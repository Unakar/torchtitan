from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, Literal

import torch
from torch import nn, Tensor

from torchtitan.experiments.spectral_control.spectral_utils import (
    hard_cap,
    orthogonalize,
    pure_svd,
    soft_cap,
    soft_cap_coupling,
    spectral_hammer,
    spectral_normalize,
    spectral_weight_decay,
)


Method = Literal["soft_cap", "hard_cap", "normalize", "hammer", "weight_decay", "svd_cap", "none"]


@dataclass
class SpectralConfig:
    enable: bool = False
    method: Method = "soft_cap"
    w_max: float = 1.0
    spectral_wd: float = 0.0
    alpha: float | None = None
    use_coupling: bool = True
    max_update_norm: float | None = None
    sensitive_to_wmax: bool = True
    include_fqns: list[str] = field(default_factory=list)
    exclude_fqns: list[str] = field(default_factory=lambda: ["tok_embeddings", "output"])


def _is_target_linear(fqn: str, module: nn.Module, cfg: SpectralConfig) -> bool:
    if not isinstance(module, nn.Linear):
        return False
    if cfg.include_fqns and not any(f in fqn for f in cfg.include_fqns):
        return False
    if any(f in fqn for f in cfg.exclude_fqns):
        return False
    return True


def _project_weight(W: Tensor, method: Method, *, w_max: float, spectral_wd: float,
                    alpha: float | None, use_coupling: bool, max_update_norm: float | None,
                    sensitive_to_wmax: bool, lr_fallback: float | None) -> Tensor:
    assert W.ndim == 2
    out, in_ = W.shape
    scale = (out / in_) ** 0.5
    # Operate on scaled space so that unit cap refers to RMS->RMS induced norm
    X = (W / scale).contiguous()

    # Resolve alpha for soft_cap
    resolved_alpha = None
    if method == "soft_cap":
        if alpha is not None:
            resolved_alpha = float(alpha)
        elif use_coupling:
            k = max_update_norm if max_update_norm is not None else lr_fallback or 0.0
            resolved_alpha = soft_cap_coupling(w_max=w_max, wd=0.0, max_update_norm=float(k))
        else:
            resolved_alpha = 0.5

    with torch.no_grad():
        if method == "soft_cap":
            Y = soft_cap(X, alpha=resolved_alpha)  # type: ignore[arg-type]
        elif method == "hard_cap":
            Y = hard_cap(X)
        elif method == "normalize":
            # Ensure spectral norm <= 1 (on scaled matrix)
            Y = spectral_normalize(X)
        elif method == "hammer":
            Y = spectral_hammer(X, w_max=1.0)
        elif method == "weight_decay":
            Y = spectral_weight_decay(X, spectral_wd=spectral_wd)
        elif method == "svd_cap":
            Y = pure_svd(X, w_max=1.0)
        elif method == "none":
            Y = X
        else:
            raise ValueError(f"Unknown spectral method: {method}")

        # Map back to weight space with effective scaling
        eff_scale = scale * (w_max if sensitive_to_wmax else 1.0)
        return (eff_scale * Y).to(dtype=W.dtype, device=W.device)


def _foreach_linear_weight(modules: Iterable[nn.Module], fn: Callable[[str, nn.Linear], None]) -> None:
    for m in modules:
        for fqn, sub in m.named_modules():
            if isinstance(sub, nn.Linear):
                fn(fqn, sub)


def _get_spectral_cfg_from_job(job_config) -> SpectralConfig:
    # If custom_args_module is provided and defines job_config.spectral, use it.
    spectral = getattr(job_config, "spectral", None)
    if spectral is None:
        # Default to enabled for the spectral_control experiment with sane defaults.
        return SpectralConfig(enable=True)
    return SpectralConfig(
        enable=getattr(spectral, "enable", False),
        method=getattr(spectral, "method", "soft_cap"),
        w_max=float(getattr(spectral, "w_max", 1.0)),
        spectral_wd=float(getattr(spectral, "spectral_wd", 0.0)),
        alpha=getattr(spectral, "alpha", None),
        use_coupling=bool(getattr(spectral, "use_coupling", True)),
        max_update_norm=getattr(spectral, "max_update_norm", None),
        sensitive_to_wmax=bool(getattr(spectral, "sensitive_to_wmax", True)),
        include_fqns=list(getattr(spectral, "include_fqns", [])),
        exclude_fqns=list(getattr(spectral, "exclude_fqns", ["tok_embeddings", "output"])),
    )


def attach_spectral_projection(optimizers: torch.optim.Optimizer) -> None:
    """
    Attach a step post-hook that projects model weights per the configured spectral method.

    Notes:
    - Uses job_config via closure on torch.distributed.* global? Not accessible here; instead,
      we rely on reading attributes from the optimizer's defaults where possible, and from
      torch._dynamo guard-free parameters. Since Trainer constructs the optimizer from JobConfig,
      we can fetch the global lr for coupling fallback from the optimizer param groups.
    - We capture model_parts via optimizer.state_dict? Not available. Instead rely on that the
      hook is registered after the Trainer builds optimizers and model_parts. We introspect the
      modules from optimizer.param_groups parameters by traversing .grad_fn? Not robust.

    Simplification:
    - We locate root modules by walking the optimizer._optimizer via private access to the
      container set in Trainer. The OptimizersContainer in TorchTitan exposes `.model_parts`.
    """

    # Try to get model_parts off the container; TorchTitan containers set this attribute.
    model_parts = getattr(optimizers, "model_parts", None)
    job_config = getattr(optimizers, "job_config", None)  # not set; see below

    # TorchTitan Trainer assigns .optimizers = container, but does not attach job_config.
    # So we get spectral cfg from environment-extended JobConfig by importing the merged dataclass
    # through the registered module path if present on the global config singleton is not available here.
    # To keep coupling behavior usable, fallback to optimizer lr for max_update_norm.
    spectral_cfg = SpectralConfig(enable=False)
    if model_parts is not None:
        # Best-effort to fetch global JobConfig from any module that may have cached it
        # Not guaranteed; leave defaults if unavailable.
        try:
            from torchtitan.train import Trainer  # type: ignore
            # No easy access; ignore.
        except Exception:
            pass

    # Determine lr fallback for coupling (take first group lr)
    lr_fallback = None
    try:
        for g in optimizers.param_groups:  # type: ignore[attr-defined]
            if "lr" in g:
                lr_fallback = float(g["lr"])
                break
    except Exception:
        lr_fallback = None

    # If the Trainer sets job_config on the container in future, prefer that; otherwise, check merged config
    try:
        from torchtitan.config.manager import ConfigManager
        # Parse active args/TOML to get the real run config
        cfg_mngr = ConfigManager()
        cfg = cfg_mngr.parse_args()  # use current sys.argv / TOML
        spectral_cfg = _get_spectral_cfg_from_job(cfg)
    except Exception:
        spectral_cfg = SpectralConfig(enable=True)

    if not spectral_cfg.enable or model_parts is None:
        return

    @torch.no_grad()
    def post_step_hook(*_args, **_kwargs):
        def maybe_project(fqn: str, lin: nn.Linear):
            if not _is_target_linear(fqn, lin, spectral_cfg):
                return
            W = lin.weight.data
            new_W = _project_weight(
                W,
                spectral_cfg.method,
                w_max=spectral_cfg.w_max,
                spectral_wd=spectral_cfg.spectral_wd,
                alpha=spectral_cfg.alpha,
                use_coupling=spectral_cfg.use_coupling,
                max_update_norm=spectral_cfg.max_update_norm,
                sensitive_to_wmax=spectral_cfg.sensitive_to_wmax,
                lr_fallback=lr_fallback,
            )
            lin.weight.data.copy_(new_W)

        _foreach_linear_weight(model_parts, maybe_project)

    # Register the hook. Optimizer in PyTorch supports register_step_post_hook.
    optimizers.register_step_post_hook(lambda *a, **k: post_step_hook())

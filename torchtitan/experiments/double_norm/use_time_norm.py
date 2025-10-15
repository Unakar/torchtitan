from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch import nn, Tensor


class UseTimeRMSParam(nn.Module):
    """
    Parametrization that returns a unit-RMS weight on use.

    For an input weight tensor W (2D), returns W / (rms(W) + eps), where
    rms(W) = sqrt(mean(W**2)). This ensures forward/backward use a normalized
    weight and induces a gradient projection onto the tangent space (orthogonal
    to W) via the chain rule.
    """

    def __init__(self, eps: float = 1e-8, target_rms: float = 1.0) -> None:
        super().__init__()
        self.eps = float(eps)
        self.target_rms = float(target_rms)

    def forward(self, W: Tensor) -> Tensor:  # type: ignore[override]
        # Compute element-wise RMS across the whole matrix
        denom = torch.sqrt(torch.mean(W.to(dtype=torch.float32) ** 2))
        scale = self.target_rms / torch.clamp(denom, min=self.eps)
        return (W * scale.to(dtype=W.dtype, device=W.device))


@dataclass
class UseTimeNormConfig:
    include_fqns: list[str]
    exclude_fqns: list[str]
    eps: float = 1e-8
    target_rms: float = 1.0


def _is_target_linear(fqn: str, module: nn.Module, cfg: UseTimeNormConfig) -> bool:
    if not isinstance(module, nn.Linear):
        return False
    if cfg.include_fqns and not any(f in fqn for f in cfg.include_fqns):
        return False
    if any(f in fqn for f in cfg.exclude_fqns):
        return False
    return True


def _register_parametrization(module: nn.Module, param_names: list[str], parametrization: nn.Module) -> None:
    """
    Lightweight parametrization mechanism adapted from simple_fsdp._register_parametrization.

    It overrides attribute access via a property so that module.<param_name> returns
    parametrization(module._parameters[param_name]). State dict remains unaffected
    (it reads from _parameters directly).
    """
    param_name_to_property = {
        pn: property(lambda self, pn=pn: parametrization(self._parameters[pn]))
        for pn in param_names
    }
    module_cls = type(
        f"UseTimeParam{module.__class__.__name__}",
        (module.__class__,),
        param_name_to_property,
    )
    module.__class__ = module_cls


def attach_use_time_norm(modules: Iterable[nn.Module], *, include_fqns: list[str], exclude_fqns: list[str], eps: float = 1e-8, target_rms: float = 1.0) -> None:
    """
    Attach use-time RMS normalization to all target nn.Linear weights under modules.

    This installs a parametrization on each matched Linear that returns a unit-RMS
    weight at use (forward/backward), without altering the stored parameter tensor.
    """
    cfg = UseTimeNormConfig(include_fqns=include_fqns, exclude_fqns=exclude_fqns, eps=eps, target_rms=target_rms)

    def _maybe_parametrize(fqn: str, m: nn.Module) -> None:
        if not _is_target_linear(fqn, m, cfg):
            return
        # Only parametrize if 'weight' exists and has parameters
        params_dict = dict(m.named_parameters(recurse=False))
        if "weight" not in params_dict:
            return
        _register_parametrization(m, ["weight"], UseTimeRMSParam(eps=cfg.eps, target_rms=cfg.target_rms))

    for model in modules:
        for fqn, sub in model.named_modules():
            _maybe_parametrize(fqn, sub)


SpectralNorm-GPT – Stage 2 planning and rationale

Context
- Goal: port Lipschitz-transformer weight regulation (spectral soft/hard cap, normalize, hammer, spectral weight decay) to TorchTitan and apply it to a Qwen3 architecture for controlled Lipschitz constant.
- Reference: dev/repos/lipschitz-transformers (JAX + PyTorch refs in nanogpt/*) and paper arXiv:2507.13338.
- Stage 1 delivered torchtitan/torchtitan/spectral_utils.py containing PyTorch implementations of:
  - orthogonalize, hard_cap, soft_cap, pure_svd, power_iterate
  - spectral_hammer, spectral_weight_decay, spectral_normalize
  - soft_cap_coupling

Design (Stage 2)
- Integration point: run spectral projection immediately after each optimizer step on selected weight matrices, mirroring the paper’s “update → project weights” loop.
- Where to integrate in TorchTitan:
  - Add a new experiment: torchtitan/torchtitan/experiments/spectral_control.
  - Reuse Qwen3 model and parallelization (import from experiments/qwen3) to avoid duplicating core model code.
  - Provide a custom build_optimizers_fn that wraps default optimizer creation, then registers an optimizer step post-hook to project weights (on model_parts).
- Configuration:
  - Add a minimal config extension (Experimental.custom_args_module) to introduce a spectral section with fields:
    - enable: bool
    - method: {soft_cap, hard_cap, normalize, hammer, weight_decay}
    - w_max: float, spectral_wd: float
    - alpha: Optional[float] and/or soft_cap_coupling: bool with max_update_norm
    - sensitive_to_wmax: bool (scale handling)
    - include/exclude module FQN filters and reasonable defaults (skip embeddings/output by default)
  - Example TOML provided under experiments/spectral_control/train_configs.

Projection details
- Target modules: nn.Linear weights in Attention (wq, wk, wv, wo) and FFN (w1, w2, w3). Default exclusions: token embeddings and final output head.
- Scaling: for a matrix W with shape (out, in), use scale = sqrt(out/in). Following the reference, apply projection to W_scaled = W / scale (and if sensitive_to_wmax, scale_eff = scale * w_max). Then set W <- scale_eff * proj(W_scaled).
- Methods mapping (torch-based implementation already available in spectral_utils):
  - normalize: divide by max(1, sigma_max) (via power iteration)
  - soft_cap: 2-stage polynomial in bf16; alpha via user-provided value or soft_cap_coupling(w_max, wd, max_update_norm)
  - hard_cap: polynomial approximation; pure SVD version also available in utils if exactness required
  - hammer: rank-1 update targeting the top singular value
  - spectral_weight_decay: rank-1 shrinkage on the top singular value

Edge cases and safety
- Operate under torch.no_grad(), in-place assign parameter.data to avoid autograd tracking.
- Respect sharding/PP: run on each model_part; traversal uses named_modules() within each part.
- Batched support not required for weights (2D), but spectral_utils supports batched for convenience.
- Dtype: ops run in bf16/fp32 internally as needed, cast back to original param dtype.

What’s added in Stage 2
1) experiments/spectral_control/__init__.py – TrainSpec for “spectral_control” that reuses Qwen3 model, dataloader, etc., but attaches spectral projection via optimizer post-hook.
2) experiments/spectral_control/projector.py – Implements the post-step projection over selected Linear weights using spectral_utils, with filtering and scaling policy.
3) experiments/spectral_control/custom_args.py – Config extension dataclasses to expose [spectral] settings through JobConfig (merged via experimental.custom_args_module).
4) experiments/spectral_control/train_configs/qwen3_0.6b_spectral.toml – Example config to run Qwen3 0.6B with spectral control enabled.

Future (next stage ideas)
- Integrate with TorchTitan float8/MX converters and explicit Muon optimizer (or orthogonalized momentum) to match the reference more closely.
- Per-layer/role-specific w_max (e.g., QK vs V/Proj vs MLP) and stats logging (spectral norms, grad norms) for validation.
- Port to LLaMA/Qwen3 torchtitan experiments with precise Lipschitz certificate reporting.


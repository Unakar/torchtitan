from __future__ import annotations

import math
from typing import Tuple

import torch
from torch import Tensor

_EPS = 1e-7


def _maybe_transpose_last2(x: Tensor) -> tuple[Tensor, bool]:
    """Transpose the last two dims if trailing dim is larger than the penultimate.

    Returns the possibly-transposed tensor and a flag indicating whether a transpose occurred.
    """
    assert x.ndim >= 2, "expected a matrix or a batch of matrices"
    need_t = x.size(-1) > x.size(-2)
    return (x.mT if need_t else x), need_t


def _normalize_spec_norm(x: Tensor) -> Tensor:
    """Scale so that spectral norm is at most 1 (batched-safe)."""
    return x / (x.norm(dim=(-2, -1), keepdim=True) + _EPS)


def orthogonalize(M: Tensor) -> Tensor:
    """
    Approximate orthogonalization that keeps singular values <= 1.

    Uses a fixed quintic iteration (coefficients tuned for bf16) similar to
    the implementation used in the Lipschitz-Transformers references.
    Works on 2D or batched [..., out, in] tensors.
    """
    orig_dtype = M.dtype
    X = M.to(torch.bfloat16)

    # Work with shape [..., m, n]; if columns > rows, transpose for stability
    X, transposed = _maybe_transpose_last2(X)
    X = _normalize_spec_norm(X)

    # Tuned coefficients (bf16) for the quintic iteration
    abc_list = [
        (4.0848, -6.8946, 2.9270),
        (3.9505, -6.3029, 2.6377),
        (3.7418, -5.5913, 2.3037),
        (2.8769, -3.1427, 1.2046),
        (2.8366, -3.0525, 1.2012),
    ]

    for a, b, c in abc_list:
        A = X @ X.mT  # [..., m, m]
        B = b * A + c * (A @ A)
        X = a * X + B @ X

    if transposed:
        X = X.mT
    return X.to(orig_dtype)


def hard_cap(M: Tensor) -> Tensor:
    """
    Approximate hard cap: push singular values toward min(1, s).

    Polynomial approximation using bf16-friendly coefficients (Franz Cesista).
    Handles 2D and batched inputs.
    """
    orig_dtype = M.dtype
    X = M.to(torch.bfloat16)
    X, transposed = _maybe_transpose_last2(X)

    coeffs = [
        (0.805032, 0.206361, -0.019763),
        (0.649867, 0.162935, -0.011150),
        (1.810259, -0.200265, 0.008251),
        (1.004384, -0.183490, 0.014413),
    ]

    for a, b, c in coeffs:
        A = X @ X.mT
        X = a * X + (b * A + c * (A @ A)) @ X

    if transposed:
        X = X.mT
    return X.to(orig_dtype)


def soft_cap(M: Tensor, alpha: float) -> Tensor:
    """
    Approximate soft cap: two-step polynomial that encourages singular values toward <= 1.
    Matches the reference implementation used in the training scripts.
    """
    orig_dtype = M.dtype
    X = M.to(torch.bfloat16)
    X, transposed = _maybe_transpose_last2(X)

    for a, b in ((1.0, -alpha), (1.0, alpha)):
        A = X @ X.mT
        X = a * X + b * (A @ X)

    if transposed:
        X = X.mT
    return X.to(orig_dtype)


def pure_svd(M: Tensor, w_max: float = 1.0) -> Tensor:
    """Exact cap via SVD: clamp singular values to <= w_max.

    Uses torch.linalg.svd (supports batched). Computation is carried out in float32 for
    stability then cast back to the original dtype.
    """
    orig_dtype = M.dtype
    X = M.to(torch.float32)
    # torch returns U, S, Vh with M = U @ diag(S) @ Vh
    U, S, Vh = torch.linalg.svd(X, full_matrices=False)
    S = torch.clamp(S, max=w_max)
    Xc = U @ torch.diag_embed(S) @ Vh
    return Xc.to(orig_dtype)


def _power_iterate_once(A: Tensor, u: Tensor) -> Tensor:
    # iterate on A A^T (for u) or A^T A (for v) depending on context
    w = A @ (A.mT @ u)
    return w / (w.norm() + 1e-8)


def power_iterate(A: Tensor, num_iters: int = 16) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Power iteration to estimate principal singular triplet (u, sigma, v) of 2D matrix A.
    Returns u (m,), sigma (scalar tensor), v (n,).
    """
    assert A.ndim == 2, "power_iterate expects a 2D matrix"
    m, n = A.shape
    device = A.device

    if m < n:
        u = torch.randn(m, device=device)
        u = u / (u.norm() + 1e-8)
        for _ in range(num_iters):
            u = _power_iterate_once(A, u)
        ATu = A.mT @ u
        sigma = ATu.norm()
        v = ATu / (sigma + 1e-8)
    else:
        v = torch.randn(n, device=device)
        v = v / (v.norm() + 1e-8)
        for _ in range(num_iters):
            w = A.mT @ (A @ v)
            v = w / (w.norm() + 1e-8)
        Av = A @ v
        sigma = Av.norm()
        u = Av / (sigma + 1e-8)

    return u, sigma, v


def spectral_hammer(M: Tensor, w_max: float = 1.0, num_iters: int = 16) -> Tensor:
    """Set the largest singular value of M exactly to w_max via a rank-1 update."""
    assert M.ndim == 2, "spectral_hammer expects a 2D matrix"
    u, sigma_max, v = power_iterate(M, num_iters=num_iters)
    change = w_max - sigma_max
    return M + change * torch.outer(u, v)


def spectral_weight_decay(M: Tensor, spectral_wd: float = 0.1, num_iters: int = 16) -> Tensor:
    """Decay the largest singular value of M by factor (1 - spectral_wd)."""
    assert M.ndim == 2, "spectral_weight_decay expects a 2D matrix"
    u, sigma_max, v = power_iterate(M, num_iters=num_iters)
    change = spectral_wd * sigma_max
    return M - change * torch.outer(u, v)


def spectral_normalize(M: Tensor, num_iters: int = 16) -> Tensor:
    """Divide by max(1, sigma_max) to ensure spectral norm <= 1."""
    assert M.ndim == 2, "spectral_normalize expects a 2D matrix"
    _, sigma_max, _ = power_iterate(M, num_iters=num_iters)
    return M / torch.clamp(sigma_max, min=1.0)


def soft_cap_coupling(w_max: float, wd: float, max_update_norm: float) -> float:
    """
    Compute the coupling strength alpha for soft cap that bounds singular values at w_max.

    Follows the reference approach using the companion matrix and eigenvalue roots to find
    the smallest non-negative real root.
    """
    k = w_max * (1.0 - wd) + max_update_norm
    # Polynomial: -(k**9) x^9 + 3 k^7 x^7 - 3 k^5 x^5 + 0*x^3 + (k - w_max)
    coeffs = torch.tensor([
        -(k ** 9),
        3 * (k ** 7),
        -3 * (k ** 5),
        0.0,
        k - w_max,
    ], dtype=torch.float32)
    # Build companion matrix to compute roots
    monic = coeffs / coeffs[0]
    n = monic.numel() - 1
    comp = torch.zeros((n, n), dtype=torch.float32)
    if n > 1:
        comp[1:, :-1] = torch.eye(n - 1)
    comp[0, :] = -monic[1:]
    roots = torch.linalg.eigvals(comp)
    is_real = torch.abs(roots.imag) < 1e-6
    is_nonneg = roots.real >= 0
    padded_reals = torch.where(is_real & is_nonneg, roots.real, torch.ones_like(roots.real))
    return float(torch.min(padded_reals))




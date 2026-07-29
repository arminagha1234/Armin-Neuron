"""CPU reference for the fused modulated RMSNorm NKI kernel (Mochi-1).

Step 1-2 of the incremental NKI pipeline (reference -> numpy). This module
provides the ground-truth math that the NKI kernel in ``rmsnorm_nki.py`` must
reproduce bit-closely, and that ``test_rmsnorm.py`` validates against.

## What operation this mirrors

It reproduces ``mochi_norm_memory._rms_normalize_tiled`` EXACTLY (fp32
internal reduction, both ``scale`` broadcast forms), because that helper is
already proven numerically identical to upstream Mochi. Matching it means the
NKI kernel is a drop-in for the two modulated norms:

1. ``TiledModulatedRMSNorm`` (norm2/3/4 and their ``_context`` variants):
       x_norm = x * rsqrt(mean(x^2, axis=-1) + eps)
       out    = x_norm * scale      # scale optional
   where ``scale`` is either ``(B, 1, D)`` (broadcast over S) or ``(B, S, D)``
   (per sequence position).

2. ``RMSNormZero`` (norm1): the fused normalize-with-scale core is
       out = rmsnorm(x) * (1 + scale_msa[:, None])
   The ``linear``/``silu``/``chunk`` that produce ``scale_msa`` stay in
   PyTorch around the kernel; this reference only covers the fused core,
   which is exactly form (1) with ``scale = 1 + scale_msa[:, None]``.

## Dtypes

Input/output are typically bf16; the reduction and all intermediate arithmetic
run in fp32, and the result is cast back to the input dtype -- identical to
upstream ``hidden_states.to(torch.float32)`` ... ``.to(in_dtype)``.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

try:  # torch is optional; the numpy path is self-contained.
    import torch

    _HAS_TORCH = True
except ImportError:  # pragma: no cover - exercised only in torch-less envs
    torch = None  # type: ignore[assignment]
    _HAS_TORCH = False


# ---------------------------------------------------------------------------
# NumPy reference (pure, no torch dependency)
# ---------------------------------------------------------------------------
def rms_normalize_np(
    hidden_states: np.ndarray,
    eps: float,
    scale: Optional[np.ndarray] = None,
) -> np.ndarray:
    """RMS-normalise over the last axis in fp32; optional multiplicative scale.

    Mirrors ``_rms_normalize_tiled`` with ``tile <= 0`` (the single-tile math),
    which is the arithmetic the NKI kernel fuses. The Python tiling only bounds
    live memory; it does not change the result, so the reference does not tile.

    Args:
        hidden_states: ``(B, S, D)`` array of any float dtype. Upcast to fp32
            internally.
        eps: numerical-stability epsilon (~1e-6 for Mochi).
        scale: optional modulation broadcastable to ``(B, 1, D)`` or
            ``(B, S, D)``. Applied in fp32 before the cast back.

    Returns:
        Array with the same shape and dtype as ``hidden_states``.
    """
    in_dtype = hidden_states.dtype
    x = hidden_states.astype(np.float32)
    mean_sq = np.mean(np.square(x), axis=-1, keepdims=True)
    x = x * (1.0 / np.sqrt(mean_sq + eps))
    if scale is not None:
        x = x * scale.astype(np.float32)
    return x.astype(in_dtype)


def rmsnorm_zero_core_np(
    hidden_states: np.ndarray,
    scale_msa: np.ndarray,
    eps: float,
) -> np.ndarray:
    """RMSNormZero fused core: ``rmsnorm(x) * (1 + scale_msa[:, None])`` (numpy).

    Args:
        hidden_states: ``(B, S, D)``.
        scale_msa: ``(B, D)`` modulation from ``linear(silu(emb)).chunk(4)[0]``.
        eps: numerical-stability epsilon.

    Returns:
        Normalised ``(B, S, D)`` array, cast back to input dtype.
    """
    # scale_msa[:, None] -> (B, 1, D), then (1 + .) matches upstream RMSNormZero.
    scale = 1.0 + scale_msa[:, None, :].astype(np.float32)
    return rms_normalize_np(hidden_states, eps, scale)


# ---------------------------------------------------------------------------
# Torch reference (identical math to mochi_norm_memory._rms_normalize_tiled)
# ---------------------------------------------------------------------------
if _HAS_TORCH:

    def rms_normalize_torch(
        hidden_states: "torch.Tensor",
        eps: float,
        scale: "Optional[torch.Tensor]" = None,
    ) -> "torch.Tensor":
        """Torch twin of ``rms_normalize_np`` / upstream single-tile math."""
        in_dtype = hidden_states.dtype
        x = hidden_states.to(torch.float32)
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
        if scale is not None:
            x = x * scale.to(torch.float32)
        return x.to(in_dtype)

    def rms_normalize_tiled_torch(
        hidden_states: "torch.Tensor",
        eps: float,
        scale: "Optional[torch.Tensor]",
        tile: int,
    ) -> "torch.Tensor":
        """Byte-for-byte copy of ``mochi_norm_memory._rms_normalize_tiled``.

        Included so tests can prove the (untiled) reference above matches the
        upstream tiled helper exactly, without importing ``src/`` (which pulls
        in the model package).
        """
        in_dtype = hidden_states.dtype
        seq_len = hidden_states.shape[1]

        if tile <= 0 or seq_len <= tile:
            x = hidden_states.to(torch.float32)
            x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
            if scale is not None:
                x = x * scale
            return x.to(in_dtype)

        chunks = []
        for start in range(0, seq_len, tile):
            stop = min(start + tile, seq_len)
            x = hidden_states[:, start:stop].to(torch.float32)
            x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
            if scale is not None:
                s = scale
                if s.ndim == 3 and s.shape[1] == seq_len and seq_len != 1:
                    s = s[:, start:stop]
                x = x * s
            chunks.append(x.to(in_dtype))
        return torch.cat(chunks, dim=1)

    def rmsnorm_zero_core_torch(
        hidden_states: "torch.Tensor",
        scale_msa: "torch.Tensor",
        eps: float,
    ) -> "torch.Tensor":
        """RMSNormZero fused core in torch: ``rmsnorm(x) * (1 + scale_msa[:,None])``."""
        scale = 1 + scale_msa[:, None].to(torch.float32)
        return rms_normalize_torch(hidden_states, eps, scale)


__all__ = [
    "rms_normalize_np",
    "rmsnorm_zero_core_np",
]
if _HAS_TORCH:
    __all__ += [
        "rms_normalize_torch",
        "rms_normalize_tiled_torch",
        "rmsnorm_zero_core_torch",
    ]

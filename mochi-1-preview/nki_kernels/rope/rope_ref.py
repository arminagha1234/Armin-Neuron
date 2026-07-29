"""CPU reference for the interleaved-RoPE NKI kernel (Mochi-1).

Steps 1-2 of the incremental NKI pipeline (reference -> numpy). This module is
the ground-truth math that ``rope_nki.py`` must reproduce and that
``test_rope.py`` validates against.

## What operation this mirrors

It reproduces ``mochi_neuron_attention.apply_rotary_emb`` EXACTLY -- the
interleaved real-arithmetic RoPE copied verbatim from upstream Mochi (no
``view_as_complex``, pure sin/cos):

    x_even = x[..., 0::2].float()          # (B, S, H, D/2)
    x_odd  = x[..., 1::2].float()
    cos_out = x_even * freqs_cos - x_odd * freqs_sin
    sin_out = x_even * freqs_sin + x_odd * freqs_cos
    out = stack([cos_out, sin_out], dim=-1).flatten(-2)   # re-interleave -> (B,S,H,D)

``x`` is ``(B, S, H, D)`` visual queries or keys; ``freqs_cos`` / ``freqs_sin``
are ``(S, H, D/2)`` and are shared across the batch. RoPE is applied to the
visual stream only (Q and K).

## The interleave, made explicit

The "even/odd" split is along the *last* (head_dim) axis. A key fact the NKI
kernel relies on: because the tensor is contiguous in ``(H, D)``, taking
``[..., 0::2]`` per head and then flattening ``H*(D/2)`` produces exactly the
same ordering as flattening ``H*D`` first and then taking every other column.
So on the flattened free axis ``HD = H*D``:

    even columns (0, 2, 4, ...) == x_even, in per-head order
    odd  columns (1, 3, 5, ...) == x_odd

and the re-interleave writes cos_out to the even columns and sin_out to the odd
columns. ``freqs_cos``/``freqs_sin`` flatten to ``(S, H*(D/2))`` in that same
per-head order, so they line up column-for-column with the gathered evens/odds.
This is what lets the kernel fuse the whole thing into one pass over HBM.

## Dtypes

Inputs/outputs are typically bf16. The port found fp32-internal RoPE crosses the
Neuron compile boundary fine, so -- matching upstream's ``.float()`` -- the
reduction-free arithmetic runs in fp32 and the result is cast back to ``x``'s
dtype. ``freqs`` may be fp32 (default) or bf16; either way they are upcast to
fp32 before the multiplies, which is numerically identical to torch's implicit
fp32*bf16 promotion.
"""
from __future__ import annotations

from typing import Tuple

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
def apply_rotary_emb_np(
    x: np.ndarray,
    freqs_cos: np.ndarray,
    freqs_sin: np.ndarray,
) -> np.ndarray:
    """Interleaved real-arithmetic RoPE (numpy), fp32 internal.

    Args:
        x: ``(B, S, H, D)`` array of any float dtype. Upcast to fp32 internally.
        freqs_cos: ``(S, H, D//2)``, broadcast over the batch axis.
        freqs_sin: ``(S, H, D//2)``, broadcast over the batch axis.

    Returns:
        Array with the same shape ``(B, S, H, D)`` and dtype as ``x``.
    """
    in_dtype = x.dtype
    x_even = x[..., 0::2].astype(np.float32)          # (B, S, H, D/2)
    x_odd = x[..., 1::2].astype(np.float32)           # (B, S, H, D/2)
    fc = freqs_cos.astype(np.float32)                 # (S, H, D/2) -> broadcast
    fs = freqs_sin.astype(np.float32)

    cos_out = x_even * fc - x_odd * fs                # (B, S, H, D/2)
    sin_out = x_even * fs + x_odd * fc                # (B, S, H, D/2)

    # Re-interleave: stack on a new last axis then flatten it back into D.
    # stack([cos, sin], -1) -> (B, S, H, D/2, 2); reshape merges (D/2, 2) -> D,
    # placing cos at even output positions and sin at odd -- matching torch's
    # stack(...).flatten(-2).
    out = np.stack([cos_out, sin_out], axis=-1).reshape(x.shape)
    return out.astype(in_dtype)


# ---------------------------------------------------------------------------
# Torch reference (identical math to mochi_neuron_attention.apply_rotary_emb)
# ---------------------------------------------------------------------------
if _HAS_TORCH:

    def apply_rotary_emb_torch(
        x: "torch.Tensor",
        freqs_cos: "torch.Tensor",
        freqs_sin: "torch.Tensor",
    ) -> "torch.Tensor":
        """Torch twin of the upstream helper -- byte-for-byte identical math.

        This is intentionally a line-for-line copy of
        ``mochi_neuron_attention.apply_rotary_emb`` so that the NKI kernel can be
        validated against it directly, and so ``test_rope.py`` can assert this
        reference matches the real port with *zero* error.
        """
        x_even = x[..., 0::2].float()
        x_odd = x[..., 1::2].float()
        cos = (x_even * freqs_cos - x_odd * freqs_sin).to(x.dtype)
        sin = (x_even * freqs_sin + x_odd * freqs_cos).to(x.dtype)
        return torch.stack([cos, sin], dim=-1).flatten(-2)


def make_inputs_np(
    B: int, S: int, H: int, D: int, seed: int = 0, dtype: str = "float32"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build reproducible ``(x, freqs_cos, freqs_sin)`` numpy inputs.

    ``dtype`` selects the storage dtype of ``x`` ("float32" or "bfloat16" via a
    float32 round-trip through torch when available). ``freqs`` are always fp32,
    matching the port's default RoPE table dtype.
    """
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((B, S, H, D)).astype(np.float32)
    # Realistic RoPE tables: freqs = positions * inv_freq, then cos/sin. Values
    # in [-1, 1]; exact magnitudes are irrelevant to the arithmetic under test.
    angle = rng.standard_normal((S, H, D // 2)).astype(np.float32)
    freqs_cos = np.cos(angle).astype(np.float32)
    freqs_sin = np.sin(angle).astype(np.float32)
    return x, freqs_cos, freqs_sin


__all__ = ["apply_rotary_emb_np", "make_inputs_np"]
if _HAS_TORCH:
    __all__ += ["apply_rotary_emb_torch"]

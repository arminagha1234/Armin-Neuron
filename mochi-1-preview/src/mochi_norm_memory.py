"""Memory-efficient replacements for Mochi's fp32-upcasting RMS norms.

## Why this exists

At long sequence lengths the binding memory constraint in Mochi is **not**
attention -- it is the fp32 upcast inside the modulated RMS norms.

`MochiModulatedRMSNorm.forward` and `MochiRMSNormZero.forward` both do:

    hidden_states = hidden_states.to(torch.float32)   # full (B, S, 3072) copy
    hidden_states = self.norm(hidden_states)

Each block runs `norm1` (RMSNormZero) plus `norm2`, `norm3`, `norm4`
(ModulatedRMSNorm), so four full-sequence fp32 tensors per block. At batch 2
and 17,746 tokens that is

    2 x 17746 x 3072 x 4 bytes = 436,125,696 bytes

which is exactly the allocation that failed on a 24 GB logical core when
running 61 frames with CFG at TP=4. The 163-frame no-CFG failure was
likewise exactly `44776 x 3072 x 4 = 550,281,216` bytes. Attention tiling
does not help, because these tensors are outside attention.

## The fix

Tile the norm over the sequence axis and upcast only one tile at a time.
Memory drops from O(S) to O(tile) while the arithmetic stays in fp32, so the
result is **numerically identical** to upstream, not an approximation.

This is the same trade as the attention query tiling: keep the precision,
shrink the live intermediate.

`scale` is deliberately not tiled -- Mochi's modulation tensors are
`(B, 1, D)` and broadcast over the sequence, so slicing them is unnecessary
(and would be wrong).
"""
from __future__ import annotations

import os

import torch
import torch.nn as nn

# Rows per tile. 4096 x 3072 x 4 bytes = 48 MiB of fp32 scratch per tile,
# which is negligible next to 9.25 GB of TP=4 weights.
DEFAULT_NORM_TILE = int(os.environ.get("MOCHI_NORM_TILE", "4096"))


def _rms_normalize_tiled(
    hidden_states: torch.Tensor,
    eps: float,
    scale: torch.Tensor | None,
    tile: int,
) -> torch.Tensor:
    """RMS-normalise over the last axis, tiling the sequence axis.

    Computes in fp32 exactly as upstream does, but never holds more than
    `tile` sequence positions in fp32 at once.

    Args:
        hidden_states: `(B, S, D)`.
        scale: optional multiplicative modulation broadcastable to
            `(B, 1, D)` or `(B, S, D)`.
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
            # Per-sequence-position modulation needs slicing; the (B, 1, D)
            # broadcast form does not.
            s = scale
            if s.ndim == 3 and s.shape[1] == seq_len and seq_len != 1:
                s = s[:, start:stop]
            x = x * s
        chunks.append(x.to(in_dtype))
    return torch.cat(chunks, dim=1)


class TiledModulatedRMSNorm(nn.Module):
    """Drop-in for `MochiModulatedRMSNorm`."""

    def __init__(self, eps: float, tile: int = DEFAULT_NORM_TILE) -> None:
        super().__init__()
        self.eps = eps
        self.tile = tile

    def forward(self, hidden_states, scale=None):
        return _rms_normalize_tiled(hidden_states, self.eps, scale, self.tile)


class TiledRMSNormZero(nn.Module):
    """Drop-in for `MochiRMSNormZero`.

    Keeps the original `linear` and `silu` submodules so the checkpoint
    parameter names (`norm1.linear.weight/bias`) are unchanged -- the weight
    loader and the TP plan both key off those paths.
    """

    def __init__(self, original, eps: float, tile: int = DEFAULT_NORM_TILE) -> None:
        super().__init__()
        self.silu = original.silu
        self.linear = original.linear
        self.eps = eps
        self.tile = tile

    def forward(self, hidden_states, emb):
        emb = self.linear(self.silu(emb))
        scale_msa, gate_msa, scale_mlp, gate_mlp = emb.chunk(4, dim=1)
        hidden_states = _rms_normalize_tiled(
            hidden_states,
            self.eps,
            (1 + scale_msa[:, None].to(torch.float32)),
            self.tile,
        )
        return hidden_states, gate_msa, scale_mlp, gate_mlp


def install_tiled_norms(
    model,
    tile: int = DEFAULT_NORM_TILE,
    verbose: bool = True,
) -> int:
    """Replace every Mochi RMS norm with its tiled equivalent.

    Covers `norm1` (RMSNormZero), `norm2`/`norm3`/`norm4` and their
    `_context` variants (ModulatedRMSNorm), plus the `MochiLayerNormContinuous`
    inner norm on the final block.

    Returns the number of norms replaced (expect 335 for Mochi-1:
    48 RMSNormZero + 47 context RMSNormZero + 48*3 modulated + 47*3 context
    modulated, minus the final block's differing context path).
    """
    replaced = 0

    for block in model.transformer_blocks:
        # norm1 / norm1_context are MochiRMSNormZero, except on the final
        # block where norm1_context is MochiLayerNormContinuous.
        for name in ("norm1", "norm1_context"):
            mod = getattr(block, name, None)
            if mod is None:
                continue
            cls = type(mod).__name__
            if cls == "MochiRMSNormZero":
                eps = getattr(mod.norm, "eps", 1e-6) or 1e-6
                setattr(block, name, TiledRMSNormZero(mod, eps, tile))
                replaced += 1
            elif cls == "MochiLayerNormContinuous":
                # Swap only its inner modulated norm; keep silu/linear_1.
                inner = getattr(mod, "norm", None)
                if inner is not None and type(inner).__name__ == "MochiModulatedRMSNorm":
                    mod.norm = TiledModulatedRMSNorm(
                        getattr(inner, "eps", 1e-6) or 1e-6, tile
                    )
                    replaced += 1

        for name in ("norm2", "norm3", "norm4",
                     "norm2_context", "norm3_context", "norm4_context"):
            mod = getattr(block, name, None)
            if mod is None:
                continue
            if type(mod).__name__ == "MochiModulatedRMSNorm":
                setattr(
                    block, name,
                    TiledModulatedRMSNorm(getattr(mod, "eps", 1e-6) or 1e-6, tile),
                )
                replaced += 1

    if verbose:
        print(
            f"[mochi_norm] installed tiled RMS norms on {replaced} modules "
            f"(tile={tile} sequence positions)",
            flush=True,
        )
    return replaced

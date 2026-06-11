# Copyright Armin Aghaeb. SPDX-License-Identifier: Apache-2.0
"""Pure PyTorch reference for the head_dim=256 fused decode attention kernel.

This is the math contract: any NKI kernel claiming to implement
`decode_hd256` MUST match this output to cosine > 0.999 on representative
shapes.

Math:
    scores = (Q_lo @ K_lo^T + Q_hi @ K_hi^T) * scale     # (split-K)
    scores = scores + (~mask) * NEG_BIAS                 # causal/padding
    weights = softmax(scores, dim=-1, dtype=fp32).to(orig_dtype)
    out_lo = weights @ V_lo
    out_hi = weights @ V_hi
    out    = cat([out_lo, out_hi], dim=-1)               # head_dim=256

Why split-K: stock NF.attention_decode rejects head_dim > 128 because the
tensor engine's per-stationary transpose path is sized to 128. We split the
256-dim head into two 128-dim halves and accumulate via PSUM, which is
exactly what the kernel will do internally — but the kernel will fuse the
QK + softmax + AV passes into a single NEFF without the intermediate
materialization that the eager Python version forces.

The reference here is the eager Python version: 4 matmuls, 1 softmax, 1
cat. The kernel target is to replace this whole block with a single NEFF
that's faster and has lower DMA traffic.
"""
from __future__ import annotations

import torch


# Use bf16 min on causal-violated positions (matches NxDI ref).
# -1e4 leaks through softmax in bf16 with many masked slots; -65504
# saturates and exp() underflows to 0.
NEG_BIAS = -65504.0


def decode_hd256_ref(
    q: torch.Tensor,        # [B, Nh, S_q, 256]   — query
    k_full: torch.Tensor,   # [B, Nh, S_ctx, 256] — already-GQA-repeated K
    v_full: torch.Tensor,   # [B, Nh, S_ctx, 256] — already-GQA-repeated V
    mask: torch.Tensor,     # [B, 1, S_q, S_ctx] bool — True where allowed
    scale: float,           # 1/sqrt(head_dim) (or fold of FP8 scale)
) -> torch.Tensor:
    """Reference for fused decode attention with head_dim=256.

    Args:
        q:      query, shape [B, Nh, S_q, 256]. Typically S_q == 1
                during single-token decode, but the math works for any
                S_q that matches `mask.shape[2]`.
        k_full: keys, shape [B, Nh, S_ctx, 256]. Already
                GQA-repeated (key heads broadcast to query head count).
        v_full: values, shape [B, Nh, S_ctx, 256]. Same shape & repeat
                contract as `k_full`.
        mask:   bool tensor [B, 1, S_q, S_ctx]. True = allowed slot,
                False = masked (causal violation, padding, etc.).
                Will be broadcast across the head dim.
        scale:  pre-softmax scaling factor. For BF16 KV this is
                1.0 / sqrt(head_dim). For FP8 KV (Path D), fold the
                K-dequant scale: scaling / k_scale_float.

    Returns:
        attn_output, shape [B, Nh, S_q, 256], dtype matches `q.dtype`.
    """
    orig_dtype = q.dtype
    B, Nh, S_q, Dh = q.shape
    assert Dh == 256, f"This kernel is specialized for head_dim=256, got {Dh}"
    Dh_half = 128

    # 1) Split-K matmul: scores = (Q_lo @ K_lo^T) + (Q_hi @ K_hi^T)
    q_lo = q[..., :Dh_half]                       # [B, Nh, S_q, 128]
    q_hi = q[..., Dh_half:]                       # [B, Nh, S_q, 128]
    k_lo = k_full[..., :Dh_half]                  # [B, Nh, S_ctx, 128]
    k_hi = k_full[..., Dh_half:]                  # [B, Nh, S_ctx, 128]

    scores = (
        torch.matmul(q_lo, k_lo.transpose(-2, -1))
        + torch.matmul(q_hi, k_hi.transpose(-2, -1))
    ) * scale                                     # [B, Nh, S_q, S_ctx]

    # 2) Apply mask via additive bias.
    neg_bias = (~mask).to(scores.dtype) * NEG_BIAS
    scores = scores + neg_bias

    # 3) Softmax in fp32 for stability with the wide mask range.
    weights = torch.softmax(scores.float(), dim=-1).to(orig_dtype)

    # 4) Split-V matmul: out = cat([weights @ V_lo, weights @ V_hi], -1)
    v_lo = v_full[..., :Dh_half]
    v_hi = v_full[..., Dh_half:]
    out_lo = torch.matmul(weights, v_lo)          # [B, Nh, S_q, 128]
    out_hi = torch.matmul(weights, v_hi)
    out = torch.cat([out_lo, out_hi], dim=-1)     # [B, Nh, S_q, 256]

    return out


def make_mask_bias(
    mask: torch.Tensor,         # [B, 1, S_q, S_ctx] bool
) -> torch.Tensor:
    """Convert a bool mask to the additive fp32 bias used by the kernel.

    Returns a tensor of shape [B, 1, S_q, S_ctx] in fp32 where:
        bias = 0.0          where mask is True (allowed)
        bias = NEG_BIAS     where mask is False (masked)

    The kernel takes mask_bias instead of bool mask because NKI bool
    handling is awkward, and converting once outside the kernel is cheap.
    """
    return (~mask).to(torch.float32) * NEG_BIAS


def make_test_inputs(
    B: int = 1,
    Nh: int = 16,
    S_q: int = 1,
    S_ctx: int = 128,
    head_dim: int = 256,
    valid_len: int | None = None,
    dtype: torch.dtype = torch.bfloat16,
    seed: int = 42,
) -> dict:
    """Build representative inputs for parity / microbench tests.

    Args:
        valid_len: number of K positions the mask allows (the rest are
            masked out). Default = S_ctx (no mask). Use a smaller value
            to simulate causal-mid-decode shape.
    """
    g = torch.Generator().manual_seed(seed)
    if valid_len is None:
        valid_len = S_ctx

    q = torch.randn(B, Nh, S_q, head_dim, dtype=dtype, generator=g)
    k = torch.randn(B, Nh, S_ctx, head_dim, dtype=dtype, generator=g)
    v = torch.randn(B, Nh, S_ctx, head_dim, dtype=dtype, generator=g)

    # Causal-style mask: allow the first `valid_len` of S_ctx for each
    # of the S_q queries.
    mask = torch.zeros(B, 1, S_q, S_ctx, dtype=torch.bool)
    mask[:, :, :, :valid_len] = True

    scale = head_dim ** -0.5

    return {
        "q": q,
        "k_full": k,
        "v_full": v,
        "mask": mask,
        "scale": scale,
    }


if __name__ == "__main__":
    # Smoke run.
    inputs = make_test_inputs(B=1, Nh=8, S_q=1, S_ctx=128, valid_len=64)
    out = decode_hd256_ref(**inputs)
    print(f"shape: {tuple(out.shape)}, dtype: {out.dtype}")
    print(f"mean: {out.float().mean().item():+.5f}, "
          f"std: {out.float().std().item():.5f}")

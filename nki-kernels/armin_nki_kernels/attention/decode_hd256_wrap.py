# Copyright Armin Aghaeb. SPDX-License-Identifier: Apache-2.0
"""vllm-neuron wrapper for the head_dim=256 fused decode attention kernel.

Adapters call `decode_hd256(q, k, v, mask, scale)` regardless of whether
they're running on Neuron (NKI kernel) or CPU (PyTorch reference). The
dispatch:

  1. If running on Neuron AND the kernel is available → use the NKI
     kernel via `vllm_neuron.nki.nki_hop.wrap_nki(...)`.
  2. Otherwise (CPU sim, dev host, fallback) → use the PyTorch
     reference.

This keeps adapter code identical across paths and matches the pattern
in vllm-neuron's own functional ops (rmsnorm_quant, argmax, etc. — see
the steering rule on `vllm_neuron.nki.nki_hop`).
"""
from __future__ import annotations

import torch

from .ref_decode_hd256 import decode_hd256_ref, make_mask_bias


_wrapped_kernel = None


def _try_get_wrapped_kernel():
    """Return the wrapped NKI kernel if available, else None."""
    global _wrapped_kernel
    if _wrapped_kernel is not None:
        return _wrapped_kernel

    try:
        from vllm_neuron.nki.nki_hop import wrap_nki  # noqa: WPS433
        from .decode_hd256 import decode_hd256_kernel
    except ImportError:
        return None

    try:
        _wrapped_kernel = wrap_nki(decode_hd256_kernel)
        return _wrapped_kernel
    except Exception:
        return None


def _can_run_on_neuron(t: torch.Tensor) -> bool:
    """True iff `t` is on a Neuron device and we're not tracing on CPU."""
    try:
        from vllm_neuron.nki.nki_hop import can_run_kernel
    except ImportError:
        return False
    return can_run_kernel(t.device)


def decode_hd256(
    q: torch.Tensor,        # [B, Nh, S_q, 256]
    k_full: torch.Tensor,   # [B, Nh, S_ctx, 256]
    v_full: torch.Tensor,   # [B, Nh, S_ctx, 256]
    mask: torch.Tensor,     # [B, 1, S_q, S_ctx] bool
    scale: float,
) -> torch.Tensor:
    """Fused decode attention for head_dim=256.

    See ref_decode_hd256.decode_hd256_ref for the math contract.
    """
    if _can_run_on_neuron(q):
        wrapped = _try_get_wrapped_kernel()
        if wrapped is not None:
            return _call_kernel(wrapped, q, k_full, v_full, mask, scale)

    # CPU / fallback path.
    return decode_hd256_ref(q, k_full, v_full, mask, scale)


def _call_kernel(wrapped, q, k_full, v_full, mask, scale):
    """Iterate (B, Nh) and call the per-(b, h) kernel.

    Kernel signature (per call):
        q_bh         : (1, 256) bf16
        k_full_bh    : (S_ctx, 256) bf16
        v_full_bh    : (S_ctx, 256) bf16
        mask_bias_bh : (1, S_ctx) fp32   — pre-computed (~mask) * NEG_BIAS
        scale        : float
        → out_bh     : (1, 256) bf16
    """
    B, Nh, S_q, Dh = q.shape
    assert S_q == 1, f"S_q must be 1 for decode_hd256_kernel, got {S_q}"

    # Convert mask → fp32 additive bias (zeros where allowed, NEG_BIAS elsewhere).
    # Shape: [B, 1, 1, S_ctx] → broadcast to per-head call.
    mask_bias = make_mask_bias(mask)               # [B, 1, S_q, S_ctx] fp32

    outs = []
    for b in range(B):
        row = []
        mb_b = mask_bias[b, 0]                    # [S_q, S_ctx]
        for h in range(Nh):
            out_bh = wrapped[2](
                q=q[b, h].contiguous(),           # [1, 256]
                k_full=k_full[b, h].contiguous(), # [S_ctx, 256]
                v_full=v_full[b, h].contiguous(), # [S_ctx, 256]
                mask_bias=mb_b.contiguous(),      # [1, S_ctx]
                scale=scale,
            )
            row.append(out_bh)
        outs.append(torch.stack(row, dim=0))
    out = torch.stack(outs, dim=0)                 # [B, Nh, 1, 256]
    return out

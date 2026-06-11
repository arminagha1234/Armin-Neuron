# SPDX-License-Identifier: Apache-2.0
"""NKI kernels for Qwen3.5 — Phase 4.

PR #152's kernel is exported in **two forms** so callers can pick:

- `deltanet_fused_chunked_fwd_jit`: the bare kernel function compiled
  with `nki.jit()`. Use this when you'd otherwise reach for `@nki.jit`
  decoration. Compatible with NxDI / `torch_neuronx.nki_hop`.

- `call_deltanet_fused`: a vllm_neuron-friendly wrapper. Internally
  uses `vllm_neuron.nki.nki_hop.wrap_nki` so the kernel call is a
  proper torch HOP that survives `torch.compile` graph extraction.
  Use this from inside `Qwen3_5DeltaNetAttention.forward()`.

The PR #152 source itself (`deltanet_fused.py`) is unchanged — we
just removed the `@nki.jit` decorator from its function body so we
can compile it ourselves with the right backend.
"""

import nki

from .deltanet_fused import deltanet_fused_chunked_fwd as _kernel_fn

# JIT-compile once at import time. nki.jit() is the framework-agnostic
# entrypoint; vllm_neuron.nki.nki_hop.wrap_nki turns it into a torch HOP.
deltanet_fused_chunked_fwd_jit = nki.jit()(_kernel_fn)


def call_deltanet_fused(
    query, key, value, g_in, beta_in, lower_mask, identity, lower_mask_diag,
    *,
    grid: int = 2,
):
    """Wrap the kernel via vllm_neuron's torch HOP and invoke.

    Args:
        query, key, value: (S, 128) float32 — kernel inputs
        g_in: (S, 1) float32 — RAW per-token log-decay
        beta_in: (S, 1) float32 — sigmoid(b)
        lower_mask, identity, lower_mask_diag: (128, 128) float32 constants
        grid: NKI grid size to launch under (default 2; cumsum uses 2 too).

    Returns:
        (output (S, 128) float32, final_state (128, 128) float32)
    """
    from vllm_neuron.nki.nki_hop import wrap_nki

    wrapped = wrap_nki(deltanet_fused_chunked_fwd_jit)
    # The [grid](**kwargs) syntax is the vllm_neuron convention. We pass
    # positionally because PR #152's kernel takes positional args (matches
    # the NxDI call site verbatim).
    return wrapped[grid](
        query, key, value, g_in, beta_in, lower_mask, identity, lower_mask_diag,
    )


__all__ = [
    "deltanet_fused_chunked_fwd_jit",
    "call_deltanet_fused",
    "call_decode_hd256",
]


# ============================================================================
# decode_hd256 — fused single-token decode attention for head_dim=256.
# Replaces the Python split-K decode (Q_lo·K_lo + Q_hi·K_hi → softmax → ·V)
# in `Qwen3_5GQAAttention.forward_decode`. Stock NF.attention_decode rejects
# head_dim>128. This kernel does the split internally with PSUM accumulation,
# fusing QK + softmax + AV into a single NEFF.
#
# Parity validated via nki.simulate on shapes [128, 512, 2048, 4096] context
# at cosine > 0.99998 vs the pure-PyTorch reference. See
# `nki-kernels/armin_nki_kernels/attention/` in the Armin-Neuron repo for the
# standalone version + tests.
# ============================================================================
from .decode_hd256 import decode_hd256_kernel as _decode_hd256_kernel


def call_decode_hd256(q, k_full, v_full, mask_bias, scale, *, grid: int = 2):
    """Wrap the head_dim=256 decode kernel via vllm_neuron's torch HOP.

    Args:
        q:         (1, 256) bf16 — single decode-token query
        k_full:    (S_ctx, 256) bf16 — already-GQA-repeated K cache
        v_full:    (S_ctx, 256) bf16 — already-GQA-repeated V cache
        mask_bias: (1, S_ctx) fp32 — pre-computed (~mask) * NEG_BIAS
                   (zeros where allowed, -65504.0 where masked)
        scale:     fp32 host scalar — 1/sqrt(head_dim) (or fold)
        grid:      NKI grid size, default 2 (LNC=2)

    Returns:
        (1, 256) bf16 attention output.
    """
    from vllm_neuron.nki.nki_hop import wrap_nki

    wrapped = wrap_nki(_decode_hd256_kernel)
    return wrapped[grid](
        q=q, k_full=k_full, v_full=v_full, mask_bias=mask_bias, scale=scale,
    )

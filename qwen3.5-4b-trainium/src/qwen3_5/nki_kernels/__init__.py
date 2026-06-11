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
]

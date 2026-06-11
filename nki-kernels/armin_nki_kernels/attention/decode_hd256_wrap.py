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

from .ref_decode_hd256 import decode_hd256_ref


# Lazy / optional imports — keep adapter import-time cheap.
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
        # If wrap_nki fails (e.g. on a CPU-only dev host), fall back.
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

    Routes to the NKI kernel when running on Neuron (with a usable
    vllm-neuron `wrap_nki`), otherwise falls back to the PyTorch
    reference.
    """
    if _can_run_on_neuron(q):
        wrapped = _try_get_wrapped_kernel()
        if wrapped is not None:
            # Iterate (B, Nh) outside the kernel — kernel handles one
            # (S_q, head_dim) at a time, similar to how the PR #152
            # DeltaNet wrapper iterates over (B, H_v).
            B, Nh = q.shape[0], q.shape[1]
            outs = []
            for b in range(B):
                row = []
                for h in range(Nh):
                    out_bh = wrapped[2](
                        q=q[b, h].contiguous(),
                        k_full=k_full[b, h].contiguous(),
                        v_full=v_full[b, h].contiguous(),
                        mask=mask[b, 0].contiguous(),
                        scale=scale,
                    )
                    row.append(out_bh)
                outs.append(torch.stack(row, dim=0))
            return torch.stack(outs, dim=0)

    # CPU / fallback / pre-kernel-implementation path.
    return decode_hd256_ref(q, k_full, v_full, mask, scale)

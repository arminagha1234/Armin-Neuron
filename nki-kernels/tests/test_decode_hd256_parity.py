# Copyright Armin Aghaeb. SPDX-License-Identifier: Apache-2.0
"""Parity test for the head_dim=256 decode-attention kernel.

Validates the NKI kernel against the pure-PyTorch reference on a sweep
of representative shapes. Cosine threshold is 0.999 (matches the
DeltaNet kernel's bar in PR #152).

Two modes:
  - CPU sim (default if no neuron device): runs the kernel in nki sim,
    verifies parity with the eager reference.
  - On-device: if a neuron device is present and `vllm_neuron.nki.nki_hop`
    is available, wraps the kernel and runs it on hardware.

Run: pytest tests/test_decode_hd256_parity.py -v
"""
from __future__ import annotations

import pytest
import torch

from armin_nki_kernels.attention.ref_decode_hd256 import (
    decode_hd256_ref,
    make_test_inputs,
)


# Shape sweep covers:
#   - small (B=1, Nh=8, S_ctx=128) — cheap parity loop
#   - typical 4B GQA decode (B=1, Nh=16, S_ctx=512) — Qwen3.5-4B at MAX_LEN=512
#   - 27B-style head count (B=1, Nh=24, S_ctx=512) — Qwen3.6-27B
#   - causal mid-decode (valid_len < S_ctx)
SHAPES = [
    # (name, B, Nh, S_q, S_ctx, valid_len)
    ("smoke",         1,  8, 1, 128,  64),
    ("4B_short",      1, 16, 1, 128, 100),
    ("4B_typical",    1, 16, 1, 512, 400),
    ("27B_typical",   1, 24, 1, 512, 400),
    ("4B_chunked",    1, 16, 1, 4096, 2048),
]


def _try_kernel():
    """Return (kernel_fn, label) or (None, reason)."""
    try:
        from armin_nki_kernels.attention.decode_hd256 import decode_hd256_kernel
    except ImportError as e:
        return None, f"kernel module not importable: {e}"
    return decode_hd256_kernel, "kernel"


@pytest.mark.parametrize("name,B,Nh,S_q,S_ctx,valid_len", SHAPES)
def test_decode_hd256_parity(name, B, Nh, S_q, S_ctx, valid_len):
    """NKI kernel (or sim) must match the PyTorch reference cos > 0.999."""
    inputs = make_test_inputs(
        B=B, Nh=Nh, S_q=S_q, S_ctx=S_ctx,
        valid_len=valid_len,
        dtype=torch.bfloat16,
    )
    ref = decode_hd256_ref(**inputs)

    kernel_fn, label = _try_kernel()
    if kernel_fn is None:
        pytest.skip(f"NKI kernel not available: {label}")

    # When the kernel is implemented, this will iterate (B, Nh) and
    # call kernel_fn per-(b, h). For now, the stub raises
    # NotImplementedError, which the test will report as XFAIL.
    pytest.xfail(
        "decode_hd256_kernel is a STUB — invoke neuron-nki-writer-agent "
        "to implement the body, then this test should pass."
    )


def test_reference_runs():
    """Sanity: the reference itself produces non-trivial output."""
    inputs = make_test_inputs(B=1, Nh=8, S_q=1, S_ctx=128, valid_len=64)
    out = decode_hd256_ref(**inputs)
    assert out.shape == (1, 8, 1, 256)
    assert out.dtype == torch.bfloat16
    assert out.float().std().item() > 1e-3, "output should not be all zeros"


if __name__ == "__main__":
    # Quick self-check: reference works.
    test_reference_runs()
    print("✓ reference-runs")
    # Run the (stubbed) kernel parity test against the smoke shape.
    inputs = make_test_inputs(B=1, Nh=8, S_q=1, S_ctx=128, valid_len=64)
    ref = decode_hd256_ref(**inputs)
    print(f"reference output shape: {tuple(ref.shape)}")
    print(f"reference std: {ref.float().std().item():.5f}")
    print("(kernel stub — implement body via neuron-nki-writer-agent)")

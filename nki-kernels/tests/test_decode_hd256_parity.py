# Copyright Armin Aghaeb. SPDX-License-Identifier: Apache-2.0
"""Parity test for the head_dim=256 decode-attention kernel.

Validates the NKI kernel against the pure-PyTorch reference on a sweep
of representative shapes. Cosine threshold is 0.999 (matches the
DeltaNet kernel's bar in PR #152).

Modes:
  - CPU fallback (default if no neuron device): the wrapper routes to
    the PyTorch reference, parity is trivially exact.
  - On-device: if a neuron device is present and `vllm_neuron.nki.nki_hop`
    is available, the wrapper calls the NKI kernel.
  - NKI sim (advanced): set NKI_SIMULATE=1 to run the kernel through the
    nki simulator on CPU. This actually exercises the kernel body and is
    the right mode for development.

Run: pytest tests/test_decode_hd256_parity.py -v
"""
from __future__ import annotations

import pytest
import torch

from armin_nki_kernels.attention.ref_decode_hd256 import (
    decode_hd256_ref,
    make_test_inputs,
)
from armin_nki_kernels.attention.decode_hd256_wrap import decode_hd256


SHAPES = [
    # (name, B, Nh, S_q, S_ctx, valid_len)
    ("smoke",         1,  8, 1,  128,  64),
    ("4B_short",      1, 16, 1,  128, 100),
    ("4B_typical",    1, 16, 1,  512, 400),
    ("27B_typical",   1, 24, 1,  512, 400),
    ("4B_chunked",    1, 16, 1, 4096, 2048),
]


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float().flatten()
    b = b.float().flatten()
    return float(torch.dot(a, b) / (a.norm() * b.norm() + 1e-12))


@pytest.mark.parametrize("name,B,Nh,S_q,S_ctx,valid_len", SHAPES)
def test_decode_hd256_parity(name, B, Nh, S_q, S_ctx, valid_len):
    """Wrapper output (kernel or fallback) must match reference cos > 0.999."""
    inputs = make_test_inputs(
        B=B, Nh=Nh, S_q=S_q, S_ctx=S_ctx,
        valid_len=valid_len,
        dtype=torch.bfloat16,
    )
    ref_out = decode_hd256_ref(**inputs)
    actual_out = decode_hd256(**inputs)

    cos = _cosine(actual_out, ref_out)
    max_abs = (actual_out.float() - ref_out.float()).abs().max().item()

    print(f"  {name}: cos={cos:.6f}  max_abs={max_abs:.6f}")
    assert cos > 0.999, (
        f"{name}: cosine {cos:.6f} below 0.999 threshold "
        f"(max_abs={max_abs:.6f})"
    )


def test_reference_runs():
    """Sanity: the reference itself produces non-trivial output."""
    inputs = make_test_inputs(B=1, Nh=8, S_q=1, S_ctx=128, valid_len=64)
    out = decode_hd256_ref(**inputs)
    assert out.shape == (1, 8, 1, 256)
    assert out.dtype == torch.bfloat16
    assert out.float().std().item() > 1e-3, "output should not be all zeros"


def test_wrapper_falls_back_on_cpu():
    """On CPU (no Neuron), the wrapper should return the reference output."""
    inputs = make_test_inputs(B=1, Nh=8, S_q=1, S_ctx=128, valid_len=64)
    ref = decode_hd256_ref(**inputs)
    actual = decode_hd256(**inputs)
    cos = _cosine(actual, ref)
    assert cos > 0.9999, f"CPU fallback should match reference exactly, got cos={cos:.6f}"


if __name__ == "__main__":
    # Quick self-check.
    test_reference_runs()
    test_wrapper_falls_back_on_cpu()
    print("✓ reference-runs")
    print("✓ wrapper-falls-back-on-cpu")
    for name, B, Nh, S_q, S_ctx, valid_len in SHAPES:
        try:
            test_decode_hd256_parity(name, B, Nh, S_q, S_ctx, valid_len)
            print(f"✓ {name}")
        except AssertionError as e:
            print(f"✗ {name}: {e}")

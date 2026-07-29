"""Tests: fused interleaved-RoPE NKI kernel vs CPU reference.

Structured so it runs on-device when the Neuron/NKI stack is importable, and
skips cleanly otherwise (e.g. on macOS, where the Neuron compiler is absent).

What is validated where:
  * CPU reference math (``rope_ref``) is proven bit-exact against the real port
    helper ``mochi_neuron_attention.apply_rotary_emb`` -- ZERO error, since the
    torch reference is a line-for-line copy. Runs wherever torch + the port
    import (skips only if the port cannot be imported). Also checks numpy vs
    torch fp32 parity.
  * NKI kernel output vs CPU reference -- runs ONLY on device (kernel compiled
    via ``@nki.jit``). Skipped otherwise. "On device" is gated on
    ``torch.device("neuron")`` being constructible (NOT torch_xla).

Tolerances (per project CLAUDE.md): bf16 atol=rtol=1e-2, fp32 atol=rtol=1e-5.

Run:  python -m pytest test_rope.py -v
  or: python test_rope.py        (prints a summary, no pytest needed)
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

import rope_ref as R

try:
    import torch

    _HAS_TORCH = True
except ImportError:
    torch = None  # type: ignore[assignment]
    _HAS_TORCH = False

# NKI import is deferred: absent on macOS / non-Neuron hosts.
try:
    import rope_nki as K  # noqa: F401  (imports nki)

    _HAS_NKI = True
except (ImportError, ModuleNotFoundError):
    K = None  # type: ignore[assignment]
    _HAS_NKI = False

# The real port helper. Importing it requires src/ on the path and the
# neuron_compat shim (torch-only, no device). Gated so a missing src/ or a
# heavier dependency does not break the whole suite.
_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.normpath(os.path.join(_HERE, "..", "..", "src"))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
try:
    from mochi_neuron_attention import apply_rotary_emb as _port_apply_rotary_emb

    _HAS_PORT = True
except Exception:  # pragma: no cover - src not importable in some envs
    _port_apply_rotary_emb = None  # type: ignore[assignment]
    _HAS_PORT = False


def _neuron_device_available() -> bool:
    """True iff a real Neuron device is addressable (NOT torch_xla)."""
    if not _HAS_TORCH:
        return False
    try:
        torch.device("neuron")  # constructible only where the backend is present
        # Constructing the device object can succeed lazily; also require the NKI
        # stack to have imported, since the kernel cannot run without it.
        return _HAS_NKI
    except (RuntimeError, AssertionError, ValueError):
        return False


_ON_DEVICE = _neuron_device_available()

# (B, S, H, D): small-first, then Mochi-like TP=4 local shapes (H=6, D=128).
# Includes exact-tile and partial-tile S, and both CFG batch sizes.
SHAPES = [
    (1, 128, 1, 128),    # smallest, single head, exact tile
    (1, 200, 6, 128),    # partial S tile, full local head count
    (2, 256, 6, 128),    # CFG batch, exact tiles
    (1, 1024, 6, 128),   # larger sequence, multi-tile
    (2, 300, 6, 128),    # CFG + partial tile
]

requires_torch = pytest.mark.skipif(not _HAS_TORCH, reason="torch not installed")
requires_port = pytest.mark.skipif(
    not _HAS_PORT, reason="mochi_neuron_attention (port) not importable"
)
requires_device = pytest.mark.skipif(
    not _ON_DEVICE, reason="no Neuron device / NKI stack (expected off-device, e.g. macOS)"
)


# ---------------------------------------------------------------------------
# Reference correctness (host-only, no device needed)
# ---------------------------------------------------------------------------
@requires_torch
@requires_port
@pytest.mark.parametrize("B,S,H,D", SHAPES)
@pytest.mark.parametrize("which", ["q", "k"])
def test_reference_matches_port(B, S, H, D, which):
    """CPU torch reference == the real port ``apply_rotary_emb`` with ZERO error.

    ``which`` is cosmetic (Q vs K take the same code path); both are exercised to
    document that RoPE is applied identically to visual queries and keys.
    """
    torch.manual_seed(0 if which == "q" else 1)
    x = torch.randn(B, S, H, D, dtype=torch.bfloat16)
    angle = torch.randn(S, H, D // 2, dtype=torch.float32)
    freqs_cos = torch.cos(angle)
    freqs_sin = torch.sin(angle)

    ref = R.apply_rotary_emb_torch(x, freqs_cos, freqs_sin)
    port = _port_apply_rotary_emb(x, freqs_cos, freqs_sin)
    assert torch.equal(ref, port), "reference diverges from the port's apply_rotary_emb"


@requires_torch
@pytest.mark.parametrize("B,S,H,D", [(1, 128, 1, 128), (2, 256, 6, 128)])
def test_numpy_matches_torch_fp32(B, S, H, D):
    """NumPy fp32 path matches the torch fp32 path within fp32 tolerance."""
    x, fc, fs = R.make_inputs_np(B, S, H, D, seed=0, dtype="float32")
    n = R.apply_rotary_emb_np(x, fc, fs)
    t = R.apply_rotary_emb_torch(
        torch.from_numpy(x), torch.from_numpy(fc), torch.from_numpy(fs)
    ).numpy()
    np.testing.assert_allclose(n, t, atol=1e-5, rtol=1e-5)


@requires_torch
def test_reference_interleave_layout():
    """Guard the crux: cos lands on even output cols, sin on odd (flattened D)."""
    B, S, H, D = 1, 4, 2, 8
    x = torch.zeros(B, S, H, D, dtype=torch.float32)
    # Choose freqs so cos_out == x_even and sin_out == x_odd: fc=1, fs=0.
    fc = torch.ones(S, H, D // 2)
    fs = torch.zeros(S, H, D // 2)
    x_even_vals = torch.arange(H * D // 2).reshape(1, 1, H, D // 2).float()
    x_odd_vals = -x_even_vals
    x[..., 0::2] = x_even_vals
    x[..., 1::2] = x_odd_vals
    out = R.apply_rotary_emb_torch(x, fc, fs)
    assert torch.equal(out[..., 0::2], x_even_vals.expand(B, S, H, D // 2))
    assert torch.equal(out[..., 1::2], x_odd_vals.expand(B, S, H, D // 2))


# ---------------------------------------------------------------------------
# NKI kernel vs CPU reference (device-only)
# ---------------------------------------------------------------------------
@requires_torch
@requires_device
@pytest.mark.parametrize("B,S,H,D", SHAPES)
def test_rope_nki(B, S, H, D):
    """apply_rotary_emb NKI kernel matches the CPU reference (bf16 tolerance)."""
    torch.manual_seed(0)
    x = torch.randn(B, S, H, D, dtype=torch.bfloat16)
    angle = torch.randn(S, H, D // 2, dtype=torch.float32)
    freqs_cos = torch.cos(angle)
    freqs_sin = torch.sin(angle)

    ref = R.apply_rotary_emb_torch(x, freqs_cos, freqs_sin)
    got = torch.as_tensor(K.apply_rotary_emb(x, freqs_cos, freqs_sin))

    torch.testing.assert_close(got.float(), ref.float(), atol=1e-2, rtol=1e-2)


# ---------------------------------------------------------------------------
# Standalone runner (no pytest)
# ---------------------------------------------------------------------------
def _main() -> None:
    if not _HAS_TORCH:
        print("torch not installed -- cannot run reference checks.")
        return

    print("== CPU reference vs the port's apply_rotary_emb (expect PASS, 0 error) ==")
    if not _HAS_PORT:
        print("  port not importable -- skipped (check src/ on path).")
    else:
        for B, S, H, D in SHAPES:
            torch.manual_seed(0)
            x = torch.randn(B, S, H, D, dtype=torch.bfloat16)
            angle = torch.randn(S, H, D // 2, dtype=torch.float32)
            fc, fs = torch.cos(angle), torch.sin(angle)
            ok = torch.equal(
                R.apply_rotary_emb_torch(x, fc, fs),
                _port_apply_rotary_emb(x, fc, fs),
            )
            print(f"  {(B, S, H, D)!s:>20}  {'PASS' if ok else 'FAIL'}")

    print("\n== NumPy vs torch fp32 parity ==")
    for B, S, H, D in [(1, 128, 1, 128), (2, 256, 6, 128)]:
        x, fc, fs = R.make_inputs_np(B, S, H, D, seed=0)
        n = R.apply_rotary_emb_np(x, fc, fs)
        t = R.apply_rotary_emb_torch(
            torch.from_numpy(x), torch.from_numpy(fc), torch.from_numpy(fs)
        ).numpy()
        maxabs = float(np.abs(n - t).max())
        print(f"  {(B, S, H, D)!s:>20}  maxabs={maxabs:.3e}")

    if _ON_DEVICE:
        print("\n== NKI kernel vs CPU reference (on device) ==")
        for B, S, H, D in SHAPES:
            torch.manual_seed(0)
            x = torch.randn(B, S, H, D, dtype=torch.bfloat16)
            angle = torch.randn(S, H, D // 2, dtype=torch.float32)
            fc, fs = torch.cos(angle), torch.sin(angle)
            ref = R.apply_rotary_emb_torch(x, fc, fs)
            got = torch.as_tensor(K.apply_rotary_emb(x, fc, fs))
            maxabs = (got.float() - ref.float()).abs().max().item()
            print(f"  {(B, S, H, D)!s:>20}  maxabs={maxabs:.4e}")
    else:
        print("\nNo Neuron device -- device checks skipped (expected on macOS).")


if __name__ == "__main__":
    _main()

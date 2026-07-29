"""Tests: fused modulated RMSNorm NKI kernel vs CPU reference.

Structured so it runs on-device when the Neuron/NKI stack is importable, and
skips cleanly otherwise (e.g. on macOS, where the Neuron compiler is absent).

What is validated where:
  * CPU reference math (``rmsnorm_ref``) is proven bit-exact against the upstream
    tiled helper in ``rmsnorm_ref``'s own docstring checks and in
    ``test_reference_matches_upstream`` below -- runs everywhere torch exists.
  * NKI kernel output vs CPU reference -- runs ONLY on device (kernels compiled
    via ``@nki.jit``). Skipped otherwise.

Tolerances (per project CLAUDE.md): bf16 atol=rtol=1e-2, fp32 atol=rtol=1e-5.

Run:  python -m pytest test_rmsnorm.py -v
  or: python test_rmsnorm.py        (prints a summary, no pytest needed)
"""
from __future__ import annotations

import numpy as np
import pytest

import rmsnorm_ref as R

try:
    import torch

    _HAS_TORCH = True
except ImportError:
    torch = None  # type: ignore[assignment]
    _HAS_TORCH = False

# NKI import is deferred: absent on macOS / non-Neuron hosts.
try:
    import rmsnorm_nki as K  # noqa: F401  (imports neuronxcc.nki)

    _HAS_NKI = True
except (ImportError, ModuleNotFoundError):
    K = None  # type: ignore[assignment]
    _HAS_NKI = False

EPS = 1e-6

# (B, S, D): small-first, then Mochi-like visual and context shapes.
SHAPES = [
    (1, 128, 256),    # smallest, exact tile
    (2, 200, 256),    # partial S tile + CFG batch
    (1, 1024, 3072),  # Mochi visual
    (2, 512, 1536),   # Mochi context, CFG
]

requires_torch = pytest.mark.skipif(not _HAS_TORCH, reason="torch not installed")
requires_nki = pytest.mark.skipif(
    not _HAS_NKI, reason="NKI/neuronxcc not available (not on a Neuron host)"
)


# ---------------------------------------------------------------------------
# Reference correctness (host-only, no device needed)
# ---------------------------------------------------------------------------
@requires_torch
@pytest.mark.parametrize("B,S,D", SHAPES)
@pytest.mark.parametrize("scale_form", ["none", "broadcast", "per_position"])
def test_reference_matches_upstream(B, S, D, scale_form):
    """CPU reference == upstream ``_rms_normalize_tiled`` math, all scale forms."""
    torch.manual_seed(0)
    x = torch.randn(B, S, D, dtype=torch.bfloat16)
    if scale_form == "none":
        scale = None
    elif scale_form == "broadcast":
        scale = torch.randn(B, 1, D, dtype=torch.bfloat16)
    else:
        scale = torch.randn(B, S, D, dtype=torch.bfloat16)

    untiled = R.rms_normalize_torch(x, EPS, scale)
    tiled = R.rms_normalize_tiled_torch(x, EPS, scale, tile=64)  # force tiling
    assert torch.equal(untiled, tiled), "reference diverges from upstream tiled math"


@requires_torch
@pytest.mark.parametrize("B,S,D", [(1, 128, 256), (2, 512, 1536)])
def test_numpy_matches_torch_fp32(B, S, D):
    """NumPy fp32 path matches torch fp32 path within fp32 tolerance."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal((B, S, D)).astype(np.float32)
    s = rng.standard_normal((B, S, D)).astype(np.float32)
    n = R.rms_normalize_np(x, EPS, s)
    t = R.rms_normalize_torch(torch.from_numpy(x), EPS, torch.from_numpy(s)).numpy()
    np.testing.assert_allclose(n, t, atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------------
# NKI kernel vs CPU reference (device-only)
# ---------------------------------------------------------------------------
@requires_torch
@requires_nki
@pytest.mark.parametrize("B,S,D", SHAPES)
@pytest.mark.parametrize("scale_form", ["none", "broadcast", "per_position"])
def test_modulated_rmsnorm_nki(B, S, D, scale_form):
    """modulated_rmsnorm kernel matches the CPU reference (bf16 tolerance)."""
    torch.manual_seed(0)
    x = torch.randn(B, S, D, dtype=torch.bfloat16)
    if scale_form == "none":
        scale = None
        scale_np = None
    elif scale_form == "broadcast":
        scale = torch.randn(B, 1, D, dtype=torch.bfloat16)
        scale_np = scale
    else:
        scale = torch.randn(B, S, D, dtype=torch.bfloat16)
        scale_np = scale

    ref = R.rms_normalize_torch(x, EPS, scale_np)

    args = (x,) if scale is None else (x, scale)
    got = K.modulated_rmsnorm(*args, eps=EPS)
    got = torch.as_tensor(got)

    torch.testing.assert_close(got.float(), ref.float(), atol=1e-2, rtol=1e-2)


@requires_torch
@requires_nki
@pytest.mark.parametrize("B,S,D", SHAPES)
def test_rmsnorm_zero_core_nki(B, S, D):
    """rmsnorm_zero_core kernel matches ``rmsnorm(x) * (1 + scale_msa[:,None])``."""
    torch.manual_seed(0)
    x = torch.randn(B, S, D, dtype=torch.bfloat16)
    scale_msa = torch.randn(B, D, dtype=torch.bfloat16)

    ref = R.rmsnorm_zero_core_torch(x, scale_msa, EPS)
    got = torch.as_tensor(K.rmsnorm_zero_core(x, scale_msa, eps=EPS))

    torch.testing.assert_close(got.float(), ref.float(), atol=1e-2, rtol=1e-2)


# ---------------------------------------------------------------------------
# Standalone runner (no pytest)
# ---------------------------------------------------------------------------
def _main() -> None:
    if not _HAS_TORCH:
        print("torch not installed -- cannot run reference checks.")
        return

    print("== CPU reference vs upstream tiled math ==")
    for B, S, D in SHAPES:
        for form in ("none", "broadcast", "per_position"):
            torch.manual_seed(0)
            x = torch.randn(B, S, D, dtype=torch.bfloat16)
            scale = {
                "none": None,
                "broadcast": torch.randn(B, 1, D, dtype=torch.bfloat16),
                "per_position": torch.randn(B, S, D, dtype=torch.bfloat16),
            }[form]
            ok = torch.equal(
                R.rms_normalize_torch(x, EPS, scale),
                R.rms_normalize_tiled_torch(x, EPS, scale, tile=64),
            )
            print(f"  {(B, S, D)!s:>18}  {form:<13} {'PASS' if ok else 'FAIL'}")

    if _HAS_NKI:
        print("\n== NKI kernel vs CPU reference (on device) ==")
        for B, S, D in SHAPES:
            torch.manual_seed(0)
            x = torch.randn(B, S, D, dtype=torch.bfloat16)
            ref = R.rms_normalize_torch(x, EPS, None)
            got = torch.as_tensor(K.modulated_rmsnorm(x, eps=EPS))
            maxabs = (got.float() - ref.float()).abs().max().item()
            print(f"  {(B, S, D)!s:>18}  maxabs={maxabs:.4e}")
    else:
        print("\nNKI not available on this host -- device checks skipped (expected on macOS).")


if __name__ == "__main__":
    _main()

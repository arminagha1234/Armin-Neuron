"""Tests for the fused SwiGLU FFN NKI kernel vs the CPU reference.

Two layers of validation, matching the CLAUDE.md pipeline:

  1. CPU-only (always runs, no Neuron needed): verifies the numpy reference is
     self-consistent — full unsharded == sum of TP-local partials — and that
     `pretranspose_weights` round-trips the layout the kernel expects.

  2. On-device (runs only when NKI + an XLA/Neuron device are importable, else
     skipped cleanly): compiles `swiglu_ffn_nki` and compares against the CPU
     reference with bf16 tolerances (atol=1e-2, rtol=1e-2).

Run:  python test_swiglu.py            # CPU checks; device checks if available
      pytest test_swiglu.py            # same, as pytest cases

Environment note: on macOS the Neuron compiler is absent, so the device tests
skip and only the numpy validation runs. On the Beta-3 DLC they execute.
"""
from __future__ import annotations

import numpy as np

from swiglu_ref import (
    IN_FEATURES,
    INNER,
    OUT_FEATURES,
    shard_w0_glu,
    shard_w2_row,
    swiglu_ffn_ref,
    swiglu_ffn_tp_allreduce,
    swiglu_ffn_tp_local,
)

BF16_ATOL = 1e-2
BF16_RTOL = 1e-2


# ── Shape sets: smallest first (CLAUDE.md), then Mochi-like ──────────────────
# (S, IN, INNER, OUT)
SMALL = (128, 256, 512, 256)
MOCHI_LOCAL_TP4 = (256, IN_FEATURES, INNER // 4, OUT_FEATURES)   # per-rank at TP=4
MOCHI_FULL = (256, IN_FEATURES, INNER, OUT_FEATURES)             # unsharded


def _rand_weights(S, IN, P, OUT, seed=0):
    rng = np.random.default_rng(seed)
    x = (rng.standard_normal((S, IN)) * 0.1).astype(np.float32)
    # W0 is [2*P, IN] already in value|gate-paired layout for a standalone run.
    w0 = (rng.standard_normal((2 * P, IN)) * 0.02).astype(np.float32)
    w2 = (rng.standard_normal((OUT, P)) * 0.02).astype(np.float32)
    return x, w0, w2


# ═══════════════════════════ CPU-only checks ════════════════════════════════
def test_reference_tp_partition_sum():
    """full unsharded == sum of TP-local partials (reproduces NOTES.md check)."""
    rng = np.random.default_rng(1)
    x = (rng.standard_normal((64, IN_FEATURES)) * 0.1).astype(np.float32)
    w0 = (rng.standard_normal((2 * INNER, IN_FEATURES)) * 0.02).astype(np.float32)
    w2 = (rng.standard_normal((OUT_FEATURES, INNER)) * 0.02).astype(np.float32)

    full = swiglu_ffn_ref(x, w0, w2)
    for ws in (1, 2, 4, 8):
        summed = swiglu_ffn_tp_allreduce(x, w0, w2, ws)
        err = np.max(np.abs(full - summed))
        assert err < 1e-2, f"TP={ws} partition-sum mismatch: {err:.2e}"


def test_pretranspose_roundtrip():
    """pretranspose_weights produces the [IN,2P]/[P,OUT] layout the kernel reads."""
    from swiglu_nki import pretranspose_weights

    _, _, P, _ = MOCHI_LOCAL_TP4
    rng = np.random.default_rng(2)
    w0_local = rng.standard_normal((2 * P, IN_FEATURES)).astype(np.float32)
    w2_local = rng.standard_normal((OUT_FEATURES, P)).astype(np.float32)
    w0t, w2t = pretranspose_weights(w0_local, w2_local)
    assert w0t.shape == (IN_FEATURES, 2 * P)
    assert w2t.shape == (P, OUT_FEATURES)
    assert np.allclose(w0t.T, w0_local)
    assert np.allclose(w2t.T, w2_local)


# ═══════════════════════════ Device availability ════════════════════════════
def _device_available():
    """Return (nki_module, device) if runnable on-device, else (None, None)."""
    try:
        try:
            import nki  # noqa: F401
        except ImportError:
            import neuronxcc.nki as nki  # noqa: F401
        import torch  # noqa: F401
        from torch_xla.core import xla_model as xm
        device = xm.xla_device()
        return nki, device
    except Exception:
        return None, None


def _run_device_case(S, IN, P, OUT, seed=0):
    """Compile+run the kernel and compare to the CPU reference. Returns max err."""
    import torch
    from torch_xla.core import xla_model as xm
    from swiglu_nki import swiglu_ffn_nki, pretranspose_weights

    x_np, w0_np, w2_np = _rand_weights(S, IN, P, OUT, seed)

    # CPU reference (fp32) — this weight is already value|gate paired, so the
    # "TP-local" reference with the identity shard is the ground truth.
    ref = swiglu_ffn_tp_local(x_np, w0_np, w2_np)

    device = xm.xla_device()
    x = torch.from_numpy(x_np).to(torch.bfloat16).to(device)
    w0_local = torch.from_numpy(w0_np).to(torch.bfloat16).to(device)
    w2_local = torch.from_numpy(w2_np).to(torch.bfloat16).to(device)
    w0t, w2t = pretranspose_weights(w0_local, w2_local)

    out = swiglu_ffn_nki(x, w0t, w2t)
    out_cpu = out.cpu().to(torch.float32).numpy()

    abs_diff = np.abs(out_cpu - ref)
    max_err = float(np.max(abs_diff))
    ok = np.allclose(out_cpu, ref, atol=BF16_ATOL, rtol=BF16_RTOL)
    return max_err, ok


def test_device_small():
    nki, device = _device_available()
    if nki is None:
        import pytest
        pytest.skip("NKI / Neuron device not available (expected on macOS)")
    max_err, ok = _run_device_case(*SMALL, seed=10)
    assert ok, f"small shape device mismatch: max|err|={max_err:.3e}"


def test_device_mochi_local_tp4():
    nki, device = _device_available()
    if nki is None:
        import pytest
        pytest.skip("NKI / Neuron device not available (expected on macOS)")
    max_err, ok = _run_device_case(*MOCHI_LOCAL_TP4, seed=11)
    assert ok, f"Mochi TP=4-local device mismatch: max|err|={max_err:.3e}"


def test_device_mochi_full():
    nki, device = _device_available()
    if nki is None:
        import pytest
        pytest.skip("NKI / Neuron device not available (expected on macOS)")
    max_err, ok = _run_device_case(*MOCHI_FULL, seed=12)
    assert ok, f"Mochi full device mismatch: max|err|={max_err:.3e}"


# ═══════════════════════════ Script entrypoint ══════════════════════════════
def _main():
    print("=" * 66)
    print("CPU reference validation (always runs)")
    print("=" * 66)
    test_reference_tp_partition_sum()
    print("  test_reference_tp_partition_sum        PASSED")
    test_pretranspose_roundtrip()
    print("  test_pretranspose_roundtrip            PASSED")

    print("=" * 66)
    nki, device = _device_available()
    if nki is None:
        print("Device tests SKIPPED — NKI/Neuron not importable on this host.")
        print("  (Expected on macOS. Re-run on the Beta-3 DLC for on-device")
        print("   validation of swiglu_ffn_nki against the CPU reference.)")
        print("=" * 66)
        return

    print("Device validation (NKI available)")
    print("=" * 66)
    for name, shape, seed in (
        ("small          ", SMALL, 10),
        ("mochi_local_tp4", MOCHI_LOCAL_TP4, 11),
        ("mochi_full     ", MOCHI_FULL, 12),
    ):
        max_err, ok = _run_device_case(*shape, seed=seed)
        status = "PASSED" if ok else "FAILED"
        print(f"  {name}  S,IN,P,OUT={shape}  max|err|={max_err:.3e}  {status}")
    print("=" * 66)


if __name__ == "__main__":
    _main()

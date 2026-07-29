"""Tests for the fused QKV-projection NKI kernel vs the CPU reference.

Two layers of validation, matching the CLAUDE.md pipeline:

  1. CPU-only (always runs, no Neuron needed): verifies the numpy reference is
     exact — fused == concat(separate q/k/v) — that the TP-local fused weight
     reassembles to the full output, and that `pretranspose_fused_weight`
     round-trips the [IN, OUT] layout the kernel expects.

  2. On-device (runs only when the NKI SDK and a `torch.device("neuron")` are
     available, else skipped cleanly): compiles `fused_qkv_projection_nki` and
     compares against the CPU reference with bf16 tolerances.

Run:  python test_fused_qkv.py          # CPU checks; device checks if available
      pytest test_fused_qkv.py          # same, as pytest cases

Device gating note: we probe `torch.device("neuron")` (per the task brief) — NOT
torch_xla, which is not installed on the target box. On macOS both the NKI SDK
and the neuron device are absent, so device tests skip and only the numpy
validation runs. Another engineer runs the device tests on the trn2 DLC.
"""
from __future__ import annotations

import numpy as np

from fused_qkv_ref import (
    INNER_DIM,
    TEXT_DIM,
    N_QKV,
    build_fused_qkv_weight,
    fused_projection_ref,
    qkv_fused_ref,
    qkv_separate_ref,
    shard_fused_qkv_weight,
    split_qkv_output,
)

# bf16 tolerances (device path); the CPU-only fused==separate check is exact.
BF16_ATOL = 1e-2
BF16_RTOL = 1e-2
# fp32 tolerance for the exactness checks.
FP32_TOL = 1e-4


# ── Shape sets: smallest first (CLAUDE.md), then Mochi-like ──────────────────
# (S, IN, PROJ_WIDTH)  where OUT_TOTAL = 3 * PROJ_WIDTH.
SMALL = (128, 256, 128)
# Visual stream at TP=4: IN=3072, per-rank q/k/v width = 3072//4 = 768.
VISUAL_TP4 = (256, INNER_DIM, INNER_DIM // 4)
# Text stream at TP=4: IN=1536, per-rank q/k/v width = 768.
TEXT_TP4 = (256, TEXT_DIM, INNER_DIM // 4)
# Visual stream full (unsharded): OUT_TOTAL = 9216.
VISUAL_FULL = (256, INNER_DIM, INNER_DIM)
# Text stream full: IN=1536, OUT_TOTAL = 9216.
TEXT_FULL = (256, TEXT_DIM, INNER_DIM)


def _rand_qkv(S, IN, proj_width, seed=0):
    """Random x and three [proj_width, IN] projection weights (fp32)."""
    rng = np.random.default_rng(seed)
    x = (rng.standard_normal((S, IN)) * 0.1).astype(np.float32)
    wq = (rng.standard_normal((proj_width, IN)) * 0.02).astype(np.float32)
    wk = (rng.standard_normal((proj_width, IN)) * 0.02).astype(np.float32)
    wv = (rng.standard_normal((proj_width, IN)) * 0.02).astype(np.float32)
    return x, wq, wk, wv


# ═══════════════════════════ CPU-only checks ════════════════════════════════
def test_fused_equals_separate_exact():
    """fused (one matmul + split) == separate (three matmuls), bit-exact fp32."""
    for IN in (INNER_DIM, TEXT_DIM):
        x, wq, wk, wv = _rand_qkv(64, IN, INNER_DIM, seed=1)
        qs, ks, vs = qkv_separate_ref(x, wq, wk, wv)
        qf, kf, vf = qkv_fused_ref(x, wq, wk, wv)
        err = max(
            np.max(np.abs(qs - qf)),
            np.max(np.abs(ks - kf)),
            np.max(np.abs(vs - vf)),
        )
        assert err <= FP32_TOL, f"IN={IN}: fused != separate, max|err|={err:.2e}"


def test_tp_local_reassembly_exact():
    """Sum/concat of per-rank fused-local outputs reconstructs the full q/k/v."""
    for IN in (INNER_DIM, TEXT_DIM):
        x, wq, wk, wv = _rand_qkv(48, IN, INNER_DIM, seed=2)
        qf, kf, vf = qkv_fused_ref(x, wq, wk, wv)
        for W in (1, 2, 4, 8):
            per = INNER_DIM // W
            q_parts, k_parts, v_parts = [], [], []
            for r in range(W):
                w_local = shard_fused_qkv_weight(wq, wk, wv, r, W)
                assert w_local.shape == (N_QKV * per, IN)
                out_local = fused_projection_ref(x, w_local)
                ql, kl, vl = split_qkv_output(out_local, per)
                q_parts.append(ql)
                k_parts.append(kl)
                v_parts.append(vl)
            q_re = np.concatenate(q_parts, axis=1)
            k_re = np.concatenate(k_parts, axis=1)
            v_re = np.concatenate(v_parts, axis=1)
            err = max(
                np.max(np.abs(q_re - qf)),
                np.max(np.abs(k_re - kf)),
                np.max(np.abs(v_re - vf)),
            )
            assert err <= FP32_TOL, f"IN={IN} TP={W}: reassembly err {err:.2e}"


def test_pretranspose_roundtrip():
    """pretranspose_fused_weight produces the [IN, OUT] layout the kernel reads."""
    from fused_qkv_nki import pretranspose_fused_weight

    x, wq, wk, wv = _rand_qkv(8, INNER_DIM, INNER_DIM, seed=3)
    w_fused = build_fused_qkv_weight(wq, wk, wv)   # [3*OUT, IN]
    assert w_fused.shape == (N_QKV * INNER_DIM, INNER_DIM)
    wt = pretranspose_fused_weight(w_fused)
    assert wt.shape == (INNER_DIM, N_QKV * INNER_DIM)
    assert np.allclose(wt.T, w_fused)


# ═══════════════════════════ Device availability ════════════════════════════
def _device_available():
    """Return (nki_module, device) if runnable on-device, else (None, None).

    Probes `torch.device("neuron")` per the task brief (NOT torch_xla, which is
    absent on the target box). Requires the NKI SDK importable too.
    """
    try:
        try:
            import nki  # noqa: F401
        except ImportError:
            import neuronxcc.nki as nki  # noqa: F401
        import torch
        device = torch.device("neuron")   # raises on hosts without the backend
        return nki, device
    except Exception:
        return None, None


def _run_device_case(S, IN, proj_width, seed=0):
    """Compile+run the kernel and compare to the CPU reference. Returns (err, ok)."""
    import torch
    from fused_qkv_nki import fused_qkv_projection_nki, pretranspose_fused_weight

    x_np, wq, wk, wv = _rand_qkv(S, IN, proj_width, seed)
    w_fused = build_fused_qkv_weight(wq, wk, wv)          # [3*proj_width, IN]

    # CPU reference (fp32).
    ref = fused_projection_ref(x_np, w_fused)            # [S, 3*proj_width]

    device = torch.device("neuron")
    x = torch.from_numpy(x_np).to(torch.bfloat16).to(device)
    w_fused_t = torch.from_numpy(w_fused).to(torch.bfloat16).to(device)
    wt = pretranspose_fused_weight(w_fused_t)            # [IN, 3*proj_width]

    out = fused_qkv_projection_nki(x, wt)
    out_cpu = out.cpu().to(torch.float32).numpy()

    max_err = float(np.max(np.abs(out_cpu - ref)))
    ok = np.allclose(out_cpu, ref, atol=BF16_ATOL, rtol=BF16_RTOL)

    # Also sanity-check the q/k/v split lands on the expected slices.
    q, k, v = split_qkv_output(out_cpu, proj_width)
    assert q.shape == k.shape == v.shape == (S, proj_width)
    return max_err, ok


def test_device_small():
    nki, device = _device_available()
    if nki is None:
        import pytest
        pytest.skip("NKI / Neuron device not available (expected on macOS)")
    max_err, ok = _run_device_case(*SMALL, seed=10)
    assert ok, f"small shape device mismatch: max|err|={max_err:.3e}"


def test_device_visual_tp4():
    nki, device = _device_available()
    if nki is None:
        import pytest
        pytest.skip("NKI / Neuron device not available (expected on macOS)")
    max_err, ok = _run_device_case(*VISUAL_TP4, seed=11)
    assert ok, f"visual TP=4 device mismatch: max|err|={max_err:.3e}"


def test_device_text_tp4():
    nki, device = _device_available()
    if nki is None:
        import pytest
        pytest.skip("NKI / Neuron device not available (expected on macOS)")
    max_err, ok = _run_device_case(*TEXT_TP4, seed=12)
    assert ok, f"text TP=4 device mismatch: max|err|={max_err:.3e}"


def test_device_visual_full():
    nki, device = _device_available()
    if nki is None:
        import pytest
        pytest.skip("NKI / Neuron device not available (expected on macOS)")
    max_err, ok = _run_device_case(*VISUAL_FULL, seed=13)
    assert ok, f"visual full device mismatch: max|err|={max_err:.3e}"


# ═══════════════════════════ Script entrypoint ══════════════════════════════
def _main():
    print("=" * 66)
    print("CPU reference validation (always runs)")
    print("=" * 66)
    test_fused_equals_separate_exact()
    print("  test_fused_equals_separate_exact       PASSED")
    test_tp_local_reassembly_exact()
    print("  test_tp_local_reassembly_exact         PASSED")
    test_pretranspose_roundtrip()
    print("  test_pretranspose_roundtrip            PASSED")

    print("=" * 66)
    nki, device = _device_available()
    if nki is None:
        print("Device tests SKIPPED — NKI/Neuron not available on this host.")
        print("  (Expected on macOS. Re-run on the trn2 DLC for on-device")
        print("   validation of fused_qkv_projection_nki vs the CPU reference.)")
        print("=" * 66)
        return

    print("Device validation (NKI available)")
    print("=" * 66)
    for name, shape, seed in (
        ("small        ", SMALL, 10),
        ("visual_tp4   ", VISUAL_TP4, 11),
        ("text_tp4     ", TEXT_TP4, 12),
        ("visual_full  ", VISUAL_FULL, 13),
        ("text_full    ", TEXT_FULL, 14),
    ):
        max_err, ok = _run_device_case(*shape, seed=seed)
        status = "PASSED" if ok else "FAILED"
        print(f"  {name}  S,IN,PROJ={shape}  max|err|={max_err:.3e}  {status}")
    print("=" * 66)


if __name__ == "__main__":
    _main()

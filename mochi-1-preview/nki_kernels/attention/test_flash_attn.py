"""Validation for the Mochi-1 flash joint-attention NKI kernel.

Compares the NKI kernel output (``flash_attn_nki.flash_attention_kernel``)
against the CPU reference (``flash_attn_ref``) with bf16 tolerances
(``atol=rtol=1e-2`` per CLAUDE.md).

Structure / environments
------------------------
* On a machine WITH the Neuron stack (torch + torch_neuronx + neuronxcc, i.e.
  a trn2 DLC): runs every case on device and asserts closeness. Device
  placement uses ``torch.device("neuron")`` and the ``import nki`` @nki.jit
  path (the ``neuronxcc.nki`` path routes through torch_xla/pyhlo which is not
  installed in this container -- see the header of flash_attn_nki.py).
* On a machine WITHOUT the Neuron device: the on-device cases are SKIPPED. The
  pure-numpy algorithm equivalence check (naive materialised vs online-softmax)
  still runs, since it needs only numpy.

Run standalone (`python test_flash_attn.py`) or under pytest. Standalone prints
a PASS/SKIP/FAIL summary and exits non-zero on failure.

Cases (small -> masked -> tile-crossing), all D=128 (Mochi head_dim):
  1. small unmasked          : P=1, S=256
  2. small unmasked          : P=4, S=256
  3. masked (padded text)    : P=B*H, Sk has padded columns -> -10000 bias
  4. tile-crossing           : P=2, S=1024  (8 k-tiles, 8 q-tiles)
  5. tile-crossing + masked  : P=2, Sq=1024, Sk=1040 padded tail
"""
from __future__ import annotations

import sys
import os

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import flash_attn_ref as ref  # noqa: E402

# ---------------------------------------------------------------------------
# Optional imports: torch + the NKI kernel. Absence -> device cases skip.
# ---------------------------------------------------------------------------
try:
    import torch

    _HAS_TORCH = True
except ImportError:
    torch = None  # type: ignore[assignment]
    _HAS_TORCH = False

_HAS_NKI = False
_NKI_IMPORT_ERR = None
try:
    import nki  # noqa: F401

    from flash_attn_nki import flash_attention_kernel

    _HAS_NKI = True
except Exception as exc:  # noqa: BLE001 - any import/env failure -> skip device path
    _NKI_IMPORT_ERR = exc

# A Neuron device must be reachable to actually place tensors and run the
# kernel. We probe torch.device("neuron") with a trivial transfer.
_HAS_DEVICE = False
_DEVICE_ERR = None
if _HAS_TORCH:
    try:
        _dev = torch.device("neuron")
        _ = torch.zeros(1).to(_dev).to("cpu")
        _HAS_DEVICE = True
    except Exception as exc:  # noqa: BLE001
        _DEVICE_ERR = exc


ATOL = 1e-2
RTOL = 1e-2
D_HEAD = 128
SCALE = 1.0 / (D_HEAD ** 0.5)


# ---------------------------------------------------------------------------
# Case construction
# ---------------------------------------------------------------------------
def _make_case(P, Sq, Sk, D, seed, keep=None, num_heads=1):
    """Build q,k,v (P,S,D) and a (P,Sk) key bias. `keep` is a (B,Sk) {0,1} mask."""
    rng = np.random.default_rng(seed)
    q = rng.standard_normal((P, Sq, D)).astype(np.float32)
    k = rng.standard_normal((P, Sk, D)).astype(np.float32)
    v = rng.standard_normal((P, Sk, D)).astype(np.float32)
    if keep is None:
        key_bias = np.zeros((P, Sk), dtype=np.float32)
    else:
        key_bias = ref.key_bias_from_keep_mask(keep, num_heads)
    return q, k, v, key_bias


def _cases():
    cases = []
    # 1-2: small unmasked.
    cases.append(("small_P1_S256", *_make_case(1, 256, 256, D_HEAD, 1)))
    cases.append(("small_P4_S256", *_make_case(4, 256, 256, D_HEAD, 2)))

    # 3: masked - B=2, H=3 (P=6), Sk=260 with the last 3 text keys padded.
    B, H, Sk = 2, 3, 260
    keep = np.ones((B, Sk), dtype=np.float32)
    keep[:, 257:] = 0.0
    cases.append(("masked_P6_S256_pad3",
                  *_make_case(B * H, 256, Sk, D_HEAD, 3, keep=keep, num_heads=H)))

    # 4: tile-crossing unmasked, S=1024 (8 q-tiles x 8 k-tiles).
    cases.append(("tilecross_P2_S1024", *_make_case(2, 1024, 1024, D_HEAD, 4)))

    # 5: tile-crossing + masked, Sq=1024, Sk=1040 with a padded tail.
    B2, H2, Sk2 = 2, 1, 1040
    keep2 = np.ones((B2, Sk2), dtype=np.float32)
    keep2[:, 1024:] = 0.0  # pad the 16-key tail
    cases.append(("tilecross_masked_P2_Sq1024_Sk1040",
                  *_make_case(B2 * H2, 1024, Sk2, D_HEAD, 5, keep=keep2, num_heads=H2)))
    return cases


# ---------------------------------------------------------------------------
# Numpy-only algorithm equivalence (always runnable, already validated).
# ---------------------------------------------------------------------------
def check_numpy_algorithm_equivalence():
    """Prove online-softmax == naive materialised in pure numpy, all cases."""
    print("\n[numpy] online-softmax vs naive materialised (fp32):")
    ok = True
    for name, q, k, v, key_bias in _cases():
        o_naive = ref.flash_attention_ref_np(q, k, v, key_bias, scale=SCALE)
        o_online = ref.flash_attention_online_np(
            q, k, v, key_bias, scale=SCALE, q_tile=128, k_tile=128
        )
        max_err = float(np.abs(o_naive - o_online).max())
        has_nan = bool(np.isnan(o_online).any())
        passed = (max_err < 1e-4) and not has_nan
        ok = ok and passed
        print(f"  {name:38s} maxerr={max_err:.2e} nan={has_nan} "
              f"{'PASS' if passed else 'FAIL'}")
    return ok


# ---------------------------------------------------------------------------
# On-device NKI validation.
# ---------------------------------------------------------------------------
def _run_kernel_on_device(q_np, k_np, v_np, key_bias_np):
    """Place inputs on the Neuron device, run the kernel, return fp32 numpy out."""
    device = torch.device("neuron")
    q_t = torch.from_numpy(q_np).to(torch.bfloat16).to(device)
    k_t = torch.from_numpy(k_np).to(torch.bfloat16).to(device)
    v_t = torch.from_numpy(v_np).to(torch.bfloat16).to(device)
    bias_t = torch.from_numpy(key_bias_np).to(torch.float32).to(device)

    out_t = flash_attention_kernel(q_t, k_t, v_t, bias_t, SCALE)
    return out_t.to("cpu").to(torch.float32).numpy()


def check_nki_kernel_on_device():
    """Run every case on device, compare to the CPU bf16 reference."""
    print("\n[device] NKI kernel vs CPU reference (bf16 atol=rtol=1e-2):")
    ok = True
    for name, q, k, v, key_bias in _cases():
        # Emulate bf16 rounding of inputs for a fair reference (kernel gets bf16).
        try:
            bf = torch.bfloat16

            def _round_bf16(x):
                return torch.from_numpy(x).to(bf).to(torch.float32).numpy()

            ref_out = ref.flash_attention_ref_np(
                _round_bf16(q), _round_bf16(k), _round_bf16(v),
                key_bias, scale=SCALE,
            )
        except Exception:  # noqa: BLE001
            ref_out = ref.flash_attention_ref_np(q, k, v, key_bias, scale=SCALE)

        nki_out = _run_kernel_on_device(q, k, v, key_bias)

        a = torch.from_numpy(nki_out)
        b = torch.from_numpy(ref_out.astype(np.float32))
        close = torch.allclose(a, b, atol=ATOL, rtol=RTOL)
        max_abs = float((a - b).abs().max())
        # Per-plane cosine similarity (secondary metric per CLAUDE.md).
        af = a.reshape(a.shape[0], -1)
        bf2 = b.reshape(b.shape[0], -1)
        cos = torch.nn.functional.cosine_similarity(af, bf2, dim=1).min().item()
        passed = close and cos >= 0.999
        ok = ok and passed
        print(f"  {name:38s} max|d|={max_abs:.3e} mincos={cos:.5f} "
              f"{'PASS' if passed else 'FAIL'}")
    return ok


# ---------------------------------------------------------------------------
# pytest entry points
# ---------------------------------------------------------------------------
def test_numpy_algorithm_equivalence():
    assert check_numpy_algorithm_equivalence()


def test_nki_kernel_on_device():
    if not (_HAS_TORCH and _HAS_NKI and _HAS_DEVICE):
        import pytest

        pytest.skip(
            "Neuron device stack unavailable "
            f"(torch={_HAS_TORCH}, nki={_HAS_NKI}, device={_HAS_DEVICE}; "
            f"nki_import_err={_NKI_IMPORT_ERR}; device_err={_DEVICE_ERR})"
        )
    assert check_nki_kernel_on_device()


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 70)
    print("Mochi-1 flash attention — validation")
    print("=" * 70)
    print(f"torch={_HAS_TORCH}  nki={_HAS_NKI}  neuron_device={_HAS_DEVICE}")
    if _NKI_IMPORT_ERR is not None:
        print(f"(nki import note: {_NKI_IMPORT_ERR})")
    if _DEVICE_ERR is not None:
        print(f"(device note: {_DEVICE_ERR})")

    numpy_ok = check_numpy_algorithm_equivalence()

    if _HAS_TORCH and _HAS_NKI and _HAS_DEVICE:
        device_ok = check_nki_kernel_on_device()
        overall = numpy_ok and device_ok
        print("\n" + "=" * 70)
        print(f"RESULT: {'ALL PASS' if overall else 'FAILURES PRESENT'}")
        sys.exit(0 if overall else 1)
    else:
        print("\n" + "=" * 70)
        print("Numpy algorithm check:", "PASS" if numpy_ok else "FAIL")
        print("Device NKI cases: SKIPPED (Neuron device not reachable here).")
        print("Run this file inside a trn2 DLC to exercise the kernel.")
        sys.exit(0 if numpy_ok else 1)

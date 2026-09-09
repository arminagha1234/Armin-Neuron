"""Direct nki.simulate parity check for decode_hd256_v2.

Runs on CPU, so it can be developed while the device is busy. Exercises the
actual kernel body (not the wrapper's PyTorch fallback), which is the only
mode that catches layout/indexing mistakes before a device compile.

fp32 first (isolates kernel logic from bf16 rounding), then bf16 (numerics).
"""
from __future__ import annotations

import sys
import numpy as np
import torch

sys.path.insert(0, "/work/v2")

from ref_decode_hd256 import decode_hd256_ref, make_test_inputs, make_mask_bias  # noqa: E402

import nki  # noqa: E402
from decode_hd256_v2 import decode_hd256_v2_kernel  # noqa: E402


_SIM = nki.simulate(decode_hd256_v2_kernel)


def _simulate(*args):
    """nki.simulate is a wrapper: nki.simulate(kernel)(*args)."""
    return _SIM(*args)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


SHAPES = [
    # name, B, Nh, S_ctx, valid_len   (S_q is always 1 for decode)
    ("smoke",      1, 2, 128, 64),
    ("4B_short",   1, 2, 128, 100),
    ("4B_typical", 1, 2, 512, 400),
    ("4B_2K",      1, 2, 2048, 1500),
]


def run(dtype, dtype_name):
    print(f"\n===== dtype={dtype_name} =====")
    all_ok = True
    for name, B, Nh, S_ctx, valid_len in SHAPES:
        inp = make_test_inputs(B=B, Nh=Nh, S_q=1, S_ctx=S_ctx,
                               valid_len=valid_len, dtype=dtype)
        ref = decode_hd256_ref(**inp)                    # [B,Nh,1,256]
        mb = make_mask_bias(inp["mask"])                 # [B,1,1,S_ctx] fp32
        scale = float(inp["scale"])

        worst = 1.0
        worst_abs = 0.0
        for b in range(B):
            for h in range(Nh):
                q_bh = inp["q"][b, h].float().numpy()            # (1,256)
                k_bh = inp["k_full"][b, h].float().numpy()       # (S_ctx,256)
                v_bh = inp["v_full"][b, h].float().numpy()
                mb_bh = mb[b, 0].float().numpy()                 # (1,S_ctx)
                try:
                    out = _simulate(q_bh, k_bh, v_bh, mb_bh, scale)
                except Exception as e:
                    print(f"  {name:11s} b{b}h{h}: SIMULATE FAILED "
                          f"{type(e).__name__}: {str(e)[:160]}")
                    return False
                r = ref[b, h].float().numpy()
                c = _cosine(np.asarray(out), r)
                a = float(np.abs(np.asarray(out, dtype=np.float64) - r).max())
                worst = min(worst, c)
                worst_abs = max(worst_abs, a)
        ok = worst > 0.999
        all_ok = all_ok and ok
        print(f"  {name:11s} S_ctx={S_ctx:5d} valid={valid_len:5d}  "
              f"cos={worst:.6f}  max_abs={worst_abs:.5f}  "
              f"{'PASS' if ok else 'FAIL'}")
    return all_ok


if __name__ == "__main__":
    ok32 = run(torch.float32, "float32")
    ok16 = run(torch.bfloat16, "bfloat16")
    print(f"\nfp32 {'PASS' if ok32 else 'FAIL'}   bf16 {'PASS' if ok16 else 'FAIL'}")
    sys.exit(0 if (ok32 and ok16) else 1)

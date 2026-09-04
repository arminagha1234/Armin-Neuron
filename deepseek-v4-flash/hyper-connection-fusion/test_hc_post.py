"""Validate `hc_post` (increment 6) against the model's own expression, in float64.

The reference here is written as a literal transcription of the model's code -- reshape,
transpose, `bmm`, broadcast-multiply, add -- NOT as a restatement of the indexing the kernel
uses. That matters: after a sinkhorn `comb` is close to symmetric (row and column sums are
both 1), so transposing it or not produces two similar-looking answers. A reference derived
from the same index expression as the kernel would agree with a mistake.

An explicit asymmetry check is included as well, so a symmetric random draw cannot let an
index slip pass unnoticed.
"""

import os
import time

import torch

os.environ.setdefault("NEURON_PLATFORM_TARGET_OVERRIDE", "trn2")

import nki  # noqa: E402
import torch_neuronx  # noqa: E402,F401

from nki_hc_post import hc_post_kernel  # noqa: E402

DEV = torch.device("neuron")
EPS_FP32 = 1.1920929e-7
BF16_RES = 2.0 ** -8
ITERS = 30
WARMUP = 3


def ref_hc_post(x, residual, post, comb, M, D):
    """Literal fp64 transcription of the model's hc_post."""
    P = x.shape[0]
    comb_3d = comb.double().reshape(P, M, M)
    comb_t = comb_3d.transpose(-1, -2)
    res_3d = residual.double().reshape(P, M, D)
    comb_residual = torch.bmm(comb_t, res_3d)              # [P, M, D]
    post_3d = post.double().reshape(P, M, 1)
    x_3d = x.double().reshape(P, 1, D)
    y = post_3d * x_3d + comb_residual
    return y.reshape(P, M * D)


def timed(fn, label):
    with torch.inference_mode():
        for _ in range(WARMUP):
            fn().to("cpu")
        t0 = time.perf_counter()
        for _ in range(ITERS):
            fn().to("cpu")
        dt = (time.perf_counter() - t0) / ITERS * 1e6
    print(f"  {label:34} {dt:9.1f} us")
    return dt


def main():
    M, D = 4, 4096
    all_ok = True

    for P in (1, 8):
        print(f"\n=== P={P}, M={M}, D={D} ===")
        torch.manual_seed(31337 + P)
        x = torch.randn(P, D, dtype=torch.float32)
        residual = torch.randn(P, M * D, dtype=torch.float32)
        post = (torch.rand(P, M, dtype=torch.float32) * 1.5 + 0.2)
        # a deliberately ASYMMETRIC comb, so a transpose slip cannot hide
        comb = torch.rand(P, M, M, dtype=torch.float32) + 0.05
        comb = comb / comb.sum(-1, keepdim=True)
        asym = (comb - comb.transpose(-1, -2)).abs().max().item()
        comb_flat = comb.reshape(P, M * M).contiguous()
        print(f"  comb asymmetry (max |c - c^T|) = {asym:.4f}  "
              f"{'(good, transpose is detectable)' if asym > 0.05 else '(TOO SYMMETRIC)'}")

        want = ref_hc_post(x, residual, post, comb_flat, M, D)
        # the inner sum runs over M terms in fp32 on device
        bound = M * EPS_FP32 * 10

        got = hc_post_kernel(x.to(DEV), residual.to(DEV),
                             post.to(DEV), comb_flat.to(DEV)).to("cpu")

        denom = want.abs().max().item()
        max_abs = (got.double() - want).abs().max().item()
        rel = max_abs / denom
        ok = rel <= bound
        all_ok &= ok
        print(f"  {'PASS' if ok else 'FAIL'}  {'hc_post vs fp64':34} "
              f"max_abs={max_abs:.3e} rel={rel:.3e} bound={bound:.3e}")
        print(f"  INFO  error is {BF16_RES / max(rel, 1e-30):.0f}x below bf16 resolution")

        # control: the WRONG (untransposed) reference must NOT match, otherwise this test
        # has no power to detect the index slip it is designed to catch
        comb_3d = comb_flat.double().reshape(P, M, M)
        wrong = (post.double().reshape(P, M, 1) * x.double().reshape(P, 1, D)
                 + torch.bmm(comb_3d, residual.double().reshape(P, M, D))).reshape(P, M * D)
        wrong_rel = ((got.double() - wrong).abs().max() / denom).item()
        discriminating = wrong_rel > 100 * max(rel, 1e-30)
        print(f"  {'PASS' if discriminating else 'FAIL'}  "
              f"{'test discriminates transpose':34} "
              f"wrong-ref rel={wrong_rel:.3e} vs correct {rel:.3e}")
        all_ok &= discriminating

        print("  --- latency (async-safe) ---")
        xd, rd, pd, cd = (x.to(DEV), residual.to(DEV), post.to(DEV), comb_flat.to(DEV))
        timed(lambda: hc_post_kernel(xd, rd, pd, cd), "hc_post kernel")

        def eager():
            c3 = cd.reshape(P, M, M).transpose(-1, -2)
            r3 = rd.reshape(P, M, D)
            return (pd.reshape(P, M, 1) * xd.reshape(P, 1, D)
                    + torch.bmm(c3, r3)).reshape(P, M * D)
        timed(eager, "eager torch (context only)")

    print("\n" + ("ALL CHECKS PASS" if all_ok else "SOME CHECKS FAILED"))
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Validate + benchmark the fused `hc_pre`-combine + `attn_norm` Phase-1 increment.

Three things are checked, in order of what actually settles the question:

  1. NUMERICS vs a torch reference, gated on the SEQUENTIAL-fp32 error bound rather than an
     arbitrary tolerance. This distinction was learned the hard way: torch's `.mean()` is
     PAIRWISE (error ~ log2(N)*eps) while the Scalar Engine reduces SEQUENTIALLY
     (error ~ N*eps), so the kernel is *expected* to differ from torch by roughly
     N/log2(N) ~ 340x at D=4096. That ratio is not a bug, and an fp64 check that flags it as
     "genuinely wrong" is itself wrong. The meaningful gate is: within the sequential bound,
     AND far below bf16 resolution, since these activations are bf16 in the real model.

  2. FUSED == UNFUSED. The fused kernel and the two-kernel path execute the same instruction
     sequence, so they should agree to the bit. This is the test that actually proves the
     fusion is behaviour-preserving.

  3. LATENCY, async-safely. Neuron execution is asynchronous: ops return before the device
     finishes, so a naive loop measures enqueue cost and yields absurd speedups. Every timed
     region below pulls its result to host with `.to("cpu")` and runs under
     `torch.inference_mode()`.

The headline comparison is FUSED vs TWO-KERNEL. The eager-torch column is context only --
in the real model these ops live inside a compiled graph that already fuses some of them,
so an eager speedup must not be reported as an end-to-end gain.
"""

import os
import time

import torch

os.environ.setdefault("NEURON_PLATFORM_TARGET_OVERRIDE", "trn2")

import nki  # noqa: E402
import torch_neuronx  # noqa: E402,F401  (registers the `neuron` device)

from nki_hc_combine_norm import (  # noqa: E402
    hc_combine_kernel,
    hc_combine_norm_kernel,
    hc_rmsnorm_only_kernel,
)

DEV = torch.device("neuron")
EPS_FP32 = 1.1920929e-7
BF16_RES = 2.0 ** -8          # 3.906e-03, the gap between consecutive bf16 mantissas
ITERS = 50
WARMUP = 5


def torch_ref(x_flat, pre, weight, eps, M, D):
    """Reference in fp64 so it is not itself the thing under test."""
    x = x_flat.double().view(-1, M, D)
    p = pre.double().unsqueeze(-1)
    y = (x * p).sum(dim=1)
    rs = torch.rsqrt(y.square().mean(-1, keepdim=True) + eps)
    return y * rs * weight.double()


def seq_bound(x_flat, pre, M, D, eps):
    """Relative error bound for a SEQUENTIAL fp32 reduction of D terms.

    Sequential summation of N values has worst-case relative error ~ N*eps scaled by the
    ratio of the sum of magnitudes to the magnitude of the sum. Both the combine (M terms)
    and the sum-of-squares (D terms) contribute.
    """
    x = x_flat.double().view(-1, M, D)
    p = pre.double().unsqueeze(-1)
    terms = x * p
    y = terms.sum(dim=1)
    # growth factor for the combine
    g_comb = terms.abs().sum(dim=1).sum() / y.abs().sum().clamp_min(1e-30)
    # sum of squares is all-positive, so its growth factor is 1
    rel = (M * EPS_FP32 * g_comb + D * EPS_FP32).item()
    return rel


def check(name, got, want, bound):
    got = got.double()
    denom = want.abs().max().clamp_min(1e-30)
    max_abs = (got - want).abs().max().item()
    rel = max_abs / denom.item()
    ok = rel <= bound
    print(f"  {'PASS' if ok else 'FAIL'}  {name:38} max_abs={max_abs:.3e} "
          f"rel={rel:.3e} bound={bound:.3e} bf16_res={BF16_RES:.3e}")
    return ok


def timed(fn, label):
    """Async-safe latency. The result is pulled to host INSIDE the timed region."""
    with torch.inference_mode():
        for _ in range(WARMUP):
            fn().to("cpu")
        t0 = time.perf_counter()
        for _ in range(ITERS):
            fn().to("cpu")
        dt = (time.perf_counter() - t0) / ITERS * 1e6
    print(f"  {label:38} {dt:9.1f} us")
    return dt


def main():
    D, M, EPS = 4096, 4, 1e-6
    all_ok = True
    results = {}

    for P in (1, 8):
        print(f"\n=== P={P} (decode batch), M={M}, D={D} ===")
        torch.manual_seed(1234 + P)
        x_flat = torch.randn(P, M * D, dtype=torch.float32)
        # `pre` comes from a sinkhorn, so it is non-negative and roughly row-normalised
        pre = torch.rand(P, M, dtype=torch.float32) + 0.1
        pre = pre / pre.sum(-1, keepdim=True)
        weight = torch.randn(1, D, dtype=torch.float32) * 0.1 + 1.0

        want = torch_ref(x_flat, pre, weight, EPS, M, D)
        bound = seq_bound(x_flat, pre, M, D, EPS)

        xd = x_flat.to(DEV)
        pd = pre.to(DEV)
        wd = weight.to(DEV)

        # ---- fused ----
        fused = hc_combine_norm_kernel(xd, pd, wd, EPS).to("cpu")
        all_ok &= check("fused vs fp64 ref", fused, want, bound)

        # ---- unfused two-kernel path ----
        y_mid = hc_combine_kernel(xd, pd)
        two = hc_rmsnorm_only_kernel(y_mid, wd, EPS).to("cpu")
        all_ok &= check("two-kernel vs fp64 ref", two, want, bound)

        # ---- the fusion-correctness test ----
        d = (fused - two).abs().max().item()
        bit_identical = d == 0.0
        print(f"  {'PASS' if d <= 1e-6 else 'FAIL'}  "
              f"{'fused vs two-kernel':38} max_abs={d:.3e} "
              f"{'(BIT-IDENTICAL)' if bit_identical else ''}")
        all_ok &= (d <= 1e-6)

        # ---- sanity: is our error actually below bf16 resolution? ----
        rel = ((fused.double() - want).abs().max() / want.abs().max()).item()
        margin = BF16_RES / max(rel, 1e-30)
        print(f"  INFO  error is {margin:.0f}x smaller than bf16 resolution")

        # ---- latency ----
        print("  --- latency (async-safe) ---")
        t_fused = timed(lambda: hc_combine_norm_kernel(xd, pd, wd, EPS), "fused (1 kernel)")

        def two_kernel():
            return hc_rmsnorm_only_kernel(hc_combine_kernel(xd, pd), wd, EPS)
        t_two = timed(two_kernel, "two separate kernels")

        def eager():
            y = (xd.view(P, M, D) * pd.unsqueeze(-1)).sum(dim=1)
            r = torch.rsqrt(y.square().mean(-1, keepdim=True) + EPS)
            return y * r * wd
        t_eager = timed(eager, "eager torch (context only)")

        results[P] = (t_fused, t_two, t_eager)
        print(f"  fused vs two-kernel : {t_two / t_fused:.2f}x")
        print(f"  fused vs eager      : {t_eager / t_fused:.2f}x  "
              f"(NOT an end-to-end claim)")

    # dispatch-bound guard: if latency is flat across an 8x problem-size change, the
    # measurement is dominated by launch overhead and work inside the kernel is not the
    # bottleneck. Same guard that settled the router kernel.
    print("\n=== scaling guard ===")
    f1, f8 = results[1][0], results[8][0]
    change = (f8 - f1) / f1 * 100
    print(f"  fused latency P=1 -> P=8: {f1:.1f} -> {f8:.1f} us ({change:+.1f}%)")
    if abs(change) < 5:
        print("  => FLAT => DISPATCH-DOMINATED: the win here is boundary removal, not compute")
    else:
        print("  => scales with work => compute-sensitive")

    print("\n" + ("ALL CHECKS PASS" if all_ok else "SOME CHECKS FAILED"))
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

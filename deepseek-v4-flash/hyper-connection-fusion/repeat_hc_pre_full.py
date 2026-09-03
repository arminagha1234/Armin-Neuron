"""Repeatability harness for the increment-5 fusion claim.

WHY THIS EXISTS
---------------
Two back-to-back runs of `test_hc_pre_full.py` disagreed about whether the fused kernel is
launch-bound:

    run A:  P=1 443.6 us   P=8 473.2 us   -> +6.7%  "scales with work"
    run B:  P=1 504.6 us   P=8 487.9 us   -> -3.3%  "flat"

The P=1 figure moved 13.7% between runs, which is larger than the P=1 -> P=8 difference either
run was trying to detect. So a single run cannot settle the scaling question, and the "this
kernel is now compute-sensitive" conclusion drawn from run A was not supported.

Two fixes, both about measurement rather than the kernel:

  1. **Interleave** the fused and unfused variants within each repetition, so any drift
     (thermal, host contention, other tenants) hits both arms of the comparison equally. The
     ratio is a paired statistic; comparing across separately-timed blocks is not.
  2. **Repeat** and report mean +- standard deviation, then judge the scaling question against
     the observed spread instead of against zero.
"""

import os
import statistics
import time

import torch

os.environ.setdefault("NEURON_PLATFORM_TARGET_OVERRIDE", "trn2")

import nki  # noqa: E402
import torch_neuronx  # noqa: E402,F401

from nki_hc_matmul_head import hc_matmul_head_kernel  # noqa: E402
from nki_hc_pre_full import hc_pre_full_kernel  # noqa: E402
from nki_hc_sinkhorn_fused import hc_sinkhorn_combine_norm_kernel  # noqa: E402

DEV = torch.device("neuron")
REPS = 7          # outer repetitions, each interleaving both variants
INNER = 20        # timed iterations per variant per repetition
WARMUP = 3


def measure(fn, n):
    """Async-safe: the result is pulled to host inside the timed region."""
    with torch.inference_mode():
        t0 = time.perf_counter()
        for _ in range(n):
            r = fn()
            (r[0] if isinstance(r, tuple) else r).to("cpu")
        return (time.perf_counter() - t0) / n * 1e6


def main():
    HC, D, EPS, NEPS, SK = 4, 4096, 1e-6, 1e-6, 20
    K = HC * D
    MIX = (2 + HC) * HC
    out = {}

    for P in (1, 8):
        torch.manual_seed(555 + P)
        x = torch.randn(P, K, dtype=torch.float32).to(DEV)
        wt = (torch.randn(MIX, K, dtype=torch.float32) / (K ** 0.5)).t().contiguous().to(DEV)
        s = (torch.rand(3, dtype=torch.float32) + 0.5).to(DEV)
        b = (torch.randn(MIX, dtype=torch.float32) * 0.1).to(DEV)
        wn = (torch.randn(1, D, dtype=torch.float32) * 0.1 + 1.0).to(DEV)

        def fused():
            return hc_pre_full_kernel(x, wt, s, b, wn, HC, SK, EPS, NEPS, False)

        def two():
            m = hc_matmul_head_kernel(x, wt, EPS, False)
            return hc_sinkhorn_combine_norm_kernel(m, s, b, x, wn, HC, SK, EPS, NEPS)

        with torch.inference_mode():
            for _ in range(WARMUP):
                fused()[0].to("cpu")
                two()[0].to("cpu")

        f_all, t_all, r_all = [], [], []
        for _ in range(REPS):
            # interleaved within the repetition: drift hits both arms
            tf = measure(fused, INNER)
            tt = measure(two, INNER)
            f_all.append(tf)
            t_all.append(tt)
            r_all.append(tt / tf)

        out[P] = (f_all, t_all, r_all)
        fm, fs = statistics.mean(f_all), statistics.stdev(f_all)
        tm, ts = statistics.mean(t_all), statistics.stdev(t_all)
        rm, rs = statistics.mean(r_all), statistics.stdev(r_all)
        print(f"\n=== P={P}  ({REPS} reps x {INNER} iters, interleaved) ===")
        print(f"  fused      {fm:8.1f} +- {fs:5.1f} us   "
              f"(min {min(f_all):.1f}, max {max(f_all):.1f}, spread {(max(f_all)-min(f_all))/fm*100:.1f}%)")
        print(f"  two-kernel {tm:8.1f} +- {ts:5.1f} us   "
              f"(min {min(t_all):.1f}, max {max(t_all):.1f})")
        print(f"  ratio      {rm:8.3f} +- {rs:5.3f}x     "
              f"(min {min(r_all):.3f}, max {max(r_all):.3f})")
        # a paired ratio is only meaningful if it clears its own noise
        print(f"  ratio > 1 in {sum(1 for r in r_all if r > 1.0)}/{REPS} repetitions")

    print("\n=== scaling: is the fused kernel launch-bound? ===")
    f1, f8 = out[1][0], out[8][0]
    m1, s1 = statistics.mean(f1), statistics.stdev(f1)
    m8, s8 = statistics.mean(f8), statistics.stdev(f8)
    delta = (m8 - m1) / m1 * 100
    # pooled noise on the difference of two means
    noise = ((s1 ** 2 + s8 ** 2) ** 0.5) / m1 * 100
    print(f"  P=1 {m1:.1f} +- {s1:.1f} us")
    print(f"  P=8 {m8:.1f} +- {s8:.1f} us")
    print(f"  difference {delta:+.1f}%,  pooled noise +-{noise:.1f}%")
    if abs(delta) <= noise:
        print("  VERDICT: difference is WITHIN noise -> scaling is NOT resolvable at this "
              "precision. No claim either way.")
    elif delta > 0:
        print("  VERDICT: grows with work beyond noise -> compute-sensitive.")
    else:
        print("  VERDICT: shrinks with work beyond noise -> investigate, unexpected.")


if __name__ == "__main__":
    raise SystemExit(main())

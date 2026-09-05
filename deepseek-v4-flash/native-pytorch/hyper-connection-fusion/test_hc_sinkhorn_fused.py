"""Validate + benchmark increment 3: sinkhorn fused with combine + RMSNorm.

Checks, in order:

  1. NUMERICS vs a float64 CPU reference that reproduces the model's own
     `_hc_split_sinkhorn_cpu` -> combine -> RMSNorm chain exactly, including its transpose
     bookkeeping. The gate is the sequential-fp32 error bound (the Scalar/Vector engines
     reduce sequentially; torch reduces pairwise, so a ~N/log2(N) discrepancy is EXPECTED
     and is not evidence of a bug), cross-checked against bf16 resolution since these
     activations are bf16 in the real model.

     The sinkhorn is 20 iterations of divide-by-sum, so error compounds across iterations.
     The bound below accounts for that explicitly rather than hand-waving a tolerance.

  2. FUSED == THREE-KERNEL. The fused kernel must agree with running the sinkhorn kernel,
     then the combine kernel, then the norm kernel. This is what proves the fusion is
     behaviour-preserving.

  3. LATENCY, async-safely (result pulled to host inside the timed region, under
     `torch.inference_mode()`), fused vs the three-kernel path.
"""

import os
import time

import torch

os.environ.setdefault("NEURON_PLATFORM_TARGET_OVERRIDE", "trn2")

import nki  # noqa: E402
import torch_neuronx  # noqa: E402,F401

from nki_hc_combine_norm import hc_combine_kernel, hc_rmsnorm_only_kernel  # noqa: E402
from nki_hc_sinkhorn_fused import (  # noqa: E402
    hc_sinkhorn_combine_norm_kernel,
    hc_sinkhorn_kernel,
)

DEV = torch.device("neuron")
EPS_FP32 = 1.1920929e-7
BF16_RES = 2.0 ** -8
ITERS = 30
WARMUP = 3


def sinkhorn_cpu(mixes, hc_scale, hc_base, hc, iters, eps):
    """float64 restatement of the model's `_hc_split_sinkhorn_cpu`, transposes and all."""
    mixes = mixes.double()
    hc_scale = hc_scale.double()
    hc_base = hc_base.double()
    pre_l = mixes[..., :hc]
    post_l = mixes[..., hc:2 * hc]
    comb_l = mixes[..., 2 * hc:]
    pre_b = hc_base[:hc]
    post_b = hc_base[hc:2 * hc]
    comb_b = hc_base[2 * hc:]

    pre = torch.sigmoid(pre_l * hc_scale[0] + pre_b) + eps
    post = 2 * torch.sigmoid(post_l * hc_scale[1] + post_b)
    comb = comb_l * hc_scale[2] + comb_b

    BS = comb.shape[0]
    M = hc
    comb = comb.reshape(BS * M, M)
    comb = torch.softmax(comb, dim=-1) + eps
    comb = comb.view(BS, M, M).transpose(-1, -2).contiguous().view(BS * M, M)
    col_sum = comb.sum(dim=-1, keepdim=True)
    comb = comb / (col_sum + eps)
    comb = comb.view(BS, M, M).transpose(-1, -2).contiguous().view(BS * M, M)
    for _ in range(iters - 1):
        row_sum = comb.sum(dim=-1, keepdim=True)
        comb = comb / (row_sum + eps)
        comb = comb.view(BS, M, M).transpose(-1, -2).contiguous().view(BS * M, M)
        col_sum = comb.sum(dim=-1, keepdim=True)
        comb = comb / (col_sum + eps)
        comb = comb.view(BS, M, M).transpose(-1, -2).contiguous().view(BS * M, M)
    comb = comb.view(BS, M, M)
    return pre, post, comb


def full_ref(mixes, hc_scale, hc_base, x_flat, weight, hc, iters, eps, norm_eps):
    pre, post, comb = sinkhorn_cpu(mixes, hc_scale, hc_base, hc, iters, eps)
    P = x_flat.shape[0]
    D = x_flat.shape[1] // hc
    x = x_flat.double().view(P, hc, D)
    y = (x * pre.unsqueeze(-1)).sum(dim=1)
    rs = torch.rsqrt(y.square().mean(-1, keepdim=True) + norm_eps)
    out = y * rs * weight.double()
    return out, post, comb.reshape(P, hc * hc)


def check(name, got, want, bound, note=""):
    got = got.double()
    denom = want.abs().max().clamp_min(1e-30).item()
    max_abs = (got - want).abs().max().item()
    rel = max_abs / denom
    ok = rel <= bound
    print(f"  {'PASS' if ok else 'FAIL'}  {name:34} max_abs={max_abs:.3e} "
          f"rel={rel:.3e} bound={bound:.3e} {note}")
    return ok


def timed(fn, label):
    with torch.inference_mode():
        for _ in range(WARMUP):
            r = fn()
            (r[0] if isinstance(r, tuple) else r).to("cpu")
        t0 = time.perf_counter()
        for _ in range(ITERS):
            r = fn()
            (r[0] if isinstance(r, tuple) else r).to("cpu")
        dt = (time.perf_counter() - t0) / ITERS * 1e6
    print(f"  {label:34} {dt:9.1f} us")
    return dt


def main():
    D, HC, ITERS_SK, EPS, NEPS = 4096, 4, 20, 1e-6, 1e-6
    MIX = (2 + HC) * HC
    all_ok = True
    lat = {}

    for P in (1, 8):
        print(f"\n=== P={P}, hc={HC}, D={D}, mix_hc={MIX}, sinkhorn_iters={ITERS_SK} ===")
        torch.manual_seed(4242 + P)
        mixes = torch.randn(P, MIX, dtype=torch.float32)
        hc_scale = torch.rand(3, dtype=torch.float32) + 0.5
        hc_base = torch.randn(MIX, dtype=torch.float32) * 0.1
        x_flat = torch.randn(P, HC * D, dtype=torch.float32)
        weight = torch.randn(1, D, dtype=torch.float32) * 0.1 + 1.0

        w_out, w_post, w_comb = full_ref(mixes, hc_scale, hc_base, x_flat, weight,
                                         HC, ITERS_SK, EPS, NEPS)

        # Error budget. The sinkhorn does 2*iters divide-by-sum steps over hc terms, so its
        # relative error accumulates roughly linearly in the number of steps; the combine
        # adds hc terms and the norm reduces D terms, both sequentially on device.
        sk_bound = 2 * ITERS_SK * HC * EPS_FP32
        bound_out = sk_bound + HC * EPS_FP32 + D * EPS_FP32
        bound_sk = sk_bound

        md, sd, bd = mixes.to(DEV), hc_scale.to(DEV), hc_base.to(DEV)
        xd, wd = x_flat.to(DEV), weight.to(DEV)

        # ---- fused ----
        f_out, f_post, f_comb = hc_sinkhorn_combine_norm_kernel(
            md, sd, bd, xd, wd, HC, ITERS_SK, EPS, NEPS)
        f_out, f_post, f_comb = f_out.to("cpu"), f_post.to("cpu"), f_comb.to("cpu")

        all_ok &= check("fused out vs fp64", f_out, w_out, bound_out,
                        f"bf16_res={BF16_RES:.2e}")
        all_ok &= check("fused post vs fp64", f_post, w_post, bound_sk)
        all_ok &= check("fused comb vs fp64", f_comb, w_comb, bound_sk)

        # ---- three-kernel path ----
        k_pre, k_post, k_comb = hc_sinkhorn_kernel(md, sd, bd, HC, ITERS_SK, EPS)
        k_y = hc_combine_kernel(xd, k_pre)
        k_out = hc_rmsnorm_only_kernel(k_y, wd, NEPS).to("cpu")

        all_ok &= check("3-kernel out vs fp64", k_out, w_out, bound_out)

        d_out = (f_out - k_out).abs().max().item()
        d_post = (f_post - k_post.to("cpu")).abs().max().item()
        d_comb = (f_comb - k_comb.to("cpu")).abs().max().item()
        bit = (d_out == 0.0 and d_post == 0.0 and d_comb == 0.0)
        ok_fuse = max(d_out, d_post, d_comb) <= 1e-6
        print(f"  {'PASS' if ok_fuse else 'FAIL'}  {'fused vs 3-kernel':34} "
              f"out={d_out:.3e} post={d_post:.3e} comb={d_comb:.3e} "
              f"{'(BIT-IDENTICAL)' if bit else ''}")
        all_ok &= ok_fuse

        rel = ((f_out.double() - w_out).abs().max() / w_out.abs().max()).item()
        print(f"  INFO  out error is {BF16_RES / max(rel, 1e-30):.0f}x "
              f"smaller than bf16 resolution")

        # sanity: sinkhorn output should be near doubly-stochastic
        cm = f_comb.view(P, HC, HC).double()
        print(f"  INFO  comb row sums in [{cm.sum(-1).min():.4f}, {cm.sum(-1).max():.4f}] "
              f"col sums in [{cm.sum(-2).min():.4f}, {cm.sum(-2).max():.4f}]")

        print("  --- latency (async-safe) ---")
        t_f = timed(lambda: hc_sinkhorn_combine_norm_kernel(
            md, sd, bd, xd, wd, HC, ITERS_SK, EPS, NEPS), "fused (1 kernel)")

        def three():
            p, _, _ = hc_sinkhorn_kernel(md, sd, bd, HC, ITERS_SK, EPS)
            return hc_rmsnorm_only_kernel(hc_combine_kernel(xd, p), wd, NEPS)
        t_3 = timed(three, "three separate kernels")

        lat[P] = (t_f, t_3)
        print(f"  fused vs 3-kernel : {t_3 / t_f:.2f}x")

    print("\n=== scaling guard ===")
    f1, f8 = lat[1][0], lat[8][0]
    ch = (f8 - f1) / f1 * 100
    print(f"  fused latency P=1 -> P=8: {f1:.1f} -> {f8:.1f} us ({ch:+.1f}%)")
    print("  => FLAT => launch-bound" if abs(ch) < 5 else "  => scales with work")

    print("\n" + ("ALL CHECKS PASS" if all_ok else "SOME CHECKS FAILED"))
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

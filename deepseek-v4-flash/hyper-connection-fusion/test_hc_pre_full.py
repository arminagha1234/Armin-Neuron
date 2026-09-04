"""Validate + benchmark increment 5: the whole `hc_pre` + RMSNorm in one kernel.

Compares against:
  * an fp64 reference for the entire boundary;
  * the two-kernel path (increment 4 then increment 3), which is the thing this fusion
    replaces -- that is the comparison that tests the fusion claim;
  * eager torch, for context only. In the real model these ops sit inside a compiled graph
    that already fuses some of them, so the eager ratio must NOT be read as an end-to-end gain.

Timing is async-safe: results are pulled to host inside the timed region, under
`torch.inference_mode()`.
"""

import os
import time

import torch

os.environ.setdefault("NEURON_PLATFORM_TARGET_OVERRIDE", "trn2")

import nki  # noqa: E402
import torch_neuronx  # noqa: E402,F401

from nki_hc_matmul_head import hc_matmul_head_kernel  # noqa: E402
from nki_hc_pre_full import hc_pre_full_kernel  # noqa: E402
from nki_hc_sinkhorn_fused import hc_sinkhorn_combine_norm_kernel  # noqa: E402

DEV = torch.device("neuron")
EPS_FP32 = 1.1920929e-7
BF16_RES = 2.0 ** -8
ITERS = 30
WARMUP = 3


def sinkhorn_cpu(mixes, s, b, hc, iters, eps):
    mixes, s, b = mixes.double(), s.double(), b.double()
    pre = torch.sigmoid(mixes[..., :hc] * s[0] + b[:hc]) + eps
    post = 2 * torch.sigmoid(mixes[..., hc:2 * hc] * s[1] + b[hc:2 * hc])
    comb = mixes[..., 2 * hc:] * s[2] + b[2 * hc:]
    P = comb.shape[0]
    comb = comb.reshape(P * hc, hc)
    comb = torch.softmax(comb, dim=-1) + eps
    comb = comb.view(P, hc, hc).transpose(-1, -2).contiguous().view(P * hc, hc)
    comb = comb / (comb.sum(-1, keepdim=True) + eps)
    comb = comb.view(P, hc, hc).transpose(-1, -2).contiguous().view(P * hc, hc)
    for _ in range(iters - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + eps)
        comb = comb.view(P, hc, hc).transpose(-1, -2).contiguous().view(P * hc, hc)
        comb = comb / (comb.sum(-1, keepdim=True) + eps)
        comb = comb.view(P, hc, hc).transpose(-1, -2).contiguous().view(P * hc, hc)
    return pre, post, comb.view(P, hc, hc).reshape(P, hc * hc)


def full_ref(x_flat, hc_fn, s, b, weight, hc, iters, eps, neps):
    x = x_flat.double()
    P = x.shape[0]
    D = x.shape[1] // hc
    rs = torch.rsqrt(x.square().mean(-1, keepdim=True) + eps)
    mixes = (x @ hc_fn.double().T) * rs
    pre, post, comb = sinkhorn_cpu(mixes.float(), s, b, hc, iters, eps)
    y = (x.view(P, hc, D) * pre.unsqueeze(-1)).sum(dim=1)
    out = y * torch.rsqrt(y.square().mean(-1, keepdim=True) + neps) * weight.double()
    return out, post, comb


def check(name, got, want, bound):
    got = got.double()
    denom = want.abs().max().clamp_min(1e-30).item()
    max_abs = (got - want).abs().max().item()
    rel = max_abs / denom
    ok = rel <= bound
    print(f"  {'PASS' if ok else 'FAIL'}  {name:34} max_abs={max_abs:.3e} "
          f"rel={rel:.3e} bound={bound:.3e}")
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
    HC, D, EPS, NEPS, SK = 4, 4096, 1e-6, 1e-6, 20
    K = HC * D
    MIX = (2 + HC) * HC
    all_ok = True
    lat = {}

    for P in (1, 8):
        print(f"\n=== P={P}, hc={HC}, D={D}, K={K}, mix_hc={MIX} ===")
        torch.manual_seed(555 + P)
        x_flat = torch.randn(P, K, dtype=torch.float32)
        hc_fn = torch.randn(MIX, K, dtype=torch.float32) / (K ** 0.5)
        hc_fn_t = hc_fn.t().contiguous()
        s = torch.rand(3, dtype=torch.float32) + 0.5
        b = torch.randn(MIX, dtype=torch.float32) * 0.1
        weight = torch.randn(1, D, dtype=torch.float32) * 0.1 + 1.0

        w_out, w_post, w_comb = full_ref(x_flat, hc_fn, s, b, weight,
                                         HC, SK, EPS, NEPS)
        bound = 2 * K * EPS_FP32 + 2 * SK * HC * EPS_FP32 + HC * EPS_FP32 + D * EPS_FP32

        xd, wtd, sd, bd, wnd = (x_flat.to(DEV), hc_fn_t.to(DEV), s.to(DEV),
                                b.to(DEV), weight.to(DEV))

        # ---- fused (increment 5) ----
        f_out, f_post, f_comb = hc_pre_full_kernel(
            xd, wtd, sd, bd, wnd, HC, SK, EPS, NEPS, False)
        f_out, f_post, f_comb = f_out.to("cpu"), f_post.to("cpu"), f_comb.to("cpu")
        all_ok &= check("fused out vs fp64", f_out, w_out, bound)
        all_ok &= check("fused post vs fp64", f_post, w_post, bound)
        all_ok &= check("fused comb vs fp64", f_comb, w_comb, bound)

        # ---- two-kernel path (increment 4 -> increment 3) ----
        mid = hc_matmul_head_kernel(xd, wtd, EPS, False)
        t_out, t_post, t_comb = hc_sinkhorn_combine_norm_kernel(
            mid, sd, bd, xd, wnd, HC, SK, EPS, NEPS)
        t_out, t_post, t_comb = t_out.to("cpu"), t_post.to("cpu"), t_comb.to("cpu")
        all_ok &= check("2-kernel out vs fp64", t_out, w_out, bound)

        d = max((f_out - t_out).abs().max().item(),
                (f_post - t_post).abs().max().item(),
                (f_comb - t_comb).abs().max().item())
        ok = d <= 1e-6
        print(f"  {'PASS' if ok else 'FAIL'}  {'fused vs 2-kernel':34} "
              f"max_abs={d:.3e} {'(BIT-IDENTICAL)' if d == 0.0 else ''}")
        all_ok &= ok

        rel = ((f_out.double() - w_out).abs().max() / w_out.abs().max()).item()
        print(f"  INFO  out error is {BF16_RES / max(rel, 1e-30):.0f}x below bf16 resolution")
        cm = f_comb.view(P, HC, HC).double()
        print(f"  INFO  comb row sums [{cm.sum(-1).min():.4f}, {cm.sum(-1).max():.4f}] "
              f"col sums [{cm.sum(-2).min():.4f}, {cm.sum(-2).max():.4f}]")

        print("  --- latency (async-safe) ---")
        t_f = timed(lambda: hc_pre_full_kernel(
            xd, wtd, sd, bd, wnd, HC, SK, EPS, NEPS, False), "fused, 1 kernel")

        def two():
            m = hc_matmul_head_kernel(xd, wtd, EPS, False)
            return hc_sinkhorn_combine_norm_kernel(m, sd, bd, xd, wnd, HC, SK, EPS, NEPS)
        t_2 = timed(two, "two kernels (inc4 -> inc3)")

        def eager():
            xx = xd.double() if False else xd
            r = torch.rsqrt(xx.square().mean(-1, keepdim=True) + EPS)
            mx = (xx @ wtd) * r
            pre = torch.sigmoid(mx[..., :HC] * sd[0] + bd[:HC]) + EPS
            yy = (xx.view(P, HC, D) * pre.unsqueeze(-1)).sum(dim=1)
            return yy * torch.rsqrt(yy.square().mean(-1, keepdim=True) + NEPS) * wnd
        t_e = timed(eager, "eager torch (partial, context)")

        lat[P] = (t_f, t_2, t_e)
        print(f"  fused vs 2-kernel : {t_2 / t_f:.2f}x")
        print(f"  (eager column is a PARTIAL chain - no sinkhorn - context only)")

    print("\n=== scaling guard ===")
    f1, f8 = lat[1][0], lat[8][0]
    ch = (f8 - f1) / f1 * 100
    print(f"  fused latency P=1 -> P=8: {f1:.1f} -> {f8:.1f} us ({ch:+.1f}%)")
    print("  => FLAT => launch-bound" if abs(ch) < 5 else "  => scales with work")

    print("\n" + ("ALL CHECKS PASS" if all_ok else "SOME CHECKS FAILED"))
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

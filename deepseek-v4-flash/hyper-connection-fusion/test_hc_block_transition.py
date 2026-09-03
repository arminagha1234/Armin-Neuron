"""Validate + benchmark increment 7: hc_post + hc_pre + RMSNorm fused.

Compared against:
  * an fp64 reference for the whole transition, built from the model's own expressions;
  * the two-kernel path (increment 6 then increment 5) -- the comparison that tests the
    fusion claim, run as a PAIRED interleaved measurement with repetitions, per the lesson
    from `repeat_hc_pre_full.py`.

All four outputs are checked, including `hidden`, which is the residual carry the next
transition consumes.
"""

import os
import statistics
import time

import torch

os.environ.setdefault("NEURON_PLATFORM_TARGET_OVERRIDE", "trn2")

import nki  # noqa: E402
import torch_neuronx  # noqa: E402,F401

from nki_hc_block_transition import hc_block_transition_kernel  # noqa: E402
from nki_hc_post import hc_post_kernel  # noqa: E402
from nki_hc_pre_full import hc_pre_full_kernel  # noqa: E402

DEV = torch.device("neuron")
EPS_FP32 = 1.1920929e-7
BF16_RES = 2.0 ** -8
REPS = 5
INNER = 15
WARMUP = 3


def ref_transition(x, residual, post_in, comb_in, hc_fn, s, b, weight,
                   hc, D, iters, eps, neps):
    P = x.shape[0]
    # --- hc_post, literal transcription ---
    c3 = comb_in.double().reshape(P, hc, hc).transpose(-1, -2)
    r3 = residual.double().reshape(P, hc, D)
    hidden = (post_in.double().reshape(P, hc, 1) * x.double().reshape(P, 1, D)
              + torch.bmm(c3, r3)).reshape(P, hc * D)
    # --- hc_pre ---
    rs = torch.rsqrt(hidden.square().mean(-1, keepdim=True) + eps)
    mixes = (hidden @ hc_fn.double().T) * rs
    sd, bd = s.double(), b.double()
    pre = torch.sigmoid(mixes[..., :hc] * sd[0] + bd[:hc]) + eps
    post = 2 * torch.sigmoid(mixes[..., hc:2 * hc] * sd[1] + bd[hc:2 * hc])
    comb = mixes[..., 2 * hc:] * sd[2] + bd[2 * hc:]
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
    comb = comb.view(P, hc, hc).reshape(P, hc * hc)
    # --- combine + norm ---
    y = (hidden.view(P, hc, D) * pre.unsqueeze(-1)).sum(dim=1)
    out = y * torch.rsqrt(y.square().mean(-1, keepdim=True) + neps) * weight.double()
    return out, hidden, post, comb


def check(name, got, want, bound):
    denom = want.abs().max().clamp_min(1e-30).item()
    max_abs = (got.double() - want).abs().max().item()
    rel = max_abs / denom
    ok = rel <= bound
    print(f"  {'PASS' if ok else 'FAIL'}  {name:32} max_abs={max_abs:.3e} "
          f"rel={rel:.3e} bound={bound:.3e}")
    return ok


def measure(fn, n):
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
    all_ok = True

    for P in (1, 8):
        print(f"\n=== P={P}, hc={HC}, D={D}, K={K} ===")
        torch.manual_seed(8888 + P)
        x = torch.randn(P, D, dtype=torch.float32)
        residual = torch.randn(P, K, dtype=torch.float32)
        post_in = torch.rand(P, HC, dtype=torch.float32) * 1.5 + 0.2
        ci = torch.rand(P, HC, HC, dtype=torch.float32) + 0.05
        ci = (ci / ci.sum(-1, keepdim=True)).reshape(P, HC * HC).contiguous()
        hc_fn = torch.randn(MIX, K, dtype=torch.float32) / (K ** 0.5)
        hc_fn_t = hc_fn.t().contiguous()
        s = torch.rand(3, dtype=torch.float32) + 0.5
        b = torch.randn(MIX, dtype=torch.float32) * 0.1
        weight = torch.randn(1, D, dtype=torch.float32) * 0.1 + 1.0

        w_out, w_hidden, w_post, w_comb = ref_transition(
            x, residual, post_in, ci, hc_fn, s, b, weight, HC, D, SK, EPS, NEPS)

        bound = 2 * K * EPS_FP32 + 2 * SK * HC * EPS_FP32 + HC * EPS_FP32 + D * EPS_FP32

        xd, rd, pid, cid = x.to(DEV), residual.to(DEV), post_in.to(DEV), ci.to(DEV)
        wtd, sd, bd, wnd = hc_fn_t.to(DEV), s.to(DEV), b.to(DEV), weight.to(DEV)

        f_out, f_hidden, f_post, f_comb = hc_block_transition_kernel(
            xd, rd, pid, cid, wtd, sd, bd, wnd, HC, SK, EPS, NEPS)
        f_out, f_hidden = f_out.to("cpu"), f_hidden.to("cpu")
        f_post, f_comb = f_post.to("cpu"), f_comb.to("cpu")

        all_ok &= check("fused out vs fp64", f_out, w_out, bound)
        all_ok &= check("fused hidden vs fp64", f_hidden, w_hidden, bound)
        all_ok &= check("fused post vs fp64", f_post, w_post, bound)
        all_ok &= check("fused comb vs fp64", f_comb, w_comb, bound)

        # ---- two-kernel path: increment 6 then increment 5 ----
        h_mid = hc_post_kernel(xd, rd, pid, cid)
        t_out, t_post, t_comb = hc_pre_full_kernel(
            h_mid, wtd, sd, bd, wnd, HC, SK, EPS, NEPS, False)
        t_out = t_out.to("cpu")
        all_ok &= check("2-kernel out vs fp64", t_out, w_out, bound)

        d = max((f_out - t_out).abs().max().item(),
                (f_hidden - h_mid.to("cpu")).abs().max().item(),
                (f_post - t_post.to("cpu")).abs().max().item(),
                (f_comb - t_comb.to("cpu")).abs().max().item())
        ok = d <= 1e-6
        print(f"  {'PASS' if ok else 'FAIL'}  {'fused vs 2-kernel':32} "
              f"max_abs={d:.3e} {'(BIT-IDENTICAL)' if d == 0.0 else ''}")
        all_ok &= ok

        rel = ((f_out.double() - w_out).abs().max() / w_out.abs().max()).item()
        print(f"  INFO  out error {BF16_RES / max(rel, 1e-30):.0f}x below bf16 resolution")
        cm = f_comb.view(P, HC, HC).double()
        print(f"  INFO  comb row sums [{cm.sum(-1).min():.4f}, {cm.sum(-1).max():.4f}] "
              f"col sums [{cm.sum(-2).min():.4f}, {cm.sum(-2).max():.4f}]")

        # ---- paired, repeated latency ----
        def fused():
            return hc_block_transition_kernel(xd, rd, pid, cid, wtd, sd, bd, wnd,
                                              HC, SK, EPS, NEPS)

        def two():
            h = hc_post_kernel(xd, rd, pid, cid)
            return hc_pre_full_kernel(h, wtd, sd, bd, wnd, HC, SK, EPS, NEPS, False)

        with torch.inference_mode():
            for _ in range(WARMUP):
                fused()[0].to("cpu")
                two()[0].to("cpu")

        fa, ta, ra = [], [], []
        for _ in range(REPS):
            tf = measure(fused, INNER)
            tt = measure(two, INNER)
            fa.append(tf); ta.append(tt); ra.append(tt / tf)
        print(f"  --- latency, {REPS} reps x {INNER} iters, interleaved ---")
        print(f"  fused      {statistics.mean(fa):8.1f} +- {statistics.stdev(fa):5.1f} us")
        print(f"  two-kernel {statistics.mean(ta):8.1f} +- {statistics.stdev(ta):5.1f} us")
        print(f"  ratio      {statistics.mean(ra):8.3f} +- {statistics.stdev(ra):5.3f}x "
              f"(>1 in {sum(1 for r in ra if r > 1)}/{REPS})")

    print("\n" + ("ALL CHECKS PASS" if all_ok else "SOME CHECKS FAILED"))
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

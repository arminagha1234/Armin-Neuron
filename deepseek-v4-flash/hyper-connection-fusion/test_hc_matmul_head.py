"""Validate the `hc_pre` matmul head (increment 4), and time both transpose paths.

Checks:
  1. numerics vs an fp64 reference, gated on a bound that accounts for the fact that the
     contraction runs over K=16384 terms in fp32 on device while the reference sums in fp64;
  2. the DMA-transpose path and the Tensor-engine-transpose path agree with each other;
  3. which transpose path is actually faster (async-safe timing).

Then, since increment 3 consumes exactly this kernel's output, it runs the two together to
confirm the full `hc_pre` + RMSNorm chain reproduces a pure-torch reference end to end.
"""

import os
import time

import torch

os.environ.setdefault("NEURON_PLATFORM_TARGET_OVERRIDE", "trn2")

import nki  # noqa: E402
import torch_neuronx  # noqa: E402,F401

from nki_hc_matmul_head import hc_matmul_head_kernel  # noqa: E402
from nki_hc_sinkhorn_fused import hc_sinkhorn_combine_norm_kernel  # noqa: E402

DEV = torch.device("neuron")
EPS_FP32 = 1.1920929e-7
BF16_RES = 2.0 ** -8
ITERS = 30
WARMUP = 3


def ref_head(x_flat, hc_fn, eps):
    x = x_flat.double()
    rs = torch.rsqrt(x.square().mean(-1, keepdim=True) + eps)
    return (x @ hc_fn.double().T) * rs


def sinkhorn_cpu(mixes, hc_scale, hc_base, hc, iters, eps):
    mixes = mixes.double()
    s = hc_scale.double()
    b = hc_base.double()
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
    return pre, post, comb.view(P, hc, hc)


def check(name, got, want, bound, extra=""):
    got = got.double()
    denom = want.abs().max().clamp_min(1e-30).item()
    max_abs = (got - want).abs().max().item()
    rel = max_abs / denom
    ok = rel <= bound
    print(f"  {'PASS' if ok else 'FAIL'}  {name:36} max_abs={max_abs:.3e} "
          f"rel={rel:.3e} bound={bound:.3e} {extra}")
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
    print(f"  {label:36} {dt:9.1f} us")
    return dt


def main():
    HC, D, EPS = 4, 4096, 1e-6
    K = HC * D
    MIX = (2 + HC) * HC
    SK_ITERS = 20
    all_ok = True

    for P in (1, 8):
        print(f"\n=== P={P}, K={K}, mix_hc={MIX} ===")
        torch.manual_seed(99 + P)
        x_flat = torch.randn(P, K, dtype=torch.float32)
        hc_fn = (torch.randn(MIX, K, dtype=torch.float32) / (K ** 0.5))
        hc_fn_t = hc_fn.t().contiguous()          # host-side, one-off: it is a constant

        want = ref_head(x_flat, hc_fn, EPS)
        # contraction over K terms in fp32 on device, plus the K-term statistic
        bound = 2 * K * EPS_FP32

        xd = x_flat.to(DEV)
        wd = hc_fn_t.to(DEV)

        got_dma = hc_matmul_head_kernel(xd, wd, EPS, True).to("cpu")
        all_ok &= check("head (dma_transpose) vs fp64", got_dma, want, bound,
                        f"bf16_res={BF16_RES:.2e}")

        got_pe = hc_matmul_head_kernel(xd, wd, EPS, False).to("cpu")
        all_ok &= check("head (nc_transpose) vs fp64", got_pe, want, bound)

        d = (got_dma - got_pe).abs().max().item()
        ok = d <= 1e-5
        print(f"  {'PASS' if ok else 'FAIL'}  {'dma vs nc transpose path':36} "
              f"max_abs={d:.3e} {'(BIT-IDENTICAL)' if d == 0.0 else ''}")
        all_ok &= ok

        rel = ((got_dma.double() - want).abs().max() / want.abs().max()).item()
        print(f"  INFO  error is {BF16_RES / max(rel, 1e-30):.0f}x below bf16 resolution")

        print("  --- latency (async-safe) ---")
        t_dma = timed(lambda: hc_matmul_head_kernel(xd, wd, EPS, True),
                      "head, dma_transpose")
        t_pe = timed(lambda: hc_matmul_head_kernel(xd, wd, EPS, False),
                     "head, nc_transpose")
        faster = "dma_transpose" if t_dma < t_pe else "nc_transpose"
        print(f"  faster path: {faster} ({max(t_dma, t_pe) / min(t_dma, t_pe):.2f}x)")

        # ---- full hc_pre + norm chain: increment 4 feeding increment 3 ----
        print("  --- full hc_pre + RMSNorm chain (increment 4 -> increment 3) ---")
        hc_scale = (torch.rand(3, dtype=torch.float32) + 0.5)
        hc_base = torch.randn(MIX, dtype=torch.float32) * 0.1
        weight = torch.randn(1, D, dtype=torch.float32) * 0.1 + 1.0
        sd, bd, wnd = hc_scale.to(DEV), hc_base.to(DEV), weight.to(DEV)

        mixes_dev = hc_matmul_head_kernel(xd, wd, EPS, True)
        out_dev, _, _ = hc_sinkhorn_combine_norm_kernel(
            mixes_dev, sd, bd, xd, wnd, HC, SK_ITERS, EPS, EPS)
        out_dev = out_dev.to("cpu")

        # fp64 reference for the whole chain
        w_mixes = ref_head(x_flat, hc_fn, EPS)
        w_pre, _, _ = sinkhorn_cpu(w_mixes.float(), hc_scale, hc_base, HC, SK_ITERS, EPS)
        xv = x_flat.double().view(P, HC, D)
        y = (xv * w_pre.unsqueeze(-1)).sum(dim=1)
        w_out = y * torch.rsqrt(y.square().mean(-1, keepdim=True) + EPS) * weight.double()

        chain_bound = bound + 2 * SK_ITERS * HC * EPS_FP32 + HC * EPS_FP32 + D * EPS_FP32
        all_ok &= check("full chain vs fp64", out_dev, w_out, chain_bound)

    print("\n" + ("ALL CHECKS PASS" if all_ok else "SOME CHECKS FAILED"))
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""`nki.simulate` gate for the HC fusion increments, at lnc=1 AND lnc=2.

This is treated as a required gate rather than a nice-to-have: the
simulator executes class-A `sendrecv` / `core_barrier` for real, so a mis-paired hand-off
between fused stages hangs or trips here instead of silently producing wrong numbers on
device. lnc=2 is the case that matters, because that is where the logical-core pairing is
actually exercised.

Run once per lnc setting (the logical-NC configuration is read at import time, so it cannot
be flipped inside a live process):

    NEURON_LOGICAL_NC_CONFIG=1 python3 sim_gate_hc.py
    NEURON_LOGICAL_NC_CONFIG=2 python3 sim_gate_hc.py

Shapes are deliberately reduced (D=256 rather than the model's 4096). The simulator
interprets every instruction in Python, so full width would take a long time for no extra
signal: this gate is about the instruction sequence and the hand-off being well formed, not
about throughput. Numerics are still checked against a NumPy reference.
"""

import os
import sys

import numpy as np

os.environ.setdefault("NEURON_PLATFORM_TARGET_OVERRIDE", "trn2")

import nki  # noqa: E402

from nki_hc_combine_norm import hc_combine_norm_kernel  # noqa: E402
from nki_hc_sinkhorn_fused import hc_sinkhorn_combine_norm_kernel  # noqa: E402

LNC = os.environ.get("NEURON_LOGICAL_NC_CONFIG", "unset")


def ref_combine_norm(x_flat, pre, weight, eps, M, D):
    x = x_flat.astype(np.float64).reshape(-1, M, D)
    y = (x * pre.astype(np.float64)[:, :, None]).sum(axis=1)
    rs = 1.0 / np.sqrt((y ** 2).mean(axis=-1, keepdims=True) + eps)
    return y * rs * weight.astype(np.float64)


def ref_sinkhorn(mixes, scale, base, hc, iters, eps):
    mixes = mixes.astype(np.float64)
    scale = scale.astype(np.float64)
    base = base.astype(np.float64)
    P = mixes.shape[0]

    def sig(v):
        return 1.0 / (1.0 + np.exp(-v))

    pre = sig(mixes[:, :hc] * scale[0] + base[:hc]) + eps
    post = 2.0 * sig(mixes[:, hc:2 * hc] * scale[1] + base[hc:2 * hc])
    comb = mixes[:, 2 * hc:] * scale[2] + base[2 * hc:]
    comb = comb.reshape(P, hc, hc)

    m = comb.max(axis=-1, keepdims=True)
    e = np.exp(comb - m)
    comb = e / e.sum(axis=-1, keepdims=True) + eps

    # column normalise
    comb = comb / (comb.sum(axis=-2, keepdims=True) + eps)
    for _ in range(iters - 1):
        comb = comb / (comb.sum(axis=-1, keepdims=True) + eps)
        comb = comb / (comb.sum(axis=-2, keepdims=True) + eps)
    return pre, post, comb.reshape(P, hc * hc)


def main():
    hc, D, iters, eps = 4, 256, 4, 1e-6
    mix = (2 + hc) * hc
    ok = True

    print(f"=== nki.simulate gate  NEURON_LOGICAL_NC_CONFIG={LNC} "
          f"(hc={hc}, D={D}, sinkhorn_iters={iters}) ===")

    for P in (1, 4):
        rng = np.random.default_rng(7 + P)
        x_flat = rng.standard_normal((P, hc * D), dtype=np.float32)
        pre = np.abs(rng.standard_normal((P, hc))).astype(np.float32) + 0.1
        pre = (pre / pre.sum(-1, keepdims=True)).astype(np.float32)
        weight = (rng.standard_normal((1, D)) * 0.1 + 1.0).astype(np.float32)
        mixes = rng.standard_normal((P, mix), dtype=np.float32)
        scale = (np.abs(rng.standard_normal(3)) + 0.5).astype(np.float32)
        base = (rng.standard_normal(mix) * 0.1).astype(np.float32)

        # ---- increment 2 ----
        got = nki.simulate(hc_combine_norm_kernel)(x_flat, pre, weight, eps)
        want = ref_combine_norm(x_flat, pre, weight, eps, hc, D)
        err = np.abs(got.astype(np.float64) - want).max() / np.abs(want).max()
        good = err < 1e-4
        ok &= good
        print(f"  {'PASS' if good else 'FAIL'}  P={P} increment2 combine+norm    rel={err:.3e}")

        # ---- increment 3 ----
        out, post, comb = nki.simulate(hc_sinkhorn_combine_norm_kernel)(
            mixes, scale, base, x_flat, weight, hc, iters, eps, eps)
        w_pre, w_post, w_comb = ref_sinkhorn(mixes, scale, base, hc, iters, eps)
        w_out = ref_combine_norm(x_flat, w_pre.astype(np.float32), weight, eps, hc, D)

        e_out = np.abs(out.astype(np.float64) - w_out).max() / np.abs(w_out).max()
        e_post = np.abs(post.astype(np.float64) - w_post).max() / np.abs(w_post).max()
        e_comb = np.abs(comb.astype(np.float64) - w_comb).max() / np.abs(w_comb).max()
        good = max(e_out, e_post, e_comb) < 1e-4
        ok &= good
        print(f"  {'PASS' if good else 'FAIL'}  P={P} increment3 sinkhorn+c+n   "
              f"out={e_out:.3e} post={e_post:.3e} comb={e_comb:.3e}")

        cm = comb.astype(np.float64).reshape(P, hc, hc)
        print(f"        comb row sums [{cm.sum(-1).min():.4f}, {cm.sum(-1).max():.4f}] "
              f"col sums [{cm.sum(-2).min():.4f}, {cm.sum(-2).max():.4f}]")

    print(f"\nlnc={LNC}: {'SIM GATE PASS' if ok else 'SIM GATE FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

r"""Phase-1 increment 5: the ENTIRE `hc_pre` plus its RMSNorm, in one kernel.

    x_flat --> statistic --> projection --> mixes --> sinkhorn --> pre --> combine --> RMSNorm
    \_______________________________________ ONE KERNEL ______________________________________/

This composes increment 4 (`head_stages`) with increment 3 (`_sinkhorn_stages`, `_combine`,
`_rmsnorm`), all of which were written as plain functions over SBUF tiles precisely so they
could be assembled here.

WHY THIS IS THE INCREMENT THAT MATTERS MOST
-------------------------------------------
The launch saving is the obvious part. The bigger part is a redundant load:

  * increment 4 loads `x_flat` for the statistic and the projection
  * increment 3 loads `x_flat` **again** for the combine

At K = hc*D = 4*4096 = 16384 fp32 that is 64 KB per partition, read twice. It is by a wide
margin the largest tensor in the boundary — `mixes` is 24 values, `pre`/`post` are 4, `comb`
is 16. Fusing reads it once, so this increment removes real bandwidth rather than only
per-launch overhead.

WHAT STILL CANNOT BE FUSED, AND WHY
-----------------------------------
`hc_post` produces the hidden state that becomes the next boundary's `x_flat`, and the
sinkhorn sits between them, so `hc_post` cannot merge into this kernel. Within `hc_pre`
though, nothing external is consumed at any step, which is what makes this fusion legal. The
earlier note calling the sinkhorn a barrier applies ACROSS boundaries, not within one.

ORDERING NOTE
-------------
The statistic must be computed from `x_flat` BEFORE the combine overwrites nothing (it does
not — the combine writes fresh accumulator tiles), but more importantly `rs` is consumed
inside `head_stages` to scale `mixes`, and the RMSNorm at the end computes its OWN,
different rsqrt over D rather than K. Two distinct statistics over two distinct axes; they
are not interchangeable and are deliberately not shared.

Written against nki 0.6.0.
"""

import nki
import nki.isa as nisa
import nki.language as nl

from nki_hc_matmul_head import head_stages
from nki_hc_sinkhorn_fused import _combine, _rmsnorm, _sinkhorn_stages

_PMAX = 128


@nki.jit
def hc_pre_full_kernel(x_flat, hc_fn_t, hc_scale, hc_base, weight,
                       hc=4, sinkhorn_iters=20, eps=1e-6, norm_eps=1e-6,
                       use_dma_transpose=False):
    """Whole-boundary kernel: `hc_pre` + RMSNorm.

    Args:
        x_flat:   [P, hc*D]     fp32 HBM -- the hc hyper-connection copies.
        hc_fn_t:  [hc*D, mix_hc] fp32 HBM -- projection weight, ALREADY TRANSPOSED.
        hc_scale: [3]           fp32 HBM.
        hc_base:  [mix_hc]      fp32 HBM.
        weight:   [1, D]        fp32 HBM -- RMSNorm per-feature scale.
        hc:       hc_mult.
        sinkhorn_iters, eps, norm_eps: as the model config.
        use_dma_transpose: see `nki_hc_matmul_head`; nc_transpose measured faster.

    Returns:
        out  [P, D]      fp32 -- normalised, combined hidden state.
        post [P, hc]     fp32 -- for the downstream `hc_post`.
        comb [P, hc*hc]  fp32 -- for the downstream `hc_post`.
    """
    P, K = x_flat.shape
    K2, mix_hc = hc_fn_t.shape
    D = K // hc

    assert P <= _PMAX, f"P must be <= {_PMAX}, got {P}"
    assert K2 == K, f"hc_fn_t partition dim {K2} != x_flat free dim {K}"
    assert mix_hc == (2 + hc) * hc, f"mix_hc {mix_hc} != (2+{hc})*{hc}"
    assert K % 128 == 0, f"K={K} must be a multiple of 128"
    assert K * 4 <= nl.tile_size.sbuf_fmax_bytes, (
        f"K={K} fp32 exceeds one SBUF partition ({nl.tile_size.sbuf_fmax_bytes} bytes)"
    )

    out = nl.ndarray((P, D), dtype=nl.float32, buffer=nl.shared_hbm)
    post_o = nl.ndarray((P, hc), dtype=nl.float32, buffer=nl.shared_hbm)
    comb_o = nl.ndarray((P, hc * hc), dtype=nl.float32, buffer=nl.shared_hbm)

    # x_flat read ONCE, then used by both the projection and the combine.
    xt = nl.ndarray((P, K), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=xt, src=x_flat[0:P, 0:K])

    scale_all = nl.ndarray((P, 3), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=scale_all, src=hc_scale.reshape((1, 3)).broadcast(0, P))
    base_all = nl.ndarray((P, mix_hc), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=base_all, src=hc_base.reshape((1, mix_hc)).broadcast(0, P))
    w = nl.ndarray((P, D), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=w, src=weight.broadcast(0, P))

    mixes_sb = head_stages(xt, x_flat, hc_fn_t, P, K, mix_hc, eps, use_dma_transpose)
    pre_sb, post_sb, comb_sb = _sinkhorn_stages(
        mixes_sb, scale_all, base_all, P, hc, sinkhorn_iters, eps)
    acc = _combine(xt, pre_sb, P, hc, D)
    y = _rmsnorm(acc, w, P, D, norm_eps)

    nisa.dma_copy(dst=out, src=y)
    nisa.dma_copy(dst=post_o, src=post_sb)
    nisa.dma_copy(dst=comb_o, src=comb_sb)
    return out, post_o, comb_o

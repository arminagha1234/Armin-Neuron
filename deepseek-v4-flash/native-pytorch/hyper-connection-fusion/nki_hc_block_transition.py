r"""Fusion increment 7: the whole inter-block transition in ONE kernel.

    attn/ffn out ──▶ hc_post ──▶ hidden ──▶ hc_pre ──▶ RMSNorm ──▶ next block's input
                  \_________________ one kernel _____________________/

This is the largest reachable fusion unit in a layer. The genuine barriers are attention
(S5) and the FFN (S11) -- both are large separate kernels and nothing fuses across them.
Everything BETWEEN two blocks is this kernel:

    S7   hidden = hc_post(attn_out, residual, post, comb)
    S8   residual = hidden                    <- a carry, not an operation
    S9   x, post, comb = hc_pre(hidden, ...)
    S10  x = ffn_norm(x)

and identically for S13 -> next layer's S2 -> S3. At 43 layers with `hc_pre`/`hc_post`
appearing 4x per layer, this transition happens **86 times per decoded token** (two blocks
per layer), so it is the unit worth collapsing.

WHY THIS IS THE BIGGEST WIN AVAILABLE
-------------------------------------
`hidden` is `[P, hc*D]` = 64 KB per partition in fp32 at D=4096 -- four times the size of
anything else crossing these boundaries. Unfused it is written to HBM by `hc_post` and read
straight back by `hc_pre`. This kernel keeps it in SBUF. That is a bandwidth saving, not just
a launch saving.

It also removes the redundant read that increment 5 removed, for the same reason: `hidden`
feeds BOTH the projection and the combine.

A NOTE ON THE TRANSPOSE PATH
----------------------------
`head_stages` can transpose via the DMA engines or the tensor engine. The DMA path needs an
HBM source, and here `hidden` never goes to HBM -- so fusion forces the tensor-engine path.
That is convenient: the tensor-engine path measured faster anyway (1.79x at P=1), so the
constraint and the preference agree.

OUTPUTS
-------
Four things leave the kernel, and all four are genuinely needed downstream:

  * `out`    [P, D]     -- normalised input for the next block
  * `hidden` [P, hc*D]  -- becomes the residual for the NEXT hc_post (the S8 carry)
  * `post`   [P, hc]    -- the next hc_post needs it
  * `comb`   [P, hc*hc] -- likewise

`hidden` has to be materialised even though it is an intermediate, because the next
transition consumes it. That is the residual carry, and it is why the chain cannot be
collapsed further than this.

Written against nki 0.6.0.
"""

import nki
import nki.isa as nisa
import nki.language as nl

from nki_hc_matmul_head import head_stages
from nki_hc_post import hc_post_stages
from nki_hc_sinkhorn_fused import _combine, _rmsnorm, _sinkhorn_stages

_PMAX = 128


@nki.jit
def hc_block_transition_kernel(x, residual, post_in, comb_in,
                               hc_fn_t, hc_scale, hc_base, weight,
                               hc=4, sinkhorn_iters=20, eps=1e-6, norm_eps=1e-6):
    """hc_post -> hc_pre -> RMSNorm, fused.

    Args:
        x:        [P, D]         fp32 HBM -- attention or FFN output.
        residual: [P, hc*D]      fp32 HBM -- residual copies carried in.
        post_in:  [P, hc]        fp32 HBM -- from the PREVIOUS hc_pre.
        comb_in:  [P, hc*hc]     fp32 HBM -- from the PREVIOUS hc_pre.
        hc_fn_t:  [hc*D, mix_hc] fp32 HBM -- projection weight, ALREADY TRANSPOSED.
        hc_scale: [3]            fp32 HBM.
        hc_base:  [mix_hc]       fp32 HBM.
        weight:   [1, D]         fp32 HBM -- RMSNorm per-feature scale.

    Returns:
        out    [P, D]
        hidden [P, hc*D]   -- residual carry for the next transition
        post   [P, hc]
        comb   [P, hc*hc]
    """
    P, D = x.shape
    _, K = residual.shape
    K2, mix_hc = hc_fn_t.shape

    assert P <= _PMAX, f"P must be <= {_PMAX}, got {P}"
    assert K == hc * D, f"residual free dim {K} is not hc*D ({hc}*{D})"
    assert K2 == K, f"hc_fn_t partition dim {K2} != {K}"
    assert mix_hc == (2 + hc) * hc, f"mix_hc {mix_hc} != (2+{hc})*{hc}"
    assert comb_in.shape[1] == hc * hc, f"comb_in free dim {comb_in.shape[1]} != {hc * hc}"
    assert K % 128 == 0, f"hc*D={K} must be a multiple of 128"

    out = nl.ndarray((P, D), dtype=nl.float32, buffer=nl.shared_hbm)
    hidden_o = nl.ndarray((P, K), dtype=nl.float32, buffer=nl.shared_hbm)
    post_o = nl.ndarray((P, hc), dtype=nl.float32, buffer=nl.shared_hbm)
    comb_o = nl.ndarray((P, hc * hc), dtype=nl.float32, buffer=nl.shared_hbm)

    xt = nl.ndarray((P, D), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=xt, src=x[0:P, 0:D])
    rt = nl.ndarray((P, K), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=rt, src=residual[0:P, 0:K])
    pin = nl.ndarray((P, hc), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=pin, src=post_in[0:P, 0:hc])
    cin = nl.ndarray((P, hc * hc), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=cin, src=comb_in[0:P, 0:hc * hc])
    scale_all = nl.ndarray((P, 3), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=scale_all, src=hc_scale.reshape((1, 3)).broadcast(0, P))
    base_all = nl.ndarray((P, mix_hc), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=base_all, src=hc_base.reshape((1, mix_hc)).broadcast(0, P))
    w = nl.ndarray((P, D), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=w, src=weight.broadcast(0, P))

    # ---- hc_post: expand back to hc copies. Stays in SBUF. ----
    hidden = hc_post_stages(xt, rt, pin, cin, P, hc, D)

    # ---- hc_pre on that hidden state, all of it ----
    # x_flat=None: the DMA-transpose path needs an HBM source and `hidden` has none, so the
    # tensor-engine path is used. It is also the faster of the two.
    mixes = head_stages(hidden, None, hc_fn_t, P, K, mix_hc, eps, False)
    pre_sb, post_sb, comb_sb = _sinkhorn_stages(
        mixes, scale_all, base_all, P, hc, sinkhorn_iters, eps)
    acc = _combine(hidden, pre_sb, P, hc, D)
    y = _rmsnorm(acc, w, P, D, norm_eps)

    nisa.dma_copy(dst=out, src=y)
    nisa.dma_copy(dst=hidden_o, src=hidden)     # the S8 residual carry
    nisa.dma_copy(dst=post_o, src=post_sb)
    nisa.dma_copy(dst=comb_o, src=comb_sb)
    return out, hidden_o, post_o, comb_o

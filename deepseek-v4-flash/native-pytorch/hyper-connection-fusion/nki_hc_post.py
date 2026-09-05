r"""Fusion increment 6: `hc_post` -- the expand side of the hyper-connection boundary.

WHAT IT COMPUTES
----------------
`hc_pre` reduces `hc` residual copies down to one tensor for attention / the FFN.
`hc_post` does the reverse: it takes that block's output and expands it back to `hc`
copies, mixing in the residual through the sinkhorn's `comb` matrix.

From the model:

    comb_t        = comb.reshape(P, M, M).transpose(-1, -2)
    comb_residual = bmm(comb_t, residual.reshape(P, M, D))
    y             = post.reshape(P, M, 1) * x.reshape(P, 1, D) + comb_residual

which per element is

    y[p, i, d] = post[p, i] * x[p, d]  +  sum_j comb[p, j, i] * residual[p, j, d]

Note the transpose: `comb_t[i, j] = comb[j, i]`, so the sum runs over the FIRST index of
`comb`. Getting that backwards produces a plausible-looking wrong answer, since `comb` is
nearly symmetric after a sinkhorn -- both row and column sums are 1. The test checks
against an fp64 reference built from the model's own expression rather than from this
restatement, so an index slip cannot pass.

A CORRECTION TO WHAT I WROTE EARLIER
------------------------------------
Increment 5's notes say `hc_post` "cannot join" the `hc_pre` kernel because it produces the
`x_flat` that kernel consumes. That conflated *attribution* with *dependency* and is wrong.
Walking the layer's `forward()`:

    S7   hidden = hc_post(attn_out, residual, post, comb)
    S8   residual = hidden                       <- a carry, not an operation
    S9   x, post, comb = hc_pre(hidden, ...)
    S10  x = ffn_norm(x)

Nothing external is consumed between S7 and S10, so `hc_post` CAN fuse forward into the
`hc_pre` + norm kernel. The same holds for S13 -> next layer's S2 -> S3.

The genuine barriers in a layer are **attention (S5) and the FFN (S11)** -- those are large
separate kernels, and no fusion crosses them. So the reachable unit is:

    hc_post -> hc_pre -> norm        (one kernel, between the two big blocks)

This file is `hc_post` on its own, which is the increment; fusing it forward is the next one.
Keeping it standalone first also gives the A/B baseline that fusion has to beat.

INSTRUCTION BUDGET
------------------
For each of the `hc` output copies: one `tensor_scalar` to start the accumulator from
`post[:, i] * x`, then `hc` fused multiply-accumulates via `scalar_tensor_tensor`, each
computing `(residual_j * comb[j, i]) + acc` in a SINGLE instruction. Both `post[:, i]` and
`comb[j, i]` are per-partition `[P, 1]` values, which is exactly the operand0 form those
instructions take.

    hc * (1 + hc)  =  20 instructions at hc=4

The result is `[P, hc*D]` = 64 KB per partition in fp32 at D=4096 -- four times the size of
`hc_pre`'s output, which is why this boundary is the more expensive of the two to leave
unfused.

Written against nki 0.6.0.
"""

import nki
import nki.isa as nisa
import nki.language as nl

_PMAX = 128


def hc_post_stages(xt, rt, post_sb, comb_sb, P, M, D):
    """`hc_post` on SBUF tiles, so a later increment can fuse it forward.

    Args:
        xt:      [P, D]     SBUF -- the block output (attention or FFN).
        rt:      [P, M*D]   SBUF -- the residual, M copies contiguous.
        post_sb: [P, M]     SBUF -- per-copy scale from the sinkhorn.
        comb_sb: [P, M*M]   SBUF -- flat row-major mixing matrix.
        P, M, D: shapes.

    Returns:
        [P, M*D] SBUF tile holding the expanded copies.
    """
    out = nl.ndarray((P, M * D), dtype=nl.float32, buffer=nl.sbuf)

    # SBUF budget: a fresh [P, D] tile per accumulation step would be M*(M+1) = 20 live
    # tiles at hc=4, i.e. 20 * 16 KB = 320 KB per partition at D=4096 -- over the partition
    # budget. Standalone the allocator's liveness analysis absorbs that, but once this stage
    # is fused with hc_pre (whose own tiles stack on top) it overflows, and SBUF overflow
    # between fused stages surfaces as NCC_INLA001. Ping-ponging between TWO tiles caps the
    # accumulator at 32 KB per partition regardless of hc, which is what makes fusion fit.
    ping = nl.ndarray((P, D), dtype=nl.float32, buffer=nl.sbuf)
    pong = nl.ndarray((P, D), dtype=nl.float32, buffer=nl.sbuf)

    for i in range(M):
        # start the accumulator: post[:, i] * x
        nisa.tensor_scalar(dst=ping, data=xt,
                           op0=nl.multiply, operand0=post_sb[0:P, i:i + 1])
        cur, nxt = ping, pong

        # + sum_j comb[j, i] * residual_j, one instruction per j
        for j in range(M):
            # comb is flat row-major [P, M*M]: element (j, i) sits at j*M + i.
            # The sum is over the FIRST index, matching comb_t = comb.transpose(-1,-2).
            cji = comb_sb[0:P, j * M + i:j * M + i + 1]
            nisa.scalar_tensor_tensor(dst=nxt,
                                      data=rt[0:P, j * D:(j + 1) * D],
                                      op0=nl.multiply, operand0=cji,
                                      op1=nl.add, operand1=cur)
            cur, nxt = nxt, cur          # swap; never read and write the same tile

        # place this copy into its slice of the output
        nisa.tensor_copy(dst=out[0:P, i * D:(i + 1) * D], src=cur)

    return out


@nki.jit
def hc_post_kernel(x, residual, post, comb):
    """Standalone `hc_post`.

    Args:
        x:        [P, D]    fp32 HBM -- block output.
        residual: [P, M*D]  fp32 HBM -- residual, M copies contiguous.
        post:     [P, M]    fp32 HBM.
        comb:     [P, M*M]  fp32 HBM -- flat row-major.

    Returns:
        [P, M*D] fp32 expanded copies.
    """
    P, D = x.shape
    _, MD = residual.shape
    _, M = post.shape

    assert P <= _PMAX, f"P must be <= {_PMAX}, got {P}"
    assert MD == M * D, f"residual free dim {MD} is not M*D ({M}*{D})"
    assert comb.shape[1] == M * M, f"comb free dim {comb.shape[1]} != M*M ({M * M})"
    assert MD * 4 <= nl.tile_size.sbuf_fmax_bytes, (
        f"M*D={MD} fp32 exceeds one SBUF partition "
        f"({nl.tile_size.sbuf_fmax_bytes} bytes); tile the free dim"
    )

    out = nl.ndarray((P, MD), dtype=nl.float32, buffer=nl.shared_hbm)

    xt = nl.ndarray((P, D), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=xt, src=x[0:P, 0:D])
    rt = nl.ndarray((P, MD), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=rt, src=residual[0:P, 0:MD])
    pt = nl.ndarray((P, M), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=pt, src=post[0:P, 0:M])
    ct = nl.ndarray((P, M * M), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=ct, src=comb[0:P, 0:M * M])

    y = hc_post_stages(xt, rt, pt, ct, P, M, D)

    nisa.dma_copy(dst=out[0:P, 0:MD], src=y)
    return out

r"""Phase-1 increment 3: fuse the HC sinkhorn with the combine + RMSNorm.

WHERE THIS SITS
---------------
Increment 2 (`nki_hc_combine_norm.py`) fused stage S2-tail with S3 and measured
1.21x (P=1) / 1.34x (P=8) against the two-kernel path, bit-identical. The discipline then
says: add ONE stage at a time. The stage immediately upstream of the combine is the
sinkhorn, so that is this file.

    mixes -> [ sinkhorn ] -> pre, post, comb -> [ combine ] -> y -> [ RMSNorm ] -> out
             \____________________ ONE KERNEL ___________________________________/

WHY THE SINKHORN IS THE STAGE THAT MATTERS
------------------------------------------
While mapping the fusion I found the structural reason the HC chain cannot simply be fused
end to end: `pre` is produced by a sinkhorn over `mixes`, and `mixes` is a projection of the
hidden state that `hc_post` produces. So the dependency chain is

    hc_post -> hidden -> statistic+matmul -> mixes -> SINKHORN -> pre -> combine -> norm

The sinkhorn is a hard serial barrier in the middle of every hyper-connection boundary. It
is also arithmetically trivial -- 20 iterations of row/column normalisation on a 4x4 matrix
per token -- so it contributes almost no FLOPs while forcing a kernel split. That makes it
the highest-value thing to absorb, and it is why this increment targets it rather than the
matmul head (which is a PE-engine job and belongs in a later increment, where a failure
cannot be confused with a Vector/Scalar-engine bug).

Fusing it removes, per HC boundary: the `pre` store+load, the `y` store+load, and two of
three kernel launches. Given the measured behaviour is launch-bound at these sizes, the
launch reduction is expected to dominate.

CREDIT / PROVENANCE
-------------------
The sinkhorn body here is a port of the model's existing `hc_split_sinkhorn_nki`
(`nki_hc.py`), reorganised so its stages can be called on SBUF tiles inside a larger
kernel rather than only as a standalone kernel with HBM in/out. Two changes:

  * `_broadcast_p0`'s `nc_stream_shuffle` is replaced with a DMA broadcast from the HBM
    source (`t.broadcast(0, P)`). Broadcasting the partition dim of an ON-CHIP tensor is
    rejected by the compiler ("Cannot broadcast partition dim (dim 0) for on-chip
    tensors"), but broadcasting an HBM source during the DMA is allowed, and it is fewer
    instructions than a shuffle group loop.
  * the per-tile `P_MAX` padding is dropped in favour of exact `P` tiles. Decode runs
    P = B*S = 1..8, so padding every tile to 128 partitions wastes SBUF for no benefit.

The column normalisation trick is kept exactly as the original had it, because it is the
clever part: `comb` is held FLAT as [P, hc*hc] row-major, so column j lives at flat
positions j, hc+j, 2*hc+j, ... and a column sum is just `hc` strided `tensor_tensor` adds
on [P, hc] slices. No transposes anywhere, which is what the CPU reference needs a pile of
`.contiguous()` calls to emulate.

Written against nki 0.6.0.
"""

import nki
import nki.isa as nisa
import nki.language as nl

_PMAX = 128


# ---------------------------------------------------------------------------
# sinkhorn stages, operating on SBUF tiles so they can be fused
# ---------------------------------------------------------------------------

def _row_normalize(comb_sb, P, hc, eps):
    """comb[row, :] /= (row_sum + eps), for each of the hc rows."""
    for row in nl.affine_range(hc):
        s, e = row * hc, row * hc + hc
        row_sum = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=row_sum, data=comb_sb[0:P, s:e], op=nl.add, axis=1)
        nisa.tensor_scalar(dst=row_sum, data=row_sum, op0=nl.add, operand0=eps)
        inv = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.reciprocal(dst=inv, data=row_sum)
        nisa.tensor_scalar(dst=comb_sb[0:P, s:e], data=comb_sb[0:P, s:e],
                           op0=nl.multiply, operand0=inv)


def _col_normalize(comb_sb, P, hc, eps):
    """comb[:, col] /= (col_sum + eps).

    `comb_sb` is flat row-major [P, hc*hc], so the hc values of row r occupy
    [r*hc, r*hc+hc) and column j is the j-th element of every such slice. Summing the hc
    row-slices elementwise therefore yields all hc column sums at once in [P, hc].
    """
    col_sum = nl.ndarray((P, hc), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=col_sum, value=0.0)
    for row in nl.affine_range(hc):
        s, e = row * hc, row * hc + hc
        nisa.tensor_tensor(dst=col_sum, data1=col_sum,
                           data2=comb_sb[0:P, s:e], op=nl.add)
    nisa.tensor_scalar(dst=col_sum, data=col_sum, op0=nl.add, operand0=eps)
    inv = nl.ndarray((P, hc), dtype=nl.float32, buffer=nl.sbuf)
    nisa.reciprocal(dst=inv, data=col_sum)
    for row in nl.affine_range(hc):
        s, e = row * hc, row * hc + hc
        nisa.tensor_tensor(dst=comb_sb[0:P, s:e], data1=comb_sb[0:P, s:e],
                           data2=inv, op=nl.multiply)


def _sinkhorn_stages(mixes_sb, scale_all, base_all, P, hc, iters, eps):
    """Split `mixes` into pre / post / comb, all as SBUF tiles.

    Returns (pre_sb [P,hc], post_sb [P,hc], comb_sb [P,hc*hc]).
    """
    mix_hc = (2 + hc) * hc

    # PRE: sigmoid(mixes[:, :hc] * scale[0] + base[:hc]) + eps
    pre_sb = nl.ndarray((P, hc), dtype=nl.float32, buffer=nl.sbuf)
    # scale is a per-partition [P,1] operand; base varies along the free dim so it must be a
    # tile -> scalar_tensor_tensor does (mixes*scale)+base in ONE instruction.
    nisa.scalar_tensor_tensor(dst=pre_sb, data=mixes_sb[0:P, 0:hc],
                              op0=nl.multiply, operand0=scale_all[0:P, 0:1],
                              op1=nl.add, operand1=base_all[0:P, 0:hc])
    nisa.activation(dst=pre_sb, data=pre_sb, op=nl.sigmoid)
    nisa.tensor_scalar(dst=pre_sb, data=pre_sb, op0=nl.add, operand0=eps)

    # POST: 2 * sigmoid(mixes[:, hc:2hc] * scale[1] + base[hc:2hc])
    post_sb = nl.ndarray((P, hc), dtype=nl.float32, buffer=nl.sbuf)
    nisa.scalar_tensor_tensor(dst=post_sb, data=mixes_sb[0:P, hc:2 * hc],
                              op0=nl.multiply, operand0=scale_all[0:P, 1:2],
                              op1=nl.add, operand1=base_all[0:P, hc:2 * hc])
    nisa.activation(dst=post_sb, data=post_sb, op=nl.sigmoid)
    nisa.tensor_scalar(dst=post_sb, data=post_sb, op0=nl.multiply, operand0=2.0)

    # COMB: (mixes[:, 2hc:] * scale[2] + base[2hc:]) -> row softmax + eps -> sinkhorn
    comb_sb = nl.ndarray((P, hc * hc), dtype=nl.float32, buffer=nl.sbuf)
    nisa.scalar_tensor_tensor(dst=comb_sb, data=mixes_sb[0:P, 2 * hc:mix_hc],
                              op0=nl.multiply, operand0=scale_all[0:P, 2:3],
                              op1=nl.add, operand1=base_all[0:P, 2 * hc:mix_hc])

    for row in nl.affine_range(hc):
        s, e = row * hc, row * hc + hc
        row_max = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=row_max, data=comb_sb[0:P, s:e], op=nl.maximum, axis=1)
        nisa.tensor_scalar(dst=comb_sb[0:P, s:e], data=comb_sb[0:P, s:e],
                           op0=nl.subtract, operand0=row_max)
        # NOTE: softmax needs BOTH exp(x) and its sum, so the fused activation+reduce form
        # is deliberately NOT used here -- it would have to run exp twice. That is the
        # documented trap (measured 19% slower); the same reason the router kernel's L1
        # denominator stayed a separate reduce.
        nisa.activation(dst=comb_sb[0:P, s:e], data=comb_sb[0:P, s:e], op=nl.exp)
        row_sum = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=row_sum, data=comb_sb[0:P, s:e], op=nl.add, axis=1)
        inv = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.reciprocal(dst=inv, data=row_sum)
        # (comb * inv) + eps in one instruction
        nisa.tensor_scalar(dst=comb_sb[0:P, s:e], data=comb_sb[0:P, s:e],
                           op0=nl.multiply, operand0=inv,
                           op1=nl.add, operand1=eps)

    _col_normalize(comb_sb, P, hc, eps)
    # sequential_range, not affine_range: each iteration reads the previous one's result.
    for _ in nl.sequential_range(iters - 1):
        _row_normalize(comb_sb, P, hc, eps)
        _col_normalize(comb_sb, P, hc, eps)

    return pre_sb, post_sb, comb_sb


# ---------------------------------------------------------------------------
# combine + norm (from increment 2, validated bit-identical)
# ---------------------------------------------------------------------------

def _combine(xt, pre_sb, P, M, D):
    """y = sum_m pre[:, m] * x[:, m*D:(m+1)*D] -- M fused multiply-accumulates."""
    accs = []
    for m in range(M):
        a = nl.ndarray((P, D), dtype=nl.float32, buffer=nl.sbuf)
        xm = xt[0:P, m * D:(m + 1) * D]
        pm = pre_sb[0:P, m:m + 1]
        if m == 0:
            nisa.tensor_scalar(dst=a, data=xm, op0=nl.multiply, operand0=pm)
        else:
            nisa.scalar_tensor_tensor(dst=a, data=xm,
                                      op0=nl.multiply, operand0=pm,
                                      op1=nl.add, operand1=accs[-1])
        accs.append(a)
    return accs[-1]


def _rmsnorm(yt, w, P, D, eps):
    """4 instructions: fused square+sum, fused scale+bias, rsqrt, fused double-multiply."""
    sq = nl.ndarray((P, D), dtype=nl.float32, buffer=nl.sbuf)
    sum_sq = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=sq, data=yt, op=nl.square,
                    reduce_op=nl.add, reduce_res=sum_sq,
                    reduce_cmd=nisa.reduce_cmd.reset_reduce)
    var = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=var, data=sum_sq,
                       op0=nl.multiply, operand0=1.0 / D,
                       op1=nl.add, operand1=eps)
    rs = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=rs, data=var, op=nl.rsqrt)
    y = nl.ndarray((P, D), dtype=nl.float32, buffer=nl.sbuf)
    nisa.scalar_tensor_tensor(dst=y, data=yt,
                              op0=nl.multiply, operand0=rs,
                              op1=nl.multiply, operand1=w)
    return y


# ---------------------------------------------------------------------------
# kernels
# ---------------------------------------------------------------------------

@nki.jit
def hc_sinkhorn_kernel(mixes, hc_scale, hc_base, hc=4, sinkhorn_iters=20, eps=1e-6):
    """Unfused reference: sinkhorn only, pre/post/comb to HBM.

    Exists so the benchmark compares the fused kernel against real kernels rather than
    against torch.
    """
    P, mix_hc = mixes.shape
    assert P <= _PMAX, f"P must be <= {_PMAX}, got {P}"
    assert mix_hc == (2 + hc) * hc, f"mix_hc {mix_hc} != (2+{hc})*{hc}"

    pre_o = nl.ndarray((P, hc), dtype=nl.float32, buffer=nl.shared_hbm)
    post_o = nl.ndarray((P, hc), dtype=nl.float32, buffer=nl.shared_hbm)
    comb_o = nl.ndarray((P, hc * hc), dtype=nl.float32, buffer=nl.shared_hbm)

    m_sb = nl.ndarray((P, mix_hc), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=m_sb, src=mixes[0:P, 0:mix_hc])
    scale_all = nl.ndarray((P, 3), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=scale_all, src=hc_scale.reshape((1, 3)).broadcast(0, P))
    base_all = nl.ndarray((P, mix_hc), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=base_all, src=hc_base.reshape((1, mix_hc)).broadcast(0, P))

    pre_sb, post_sb, comb_sb = _sinkhorn_stages(
        m_sb, scale_all, base_all, P, hc, sinkhorn_iters, eps)

    nisa.dma_copy(dst=pre_o, src=pre_sb)
    nisa.dma_copy(dst=post_o, src=post_sb)
    nisa.dma_copy(dst=comb_o, src=comb_sb)
    return pre_o, post_o, comb_o


@nki.jit
def hc_sinkhorn_combine_norm_kernel(mixes, hc_scale, hc_base, x_flat, weight,
                                    hc=4, sinkhorn_iters=20, eps=1e-6,
                                    norm_eps=1e-6):
    """FUSED sinkhorn + combine + RMSNorm: the whole tail of `hc_pre` plus its norm.

    `pre` and the combined hidden state `y` never leave SBUF.

    Args:
        mixes:    [P, (2+hc)*hc] fp32 HBM -- the projection output.
        hc_scale: [3]            fp32 HBM.
        hc_base:  [(2+hc)*hc]    fp32 HBM.
        x_flat:   [P, hc*D]      fp32 HBM -- the hc hyper-connection copies.
        weight:   [1, D]         fp32 HBM -- RMSNorm per-feature scale.
        hc:       number of HC copies (hc_mult).
        sinkhorn_iters, eps, norm_eps: as the model config.

    Returns:
        out  [P, D]      fp32 -- normalised, combined hidden state.
        post [P, hc]     fp32 -- needed downstream by `hc_post`.
        comb [P, hc*hc]  fp32 -- needed downstream by `hc_post`.
    """
    P, mix_hc = mixes.shape
    _, hcD = x_flat.shape
    D = hcD // hc

    assert P <= _PMAX, f"P must be <= {_PMAX}, got {P}"
    assert mix_hc == (2 + hc) * hc, f"mix_hc {mix_hc} != (2+{hc})*{hc}"
    assert hcD == hc * D, f"x_flat free dim {hcD} is not hc*D ({hc}*{D})"
    assert hcD * 4 <= nl.tile_size.sbuf_fmax_bytes, (
        f"hc*D={hcD} fp32 exceeds one SBUF partition "
        f"({nl.tile_size.sbuf_fmax_bytes} bytes); tile the free dim"
    )

    out = nl.ndarray((P, D), dtype=nl.float32, buffer=nl.shared_hbm)
    post_o = nl.ndarray((P, hc), dtype=nl.float32, buffer=nl.shared_hbm)
    comb_o = nl.ndarray((P, hc * hc), dtype=nl.float32, buffer=nl.shared_hbm)

    m_sb = nl.ndarray((P, mix_hc), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=m_sb, src=mixes[0:P, 0:mix_hc])
    scale_all = nl.ndarray((P, 3), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=scale_all, src=hc_scale.reshape((1, 3)).broadcast(0, P))
    base_all = nl.ndarray((P, mix_hc), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=base_all, src=hc_base.reshape((1, mix_hc)).broadcast(0, P))
    xt = nl.ndarray((P, hcD), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=xt, src=x_flat[0:P, 0:hcD])
    w = nl.ndarray((P, D), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=w, src=weight.broadcast(0, P))

    pre_sb, post_sb, comb_sb = _sinkhorn_stages(
        m_sb, scale_all, base_all, P, hc, sinkhorn_iters, eps)

    acc = _combine(xt, pre_sb, P, hc, D)     # pre never hits HBM
    y = _rmsnorm(acc, w, P, D, norm_eps)     # y never hits HBM

    nisa.dma_copy(dst=out, src=y)
    nisa.dma_copy(dst=post_o, src=post_sb)
    nisa.dma_copy(dst=comb_o, src=comb_sb)
    return out, post_o, comb_o

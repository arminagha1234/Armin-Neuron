"""Fusion increment: fuse the `hc_pre` combine (S2 tail) with `attn_norm` (S3).

WHAT THIS FUSES AND WHY THESE TWO
---------------------------------
The discipline followed here is to fuse exactly TWO adjacent stages, compile,
sim-check at lnc=1 AND lnc=2, and only then add a third. From the stage list, the adjacent
pair with the best ratio of (boundaries removed) to (implementation risk) is:

    S2 tail   y = sum_m pre[:, m] * x_flat.view(P, M, D)[:, m, :]      -> [P, D]
    S3        attn_norm(y) = y * rsqrt(mean(y^2) + eps) * weight       -> [P, D]

`hc_pre` runs 4x per layer (S2, S7, S9, S13); over 43 layers that is 172 HC boundaries per
decoded token. Unfused, every one of them writes `y` to HBM and immediately reads it back.

Deliberately NOT included in this increment:
  * the sinkhorn (`_hc_split_sinkhorn`) that produces `pre` -- it already has an NKI twin,
    and pulling a 20-iteration normalisation into the first increment violates the
    two-adjacent-ops rule.
  * the `mixes = matmul(x_flat, hc_fn.t()) * rsqrt` head of `hc_pre` -- that is a PE-engine
    op; mixing engine classes into a first increment makes a failure ambiguous.

Real V4-Flash shapes: D = hidden_size = 4096, M = hc_mult = 4, so `x_flat` is [P, 16384]
(64 KB per partition in fp32, inside the SBUF partition budget) and `pre` is [P, 4].
At decode P = B*S = 1..8.

INSTRUCTION BUDGET
------------------
Combine is M fused multiply-accumulates: the first is a `tensor_scalar` (no accumulator to
add yet), each subsequent one is a `scalar_tensor_tensor` computing
`(x_m * pre_m) + acc` in a SINGLE instruction. `pre[:, m]` is a per-partition [P,1] vector,
which is precisely the operand0 form both instructions accept.

Then RMSNorm proper is the 4-instruction sequence established in `nki_rmsnorm_fused.py`
(fused square+sum, fused scale+bias, rsqrt, fused double-multiply).

    combine: M     instructions  (4 at hc_mult=4)
    norm:    4     instructions
    total:   8     instructions, one HBM read of x, one HBM write of the result

The unfused pair costs the same arithmetic PLUS a [P, D] store and a [P, D] load. That is
the boundary this kernel deletes, 172 times per token.

HONEST SCOPE
------------
The comparison that actually tests the fusion claim is fused-NKI vs two-separate-NKI, and
that is what `test_hc_combine_norm.py` measures. A comparison against eager torch is
context only: in the real model these ops sit inside a compiled graph that already fuses
some of them, so an eager speedup must NOT be read as an end-to-end gain. This is the same
correction that applied to the router kernel.

Written against nki 0.6.0 (`nisa.rsqrt` does not exist in this version; the 0.4.0 constraint
sheet is wrong about that -- `nisa.activation(op=nl.rsqrt)` is the working form).
"""

import nki
import nki.isa as nisa
import nki.language as nl

_PMAX = 128


@nki.jit
def hc_combine_kernel(x_flat, pre):
    """Unfused reference stage S2-tail: `y = sum_m pre[:, m] * x_flat.view(P, M, D)[:, m, :]`.

    Exists so the benchmark can measure the fused kernel against two real kernels rather
    than against torch.

    Args:
        x_flat: [P, M*D] fp32 in HBM -- the M hyper-connection copies, contiguous.
        pre:    [P, M]   fp32 in HBM -- per-copy mix weights from the sinkhorn.

    Returns:
        [P, D] fp32 combined hidden state.
    """
    P, MD = x_flat.shape
    _, M = pre.shape
    D = MD // M
    out = nl.ndarray((P, D), dtype=nl.float32, buffer=nl.shared_hbm)

    assert P <= _PMAX, f"P must be <= {_PMAX}, got {P}"
    assert MD == M * D, f"x_flat free dim {MD} is not M*D ({M}*{D})"
    assert MD * 4 <= nl.tile_size.sbuf_fmax_bytes, (
        f"M*D={MD} fp32 exceeds one SBUF partition "
        f"({nl.tile_size.sbuf_fmax_bytes} bytes); tile the free dim"
    )

    xt = nl.ndarray((P, MD), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=xt, src=x_flat[0:P, 0:MD])
    pt = nl.ndarray((P, M), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=pt, src=pre[0:P, 0:M])

    acc = _combine(xt, pt, P, M, D)
    nisa.dma_copy(dst=out[0:P, 0:D], src=acc)
    return out


def _combine(xt, pt, P, M, D):
    """M fused multiply-accumulates over the hyper-connection copies.

    Returns an SBUF [P, D] tile. Separate destination tiles are used per step rather than
    accumulating in place: an instruction that both reads and writes the same tile is a
    read/write hazard, and the rule followed here is to stay on auto-alloc and
    avoids hand-managed aliasing.
    """
    accs = []
    for m in range(M):
        a = nl.ndarray((P, D), dtype=nl.float32, buffer=nl.sbuf)
        xm = xt[0:P, m * D:(m + 1) * D]
        pm = pt[0:P, m:m + 1]          # [P,1] per-partition scalar
        if m == 0:
            # nothing to accumulate onto yet
            nisa.tensor_scalar(dst=a, data=xm, op0=nl.multiply, operand0=pm)
        else:
            # (x_m * pre_m) + acc_{m-1}  in ONE instruction
            nisa.scalar_tensor_tensor(dst=a, data=xm,
                                      op0=nl.multiply, operand0=pm,
                                      op1=nl.add, operand1=accs[-1])
        accs.append(a)
    return accs[-1]


def _rmsnorm_inplace(yt, w, P, D, eps):
    """The 4-instruction RMSNorm on an SBUF tile. Returns a new SBUF [P, D] tile."""
    # (1) squares AND their sum in one Scalar Engine instruction. reset_reduce clears the
    # shared reduction registers, so the result does not depend on prior activation calls.
    sq = nl.ndarray((P, D), dtype=nl.float32, buffer=nl.sbuf)
    sum_sq = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=sq, data=yt, op=nl.square,
                    reduce_op=nl.add, reduce_res=sum_sq,
                    reduce_cmd=nisa.reduce_cmd.reset_reduce)

    # (2) var = sum_sq * (1/D) + eps -- two operators, one instruction.
    var = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=var, data=sum_sq,
                       op0=nl.multiply, operand0=1.0 / D,
                       op1=nl.add, operand1=eps)

    # (3) rsqrt.
    rs = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=rs, data=var, op=nl.rsqrt)

    # (4) (y * rsqrt) * weight -- both multiplies in one instruction.
    y = nl.ndarray((P, D), dtype=nl.float32, buffer=nl.sbuf)
    nisa.scalar_tensor_tensor(dst=y, data=yt,
                              op0=nl.multiply, operand0=rs,
                              op1=nl.multiply, operand1=w)
    return y


@nki.jit
def hc_rmsnorm_only_kernel(y_in, weight, eps):
    """Unfused reference stage S3: RMSNorm of an already-combined hidden state.

    Same body as `nki_rmsnorm_fused.rmsnorm_kernel`, restated here so the A/B benchmark
    measures the two-kernel path with the exact code the fused kernel is built from.
    """
    P, D = y_in.shape
    out = nl.ndarray((P, D), dtype=nl.float32, buffer=nl.shared_hbm)

    assert P <= _PMAX, f"P must be <= {_PMAX}, got {P}"
    assert D * 4 <= nl.tile_size.sbuf_fmax_bytes, (
        f"D={D} fp32 exceeds one SBUF partition; tile the free dim"
    )

    yt = nl.ndarray((P, D), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=yt, src=y_in[0:P, 0:D])

    # weight is per-FEATURE so it must be a full tile. Broadcasting the partition dim of an
    # ON-CHIP tensor is rejected ("Cannot broadcast partition dim (dim 0) for on-chip
    # tensors"); broadcasting the HBM source during the DMA is allowed.
    w = nl.ndarray((P, D), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=w, src=weight.broadcast(0, P))

    y = _rmsnorm_inplace(yt, w, P, D, eps)
    nisa.dma_copy(dst=out[0:P, 0:D], src=y)
    return out


@nki.jit
def hc_combine_norm_kernel(x_flat, pre, weight, eps):
    """FUSED S2-tail + S3: combine the M hyper-connection copies, then RMSNorm, in one kernel.

    This is the Phase-1 increment. The [P, D] intermediate `y` never leaves SBUF, which is
    the whole point: it removes one HBM store + one HBM load per HC boundary, 172 times per
    decoded token at 43 layers.

    Args:
        x_flat: [P, M*D] fp32 in HBM -- M hyper-connection copies, contiguous.
        pre:    [P, M]   fp32 in HBM -- per-copy mix weights.
        weight: [1, D]   fp32 in HBM -- RMSNorm per-feature scale.
        eps:    variance epsilon.

    Returns:
        [P, D] fp32 normalised, combined hidden state.
    """
    P, MD = x_flat.shape
    _, M = pre.shape
    D = MD // M
    out = nl.ndarray((P, D), dtype=nl.float32, buffer=nl.shared_hbm)

    assert P <= _PMAX, f"P must be <= {_PMAX}, got {P}"
    assert MD == M * D, f"x_flat free dim {MD} is not M*D ({M}*{D})"
    assert MD * 4 <= nl.tile_size.sbuf_fmax_bytes, (
        f"M*D={MD} fp32 exceeds one SBUF partition "
        f"({nl.tile_size.sbuf_fmax_bytes} bytes); tile the free dim"
    )

    xt = nl.ndarray((P, MD), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=xt, src=x_flat[0:P, 0:MD])
    pt = nl.ndarray((P, M), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=pt, src=pre[0:P, 0:M])
    w = nl.ndarray((P, D), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=w, src=weight.broadcast(0, P))

    acc = _combine(xt, pt, P, M, D)      # stays in SBUF -- the deleted boundary
    y = _rmsnorm_inplace(acc, w, P, D, eps)

    nisa.dma_copy(dst=out[0:P, 0:D], src=y)
    return out

"""Phase-1 increment 4: the `hc_pre` matmul head -- statistic + projection -> `mixes`.

This is the last stage of `hc_pre` that was still outside NKI. With it, the whole of
`hc_pre` plus its RMSNorm can be expressed as kernels:

    x_flat --> [ statistic + matmul ]  -->  mixes
                       (this file)            |
                                              v
               [ sinkhorn + combine + RMSNorm ]  --> normalised hidden state
                       (increment 3)

The maths:

    rsqrt = 1 / sqrt( mean(x_flat^2, axis=-1) + eps )        # over K = hc*D
    mixes = (x_flat @ hc_fn.T) * rsqrt                       # [P, mix_hc]

WHY THIS WAS DEFERRED UNTIL LAST
--------------------------------
Increments 2 and 3 live entirely on the Scalar/Vector engines. This one needs the Tensor
engine, because the contraction is over K = hc*D = 16384 and `nc_matmul` only ever contracts
over the PARTITION axis. Keeping engine classes separate across increments was deliberate:
if a fused kernel spanning three engine types fails, the failure is ambiguous.

THE OPERAND CHOICE THAT AVOIDS A FINAL TRANSPOSE
------------------------------------------------
`nc_matmul(dst, stationary, moving)` computes `dst = stationary.T @ moving`, contracting over
the partition axis of both operands. The obvious assignment (weight stationary, activation
moving) yields `[mix_hc, P]`, i.e. `mixes` transposed, needing another transpose to fix.

Putting the ACTIVATION in the stationary slot instead:

    stationary = x_chunk^T   [128 (K), P]        -> stationary.T = [P, 128]
    moving     = hc_fn^T     [128 (K), mix_hc]
    dst        = [P, mix_hc]                     <- already the layout we want

and the engine limits are satisfied comfortably: stationary free dim = P <= 8 (limit 128),
moving free dim = mix_hc = 24 (limit 512), PSUM free dim 24 (limit 512).

GETTING K ONTO THE PARTITION AXIS
---------------------------------
`x_flat` arrives as [P, K] -- K is on the free axis, so it has to be transposed in 128-wide
chunks. Two ways, both implemented here:

  * `nc_transpose` (DEFAULT): the pattern the model's existing kernels use. Tensor-engine
    transpose writes to PSUM, so it needs a PSUM tile plus a copy back to SBUF before the
    result can serve as a matmul operand -- 3 ops per chunk, and it shares an engine with
    the matmul.
  * `dma_transpose` (`use_dma_transpose=True`): transposes during the HBM->SBUF copy, so a
    chunk costs one DMA and the work lands on the DMA engines instead. No PSUM round-trip.

I expected `dma_transpose` to win, on the reasoning that it moves work off the Tensor engine
and skips the PSUM hop. **Measured, it loses**, so the default is `nc_transpose`:

    P=1:  nc_transpose 287.5 us   vs   dma_transpose 514.8 us   (1.79x)
    P=8:  nc_transpose 273.4 us   vs   dma_transpose 300.1 us   (1.10x)

128 small DMA transposes (one per contraction chunk) cost more than 128 Tensor-engine
transposes plus their PSUM copies, and the penalty is worst at P=1 where each DMA moves only
128 x 1 elements -- i.e. per-descriptor overhead dominates, exactly the regime where DMA is a
bad trade. Both paths produce bit-identical results, so this is purely a scheduling choice.

The weight is passed **pre-transposed** as `hc_fn_t` of shape [K, mix_hc]. `hc_fn` is a
constant, so transposing it once on the host is free and it removes 128 transposes from the
steady-state path. Callers hold the transposed copy.

PSUM DISCIPLINE
---------------
Both PSUM tiles are allocated ONCE outside the chunk loop and reused. Allocating a fresh PSUM
tile per iteration is the classic way to blow the allocator and get `NCC_INLA001` -- the same
error code the activation-table ceiling reports, since it is the compiler's general allocator
error. Reuse creates a write-after-read dependency that serialises the loop slightly, which
is the right trade at 128 chunks.

Written against nki 0.6.0.
"""

import nki
import nki.isa as nisa
import nki.language as nl

_PMAX = 128
_K_TILE = 128          # contraction chunk = PE partition limit


@nki.jit
def hc_matmul_head_kernel(x_flat, hc_fn_t, eps, use_dma_transpose=False):
    """`mixes = (x_flat @ hc_fn.T) * rsqrt(mean(x_flat^2) + eps)`.

    Args:
        x_flat:   [P, K]        fp32 HBM -- hc*D wide hidden state.
        hc_fn_t:  [K, mix_hc]   fp32 HBM -- the projection weight, ALREADY TRANSPOSED.
        eps:      variance epsilon.
        use_dma_transpose: transpose x chunks on the DMA engines (default) rather than the
            Tensor engine.

    Returns:
        [P, mix_hc] fp32 `mixes`.
    """
    P, K = x_flat.shape
    K2, mix_hc = hc_fn_t.shape

    assert P <= _PMAX, f"P must be <= {_PMAX}, got {P}"
    assert K2 == K, f"hc_fn_t partition dim {K2} != x_flat free dim {K}"
    assert K % _K_TILE == 0, f"K={K} must be a multiple of {_K_TILE}"
    assert mix_hc <= nl.tile_size.gemm_moving_fmax, (
        f"mix_hc={mix_hc} exceeds moving free-dim limit "
        f"{nl.tile_size.gemm_moving_fmax}"
    )
    assert K * 4 <= nl.tile_size.sbuf_fmax_bytes, (
        f"K={K} fp32 exceeds one SBUF partition ({nl.tile_size.sbuf_fmax_bytes} bytes)"
    )

    out = nl.ndarray((P, mix_hc), dtype=nl.float32, buffer=nl.shared_hbm)

    xt = nl.ndarray((P, K), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=xt, src=x_flat[0:P, 0:K])

    y = head_stages(xt, x_flat, hc_fn_t, P, K, mix_hc, eps, use_dma_transpose)

    nisa.dma_copy(dst=out, src=y)
    return out


def head_stages(xt, x_flat, hc_fn_t, P, K, mix_hc, eps, use_dma_transpose=False):
    """The statistic + projection, on SBUF tiles, so increment 5 can fuse it.

    `xt` is the already-loaded [P, K] tile. `x_flat` is still needed as the HBM source when
    `use_dma_transpose` is set, because that path transposes during the HBM->SBUF copy.

    Returns an SBUF [P, mix_hc] tile holding `mixes`.
    """
    n_chunks = K // _K_TILE

    # ---- the RMS statistic, on data we have to load anyway ----
    # squares AND their sum in ONE Scalar Engine instruction. Only the sum is needed, never
    # the individual squares, which is exactly the condition under which the fused
    # activation+reduce form is the right choice (unlike softmax, where it costs a second
    # exp and measured slower).
    sq = nl.ndarray((P, K), dtype=nl.float32, buffer=nl.sbuf)
    sum_sq = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=sq, data=xt, op=nl.square,
                    reduce_op=nl.add, reduce_res=sum_sq,
                    reduce_cmd=nisa.reduce_cmd.reset_reduce)
    var = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=var, data=sum_sq,
                       op0=nl.multiply, operand0=1.0 / K,
                       op1=nl.add, operand1=eps)
    rs = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=rs, data=var, op=nl.rsqrt)

    # ---- the projection ----
    # Allocated once, reused every chunk: see PSUM DISCIPLINE above.
    acc = nl.ndarray((P, mix_hc), dtype=nl.float32, buffer=nl.psum)
    xcT = nl.ndarray((_K_TILE, P), dtype=nl.float32, buffer=nl.sbuf)
    # BOTH nc_matmul operands must live in SBUF ("moving must be in [sbuf], got shared_hbm"),
    # and hc_fn_t is [K, mix_hc] with K=16384 >> 128 partitions, so the weight cannot be
    # staged as a single tile -- it is streamed one contraction chunk at a time.
    wc = nl.ndarray((_K_TILE, mix_hc), dtype=nl.float32, buffer=nl.sbuf)
    if not use_dma_transpose:
        xcT_psum = nl.ndarray((_K_TILE, P), dtype=nl.float32, buffer=nl.psum)

    for c in nl.sequential_range(n_chunks):
        s = c * _K_TILE
        if use_dma_transpose:
            # transpose during the HBM -> SBUF copy; no Tensor-engine op, no PSUM hop
            nisa.dma_transpose(dst=xcT, src=x_flat[0:P, s:s + _K_TILE])
        else:
            # Tensor-engine transpose lands in PSUM and must be copied to SBUF before it can
            # serve as a matmul operand -- the pattern the model's own kernels use.
            nisa.nc_transpose(dst=xcT_psum, data=xt[0:P, s:s + _K_TILE])
            nisa.tensor_copy(dst=xcT, src=xcT_psum)

        nisa.dma_copy(dst=wc, src=hc_fn_t[s:s + _K_TILE, 0:mix_hc])

        # Repeated nc_matmul into the SAME PSUM tile accumulates -- this is the documented
        # behaviour the model's block-matmul kernels rely on ("psum accumulates across H").
        nisa.nc_matmul(dst=acc, stationary=xcT, moving=wc)

    # ---- scale by the RMS statistic ----
    # rs is a per-partition [P,1] vector, so it rides in a tensor_scalar operand.
    y = nl.ndarray((P, mix_hc), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=y, data=acc, op0=nl.multiply, operand0=rs)
    return y

"""Multi-head batched decode attention for head_dim=256 (Gemma4 SWA layers).

The single-head kernel (nki_decode_attention_hd256) is faster than one SDPA
call, but a real layer has 32 heads. Calling the single-head kernel 32× incurs
32× the eager dispatch overhead (~0.19ms each) which loses to torch's single
batched bmm. This kernel does ALL heads in ONE dispatch — heads are looped
internally so there's a single compiled kernel and a single launch.

Layout (head-major, transposed Q/K to satisfy the 128-partition DMA rule):
  q_t : [NH*HD, 1]   — head h query occupies rows [h*HD : (h+1)*HD]
  k_t : [NH*HD, S]   — head h keys occupy rows  [h*HD : (h+1)*HD]
  v   : [NH*S, HD]   — head h values occupy rows [h*S  : (h+1)*S]
  out : [NH, HD]

GQA is handled by the caller (expand KV heads to NH before calling), matching
the torch reference. head_dim=256 split into two 128 halves; scores tiled over
S in <=512 chunks (matmul moving free dim limit).
"""
import nki
import nki.isa as nisa
import nki.language as nl


@nki.jit
def nki_decode_attention_hd256_mh(
    q_t,        # [NH*HD, 1]
    k_t,        # [NH*HD, S]
    v,          # [NH*S, HD]
    scale_val,  # float
    num_heads,  # int (static)
):
    HD = 256
    HALF = 128
    NH = num_heads
    S = k_t.shape[1]
    SCORE_TILE = 512

    output = nl.ndarray((NH, HD), dtype=q_t.dtype, buffer=nl.shared_hbm)

    for h in nl.affine_range(NH):
        hb = h * HD
        vb = h * S

        q_lo = nl.ndarray((HALF, 1), dtype=nl.float32, buffer=nl.sbuf)
        q_hi = nl.ndarray((HALF, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=q_lo, src=q_t[hb:hb + HALF, 0:1])
        nisa.dma_copy(dst=q_hi, src=q_t[hb + HALF:hb + HD, 0:1])

        k_lo = nl.ndarray((HALF, S), dtype=nl.float32, buffer=nl.sbuf)
        k_hi = nl.ndarray((HALF, S), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=k_lo, src=k_t[hb:hb + HALF, 0:S])
        nisa.dma_copy(dst=k_hi, src=k_t[hb + HALF:hb + HD, 0:S])

        # scores [1,S], tiled over S
        scores = nl.ndarray((1, S), dtype=nl.float32, buffer=nl.sbuf)
        n_tiles = (S + SCORE_TILE - 1) // SCORE_TILE
        for sc in nl.affine_range(n_tiles):
            c0 = sc * SCORE_TILE
            c1 = min(c0 + SCORE_TILE, S)
            csz = c1 - c0
            sp = nl.ndarray((1, csz), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(dst=sp, stationary=q_lo[0:HALF, 0:1], moving=k_lo[0:HALF, c0:c1])
            nisa.nc_matmul(dst=sp, stationary=q_hi[0:HALF, 0:1], moving=k_hi[0:HALF, c0:c1])
            nisa.tensor_copy(dst=scores[0:1, c0:c1], src=sp[0:1, 0:csz])
        nisa.tensor_scalar(dst=scores, data=scores, op0=nl.multiply, operand0=scale_val)

        # softmax
        smax = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=smax, data=scores, op=nl.maximum, axis=(1,))
        nisa.tensor_scalar(dst=scores, data=scores, op0=nl.subtract, operand0=smax)
        nisa.activation(dst=scores, data=scores, op=nl.exp)
        ssum = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=ssum, data=scores, op=nl.add, axis=(1,))
        inv = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.reciprocal(dst=inv, data=ssum)
        nisa.tensor_scalar(dst=scores, data=scores, op0=nl.multiply, operand0=inv)

        # output halves
        out_lo = nl.ndarray((1, HALF), dtype=nl.float32, buffer=nl.sbuf)
        out_hi = nl.ndarray((1, HALF), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(dst=out_lo, value=0.0)
        nisa.memset(dst=out_hi, value=0.0)
        ident = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(dst=ident, value=1.0)

        num_s_tiles = (S + HALF - 1) // HALF
        for st in nl.affine_range(num_s_tiles):
            s0 = st * HALF
            s1 = min(s0 + HALF, S)
            ssz = s1 - s0

            w_slice = nl.ndarray((1, ssz), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=w_slice[0:1, 0:ssz], src=scores[0:1, s0:s1])
            wt_psum = nl.ndarray((ssz, 1), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(dst=wt_psum, stationary=w_slice[0:1, 0:ssz], moving=ident[0:1, 0:1])
            w_t = nl.ndarray((ssz, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=w_t[0:ssz, 0:1], src=wt_psum[0:ssz, 0:1])

            v_lo = nl.ndarray((ssz, HALF), dtype=nl.float32, buffer=nl.sbuf)
            v_hi = nl.ndarray((ssz, HALF), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=v_lo, src=v[vb + s0:vb + s1, 0:HALF])
            nisa.dma_copy(dst=v_hi, src=v[vb + s0:vb + s1, HALF:HD])

            ov_lo = nl.ndarray((1, HALF), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(dst=ov_lo, stationary=w_t[0:ssz, 0:1], moving=v_lo[0:ssz, 0:HALF])
            ov_hi = nl.ndarray((1, HALF), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(dst=ov_hi, stationary=w_t[0:ssz, 0:1], moving=v_hi[0:ssz, 0:HALF])

            ov_lo_s = nl.ndarray((1, HALF), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=ov_lo_s[0:1, 0:HALF], src=ov_lo[0:1, 0:HALF])
            nisa.tensor_tensor(dst=out_lo, data1=out_lo, data2=ov_lo_s, op=nl.add)
            ov_hi_s = nl.ndarray((1, HALF), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=ov_hi_s[0:1, 0:HALF], src=ov_hi[0:1, 0:HALF])
            nisa.tensor_tensor(dst=out_hi, data1=out_hi, data2=ov_hi_s, op=nl.add)

        nisa.dma_copy(dst=output[h:h + 1, 0:HALF], src=out_lo)
        nisa.dma_copy(dst=output[h:h + 1, HALF:HD], src=out_hi)

    return output

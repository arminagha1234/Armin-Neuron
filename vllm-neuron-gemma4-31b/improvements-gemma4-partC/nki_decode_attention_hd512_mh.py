"""Multi-head batched decode attention for head_dim=512 (Gemma4 Global layers).

The global layers (11/60) use head_dim=512 with 4 KV heads. Same batched
all-heads-in-one-dispatch design as nki_decode_attention_hd256_mh, but head_dim
splits into FOUR 128-dim quarters (4-way split-K) instead of two halves.

Calling the single-head hd512 kernel 32× would pay 32× the eager dispatch
overhead and lose to torch's batched bmm — so all heads are looped INSIDE one
kernel (single compiled NEFF, single launch).

Layout (head-major, transposed Q/K to satisfy the 128-partition DMA rule):
  q_t : [NH*HD, 1]   — head h query at rows [h*HD : (h+1)*HD]
  k_t : [NH*HD, S]   — head h keys at rows  [h*HD : (h+1)*HD]
  v   : [NH*S, HD]   — head h values at rows [h*S  : (h+1)*S]
  out : [NH, HD]

GQA expansion (4 kv -> 32 q heads) is done by the caller, matching the torch
reference. Scores tiled over S in <=512 chunks (matmul moving free-dim limit).
"""
import nki
import nki.isa as nisa
import nki.language as nl


@nki.jit
def nki_decode_attention_hd512_mh(
    q_t,        # [NH*HD, 1]
    k_t,        # [NH*HD, S]
    v,          # [NH*S, HD]
    scale_val,  # float
    num_heads,  # int (static)
):
    HD = 512
    Q = 128            # quarter size
    NQ = HD // Q       # 4 quarters
    NH = num_heads
    S = k_t.shape[1]
    SCORE_TILE = 512

    output = nl.ndarray((NH, HD), dtype=q_t.dtype, buffer=nl.shared_hbm)

    for h in nl.affine_range(NH):
        hb = h * HD
        vb = h * S

        # --- scores [1,S] = sum over 4 quarters of Q_q^T @ K_q, S-tiled ---
        scores = nl.ndarray((1, S), dtype=nl.float32, buffer=nl.sbuf)
        n_score_tiles = (S + SCORE_TILE - 1) // SCORE_TILE
        for sc in nl.affine_range(n_score_tiles):
            c0 = sc * SCORE_TILE
            c1 = min(c0 + SCORE_TILE, S)
            csz = c1 - c0
            sp = nl.ndarray((1, csz), dtype=nl.float32, buffer=nl.psum)
            for qi in nl.affine_range(NQ):
                d0 = hb + qi * Q
                q_q = nl.ndarray((Q, 1), dtype=nl.float32, buffer=nl.sbuf)
                k_q = nl.ndarray((Q, csz), dtype=nl.float32, buffer=nl.sbuf)
                nisa.dma_copy(dst=q_q, src=q_t[d0:d0 + Q, 0:1])
                nisa.dma_copy(dst=k_q, src=k_t[d0:d0 + Q, c0:c1])
                nisa.nc_matmul(dst=sp, stationary=q_q[0:Q, 0:1], moving=k_q[0:Q, 0:csz])
            nisa.tensor_copy(dst=scores[0:1, c0:c1], src=sp[0:1, 0:csz])
        nisa.tensor_scalar(dst=scores, data=scores, op0=nl.multiply, operand0=scale_val)

        # --- softmax over S ---
        smax = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=smax, data=scores, op=nl.maximum, axis=(1,))
        nisa.tensor_scalar(dst=scores, data=scores, op0=nl.subtract, operand0=smax)
        nisa.activation(dst=scores, data=scores, op=nl.exp)
        ssum = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=ssum, data=scores, op=nl.add, axis=(1,))
        inv = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.reciprocal(dst=inv, data=ssum)
        nisa.tensor_scalar(dst=scores, data=scores, op0=nl.multiply, operand0=inv)

        # --- output: weights @ V, 4 output quarters, accumulate over S-tiles ---
        out_q = []
        for qi in nl.static_range(NQ):
            oq = nl.ndarray((1, Q), dtype=nl.float32, buffer=nl.sbuf)
            nisa.memset(dst=oq, value=0.0)
            out_q.append(oq)

        ident = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(dst=ident, value=1.0)

        num_s_tiles = (S + Q - 1) // Q
        for st in nl.affine_range(num_s_tiles):
            s0 = st * Q
            s1 = min(s0 + Q, S)
            ssz = s1 - s0

            w_slice = nl.ndarray((1, ssz), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=w_slice[0:1, 0:ssz], src=scores[0:1, s0:s1])
            w_t_psum = nl.ndarray((ssz, 1), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(dst=w_t_psum, stationary=w_slice[0:1, 0:ssz], moving=ident[0:1, 0:1])
            w_t = nl.ndarray((ssz, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=w_t[0:ssz, 0:1], src=w_t_psum[0:ssz, 0:1])

            for qi in nl.static_range(NQ):
                d0 = qi * Q
                v_q = nl.ndarray((ssz, Q), dtype=nl.float32, buffer=nl.sbuf)
                nisa.dma_copy(dst=v_q, src=v[vb + s0:vb + s1, d0:d0 + Q])
                ov = nl.ndarray((1, Q), dtype=nl.float32, buffer=nl.psum)
                nisa.nc_matmul(dst=ov, stationary=w_t[0:ssz, 0:1], moving=v_q[0:ssz, 0:Q])
                ov_s = nl.ndarray((1, Q), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_copy(dst=ov_s[0:1, 0:Q], src=ov[0:1, 0:Q])
                nisa.tensor_tensor(dst=out_q[qi], data1=out_q[qi], data2=ov_s, op=nl.add)

        for qi in nl.static_range(NQ):
            d0 = qi * Q
            nisa.dma_copy(dst=output[h:h + 1, d0:d0 + Q], src=out_q[qi])

    return output

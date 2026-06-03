"""NKI decode attention for head_dim=512 (Gemma4 Global layers, 11/60 layers).

Same split-K pattern as hd256 but splits head_dim=512 into FOUR 128-dim quarters.
Q/K are passed head_dim-major (transposed) so the 128-partition DMA constraint holds.

Validated pattern (from hd256 bring-up): identity-matmul transpose, no auto-broadcast.
"""
import nki
import nki.isa as nisa
import nki.language as nl


@nki.jit
def nki_decode_attention_hd512(
    q_tensor,     # [512, 1] — query, head_dim-major (transposed)
    k_cache_t,    # [512, S] — keys, head_dim-major (transposed)
    v_cache,      # [S, 512] — values, seq-major
    scale_val,    # float scalar
):
    """Decode attention for head_dim=512 via 4-way split-K."""
    HD = 512
    Q = 128            # quarter size
    NQ = HD // Q       # 4 quarters
    S = k_cache_t.shape[1]

    output = nl.ndarray((1, HD), dtype=q_tensor.dtype, buffer=nl.shared_hbm)

    # --- Scores: sum over 4 quarters of Q_q^T @ K_q -> [1, S], tiled over S ---
    # Hardware caps the matmul moving free dim and PSUM bank at 512, so tile S.
    SCORE_TILE = 512
    scores = nl.ndarray((1, S), dtype=nl.float32, buffer=nl.sbuf)
    num_score_tiles = (S + SCORE_TILE - 1) // SCORE_TILE
    for sc in nl.affine_range(num_score_tiles):
        c0 = sc * SCORE_TILE
        c1 = min(c0 + SCORE_TILE, S)
        csz = c1 - c0
        score_psum = nl.ndarray((1, csz), dtype=nl.float32, buffer=nl.psum)
        for qi in nl.affine_range(NQ):
            d0 = qi * Q
            q_q = nl.ndarray((Q, 1), dtype=nl.float32, buffer=nl.sbuf)
            k_q = nl.ndarray((Q, csz), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=q_q, src=q_tensor[d0:d0 + Q, 0:1])
            nisa.dma_copy(dst=k_q, src=k_cache_t[d0:d0 + Q, c0:c1])
            # accumulate into the same PSUM (hardware accumulation)
            nisa.nc_matmul(dst=score_psum, stationary=q_q[0:Q, 0:1], moving=k_q[0:Q, 0:csz])
        nisa.tensor_copy(dst=scores[0:1, c0:c1], src=score_psum[0:1, 0:csz])

    nisa.tensor_scalar(dst=scores, data=scores, op0=nl.multiply, operand0=scale_val)

    # --- Softmax over S ---
    smax = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=smax, data=scores, op=nl.maximum, axis=(1,))
    nisa.tensor_scalar(dst=scores, data=scores, op0=nl.subtract, operand0=smax)
    nisa.activation(dst=scores, data=scores, op=nl.exp)
    ssum = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=ssum, data=scores, op=nl.add, axis=(1,))
    inv_sum = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.reciprocal(dst=inv_sum, data=ssum)
    nisa.tensor_scalar(dst=scores, data=scores, op0=nl.multiply, operand0=inv_sum)

    # --- Output: weights @ V, accumulate over S-tiles, 4 output quarters ---
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

        # transpose weights slice [1,ssz] -> [ssz,1] via identity matmul
        w_slice = nl.ndarray((1, ssz), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=w_slice[0:1, 0:ssz], src=scores[0:1, s0:s1])
        w_t_psum = nl.ndarray((ssz, 1), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=w_t_psum, stationary=w_slice[0:1, 0:ssz], moving=ident[0:1, 0:1])
        w_t = nl.ndarray((ssz, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=w_t[0:ssz, 0:1], src=w_t_psum[0:ssz, 0:1])

        for qi in nl.static_range(NQ):
            d0 = qi * Q
            v_q = nl.ndarray((ssz, Q), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=v_q, src=v_cache[s0:s1, d0:d0 + Q])
            ov = nl.ndarray((1, Q), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(dst=ov, stationary=w_t[0:ssz, 0:1], moving=v_q[0:ssz, 0:Q])
            ov_s = nl.ndarray((1, Q), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=ov_s[0:1, 0:Q], src=ov[0:1, 0:Q])
            nisa.tensor_tensor(dst=out_q[qi], data1=out_q[qi], data2=ov_s, op=nl.add)

    # --- Store 4 quarters ---
    for qi in nl.static_range(NQ):
        d0 = qi * Q
        nisa.dma_copy(dst=output[0:1, d0:d0 + Q], src=out_q[qi])

    return output

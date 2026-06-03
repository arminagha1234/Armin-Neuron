"""NKI decode attention kernel for head_dim=256 (Gemma4 SWA layers).

Standard NKI decode megakernel only supports head_dim≤128. This kernel handles
head_dim=256 by splitting into two 128-dim halves.

Layout strategy (to satisfy the 128-partition DMA alignment constraint):
  - K/V are loaded with head_dim-half (128) as the PARTITION dim and S as the FREE dim.
  - Q is loaded with head_dim-half (128) as the partition dim, 1 token as free.
  - score = Q^T @ K  via nc_matmul (stationary=Q[128,1], moving=K[128,S]) -> [1, S]

This replaces the PyTorch SDPA fallback for decode that #1552 flagged as the
bottleneck (~350ms/token).

Constraint handled: S (cached seq len) does NOT need to be divisible by 128
because S is the FREE dimension here, not the partition dimension. head_dim-half
= 128 is always the partition dim (exactly pmax).
"""
import nki
import nki.isa as nisa
import nki.language as nl


@nki.jit
def nki_decode_attention_hd256(
    q_tensor,     # [256, 1] — single query token, head_dim-major (transposed)
    k_cache_t,    # [256, S] — keys, head_dim-major (transposed): k_cache_t[d, s]
    v_cache,      # [S, 256] — values, seq-major
    scale_val,    # float scalar
):
    """Decode attention for head_dim=256, head_dim-major Q/K layout.

    Args:
        q_tensor:  [256, 1] query (head_dim as partition, transposed)
        k_cache_t: [256, S] keys transposed (head_dim as partition)
        v_cache:   [S, 256] values (seq as outer dim) — S must be ≤512 per call,
                   or tiled by the caller
        scale_val: attention scale (1.0 for Gemma4)

    Returns:
        output: [1, 256] attention output
    """
    HD = 256
    HALF = 128
    S = k_cache_t.shape[1]

    output = nl.ndarray((1, HD), dtype=q_tensor.dtype, buffer=nl.shared_hbm)

    # --- Load Q halves: [128, 1] each (head_dim-half as partition) ---
    q_lo = nl.ndarray((HALF, 1), dtype=nl.float32, buffer=nl.sbuf)
    q_hi = nl.ndarray((HALF, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=q_lo, src=q_tensor[0:HALF, 0:1])
    nisa.dma_copy(dst=q_hi, src=q_tensor[HALF:HD, 0:1])

    # --- Load K halves: [128, S] each (head_dim-half as partition, S free) ---
    k_lo = nl.ndarray((HALF, S), dtype=nl.float32, buffer=nl.sbuf)
    k_hi = nl.ndarray((HALF, S), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=k_lo, src=k_cache_t[0:HALF, 0:S])
    nisa.dma_copy(dst=k_hi, src=k_cache_t[HALF:HD, 0:S])

    # --- Scores: Q^T @ K via split-K, tiled over S in <=512 chunks ---
    # nc_matmul: stationary=[K,M], moving=[K,N] -> result=[M,N]
    # Hardware caps the moving free dim (N) and PSUM bank at 512, so tile S.
    SCORE_TILE = 512
    scores = nl.ndarray((1, S), dtype=nl.float32, buffer=nl.sbuf)
    num_score_tiles = (S + SCORE_TILE - 1) // SCORE_TILE
    for sc in nl.affine_range(num_score_tiles):
        c0 = sc * SCORE_TILE
        c1 = min(c0 + SCORE_TILE, S)
        csz = c1 - c0
        score_psum = nl.ndarray((1, csz), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=score_psum, stationary=q_lo[0:HALF, 0:1], moving=k_lo[0:HALF, c0:c1])
        nisa.nc_matmul(dst=score_psum, stationary=q_hi[0:HALF, 0:1], moving=k_hi[0:HALF, c0:c1])
        nisa.tensor_copy(dst=scores[0:1, c0:c1], src=score_psum[0:1, 0:csz])

    # apply scale
    nisa.tensor_scalar(dst=scores, data=scores, op0=nl.multiply, operand0=scale_val)

    # --- Softmax over S (free dim) ---
    smax = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=smax, data=scores, op=nl.maximum, axis=(1,))
    # scores = exp(scores - max)
    nisa.tensor_scalar(dst=scores, data=scores, op0=nl.subtract, operand0=smax)
    nisa.activation(dst=scores, data=scores, op=nl.exp)
    # sum
    ssum = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=ssum, data=scores, op=nl.add, axis=(1,))
    inv_sum = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.reciprocal(dst=inv_sum, data=ssum)
    # normalize weights
    nisa.tensor_scalar(dst=scores, data=scores, op0=nl.multiply, operand0=inv_sum)

    # --- Output: weights @ V ---
    # We need out[1,256] = weights[1,S] @ V[S,256].
    # nc_matmul: stationary=[K,M], moving=[K,N] -> [M,N], with K as partition.
    # Put S as the contraction (partition) dim: need weights[S,1] and V[S,256].
    # weights is [1,S] -> transpose to [S,1]. V is [S,256] already seq-major.
    # But S must be ≤128 for the partition dim of nc_matmul. Tile S by 128.
    HALF_OUT = HALF
    out_lo = nl.ndarray((1, HALF_OUT), dtype=nl.float32, buffer=nl.sbuf)
    out_hi = nl.ndarray((1, HALF_OUT), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=out_lo, value=0.0)
    nisa.memset(dst=out_hi, value=0.0)

    num_s_tiles = (S + HALF - 1) // HALF
    for st in nl.affine_range(num_s_tiles):
        s0 = st * HALF
        s1 = min(s0 + HALF, S)
        ssz = s1 - s0

        # weights tile transposed to [ssz, 1] (S as partition) via identity matmul.
        # nc_transpose is limited to <=[32,32]; identity-matmul transpose handles any size.
        # scores slice [1, ssz] @ identity... instead: use nc_matmul with the slice as
        # moving and a [1,1] identity stationary to land weights as [ssz,1].
        w_slice = nl.ndarray((1, ssz), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=w_slice[0:1, 0:ssz], src=scores[0:1, s0:s1])
        ident = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(dst=ident, value=1.0)
        w_t_psum = nl.ndarray((ssz, 1), dtype=nl.float32, buffer=nl.psum)
        # stationary=w_slice[1,ssz] (K=1, M=ssz), moving=ident[1,1] (K=1, N=1) -> [ssz,1]
        nisa.nc_matmul(dst=w_t_psum, stationary=w_slice[0:1, 0:ssz], moving=ident[0:1, 0:1])
        w_t = nl.ndarray((ssz, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=w_t[0:ssz, 0:1], src=w_t_psum[0:ssz, 0:1])

        # V tile [ssz, 256] -> halves [ssz, 128]
        v_lo = nl.ndarray((ssz, HALF), dtype=nl.float32, buffer=nl.sbuf)
        v_hi = nl.ndarray((ssz, HALF), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=v_lo, src=v_cache[s0:s1, 0:HALF])
        nisa.dma_copy(dst=v_hi, src=v_cache[s0:s1, HALF:HD])

        # out += weights^T @ V : stationary=w_t[ssz,1], moving=v_lo[ssz,128] -> [1,128]
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

    # --- Store output [1,256] ---
    nisa.dma_copy(dst=output[0:1, 0:HALF], src=out_lo)
    nisa.dma_copy(dst=output[0:1, HALF:HD], src=out_hi)

    return output

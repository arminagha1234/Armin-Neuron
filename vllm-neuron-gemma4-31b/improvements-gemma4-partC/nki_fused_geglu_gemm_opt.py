"""Throughput-optimized fully-fused GeGLU up-projection.

act[M,N] = gelu_tanh(x @ Wg) * (x @ Wu), one kernel, hoisted lhsT reuse.

Key throughput tricks (vs the first nki_fused_geglu_gemm):
  - x (lhsT) K-tiles for an M-block loaded ONCE, reused across all N-tiles AND
    across both the gate and up matmuls (x is shared by both projections).
  - gate & up accumulate K in PSUM (two PSUM banks).
  - gelu(gate)*up fused in SBUF; intermediate never hits HBM.

Contract:
  x_t : [K, M]  (hidden on partition, M = tokens, transposed)
  wg  : [K, N]  gate weight [hidden, inter]
  wu  : [K, N]  up   weight [hidden, inter]
  out : [M, N]
K mult of 128, M mult of 128, N mult of 512 (caller pads).
"""
import nki
import nki.isa as nisa
import nki.language as nl


@nki.jit
def nki_fused_geglu_gemm_opt(x_t, wg, wu):
    K, M = x_t.shape
    _, N = wg.shape
    TILE_M = 128
    TILE_K = 128
    TILE_N = 512

    out = nl.ndarray((M, N), dtype=x_t.dtype, buffer=nl.shared_hbm)

    n_m = (M + TILE_M - 1) // TILE_M
    n_k = (K + TILE_K - 1) // TILE_K
    n_n = (N + TILE_N - 1) // TILE_N

    for m in nl.affine_range(n_m):
        # hoist x: load all K-tiles of this M-block once, reuse for gate AND up
        x_tiles = []
        for k in nl.affine_range(n_k):
            xt = nl.ndarray((TILE_K, TILE_M), dtype=x_t.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=xt, src=x_t[k * TILE_K:(k + 1) * TILE_K,
                                          m * TILE_M:(m + 1) * TILE_M])
            x_tiles.append(xt)

        for n in nl.affine_range(n_n):
            gate_psum = nl.ndarray((TILE_M, TILE_N), dtype=nl.float32, buffer=nl.psum)
            up_psum = nl.ndarray((TILE_M, TILE_N), dtype=nl.float32, buffer=nl.psum)
            for k in nl.affine_range(n_k):
                wg_t = nl.ndarray((TILE_K, TILE_N), dtype=wg.dtype, buffer=nl.sbuf)
                wu_t = nl.ndarray((TILE_K, TILE_N), dtype=wu.dtype, buffer=nl.sbuf)
                nisa.dma_copy(dst=wg_t, src=wg[k * TILE_K:(k + 1) * TILE_K,
                                               n * TILE_N:(n + 1) * TILE_N])
                nisa.dma_copy(dst=wu_t, src=wu[k * TILE_K:(k + 1) * TILE_K,
                                               n * TILE_N:(n + 1) * TILE_N])
                nisa.nc_matmul(dst=gate_psum, stationary=x_tiles[k], moving=wg_t)
                nisa.nc_matmul(dst=up_psum, stationary=x_tiles[k], moving=wu_t)

            g_sb = nl.ndarray((TILE_M, TILE_N), dtype=nl.float32, buffer=nl.sbuf)
            u_sb = nl.ndarray((TILE_M, TILE_N), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=g_sb, src=gate_psum)
            nisa.tensor_copy(dst=u_sb, src=up_psum)
            nisa.activation(dst=g_sb, data=g_sb, op=nl.gelu_apprx_tanh)
            nisa.tensor_tensor(dst=g_sb, data1=g_sb, data2=u_sb, op=nl.multiply)

            res = nl.ndarray((TILE_M, TILE_N), dtype=out.dtype, buffer=nl.sbuf)
            nisa.tensor_copy(dst=res, src=g_sb)
            nisa.dma_copy(dst=out[m * TILE_M:(m + 1) * TILE_M,
                                  n * TILE_N:(n + 1) * TILE_N], src=res)

    return out

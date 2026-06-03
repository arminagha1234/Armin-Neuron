"""Throughput-optimized tiled NKI GEMM (hoist-load pattern).

Adapted from the official NKI matmul tutorial (nki_matmul_hoist_load_):
  - lhsT tiles for an M-block are loaded ONCE and reused across all N-tiles
  - K accumulation happens in PSUM (no SBUF add-tree)
  - rhs tiles loaded per (n,k)

My earlier naive nki_gemm reloaded lhsT inside the N loop, which is why it lost
~2.2x to torch at compute-bound (large-M) shapes. This version hoists the lhsT
load and reuses it, which is the standard throughput trick.

Contract: C[M,N] = A[M,K] @ B[K,N], with A passed transposed as lhsT [K, M].
  lhsT : [K, M]   (K on partition)
  rhs  : [K, N]
  out  : [M, N]
K must be a multiple of 128, M of 128, N of 512 (caller pads). For Gemma4
hidden=5376 = 42*128 ✓, inter=21504 = 42*512 ✓.
"""
import nki
import nki.isa as nisa
import nki.language as nl


@nki.jit
def nki_gemm_opt(lhsT, rhs):
    K, M = lhsT.shape
    K_, N = rhs.shape
    TILE_M = 128
    TILE_K = 128
    TILE_N = 512

    out = nl.ndarray((M, N), dtype=lhsT.dtype, buffer=nl.shared_hbm)

    n_m = (M + TILE_M - 1) // TILE_M
    n_k = (K + TILE_K - 1) // TILE_K
    n_n = (N + TILE_N - 1) // TILE_N

    for m in nl.affine_range(n_m):
        # hoist: load all K-tiles of this M-column of lhsT once, reuse across N
        lhsT_tiles = []
        for k in nl.affine_range(n_k):
            lt = nl.ndarray((TILE_K, TILE_M), dtype=lhsT.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=lt, src=lhsT[k * TILE_K:(k + 1) * TILE_K,
                                           m * TILE_M:(m + 1) * TILE_M])
            lhsT_tiles.append(lt)

        for n in nl.affine_range(n_n):
            res_psum = nl.ndarray((TILE_M, TILE_N), dtype=nl.float32, buffer=nl.psum)
            for k in nl.affine_range(n_k):
                rt = nl.ndarray((TILE_K, TILE_N), dtype=rhs.dtype, buffer=nl.sbuf)
                nisa.dma_copy(dst=rt, src=rhs[k * TILE_K:(k + 1) * TILE_K,
                                              n * TILE_N:(n + 1) * TILE_N])
                nisa.nc_matmul(dst=res_psum, stationary=lhsT_tiles[k], moving=rt)

            res_sb = nl.ndarray((TILE_M, TILE_N), dtype=out.dtype, buffer=nl.sbuf)
            nisa.tensor_copy(dst=res_sb, src=res_psum)
            nisa.dma_copy(dst=out[m * TILE_M:(m + 1) * TILE_M,
                                  n * TILE_N:(n + 1) * TILE_N], src=res_sb)

    return out

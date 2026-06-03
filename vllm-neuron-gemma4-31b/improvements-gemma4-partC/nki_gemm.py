"""General tiled NKI GEMM: C[M,N] = A[M,K] @ B[K,N].

nc_matmul contract: stationary=[K,M], moving=[K,N] -> psum=[M,N], with the
contraction dim K on the partition axis. Hardware limits:
  - K (partition)        <= 128  -> tile K by 128, accumulate in PSUM
  - M (stationary free)  <= 128  -> tile M by 128
  - N (moving free)      <= 512  -> tile N by 512

A is passed PRE-TRANSPOSED as A_t [K, M] so we can slice [k_tile, m_tile]
straight into the stationary operand (the caller transposes; for decode the
activation [1,K] -> [K,1] is trivial). B is [K, N] (the projection weight
stored as W.T, i.e. [in_features, out_features]).

This is the baseline: can hand-written NKI match the framework matmul on the
eager build? Decode projections are memory-bound (read the whole weight once),
so the ceiling is the weight-read bandwidth either way.
"""
import nki
import nki.isa as nisa
import nki.language as nl


@nki.jit
def nki_gemm(
    a_t,   # [K, M] — A transposed (contraction dim on partition)
    b,     # [K, N] — B (contraction dim on partition)
):
    K, M = a_t.shape
    _, N = b.shape
    KT, MT, NT = 128, 128, 512

    out = nl.ndarray((M, N), dtype=a_t.dtype, buffer=nl.shared_hbm)

    n_k = (K + KT - 1) // KT
    n_m = (M + MT - 1) // MT
    n_n = (N + NT - 1) // NT

    for mi in nl.affine_range(n_m):
        m0 = mi * MT
        m1 = min(m0 + MT, M)
        msz = m1 - m0
        for ni in nl.affine_range(n_n):
            nn0 = ni * NT
            nn1 = min(nn0 + NT, N)
            nsz = nn1 - nn0

            psum = nl.ndarray((msz, nsz), dtype=nl.float32, buffer=nl.psum)
            for ki in nl.affine_range(n_k):
                k0 = ki * KT
                k1 = min(k0 + KT, K)
                ksz = k1 - k0
                a_tile = nl.ndarray((ksz, msz), dtype=nl.float32, buffer=nl.sbuf)
                b_tile = nl.ndarray((ksz, nsz), dtype=nl.float32, buffer=nl.sbuf)
                nisa.dma_copy(dst=a_tile, src=a_t[k0:k1, m0:m1])
                nisa.dma_copy(dst=b_tile, src=b[k0:k1, nn0:nn1])
                # accumulate over K tiles into the same PSUM
                nisa.nc_matmul(dst=psum, stationary=a_tile[0:ksz, 0:msz],
                               moving=b_tile[0:ksz, 0:nsz])

            c_tile = nl.ndarray((msz, nsz), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=c_tile[0:msz, 0:nsz], src=psum[0:msz, 0:nsz])
            nisa.dma_copy(dst=out[m0:m1, nn0:nn1], src=c_tile)

    return out

"""Fully-fused GeGLU up-projection for Gemma4 MLP.

Computes, in ONE kernel:  act[M,I] = gelu_tanh(x @ Wg) * (x @ Wu)

Standard path is 4 ops with 3 large HBM round-trips of the [M,I] intermediate:
  gate = x @ Wg            -> write gate [M,I]
  up   = x @ Wu            -> write up   [M,I]
  g    = gelu_tanh(gate)   -> read gate, write g
  act  = g * up            -> read g, read up, write act

Fused path: for each N-tile of the intermediate, do BOTH matmuls into PSUM,
apply gelu to the gate result in SBUF, multiply by the up result, write the
[M, n_tile] product once. The intermediate never hits HBM.

nc_matmul: stationary=[K,M], moving=[K,N] -> [M,N], K on partition.
  x_t : [K, M]   (activation transposed, K=hidden on partition)
  Wg  : [K, N]   gate weight [hidden, inter]
  Wu  : [K, N]   up   weight [hidden, inter]
Tiles: K by 128 (accumulate), M by 128, N by 512.
"""
import nki
import nki.isa as nisa
import nki.language as nl


@nki.jit
def nki_fused_geglu_gemm(
    x_t,   # [K, M] — activation transposed
    wg,    # [K, N] — gate weight
    wu,    # [K, N] — up weight
):
    K, M = x_t.shape
    _, N = wg.shape
    KT, MT, NT = 128, 128, 512

    out = nl.ndarray((M, N), dtype=x_t.dtype, buffer=nl.shared_hbm)

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

            gate_psum = nl.ndarray((msz, nsz), dtype=nl.float32, buffer=nl.psum)
            up_psum = nl.ndarray((msz, nsz), dtype=nl.float32, buffer=nl.psum)
            for ki in nl.affine_range(n_k):
                k0 = ki * KT
                k1 = min(k0 + KT, K)
                ksz = k1 - k0
                x_tile = nl.ndarray((ksz, msz), dtype=nl.float32, buffer=nl.sbuf)
                wg_tile = nl.ndarray((ksz, nsz), dtype=nl.float32, buffer=nl.sbuf)
                wu_tile = nl.ndarray((ksz, nsz), dtype=nl.float32, buffer=nl.sbuf)
                nisa.dma_copy(dst=x_tile, src=x_t[k0:k1, m0:m1])
                nisa.dma_copy(dst=wg_tile, src=wg[k0:k1, nn0:nn1])
                nisa.dma_copy(dst=wu_tile, src=wu[k0:k1, nn0:nn1])
                nisa.nc_matmul(dst=gate_psum, stationary=x_tile[0:ksz, 0:msz],
                               moving=wg_tile[0:ksz, 0:nsz])
                nisa.nc_matmul(dst=up_psum, stationary=x_tile[0:ksz, 0:msz],
                               moving=wu_tile[0:ksz, 0:nsz])

            g_sb = nl.ndarray((msz, nsz), dtype=nl.float32, buffer=nl.sbuf)
            u_sb = nl.ndarray((msz, nsz), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=g_sb[0:msz, 0:nsz], src=gate_psum[0:msz, 0:nsz])
            nisa.tensor_copy(dst=u_sb[0:msz, 0:nsz], src=up_psum[0:msz, 0:nsz])
            # gelu(gate) * up, fused in SBUF
            nisa.activation(dst=g_sb, data=g_sb, op=nl.gelu_apprx_tanh)
            nisa.tensor_tensor(dst=g_sb, data1=g_sb, data2=u_sb, op=nl.multiply)
            nisa.dma_copy(dst=out[m0:m1, nn0:nn1], src=g_sb)

    return out

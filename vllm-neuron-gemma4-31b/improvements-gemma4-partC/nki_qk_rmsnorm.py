"""NKI QK-RMSNorm kernel for Gemma4 attention.

Gemma4 applies RMSNorm to the query and key projections (per head, over
head_dim) before RoPE:
    q = q_norm(q)   # RMSNorm over head_dim, weight shape [head_dim]
    k = k_norm(k)

This is a plain RMSNorm (no residual) over the free dimension. The naive
path reads q/k from HBM, norms, writes back — this kernel does it in a
single fused pass and can be reused for both q_norm and k_norm.

  Input:  x      [T, D]  (D = head_dim, 256 for Gemma4; T = num_heads*tokens)
          weight [1, D]  (RMSNorm weight)
  Output: out    [T, D] = RMSNorm(x) * weight

Gemma4 head_dim = 256, eps = 1e-6.
"""
import nki
import nki.isa as nisa
import nki.language as nl


@nki.jit
def nki_qk_rmsnorm(
    x,        # [T, D]
    weight,   # [1, D]
    eps_val,  # float
):
    """RMSNorm over the free dim D, with per-feature weight, single HBM pass."""
    T, D = x.shape
    TILE_P = 128
    inv_d = 1.0 / D

    output = nl.ndarray((T, D), dtype=x.dtype, buffer=nl.shared_hbm)

    num_t_tiles = (T + TILE_P - 1) // TILE_P
    for t_idx in nl.affine_range(num_t_tiles):
        t0 = t_idx * TILE_P
        t1 = min(t0 + TILE_P, T)
        t_sz = t1 - t0

        x_tile = nl.ndarray((t_sz, D), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=x_tile, src=x[t0:t1, 0:D])

        # replicate [1,D] weight across t_sz partitions (broadcast DMA, stride 0)
        w_rep = nl.ndarray((t_sz, D), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=w_rep, src=weight.ap(pattern=[[0, t_sz], [1, D]]))

        # variance reduce + rsqrt(sum/D + eps) fused into ONE activation call.
        # (nkilib rmsnorm_tkg technique: activation(rsqrt, scale=1/D, bias=eps)
        # replaces the 3 ops mul-1/D, add-eps, rsqrt.)
        x_sq = nl.ndarray((t_sz, D), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=x_sq, data1=x_tile, data2=x_tile, op=nl.multiply)
        var = nl.ndarray((t_sz, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=var, data=x_sq, op=nl.add, axis=(1,))
        rstd = nl.ndarray((t_sz, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=rstd, data=var, op=nl.rsqrt, scale=inv_d, bias=eps_val)

        # x * rstd (per-partition scalar), then * weight (per-feature)
        nisa.tensor_scalar(dst=x_tile, data=x_tile, op0=nl.multiply, operand0=rstd)
        nisa.tensor_tensor(dst=x_tile, data1=x_tile, data2=w_rep, op=nl.multiply)

        nisa.dma_copy(dst=output[t0:t1, 0:D], src=x_tile)

    return output

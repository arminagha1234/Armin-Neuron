"""NKI fused RMSNorm + residual add kernel for Gemma4.

Gemma4 has 4 norms per layer × 60 layers = 240 norm operations. The standard
path reads hidden_states from HBM, norms, writes back, then reads again for the
residual add. This kernel fuses: RMSNorm(x) + residual_add in one HBM pass.

Saves 2 HBM round-trips per fusion point × 2 points per layer × 60 layers = 240 DMAs eliminated.

Pattern:
  Input: residual [T, H], module_output [T, H], norm_weight [H]
  Output: new_hidden = residual + RMSNorm(module_output, weight, eps)
"""
import nki
import nki.isa as nisa
import nki.language as nl


@nki.jit
def nki_fused_rmsnorm_residual(
    residual,       # [T, H] — residual stream
    module_output,  # [T, H] — output from attention/MLP
    norm_weight,    # [H] — RMSNorm weight (1D)
    eps_val,        # float — RMSNorm epsilon (1e-6)
):
    """Fused: residual + RMSNorm(module_output) in single HBM pass.

    Standard path (3 HBM ops):
      1. Read module_output → norm → write normed
      2. Read residual + read normed → add → write result

    Fused path (2 HBM ops):
      1. Read module_output + residual + weight → norm + add → write result
    """
    T, H = residual.shape
    TILE_P = 128

    output = nl.ndarray((T, H), dtype=residual.dtype, buffer=nl.shared_hbm)

    num_t_tiles = (T + TILE_P - 1) // TILE_P

    for t_idx in nl.affine_range(num_t_tiles):
        t_start = t_idx * TILE_P
        t_end = min(t_start + TILE_P, T)
        t_sz = t_end - t_start

        # Load module_output tile [t_sz, H]
        x_tile = nl.ndarray((t_sz, H), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=x_tile, src=module_output[t_start:t_end, 0:H])

        # Load residual tile [t_sz, H]
        res_tile = nl.ndarray((t_sz, H), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=res_tile, src=residual[t_start:t_end, 0:H])

        # Replicate the [1,H] weight across t_sz partitions via broadcast DMA
        # (partition stride 0 reads the same row into every partition).
        w_rep = nl.ndarray((t_sz, H), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=w_rep, src=norm_weight.ap(pattern=[[0, t_sz], [1, H]]))

        # Compute RMSNorm: variance = mean(x^2), rsqrt, scale
        # Step 1: x^2
        x_sq = nl.ndarray((t_sz, H), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=x_sq, data1=x_tile, data2=x_tile, op=nl.multiply)

        # Step 2: sum(x^2) along H dim -> [t_sz, 1]
        x_sq_sum = nl.ndarray((t_sz, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=x_sq_sum, data=x_sq, op=nl.add, axis=(1,))

        # Step 3: rsqrt(sum/H + eps) in ONE fused activation call.
        # nisa.activation(op=rsqrt, scale, bias) computes rsqrt(scale*data + bias).
        # (Technique from nkilib core/subkernels/rmsnorm_tkg.py — fuses the
        # mean-divide + eps-add + rsqrt that were 3 separate tensor_scalar ops.)
        inv_h = 1.0 / H
        rsqrt_val = nl.ndarray((t_sz, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=rsqrt_val, data=x_sq_sum, op=nl.rsqrt, scale=inv_h, bias=eps_val)

        # Step 4: x_normed = x * rsqrt (per-partition scalar broadcast across H)
        # tensor_scalar with a [t_sz,1] operand broadcasts across the free dim
        nisa.tensor_scalar(dst=x_tile, data=x_tile, op0=nl.multiply, operand0=rsqrt_val)

        # Step 5: apply weight: x_normed * weight (replicated across partitions)
        nisa.tensor_tensor(dst=x_tile, data1=x_tile, data2=w_rep, op=nl.multiply)

        # Step 6: residual add: output = residual + normed
        nisa.tensor_tensor(dst=res_tile, data1=res_tile, data2=x_tile, op=nl.add)

        # Store result
        nisa.dma_copy(dst=output[t_start:t_end, 0:H], src=res_tile)

    return output

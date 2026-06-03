"""NKI fused GeGLU activation gate for Gemma4 MLP.

Gemma4 MLP is GeGLU:  down_proj( gelu(gate_proj(x)) * up_proj(x) )
The gate_proj and up_proj matmuls are large GEMMs best left to the
matmul engine, but the elementwise gate  `gelu(gate) * up`  is a
separate op that, in the naive path, reads `gate` and `up` from HBM,
applies gelu, multiplies, and writes the product back to HBM before
down_proj reads it again.

This kernel fuses the gate activation + multiply into one HBM pass:
  Input:  gate [T, I], up [T, I]   (I = intermediate_size, e.g. 30720)
  Output: act  [T, I] = gelu_tanh(gate) * up

Gemma4 uses the tanh approximation of GeLU (gelu_pytorch_tanh), so we
use `nl.gelu_apprx_tanh`.

Tiling: the intermediate dim I (e.g. 30720) is huge, so a [128, I] fp32
tile (~15MB) plus its `up` partner overflows the ~28MB SBUF. We tile BOTH
the partition dim (T, by 128) and the free dim (I, by TILE_F) so each
working set is small.

Saves 1 HBM round-trip of the [T, I] intermediate per layer × 60 layers.
"""
import nki
import nki.isa as nisa
import nki.language as nl


@nki.jit
def nki_geglu_mlp(
    gate,   # [T, I] — gate_proj output
    up,     # [T, I] — up_proj output
):
    """Fused GeGLU activation: gelu_tanh(gate) * up in a single HBM pass."""
    T, I = gate.shape
    TILE_P = 128
    TILE_F = 2048  # free-dim tile so [128, TILE_F] fp32 working set stays small

    output = nl.ndarray((T, I), dtype=gate.dtype, buffer=nl.shared_hbm)

    num_t_tiles = (T + TILE_P - 1) // TILE_P
    num_f_tiles = (I + TILE_F - 1) // TILE_F
    for t_idx in nl.affine_range(num_t_tiles):
        t0 = t_idx * TILE_P
        t1 = min(t0 + TILE_P, T)

        for f_idx in nl.affine_range(num_f_tiles):
            f0 = f_idx * TILE_F
            f1 = min(f0 + TILE_F, I)

            g_tile = nl.ndarray((t1 - t0, f1 - f0), dtype=nl.float32, buffer=nl.sbuf)
            u_tile = nl.ndarray((t1 - t0, f1 - f0), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=g_tile, src=gate[t0:t1, f0:f1])
            nisa.dma_copy(dst=u_tile, src=up[t0:t1, f0:f1])

            # gelu(gate) (tanh approximation, matching Gemma4's gelu_pytorch_tanh)
            nisa.activation(dst=g_tile, data=g_tile, op=nl.gelu_apprx_tanh)
            # multiply by up
            nisa.tensor_tensor(dst=g_tile, data1=g_tile, data2=u_tile, op=nl.multiply)

            nisa.dma_copy(dst=output[t0:t1, f0:f1], src=g_tile)

    return output

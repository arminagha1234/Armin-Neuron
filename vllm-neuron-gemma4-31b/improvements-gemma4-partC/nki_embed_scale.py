"""NKI embedding-scale kernel for Gemma4.

Gemma models scale token embeddings by sqrt(hidden_size) right after the
embedding lookup:
    hidden = embed(tokens) * sqrt(hidden_size)

For Gemma4 31B, hidden_size = 5376, so the scale is sqrt(5376) ≈ 73.32.
This is a trivial elementwise scale, fused here into a single HBM pass so
it can be chained with the embedding DMA without a separate framework op.

  Input:  embeds [T, H]
  Output: out    [T, H] = embeds * scale
"""
import nki
import nki.isa as nisa
import nki.language as nl


@nki.jit
def nki_embed_scale(
    embeds,     # [T, H]
    scale_val,  # float scalar (sqrt(hidden_size))
):
    """Scale embeddings by a constant in a single HBM pass."""
    T, H = embeds.shape
    TILE_P = 128

    output = nl.ndarray((T, H), dtype=embeds.dtype, buffer=nl.shared_hbm)

    num_t_tiles = (T + TILE_P - 1) // TILE_P
    for t_idx in nl.affine_range(num_t_tiles):
        t0 = t_idx * TILE_P
        t1 = min(t0 + TILE_P, T)
        t_sz = t1 - t0

        e_tile = nl.ndarray((t_sz, H), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=e_tile, src=embeds[t0:t1, 0:H])
        nisa.tensor_scalar(dst=e_tile, data=e_tile, op0=nl.multiply, operand0=scale_val)
        nisa.dma_copy(dst=output[t0:t1, 0:H], src=e_tile)

    return output

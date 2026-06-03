"""NKI fused logit soft-capping for Gemma4.

Gemma4 applies soft-capping to attention logits and final logits:
    capped = cap * tanh(x / cap)

This bounds the values to (-cap, +cap) smoothly. In the naive path this
is three separate elementwise passes over a large tensor (divide, tanh,
multiply), each reading and writing HBM.

This kernel fuses divide + tanh + multiply into one HBM pass.

  Input:  x   [T, V]   (V = vocab_size for final logits ~262144,
                        or seq for attention logits)
  Output: out [T, V] = cap * tanh(x / cap)

Gemma4 caps:  attn_logit_softcapping = 50.0,  final_logit_softcapping = 30.0
"""
import nki
import nki.isa as nisa
import nki.language as nl


@nki.jit
def nki_logit_softcap(
    x,        # [T, V]
    cap_val,  # float scalar (e.g. 30.0 for final logits, 50.0 for attn)
):
    """Fused soft-cap: cap * tanh(x / cap) in a single HBM pass."""
    T, V = x.shape
    TILE_P = 128
    inv_cap = 1.0 / cap_val

    output = nl.ndarray((T, V), dtype=x.dtype, buffer=nl.shared_hbm)

    num_t_tiles = (T + TILE_P - 1) // TILE_P
    for t_idx in nl.affine_range(num_t_tiles):
        t0 = t_idx * TILE_P
        t1 = min(t0 + TILE_P, T)
        t_sz = t1 - t0

        x_tile = nl.ndarray((t_sz, V), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=x_tile, src=x[t0:t1, 0:V])

        # x / cap
        nisa.tensor_scalar(dst=x_tile, data=x_tile, op0=nl.multiply, operand0=inv_cap)
        # tanh
        nisa.activation(dst=x_tile, data=x_tile, op=nl.tanh)
        # * cap
        nisa.tensor_scalar(dst=x_tile, data=x_tile, op0=nl.multiply, operand0=cap_val)

        nisa.dma_copy(dst=output[t0:t1, 0:V], src=x_tile)

    return output

"""Flash non-causal joint self-attention NKI kernel for Mochi-1 (trn2).

OPTIMIZED VARIANT (v2). Same math as the validated online-softmax kernel and
the materialised numpy reference (`flash_attn_ref.py`), but restructured for
the Mochi geometry (S ~6616, D=128, P=6) where the original streaming kernel
was overhead-bound (~44 ms, 16k inner iterations).

Key changes vs the original streaming kernel:
  1. K^T and V are loaded + transposed ONCE PER PLANE and kept resident in
     SBUF, instead of being reloaded/re-transposed for every one of the ~52
     q-tiles (was the dominant DMA + transpose cost).
  2. The per-key-column bias is broadcast across the 128 query partitions ONCE
     PER PLANE, instead of once per (q-tile, k-tile) inner iteration.
  3. MM1 (QK^T) uses a 512-wide moving operand (kT tiled by 512) so the tensor
     engine runs near its 128x512 moving limit instead of wasting ~75% at 128.
  4. Because the full key row (Sk<=~10k) fits in SBUF, softmax is done in a
     single pass per q-tile (one reduce-max + one fused exp/row-sum over the
     whole row) rather than an online rescale loop -- this removes all the
     per-tile correction/rescale vector traffic. Numerically identical to the
     materialised reference (exact full-row softmax).

Correctness is validated by test_flash_attn.py (5 cases, bf16 atol=rtol=1e-2,
min per-plane cosine >= 0.999) against the CPU reference.

Exact operation (non-causal):
    scores[p,q,k] = (sum_d Q[p,q,d] * K[p,k,d]) * scale + key_bias[p,k]
    probs         = softmax(scores, axis=k)
    out[p,q,d]    = sum_k probs[p,q,k] * V[p,k,d]

Q,K,V: (P, S, D) bf16, D=head_dim<=128. key_bias: (P, Sk) additive per-key
bias (0 keep / -10000 masked), broadcast across query rows. scale: 1/sqrt(D).

License: Apache-2.0.
"""
from __future__ import annotations

import nki
import nki.language as nl
import nki.isa as nisa


_P = 128
_Q_TILE = 128
_K_TILE = 128          # K/V contraction tile for MM2 (partition dim cap)
_MM1_MOVING = 512      # MM1 QK^T moving-operand width (trn2 128x512 cap)

_NEG_SENTINEL = -1.0e30


def kernel_assert(condition: bool, error_text: str) -> None:
    assert condition, f"[INTERNAL_ERROR] [NCC_INKI016] Kernel validation exception: {error_text}"


def div_ceil(n: int, d: int) -> int:
    return (n + d - 1) // d


def _stream_shuffle_broadcast(src: nl.ndarray, dst: nl.ndarray) -> None:
    """Broadcast ``src[0:1, :]`` onto every partition of ``dst`` (both SBUF)."""
    dst_npar = dst.shape[0]
    kernel_assert(
        len(src.shape) == 2 and len(dst.shape) == 2,
        "stream_shuffle_broadcast: src and dst must be 2D",
    )
    kernel_assert(
        src.shape[1] == dst.shape[1],
        "stream_shuffle_broadcast: matching free dim required",
    )
    shuffle_mask = [0] * 32
    for i in range((dst_npar + 31) // 32):
        cur_npar = min(32, dst_npar - i * 32)
        nisa.nc_stream_shuffle(
            src=src[0:1, :],
            dst=dst[i * 32 : i * 32 + cur_npar, 0 : dst.shape[1]],
            shuffle_mask=shuffle_mask,
        )


@nki.jit
def flash_attention_kernel(
    q: nl.ndarray,          # (P, Sq, D) bf16
    k: nl.ndarray,          # (P, Sk, D) bf16
    v: nl.ndarray,          # (P, Sk, D) bf16
    key_bias: nl.ndarray,   # (P, Sk) additive per-key-column bias
    scale: float,
) -> nl.ndarray:
    P, Sq, D = q.shape
    Pk, Sk, Dk = k.shape
    Pv, Skv, Dv = v.shape
    Pb, Skb = key_bias.shape

    kernel_assert(P == Pk == Pv == Pb, "plane count mismatch across q/k/v/bias")
    kernel_assert(D == Dk == Dv, "head_dim mismatch across q/k/v")
    kernel_assert(Sk == Skv == Skb, "key length mismatch across k/v/bias")
    kernel_assert(D <= _P, f"head_dim {D} must be <= {_P}")

    n_q_tiles = div_ceil(Sq, _Q_TILE)
    n_k_tiles = div_ceil(Sk, _K_TILE)          # 128-tiling for K^T build + MM2
    n_mm1_chunks = div_ceil(Sk, _MM1_MOVING)   # 512-chunking for MM1 moving

    out = nl.ndarray((P, Sq, D), dtype=q.dtype, buffer=nl.shared_hbm)

    for p in nl.affine_range(P):
        # ============================================================
        # Per-plane resident tensors: K^T [D, Sk] and V [k, tile*D].
        # Loaded / transposed ONCE and reused across all q-tiles.
        # ============================================================
        kT = nl.ndarray((D, n_k_tiles * _K_TILE), dtype=k.dtype, buffer=nl.sbuf)
        v_sb = nl.ndarray((_K_TILE, n_k_tiles * D), dtype=v.dtype, buffer=nl.sbuf)

        for kt in nl.static_range(n_k_tiles):
            k_start = kt * _K_TILE
            k_size = min(_K_TILE, Sk - k_start)

            k_rows = nl.ndarray((_K_TILE, D), dtype=k.dtype, buffer=nl.sbuf)
            nisa.dma_copy(
                dst=k_rows[0:k_size, 0:D],
                src=k[p, nl.ds(k_start, k_size), 0:D],
            )
            kT_psum = nl.ndarray((D, _K_TILE), dtype=k.dtype, buffer=nl.psum)
            nisa.nc_transpose(
                dst=kT_psum[0:D, 0:k_size],
                data=k_rows[0:k_size, 0:D],
                engine=nisa.engine.tensor,
            )
            nisa.tensor_copy(
                dst=kT[0:D, nl.ds(k_start, k_size)],
                src=kT_psum[0:D, 0:k_size],
            )

            # V tile: (k_size, D) laid out at free offset kt*D. k on partition.
            nisa.dma_copy(
                dst=v_sb[0:k_size, nl.ds(kt * D, D)],
                src=v[p, nl.ds(k_start, k_size), 0:D],
            )

        # ------------------------------------------------------------
        # Broadcast the per-key bias across all 128 query partitions ONCE.
        # ------------------------------------------------------------
        bias_row = nl.ndarray((1, n_k_tiles * _K_TILE), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=bias_row[0:1, 0:Sk],
            src=key_bias[nl.ds(p, 1), 0:Sk],
        )
        bias_bcast = nl.ndarray((_P, n_k_tiles * _K_TILE), dtype=nl.float32, buffer=nl.sbuf)
        _stream_shuffle_broadcast(
            src=bias_row[0:1, 0:Sk],
            dst=bias_bcast[0:_P, 0:Sk],
        )

        # ============================================================
        # Per q-tile: full-row scores -> single-pass softmax -> PV.
        # ============================================================
        for qt in nl.affine_range(n_q_tiles):
            q_start = qt * _Q_TILE
            q_size = min(_Q_TILE, Sq - q_start)

            # Load Q tile (q_size, D), transpose to Q^T (D, q_size), scale.
            q_rows = nl.ndarray((_Q_TILE, D), dtype=q.dtype, buffer=nl.sbuf)
            nisa.dma_copy(
                dst=q_rows[0:q_size, 0:D],
                src=q[p, nl.ds(q_start, q_size), 0:D],
            )
            qT_psum = nl.ndarray((D, _Q_TILE), dtype=q.dtype, buffer=nl.psum)
            nisa.nc_transpose(
                dst=qT_psum[0:D, 0:q_size],
                data=q_rows[0:q_size, 0:D],
                engine=nisa.engine.tensor,
            )
            qT = nl.ndarray((D, _Q_TILE), dtype=q.dtype, buffer=nl.sbuf)
            # Fold softmax scale into Q^T (bf16); MM1 product lands scaled in PSUM.
            nisa.tensor_scalar(
                dst=qT[0:D, 0:q_size],
                data=qT_psum[0:D, 0:q_size],
                op0=nl.multiply,
                operand0=scale,
            )

            # --- MM1: scores[q, Sk] = (Q@K^T)*scale + bias, 512-wide moving. ---
            scores = nl.ndarray((_Q_TILE, n_k_tiles * _K_TILE), dtype=nl.float32, buffer=nl.sbuf)
            for cc in nl.static_range(n_mm1_chunks):
                c_start = cc * _MM1_MOVING
                c_size = min(_MM1_MOVING, Sk - c_start)
                sc_psum = nl.ndarray((_Q_TILE, _MM1_MOVING), dtype=nl.float32, buffer=nl.psum)
                nisa.nc_matmul(
                    dst=sc_psum[0:q_size, 0:c_size],
                    stationary=qT[0:D, 0:q_size],
                    moving=kT[0:D, nl.ds(c_start, c_size)],
                )
                # scores = psum + bias (scale already folded into Q^T).
                nisa.tensor_tensor(
                    dst=scores[0:q_size, nl.ds(c_start, c_size)],
                    data1=sc_psum[0:q_size, 0:c_size],
                    data2=bias_bcast[0:q_size, nl.ds(c_start, c_size)],
                    op=nl.add,
                )

            # --- Single-pass softmax over the full key row. ---
            m_row = nl.ndarray((_Q_TILE, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_reduce(
                dst=m_row[0:q_size, 0:1],
                data=scores[0:q_size, 0:Sk],
                op=nl.maximum,
                axis=1,
            )
            neg_m = nl.ndarray((_Q_TILE, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(
                dst=neg_m[0:q_size, 0:1],
                data=m_row[0:q_size, 0:1],
                op0=nl.multiply,
                operand0=-1.0,
            )
            probs = nl.ndarray((_Q_TILE, n_k_tiles * _K_TILE), dtype=q.dtype, buffer=nl.sbuf)
            l_row = nl.ndarray((_Q_TILE, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(
                dst=probs[0:q_size, 0:Sk],
                data=scores[0:q_size, 0:Sk],
                op=nl.exp,
                bias=neg_m[0:q_size, 0:1],
                reduce_op=nl.add,
                reduce_res=l_row[0:q_size, 0:1],
                reduce_cmd=nisa.reduce_cmd.reset_reduce,
            )
            recip = nl.ndarray((_Q_TILE, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.reciprocal(dst=recip[0:q_size, 0:1], data=l_row[0:q_size, 0:1])

            # --- MM2: out[q, d] = probs @ V. Contraction over k, 128-tiled. ---
            out_psum = nl.ndarray((_Q_TILE, D), dtype=nl.float32, buffer=nl.psum)
            for kt in nl.static_range(n_k_tiles):
                k_start = kt * _K_TILE
                k_size = min(_K_TILE, Sk - k_start)
                pT_psum = nl.ndarray((_K_TILE, _Q_TILE), dtype=q.dtype, buffer=nl.psum)
                nisa.nc_transpose(
                    dst=pT_psum[0:k_size, 0:q_size],
                    data=probs[0:q_size, nl.ds(k_start, k_size)],
                    engine=nisa.engine.tensor,
                )
                pT = nl.ndarray((_K_TILE, _Q_TILE), dtype=q.dtype, buffer=nl.sbuf)
                nisa.tensor_copy(dst=pT[0:k_size, 0:q_size], src=pT_psum[0:k_size, 0:q_size])
                nisa.nc_matmul(
                    dst=out_psum[0:q_size, 0:D],
                    stationary=pT[0:k_size, 0:q_size],
                    moving=v_sb[0:k_size, nl.ds(kt * D, D)],
                    accumulate=(kt > 0),
                )

            # --- Finalise: out = out_psum / l_row. ---
            out_sb = nl.ndarray((_Q_TILE, D), dtype=q.dtype, buffer=nl.sbuf)
            nisa.tensor_scalar(
                dst=out_sb[0:q_size, 0:D],
                data=out_psum[0:q_size, 0:D],
                op0=nl.multiply,
                operand0=recip[0:q_size, 0:1],
            )
            nisa.dma_copy(
                dst=out[p, nl.ds(q_start, q_size), 0:D],
                src=out_sb[0:q_size, 0:D],
            )

    return out

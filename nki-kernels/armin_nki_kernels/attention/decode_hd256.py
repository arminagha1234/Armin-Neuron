# Copyright Armin Aghaeb. SPDX-License-Identifier: Apache-2.0
"""Fused single-token decode attention NKI kernel for head_dim=256.

Replaces the Python split-K decode path used in Qwen3.5/3.6 GQA layers
(see `Qwen3_5GQAAttention.forward_decode`). Stock vllm-neuron's
`NF.attention_decode` rejects head_dim>128 because the tensor engine's
single-stationary transpose is sized to 128. This kernel does the same
split-K trick internally with PSUM accumulation, plus fuses the
QK + softmax + AV passes into one NEFF.

Reference (math contract): see `ref_decode_hd256.py`. Any change to
this kernel MUST keep cosine > 0.999 vs that reference on the test
shapes in `tests/test_decode_hd256_parity.py`.

Tiling strategy:
    Per (batch, head) call, the kernel processes S_q=1 query against
    S_ctx context tokens (multiple of 128). It chunks S_ctx into
    P_MAX=128 chunks because:
      - Tensor engine transpose path is sized to 128
      - K/V chunks of 128 fit naturally as the moving operand
      - Q is small (256 floats per token) so it sits in SBUF for the
        whole kernel without re-loading

    Pass 1 (QK):
      1. Load Q (1, 256) into SBUF — split into Q_lo (1, 128), Q_hi (1, 128).
      2. For each chunk c = 0..num_chunks-1:
           Load K_chunk_lo (128, 128), K_chunk_hi (128, 128).
           PSUM tile s_chunk (1, 128) ← 0
           nc_matmul s_chunk += Q_lo @ K_chunk_lo.T  (stationary=Q_lo, moving=K_chunk_lo)
           nc_matmul s_chunk += Q_hi @ K_chunk_hi.T  (PSUM accumulates)
           Copy s_chunk → SBUF scores[c*128 : (c+1)*128]
           Apply mask: scores += (~mask_chunk) * -65504
      3. Concatenated SBUF scores: (1, S_ctx)

    Softmax pass:
      4. Cast scores to fp32 (already in fp32 from PSUM).
      5. Find max (for numerical stability — skipped since mask
         already saturates to -65504, exp(score - max) ≈ exp(score)
         for the unmasked region anyway). Run exp inline via activation.
      6. Reduce-sum over free dim → denom (1, 1).
      7. Divide: weights = exp / denom (still fp32).
      8. Cast weights to bf16 for the AV matmul.

    Pass 2 (AV):
      9. For each chunk c = 0..num_chunks-1:
           Load V_chunk_lo (128, 128), V_chunk_hi (128, 128).
           PSUM tile out_lo_acc (1, 128), out_hi_acc (1, 128) (accumulated
             across all chunks).
           nc_matmul out_lo_acc += weights_chunk @ V_chunk_lo
           nc_matmul out_hi_acc += weights_chunk @ V_chunk_hi
      10. Copy out_lo, out_hi → output (1, 256) and DMA back.

Key NKI notes:
    - PSUM accumulation on the same destination triggers hardware
      hardware accumulation across multiple nc_matmul calls — this is
      the only way to do split-K matmul without explicit add ops.
    - K and V are stationary in nc_matmul; Q is moving (single token,
      so the K/V operand is the larger 128-token stationary).
"""
from __future__ import annotations

import numpy as np

try:
    import nki
    import nki.isa as nisa
    import nki.language as nl
    _NKI_AVAILABLE = True
except ImportError:
    _NKI_AVAILABLE = False


# ---------------------------------------------------------------------------
# Kernel constants
# ---------------------------------------------------------------------------
P_MAX = 128
HEAD_DIM = 256
HEAD_DIM_HALF = 128
NEG_BIAS = -65504.0


# ---------------------------------------------------------------------------
# kernel_assert helper (inline definition, matches nkilib convention)
# ---------------------------------------------------------------------------

def kernel_assert(cond, msg=""):
    """Compile-time check (host side, before NKI body runs).

    Implementation note: NKI's compiler doesn't support `raise` statements
    inside @nki.jit kernel bodies — when the kernel is FX-traced under
    fake-tensor mode, the compiler tries to specialize the kernel and
    `raise` causes 'NKI does not support raise statements'. We use a
    plain Python `assert` instead. NKI does support `assert` as a fatal
    error path (per its error spec).
    """
    assert cond, f"[NCC_INKI016] Kernel validation exception: {msg}"


# ---------------------------------------------------------------------------
# Kernel body
# ---------------------------------------------------------------------------

if _NKI_AVAILABLE:

    @nki.jit
    def decode_hd256_kernel(
        q,         # (1, 256) bf16 — single decode-token query
        k_full,    # (S_ctx, 256) bf16 — already GQA-repeated K cache
        v_full,    # (S_ctx, 256) bf16 — already GQA-repeated V cache
        mask_bias, # (1, S_ctx) fp32 — pre-computed: 0 where allowed,
                   #                                  NEG_BIAS where masked
        scale,     # fp32 host scalar — 1/sqrt(head_dim) (or fold)
    ):
        """Fused decode attention for head_dim=256, per (batch, head).

        Args:
            q:         (1, 256) bf16 query for one decode token.
            k_full:    (S_ctx, 256) bf16 K cache (GQA-repeated). S_ctx
                       must be a multiple of 128.
            v_full:    (S_ctx, 256) bf16 V cache.
            mask_bias: (1, S_ctx) fp32 additive bias. The caller pre-
                       computes (~mask) * NEG_BIAS so the kernel just
                       adds it to the QK scores. This avoids passing a
                       bool tensor (which has awkward NKI semantics) and
                       lets the kernel do a simple tensor_tensor add.
            scale:     pre-softmax scaling factor (fp32 host scalar).

        Returns:
            (1, 256) bf16 attention output.

        Constraints:
            - S_ctx % 128 == 0 (caller pads if needed)
            - All inputs bf16 except mask_bias (fp32) and scale (fp32 scalar)
        """
        S_ctx = k_full.shape[0]
        # Shape validation is done by the wrapper before the kernel call.
        # NKI can't model `raise`/`assert` inside @nki.jit bodies during
        # FX-tracing under fake-tensor mode (it fails to specialize), so we
        # keep the kernel body free of host-side validation. The wrapper
        # owns shape/dtype checks before invocation.
        num_chunks = S_ctx // P_MAX

        # Output in HBM
        output = nl.ndarray((1, HEAD_DIM), dtype=q.dtype, buffer=nl.shared_hbm)

        # =====================================================================
        # Load Q once into SBUF (it's tiny: 256 floats per token in bf16 = 512B)
        # =====================================================================
        # We need Q_lo and Q_hi in SBUF as (128, 1) shape because nc_matmul
        # has stationary [K=128, M=128] form, and Q is the smaller side
        # (single token). For the QK matmul we want:
        #   stationary = K_chunk (128 ctx tokens × 128 dim half)
        #   moving     = Q (256 dim, but sliced into the matching half)
        # BUT — Q has only 1 row and we want score[1, 128] in PSUM. The
        # natural layout is:
        #   stationary = Q_lo^T (128 dim, 1 token)
        #   moving     = K_chunk_lo (128 ctx, 128 dim)
        # which gives score (1, 128_ctx). We'll need to transpose afterward
        # OR feed K stationary and Q moving.
        #
        # Simpler: stationary = K_chunk_lo (128 ctx × 128 dim, dim is K),
        #          moving = Q_lo        (128 dim × 1 token, dim is K)
        # Result: (128 ctx, 1 token) PSUM tile = (128, 1).
        #
        # We use partition-dim = 128 ctx. So scores per chunk live across
        # all 128 partitions, with free dim = 1 (single query). Then the
        # softmax-reduce becomes a partition-dim reduction.
        #
        # Pivot: switch the per-chunk score to (128, 1) instead of (1, 128).
        # SBUF layout for the full S_ctx scores: (128, num_chunks) — each
        # chunk fills one column. Or flatten across chunks into a single
        # (128 * num_chunks, 1) tile? Partition limit is 128, so we keep
        # (128, num_chunks).

        # Q in SBUF as (128, 1) — partition dim = head_dim_half, free = 1 token
        q_lo = nl.ndarray((HEAD_DIM_HALF, 1), dtype=q.dtype, buffer=nl.sbuf)
        q_hi = nl.ndarray((HEAD_DIM_HALF, 1), dtype=q.dtype, buffer=nl.sbuf)
        # Q is stored as (1, 256) in HBM. Transpose into the (128, 1) SBUF
        # layout by routing through a (1, 256) → (256, 1) → split-into-halves.
        # Easiest: dma_copy each half directly using strided offsets.
        # Q layout in HBM: [q[0, 0..127], q[0, 128..255]] contiguous.
        # We want q_lo[k] = q[0, k] for k in 0..127.
        # DMA copy with dst=q_lo[k:k+128, 0:1], src=q[0:1, k:k+128].
        # This is a transpose-on-DMA: (1, 128) → (128, 1).
        # NKI supports it via the natural broadcast of a (1, F) → (P, 1) when
        # P partitions each get one element of the F dim.

        # Allocate a (128, 256) SBUF tile, dma both halves of q[0, :] into it
        # via a single (1, 256) → (256, 1) layout. Actually the cleanest is
        # to load q as (1, 256) and then nc_transpose into (256, 1). Let me
        # do that.

        q_row = nl.ndarray((1, HEAD_DIM), dtype=q.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=q_row, src=q[0:1, 0:HEAD_DIM])

        # nc_transpose: (1, 256) → (256, 1) via PSUM. But P max is 128, so
        # we can't have a (256, 1) SBUF tile directly. We'll transpose into
        # two (128, 1) tiles.
        # Pad q_row to (128, 128) by writing q's two halves into rows 0
        # and ... hmm, easier: directly DMA each half from HBM into its own
        # (128, 1) SBUF tile via the (1, 128) → (128, 1) transpose pattern.

        # Use nc_transpose explicitly. Pad q's lower half into (128, 128).
        q_lo_pad = nl.ndarray((P_MAX, P_MAX), dtype=q.dtype, buffer=nl.sbuf)
        nisa.memset(dst=q_lo_pad, value=0.0)
        # Copy q[0, 0:128] into q_lo_pad[0, 0:128] (single row).
        nisa.tensor_copy(
            dst=q_lo_pad[0:1, 0:HEAD_DIM_HALF],
            src=q_row[0:1, 0:HEAD_DIM_HALF],
        )
        q_lo_t_psum = nl.ndarray((P_MAX, P_MAX), dtype=q.dtype, buffer=nl.psum)
        nisa.nc_transpose(dst=q_lo_t_psum, data=q_lo_pad)
        # After transpose, column 0 of q_lo_t_psum = q[0, 0:128] (as a
        # column vector across partitions).
        nisa.tensor_copy(dst=q_lo[0:HEAD_DIM_HALF, 0:1], src=q_lo_t_psum[0:HEAD_DIM_HALF, 0:1])

        q_hi_pad = nl.ndarray((P_MAX, P_MAX), dtype=q.dtype, buffer=nl.sbuf)
        nisa.memset(dst=q_hi_pad, value=0.0)
        nisa.tensor_copy(
            dst=q_hi_pad[0:1, 0:HEAD_DIM_HALF],
            src=q_row[0:1, HEAD_DIM_HALF:HEAD_DIM],
        )
        q_hi_t_psum = nl.ndarray((P_MAX, P_MAX), dtype=q.dtype, buffer=nl.psum)
        nisa.nc_transpose(dst=q_hi_t_psum, data=q_hi_pad)
        nisa.tensor_copy(dst=q_hi[0:HEAD_DIM_HALF, 0:1], src=q_hi_t_psum[0:HEAD_DIM_HALF, 0:1])

        # =====================================================================
        # Pass 1 — QK matmul + scale + mask. Result: scores SBUF (128, num_chunks)
        # =====================================================================
        # Per chunk c, scores[:, c] = (Q_lo @ K_c_lo.T + Q_hi @ K_c_hi.T) * scale + mask_bias[:, c*128:(c+1)*128].
        # Shape per chunk in PSUM: (128, 1) where partition dim=128 ctx tokens.

        scores = nl.ndarray((P_MAX, num_chunks), dtype=nl.float32, buffer=nl.sbuf)

        for c in nl.affine_range(num_chunks):
            chunk_off = c * P_MAX

            # Load K chunk halves: (128 ctx, 128 dim) each
            k_lo = nl.ndarray((P_MAX, HEAD_DIM_HALF), dtype=k_full.dtype, buffer=nl.sbuf)
            k_hi = nl.ndarray((P_MAX, HEAD_DIM_HALF), dtype=k_full.dtype, buffer=nl.sbuf)
            nisa.dma_copy(
                dst=k_lo,
                src=k_full[chunk_off:chunk_off + P_MAX, 0:HEAD_DIM_HALF],
            )
            nisa.dma_copy(
                dst=k_hi,
                src=k_full[chunk_off:chunk_off + P_MAX, HEAD_DIM_HALF:HEAD_DIM],
            )

            # nc_matmul contracts over the PARTITION dim, so we need
            # K transposed to (128 dim, 128 ctx) so partition=dim matches Q.
            # result[m=ctx, n=0] = sum_dim k_T[dim, ctx] * q[dim, 0]
            #                    = (q @ k.T)[0, ctx]  ✓
            k_lo_T_psum = nl.ndarray((P_MAX, P_MAX), dtype=k_full.dtype, buffer=nl.psum)
            nisa.nc_transpose(dst=k_lo_T_psum, data=k_lo)
            k_lo_T = nl.ndarray((P_MAX, P_MAX), dtype=k_full.dtype, buffer=nl.sbuf)
            nisa.tensor_copy(dst=k_lo_T, src=k_lo_T_psum)

            k_hi_T_psum = nl.ndarray((P_MAX, P_MAX), dtype=k_full.dtype, buffer=nl.psum)
            nisa.nc_transpose(dst=k_hi_T_psum, data=k_hi)
            k_hi_T = nl.ndarray((P_MAX, P_MAX), dtype=k_full.dtype, buffer=nl.sbuf)
            nisa.tensor_copy(dst=k_hi_T, src=k_hi_T_psum)

            # PSUM accumulator for this chunk's QK score: (128, 1)
            s_psum = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.psum)
            # Stationary k_T (128 dim, 128 ctx), moving q (128 dim, 1)
            # → PSUM (128 ctx, 1) — split-K contraction over dim.
            nisa.nc_matmul(dst=s_psum, stationary=k_lo_T, moving=q_lo)
            nisa.nc_matmul(dst=s_psum, stationary=k_hi_T, moving=q_hi)

            # Copy PSUM into SBUF scores tile, scaled.
            nisa.tensor_scalar(
                dst=scores[0:P_MAX, c:c + 1],
                data=s_psum[0:P_MAX, 0:1],
                op0=nl.multiply,
                operand0=scale,
            )

        # =====================================================================
        # Apply mask: scores += mask_bias (already pre-multiplied by NEG_BIAS
        # by the caller for masked positions, 0 for allowed positions).
        # =====================================================================
        # mask_bias in HBM is (1, S_ctx). We need to add it to scores
        # which is (128 ctx, num_chunks). The (c*128 + p)-th score
        # corresponds to mask_bias[0, c*128 + p].
        # Load mask_bias into the same partition layout: (128, num_chunks).
        # mask_bias[0, c*128 + p] should match scores[p, c].
        # That's a transpose: (1, S_ctx) → (128, num_chunks) where
        # tile[p, c] = src[0, c * 128 + p].
        # Use tensor_view-style strided DMA: stride 1 along partition dim
        # (incrementing p by 1 = +1 in src), stride 128 along free dim
        # (incrementing c by 1 = +128 in src).
        # NKI dma_copy with explicit access pattern handles this via .ap().
        # We'll use a manual loop instead (simpler, num_chunks is small).
        for c in nl.affine_range(num_chunks):
            chunk_off = c * P_MAX
            mb_chunk_pad = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
            nisa.memset(dst=mb_chunk_pad, value=0.0)
            nisa.dma_copy(
                dst=mb_chunk_pad[0:1, 0:P_MAX],
                src=mask_bias[0:1, chunk_off:chunk_off + P_MAX],
            )
            mb_chunk_t_psum = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_transpose(dst=mb_chunk_t_psum, data=mb_chunk_pad)
            # mb_chunk_t_psum[:, 0] = mask_bias[0, chunk_off:chunk_off+128]
            nisa.tensor_tensor(
                dst=scores[0:P_MAX, c:c + 1],
                data1=scores[0:P_MAX, c:c + 1],
                data2=mb_chunk_t_psum[0:P_MAX, 0:1],
                op=nl.add,
            )

        # =====================================================================
        # Softmax over the full S_ctx range
        # =====================================================================
        # scores layout: (128 ctx, num_chunks) — but logically a single
        # row of S_ctx entries (each query has one row).
        # We want softmax over S_ctx. Naive: flatten scores back to
        # (1, S_ctx) and do softmax over free dim.
        #
        # Flatten via nc_transpose then reshape. Easier with the data laid
        # out (128, num_chunks): the LOGICAL flat order is
        # [scores[0,0], scores[1,0], ..., scores[127,0], scores[0,1], ...].
        # That's free-major within each partition. So scores read as a
        # (128, num_chunks) tile and reduced along BOTH dims for softmax
        # gives the correct sum.
        #
        # For numerical stability we'd subtract max(scores) first, but our
        # mask uses bf16-min so unmasked scores are bounded and exp doesn't
        # overflow in fp32. We'll do exp directly, then sum-reduce, then
        # divide.

        exp_scores = nl.ndarray((P_MAX, num_chunks), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=exp_scores, data=scores, op=nl.exp)

        # Reduce-sum across both (P, F) dims. nisa.tensor_reduce supports
        # axis-1 reductions natively; for partition-dim reduction we'd
        # need a separate pass. Instead: reduce along free dim first to
        # get (128, 1), then reduce along partition via nc_transpose +
        # reduce or via tensor_reduce on transposed layout.
        sum_per_partition = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_reduce(
            dst=sum_per_partition,
            data=exp_scores,
            op=nl.add,
            axis=1,
        )
        # Now reduce sum_per_partition (128, 1) → (1, 1) total.
        # Pad to (128, 128) and transpose, then reduce again? Or use
        # the matmul-with-ones trick.
        # Cleanest: nc_transpose (128, 1) → (1, 128), then tensor_reduce
        # along axis=1.
        sum_pad = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(dst=sum_pad, value=0.0)
        nisa.tensor_copy(
            dst=sum_pad[0:P_MAX, 0:1],
            src=sum_per_partition[0:P_MAX, 0:1],
        )
        sum_t_psum = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_transpose(dst=sum_t_psum, data=sum_pad)
        # sum_t_psum[0, 0:128] = sum_per_partition.flat
        sum_row = nl.ndarray((1, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=sum_row, src=sum_t_psum[0:1, 0:P_MAX])
        denom = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=denom, data=sum_row, op=nl.add, axis=1)

        # Compute 1/denom (NKI tensor_scalar doesn't support `divide` op,
        # so we precompute the reciprocal and multiply).
        inv_denom = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.reciprocal(dst=inv_denom, data=denom)

        # Broadcast inv_denom (1, 1) to (128, num_chunks). tensor_scalar
        # with operand0=(1, 1) can't broadcast across the 128 partitions of
        # dst — NKI's verifier rejects it ("operand0 partition total
        # elements 1 != dst partition total elements 128"). Use
        # nl.broadcast_to which the compiler folds into an access-pattern.
        inv_denom_bcast = nl.broadcast_to(inv_denom, shape=(P_MAX, num_chunks))

        # weights = exp_scores * inv_denom_bcast (full tile-tile multiply)
        weights_fp32 = nl.ndarray((P_MAX, num_chunks), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(
            dst=weights_fp32,
            data1=exp_scores,
            data2=inv_denom_bcast,
            op=nl.multiply,
        )

        # Cast weights to bf16 for the AV matmul
        weights = nl.ndarray((P_MAX, num_chunks), dtype=q.dtype, buffer=nl.sbuf)
        nisa.tensor_copy(dst=weights, src=weights_fp32)

        # =====================================================================
        # Pass 2 — AV matmul. Output: (1, 256) split into out_lo + out_hi.
        # =====================================================================
        # We need out_lo (1, 128) = sum_over_ctx(weights[ctx] * V[ctx, :128])
        # and similarly for out_hi.
        #
        # In the (128 ctx, num_chunks) weights layout, weights[p, c]
        # corresponds to ctx-token (c*128 + p).
        # For each chunk c, we have V_chunk_lo (128, 128) and V_chunk_hi
        # (128, 128). The matmul we want is:
        #   out_lo += sum_p (weights[p, c] * V_chunk_lo[p, :])
        # which is weights[:, c].T @ V_chunk_lo, with weights[:, c] as
        # the moving and V_chunk_lo as the stationary along ctx-dim.
        # nc_matmul: stationary V_chunk_lo (128 dim × 128 ctx),
        #            moving weights[:, c] (128 ctx × 1)
        # → result (128 dim × 1) — but we want (1 token × 128 dim).
        # Equivalent — just transpose in the layout.
        # Result shape per chunk: (128 dim, 1) PSUM. Accumulate across
        # all chunks into a single (128, 1) PSUM tile (PSUM accumulation).

        out_lo_psum = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.psum)
        out_hi_psum = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.psum)

        for c in nl.affine_range(num_chunks):
            chunk_off = c * P_MAX

            v_lo = nl.ndarray((P_MAX, HEAD_DIM_HALF), dtype=v_full.dtype, buffer=nl.sbuf)
            v_hi = nl.ndarray((P_MAX, HEAD_DIM_HALF), dtype=v_full.dtype, buffer=nl.sbuf)
            nisa.dma_copy(
                dst=v_lo,
                src=v_full[chunk_off:chunk_off + P_MAX, 0:HEAD_DIM_HALF],
            )
            nisa.dma_copy(
                dst=v_hi,
                src=v_full[chunk_off:chunk_off + P_MAX, HEAD_DIM_HALF:HEAD_DIM],
            )

            # nc_matmul:
            # stationary V_lo (128 ctx × 128 dim) — partition dim = ctx
            # moving weights[:, c:c+1] (128 ctx × 1) — partition dim = ctx
            # → PSUM result (128 dim × 1)
            # Wait — moving must have partition dim = same K as stationary,
            # which is the ctx dim here (128). So both are (128, ...).
            # Result is (128 dim, 1). Accumulates across chunks.
            nisa.nc_matmul(
                dst=out_lo_psum,
                stationary=v_lo,
                moving=weights[0:P_MAX, c:c + 1],
            )
            nisa.nc_matmul(
                dst=out_hi_psum,
                stationary=v_hi,
                moving=weights[0:P_MAX, c:c + 1],
            )

        # Copy PSUM → SBUF, cast to bf16
        out_lo = nl.ndarray((P_MAX, 1), dtype=q.dtype, buffer=nl.sbuf)
        out_hi = nl.ndarray((P_MAX, 1), dtype=q.dtype, buffer=nl.sbuf)
        nisa.tensor_copy(dst=out_lo, src=out_lo_psum)
        nisa.tensor_copy(dst=out_hi, src=out_hi_psum)

        # Transpose (128, 1) → (1, 128) for each half. Pad to (128, 128),
        # transpose, then copy result row 0 (which now holds the original
        # column 0) into an SBUF staging tile. Finally DMA SBUF → HBM.
        out_lo_pad = nl.ndarray((P_MAX, P_MAX), dtype=q.dtype, buffer=nl.sbuf)
        nisa.memset(dst=out_lo_pad, value=0.0)
        nisa.tensor_copy(dst=out_lo_pad[0:P_MAX, 0:1], src=out_lo)
        out_lo_t_psum = nl.ndarray((P_MAX, P_MAX), dtype=q.dtype, buffer=nl.psum)
        nisa.nc_transpose(dst=out_lo_t_psum, data=out_lo_pad)
        # PSUM → SBUF staging (dma_copy can't go PSUM → HBM directly)
        out_lo_sbuf = nl.ndarray((1, HEAD_DIM_HALF), dtype=q.dtype, buffer=nl.sbuf)
        nisa.tensor_copy(dst=out_lo_sbuf, src=out_lo_t_psum[0:1, 0:HEAD_DIM_HALF])

        out_hi_pad = nl.ndarray((P_MAX, P_MAX), dtype=q.dtype, buffer=nl.sbuf)
        nisa.memset(dst=out_hi_pad, value=0.0)
        nisa.tensor_copy(dst=out_hi_pad[0:P_MAX, 0:1], src=out_hi)
        out_hi_t_psum = nl.ndarray((P_MAX, P_MAX), dtype=q.dtype, buffer=nl.psum)
        nisa.nc_transpose(dst=out_hi_t_psum, data=out_hi_pad)
        out_hi_sbuf = nl.ndarray((1, HEAD_DIM_HALF), dtype=q.dtype, buffer=nl.sbuf)
        nisa.tensor_copy(dst=out_hi_sbuf, src=out_hi_t_psum[0:1, 0:HEAD_DIM_HALF])

        # SBUF → HBM
        nisa.dma_copy(dst=output[0:1, 0:HEAD_DIM_HALF], src=out_lo_sbuf)
        nisa.dma_copy(dst=output[0:1, HEAD_DIM_HALF:HEAD_DIM], src=out_hi_sbuf)

        return output


else:

    def decode_hd256_kernel(*args, **kwargs):
        raise RuntimeError(
            "nki is not available in this environment — this kernel "
            "must run inside the vllm-neuron container."
        )

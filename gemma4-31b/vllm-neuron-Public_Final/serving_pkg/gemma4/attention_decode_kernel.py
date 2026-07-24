# SPDX-License-Identifier: Apache-2.0
"""
Gemma4 Decode Attention — custom NKI kernel hook (CLEAN, post-fix).

This is the clean, single-path version of the Gemma4 decode-attention NKI
kernel after the q-load DMA-aliasing root cause was found and fixed. All of
the `GEMMA4_DIAG_*` bisection scaffolding (~57 gates) has been removed; only
the proven-correct path remains.

Background
----------
Gemma4's per-layer head_dim (256 SWA / 512 global) exceeds the
``attention_block_tkg`` megakernel's 128 limit, so the standard
``vllm_neuron.functional.attention_decode`` path can't be used. This kernel
handles the larger head dims directly with flash-attention-style online
softmax (ported from nkilib/core/attention/attention_tkg.py).

Decode has S_q=1, so attention reduces to:
    score[h, k] = sum_d q[h, d] * k_cache[h_kv, d, k]
    p[h, k]     = softmax(score[h, :])
    out[h, d]   = sum_k p[h, k] * v_cache[h_kv, k, d]

trn2 partition dim is hardware-capped at 128, so head_dim>128 needs d-tiling
(d split into chunks of 128). K cache is PRE-TRANSPOSED in HBM as
(B, Hkv, D, S_total) so MM1 loads K directly without computing K^T.

THE ROOT CAUSE AND FIX (see FIX_SUMMARY.md)
-------------------------------------------
Under torch-xla `wrap_nki` (this vllm plugin's lowering stack — DIFFERENT from
torch_neuronx, where this kernel originally worked), issuing MULTIPLE separate
single-slice ``nisa.dma_copy`` CALLS to one operand, keyed by a scalar index,
collapses to the FIRST index. The original q-load packed the GQA query heads
with a per-head loop of separate single-column DMA calls keyed by a scalar head
index — so every packed column aliased to head-0, duplicating head-0 across all
heads and producing garbage tokens (-12545868, incoherent text).

THE FIX: load all local query heads with ONE multi-partition DMA (heads on the
partition axis, mirroring K's faithful geometry), then ``nc_transpose`` into
q_packed's (D-on-partition, heads-on-free) layout. The SAME single-DMA discipline
is applied to the output write-back. Proven: per-head cosine vs CPU-fp32 oracle
-> ~1.0 for both heads (kdiff raysubmit_i9bAP1eRB3TpCeCq), and coherent
end-to-end tokens (prod raysubmit_eXh6zw6YEBvwLtea: France->Paris, 2+2=4, etc.).

Key invariant for any wrap_nki NKI kernel: NEVER load or store a packed axis with
a per-index loop of separate single-slice DMAs — use ONE multi-partition DMA
(axis on the partition dim) plus an on-chip transpose.

Optimizations retained from attention_tkg:
  * Negated-max trick: running_max stored as -max so exp(qk - max) becomes
    exp(qk + bias) with bias=-max via the activation's bias arg.
  * Fused exp+reduce: one Scalar Engine instruction folds exp(qk-max),
    tile_exp materialize, and the tile_sum reduction.
  * NaN guard: _POS_FINITE sentinel + min-clamp prevents +inf-+inf=NaN on
    fully-masked tiles.
  * GQA query-head packing: query heads sharing one kv_h are packed on the MM1
    output partition so softmax / running-buffer updates run in parallel.
"""
from __future__ import annotations

import torch
from torch import Tensor
import nki
import nki.isa as nisa
import nki.language as nl

from vllm_neuron.nki.nki_hop import can_run_kernel, wrap_nki

# trn2 partition dim cap; also the K-tile width for MM1 (must be equal).
_P = 128
_K_TILE = 128

# Finite sentinel for the running-max init (negated-max convention). Using +inf
# would risk +inf - (+inf) = NaN per IEEE 754 on fully-masked first tiles; a
# finite sentinel makes the diff finite-but-very-negative so exp(diff)=0 cleanly.
# Mirrors attention_tkg's _clamp_max_to_finite.
_POS_FINITE = 1.0e30


# ---------------------------------------------------------------------------
# Eligibility check
# ---------------------------------------------------------------------------


def _can_use_nki_kernel(
    q: Tensor,
    k: Tensor,
    num_kv_groups: int,
) -> bool:
    """Check whether tensor shapes satisfy the NKI kernel's static constraints.

    The kernel requires:
      - tensors on Neuron (or NKI sim)
      - S_decode == 1 (single-token decode; spec decode S_decode>1 unsupported)
      - head_dim divisible by 128 (partition dim)
      - S_ctx divisible by 128 (K-tile width)
      - num_q_heads == num_kv_heads * num_kv_groups
    """
    if not can_run_kernel(q):
        return False

    _, num_q_heads, S_decode, head_dim = q.shape
    _, num_kv_heads, S_ctx, _ = k.shape

    if S_decode != 1:
        return False
    if head_dim % _P != 0:
        return False
    if S_ctx % _K_TILE != 0:
        return False
    if num_q_heads != num_kv_heads * num_kv_groups:
        return False
    return True


# ---------------------------------------------------------------------------
# PyTorch reference implementation (fallback + correctness oracle)
# ---------------------------------------------------------------------------


def _torch_attention_decode(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    attn_mask: Tensor,
    softmax_scale: float,
    num_kv_groups: int,
) -> Tensor:
    """PyTorch reference for Gemma4 decode attention.

    Args:
        q:           [B, num_q_heads, S_decode, head_dim]
        k:           [B, num_kv_heads, S_ctx, head_dim]
        v:           [B, num_kv_heads, S_ctx, head_dim]
        attn_mask:   [B, 1, S_decode, S_ctx] additive mask (0 keep, -inf mask).
        softmax_scale: Multiplied into QK^T before softmax (1.0 for Gemma4).
        num_kv_groups: GQA group size (num_q_heads // num_kv_heads).

    Returns:
        [B, num_q_heads, S_decode, head_dim] attention output.
    """
    B, num_q_heads, S_decode, head_dim = q.shape
    _, num_kv_heads, S_ctx, _ = k.shape

    if num_kv_groups > 1:
        k = (
            k.unsqueeze(2)
            .expand(B, num_kv_heads, num_kv_groups, S_ctx, head_dim)
            .reshape(B, num_q_heads, S_ctx, head_dim)
        )
        v = (
            v.unsqueeze(2)
            .expand(B, num_kv_heads, num_kv_groups, S_ctx, head_dim)
            .reshape(B, num_q_heads, S_ctx, head_dim)
        )

    return torch.nn.functional.scaled_dot_product_attention(
        q, k, v,
        attn_mask=attn_mask,
        is_causal=False,
        scale=softmax_scale,
    )


# ---------------------------------------------------------------------------
# NKI kernel
#
# Inputs are in kernel-native layout (caller-side adapter handles the
# permute / reshape from the model's [B, H, S_dec, D] tensors).
#
# Dispatched single-core (n_prgs=1): one program owns all GQA query heads of
# each kv_h, so local_kv_groups == kv_groups and hi_base == 0. The output rows
# are contiguous, which lets the write-back use ONE multi-partition DMA.
# ---------------------------------------------------------------------------


@nki.jit
def _nki_attention_decode_kernel(
    q: nl.ndarray,            # [B, H, D]
    k_cache: nl.ndarray,      # [B, Hkv, D, S_total]   pre-transposed
    v_cache: nl.ndarray,      # [B, Hkv, S_total, D]
    attn_mask: nl.ndarray,    # [B, 1, 1, S_total]  (added to scores; 0=keep, -inf=mask)
    scale: float = 1.0,
):
    B, H, D = q.shape
    Bk, Hkv, Dk, S_total = k_cache.shape
    Bv, Hv, Sv, Dv = v_cache.shape
    assert B == Bk == Bv
    assert D == Dk == Dv
    assert Hkv == Hv
    assert H % Hkv == 0
    assert D % _P == 0
    assert S_total % _P == 0
    assert S_total % _K_TILE == 0, f"S_total {S_total} must be multiple of {_K_TILE}"
    assert _K_TILE == _P, "_K_TILE must equal _P for MM2 V tile alignment"

    n_d = D // _P
    n_s = S_total // _P
    n_kt = S_total // _K_TILE
    kv_groups = H // Hkv

    # Single-core dispatch (wrapped[1]): one program owns every GQA head.
    n_prgs = nl.num_programs(axes=0)
    prg_id = nl.program_id(axis=0)
    assert kv_groups % n_prgs == 0
    local_kv_groups = kv_groups // n_prgs
    hi_base = prg_id * local_kv_groups

    o = nl.ndarray((B, H, D), dtype=q.dtype, buffer=nl.shared_hbm)

    for b in nl.affine_range(B):
        for kv_h in nl.affine_range(Hkv):
            # ----- Pre-load K once per (b, kv_h): one multi-partition DMA per
            # d-chunk (D on partition, full seq on free). K's faithful geometry. -----
            k_full_list = []
            for dc in nl.static_range(n_d):
                k_d_buf = nl.ndarray((_P, S_total), dtype=k_cache.dtype, buffer=nl.sbuf)
                nisa.dma_copy(
                    dst=k_d_buf[:, :],
                    src=k_cache[b, kv_h, nl.ds(dc * _P, _P), :],
                )
                k_full_list.append(k_d_buf)

            mm2_dtype = v_cache.dtype

            # ----- Pre-load V once per (b, kv_h), shared across kv_groups. -----
            v_tile_list = []
            for sc in nl.static_range(n_s):
                for dc in nl.static_range(n_d):
                    v_chunk = nl.ndarray((_P, _P), dtype=mm2_dtype, buffer=nl.sbuf)
                    nisa.dma_copy(
                        dst=v_chunk[:, :],
                        src=v_cache[b, kv_h, nl.ds(sc * _P, _P), nl.ds(dc * _P, _P)],
                    )
                    v_tile_list.append(v_chunk)

            # ----- Pre-load full mask once per (b, kv_h) -----
            # Replicate the single-row mask across local_kv_groups partitions so
            # it broadcasts against the head-packed qk_tile in the per-kt add.
            mask_b = 0 if attn_mask.shape[0] == 1 else b
            mask_full = nl.ndarray((local_kv_groups, S_total), dtype=nl.float32, buffer=nl.sbuf)
            for hi in nl.static_range(local_kv_groups):
                nisa.dma_copy(
                    dst=mask_full[nl.ds(hi, 1), :],
                    src=attn_mask[mask_b, 0:1, 0, :],
                )

            # ----- GQA query-head packing (ROOT-CAUSE FIX: Q_1DMA) -----
            # Load all local query heads with ONE multi-partition DMA (heads on
            # the partition axis, D-chunk on free) — exactly K's faithful
            # geometry, so there is no per-head scalar offset to alias. Pad to a
            # full (_P,_P) tile and nc_transpose (both transpose partitions are
            # full-128), then slice cols 0:lkg into q_packed (D on partition,
            # heads on free). MM1 then produces (local_kv_groups, _K_TILE).
            q_packed_list = []
            h_base_q = kv_h * kv_groups + hi_base
            for dc in nl.static_range(n_d):
                # q_packed is (D-on-partition, heads-on-free); stage the transpose
                # through full-128 buffers, then slice the real head columns back.
                q_hpart = nl.ndarray((_P, _P), dtype=q.dtype, buffer=nl.sbuf)
                nisa.memset(q_hpart, value=0.0)
                nisa.dma_copy(
                    dst=q_hpart[nl.ds(0, local_kv_groups), :],
                    src=q[b, nl.ds(h_base_q, local_kv_groups), nl.ds(dc * _P, _P)],
                )
                q_t_psum = nl.ndarray((_P, _P), dtype=q.dtype, buffer=nl.psum)
                nisa.nc_transpose(dst=q_t_psum, data=q_hpart, engine=nisa.engine.tensor)
                q_packed_full = nl.ndarray((_P, local_kv_groups), dtype=q.dtype, buffer=nl.sbuf)
                nisa.tensor_copy(
                    dst=q_packed_full[:, nl.ds(0, local_kv_groups)],
                    src=q_t_psum[:, nl.ds(0, local_kv_groups)],
                )
                q_packed_list.append(q_packed_full)

            # ----- Flash-attention online-softmax running buffers (per-head packed) -----
            running_max_neg = nl.ndarray((local_kv_groups, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.memset(running_max_neg, value=_POS_FINITE)
            running_sum = nl.ndarray((local_kv_groups, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.memset(running_sum, value=0.0)
            running_output = nl.ndarray((local_kv_groups, D), dtype=nl.float32, buffer=nl.sbuf)
            nisa.memset(running_output, value=0.0)

            # ----- Fused MM1 -> softmax -> MM2 loop over K-tiles -----
            # MUST be sequential_range: the running buffers are read-modify-
            # written every iteration (loop-carried dependency). affine_range
            # would let the compiler reorder iterations and corrupt the
            # accumulation order, yielding valid-ranged-but-wrong output.
            for kt in nl.sequential_range(n_kt):
                k_off = kt * _K_TILE

                # MM1: tile_psum[local_kv_groups, _K_TILE] = q_packed^T @ K_tile,
                # accumulated over d-chunks.
                tile_psum = nl.ndarray((local_kv_groups, _K_TILE), dtype=nl.float32, buffer=nl.psum)
                for dc in nl.static_range(n_d):
                    nisa.nc_matmul(
                        tile_psum,
                        stationary=q_packed_list[dc],
                        moving=k_full_list[dc][:, nl.ds(k_off, _K_TILE)],
                        accumulate=(dc > 0),
                    )

                # scores = scale*qk + mask (mask broadcasts across heads).
                qk_tile = nl.ndarray((local_kv_groups, _K_TILE), dtype=nl.float32, buffer=nl.sbuf)
                if scale != 1.0:
                    nisa.tensor_scalar(
                        dst=qk_tile, data=tile_psum,
                        op0=nl.multiply, operand0=scale,
                    )
                    nisa.tensor_tensor(
                        dst=qk_tile, data1=qk_tile,
                        data2=mask_full[:, nl.ds(k_off, _K_TILE)],
                        op=nl.add,
                    )
                else:
                    nisa.tensor_tensor(
                        dst=qk_tile, data1=tile_psum,
                        data2=mask_full[:, nl.ds(k_off, _K_TILE)],
                        op=nl.add,
                    )

                # ----- Online softmax update (per-head, local_kv_groups in parallel) -----
                neg_tile_max = nl.ndarray((local_kv_groups, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_reduce(
                    dst=neg_tile_max, data=qk_tile, op=nl.maximum,
                    axis=1, keepdims=True, negate=True,
                )
                nisa.tensor_scalar(
                    dst=neg_tile_max, data=neg_tile_max,
                    op0=nl.minimum, operand0=_POS_FINITE,
                )
                new_max_neg = nl.ndarray((local_kv_groups, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_tensor(
                    dst=new_max_neg, data1=running_max_neg, data2=neg_tile_max,
                    op=nl.minimum,
                )
                diff_for_corr = nl.ndarray((local_kv_groups, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_tensor(
                    dst=diff_for_corr, data1=new_max_neg, data2=running_max_neg,
                    op=nl.subtract,
                )
                correction = nl.ndarray((local_kv_groups, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.activation(dst=correction, data=diff_for_corr, op=nl.exp)

                # tile_exp = exp(qk - new_max), with the row-sum reduction fused.
                tile_exp_bf = nl.ndarray((local_kv_groups, _K_TILE), dtype=mm2_dtype, buffer=nl.sbuf)
                tile_sum = nl.ndarray((local_kv_groups, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.activation(
                    dst=tile_exp_bf, op=nl.exp, data=qk_tile,
                    bias=new_max_neg,
                    reduce_op=nl.add,
                    reduce_res=tile_sum,
                    reduce_cmd=nisa.reduce_cmd.reset_reduce,
                )

                # ----- MM2: transpose tile_exp -> p_t, then p_t @ V -----
                p_t = nl.ndarray((_K_TILE, local_kv_groups), dtype=mm2_dtype, buffer=nl.sbuf)
                p_t_psum = nl.ndarray((_K_TILE, local_kv_groups), dtype=mm2_dtype, buffer=nl.psum)
                nisa.nc_transpose(dst=p_t_psum, data=tile_exp_bf, engine=nisa.engine.tensor)
                nisa.tensor_copy(dst=p_t, src=p_t_psum)

                for dc in nl.static_range(n_d):
                    out_psum = nl.ndarray((local_kv_groups, _P), dtype=nl.float32, buffer=nl.psum)
                    nisa.nc_matmul(
                        out_psum,
                        stationary=p_t,
                        moving=v_tile_list[kt * n_d + dc],
                        accumulate=False,
                    )
                    # running_output[:, dc] = running_output[:, dc]*correction + out_psum
                    ro_slice = running_output[:, nl.ds(dc * _P, _P)]
                    nisa.tensor_scalar(
                        dst=ro_slice, data=ro_slice,
                        op0=nl.multiply, operand0=correction,
                    )
                    nisa.tensor_tensor(
                        dst=ro_slice, data1=ro_slice, data2=out_psum, op=nl.add,
                    )

                # running_sum = running_sum*correction + tile_sum
                nisa.tensor_tensor(
                    dst=running_sum, data1=running_sum, data2=correction, op=nl.multiply,
                )
                nisa.tensor_tensor(
                    dst=running_sum, data1=running_sum, data2=tile_sum, op=nl.add,
                )
                nisa.tensor_copy(dst=running_max_neg, src=new_max_neg)

            # ----- Finalize: out = running_output / running_sum -----
            recip_sum = nl.ndarray((local_kv_groups, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.reciprocal(dst=recip_sum, data=running_sum)
            out_bf = nl.ndarray((local_kv_groups, D), dtype=q.dtype, buffer=nl.sbuf)
            nisa.tensor_scalar(
                dst=out_bf, data=running_output,
                op0=nl.multiply, operand0=recip_sum,
            )

            # ----- Write-back (ROOT-CAUSE FIX: single multi-partition DMA) -----
            # ONE dma_copy writes all local heads of this kv group in a single
            # shot. out_bf rows 0:local_kv_groups are the distinct heads (packed
            # MM1 lands each head on its own row). With n_prgs=1, hi_base=0, so
            # heads h0 .. h0+local_kv_groups are CONTIGUOUS rows of o[b]. A single
            # multi-partition write of the returned shared_hbm tensor is the
            # single-producer write nkilib uses; the multi-CALL slice-drop that
            # zeroed downstream heads cannot occur.
            h0 = kv_h * kv_groups + hi_base
            nisa.dma_copy(
                dst=o[b, nl.ds(h0, local_kv_groups), :],
                src=out_bf[nl.ds(0, local_kv_groups), :],
            )

    return o


# ---------------------------------------------------------------------------
# NKI adapter — converts model-side layout to kernel-native layout, dispatches
# the kernel through wrap_nki (so it shows up as an HOP in the FX graph), and
# reshapes the output back.
# ---------------------------------------------------------------------------


def _nki_attention_decode(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    attn_mask: Tensor,
    softmax_scale: float,
) -> Tensor:
    """Caller-side adapter for the NKI kernel.

    Converts:
        q:         [B, H,   1,    D]  -> [B, H,   D]      (squeeze S_decode)
        k:         [B, Hkv, S,    D]  -> [B, Hkv, D, S]   (transpose for K-stationary MM1)
        v:         [B, Hkv, S,    D]  (passed through)
        attn_mask: [B, 1,   1,    S]  -> [B, 1, 1, S]     (already correct)

    And reshapes the kernel's [B, H, D] output back to [B, H, 1, D].
    """
    q_kernel = q.squeeze(2)                       # [B, H, D]
    k_kernel = k.transpose(-1, -2).contiguous()   # [B, Hkv, D, S_ctx]
    v_kernel = v.contiguous()                     # [B, Hkv, S_ctx, D]
    mask_kernel = attn_mask                        # [B, 1, 1, S_ctx]

    wrapped = wrap_nki(_nki_attention_decode_kernel)
    # Single-core dispatch (n_prgs=1): one program owns all GQA heads, so the
    # output head rows are contiguous and the write-back is one multi-partition
    # DMA. This is the proven-correct configuration (see FIX_SUMMARY.md).
    out = wrapped[1](
        q=q_kernel,
        k_cache=k_kernel,
        v_cache=v_kernel,
        attn_mask=mask_kernel,
        scale=softmax_scale,
    )
    return out.unsqueeze(2)  # [B, H, 1, D]


# ---------------------------------------------------------------------------
# Public dispatch
# ---------------------------------------------------------------------------


def attention_decode(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    attn_mask: Tensor,
    softmax_scale: float,
    num_kv_groups: int,
    force_torch: bool = False,
) -> Tensor:
    """Dispatch Gemma4 decode attention to the NKI kernel or torch fallback.

    Args:
        q:             [B, num_q_heads, S_decode, head_dim]
        k:             [B, num_kv_heads, S_ctx, head_dim]
        v:             [B, num_kv_heads, S_ctx, head_dim]
        attn_mask:     [B, 1, S_decode, S_ctx] additive mask.
        softmax_scale: Scale applied before softmax (1.0 for Gemma4).
        num_kv_groups: GQA group size.
        force_torch:   When True, always use the PyTorch reference.

    Returns:
        [B, num_q_heads, S_decode, head_dim]
    """
    if force_torch or not _can_use_nki_kernel(q, k, num_kv_groups):
        return _torch_attention_decode(
            q, k, v, attn_mask, softmax_scale, num_kv_groups
        )

    return _nki_attention_decode(q, k, v, attn_mask, softmax_scale)

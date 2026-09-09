# SPDX-License-Identifier: Apache-2.0
"""Dual-source flash attention with sink bias for DeepSeek-V4-Flash decode.

Fuses the model's two-source attention (sliding-window prior_kv + compressed
csa_kv sharing ONE softmax max/denominator + a per-head sink term) into a single
NKI kernel. Online softmax is order-independent, so we run the SAME running
max/sum/accumulator over source-A blocks then source-B blocks, add the sink to
the denominator once, and normalize. Numerically identical to:
    M = max(max(q@kvA^T*sc + maskA), max(q@kvB^T*sc + maskB), sink)
    denom = sum(exp(A-M)) + sum(exp(B-M)) + exp(sink-M)
    out = (sum(exp(A-M)@kvA) + sum(exp(B-M)@kvB)) / denom

Derived from the single-source sparse_attn_nki (same block-processing body).
"""
import nki
import nki.isa as nisa
import nki.language as nl
import numpy as np
import os

# Batch-loop scheduling toggle: affine_range (default, full-unroll, banked)
# vs sequential_range (no unroll -> smaller graph, lower SBUF pressure at high batch).
_B_LOOP_SEQ = os.environ.get("V4_ATTN_B_SEQ", "0") == "1"

P_MAX = 128
PSUM_FMAX = 512
_MIN_FLOAT32 = float(np.finfo(np.float32).min)
KV_BLOCK_MIN = 128


def kernel_assert(condition: bool, error_text: str):
    assert condition, f"[INTERNAL_ERROR] [NCC_INKI016] Kernel validation exception: {error_text}"


def div_ceil(n: int, d: int) -> int:
    return (n + d - 1) // d


def _broadcast_p0(src, dst):
    """Broadcast partition 0 of src to all partitions of dst via stream shuffle."""
    dst_npar = dst.shape[0]
    SG = 32
    shuffle_mask = [0] * SG
    for gi in range((dst_npar + SG - 1) // SG):
        cur = min(SG, dst_npar - gi * SG)
        nisa.nc_stream_shuffle(
            src=src[0:1, :],
            dst=dst[gi * SG:gi * SG + cur, 0:dst.shape[1]],
            shuffle_mask=shuffle_mask,
        )


def _pick_kv_block(T):
    kv_block = KV_BLOCK_MIN
    for c in range(KV_BLOCK_MIN, PSUM_FMAX + 1, KV_BLOCK_MIN):
        if T % c == 0:
            kv_block = c
    return kv_block


def _process_source(q_t_tiles, num_d_tiles_k, kv_src, valid_4d, b_idx, s_idx,
                    H, D, T, scale, running_max, running_sum, acc_out):
    """Online-softmax over ONE KV source's blocks; updates running_max/sum/acc in place."""
    kv_block = _pick_kv_block(T)
    num_loads = kv_block // P_MAX
    num_blocks = T // kv_block
    for blk in nl.sequential_range(num_blocks):
        blk_start = blk * kv_block
        # Load additive validity mask [1, kv_block] -> broadcast to [H, kv_block]
        mask_sb = nl.ndarray((1, kv_block), dtype=nl.float32, buffer=nl.sbuf)
        for ld in nl.affine_range(num_loads):
            ld_start = blk_start + ld * P_MAX
            f_start = ld * P_MAX
            mc = nl.ndarray((1, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=mc[0:1, 0:P_MAX], src=valid_4d[b_idx, s_idx, 0:1, ld_start:ld_start + P_MAX])
            nisa.tensor_copy(dst=mask_sb[0:1, f_start:f_start + P_MAX], src=mc)
        mask_bc = nl.ndarray((H, kv_block), dtype=nl.float32, buffer=nl.sbuf)
        _broadcast_p0(mask_sb, mask_bc)
        # Load KV sub-blocks [P_MAX, D]
        kv_subs = [None] * num_loads
        for ld in nl.affine_range(num_loads):
            ld_start = blk_start + ld * P_MAX
            ks = nl.ndarray((P_MAX, D), dtype=kv_src.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=ks[0:P_MAX, 0:D], src=kv_src[b_idx, s_idx, ld_start:ld_start + P_MAX, 0:D])
            kv_subs[ld] = ks
        # Scores: Q_t^T @ KV^T -> [H, kv_block]
        scores_psum = nl.ndarray((H, kv_block), dtype=nl.float32, buffer=nl.psum)
        for d in nl.affine_range(num_d_tiles_k):
            d_start = d * P_MAX
            d_end = min(d_start + P_MAX, D)
            d_sz = d_end - d_start
            kv_t = nl.ndarray((d_sz, kv_block), dtype=kv_src.dtype, buffer=nl.sbuf)
            for ld in nl.affine_range(num_loads):
                f_start = ld * P_MAX
                ktp = nl.ndarray((d_sz, P_MAX), dtype=kv_src.dtype, buffer=nl.psum)
                nisa.nc_transpose(dst=ktp[0:d_sz, 0:P_MAX], data=kv_subs[ld][0:P_MAX, d_start:d_end])
                nisa.tensor_copy(dst=kv_t[0:d_sz, f_start:f_start + P_MAX], src=ktp)
            nisa.nc_matmul(dst=scores_psum, stationary=q_t_tiles[d][0:d_sz, 0:H], moving=kv_t[0:d_sz, 0:kv_block])
        # Scale + additive mask
        ss = nl.ndarray((H, kv_block), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=ss, data=scores_psum, op0=nl.multiply, operand0=scale)
        sm = nl.ndarray((H, kv_block), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=sm, data1=ss, data2=mask_bc, op=nl.add)
        # Online softmax update
        tile_max = nl.ndarray((H, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=tile_max, data=sm, op=nl.maximum, axis=1)
        new_max = nl.ndarray((H, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=new_max, data1=running_max, data2=tile_max, op=nl.maximum)
        max_diff = nl.ndarray((H, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=max_diff, data1=running_max, data2=new_max, op=nl.subtract)
        corr = nl.ndarray((H, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=corr, data=max_diff, op=nl.exp)
        shifted = nl.ndarray((H, kv_block), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=shifted, data1=sm, data2=new_max.ap(pattern=[[1, H], [0, kv_block]], offset=0), op=nl.subtract)
        exps = nl.ndarray((H, kv_block), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=exps, data=shifted, op=nl.exp)
        tile_sum = nl.ndarray((H, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=tile_sum, data=exps, op=nl.add, axis=1)
        csum = nl.ndarray((H, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=csum, data1=running_sum, data2=corr, op=nl.multiply)
        nisa.tensor_tensor(dst=running_sum, data1=csum, data2=tile_sum, op=nl.add)
        nisa.tensor_copy(dst=running_max, src=new_max)
        # AV matmul: exps @ KV -> [H, D], with rescale of accumulator
        av_psum = nl.ndarray((H, D), dtype=nl.float32, buffer=nl.psum)
        for sk in nl.affine_range(num_loads):
            sk_start = sk * P_MAX
            eb = nl.ndarray((H, P_MAX), dtype=kv_src.dtype, buffer=nl.sbuf)
            nisa.tensor_copy(dst=eb[0:H, 0:P_MAX], src=exps[0:H, sk_start:sk_start + P_MAX])
            etp = nl.ndarray((P_MAX, H), dtype=kv_src.dtype, buffer=nl.psum)
            nisa.nc_transpose(dst=etp[0:P_MAX, 0:H], data=eb[0:H, 0:P_MAX])
            ets = nl.ndarray((P_MAX, H), dtype=kv_src.dtype, buffer=nl.sbuf)
            nisa.tensor_copy(dst=ets, src=etp)
            nisa.nc_matmul(dst=av_psum, stationary=ets[0:P_MAX, 0:H], moving=kv_subs[sk][0:P_MAX, 0:D])
        cacc = nl.ndarray((H, D), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=cacc, data1=acc_out, data2=corr.ap(pattern=[[1, H], [0, D]], offset=0), op=nl.multiply)
        nisa.tensor_tensor(dst=acc_out, data1=cacc, data2=av_psum, op=nl.add)


@nki.jit
def dual_sparse_attn_nki(q, kv_a, valid_a, kv_b, valid_b, attn_sink, softmax_scale):
    """Dual-source flash attention with sink. See module docstring.

    Args:
        q:        [B, S, H, D]
        kv_a:     [B, S, Ta, D]   (e.g. sliding-window prior KV)
        valid_a:  [B, S, Ta]      additive mask (0 valid, large-negative masked)
        kv_b:     [B, S, Tb, D]   (e.g. compressed CSA KV)
        valid_b:  [B, S, Tb]      additive mask
        attn_sink:[H]             per-head sink bias in the softmax denominator
        softmax_scale: 1/sqrt(D)
    Returns:
        out: [B, S, H, D]
    """
    B, S, H, D = q.shape
    Ta = kv_a.shape[2]
    Tb = kv_b.shape[2]
    kernel_assert(H <= P_MAX, f"H ({H}) must be <= {P_MAX}")
    kernel_assert(D <= PSUM_FMAX, f"D ({D}) must be <= {PSUM_FMAX}")
    kernel_assert(Ta % KV_BLOCK_MIN == 0, f"Ta ({Ta}) must be multiple of {KV_BLOCK_MIN}")
    kernel_assert(Tb % KV_BLOCK_MIN == 0, f"Tb ({Tb}) must be multiple of {KV_BLOCK_MIN}")
    num_d_tiles_k = div_ceil(D, P_MAX)
    out = nl.ndarray(q.shape, dtype=q.dtype, buffer=nl.shared_hbm)
    valid_a_4d = valid_a.reshape((B, S, 1, Ta))
    valid_b_4d = valid_b.reshape((B, S, 1, Tb))
    _b_range = nl.sequential_range if _B_LOOP_SEQ else nl.affine_range
    for b_idx in _b_range(B):
        sink_sb = nl.ndarray((H, 1), dtype=nl.float32, buffer=nl.sbuf)
        sink_tmp = nl.ndarray((H, 1), dtype=attn_sink.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=sink_tmp[0:H, 0:1], src=attn_sink.reshape((H, 1))[0:H, 0:1])
        nisa.tensor_copy(dst=sink_sb, src=sink_tmp)
        for s_idx in nl.affine_range(S):
            # Transpose Q [H,D] -> [D,H] tiles for the score matmul
            q_sb = nl.ndarray((H, D), dtype=q.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=q_sb[0:H, 0:D], src=q[b_idx, s_idx, 0:H, 0:D])
            q_t_tiles = [None] * num_d_tiles_k
            for d in nl.affine_range(num_d_tiles_k):
                d_start = d * P_MAX
                d_end = min(d_start + P_MAX, D)
                d_sz = d_end - d_start
                qtp = nl.ndarray((d_sz, H), dtype=q.dtype, buffer=nl.psum)
                nisa.nc_transpose(dst=qtp[0:d_sz, 0:H], data=q_sb[0:H, d_start:d_end])
                qts = nl.ndarray((d_sz, H), dtype=q.dtype, buffer=nl.sbuf)
                nisa.tensor_copy(dst=qts, src=qtp)
                q_t_tiles[d] = qts
            # Shared online-softmax state across BOTH sources
            running_max = nl.ndarray((H, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.memset(dst=running_max, value=_MIN_FLOAT32)
            running_sum = nl.ndarray((H, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.memset(dst=running_sum, value=0.0)
            acc_out = nl.ndarray((H, D), dtype=nl.float32, buffer=nl.sbuf)
            nisa.memset(dst=acc_out, value=0.0)
            _process_source(q_t_tiles, num_d_tiles_k, kv_a, valid_a_4d, b_idx, s_idx,
                            H, D, Ta, softmax_scale, running_max, running_sum, acc_out)
            _process_source(q_t_tiles, num_d_tiles_k, kv_b, valid_b_4d, b_idx, s_idx,
                            H, D, Tb, softmax_scale, running_max, running_sum, acc_out)
            # Add per-head sink to denominator
            sink_shifted = nl.ndarray((H, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=sink_shifted, data1=sink_sb, data2=running_max, op=nl.subtract)
            sink_exp = nl.ndarray((H, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(dst=sink_exp, data=sink_shifted, op=nl.exp)
            nisa.tensor_tensor(dst=running_sum, data1=running_sum, data2=sink_exp, op=nl.add)
            # Normalize: out = acc / running_sum
            inv = nl.ndarray((H, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.reciprocal(dst=inv, data=running_sum)
            of = nl.ndarray((H, D), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=of, data1=acc_out, data2=inv.ap(pattern=[[1, H], [0, D]], offset=0), op=nl.multiply)
            osb = nl.ndarray((H, D), dtype=q.dtype, buffer=nl.sbuf)
            nisa.tensor_copy(dst=osb, src=of)
            nisa.dma_copy(dst=out[b_idx, s_idx, 0:H, 0:D], src=osb)
    return out

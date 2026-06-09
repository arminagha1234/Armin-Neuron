# SPDX-License-Identifier: Apache-2.0
"""
Fused attention NKI kernel for BERT, v2 (correct nc_matmul layout).

Per-head shape: Q, K, V each [S=128, D=32 or 64], mask [1, S].

NKI nc_matmul contract:
  stationary [K, M], moving [K, N] -> result [M, N]
  where K is the contracted dim (= partition dim).

For scores = Q · K^T  ([S,S] = [S,D] · [D,S]): K_contracted = D
  stationary needs [K=D, M=S]  -> Q transposed (P=D, F=S)
  moving     needs [K=D, N=S]  -> K transposed (P=D, F=S)

For out = probs · V  ([S,D] = [S,S] · [S,D]): K_contracted = S
  stationary needs [K=S, M=S]  -> probs transposed (P=S, F=S)
  moving     needs [K=S, N=D]  -> V as-is (P=S, F=D)

Use nisa.nc_transpose for the P<->F transposes.
"""
import math

import nki
import nki.isa as nisa
import nki.language as nl


@nki.jit
def fused_attention_kernel(q_hbm, k_hbm, v_hbm, mask_hbm):
    """One head: Q,K,V [S,D]; mask [1,S] additive. Output [S,D]."""
    S, D = q_hbm.shape
    scale = 1.0 / math.sqrt(D)

    out_hbm = nl.ndarray((S, D), dtype=q_hbm.dtype, buffer=nl.shared_hbm)

    # Load Q, K, V into SBUF in [S, D] layout (P=S, F=D)
    q_sbuf = nl.ndarray((S, D), dtype=q_hbm.dtype, buffer=nl.sbuf)
    k_sbuf = nl.ndarray((S, D), dtype=k_hbm.dtype, buffer=nl.sbuf)
    v_sbuf = nl.ndarray((S, D), dtype=v_hbm.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=q_sbuf, src=q_hbm)
    nisa.dma_copy(dst=k_sbuf, src=k_hbm)
    nisa.dma_copy(dst=v_sbuf, src=v_hbm)

    # Mask comes in pre-broadcast as [S, S] additive (host-side replication
    # avoids needing tensor_tensor partition-dim broadcast in NKI).
    mask_sbuf = nl.ndarray((S, S), dtype=mask_hbm.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=mask_sbuf, src=mask_hbm)

    # ── Transpose Q and K into [D, S] layout (P=D, F=S) for nc_matmul ─────
    # Tensor Engine transpose handles up to [128, 128] but writes to PSUM,
    # so allocate PSUM destinations and copy to SBUF afterwards.
    q_t_psum = nl.ndarray((D, S), dtype=q_hbm.dtype, buffer=nl.psum)
    k_t_psum = nl.ndarray((D, S), dtype=k_hbm.dtype, buffer=nl.psum)
    nisa.nc_transpose(dst=q_t_psum, data=q_sbuf)
    nisa.nc_transpose(dst=k_t_psum, data=k_sbuf)
    q_t = nl.ndarray((D, S), dtype=q_hbm.dtype, buffer=nl.sbuf)
    k_t = nl.ndarray((D, S), dtype=k_hbm.dtype, buffer=nl.sbuf)
    nisa.tensor_copy(dst=q_t, src=q_t_psum)
    nisa.tensor_copy(dst=k_t, src=k_t_psum)

    # ── scores = Q @ K^T  ────────────────────────────────────────────────
    # stationary [K=D, M=S] = q_t, moving [K=D, N=S] = k_t -> result [S, S]
    scores_psum = nl.ndarray((S, S), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=scores_psum, stationary=q_t, moving=k_t)
    scores_sbuf = nl.ndarray((S, S), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=scores_sbuf, src=scores_psum)

    # ── scale + add mask ─────────────────────────────────────────────────
    nisa.tensor_scalar(dst=scores_sbuf, data=scores_sbuf, op0=nl.multiply, operand0=scale)
    # mask is [1, S]; broadcasts row-wise across [S, S]
    nisa.tensor_tensor(dst=scores_sbuf, data1=scores_sbuf, data2=mask_sbuf, op=nl.add)

    # ── Softmax (manual: max -> exp -> sum -> reciprocal -> multiply) ─────
    row_max = nl.ndarray((S, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=row_max, data=scores_sbuf, op=nl.max, axis=1)
    neg_max = nl.ndarray((S, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=neg_max, data=row_max, op0=nl.multiply, operand0=-1.0)
    nisa.tensor_tensor(dst=scores_sbuf, data1=scores_sbuf, data2=neg_max, op=nl.add)
    exp_sbuf = nl.ndarray((S, S), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=exp_sbuf, data=scores_sbuf, op=nl.exp)
    row_sum = nl.ndarray((S, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=row_sum, data=exp_sbuf, op=nl.add, axis=1)
    inv_sum = nl.ndarray((S, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=inv_sum, data=row_sum, op=nl.reciprocal)
    probs_sbuf = nl.ndarray((S, S), dtype=q_hbm.dtype, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=probs_sbuf, data1=exp_sbuf, data2=inv_sum, op=nl.multiply)

    # ── Transpose probs to put S (cols, contracted) on partition dim ─────
    probs_t_psum = nl.ndarray((S, S), dtype=q_hbm.dtype, buffer=nl.psum)
    nisa.nc_transpose(dst=probs_t_psum, data=probs_sbuf)
    probs_t = nl.ndarray((S, S), dtype=q_hbm.dtype, buffer=nl.sbuf)
    nisa.tensor_copy(dst=probs_t, src=probs_t_psum)

    # ── out = probs @ V  ─────────────────────────────────────────────────
    # stationary [K=S, M=S] = probs_t, moving [K=S, N=D] = v_sbuf -> [S, D]
    out_psum = nl.ndarray((S, D), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=out_psum, stationary=probs_t, moving=v_sbuf)
    out_sbuf = nl.ndarray((S, D), dtype=q_hbm.dtype, buffer=nl.sbuf)
    nisa.tensor_copy(dst=out_sbuf, src=out_psum)

    nisa.dma_copy(dst=out_hbm, src=out_sbuf)
    return out_hbm


def fused_attention(q, k, v, mask):
    """q,k,v: [B,H,S,D]; mask: [B,1,1,S] additive. Returns [B,H,S,D]."""
    import torch
    B, H, S, D = q.shape
    out = torch.empty_like(q)
    # Pre-broadcast mask [B, 1, 1, S] -> [B, S, S] on host side.
    # This avoids needing tensor_tensor partition-dim broadcast in NKI.
    m_bs = mask.reshape(B, S)
    for b in range(B):
        m_ss = m_bs[b].reshape(1, S).expand(S, S).contiguous()  # [S, S]
        for h in range(H):
            qh = q[b, h].contiguous()
            kh = k[b, h].contiguous()
            vh = v[b, h].contiguous()
            out[b, h] = fused_attention_kernel(qh, kh, vh, m_ss)
    return out

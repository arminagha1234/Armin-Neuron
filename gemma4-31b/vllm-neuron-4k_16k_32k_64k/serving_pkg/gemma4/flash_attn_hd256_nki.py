import torch
import torch.nn.functional as F


def flash_attention_hd256(q, k, v, scale=1.0):
    """Split-K flash attention for head_dim=256. torch.compile safe (no branches).

    q: [Nh, T, 256], k: [Nkv, T, 256], v: [Nkv, T, 256]
    Returns: [Nh, T, 256]
    """
    # Split head_dim into two 128-dim halves
    q_lo = q[:, :, :128]
    q_hi = q[:, :, 128:]
    k_lo = k[:, :, :128]
    k_hi = k[:, :, 128:]

    # Split-K: score = Q_lo @ K_lo^T + Q_hi @ K_hi^T
    scores = torch.matmul(q_lo, k_lo.transpose(-2, -1)) + torch.matmul(q_hi, k_hi.transpose(-2, -1))
    scores = scores * scale

    # Causal mask via additive bias (always applied, no branch)
    T = q.shape[1]
    mask = torch.triu(torch.ones(T, T, dtype=scores.dtype, device=scores.device), diagonal=1) * (-1e9)
    scores = scores + mask.unsqueeze(0)

    # Softmax
    attn_weights = torch.softmax(scores, dim=-1)

    # Output: split V into halves
    v_lo = v[:, :, :128]
    v_hi = v[:, :, 128:]
    out_lo = torch.matmul(attn_weights, v_lo)
    out_hi = torch.matmul(attn_weights, v_hi)

    return torch.cat([out_lo, out_hi], dim=-1)


# Alias for import compatibility
gemma4_flash_attention = flash_attention_hd256

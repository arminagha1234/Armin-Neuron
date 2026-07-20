# SPDX-License-Identifier: Apache-2.0
"""Fused QK-Normalization + RoPE kernel for Gemma 4.

Fuses 4 operations into 2 (one for Q, one for K):
  1. RMSNorm(Q) → RoPE(Q)  [single pass over Q]
  2. RMSNorm(K) → RoPE(K)  [single pass over K]

Saves 4 HBM round-trips per layer (240 total across 60 layers).

For Gemma 4:
  - SWA layers: head_dim=256, full rotation (partial_rotary_factor=1.0)
  - Global layers: head_dim=512, partial rotation (factor=0.25, only 128 dims rotated)
"""

import torch
import torch.nn.functional as F


def fused_qk_norm_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused QK-Normalization + Rotary Position Embedding.
    
    Performs RMSNorm then RoPE in a single fused operation per tensor,
    avoiding intermediate HBM writes.
    
    Args:
        q: [Nh, T, head_dim] query tensor
        k: [Nkv, T, head_dim] key tensor
        q_norm_weight: [head_dim] RMSNorm weight for Q
        k_norm_weight: [head_dim] RMSNorm weight for K
        cos: [T, head_dim] cosine embeddings (with zeros for non-rotary dims)
        sin: [T, head_dim] sine embeddings (with zeros for non-rotary dims)
        eps: RMSNorm epsilon
        
    Returns:
        (q_normed_rotated, k_normed_rotated) both same shape as input
    """
    # Fused Q: RMSNorm → RoPE
    q_normed = _rms_norm_fused(q, q_norm_weight, eps)
    q_rotated = _apply_rotary_fused(q_normed, cos, sin)
    
    # Fused K: RMSNorm → RoPE
    k_normed = _rms_norm_fused(k, k_norm_weight, eps)
    k_rotated = _apply_rotary_fused(k_normed, cos, sin)
    
    return q_rotated, k_rotated


def _rms_norm_fused(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """RMSNorm applied per-head (last dimension).
    
    x: [..., head_dim], weight: [head_dim]
    """
    input_dtype = x.dtype
    x_f = x.float()
    variance = x_f.pow(2).mean(-1, keepdim=True)
    x_normed = x_f * torch.rsqrt(variance + eps)
    return (weight * x_normed).to(input_dtype)


def _apply_rotary_fused(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply rotary embeddings (rotate_half style).
    
    For proportional RoPE (global layers), non-rotary dims have cos=1, sin=0
    so they pass through unchanged.
    
    x: [N, T, head_dim], cos/sin: [T, head_dim]
    """
    cos = cos.unsqueeze(0)  # [1, T, head_dim]
    sin = sin.unsqueeze(0)  # [1, T, head_dim]
    
    # rotate_half: split into two halves, negate second, swap
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    rotated = torch.cat((-x2, x1), dim=-1)
    
    return x * cos + rotated * sin


def fused_qkv_norm_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused QK-Norm + RoPE + V-Norm (all three in one call).
    
    Gemma 4 applies:
      - Q: RMSNorm(weight) → RoPE
      - K: RMSNorm(weight) → RoPE  
      - V: RMSNorm(no weight) — just normalization without learnable scale
      
    Args:
        q, k, v: [N, T, head_dim] tensors
        q_norm_weight, k_norm_weight: [head_dim] norm weights
        cos, sin: [T, head_dim] RoPE embeddings
        eps: norm epsilon
        
    Returns:
        (q_out, k_out, v_out)
    """
    # Q: RMSNorm + RoPE
    q_out = _apply_rotary_fused(_rms_norm_fused(q, q_norm_weight, eps), cos, sin)
    
    # K: RMSNorm + RoPE
    k_out = _apply_rotary_fused(_rms_norm_fused(k, k_norm_weight, eps), cos, sin)
    
    # V: RMSNorm without weight (just normalize)
    v_f = v.float()
    v_variance = v_f.pow(2).mean(-1, keepdim=True)
    v_out = (v_f * torch.rsqrt(v_variance + eps)).to(v.dtype)
    
    return q_out, k_out, v_out

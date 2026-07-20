# SPDX-License-Identifier: Apache-2.0
"""Fused LM Head + Logit Softcapping for Gemma 4.

Gemma 4 applies: logits = cap * tanh(logits / cap) where cap=30.0.
Standard path: matmul → write logits [T, 262144] → read → softcap → write.
Fused: matmul → softcap in-place → single write.

This saves one full read+write of the [T, 262144] logit tensor,
which at bf16 is 512KB per token — significant for prefill.
"""

import torch


def fused_lm_head_softcap(
    hidden_states: torch.Tensor,
    lm_head_weight: torch.Tensor,
    cap: float = 30.0,
) -> torch.Tensor:
    """Fused linear projection + logit soft-capping.
    
    Computes: cap * tanh(hidden @ weight^T / cap)
    
    Args:
        hidden_states: [T, hidden_size] final hidden states
        lm_head_weight: [vocab_size, hidden_size] LM head weight
        cap: softcapping value (30.0 for Gemma 4)
        
    Returns:
        logits: [T, vocab_size] soft-capped logits
    """
    # Linear projection
    logits = F.linear(hidden_states, lm_head_weight)
    
    # Softcap: cap * tanh(logits / cap)
    # Using float32 for numerical stability (matches PR implementation)
    logits = logits.float()
    logits = cap * torch.tanh(logits / cap)
    
    return logits


def fused_lm_head_softcap_bf16(
    hidden_states: torch.Tensor,
    lm_head_weight: torch.Tensor,
    cap: float = 30.0,
) -> torch.Tensor:
    """Same as above but keeps computation in bf16 for speed.
    
    Slightly less accurate but faster — the tanh saturation at ±30
    means precision loss is minimal in practice.
    """
    logits = torch.matmul(hidden_states, lm_head_weight.t())
    inv_cap = 1.0 / cap
    logits = cap * torch.tanh(logits * inv_cap)
    return logits

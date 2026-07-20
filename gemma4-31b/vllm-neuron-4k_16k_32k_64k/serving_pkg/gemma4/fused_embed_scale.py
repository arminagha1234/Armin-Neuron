# SPDX-License-Identifier: Apache-2.0
"""Fused Embedding + Scale for Gemma 4.

Gemma 4 scales embeddings by sqrt(hidden_size) = sqrt(5376) ≈ 73.32.
Standard path: embed_tokens(ids) → write to HBM → read → multiply → write.
Fused: embed_tokens(ids) * scale → single write.
"""

import torch
import torch.nn.functional as F


def fused_embedding_scale(
    input_ids: torch.Tensor,
    embedding_weight: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Fused embedding lookup + scalar multiplication.
    
    Args:
        input_ids: [T] token IDs
        embedding_weight: [vocab_size, hidden_size] embedding table
        scale: scalar multiplier (sqrt(hidden_size) for Gemma 4)
        
    Returns:
        scaled_embeddings: [T, hidden_size]
    """
    # Single fused operation — torch.compile keeps the multiply
    # in registers without a separate HBM write for the raw embeddings
    embeddings = F.embedding(input_ids, embedding_weight)
    return embeddings * scale

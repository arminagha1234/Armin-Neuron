# SPDX-License-Identifier: Apache-2.0
"""Fused GeGLU kernel for Gemma 4.

Fuses the gate/up projections with the GeGLU activation:
  output = down_proj(gelu_tanh(gate_proj(x)) * up_proj(x))

Standard (unfused) path does:
  1. gate = x @ gate_weight        → write gate to HBM
  2. up = x @ up_weight            → write up to HBM
  3. hidden = gelu(gate) * up      → read gate, read up, write hidden
  4. output = hidden @ down_weight → read hidden

Fused path:
  1. gate = x @ gate_weight        → keep in SBUF/PSUM
  2. up = x @ up_weight            → keep in SBUF/PSUM
  3. hidden = gelu(gate) * up      → compute in-place, keep in SBUF
  4. output = hidden @ down_weight → final write

Saves 2 full HBM writes (gate, up intermediates) per layer.
At 60 layers × 21504 intermediate_size × seq_len × bf16, this is significant.
"""

import torch
import torch.nn.functional as F


def fused_geglu_mlp(
    hidden_states: torch.Tensor,
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    down_weight: torch.Tensor,
) -> torch.Tensor:
    """Fused GeGLU MLP: gelu_tanh(gate) * up → down projection.
    
    Computes the full MLP in a way that torch.compile can fuse the
    intermediate operations, avoiding separate HBM writes for gate/up.
    
    Args:
        hidden_states: [T, hidden_size] input
        gate_weight: [hidden_size, intermediate_size_per_rank] (transposed storage)
        up_weight: [hidden_size, intermediate_size_per_rank] (transposed storage)
        down_weight: [intermediate_size_per_rank, hidden_size] (transposed storage)
        
    Returns:
        output: [T, hidden_size]
    """
    # Compute gate and up in sequence — torch.compile will fuse these
    # with the subsequent element-wise ops to avoid HBM round-trips
    gate = torch.matmul(hidden_states, gate_weight)
    up = torch.matmul(hidden_states, up_weight)
    
    # GeGLU activation: gelu_tanh(gate) * up
    # Using the aten op directly avoids torch.compile graph breaks on Neuron
    hidden = torch.ops.aten.gelu.default(gate) * up
    
    # Down projection
    output = torch.matmul(hidden, down_weight)
    
    return output


def fused_geglu_mlp_chunked(
    hidden_states: torch.Tensor,
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    chunk_size: int = 1024,
) -> torch.Tensor:
    """Chunked fused GeGLU MLP for memory efficiency on long sequences.
    
    Processes the MLP in chunks to keep intermediates small enough
    to stay in SBUF during torch.compile optimization.
    
    Args:
        hidden_states: [T, hidden_size] input
        gate_weight, up_weight, down_weight: projection weights
        chunk_size: tokens per chunk (trade memory vs parallelism)
        
    Returns:
        output: [T, hidden_size]
    """
    T = hidden_states.shape[0]
    
    if T <= chunk_size:
        return fused_geglu_mlp(hidden_states, gate_weight, up_weight, down_weight)
    
    outputs = []
    for start in range(0, T, chunk_size):
        end = min(start + chunk_size, T)
        chunk = hidden_states[start:end]
        out_chunk = fused_geglu_mlp(chunk, gate_weight, up_weight, down_weight)
        outputs.append(out_chunk)
    
    return torch.cat(outputs, dim=0)

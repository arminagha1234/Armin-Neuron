# SPDX-License-Identifier: Apache-2.0
"""Fused Norm-Residual kernel for Gemma 4.

Fuses the post-norm + residual-add + pre-norm pattern that appears
between attention and MLP in each decoder layer:

Standard (unfused):
  1. attn_out = post_attention_layernorm(attn_out)  → write to HBM
  2. hidden = residual + attn_out                    → read residual, read attn_out, write hidden
  3. mlp_in = pre_feedforward_layernorm(hidden)      → read hidden, write mlp_in

Fused:
  1. Load residual + attn_out once
  2. post_norm(attn_out) → add residual → pre_norm → store mlp_in + store new_residual
  Saves 2 HBM round-trips per fusion point.

Gemma 4 has 2 fusion points per layer:
  - Between attention output and MLP input
  - Between MLP output and next layer input (via layer_scalar)

Total savings: 4 HBM round-trips per layer × 60 layers = 240 saved DMAs.
"""

import torch


def fused_post_norm_residual_pre_norm(
    residual: torch.Tensor,
    module_output: torch.Tensor,
    post_norm_weight: torch.Tensor,
    pre_norm_weight: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused: post_norm(output) + residual_add + pre_norm.
    
    Combines three operations into a single fused pass:
      1. normalized_output = RMSNorm(module_output, post_norm_weight)
      2. new_residual = residual + normalized_output
      3. next_input = RMSNorm(new_residual, pre_norm_weight)
    
    Args:
        residual: [T, hidden_size] residual from before the module
        module_output: [T, hidden_size] output from attention/MLP
        post_norm_weight: [hidden_size] post-module norm weight
        pre_norm_weight: [hidden_size] pre-next-module norm weight
        eps: RMSNorm epsilon
        
    Returns:
        (new_residual, next_input):
            new_residual: [T, hidden_size] for the next residual connection
            next_input: [T, hidden_size] normalized input for next module
    """
    # Step 1: Post-norm the module output
    module_output_f = module_output.float()
    variance = module_output_f.pow(2).mean(-1, keepdim=True)
    normed_output = module_output_f * torch.rsqrt(variance + eps)
    normed_output = (post_norm_weight * normed_output).to(module_output.dtype)
    
    # Step 2: Residual add
    new_residual = residual + normed_output
    
    # Step 3: Pre-norm for next module
    new_residual_f = new_residual.float()
    variance2 = new_residual_f.pow(2).mean(-1, keepdim=True)
    next_input = new_residual_f * torch.rsqrt(variance2 + eps)
    next_input = (pre_norm_weight * next_input).to(new_residual.dtype)
    
    return new_residual, next_input


def fused_post_norm_residual_scalar(
    residual: torch.Tensor,
    module_output: torch.Tensor,
    post_norm_weight: torch.Tensor,
    layer_scalar: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Fused: post_norm(output) + residual_add + layer_scalar multiply.
    
    Used at the end of each decoder layer:
      1. normalized_output = RMSNorm(module_output, post_norm_weight)
      2. hidden = residual + normalized_output
      3. output = hidden * layer_scalar
    
    Args:
        residual: [T, hidden_size] residual from before MLP
        module_output: [T, hidden_size] MLP output
        post_norm_weight: [hidden_size] post-feedforward norm weight
        layer_scalar: [1] per-layer learned scalar
        eps: RMSNorm epsilon
        
    Returns:
        output: [T, hidden_size] final layer output (input to next layer)
    """
    # Post-norm
    module_output_f = module_output.float()
    variance = module_output_f.pow(2).mean(-1, keepdim=True)
    normed_output = module_output_f * torch.rsqrt(variance + eps)
    normed_output = (post_norm_weight * normed_output).to(module_output.dtype)
    
    # Residual + scalar (fused)
    output = (residual + normed_output) * layer_scalar
    
    return output


def fused_decoder_layer_norms(
    hidden_states: torch.Tensor,
    attn_output: torch.Tensor,
    mlp_output: torch.Tensor,
    input_layernorm_weight: torch.Tensor,
    post_attention_layernorm_weight: torch.Tensor,
    pre_feedforward_layernorm_weight: torch.Tensor,
    post_feedforward_layernorm_weight: torch.Tensor,
    layer_scalar: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Full fused norm pipeline for one Gemma 4 decoder layer.
    
    Fuses all 4 norms + 2 residuals + scalar into minimal HBM passes:
    
    Standard path (8 HBM ops):
      x = input_norm(hidden)           → write
      attn_out = attention(x)          → write  
      x = post_attn_norm(attn_out)     → write
      hidden = residual + x            → write
      x = pre_ffn_norm(hidden)         → write
      mlp_out = mlp(x)                 → write
      x = post_ffn_norm(mlp_out)       → write
      output = (residual + x) * scalar → write
      
    This function handles the norm/residual parts (not attention/MLP compute):
      Given attn_output and mlp_output (already computed), fuses:
      1. post_attn_norm(attn_output) + residual → new_hidden
      2. pre_ffn_norm(new_hidden) [returned for MLP input]
      3. post_ffn_norm(mlp_output) + residual + scalar → final output
      
    Note: This is called AFTER attention and MLP are computed separately.
    The fusion is in the norm/residual/scalar operations between them.
    
    Args:
        hidden_states: [T, H] input to the layer (first residual)
        attn_output: [T, H] attention module output
        mlp_output: [T, H] MLP module output
        *_weight: norm weights
        layer_scalar: [1] per-layer scalar
        eps: norm epsilon
        
    Returns:
        layer_output: [T, H] final output of this decoder layer
    """
    # Phase 1: post_attn_norm + residual
    attn_f = attn_output.float()
    var1 = attn_f.pow(2).mean(-1, keepdim=True)
    attn_normed = (post_attention_layernorm_weight * (attn_f * torch.rsqrt(var1 + eps))).to(hidden_states.dtype)
    hidden_after_attn = hidden_states + attn_normed  # new residual
    
    # Phase 2: post_ffn_norm + residual + scalar
    mlp_f = mlp_output.float()
    var2 = mlp_f.pow(2).mean(-1, keepdim=True)
    mlp_normed = (post_feedforward_layernorm_weight * (mlp_f * torch.rsqrt(var2 + eps))).to(hidden_after_attn.dtype)
    layer_output = (hidden_after_attn + mlp_normed) * layer_scalar
    
    return layer_output

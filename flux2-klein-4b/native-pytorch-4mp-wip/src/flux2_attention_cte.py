"""Step 1.1 — `attention_cte` NKI kernel wrapper for FLUX.2-klein-4B.

This module provides a drop-in replacement for the SDPA call inside
diffusers' `Flux2AttnProcessor.__call__`. It calls NxDI's
`attention_cte` NKI kernel (the one used in production by AWS for
FLUX.1) instead of `dispatch_attention_fn(...)` → `F.scaled_dot_product_attention`.

Why bother (vs the v3 wrap_nki test that was 11% slower):
- v3 was tested on a different shape; klein-4B has seq=4608, head_dim=128
- The kernel uses LNC=2 sharding when invoked as `attention_cte[2](...)`
- This is the single-rank variant — no TP=4 plumbing required, no NxD
  parallel-state setup. Tests whether the kernel alone wins at this
  shape, before committing to the full TP=4 lift (Step 1.2).

If the kernel is faster: ship it as Step 1.1 (kernel-only).
If the kernel is slower or equal: skip it, go straight to Step 1.2
(TP=4 with native-PyTorch parallelize_module).

Layout note:
- diffusers Flux2 attention runs at `[B, S, H, D]` (sequence_dim=1)
- attention_cte wants `[B*H, S, D]`
- This wrapper handles the permute/reshape boundary
"""
from __future__ import annotations

import math
import os

import torch
import torch.nn.functional as F


# Lazily import the kernel: it's only available when nkilib is on path.
_KERNEL = None
_KERNEL_SHARDED = None  # wrap_nki-wrapped, callable from compiled graph


def _get_kernel():
    """Return the LNC=2-sharded `attention_cte[2]` wrapped via wrap_nki.

    Beta 3 native: `torch_neuronx.nki_hop.wrap_nki` is the bridge that
    makes a NKI kernel traceable by torch.compile. Bare kernel calls
    fail with "'Kernel' object has no attribute 'func'" inside Dynamo.
    """
    global _KERNEL, _KERNEL_SHARDED
    if _KERNEL_SHARDED is not None:
        return _KERNEL_SHARDED
    from nkilib.core.attention.attention_cte import attention_cte
    from torch_neuronx.nki_hop import wrap_nki
    _KERNEL = attention_cte
    # attention_cte[2] is the LNC=2 sharded grid variant.
    _KERNEL_SHARDED = wrap_nki(attention_cte[2])
    return _KERNEL_SHARDED


def _kernel_call(q, k, v, scale: float):
    """Invoke attention_cte with klein-4B-appropriate flags.

    Args:
        q, k, v: [B*H, S, D] bf16 contiguous tensors
        scale:   1 / sqrt(D)

    Returns:
        [B*H, S, D] bf16 output

    Always uses the LNC=2 sharded grid variant (`attention_cte[2]`)
    via `wrap_nki`. Requires `NEURON_RT_VIRTUAL_CORE_SIZE=2`.
    """
    kernel = _get_kernel()

    # FLUX is bidirectional → causal_mask=False.
    # tp_q=True, tp_k=True lets the kernel handle internal Q/K transposes
    # via DMA, which is faster than user-side permutes.
    # tp_out=False keeps output in [B*H, S, D] layout.
    return kernel(
        q=q, k=k, v=v, scale=scale,
        causal_mask=False,
        tp_q=True,
        tp_k=True,
        tp_out=False,
    )


def attention_cte_flux2(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Drop-in replacement for `dispatch_attention_fn(...)` in
    `Flux2AttnProcessor`.

    Args:
        query, key, value: shape `[B, S, H, D]` bf16 (diffusers Flux2 layout,
            sequence_dim=1, head_dim last)
        attn_mask: optional additive mask (klein-4B uses None in the
            standard path; if provided, falls back to SDPA — the kernel
            doesn't support arbitrary additive masks)

    Returns:
        `[B, S, H, D]` bf16 output, same layout as input.

    Falls back to PyTorch SDPA when:
      - device is CPU (compile-time tracing, fake-tensor mode)
      - head_dim != 128 (kernel constraint)
      - attn_mask is provided (kernel doesn't support arbitrary mask)
    """
    B, S, H, D = query.shape
    on_neuron = query.device.type == "neuron"
    use_kernel = (
        on_neuron
        and D == 128
        and attn_mask is None
    )
    if not use_kernel:
        # Fallback: standard SDPA. Need [B, H, S, D] for SDPA.
        q_sdpa = query.permute(0, 2, 1, 3).contiguous()
        k_sdpa = key.permute(0, 2, 1, 3).contiguous()
        v_sdpa = value.permute(0, 2, 1, 3).contiguous()
        out = F.scaled_dot_product_attention(q_sdpa, k_sdpa, v_sdpa, attn_mask=attn_mask)
        # Back to [B, S, H, D]
        return out.permute(0, 2, 1, 3).contiguous()

    # --- Kernel path ---
    # Klein-4B: layout is [B, S, H, D]. Kernel needs [B*H, S, D].
    # permute first to [B, H, S, D], then reshape to [B*H, S, D].
    q = query.permute(0, 2, 1, 3).reshape(B * H, S, D).contiguous()
    k = key.permute(0, 2, 1, 3).reshape(B * H, S, D).contiguous()
    v = value.permute(0, 2, 1, 3).reshape(B * H, S, D).contiguous()

    scale = 1.0 / math.sqrt(D)
    out = _kernel_call(q, k, v, scale)

    # Back to [B, S, H, D]
    out = out.reshape(B, H, S, D).permute(0, 2, 1, 3).contiguous()
    return out


def install_attention_cte_processor(transformer):
    """Install the kernel-backed attention into a diffusers Flux2 transformer.

    Walks the transformer's modules and replaces every `Flux2AttnProcessor`
    instance's `__call__` to dispatch through `attention_cte_flux2` instead
    of `dispatch_attention_fn`.

    Idempotent. Safe to call before or after `apply_neuron_patches`.
    """
    from diffusers.models.transformers.transformer_flux2 import (
        Flux2AttnProcessor,
    )

    # We patch the class, not each instance, so all blocks get the
    # kernel call. (Each block has its own processor instance, so a
    # module-walk replacement also works — class patch is simpler.)

    if getattr(Flux2AttnProcessor, "_attention_cte_installed", False):
        return  # idempotent

    orig_call = Flux2AttnProcessor.__call__

    def patched_call(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        image_rotary_emb=None,
    ):
        # Reproduce the original processor's pre-attention work, swap
        # only the attention compute, and reproduce the post-attention
        # work. Necessary because we can't intercept `dispatch_attention_fn`
        # cleanly (it's a function, not a method).
        from diffusers.models.transformers.transformer_flux2 import (
            _get_qkv_projections,
        )
        from diffusers.models.embeddings import apply_rotary_emb

        query, key, value, encoder_query, encoder_key, encoder_value = (
            _get_qkv_projections(attn, hidden_states, encoder_hidden_states)
        )

        query = query.unflatten(-1, (attn.heads, -1))
        key = key.unflatten(-1, (attn.heads, -1))
        value = value.unflatten(-1, (attn.heads, -1))

        query = attn.norm_q(query)
        key = attn.norm_k(key)

        if attn.added_kv_proj_dim is not None:
            encoder_query = encoder_query.unflatten(-1, (attn.heads, -1))
            encoder_key = encoder_key.unflatten(-1, (attn.heads, -1))
            encoder_value = encoder_value.unflatten(-1, (attn.heads, -1))

            encoder_query = attn.norm_added_q(encoder_query)
            encoder_key = attn.norm_added_k(encoder_key)

            query = torch.cat([encoder_query, query], dim=1)
            key = torch.cat([encoder_key, key], dim=1)
            value = torch.cat([encoder_value, value], dim=1)

        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
            key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)

        # ---- Kernel-backed attention compute ----
        hidden_states_out = attention_cte_flux2(query, key, value, attn_mask=attention_mask)

        hidden_states_out = hidden_states_out.flatten(2, 3)
        hidden_states_out = hidden_states_out.to(query.dtype)

        if encoder_hidden_states is not None:
            encoder_hidden_states_out, hidden_states_out = (
                hidden_states_out.split_with_sizes(
                    [
                        encoder_hidden_states.shape[1],
                        hidden_states_out.shape[1] - encoder_hidden_states.shape[1],
                    ],
                    dim=1,
                )
            )
            encoder_hidden_states_out = attn.to_add_out(encoder_hidden_states_out)

        hidden_states_out = attn.to_out[0](hidden_states_out)
        hidden_states_out = attn.to_out[1](hidden_states_out)

        if encoder_hidden_states is not None:
            return hidden_states_out, encoder_hidden_states_out
        return hidden_states_out

    Flux2AttnProcessor.__call__ = patched_call
    Flux2AttnProcessor._attention_cte_installed = True
    Flux2AttnProcessor._original_call = orig_call

    # Also patch the single-stream parallel self-attention processor.
    _install_parallel_processor()


def _install_parallel_processor():
    """Patch Flux2ParallelSelfAttnProcessor (single-stream fused block) to
    use attention_cte for the attention compute.
    """
    from diffusers.models.transformers.transformer_flux2 import (
        Flux2ParallelSelfAttnProcessor,
    )
    from diffusers.models.embeddings import apply_rotary_emb

    if getattr(Flux2ParallelSelfAttnProcessor, "_attention_cte_installed", False):
        return
    orig = Flux2ParallelSelfAttnProcessor.__call__

    def patched_parallel_call(self, attn, hidden_states, attention_mask=None,
                              image_rotary_emb=None):
        # Fused QKV + MLP-in projection
        hs = attn.to_qkv_mlp_proj(hidden_states)
        qkv, mlp_hidden_states = torch.split(
            hs,
            [3 * attn.inner_dim, attn.mlp_hidden_dim * attn.mlp_mult_factor],
            dim=-1,
        )
        query, key, value = qkv.chunk(3, dim=-1)
        query = query.unflatten(-1, (attn.heads, -1))
        key = key.unflatten(-1, (attn.heads, -1))
        value = value.unflatten(-1, (attn.heads, -1))
        query = attn.norm_q(query)
        key = attn.norm_k(key)
        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
            key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)

        # Kernel-backed attention (flash-tiled for long seq)
        attn_out = attention_cte_flux2(query, key, value, attn_mask=attention_mask)
        attn_out = attn_out.flatten(2, 3).to(query.dtype)

        # FF tail
        mlp_hidden_states = attn.mlp_act_fn(mlp_hidden_states)
        combined = torch.cat([attn_out, mlp_hidden_states], dim=-1)
        return attn.to_out(combined)

    Flux2ParallelSelfAttnProcessor.__call__ = patched_parallel_call
    Flux2ParallelSelfAttnProcessor._attention_cte_installed = True
    Flux2ParallelSelfAttnProcessor._original_call = orig


def restore_default_attention(transformer=None):
    """Undo install_attention_cte_processor (for A/B benching)."""
    from diffusers.models.transformers.transformer_flux2 import (
        Flux2AttnProcessor,
    )
    if getattr(Flux2AttnProcessor, "_attention_cte_installed", False):
        Flux2AttnProcessor.__call__ = Flux2AttnProcessor._original_call
        Flux2AttnProcessor._attention_cte_installed = False

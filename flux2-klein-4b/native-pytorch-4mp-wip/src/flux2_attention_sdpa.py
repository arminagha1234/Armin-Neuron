"""SDPA-based attention for FLUX.2 under TP+v3.

The Neuron compiler can lower a single F.scaled_dot_product_attention
call to a native flash kernel. The pure-Python tile-based manual flash
was correctness-first and is the speed bottleneck (227-579s warm).
With v3 full sharding + fp32 active, precision is no longer an issue
and we can swap to SDPA for a large speedup.

Layout: same as manual flash — input [B, S, H, D], output [B, S, H, D].
SDPA wants [B, H, S, D] internally; permute round-trips.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def sdpa_attention(query, key, value, attn_mask=None):
    """[B, S, H, D] -> [B, S, H, D] via SDPA."""
    q = query.permute(0, 2, 1, 3).contiguous()  # [B,H,S,D]
    k = key.permute(0, 2, 1, 3).contiguous()
    v = value.permute(0, 2, 1, 3).contiguous()
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
    return out.permute(0, 2, 1, 3).contiguous()


def install_sdpa_processor():
    """Patch Flux2AttnProcessor + Flux2ParallelSelfAttnProcessor to use
    SDPA (single op, Neuron-fusible)."""
    from diffusers.models.transformers.transformer_flux2 import (
        Flux2AttnProcessor, Flux2ParallelSelfAttnProcessor,
    )
    from diffusers.models.embeddings import apply_rotary_emb

    if getattr(Flux2AttnProcessor, "_sdpa_installed", False):
        return

    def patched_call(
        self, attn, hidden_states, encoder_hidden_states=None,
        attention_mask=None, image_rotary_emb=None,
    ):
        from diffusers.models.transformers.transformer_flux2 import (
            _get_qkv_projections,
        )
        q, k, v, eq, ek, ev = _get_qkv_projections(
            attn, hidden_states, encoder_hidden_states)

        q = q.unflatten(-1, (attn.heads, -1))
        k = k.unflatten(-1, (attn.heads, -1))
        v = v.unflatten(-1, (attn.heads, -1))
        q = attn.norm_q(q)
        k = attn.norm_k(k)

        if attn.added_kv_proj_dim is not None:
            eq = eq.unflatten(-1, (attn.heads, -1))
            ek = ek.unflatten(-1, (attn.heads, -1))
            ev = ev.unflatten(-1, (attn.heads, -1))
            eq = attn.norm_added_q(eq)
            ek = attn.norm_added_k(ek)
            q = torch.cat([eq, q], dim=1)
            k = torch.cat([ek, k], dim=1)
            v = torch.cat([ev, v], dim=1)

        if image_rotary_emb is not None:
            q = apply_rotary_emb(q, image_rotary_emb, sequence_dim=1)
            k = apply_rotary_emb(k, image_rotary_emb, sequence_dim=1)

        out = sdpa_attention(q, k, v, attn_mask=attention_mask)
        out = out.flatten(2, 3).to(q.dtype)

        if encoder_hidden_states is not None:
            eo, out = out.split_with_sizes(
                [encoder_hidden_states.shape[1],
                 out.shape[1] - encoder_hidden_states.shape[1]],
                dim=1)
            eo = attn.to_add_out(eo)

        out = attn.to_out[0](out)
        out = attn.to_out[1](out)

        if encoder_hidden_states is not None:
            return out, eo
        return out

    Flux2AttnProcessor.__call__ = patched_call
    Flux2AttnProcessor._sdpa_installed = True

    if not getattr(Flux2ParallelSelfAttnProcessor, "_sdpa_installed", False):
        # The single-stream block uses the v3-installed processor; we
        # override the v3 processor here too so it uses SDPA.
        def patched_parallel_call(
            self, attn, hidden_states, attention_mask=None,
            image_rotary_emb=None,
        ):
            if not getattr(attn, "_v3_split", False):
                # fallback to fused path (shouldn't happen post-v3)
                from diffusers.models.transformers.transformer_flux2 import (
                    Flux2ParallelSelfAttnProcessor as P,
                )
                return P.__call__.__wrapped__(self, attn, hidden_states,
                                              attention_mask, image_rotary_emb)

            q = attn.to_q_s(hidden_states)
            k = attn.to_k_s(hidden_states)
            v = attn.to_v_s(hidden_states)
            g = attn.mlp_gate(hidden_states)
            m = attn.mlp_value(hidden_states)

            q = q.unflatten(-1, (attn.heads, -1))
            k = k.unflatten(-1, (attn.heads, -1))
            v = v.unflatten(-1, (attn.heads, -1))
            q = attn.norm_q(q)
            k = attn.norm_k(k)
            if image_rotary_emb is not None:
                q = apply_rotary_emb(q, image_rotary_emb, sequence_dim=1)
                k = apply_rotary_emb(k, image_rotary_emb, sequence_dim=1)

            attn_out = sdpa_attention(q, k, v, attn_mask=attention_mask)
            attn_out = attn_out.flatten(2, 3).to(q.dtype)

            mlp_out = attn._silu(g) * m

            out = attn.attn_out_proj(attn_out) + attn.mlp_out_proj(mlp_out)
            return out

        Flux2ParallelSelfAttnProcessor.__call__ = patched_parallel_call
        Flux2ParallelSelfAttnProcessor._sdpa_installed = True

    print("[sdpa] installed SDPA-based attention into Flux2 processors",
          flush=True)

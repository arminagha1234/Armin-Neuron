"""Manual flash attention for FLUX.2-klein-4B under TP.

Replaces the buggy `attention_cte` kernel under TP with a hand-written
tile-based attention that processes the sequence in chunks small enough
to fit in SBUF. Slower than a real flash kernel but **correctness-first**.

Once we verify TP gives matching output (std~=18 vs single-core 18.16),
we can swap this back for `attention_cte` with a fixed wrapper, or write
a proper NKI flash attention.

Mathematics: standard attention computed in tiles along the sequence dim.
For each Q tile, we compute exp(Q @ K^T) @ V tile-by-tile and combine
with the running softmax denominator.
"""
from __future__ import annotations

import math
import torch
import torch.nn.functional as F

# When True, manual_flash_attention upcasts q/k/v to fp32 for the QK and
# PV matmuls (the score tiles are bounded by tile size, so memory stays
# safe). This pushes past the bf16-matmul precision ceiling at high res
# without the full-fp32 activation OOM.
COMPUTE_FP32 = False

# Flash tile sizes along the sequence dim. At very high token counts
# (4MP ≈ 65K tokens) a small tile means many online-softmax accumulation
# steps (65K/1024 = 64 tiles), which can degrade. Larger tiles = fewer
# accumulation steps. Override via set_tile_size().
TILE_Q = 1024
TILE_KV = 1024

# When True, skip the Python tile loop entirely and dispatch to
# F.scaled_dot_product_attention, which on Neuron Beta3 maps to a fused
# flash kernel (orders of magnitude faster than the Python loop). Keeps
# whatever dtype q/k/v are in (fp32 for the high-res correct path).
USE_SDPA = False


def set_compute_fp32(enabled: bool):
    global COMPUTE_FP32
    COMPUTE_FP32 = bool(enabled)


def set_tile_size(q: int, kv: int):
    global TILE_Q, TILE_KV
    TILE_Q, TILE_KV = int(q), int(kv)


def set_use_sdpa(enabled: bool):
    global USE_SDPA
    USE_SDPA = bool(enabled)


def manual_flash_attention(
    query: torch.Tensor,    # [B, S, H, D]
    key: torch.Tensor,      # [B, S_kv, H, D]
    value: torch.Tensor,    # [B, S_kv, H, D]
    attn_mask: torch.Tensor | None = None,
    tile_size_q: int | None = None,
    tile_size_kv: int | None = None,
) -> torch.Tensor:
    """Tile-based flash-style attention.

    Returns: [B, S, H, D] — same layout as input.
    """
    if tile_size_q is None:
        tile_size_q = TILE_Q
    if tile_size_kv is None:
        tile_size_kv = TILE_KV
    B, S, H, D = query.shape

    # Fast path: dispatch to the fused SDPA flash kernel (no Python loop).
    if USE_SDPA:
        qf = query.permute(0, 2, 1, 3)   # [B,H,S,D]
        kf = key.permute(0, 2, 1, 3)
        vf = value.permute(0, 2, 1, 3)
        out = F.scaled_dot_product_attention(qf, kf, vf, attn_mask=attn_mask)
        return out.permute(0, 2, 1, 3).contiguous()
    B_kv, S_kv, H_kv, D_kv = key.shape
    assert D == D_kv, f"head_dim mismatch: {D} vs {D_kv}"
    assert H == H_kv, f"num_heads mismatch: {H} vs {H_kv}"
    assert B == B_kv

    # Permute to [B, H, S, D] for attention computation
    q = query.permute(0, 2, 1, 3).contiguous()  # [B, H, S, D]
    k = key.permute(0, 2, 1, 3).contiguous()    # [B, H, S_kv, D]
    v = value.permute(0, 2, 1, 3).contiguous()  # [B, H, S_kv, D]

    # Optional fp32 compute: upcast q/k/v so the QK and PV matmuls run in
    # fp32 (score tiles are bounded by tile size → memory-safe). Output is
    # downcast back to the input dtype at the end.
    mm_dtype = torch.float32 if COMPUTE_FP32 else q.dtype
    if COMPUTE_FP32:
        q = q.float()
        k = k.float()
        v = v.float()

    scale = 1.0 / math.sqrt(D)

    # Output accumulator
    out = torch.zeros_like(q)  # [B, H, S, D]

    # Tile along Q sequence dim
    for q_start in range(0, S, tile_size_q):
        q_end = min(q_start + tile_size_q, S)
        q_tile = q[:, :, q_start:q_end, :]  # [B, H, tq, D]

        # Running maxes / sums for online softmax across KV tiles
        # Init with very negative max so first tile takes over
        m_running = torch.full(
            (B, H, q_end - q_start, 1),
            float("-inf"),
            dtype=torch.float32,
            device=q.device,
        )
        l_running = torch.zeros(
            (B, H, q_end - q_start, 1),
            dtype=torch.float32,
            device=q.device,
        )
        out_tile = torch.zeros(
            (B, H, q_end - q_start, D),
            dtype=torch.float32,
            device=q.device,
        )

        for kv_start in range(0, S_kv, tile_size_kv):
            kv_end = min(kv_start + tile_size_kv, S_kv)
            k_tile = k[:, :, kv_start:kv_end, :]  # [B, H, tk, D]
            v_tile = v[:, :, kv_start:kv_end, :]  # [B, H, tk, D]

            # Compute attention scores for this Q-tile vs this K-tile
            # scores: [B, H, tq, tk]
            scores = torch.matmul(q_tile, k_tile.transpose(-2, -1)) * scale
            scores = scores.float()

            if attn_mask is not None:
                # We don't expect attn_mask in FLUX.2 standard path, but support it
                m_slice = attn_mask[..., q_start:q_end, kv_start:kv_end]
                scores = scores + m_slice

            # Online softmax update (flash attention style)
            m_tile = scores.amax(dim=-1, keepdim=True)        # [B, H, tq, 1]
            m_new = torch.maximum(m_running, m_tile)          # [B, H, tq, 1]
            alpha = torch.exp(m_running - m_new)              # rescale prior
            beta_scores = torch.exp(scores - m_new)           # current tile
            l_new = alpha * l_running + beta_scores.sum(
                dim=-1, keepdim=True
            )

            # Accumulate output: rescale prior + add current tile contribution
            out_tile = (
                alpha * out_tile
                + torch.matmul(beta_scores.to(v_tile.dtype), v_tile).float()
            )

            m_running = m_new
            l_running = l_new

        # Final normalization
        out_tile = out_tile / l_running
        out[:, :, q_start:q_end, :] = out_tile.to(q.dtype)

    # Back to [B, S, H, D]
    out = out.permute(0, 2, 1, 3).contiguous()
    return out


def install_manual_flash_processor(transformer=None):
    """Patch Flux2AttnProcessor + Flux2ParallelSelfAttnProcessor to use
    the manual flash attention instead of `dispatch_attention_fn`.

    Same install pattern as `flux2_attention_cte.install_attention_cte_processor`
    but with our manual flash for the actual compute step.
    """
    from diffusers.models.transformers.transformer_flux2 import (
        Flux2AttnProcessor,
        Flux2ParallelSelfAttnProcessor,
    )
    from diffusers.models.embeddings import apply_rotary_emb

    if getattr(Flux2AttnProcessor, "_manual_flash_installed", False):
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
        from diffusers.models.transformers.transformer_flux2 import (
            _get_qkv_projections,
        )

        q, k, v, eq, ek, ev = _get_qkv_projections(
            attn, hidden_states, encoder_hidden_states
        )

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

        # Manual flash attention compute
        out = manual_flash_attention(q, k, v, attn_mask=attention_mask)

        out = out.flatten(2, 3).to(q.dtype)

        if encoder_hidden_states is not None:
            eo, out = out.split_with_sizes(
                [encoder_hidden_states.shape[1],
                 out.shape[1] - encoder_hidden_states.shape[1]],
                dim=1,
            )
            eo = attn.to_add_out(eo)

        out = attn.to_out[0](out)
        out = attn.to_out[1](out)

        if encoder_hidden_states is not None:
            return out, eo
        return out

    Flux2AttnProcessor.__call__ = patched_call
    Flux2AttnProcessor._manual_flash_installed = True
    Flux2AttnProcessor._original_call = orig_call

    # Single-stream parallel processor
    if not getattr(Flux2ParallelSelfAttnProcessor, "_manual_flash_installed", False):
        orig_par = Flux2ParallelSelfAttnProcessor.__call__

        def patched_parallel_call(
            self, attn, hidden_states, attention_mask=None, image_rotary_emb=None,
        ):
            hs = attn.to_qkv_mlp_proj(hidden_states)
            qkv, mlp_hs = torch.split(
                hs,
                [3 * attn.inner_dim,
                 attn.mlp_hidden_dim * attn.mlp_mult_factor],
                dim=-1,
            )
            q, k, v = qkv.chunk(3, dim=-1)
            q = q.unflatten(-1, (attn.heads, -1))
            k = k.unflatten(-1, (attn.heads, -1))
            v = v.unflatten(-1, (attn.heads, -1))
            q = attn.norm_q(q)
            k = attn.norm_k(k)
            if image_rotary_emb is not None:
                q = apply_rotary_emb(q, image_rotary_emb, sequence_dim=1)
                k = apply_rotary_emb(k, image_rotary_emb, sequence_dim=1)

            attn_out = manual_flash_attention(q, k, v, attn_mask=attention_mask)
            attn_out = attn_out.flatten(2, 3).to(q.dtype)

            mlp_hs = attn.mlp_act_fn(mlp_hs)
            combined = torch.cat([attn_out, mlp_hs], dim=-1)
            return attn.to_out(combined)

        Flux2ParallelSelfAttnProcessor.__call__ = patched_parallel_call
        Flux2ParallelSelfAttnProcessor._manual_flash_installed = True
        Flux2ParallelSelfAttnProcessor._original_call = orig_par

    print("[manual_flash] installed manual flash attention "
          "into Flux2 attention processors", flush=True)

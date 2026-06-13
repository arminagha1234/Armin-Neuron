#!/usr/bin/env python3
"""
LTX-2 Optimized — Pre-Compiled NEFF Loading for Maximum Speed
==============================================================
Single-file script that loads PRE-COMPILED Neuron NEFFs for all three
components (Text Encoder + DiT Backbone + VAE Decoder) to achieve ~30s
generation vs 179s in eager mode.

Architecture:
  - Neuron Gemma-3 text encoder: 4 pre-compiled NEFF shards via torch.jit.load
  - Neuron DiT backbone: 1 pre-compiled NEFF via NxDI NeuronApplicationBase.load()
  - Neuron VAE decoder: 4 pre-compiled NEFF shards via torch.jit.load (tiled decode)

All components coexist on the same 4 NeuronCores and execute sequentially:
  text encoding → denoising (48 blocks) → VAE decode (tiled)

Launch (SINGLE process — TP handled internally by TensorParallelNeuronModel):
  NEURON_FUSE_SOFTMAX=1 NEURON_CUSTOM_SILU=1 NEURON_RT_STOCHASTIC_ROUNDING_EN=0 \\
    python neuron_ltx2_optimized.py

Prerequisites:
  1. Gemma3 compiled: /mnt/data/gemma3_compiled/tp_{0..3}.pt
  2. Gemma3 sharded: /mnt/data/gemma3_sharded/rank_{0..3}.pt
  3. DiT compiled:   /mnt/data/ltx2_compiled/model.pt (NxDI format)
  4. VAE compiled:   /mnt/data/ltx2_vae_compiled/tp_{0..3}.pt

Requirements:
  - Trainium2 instance (trn2.48xlarge)
  - NxDI venv with torch_neuronx, neuronx_distributed
  - diffusers >= 0.31.0 (with LTX2Pipeline support)
"""

import argparse
import gc
import json
import logging
import math
import os
import sys
import time
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# ─────────────────────────────────────────────────────────────────────────────
# Environment defaults
# ─────────────────────────────────────────────────────────────────────────────

os.environ.setdefault("NEURON_FUSE_SOFTMAX", "1")
os.environ.setdefault("NEURON_CUSTOM_SILU", "1")
os.environ.setdefault("NEURON_RT_STOCHASTIC_ROUNDING_EN", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 1. BMM-SDPA Replacement (for CPU preprocessing calls)
# ─────────────────────────────────────────────────────────────────────────────

_sdpa_original = None
_sdpa_replaced = False


def install_bmm_sdpa():
    """Replace F.scaled_dot_product_attention with explicit BMM+softmax for Neuron.

    Stock SDPA miscomputes on Neuron device. This replacement:
    - CPU calls (query.device.type == "cpu") fall through to original SDPA
    - Neuron calls use explicit torch.bmm(Q, K^T) * scale + mask → softmax → bmm(probs, V)
    - Handles 4D→3D reshape transparently
    """
    global _sdpa_replaced, _sdpa_original

    if _sdpa_replaced:
        return _sdpa_original

    _sdpa_original = torch.nn.functional.scaled_dot_product_attention

    def neuron_sdpa(
        query,
        key,
        value,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=False,
        scale=None,
        enable_gqa=False,
    ):
        # CPU fallback — use original SDPA (for text encoder preprocessing)
        if query.device.type == "cpu":
            return _sdpa_original(
                query,
                key,
                value,
                attn_mask=attn_mask,
                dropout_p=dropout_p,
                is_causal=is_causal,
                scale=scale,
            )

        # ── BMM path for Neuron device ──────────────────────────────────
        d = query.shape[-1]
        if scale is None:
            scale = 1.0 / math.sqrt(d)

        orig_shape = None
        if len(query.shape) == 4:
            orig_shape = query.shape
            b, h, sq, d_head = query.shape
            query = query.reshape(b * h, sq, d_head)
            key = key.reshape(b * h, -1, d_head)
            value = value.reshape(b * h, -1, d_head)
            if attn_mask is not None:
                if attn_mask.ndim == 4:
                    attn_mask = attn_mask.reshape(
                        b * h, attn_mask.shape[-2], attn_mask.shape[-1]
                    )
                elif attn_mask.ndim == 2:
                    attn_mask = attn_mask.unsqueeze(0)
                elif attn_mask.ndim == 3:
                    if attn_mask.shape[0] == orig_shape[0]:
                        attn_mask = (
                            attn_mask.unsqueeze(1)
                            .expand(orig_shape[0], orig_shape[1], -1, -1)
                            .reshape(
                                orig_shape[0] * orig_shape[1],
                                attn_mask.shape[-2],
                                attn_mask.shape[-1],
                            )
                        )

        # Explicit BMM attention
        scores = torch.bmm(query, key.transpose(-1, -2)) * scale

        if attn_mask is not None:
            scores = scores + attn_mask

        probs = scores.softmax(dim=-1)
        out = torch.bmm(probs, value)

        if orig_shape is not None:
            out = out.reshape(orig_shape[0], orig_shape[1], -1, orig_shape[3])

        return out

    torch.nn.functional.scaled_dot_product_attention = neuron_sdpa
    _sdpa_replaced = True
    logger.info("Installed BMM-based SDPA replacement for Neuron")
    return _sdpa_original


# ─────────────────────────────────────────────────────────────────────────────
# 2. Neuron Text Encoder Wrapper (Gemma-3, TP=4 pre-compiled NEFFs)
# ─────────────────────────────────────────────────────────────────────────────


class NeuronTextEncoderOutput:
    """Mimics the HuggingFace text encoder output with hidden_states tuple."""

    def __init__(self, hidden_states):
        self.hidden_states = hidden_states


class NeuronTextEncoderWrapper:
    """Drop-in replacement for Gemma3ForConditionalGeneration text encoder.

    Loads 4 pre-compiled NEFF shards + pre-sharded weights. The compiled model
    returns a stacked tensor of shape (1, 1024, 3840, num_hidden_states) which
    is unpacked into the hidden_states tuple expected by the pipeline.

    The pipeline calls:
        pipe.text_encoder(input_ids=..., attention_mask=..., output_hidden_states=True)
    """

    def __init__(self, compiled_gemma3, dtype=torch.bfloat16):
        self.compiled_model = compiled_gemma3
        self.dtype = dtype
        self._device = torch.device("cpu")
        self.config = type("Config", (), {"output_hidden_states": True})()

    def __call__(
        self, input_ids=None, attention_mask=None, output_hidden_states=True, **kwargs
    ):
        """Run compiled Gemma-3 text encoder.

        Args:
            input_ids: (B, seq_len) token IDs
            attention_mask: (B, seq_len) binary mask

        Returns:
            NeuronTextEncoderOutput with hidden_states tuple
        """
        with torch.no_grad():
            # Compiled model returns stacked hidden states: (1, seq, dim, num_states)
            stacked = self.compiled_model(input_ids, attention_mask)
            num_states = stacked.shape[-1]
            hidden_states = tuple(stacked[:, :, :, i] for i in range(num_states))
        return NeuronTextEncoderOutput(hidden_states=hidden_states)

    def eval(self):
        return self

    def to(self, *args, **kwargs):
        return self

    @property
    def device(self):
        return self._device


def load_neuron_gemma3(sharded_dir: str, compile_dir: str, tp_degree: int = 4):
    """Load TP=4 compiled Gemma3 encoder with pre-sharded weights.

    Args:
        sharded_dir: Directory containing rank_0.pt ... rank_3.pt (pre-sharded weights)
        compile_dir: Directory containing tp_0.pt ... tp_3.pt (compiled NEFFs)
        tp_degree: Tensor parallel degree (default 4)

    Returns:
        TensorParallelNeuronModel wrapping the 4 compiled shards
    """
    import torch_neuronx
    from neuronx_distributed.trace.trace import (
        TensorParallelNeuronModel,
        replace_weights,
    )

    models = []
    for rank in range(tp_degree):
        t0 = time.time()

        # Load pre-sharded weights for this rank
        rank_ckpt_path = os.path.join(sharded_dir, f"rank_{rank}.pt")
        if not os.path.exists(rank_ckpt_path):
            raise FileNotFoundError(
                f"Sharded weight not found: {rank_ckpt_path}\n"
                f"Run shard_gemma3_weights.py to generate sharded weights."
            )
        ckpt = torch.load(rank_ckpt_path, weights_only=True)

        # Load compiled NEFF (without loading to NeuronRT yet)
        neff_path = os.path.join(compile_dir, f"tp_{rank}.pt")
        if not os.path.exists(neff_path):
            raise FileNotFoundError(
                f"Compiled NEFF not found: {neff_path}\n"
                f"Run compile_gemma3.py to compile the text encoder."
            )
        with torch_neuronx.contexts.disable_nrt_load():
            traced_model = torch.jit.load(neff_path)

        # Replace placeholder weights with real sharded weights
        replace_weights(traced_model, ckpt)
        models.append(traced_model)
        del ckpt
        gc.collect()

        logger.info(f"  [Gemma3 rank {rank}] loaded in {time.time() - t0:.1f}s")

    compiled = TensorParallelNeuronModel(models)
    logger.info(f"  Gemma3: all {tp_degree} ranks loaded and ready")
    return compiled


# ─────────────────────────────────────────────────────────────────────────────
# 3. Neuron Transformer Wrapper (DiT backbone, CPU preprocessing → compiled)
# ─────────────────────────────────────────────────────────────────────────────


class NeuronTransformerWrapper(nn.Module):
    """Drop-in replacement for LTX2VideoTransformer3DModel in the Diffusers pipeline.

    Keeps CPU copies of preprocessing modules:
      proj_in, audio_proj_in, time_embed, audio_time_embed,
      caption_projection, audio_caption_projection,
      rope, audio_rope, cross_attn_rope, cross_attn_audio_rope,
      av_cross_attn_video_scale_shift, av_cross_attn_video_a2v_gate,
      av_cross_attn_audio_scale_shift, av_cross_attn_audio_v2a_gate

    In forward():
      1. CPU preprocessing (proj_in, time_embed, caption_projection, RoPE, masks)
      2. Call compiled backbone with 22 positional tensor args
      3. Return (video_output, audio_output) as the pipeline expects
    """

    def __init__(self, compiled_backbone, cpu_transformer, text_seq: int = 1024):
        """
        Args:
            compiled_backbone: NeuronLTX2BackboneApplication (loaded via NxDI)
            cpu_transformer: Original LTX2VideoTransformer3DModel (for preprocessing)
            text_seq: Maximum text sequence length (must match compile-time)
        """
        super().__init__()
        self.compiled_backbone = compiled_backbone
        self.text_seq = text_seq

        # Copy config and attributes the pipeline expects
        self.config = cpu_transformer.config
        self.dtype = cpu_transformer.dtype
        self.device = cpu_transformer.device

        # Keep CPU preprocessing layers
        self.proj_in = cpu_transformer.proj_in
        self.audio_proj_in = cpu_transformer.audio_proj_in
        self.time_embed = cpu_transformer.time_embed
        self.audio_time_embed = cpu_transformer.audio_time_embed
        self.caption_projection = cpu_transformer.caption_projection
        self.audio_caption_projection = cpu_transformer.audio_caption_projection
        self.rope = cpu_transformer.rope
        self.audio_rope = cpu_transformer.audio_rope
        self.cross_attn_rope = cpu_transformer.cross_attn_rope
        self.cross_attn_audio_rope = cpu_transformer.cross_attn_audio_rope
        self.av_cross_attn_video_scale_shift = cpu_transformer.av_cross_attn_video_scale_shift
        self.av_cross_attn_video_a2v_gate = cpu_transformer.av_cross_attn_video_a2v_gate
        self.av_cross_attn_audio_scale_shift = cpu_transformer.av_cross_attn_audio_scale_shift
        self.av_cross_attn_audio_v2a_gate = cpu_transformer.av_cross_attn_audio_v2a_gate

        # Step-invariant cache (cleared on new prompt)
        self._step_cache = None
        self._step_cache_key = None

    def _compute_step_invariant(
        self,
        encoder_hidden_states,
        audio_encoder_hidden_states,
        encoder_attention_mask,
        audio_encoder_attention_mask,
        video_coords,
        audio_coords,
        batch_size,
        inner_dim,
        audio_inner_dim,
        dtype,
    ):
        """Compute and cache step-invariant preprocessing (caption proj, RoPE, masks).

        These depend only on the prompt and spatial layout, not the timestep,
        so they are identical across all denoising steps.
        """
        with torch.no_grad():
            # Caption projection (CPU)
            enc_hs = self.caption_projection(encoder_hidden_states)
            enc_hs = enc_hs.view(batch_size, -1, inner_dim)
            audio_enc_hs = self.audio_caption_projection(audio_encoder_hidden_states)
            audio_enc_hs = audio_enc_hs.view(batch_size, -1, audio_inner_dim)

            # RoPE (CPU) — compute from spatial coords
            video_rotary_emb = self.rope(video_coords, device="cpu")
            audio_rotary_emb = self.audio_rope(audio_coords, device="cpu")
            video_cross_rotary_emb = self.cross_attn_rope(
                video_coords[:, 0:1, :], device="cpu"
            )
            audio_cross_rotary_emb = self.cross_attn_audio_rope(
                audio_coords[:, 0:1, :], device="cpu"
            )

        # Cast RoPE from float32 to bfloat16 for Neuron
        video_rotary_emb = (video_rotary_emb[0].to(dtype), video_rotary_emb[1].to(dtype))
        audio_rotary_emb = (audio_rotary_emb[0].to(dtype), audio_rotary_emb[1].to(dtype))
        video_cross_rotary_emb = (
            video_cross_rotary_emb[0].to(dtype),
            video_cross_rotary_emb[1].to(dtype),
        )
        audio_cross_rotary_emb = (
            audio_cross_rotary_emb[0].to(dtype),
            audio_cross_rotary_emb[1].to(dtype),
        )

        # Attention masks — convert binary (B, text_seq) to additive bias (B, 1, text_seq)
        with torch.no_grad():
            if encoder_attention_mask is not None and encoder_attention_mask.ndim == 2:
                enc_mask = (1 - encoder_attention_mask.to(dtype)) * -10000.0
                enc_mask = enc_mask.unsqueeze(1)
            else:
                enc_mask = encoder_attention_mask

            if (
                audio_encoder_attention_mask is not None
                and audio_encoder_attention_mask.ndim == 2
            ):
                audio_enc_mask = (1 - audio_encoder_attention_mask.to(dtype)) * -10000.0
                audio_enc_mask = audio_enc_mask.unsqueeze(1)
            else:
                audio_enc_mask = audio_encoder_attention_mask

            if enc_mask is None:
                enc_mask = torch.zeros(batch_size, 1, self.text_seq, dtype=dtype)
            if audio_enc_mask is None:
                audio_enc_mask = torch.zeros(batch_size, 1, self.text_seq, dtype=dtype)

        return (
            enc_hs,
            audio_enc_hs,
            video_rotary_emb,
            audio_rotary_emb,
            video_cross_rotary_emb,
            audio_cross_rotary_emb,
            enc_mask,
            audio_enc_mask,
        )

    def forward(
        self,
        hidden_states,
        audio_hidden_states=None,
        encoder_hidden_states=None,
        audio_encoder_hidden_states=None,
        timestep=None,
        encoder_attention_mask=None,
        audio_encoder_attention_mask=None,
        num_frames=None,
        height=None,
        width=None,
        fps=None,
        audio_num_frames=None,
        video_coords=None,
        audio_coords=None,
        return_dict=False,
        **kwargs,
    ):
        """Preprocess on CPU, run 48 blocks on Neuron, return results.

        Step-invariant computations (caption projection, RoPE, attention masks) are
        cached after the first call and reused for subsequent denoising steps.
        """
        batch_size = hidden_states.shape[0]
        dtype = torch.bfloat16

        with torch.no_grad():
            # ── 1. Project inputs (CPU) — step-varying ──────────────────
            hs = self.proj_in(hidden_states)
            ahs = self.audio_proj_in(audio_hidden_states)

            # ── 2. Time embeddings (CPU) — step-varying ─────────────────
            temb, embedded_ts = self.time_embed(
                timestep.flatten(), batch_size=batch_size, hidden_dtype=dtype
            )
            temb = temb.view(batch_size, -1, temb.size(-1))
            embedded_ts = embedded_ts.view(batch_size, -1, embedded_ts.size(-1))

            temb_audio, audio_embedded_ts = self.audio_time_embed(
                timestep.flatten(), batch_size=batch_size, hidden_dtype=dtype
            )
            temb_audio = temb_audio.view(batch_size, -1, temb_audio.size(-1))
            audio_embedded_ts = audio_embedded_ts.view(
                batch_size, -1, audio_embedded_ts.size(-1)
            )

            # ── 3. Cross-attention conditioning (CPU) — step-varying ────
            ts_scale = (
                self.config.cross_attn_timestep_scale_multiplier
                / self.config.timestep_scale_multiplier
            )

            video_ca_ss, _ = self.av_cross_attn_video_scale_shift(
                timestep.flatten(), batch_size=batch_size, hidden_dtype=dtype
            )
            video_ca_gate, _ = self.av_cross_attn_video_a2v_gate(
                timestep.flatten() * ts_scale, batch_size=batch_size, hidden_dtype=dtype
            )
            video_ca_ss = video_ca_ss.view(batch_size, -1, video_ca_ss.shape[-1])
            video_ca_gate = video_ca_gate.view(batch_size, -1, video_ca_gate.shape[-1])

            audio_ca_ss, _ = self.av_cross_attn_audio_scale_shift(
                timestep.flatten(), batch_size=batch_size, hidden_dtype=dtype
            )
            audio_ca_v2a_gate, _ = self.av_cross_attn_audio_v2a_gate(
                timestep.flatten() * ts_scale, batch_size=batch_size, hidden_dtype=dtype
            )
            audio_ca_ss = audio_ca_ss.view(batch_size, -1, audio_ca_ss.shape[-1])
            audio_ca_v2a_gate = audio_ca_v2a_gate.view(
                batch_size, -1, audio_ca_v2a_gate.shape[-1]
            )

        # ── 4-6. Step-invariant: caption projection, RoPE, masks (cached) ──
        cache_key = encoder_hidden_states.data_ptr()
        if self._step_cache is None or self._step_cache_key != cache_key:
            (
                enc_hs,
                audio_enc_hs,
                video_rotary_emb,
                audio_rotary_emb,
                video_cross_rotary_emb,
                audio_cross_rotary_emb,
                enc_mask,
                audio_enc_mask,
            ) = self._compute_step_invariant(
                encoder_hidden_states,
                audio_encoder_hidden_states,
                encoder_attention_mask,
                audio_encoder_attention_mask,
                video_coords,
                audio_coords,
                batch_size,
                hs.size(-1),
                ahs.size(-1),
                dtype,
            )
            self._step_cache = (
                enc_hs,
                audio_enc_hs,
                video_rotary_emb,
                audio_rotary_emb,
                video_cross_rotary_emb,
                audio_cross_rotary_emb,
                enc_mask,
                audio_enc_mask,
            )
            self._step_cache_key = cache_key
        else:
            (
                enc_hs,
                audio_enc_hs,
                video_rotary_emb,
                audio_rotary_emb,
                video_cross_rotary_emb,
                audio_cross_rotary_emb,
                enc_mask,
                audio_enc_mask,
            ) = self._step_cache

        # ── 7. Call compiled Neuron backbone (22 positional args) ────────
        video_output, audio_output = self.compiled_backbone(
            hs,                          # 1. hidden_states
            ahs,                         # 2. audio_hidden_states
            enc_hs,                      # 3. encoder_hidden_states
            audio_enc_hs,                # 4. audio_encoder_hidden_states
            temb,                        # 5. temb
            temb_audio,                  # 6. temb_audio
            embedded_ts,                 # 7. embedded_timestep
            audio_embedded_ts,           # 8. audio_embedded_timestep
            video_ca_ss,                 # 9. video_cross_attn_scale_shift
            audio_ca_ss,                 # 10. audio_cross_attn_scale_shift
            video_ca_gate,               # 11. video_cross_attn_a2v_gate
            audio_ca_v2a_gate,           # 12. audio_cross_attn_v2a_gate
            video_rotary_emb[0],         # 13. video_rotary_cos
            video_rotary_emb[1],         # 14. video_rotary_sin
            audio_rotary_emb[0],         # 15. audio_rotary_cos
            audio_rotary_emb[1],         # 16. audio_rotary_sin
            video_cross_rotary_emb[0],   # 17. cross_video_rotary_cos
            video_cross_rotary_emb[1],   # 18. cross_video_rotary_sin
            audio_cross_rotary_emb[0],   # 19. cross_audio_rotary_cos
            audio_cross_rotary_emb[1],   # 20. cross_audio_rotary_sin
            enc_mask,                    # 21. encoder_attention_mask
            audio_enc_mask,              # 22. audio_encoder_attention_mask
        )

        return video_output, audio_output


# ─────────────────────────────────────────────────────────────────────────────
# 4. Neuron Tiled VAE Decoder (TP=4 pre-compiled NEFFs)
# ─────────────────────────────────────────────────────────────────────────────


def _create_blend_mask_1d(length: int, blend_size: int, device: str = "cpu") -> torch.Tensor:
    """Create a 1D linear blending mask.

    Returns a tensor of shape [length] where:
    - First blend_size pixels ramp from 0 to 1
    - Middle pixels are 1
    - Last blend_size pixels ramp from 1 to 0
    """
    mask = torch.ones(length, device=device)
    if blend_size > 0:
        ramp = torch.linspace(0, 1, blend_size + 2, device=device)[1:-1]
        mask[:blend_size] = ramp
        mask[-blend_size:] = ramp.flip(0)
    return mask


def _tiled_decode(
    latent: torch.Tensor,
    compiled_model,
    tile_latent_h: int = 8,
    tile_latent_w: int = 8,
    overlap_latent_h: int = 2,
    overlap_latent_w: int = 2,
    spatial_scale: int = 32,
) -> torch.Tensor:
    """Decode latent tensor using spatial tiling with overlap blending.

    Splits the latent [B, C, T, H, W] into spatial tiles of size
    [B, C, T, tile_h, tile_w], decodes each through the compiled model,
    and blends overlapping regions with a linear ramp.

    Args:
        latent: [1, 128, T, H_lat, W_lat] latent tensor (float32)
        compiled_model: Neuron-compiled TP VAE decoder
        tile_latent_h: Tile height in latent pixels
        tile_latent_w: Tile width in latent pixels
        overlap_latent_h: Overlap in latent H pixels
        overlap_latent_w: Overlap in latent W pixels
        spatial_scale: Latent-to-pixel spatial scale (32 for LTX-2)

    Returns:
        [1, 3, T_out, H_out, W_out] decoded video tensor (float32)
    """
    B, C, T, H_lat, W_lat = latent.shape
    assert B == 1, "Batch size must be 1 for tiled decode"

    H_out = H_lat * spatial_scale
    W_out = W_lat * spatial_scale

    stride_h = tile_latent_h - overlap_latent_h
    stride_w = tile_latent_w - overlap_latent_w
    assert stride_h > 0, f"stride_h={stride_h} must be > 0"
    assert stride_w > 0, f"stride_w={stride_w} must be > 0"

    overlap_h_pixels = overlap_latent_h * spatial_scale
    overlap_w_pixels = overlap_latent_w * spatial_scale
    tile_h_pixels = tile_latent_h * spatial_scale
    tile_w_pixels = tile_latent_w * spatial_scale

    # Compute tile start positions
    tile_starts_h = []
    i = 0
    while True:
        start = i * stride_h
        if start + tile_latent_h > H_lat:
            start = H_lat - tile_latent_h
            tile_starts_h.append(max(0, start))
            break
        tile_starts_h.append(start)
        if start + tile_latent_h >= H_lat:
            break
        i += 1
    tile_starts_h = sorted(set(tile_starts_h))

    tile_starts_w = []
    i = 0
    while True:
        start = i * stride_w
        if start + tile_latent_w > W_lat:
            start = W_lat - tile_latent_w
            tile_starts_w.append(max(0, start))
            break
        tile_starts_w.append(start)
        if start + tile_latent_w >= W_lat:
            break
        i += 1
    tile_starts_w = sorted(set(tile_starts_w))

    total_tiles = len(tile_starts_h) * len(tile_starts_w)
    logger.info(
        f"  VAE tiling: {len(tile_starts_h)}x{len(tile_starts_w)} = {total_tiles} tiles, "
        f"tile={tile_latent_h}x{tile_latent_w}, overlap=({overlap_latent_h},{overlap_latent_w})"
    )

    # Output temporal dimension: T_out = (T-1)*8 + 1
    T_out = (T - 1) * 8 + 1

    output_accum = torch.zeros(1, 3, T_out, H_out, W_out, dtype=torch.float32)
    weight_accum = torch.zeros(1, 1, 1, H_out, W_out, dtype=torch.float32)

    decode_times = []

    for ti, h_start_lat in enumerate(tile_starts_h):
        for tj, w_start_lat in enumerate(tile_starts_w):
            tile_idx = ti * len(tile_starts_w) + tj + 1

            h_end_lat = h_start_lat + tile_latent_h
            w_end_lat = w_start_lat + tile_latent_w
            tile_latent = latent[:, :, :, h_start_lat:h_end_lat, w_start_lat:w_end_lat]

            t0 = time.time()
            with torch.no_grad():
                tile_output = compiled_model(tile_latent)
            dt = time.time() - t0
            decode_times.append(dt)

            # Pixel coordinates for this tile
            h_start_px = h_start_lat * spatial_scale
            w_start_px = w_start_lat * spatial_scale
            h_end_px = h_start_px + tile_h_pixels
            w_end_px = w_start_px + tile_w_pixels

            # Create spatial blend masks
            blend_h = _create_blend_mask_1d(
                tile_h_pixels, overlap_h_pixels if h_start_lat > 0 else 0
            )
            blend_w = _create_blend_mask_1d(
                tile_w_pixels, overlap_w_pixels if w_start_lat > 0 else 0
            )

            # Handle end tiles (ramp down at the boundary if not last)
            if h_end_lat < H_lat and overlap_h_pixels > 0:
                blend_h[-overlap_h_pixels:] = torch.linspace(
                    1, 0, overlap_h_pixels + 2
                )[1:-1]
            if w_end_lat < W_lat and overlap_w_pixels > 0:
                blend_w[-overlap_w_pixels:] = torch.linspace(
                    1, 0, overlap_w_pixels + 2
                )[1:-1]

            # 2D blend mask: [1, 1, 1, H, W]
            blend_mask = blend_h.unsqueeze(1) * blend_w.unsqueeze(0)
            blend_mask = blend_mask.unsqueeze(0).unsqueeze(0).unsqueeze(0)

            output_accum[:, :, :, h_start_px:h_end_px, w_start_px:w_end_px] += (
                tile_output.float() * blend_mask
            )
            weight_accum[:, :, :, h_start_px:h_end_px, w_start_px:w_end_px] += blend_mask

            if tile_idx <= 4 or tile_idx == total_tiles:
                logger.info(
                    f"    Tile {tile_idx}/{total_tiles}: "
                    f"lat[{h_start_lat}:{h_end_lat},{w_start_lat}:{w_end_lat}] "
                    f"{dt * 1000:.0f}ms"
                )

    output = output_accum / weight_accum.clamp(min=1e-6)
    total_decode = sum(decode_times)
    logger.info(
        f"  VAE decode total: {total_decode:.2f}s "
        f"({total_tiles} tiles, avg {total_decode / total_tiles * 1000:.0f}ms/tile)"
    )
    return output


class NeuronTiledVAEDecoder(nn.Module):
    """Drop-in replacement for the Diffusers VAE decoder.

    Loads 4 pre-compiled NEFF shards as a TensorParallelNeuronModel and
    performs tiled spatial decoding with overlap blending.

    The outer AutoencoderKLLTX2Video.decode() calls:
        self.decoder(hidden_states, temb=None, causal=False)
    This wrapper matches that interface.
    """

    def __init__(
        self,
        compiled_dir: str,
        tile_latent_h: int = 8,
        tile_latent_w: int = 8,
        overlap_latent_h: int = 2,
        overlap_latent_w: int = 2,
        original_decoder=None,
        tp_degree: int = 4,
    ):
        """
        Args:
            compiled_dir: Path to directory with compiled TP model (tp_0.pt, etc.)
            tile_latent_h: Tile height in latent pixels
            tile_latent_w: Tile width in latent pixels
            overlap_latent_h: Overlap in latent H
            overlap_latent_w: Overlap in latent W
            original_decoder: Original decoder (for attribute copying)
            tp_degree: Number of TP shards
        """
        super().__init__()
        self.tile_latent_h = tile_latent_h
        self.tile_latent_w = tile_latent_w
        self.overlap_latent_h = overlap_latent_h
        self.overlap_latent_w = overlap_latent_w
        self.tp_degree = tp_degree

        # Load compiled TP model
        logger.info(f"Loading compiled VAE from {compiled_dir} (TP={tp_degree})...")
        t0 = time.time()
        self.compiled_model = self._load_compiled_vae(compiled_dir)
        logger.info(f"  VAE loaded in {time.time() - t0:.1f}s")

        # Copy attributes from original decoder
        if original_decoder is not None:
            self.patch_size = getattr(original_decoder, "patch_size", 4)
            self.patch_size_t = getattr(original_decoder, "patch_size_t", 1)
            self.is_causal = getattr(original_decoder, "is_causal", False)
        else:
            self.patch_size = 4
            self.patch_size_t = 1
            self.is_causal = False

        self._warmed_up = False

    def _load_compiled_vae(self, compiled_dir: str):
        """Load compiled TP VAE from tp_0.pt ... tp_N.pt.

        Uses neuronx_distributed.trace.parallel_model_load if available,
        otherwise falls back to manual TensorParallelNeuronModel construction.
        """
        try:
            import neuronx_distributed
            return neuronx_distributed.trace.parallel_model_load(compiled_dir)
        except (ImportError, AttributeError):
            # Fallback: manual load
            import torch_neuronx
            from neuronx_distributed.trace.trace import TensorParallelNeuronModel

            models = []
            for rank in range(self.tp_degree):
                neff_path = os.path.join(compiled_dir, f"tp_{rank}.pt")
                if not os.path.exists(neff_path):
                    raise FileNotFoundError(f"VAE NEFF not found: {neff_path}")
                traced = torch.jit.load(neff_path)
                models.append(traced)
            return TensorParallelNeuronModel(models)

    def warmup(self, num_frames: int = 121):
        """Run 2 warmup iterations to prime the Neuron model."""
        if self._warmed_up:
            return
        logger.info("  Warming up VAE decoder...")
        latent_t = (num_frames - 1) // 8 + 1
        dummy = torch.randn(
            1, 128, latent_t, self.tile_latent_h, self.tile_latent_w, dtype=torch.float32
        )
        for _ in range(2):
            with torch.no_grad():
                self.compiled_model(dummy)
        self._warmed_up = True
        logger.info("  VAE warmup complete")

    def forward(self, hidden_states, temb=None, causal=None):
        """Decode latent tensor using Neuron tiled decode.

        Args:
            hidden_states: [B, C, T, H, W] latent tensor
            temb: Time embedding (always None for LTX-2)
            causal: Causal mode flag (always False for inference)

        Returns:
            Decoded tensor [B, 3, T_out, H_out, W_out]
        """
        if not self._warmed_up:
            num_frames_approx = (hidden_states.shape[2] - 1) * 8 + 1
            self.warmup(num_frames=num_frames_approx)

        # Tiled decode expects float32 input
        latent_fp32 = hidden_states.float()

        output = _tiled_decode(
            latent_fp32,
            self.compiled_model,
            tile_latent_h=self.tile_latent_h,
            tile_latent_w=self.tile_latent_w,
            overlap_latent_h=self.overlap_latent_h,
            overlap_latent_w=self.overlap_latent_w,
            spatial_scale=32,
        )

        return output


# ─────────────────────────────────────────────────────────────────────────────
# 5. DiT Backbone Loader (NxDI-based)
# ─────────────────────────────────────────────────────────────────────────────


def load_neuron_dit(
    transformer_model_path: str,
    compile_dir: str,
    height: int,
    width: int,
    num_frames: int,
    tp_degree: int = 4,
):
    """Load the pre-compiled DiT backbone via NxDI's NeuronApplicationBase.

    This is the one component that still uses NxDI at load time for initial
    weight sharding into the traced model format.

    Args:
        transformer_model_path: Path to transformer weights (HF format)
        compile_dir: Path to compiled DiT directory (model.pt + config)
        height: Video height in pixels
        width: Video width in pixels
        num_frames: Number of video frames
        tp_degree: Tensor parallel degree

    Returns:
        backbone_app: The loaded NeuronLTX2BackboneApplication
    """
    from neuronx_distributed_inference.models.config import NeuronConfig

    # Import the NxDI LTX2 model classes
    # These come from the NxDI contrib models package
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    try:
        from modeling_ltx2 import LTX2BackboneInferenceConfig, NeuronLTX2BackboneApplication
    except ImportError:
        # Try the NxDI installed path
        from neuronx_distributed_inference.contrib.models.ltx2_video_audio.modeling_ltx2 import (
            LTX2BackboneInferenceConfig,
            NeuronLTX2BackboneApplication,
        )

    # Build config from the transformer's config.json
    config_path = os.path.join(transformer_model_path, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Transformer config not found: {config_path}\n"
            f"Expected HF-format transformer directory."
        )

    with open(config_path) as f:
        hf_config = json.load(f)

    num_heads = hf_config["num_attention_heads"]
    head_dim = hf_config["attention_head_dim"]
    inner_dim = num_heads * head_dim
    audio_num_heads = hf_config["audio_num_attention_heads"]
    audio_head_dim = hf_config["audio_attention_head_dim"]
    audio_inner_dim = audio_num_heads * audio_head_dim
    audio_ca_dim = hf_config.get("audio_cross_attention_dim", audio_inner_dim)

    latent_num_frames = (num_frames - 1) // 8 + 1
    latent_height = height // 32
    latent_width = width // 32
    video_seq = latent_num_frames * latent_height * latent_width
    audio_num_frames_val = round((num_frames / 24.0) * 24.97)

    backbone_neuron_config = NeuronConfig(
        tp_degree=tp_degree,
        world_size=tp_degree,
        torch_dtype=torch.bfloat16,
    )

    config = LTX2BackboneInferenceConfig(
        neuron_config=backbone_neuron_config,
        num_layers=hf_config["num_layers"],
        num_attention_heads=num_heads,
        attention_head_dim=head_dim,
        inner_dim=inner_dim,
        audio_num_attention_heads=audio_num_heads,
        audio_attention_head_dim=audio_head_dim,
        audio_inner_dim=audio_inner_dim,
        audio_cross_attention_dim=audio_ca_dim,
        caption_channels=hf_config.get("caption_channels", 3840),
        video_seq=video_seq,
        audio_seq=audio_num_frames_val,
        text_seq=1024,
        height=height,
        width=width,
        num_frames=num_frames,
    )
    config.hf_config_dict = hf_config

    logger.info(
        f"  DiT config: {hf_config['num_layers']} blocks, TP={tp_degree}, "
        f"{height}x{width}, {num_frames} frames, video_seq={video_seq}"
    )

    # Load the compiled backbone
    backbone_app = NeuronLTX2BackboneApplication(
        model_path=transformer_model_path, config=config
    )
    backbone_app.load(compile_dir)

    return backbone_app


# ─────────────────────────────────────────────────────────────────────────────
# 6. Main — Load all pre-compiled components and generate
# ─────────────────────────────────────────────────────────────────────────────


def parse_args():
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="LTX-2 Optimized — Pre-Compiled NEFF Loading",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Compiled model paths
    parser.add_argument(
        "--gemma3-compile-dir",
        type=str,
        default="/mnt/data/gemma3_compiled",
        help="Directory with compiled Gemma3 NEFFs (tp_0.pt ... tp_3.pt)",
    )
    parser.add_argument(
        "--gemma3-sharded-dir",
        type=str,
        default="/mnt/data/gemma3_sharded",
        help="Directory with pre-sharded Gemma3 weights (rank_0.pt ... rank_3.pt)",
    )
    parser.add_argument(
        "--dit-compile-dir",
        type=str,
        default="/mnt/data/ltx2_compiled",
        help="Directory with compiled DiT backbone (NxDI format)",
    )
    parser.add_argument(
        "--vae-compile-dir",
        type=str,
        default="/mnt/data/ltx2_vae_compiled",
        help="Directory with compiled VAE NEFFs (tp_0.pt ... tp_3.pt)",
    )

    # Model/generation settings
    parser.add_argument(
        "--transformer-path",
        type=str,
        default=None,
        help="Local path to transformer weights (HF format). If None, downloads from HF.",
    )
    parser.add_argument("--height", type=int, default=512, help="Video height in pixels")
    parser.add_argument("--width", type=int, default=768, help="Video width in pixels")
    parser.add_argument("--num-frames", type=int, default=121, help="Number of video frames")
    parser.add_argument("--num-steps", type=int, default=40, help="Number of denoising steps")
    parser.add_argument("--guidance-scale", type=float, default=4.0, help="CFG guidance scale")
    parser.add_argument(
        "--prompt",
        type=str,
        default=(
            "A close-up shot of a young waitress in a retro 1950s diner, her warm brown eyes "
            "meeting the camera with a gentle smile. She wears a black polka-dot dress with an "
            "elegant cream lace collar, her reddish-brown hair styled in an elaborate updo with "
            "delicate curls framing her freckled face. Soft, warm light from overhead fixtures "
            "illuminates her features as she stands behind a yellow counter."
        ),
        help="Text prompt for generation",
    )
    parser.add_argument("--seed", type=int, default=10, help="Random seed")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for frames and video (default: ../results/)",
    )

    # VAE tiling config
    parser.add_argument("--vae-tile-h", type=int, default=8, help="VAE tile height in latent pixels")
    parser.add_argument("--vae-tile-w", type=int, default=8, help="VAE tile width in latent pixels")
    parser.add_argument("--vae-overlap-h", type=int, default=2, help="VAE overlap height in latent pixels")
    parser.add_argument("--vae-overlap-w", type=int, default=2, help="VAE overlap width in latent pixels")

    # TP
    parser.add_argument("--tp-degree", type=int, default=4, help="Tensor parallel degree")

    return parser.parse_args()


def main():
    """Entry point: load pre-compiled NEFFs, assemble pipeline, generate.

    Launch with single process (no torchrun needed):
      NEURON_FUSE_SOFTMAX=1 NEURON_CUSTOM_SILU=1 NEURON_RT_STOCHASTIC_ROUNDING_EN=0 \\
        python neuron_ltx2_optimized.py
    """
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    # ── Install BMM-SDPA globally (still needed for CPU SDPA calls) ─────
    install_bmm_sdpa()

    print("=" * 70)
    print("LTX-2 OPTIMIZED — Pre-Compiled NEFF Loading")
    print("=" * 70)
    print(f"  Text Encoder: Neuron Gemma3-12B (TP={args.tp_degree})")
    print(f"  DiT Backbone: Neuron LTX2 48-block (TP={args.tp_degree})")
    print(f"  VAE Decoder:  Neuron tiled (TP={args.tp_degree})")
    print(f"  Resolution:   {args.width}x{args.height}, {args.num_frames} frames")
    print(f"  Steps:        {args.num_steps}")
    print("=" * 70)

    t_total = time.time()
    timings = {}

    # ────────────────────────────────────────────────────────────────────
    # Step 1: Load Diffusers LTX2Pipeline (CPU)
    # ────────────────────────────────────────────────────────────────────
    print("\n[1/6] Loading Diffusers LTX2Pipeline (CPU)...")
    t0 = time.time()

    from diffusers import LTX2Pipeline

    pipe = LTX2Pipeline.from_pretrained("Lightricks/LTX-2", torch_dtype=torch.bfloat16)
    timings["pipeline_load"] = time.time() - t0
    logger.info(f"  Pipeline loaded in {timings['pipeline_load']:.1f}s")

    # ────────────────────────────────────────────────────────────────────
    # Step 2: Load compiled Gemma3 text encoder → swap pipe.text_encoder
    # ────────────────────────────────────────────────────────────────────
    print(f"\n[2/6] Loading Neuron Gemma3 text encoder...")
    print(f"  Compile dir: {args.gemma3_compile_dir}")
    print(f"  Sharded dir: {args.gemma3_sharded_dir}")
    t0 = time.time()

    # Verify directories exist
    if not os.path.isdir(args.gemma3_compile_dir):
        raise FileNotFoundError(
            f"Gemma3 compile directory not found: {args.gemma3_compile_dir}\n"
            f"Run compile_gemma3.py first to generate the compiled NEFFs."
        )
    if not os.path.isdir(args.gemma3_sharded_dir):
        raise FileNotFoundError(
            f"Gemma3 sharded directory not found: {args.gemma3_sharded_dir}\n"
            f"Run shard_gemma3_weights.py first to generate sharded weights."
        )

    # Free the CPU text encoder before loading Neuron version
    del pipe.text_encoder
    gc.collect()

    compiled_gemma3 = load_neuron_gemma3(
        sharded_dir=args.gemma3_sharded_dir,
        compile_dir=args.gemma3_compile_dir,
        tp_degree=args.tp_degree,
    )
    pipe.text_encoder = NeuronTextEncoderWrapper(compiled_gemma3)
    timings["gemma3_load"] = time.time() - t0
    logger.info(f"  Gemma3 loaded and swapped in {timings['gemma3_load']:.1f}s")

    # ────────────────────────────────────────────────────────────────────
    # Step 3: Load compiled DiT backbone → wrap → swap pipe.transformer
    # ────────────────────────────────────────────────────────────────────
    print(f"\n[3/6] Loading Neuron DiT backbone...")
    print(f"  Compile dir: {args.dit_compile_dir}")
    t0 = time.time()

    if not os.path.isdir(args.dit_compile_dir):
        raise FileNotFoundError(
            f"DiT compile directory not found: {args.dit_compile_dir}\n"
            f"Run the NxDI compile step first."
        )

    # Resolve transformer model path (for config.json + weight sharding)
    transformer_path = args.transformer_path
    if transformer_path is None:
        from huggingface_hub import snapshot_download

        logger.info("  Downloading transformer weights from HuggingFace...")
        local_path = snapshot_download("Lightricks/LTX-2", allow_patterns=["transformer/*"])
        transformer_path = os.path.join(local_path, "transformer")

    cpu_transformer = pipe.transformer

    backbone_app = load_neuron_dit(
        transformer_model_path=transformer_path,
        compile_dir=args.dit_compile_dir,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        tp_degree=args.tp_degree,
    )

    # Wrap with NeuronTransformerWrapper (CPU preprocessing → compiled backbone)
    wrapper = NeuronTransformerWrapper(
        compiled_backbone=backbone_app,
        cpu_transformer=cpu_transformer,
        text_seq=1024,
    )

    # Free heavy transformer blocks from CPU
    del cpu_transformer.transformer_blocks
    del cpu_transformer.norm_out, cpu_transformer.proj_out
    del cpu_transformer.audio_norm_out, cpu_transformer.audio_proj_out
    gc.collect()

    pipe.transformer = wrapper
    timings["dit_load"] = time.time() - t0
    logger.info(f"  DiT loaded and swapped in {timings['dit_load']:.1f}s")

    # ────────────────────────────────────────────────────────────────────
    # Step 4: Load compiled VAE decoder → swap pipe.vae.decoder
    # ────────────────────────────────────────────────────────────────────
    use_neuron_vae = os.path.isdir(args.vae_compile_dir) and os.path.exists(
        os.path.join(args.vae_compile_dir, "tp_0.pt")
    )

    if use_neuron_vae:
        print(f"\n[4/6] Loading Neuron VAE decoder...")
        print(f"  Compile dir: {args.vae_compile_dir}")
        print(f"  Tile: {args.vae_tile_h}x{args.vae_tile_w}, overlap: ({args.vae_overlap_h},{args.vae_overlap_w})")
        t0 = time.time()

        original_decoder = pipe.vae.decoder
        neuron_vae = NeuronTiledVAEDecoder(
            compiled_dir=args.vae_compile_dir,
            tile_latent_h=args.vae_tile_h,
            tile_latent_w=args.vae_tile_w,
            overlap_latent_h=args.vae_overlap_h,
            overlap_latent_w=args.vae_overlap_w,
            original_decoder=original_decoder,
            tp_degree=args.tp_degree,
        )
        del original_decoder
        gc.collect()

        pipe.vae.decoder = neuron_vae

        # Warmup the VAE
        logger.info("  Running VAE warmup...")
        neuron_vae.warmup(num_frames=args.num_frames)

        timings["vae_load"] = time.time() - t0
        logger.info(f"  VAE loaded, swapped, and warmed up in {timings['vae_load']:.1f}s")
    else:
        print(f"\n[4/6] Neuron VAE NOT FOUND at {args.vae_compile_dir}")
        print(f"  Falling back to CPU VAE decoder")
        timings["vae_load"] = 0.0

    # ────────────────────────────────────────────────────────────────────
    # Step 5: Generate
    # ────────────────────────────────────────────────────────────────────
    print(f"\n[5/6] Generating video+audio...")
    print(f"  Prompt: {args.prompt[:80]}...")
    print(f"  {args.width}x{args.height}, {args.num_frames} frames, {args.num_steps} steps")
    print(f"  Guidance scale: {args.guidance_scale}, Seed: {args.seed}")

    generator = torch.Generator(device="cpu").manual_seed(args.seed)

    t0 = time.time()
    output = pipe(
        prompt=args.prompt,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        num_inference_steps=args.num_steps,
        guidance_scale=args.guidance_scale,
        generator=generator,
        output_type="pil",
    )
    timings["generation"] = time.time() - t0
    per_step = timings["generation"] / args.num_steps
    logger.info(f"  Generation complete in {timings['generation']:.1f}s ({per_step:.2f}s/step)")

    # ────────────────────────────────────────────────────────────────────
    # Step 6: Save outputs
    # ────────────────────────────────────────────────────────────────────
    print(f"\n[6/6] Saving outputs...")

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
    os.makedirs(output_dir, exist_ok=True)

    frames = output.frames[0]

    # Save individual frames
    frames_dir = os.path.join(output_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)
    for i, frame in enumerate(frames):
        frame.save(os.path.join(frames_dir, f"frame_{i:04d}.png"))
    logger.info(f"  Saved {len(frames)} frames to {frames_dir}/")

    # Save video
    try:
        from diffusers.utils import export_to_video

        video_path = os.path.join(output_dir, "ltx2_optimized_output.mp4")
        export_to_video(frames, video_path, fps=24)
        logger.info(f"  Video saved to {video_path}")
    except Exception as e:
        logger.warning(f"  Video export failed: {e}")
        video_path = None

    # Save metadata
    total_time = time.time() - t_total
    metadata = {
        "model": "Lightricks/LTX-2",
        "script": "neuron_ltx2_optimized.py",
        "prompt": args.prompt,
        "resolution": f"{args.width}x{args.height}",
        "num_frames": args.num_frames,
        "num_steps": args.num_steps,
        "guidance_scale": args.guidance_scale,
        "seed": args.seed,
        "tp_degree": args.tp_degree,
        "timings": {
            "pipeline_load_s": round(timings["pipeline_load"], 2),
            "gemma3_load_s": round(timings["gemma3_load"], 2),
            "dit_load_s": round(timings["dit_load"], 2),
            "vae_load_s": round(timings["vae_load"], 2),
            "generation_s": round(timings["generation"], 2),
            "per_step_s": round(per_step, 2),
            "total_s": round(total_time, 2),
        },
        "components": {
            "text_encoder": f"Neuron Gemma3-12B (TP={args.tp_degree})",
            "dit_backbone": f"Neuron LTX2 48-block (TP={args.tp_degree})",
            "vae_decoder": (
                f"Neuron tiled {args.vae_tile_h}x{args.vae_tile_w} (TP={args.tp_degree})"
                if use_neuron_vae
                else "CPU (Diffusers default)"
            ),
        },
        "paths": {
            "gemma3_compile_dir": args.gemma3_compile_dir,
            "gemma3_sharded_dir": args.gemma3_sharded_dir,
            "dit_compile_dir": args.dit_compile_dir,
            "vae_compile_dir": args.vae_compile_dir,
        },
    }
    metadata_path = os.path.join(output_dir, "optimized_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"  Metadata saved to {metadata_path}")

    # ────────────────────────────────────────────────────────────────────
    # Summary
    # ────────────────────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("SUMMARY — LTX-2 Optimized (Pre-Compiled NEFFs)")
    print(f"{'=' * 70}")
    print(f"  Pipeline load:     {timings['pipeline_load']:6.1f}s")
    print(f"  Gemma3 load:       {timings['gemma3_load']:6.1f}s")
    print(f"  DiT load:          {timings['dit_load']:6.1f}s")
    print(f"  VAE load+warmup:   {timings['vae_load']:6.1f}s")
    print(f"  ─────────────────────────────")
    print(f"  Generation:        {timings['generation']:6.1f}s  ({per_step:.2f}s/step)")
    print(f"  ─────────────────────────────")
    print(f"  Total wall time:   {total_time:6.1f}s")
    print(f"")
    print(f"  Output: {len(frames)} frames")
    if video_path:
        print(f"  Video:  {video_path}")
    print(f"  Text encoder: Neuron Gemma3-12B")
    print(f"  DiT backbone: Neuron LTX2 48 blocks")
    print(f"  VAE decoder:  {'Neuron tiled' if use_neuron_vae else 'CPU fallback'}")
    print(f"  Output dir:   {output_dir}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()

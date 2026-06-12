"""TP=4 plan for `LTX2VideoTransformer3DModel` on Beta 3 / native PyTorch.

Per-block layout (48 blocks):
  attn1            (video self-attention,  inner_dim=4096, 32 heads × 128)
  audio_attn1      (audio self-attention,  audio_dim=2048, 32 heads × 64)
  attn2            (video cross-attention, inner=4096 → kv from text 4096)
  audio_attn2      (audio cross-attention, inner=2048 → kv from text 2048)
  audio_to_video_attn   (a2v cross, inner=4096)
  ff               (video FFN, dim=4096)
  audio_ff         (audio FFN, dim=2048)

ColwiseParallel splits the output dim of qkv-projection (heads/N per rank)
RowwiseParallel splits the input dim of out-projection (all-reduce sum)

Scope of this file:
  - qwen_edit_tp_plan() equivalent for LTX-2: returns the parallelize_module
    plan (ColwiseParallel / RowwiseParallel mapping)
  - apply_tp_fixes() — patches attn.heads on each LTX2Attention module
    after sharding (since each rank sees heads/N), and replaces the
    per-head RMSNorm with a TP-aware version

This is the "five fixes" pattern documented in
.kiro/steering/neuron-tp-on-beta2.md but adapted to the LTX-2 layer
naming and the Beta 3 device API.
"""
from __future__ import annotations

import torch
import torch.nn as nn

# Architecture constants from LTX-2 transformer/config.json
N_LAYERS = 48
N_HEADS_FULL = 32
HEAD_DIM = 128
INNER_DIM = N_HEADS_FULL * HEAD_DIM  # 4096

AUDIO_HEADS = 32
AUDIO_HEAD_DIM = 64
AUDIO_INNER_DIM = AUDIO_HEADS * AUDIO_HEAD_DIM  # 2048


def ltx2_tp_plan(world_size: int) -> dict:
    """Return parallelize_module plan keyed by submodule path.

    Path format: `transformer_blocks.<layer>.<attn>.<linear>` matches
    the layer names diffusers exposes under the `LTX2VideoTransformer3DModel`.
    """
    from torch.distributed.tensor.parallel import (
        ColwiseParallel, RowwiseParallel,
    )
    plan = {}
    for layer in range(N_LAYERS):
        prefix = f"transformer_blocks.{layer}"

        # Video self-attention (attn1)
        plan[f"{prefix}.attn1.to_q"] = ColwiseParallel()
        plan[f"{prefix}.attn1.to_k"] = ColwiseParallel()
        plan[f"{prefix}.attn1.to_v"] = ColwiseParallel()
        plan[f"{prefix}.attn1.to_out.0"] = RowwiseParallel()

        # Audio self-attention (audio_attn1)
        plan[f"{prefix}.audio_attn1.to_q"] = ColwiseParallel()
        plan[f"{prefix}.audio_attn1.to_k"] = ColwiseParallel()
        plan[f"{prefix}.audio_attn1.to_v"] = ColwiseParallel()
        plan[f"{prefix}.audio_attn1.to_out.0"] = RowwiseParallel()

        # Video cross-attention (attn2) — Q from video, K/V from text
        plan[f"{prefix}.attn2.to_q"] = ColwiseParallel()
        plan[f"{prefix}.attn2.to_k"] = ColwiseParallel()
        plan[f"{prefix}.attn2.to_v"] = ColwiseParallel()
        plan[f"{prefix}.attn2.to_out.0"] = RowwiseParallel()

        # Audio cross-attention (audio_attn2)
        plan[f"{prefix}.audio_attn2.to_q"] = ColwiseParallel()
        plan[f"{prefix}.audio_attn2.to_k"] = ColwiseParallel()
        plan[f"{prefix}.audio_attn2.to_v"] = ColwiseParallel()
        plan[f"{prefix}.audio_attn2.to_out.0"] = RowwiseParallel()

        # Audio→Video cross-attention
        plan[f"{prefix}.audio_to_video_attn.to_q"] = ColwiseParallel()
        plan[f"{prefix}.audio_to_video_attn.to_k"] = ColwiseParallel()
        plan[f"{prefix}.audio_to_video_attn.to_v"] = ColwiseParallel()
        plan[f"{prefix}.audio_to_video_attn.to_out.0"] = RowwiseParallel()

        # Video FFN (FeedForward typically has net.0.proj (gate+up) and net.2 (down))
        plan[f"{prefix}.ff.net.0.proj"] = ColwiseParallel()
        plan[f"{prefix}.ff.net.2"] = RowwiseParallel()

        # Audio FFN
        plan[f"{prefix}.audio_ff.net.0.proj"] = ColwiseParallel()
        plan[f"{prefix}.audio_ff.net.2"] = RowwiseParallel()

    return plan


def apply_tp_fixes(model, world_size: int, rank: int):
    """Patch attn.heads to heads/N on each LTX2Attention module.

    After ColwiseParallel splits to_q/k/v, each rank sees only
    inner_dim/N. The block's forward computes
        query.unflatten(-1, (attn.heads, head_dim))
    so attn.heads must be patched to heads/N to give the correct
    head_dim back.

    Also no per-head RMSNorm sharding to do here — LTX-2's
    `qk_norm: rms_norm_across_heads` means the norm is applied
    PER HEAD over head_dim, NOT across all heads. So the norm
    weight shape is `[head_dim]` (= 128) which is invariant under
    head-count sharding. No replacement needed.
    """
    new_video_heads = N_HEADS_FULL // world_size
    new_audio_heads = AUDIO_HEADS // world_size

    for layer in range(N_LAYERS):
        block = model.transformer_blocks[layer]
        # Video attention modules — full inner_dim sharded
        for attn_name in ("attn1", "attn2", "audio_to_video_attn"):
            attn = getattr(block, attn_name, None)
            if attn is not None and hasattr(attn, "heads"):
                attn.heads = new_video_heads
        # Audio attention modules
        for attn_name in ("audio_attn1", "audio_attn2"):
            attn = getattr(block, attn_name, None)
            if attn is not None and hasattr(attn, "heads"):
                attn.heads = new_audio_heads

    if rank == 0:
        print(f"[ltx2_tp_plan] patched attn.heads: video={new_video_heads}, "
              f"audio={new_audio_heads} on {N_LAYERS} blocks", flush=True)

    # NOTE: adaptive QK norm install moved to a SEPARATE function
    # (install_adaptive_qk_norm) that MUST run AFTER load_weights_sharded,
    # because it replaces the norm_q/norm_k modules — if done before load,
    # the loader can't find `norm_q.weight` and the weights stay on meta.


def install_adaptive_qk_norm(model, world_size: int, rank: int):
    import torch.distributed as dist

    class _AdaptiveQKNorm(nn.Module):
        def __init__(self, weight, eps, world_size, rank):
            super().__init__()
            # Keep the full replicated weight as a buffer-like Parameter
            self.full_weight = weight  # nn.Parameter, full inner_dim
            self.eps = eps
            self.world_size = world_size
            self.rank = rank

        def forward(self, x):
            in_dim = x.shape[-1]
            full_dim = self.full_weight.shape[0]
            sharded = in_dim < full_dim
            local_sq = (x.float() ** 2).sum(dim=-1, keepdim=True)
            if sharded and self.world_size > 1 and dist.is_initialized():
                dist.all_reduce(local_sq, op=dist.ReduceOp.SUM)
                denom = full_dim
                w = self.full_weight.narrow(0, self.rank * in_dim, in_dim)
            else:
                denom = in_dim
                w = self.full_weight if in_dim == full_dim else \
                    self.full_weight.narrow(0, 0, in_dim)
            rms = (local_sq / denom + self.eps).rsqrt()
            out = (x.float() * rms).to(x.dtype)
            return out * w

    n_patched = 0
    eps = 1e-6
    for layer in range(N_LAYERS):
        block = model.transformer_blocks[layer]
        for attn_name in ("attn1", "attn2", "audio_attn1", "audio_attn2",
                          "audio_to_video_attn", "video_to_audio_attn"):
            attn = getattr(block, attn_name, None)
            if attn is None:
                continue
            for norm_name in ("norm_q", "norm_k"):
                norm = getattr(attn, norm_name, None)
                if norm is None or not hasattr(norm, "weight") or norm.weight is None:
                    continue
                eps_val = getattr(norm, "eps", eps) or eps
                setattr(attn, norm_name,
                        _AdaptiveQKNorm(norm.weight, eps_val, world_size, rank))
                n_patched += 1

    if rank == 0:
        print(f"[ltx2_tp_plan] installed adaptive QK norm on {n_patched} norms",
              flush=True)


def patch_rope_rank_slice(model, world_size: int, rank: int):
    """Slice every RoPE module's cos/sin output by this rank's head range.

    LTX-2 has FOUR RoPE modules on the top-level transformer:
        - rope                  (video self-attn,  num_attention_heads=32)
        - audio_rope            (audio self-attn,  audio_num_attention_heads=32)
        - cross_attn_rope       (video cross-attn, num_attention_heads=32)
        - cross_attn_audio_rope (audio cross-attn, audio_num_attention_heads=32)

    Each returns cos/sin of shape (B, H, T, D//2) for `split` rope (or
    (B, T, 2r) for interleaved). After ColwiseParallel shards q/k/v, each
    rank only holds H/world_size heads, so we slice axis 1 of cos/sin to
    [rank*H_local : (rank+1)*H_local].

    We read each module's own `num_attention_heads` so audio (which may
    differ) is handled correctly.
    """
    rope_attrs = ["rope", "audio_rope", "cross_attn_rope", "cross_attn_audio_rope"]
    patched = []

    for attr in rope_attrs:
        rope = getattr(model, attr, None)
        if rope is None:
            continue

        full_heads = getattr(rope, "num_attention_heads", N_HEADS_FULL)
        if full_heads % world_size != 0:
            if rank == 0:
                print(f"[ltx2_tp_plan] WARN: {attr}.num_attention_heads={full_heads} "
                      f"not divisible by world_size={world_size}; skipping slice",
                      flush=True)
            continue
        h_local = full_heads // world_size
        start = rank * h_local
        end = start + h_local

        # Bind loop vars via default args to avoid late-binding bug
        _orig = rope.forward

        def _sliced(*args, _orig=_orig, _start=start, _end=end, **kwargs):
            out = _orig(*args, **kwargs)
            if isinstance(out, tuple) and len(out) == 2 and torch.is_tensor(out[0]):
                cos, sin = out
                # split rope: cos/sin are (B, H, T, D//2) — slice axis 1
                if cos.dim() == 4:
                    cos = cos[:, _start:_end]
                    sin = sin[:, _start:_end]
                return cos, sin
            return out

        rope.forward = _sliced
        patched.append(f"{attr}(heads {start}:{end})")

    if rank == 0:
        print(f"[ltx2_tp_plan] patched RoPE rank slices: {patched}", flush=True)

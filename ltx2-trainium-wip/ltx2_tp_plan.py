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

        # Audio→Video cross-attention (Q: Video, K,V: Audio; uses audio_num_attention_heads=32)
        plan[f"{prefix}.audio_to_video_attn.to_q"] = ColwiseParallel()
        plan[f"{prefix}.audio_to_video_attn.to_k"] = ColwiseParallel()
        plan[f"{prefix}.audio_to_video_attn.to_v"] = ColwiseParallel()
        plan[f"{prefix}.audio_to_video_attn.to_out.0"] = RowwiseParallel()

        # Video→Audio cross-attention (Q: Audio, K,V: Video)
        plan[f"{prefix}.video_to_audio_attn.to_q"] = ColwiseParallel()
        plan[f"{prefix}.video_to_audio_attn.to_k"] = ColwiseParallel()
        plan[f"{prefix}.video_to_audio_attn.to_v"] = ColwiseParallel()
        plan[f"{prefix}.video_to_audio_attn.to_out.0"] = RowwiseParallel()

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

    # Detect actual sharding per attention by inspecting to_q output dim.
    # ColwiseParallel converts to_q.weight to a DTensor sharded on dim 0;
    # its local shape[0] will be full/world_size. Only patch attn.heads
    # for modules whose to_q actually sharded — otherwise unflatten gives
    # the wrong head_dim.
    def _is_sharded(linear):
        if linear is None or not hasattr(linear, "weight"):
            return False
        w = linear.weight
        # DTensor: check placements for a Shard
        try:
            from torch.distributed.tensor import DTensor
            if isinstance(w, DTensor):
                return any(getattr(p, "is_shard", lambda: False)() or
                           p.__class__.__name__ == "Shard" for p in w.placements)
        except ImportError:
            pass
        # Fallback: compare local out dim to expected full
        return False

    for layer in range(N_LAYERS):
        block = model.transformer_blocks[layer]
        for attn_name in ("attn1", "attn2", "audio_to_video_attn"):
            attn = getattr(block, attn_name, None)
            if attn is not None and hasattr(attn, "heads") and _is_sharded(getattr(attn, "to_q", None)):
                attn.heads = new_video_heads
        for attn_name in ("audio_attn1", "audio_attn2", "video_to_audio_attn"):
            attn = getattr(block, attn_name, None)
            if attn is not None and hasattr(attn, "heads") and _is_sharded(getattr(attn, "to_q", None)):
                attn.heads = new_audio_heads

    # Diagnostic: report sharding state of block 0's attentions
    if rank == 0:
        b0 = model.transformer_blocks[0]
        for an in ("attn1", "attn2", "audio_attn1", "audio_attn2",
                   "audio_to_video_attn", "video_to_audio_attn"):
            a = getattr(b0, an, None)
            if a is not None:
                print(f"[ltx2_tp_plan] block0.{an}.to_q sharded={_is_sharded(getattr(a,'to_q',None))} heads={getattr(a,'heads',None)}",
                      flush=True)

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
    """Slice every RoPE module's cos/sin output by this rank's head range,
    AND force-build coords on CPU then move to neuron (eliminates meta leak).

    LTX-2 has FOUR RoPE modules on the top-level transformer:
        - rope                  (video self-attn,  num_attention_heads=32)
        - audio_rope            (audio self-attn,  audio_num_attention_heads=32)
        - cross_attn_rope       (video cross-attn, num_attention_heads=32)
        - cross_attn_audio_rope (audio cross-attn, audio_num_attention_heads=32)

    Each returns cos/sin of shape (B, H, T, D//2) for split rope, or
    (B, T, 2r) for interleaved. After ColwiseParallel shards q/k/v,
    each rank only holds H/world_size heads, so we slice axis 1 of
    cos/sin to [rank*H_local : (rank+1)*H_local].
    """
    rope_attrs = ["rope", "audio_rope", "cross_attn_rope", "cross_attn_audio_rope"]

    # Find the rope class
    rope_cls = None
    for attr in rope_attrs:
        rope = getattr(model, attr, None)
        if rope is not None:
            rope_cls = type(rope)
            break
    if rope_cls is None:
        if rank == 0:
            print(f"[ltx2_tp_plan] no rope class found", flush=True)
        return

    neuron = torch.device("neuron")
    cpu = torch.device("cpu")

    # Per-rope-instance head ranges
    head_ranges = {}
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
        # Mark this instance with its head range
        rope._tp_head_start = rank * h_local
        rope._tp_head_end = rope._tp_head_start + h_local
        head_ranges[attr] = (rope._tp_head_start, rope._tp_head_end)

    # Save class originals once
    if not hasattr(rope_cls, "_orig_prepare_video_coords"):
        rope_cls._orig_prepare_video_coords = rope_cls.prepare_video_coords
        rope_cls._orig_prepare_audio_coords = rope_cls.prepare_audio_coords
        rope_cls._orig_forward = rope_cls.forward

    def _patched_video_coords(self, *args, **kwargs):
        # Build on CPU (eliminates any meta-device leakage), move to neuron
        new_args = [cpu if isinstance(a, torch.device) else a for a in args]
        if "device" in kwargs:
            kwargs["device"] = cpu
        out = rope_cls._orig_prepare_video_coords(self, *new_args, **kwargs)
        return out.to(neuron) if torch.is_tensor(out) else out

    def _patched_audio_coords(self, *args, **kwargs):
        new_args = [cpu if isinstance(a, torch.device) else a for a in args]
        if "device" in kwargs:
            kwargs["device"] = cpu
        out = rope_cls._orig_prepare_audio_coords(self, *new_args, **kwargs)
        return out.to(neuron) if torch.is_tensor(out) else out

    def _patched_forward(self, *args, **kwargs):
        # Move coords positional to CPU for freq computation, force device=cpu,
        # then move outputs to neuron AND slice by this rank's head range.
        new_args = list(args)
        if new_args and torch.is_tensor(new_args[0]):
            new_args[0] = new_args[0].to(cpu)
        if "device" in kwargs:
            kwargs["device"] = cpu
        out = rope_cls._orig_forward(self, *new_args, **kwargs)
        if isinstance(out, tuple) and len(out) == 2 and torch.is_tensor(out[0]):
            cos, sin = out
            cos = cos.to(neuron); sin = sin.to(neuron)
            start = getattr(self, "_tp_head_start", None)
            end = getattr(self, "_tp_head_end", None)
            if start is not None and cos.dim() == 4:
                cos = cos[:, start:end]
                sin = sin[:, start:end]
            return cos, sin
        return out

    rope_cls.prepare_video_coords = _patched_video_coords
    rope_cls.prepare_audio_coords = _patched_audio_coords
    rope_cls.forward = _patched_forward

    if rank == 0:
        print(f"[ltx2_tp_plan] monkey-patched {rope_cls.__name__}: "
              f"coords build on CPU → .to(neuron); per-rank head slice "
              f"applied. ranges={head_ranges}", flush=True)


def _force_rope_device_neuron(model, rope_attrs, rank):
    """Deprecated — kept for backward compat. Class-level patch in
    patch_rope_rank_slice now handles both device coercion AND head slice."""
    pass

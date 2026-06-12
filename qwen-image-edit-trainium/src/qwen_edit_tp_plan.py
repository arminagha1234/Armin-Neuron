"""TP plan + post-hoc TP fixes for QwenImageTransformer2DModel.

Architecture (Qwen-Image-Edit-2511 transformer):
    - 48 attention heads, head_dim = 128, inner_dim = 6144
    - 20 dual-stream blocks (separate image + text attention paths)
    - 40 single-stream blocks (joint attention)
    - Modulation layer for guidance / timestep injection

TP plan rules (mirrors LTX-2.3 v3 in
neuron/examples/LTX/ltx23_pipeline_v3.py):
    - to_q, to_k, to_v             → ColwiseParallel (shard inner_dim)
    - to_out (linear → bias)        → RowwiseParallel (gather along inner_dim)
    - ff.net.0.proj (gated)         → ColwiseParallel
    - ff.net.2 (output projection)  → RowwiseParallel
    - modulation linears            → ColumnParallelLinear with gather
    - final out_proj                → RowwiseParallel

After parallelize_module wraps the modules, we run apply_tp_fixes() to:
    1. Confirm we have the right model class (Plus = multi-image)
    2. Replace nn.RMSNorm in attn.norm_q / norm_k with TPRMSNorm
    3. Patch attn.heads = full_heads // world_size
    4. Wrap RoPE forward to slice cos/sin by rank

References:
    - .kiro/steering/neuron-tp-on-beta2.md (fixes 1-5)
    - neuron/examples/LTX/ltx23_pipeline_v3.py
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn

if TYPE_CHECKING:
    from torch.distributed.tensor.parallel import ParallelStyle  # type: ignore

from rope_functional import slice_rope_for_rank
from tprmsnorm import TPRMSNorm

# Qwen-Image-Edit-2511 transformer architectural constants
# Confirmed from snapshot transformer/config.json (2026-06-11):
#   num_attention_heads: 24
#   attention_head_dim:  128
#   num_layers:          60   (unified — single transformer_blocks list)
#   in_channels:         64
#   out_channels:        16
#   patch_size:          2
#   joint_attention_dim: 3584 (matches Qwen2.5-VL hidden)
#   axes_dims_rope:      [16, 56, 56]   (3D RoPE — t, h, w)
#   guidance_embeds:     False
#   zero_cond_t:         True
N_HEADS_FULL = 24
HEAD_DIM = 128
INNER_DIM = N_HEADS_FULL * HEAD_DIM  # 3072
N_LAYERS = 60


def qwen_edit_tp_plan(world_size: int) -> dict[str, "ParallelStyle"]:
    """Build the parallelize_module plan dict.

    Returns a dict suitable for `parallelize_module(model, mesh, plan)`.
    """
    from torch.distributed.tensor.parallel import (
        ColwiseParallel,
        RowwiseParallel,
    )

    if INNER_DIM % world_size != 0:
        raise ValueError(
            f"inner_dim={INNER_DIM} not divisible by world_size={world_size}"
        )
    if N_HEADS_FULL % world_size != 0:
        raise ValueError(
            f"n_heads={N_HEADS_FULL} not divisible by world_size={world_size}"
        )

    plan: dict[str, ParallelStyle] = {}

    # 60 unified transformer blocks. Per `_inspect.py` on real 2511
    # snapshot, each block has BOTH image and text paths inside one
    # `QwenImageTransformerBlock`:
    #   attn.{to_q, to_k, to_v}                — image-side QKV
    #   attn.{add_q_proj, add_k_proj, add_v_proj} — text-side QKV
    #   attn.to_out.0                          — image-side out
    #   attn.to_add_out                        — text-side out
    #   img_mlp.net.0.proj / img_mlp.net.2     — image-side FFN
    #   txt_mlp.net.0.proj / txt_mlp.net.2     — text-side FFN
    #   img_mod.1 / txt_mod.1                  — modulation linears (replicate)
    for block_idx in range(N_LAYERS):
        prefix = f"transformer_blocks.{block_idx}"
        # Image-side attention QKV → column shard
        plan[f"{prefix}.attn.to_q"] = ColwiseParallel()
        plan[f"{prefix}.attn.to_k"] = ColwiseParallel()
        plan[f"{prefix}.attn.to_v"] = ColwiseParallel()
        # Image-side attention output → row shard
        plan[f"{prefix}.attn.to_out.0"] = RowwiseParallel()
        # Text-side attention QKV → column shard
        plan[f"{prefix}.attn.add_q_proj"] = ColwiseParallel()
        plan[f"{prefix}.attn.add_k_proj"] = ColwiseParallel()
        plan[f"{prefix}.attn.add_v_proj"] = ColwiseParallel()
        # Text-side attention output → row shard
        plan[f"{prefix}.attn.to_add_out"] = RowwiseParallel()
        # Image-side FFN
        plan[f"{prefix}.img_mlp.net.0.proj"] = ColwiseParallel()
        plan[f"{prefix}.img_mlp.net.2"] = RowwiseParallel()
        # Text-side FFN
        plan[f"{prefix}.txt_mlp.net.0.proj"] = ColwiseParallel()
        plan[f"{prefix}.txt_mlp.net.2"] = RowwiseParallel()
        # img_mod.1 / txt_mod.1 are modulation linears; we leave them
        # replicated (they project from a small embedding to inner_dim,
        # cost is negligible and replicating avoids a DTensor placement
        # for a tiny tensor).

    return plan


def apply_tp_fixes(
    transformer: nn.Module,
    *,
    world_size: int,
    rank: int,
) -> None:
    """Post-hoc TP fixes (run AFTER parallelize_module).

    Mutates transformer in place:
        - validates model class (fix 1)
        - swaps RMSNorm for TPRMSNorm on each attention's norm_q / norm_k (fix 2)
        - patches attn.heads to heads/N (fix 3)
        - wraps RoPE forward to slice by rank (fix 4)

    Functional RoPE (fix 5) is applied at module construction time via
    monkey-patch in run_native.py — see the apply_split_rotary_emb hook.
    """
    # Fix 1: model class sanity check
    cls_name = type(transformer).__name__
    if cls_name != "QwenImageTransformer2DModel":
        raise RuntimeError(
            f"Expected QwenImageTransformer2DModel, got {cls_name}. "
            "Confirm via model_index.json — wrong class compiles but produces "
            "wrong architecture."
        )

    # Fix 2 + 3: per-attention swaps
    heads_per_rank = N_HEADS_FULL // world_size
    for name, module in transformer.named_modules():
        # Identify attention modules — diffusers uses Attention class
        if not hasattr(module, "to_q") or not hasattr(module, "to_v"):
            continue

        # Fix 3: heads/N
        if hasattr(module, "heads"):
            module.heads = heads_per_rank

        # Fix 2: TP-aware RMSNorm — but ONLY if the RMSNorm normalizes
        # the sharded (inner_dim) channel.
        #
        # In Qwen-Image-Edit-2511 the norm_q / norm_k / norm_added_q /
        # norm_added_k weights are shape [head_dim] = [128], not
        # [inner_dim] = [3072]. They are applied PER-HEAD (after the
        # tensor is reshaped to (batch, seq, n_heads, head_dim)). The
        # head_dim axis is NOT split across TP ranks (inner_dim is split
        # by giving each rank fewer heads, but each head's full
        # head_dim stays on one rank). So the standard nn.RMSNorm(128)
        # works untouched — every rank holds the same per-head norm.
        #
        # We keep the TPRMSNorm code in the repo because if a future
        # variant uses inner_dim-shaped norms, it'll be needed.
        for norm_name in ("norm_q", "norm_k", "norm_added_q", "norm_added_k"):
            existing = getattr(module, norm_name, None)
            if existing is None:
                continue
            # Inspect the existing weight shape. If it's head_dim-sized,
            # this is a per-head norm — keep stock nn.RMSNorm.
            full_dim = INNER_DIM
            local_dim = full_dim // world_size
            if existing.weight is not None:
                w_size = existing.weight.numel()
                if w_size == HEAD_DIM:
                    # Per-head norm. Nothing to do; stock RMSNorm(head_dim)
                    # operates on the unshared head_dim axis.
                    continue
                if w_size != full_dim and w_size != local_dim:
                    raise RuntimeError(
                        f"{name}.{norm_name}.weight has unexpected "
                        f"shape {existing.weight.shape} (expected "
                        f"{HEAD_DIM}, {local_dim} or {full_dim})"
                    )
            tp_norm = TPRMSNorm(
                full_dim=full_dim,
                world_size=world_size,
                eps=getattr(existing, "eps", 1e-6),
                elementwise_affine=existing.weight is not None,
            )
            if existing.weight is not None:
                with torch.no_grad():
                    full_weight = existing.weight.detach()
                    if full_weight.numel() == local_dim:
                        tp_norm.weight.copy_(full_weight)
                    elif full_weight.numel() == full_dim:
                        start = rank * local_dim
                        end = start + local_dim
                        tp_norm.weight.copy_(full_weight[start:end])
            tp_norm = tp_norm.to(
                device=existing.weight.device if existing.weight is not None else "cpu",
                dtype=existing.weight.dtype if existing.weight is not None else torch.bfloat16,
            )
            setattr(module, norm_name, tp_norm)


def install_rope_slice_hook(
    rope_module: nn.Module,
    *,
    rank: int,
    world_size: int,
) -> None:
    """Wrap a RoPE module's forward to slice cos/sin by rank.

    Diffusers RoPE typically returns (cos, sin) shaped
    [B, H_full, T, head_dim/2]. Each rank only needs
    [B, H_full / N, T, head_dim/2] matching its head shard.
    """
    original_forward = rope_module.forward

    def sliced_forward(*args, **kwargs):
        cos, sin = original_forward(*args, **kwargs)
        cos_s, sin_s = slice_rope_for_rank(
            cos, sin, rank=rank, world_size=world_size, head_dim=1
        )
        return cos_s, sin_s

    rope_module.forward = sliced_forward  # type: ignore[assignment]

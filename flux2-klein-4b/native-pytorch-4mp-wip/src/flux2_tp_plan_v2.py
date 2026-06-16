"""TP plan v2 — only shard ATTENTION linears, leave FFN replicated.

The original plan sharded the SwiGLU FFN's `linear_in`, which produces
`[gate; value]` concatenated. ColwiseParallel splits this on the last
dim, so each rank gets a mix of gate/value columns instead of half-each.
The `Flux2SwiGLU` then does `chunk(2, dim=-1)` and multiplies wrong pairs.
That's the root cause of std=45 with TP>1.

This v2 plan:
  - Shards attention Q/K/V/O (Colwise/Rowwise) — same as before
  - Does NOT shard FFN linear_in/linear_out — each rank computes the
    full FFN, slower but correct.
  - For single-stream blocks, also does NOT shard `to_qkv_mlp_proj` because
    it fuses the GLU MLP in. Single-stream blocks fall back to per-rank
    full compute. (We can revisit with a custom `to_qkv_mlp_proj` shard
    later, but correctness first.)

Trade-off vs the v1 plan: FFN compute and inner-dim memory are NOT halved.
But attention scores ARE halved (the bigger memory pressure at high res).
For 4 MP we still get the activation-memory win on attention, which is
what unblocks higher resolutions in the first place.
"""
from __future__ import annotations

import torch

N_DOUBLE = 5
N_SINGLE = 20
N_HEADS_FULL = 24
HEAD_DIM = 128
INNER_DIM = N_HEADS_FULL * HEAD_DIM  # 3072


def flux2_tp_plan_v2(world_size: int) -> dict:
    """Attention-only TP plan. FFN runs replicated on every rank."""
    from torch.distributed.tensor.parallel import (
        ColwiseParallel, RowwiseParallel,
    )

    plan = {}

    # Double-stream blocks: shard ONLY attention Q/K/V/O, not FFN
    for i in range(N_DOUBLE):
        p = f"transformer_blocks.{i}"
        plan[f"{p}.attn.to_q"] = ColwiseParallel()
        plan[f"{p}.attn.to_k"] = ColwiseParallel()
        plan[f"{p}.attn.to_v"] = ColwiseParallel()
        plan[f"{p}.attn.to_out.0"] = RowwiseParallel()
        plan[f"{p}.attn.add_q_proj"] = ColwiseParallel()
        plan[f"{p}.attn.add_k_proj"] = ColwiseParallel()
        plan[f"{p}.attn.add_v_proj"] = ColwiseParallel()
        plan[f"{p}.attn.to_add_out"] = RowwiseParallel()
        # NOTE: ff.linear_in / ff.linear_out NOT sharded — see docstring
        # NOTE: ff_context.linear_in / linear_out NOT sharded — same reason

    # Single-stream blocks: SKIP entirely — `to_qkv_mlp_proj` fuses the
    # GLU MLP, can't safely shard with simple ColwiseParallel.
    # TODO: revisit with custom shard that splits gate/value separately.

    return plan


def _is_sharded(linear) -> bool:
    if linear is None or not hasattr(linear, "weight"):
        return False
    w = linear.weight
    try:
        from torch.distributed.tensor import DTensor
        if isinstance(w, DTensor):
            return any(p.__class__.__name__ == "Shard" for p in w.placements)
    except ImportError:
        pass
    return False


def apply_tp_fixes_v2(model, world_size: int, rank: int) -> None:
    """Patch attn.heads to heads/N on the SHARDED attention modules only.

    Single-stream blocks are not sharded in v2, so don't touch their attn.heads.
    """
    new_heads = N_HEADS_FULL // world_size

    patched = 0
    # Double-stream — sharded
    for i in range(N_DOUBLE):
        block = model.transformer_blocks[i]
        attn = getattr(block, "attn", None)
        if attn is not None and hasattr(attn, "heads") and _is_sharded(
            getattr(attn, "to_q", None)
        ):
            attn.heads = new_heads
            patched += 1

    # Single-stream — NOT sharded in v2, leave heads alone

    if rank == 0:
        print(f"[flux2_tp_v2] patched attn.heads -> {new_heads} on "
              f"{patched} double-stream blocks (single-stream NOT sharded)",
              flush=True)


def flux2_tp_plan_empty(world_size):
    # Control: shard NOTHING. Tests if TP sharding is the high-res bug.
    return {}

def apply_tp_fixes_empty(model, world_size, rank):
    if rank == 0:
        print('[flux2_tp_empty] no sharding, no head patch', flush=True)

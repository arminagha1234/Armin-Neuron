"""TP=4 plan + Neuron-correctness fixes for FLUX.2-klein-4B (diffusers).

Adapts the proven LTX-2 TP recipe (neuron/examples/ltx2/.../ltx2_tp_plan.py)
to the diffusers Flux2 transformer.

Architecture (klein-4B, from transformer_flux2.py + config):
  transformer_blocks.{0-7}        double-stream (Flux2Attention)
  single_transformer_blocks.{0-47} single-stream (Flux2ParallelSelfAttention)
  num_attention_heads = 48, attention_head_dim = 128, inner_dim = 6144

Double-stream attention submodules:
  attn.to_q / to_k / to_v               Colwise  (separate projections)
  attn.to_out.0                         Rowwise
  attn.add_q_proj / add_k_proj / add_v_proj  Colwise
  attn.to_add_out                       Rowwise
  ff.linear_in / ff_context.linear_in   Colwise
  ff.linear_out / ff_context.linear_out Rowwise

Single-stream attention submodules:
  attn.to_qkv_mlp_proj                  Colwise (FUSED qkv+mlp-gate)
  attn.to_out                           Rowwise (FUSED attn-out + mlp-out)

Fixes (same numbering as the LTX-2 recipe):
  3. TP=4 plan (this file)
  4. (klein qk_norm is per-head RMSNorm on head_dim=128, NOT across the
     sharded inner_dim — so it does NOT need the adaptive all-reduce
     norm. head_dim is unchanged by sharding; only head COUNT shards.
     This is simpler than LTX-2.)
  6. attn.heads patched to heads/N (sharding-aware)
  layout. The diffusers processor does query.unflatten(-1, (attn.heads, -1)),
  so after Colwise on to_q the local width is inner_dim/N and attn.heads
  must be heads/N for unflatten to recover head_dim=128.
"""
from __future__ import annotations

import torch

N_DOUBLE = 5
N_SINGLE = 20
N_HEADS_FULL = 24
HEAD_DIM = 128
INNER_DIM = N_HEADS_FULL * HEAD_DIM  # 3072


def flux2_tp_plan(world_size: int) -> dict:
    """parallelize_module plan keyed by submodule path."""
    from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel

    plan = {}

    # Double-stream blocks
    for i in range(N_DOUBLE):
        p = f"transformer_blocks.{i}"
        # main stream attention
        plan[f"{p}.attn.to_q"] = ColwiseParallel()
        plan[f"{p}.attn.to_k"] = ColwiseParallel()
        plan[f"{p}.attn.to_v"] = ColwiseParallel()
        plan[f"{p}.attn.to_out.0"] = RowwiseParallel()
        # added (context/text) stream projections
        plan[f"{p}.attn.add_q_proj"] = ColwiseParallel()
        plan[f"{p}.attn.add_k_proj"] = ColwiseParallel()
        plan[f"{p}.attn.add_v_proj"] = ColwiseParallel()
        plan[f"{p}.attn.to_add_out"] = RowwiseParallel()
        # feed-forwards (img + context)
        plan[f"{p}.ff.linear_in"] = ColwiseParallel()
        plan[f"{p}.ff.linear_out"] = RowwiseParallel()
        plan[f"{p}.ff_context.linear_in"] = ColwiseParallel()
        plan[f"{p}.ff_context.linear_out"] = RowwiseParallel()

    # Single-stream blocks — fused qkv+mlp projection
    for i in range(N_SINGLE):
        p = f"single_transformer_blocks.{i}"
        plan[f"{p}.attn.to_qkv_mlp_proj"] = ColwiseParallel()
        plan[f"{p}.attn.to_out"] = RowwiseParallel()

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


def apply_tp_fixes(model, world_size: int, rank: int) -> None:
    """Patch attn.heads to heads/N on every attention module after sharding.

    Both Flux2Attention (double) and Flux2ParallelSelfAttention (single)
    have a `.heads` attribute used by their processors for
    query.unflatten(-1, (attn.heads, -1)). After Colwise sharding, each
    rank's local projection width is inner_dim/N, so heads must be
    heads/N to keep head_dim=128.
    """
    new_heads = N_HEADS_FULL // world_size

    patched = 0
    # Double-stream
    for i in range(N_DOUBLE):
        block = model.transformer_blocks[i]
        attn = getattr(block, "attn", None)
        if attn is not None and hasattr(attn, "heads") and _is_sharded(
            getattr(attn, "to_q", None)
        ):
            attn.heads = new_heads
            patched += 1
    # Single-stream — also patch inner_dim + mlp_hidden_dim because the
    # processor splits the fused projection by these sizes:
    #   torch.split(x, [3*inner_dim, mlp_hidden_dim*mlp_mult_factor], -1)
    # After Colwise sharding the local width is 1/N, so the split sizes
    # must shard too.
    for i in range(N_SINGLE):
        block = model.single_transformer_blocks[i]
        attn = getattr(block, "attn", None)
        if attn is not None and hasattr(attn, "heads") and _is_sharded(
            getattr(attn, "to_qkv_mlp_proj", None)
        ):
            attn.heads = new_heads
            if hasattr(attn, "inner_dim"):
                attn.inner_dim = attn.inner_dim // world_size
            if hasattr(attn, "mlp_hidden_dim"):
                attn.mlp_hidden_dim = attn.mlp_hidden_dim // world_size
            patched += 1

    if rank == 0:
        print(f"[flux2_tp] patched attn.heads -> {new_heads} on {patched} "
              f"blocks (expected {N_DOUBLE + N_SINGLE})", flush=True)

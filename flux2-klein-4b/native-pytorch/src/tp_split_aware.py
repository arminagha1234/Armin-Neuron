"""Split-aware tensor parallelism for FLUX.2 single-stream blocks.

Background
----------

FLUX.2 has 48 single-stream "parallel" transformer blocks (the ViT-22B
style design where attention QKV and MLP-in are computed by a single
fused Linear, and the attention output + MLP output go through a second
fused Linear). The fused projections look like:

    to_qkv_mlp_proj : [d]              -> [3*d_inner + 2*d_mlp]
    to_out          : [d_inner + d_mlp] -> [d_out]

Vanilla `parallelize_module` with `ColwiseParallel` shards both Linears
on the output dimension, which sounds correct, but the **forward path**
splits the to_qkv_mlp_proj output into Q, K, V, and MLP segments. After
naive ColwiseParallel sharding, each rank holds an equal slice of the
full output dim — which means each rank holds part of Q's last channels,
part of K's last channels, part of V's last channels, and part of MLP.
The `torch.split([3*d_inner, 2*d_mlp])` then produces wrong slices on
each rank, the `chunk(3)` returns wrong Q/K/V boundaries, and attention
math degrades silently.

What we do here
---------------

`ShardedFlux2ParallelSelfAttention` is a drop-in replacement for the
diffusers `Flux2ParallelSelfAttention` that does sharding correctly:

    - to_qkv_mlp_proj output dim is logically [Q | K | V | MLP_gate | MLP_value]
      (the SwiGLU input is doubled, hence mlp_mult_factor=2).
    - On rank r of N, we hold:
        Q[r*d_inner//N : (r+1)*d_inner//N]
        K[r*d_inner//N : (r+1)*d_inner//N]
        V[r*d_inner//N : (r+1)*d_inner//N]
        MLP_gate[r*d_mlp//N : (r+1)*d_mlp//N]
        MLP_value[r*d_mlp//N : (r+1)*d_mlp//N]
    - to_out input dim is [attn | mlp]. Each rank holds the matching
      slice of attn (heads/N) and mlp (d_mlp/N) and produces a partial
      output. We `all_reduce` to combine.

This is morphologically identical to NxDI's `MergedColumnParallelLinear`
plus the matching `RowParallelLinear`, but expressed at the
diffusers-attention level so we don't depend on NxDI.

Usage
-----

After model construction, call:

    from tp_split_aware import apply_split_aware_tp
    apply_split_aware_tp(transformer, mesh)

where `mesh` is a 1D `init_device_mesh('neuron', (world_size,))`. This
walks every `Flux2ParallelSelfAttention` module, replaces it with the
split-aware version, and slices the weights accordingly.

Performance expectation
-----------------------

On trn2.3xl with TP=2 inside a single LNC=2 logical core (or
NEURON_RT_VIRTUAL_CORE_SIZE=2 physical world), this gives roughly
1.6-1.8x per-step speedup vs single core. Combined with batch
parallelism (2 procs x TP=2 spread across the 4 physical cores via
LNC=1) this would target ~$0.012/image, but LNC=1 requires a host
driver reconfig that is not currently exposed as a runtime tunable on
trn2 instances (verified 2026-06-13).

WARNING: untested at the time of writing. The shapes and slicing match
the diffusers transformer_flux2.py source as of 0.39.0.dev. Use the
companion `tp_split_aware_smoke.py` test before relying on this in
production.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


def _slice_for_rank(x: torch.Tensor, dim: int, rank: int, world: int) -> torch.Tensor:
    """Return the rank-th slice of x along dim, evenly split across world."""
    n = x.shape[dim]
    assert n % world == 0, f"dim {dim} of size {n} not divisible by {world}"
    chunk = n // world
    start = rank * chunk
    end = start + chunk
    return x.narrow(dim, start, end - start).contiguous()


class ShardedFlux2ParallelSelfAttention(nn.Module):
    """Split-aware TP version of Flux2ParallelSelfAttention.

    Holds the local slice of:
        - to_qkv_mlp_proj: 5 segments [Q, K, V, MLP_gate, MLP_value],
          each sharded by output channel.
        - to_out: input is [attn | mlp], each sharded matching above.

    The forward `all_reduce`s the to_out output across the TP group.
    """

    def __init__(
        self,
        original: nn.Module,
        rank: int,
        world: int,
        process_group: Optional[dist.ProcessGroup] = None,
    ):
        super().__init__()
        self.rank = rank
        self.world = world
        self.process_group = process_group

        # Copy attributes the processor reads.
        self.head_dim = original.head_dim
        self.inner_dim = original.inner_dim
        self.query_dim = original.query_dim
        self.out_dim = original.out_dim
        self.use_bias = original.use_bias
        self.dropout = original.dropout
        self.mlp_ratio = original.mlp_ratio
        self.mlp_hidden_dim = original.mlp_hidden_dim
        self.mlp_mult_factor = original.mlp_mult_factor
        self.heads = original.heads

        # Sharded sizes
        assert self.inner_dim % world == 0, \
            f"inner_dim={self.inner_dim} not divisible by world={world}"
        assert self.mlp_hidden_dim % world == 0, \
            f"mlp_hidden_dim={self.mlp_hidden_dim} not divisible by world={world}"
        assert self.heads % world == 0, \
            f"heads={self.heads} not divisible by world={world}"

        d_in = self.query_dim
        d_inner_local = self.inner_dim // world
        d_mlp_local = self.mlp_hidden_dim // world

        # to_qkv_mlp_proj: split the original [3*d_inner + mlp_mult * d_mlp]
        # output into 5 segments and slice each on its inner dim.
        # Original weight: [out, in], where out = 3*d_inner + mlp_mult*d_mlp.
        orig_w = original.to_qkv_mlp_proj.weight    # [out, in]
        orig_b = original.to_qkv_mlp_proj.bias       # [out] or None

        # Segment sizes in the ORIGINAL fused output:
        # [Q (d_inner) | K (d_inner) | V (d_inner) | MLP_gate (d_mlp) | MLP_value (d_mlp)]
        q_w = orig_w[0 * self.inner_dim:1 * self.inner_dim, :]
        k_w = orig_w[1 * self.inner_dim:2 * self.inner_dim, :]
        v_w = orig_w[2 * self.inner_dim:3 * self.inner_dim, :]
        mlp_offset = 3 * self.inner_dim
        mlp_gate_w = orig_w[mlp_offset + 0 * self.mlp_hidden_dim
                            : mlp_offset + 1 * self.mlp_hidden_dim, :]
        mlp_val_w = orig_w[mlp_offset + 1 * self.mlp_hidden_dim
                           : mlp_offset + 2 * self.mlp_hidden_dim, :]

        # Slice each segment on its inner channel dim for this rank.
        q_local = _slice_for_rank(q_w, dim=0, rank=rank, world=world)
        k_local = _slice_for_rank(k_w, dim=0, rank=rank, world=world)
        v_local = _slice_for_rank(v_w, dim=0, rank=rank, world=world)
        mg_local = _slice_for_rank(mlp_gate_w, dim=0, rank=rank, world=world)
        mv_local = _slice_for_rank(mlp_val_w, dim=0, rank=rank, world=world)

        # Concatenate in the same QKV-MLP order so a single matmul gives
        # us the local [Q_local | K_local | V_local | MLP_gate_local | MLP_value_local].
        local_w = torch.cat([q_local, k_local, v_local, mg_local, mv_local], dim=0)
        self.to_qkv_mlp_proj_weight = nn.Parameter(local_w, requires_grad=False)

        if orig_b is not None:
            q_b = orig_b[0 * self.inner_dim:1 * self.inner_dim]
            k_b = orig_b[1 * self.inner_dim:2 * self.inner_dim]
            v_b = orig_b[2 * self.inner_dim:3 * self.inner_dim]
            mg_b = orig_b[mlp_offset + 0 * self.mlp_hidden_dim
                          : mlp_offset + 1 * self.mlp_hidden_dim]
            mv_b = orig_b[mlp_offset + 1 * self.mlp_hidden_dim
                          : mlp_offset + 2 * self.mlp_hidden_dim]

            q_b_local = _slice_for_rank(q_b, dim=0, rank=rank, world=world)
            k_b_local = _slice_for_rank(k_b, dim=0, rank=rank, world=world)
            v_b_local = _slice_for_rank(v_b, dim=0, rank=rank, world=world)
            mg_b_local = _slice_for_rank(mg_b, dim=0, rank=rank, world=world)
            mv_b_local = _slice_for_rank(mv_b, dim=0, rank=rank, world=world)
            local_b = torch.cat([q_b_local, k_b_local, v_b_local, mg_b_local, mv_b_local], dim=0)
            self.to_qkv_mlp_proj_bias = nn.Parameter(local_b, requires_grad=False)
        else:
            self.register_parameter("to_qkv_mlp_proj_bias", None)

        # to_out: original is Linear([d_inner + d_mlp] -> d_out).
        # The input layout is [attn (d_inner) | mlp (d_mlp)]. After our
        # local-output forward, we have [d_inner_local + d_mlp_local].
        # We need a row-parallel-style matmul: input is sharded, output
        # is full d_out, then all_reduce across TP.
        orig_out_w = original.to_out.weight    # [d_out, d_inner + d_mlp]
        orig_out_b = original.to_out.bias       # [d_out] or None

        attn_part = orig_out_w[:, :self.inner_dim]    # [d_out, d_inner]
        mlp_part = orig_out_w[:, self.inner_dim:]      # [d_out, d_mlp]

        attn_local = _slice_for_rank(attn_part, dim=1, rank=rank, world=world)
        mlp_local = _slice_for_rank(mlp_part, dim=1, rank=rank, world=world)
        local_out_w = torch.cat([attn_local, mlp_local], dim=1)
        self.to_out_weight = nn.Parameter(local_out_w, requires_grad=False)

        if orig_out_b is not None:
            # Bias is added once after all_reduce; only rank 0 holds it
            # (or we divide by world; we choose rank-0 for clarity).
            if rank == 0:
                self.to_out_bias = nn.Parameter(orig_out_b.clone(), requires_grad=False)
            else:
                self.to_out_bias = nn.Parameter(torch.zeros_like(orig_out_b), requires_grad=False)
        else:
            self.register_parameter("to_out_bias", None)

        # Norms — these are per-channel and don't need sharding because
        # they operate on the head_dim, not inner_dim.
        self.norm_q = original.norm_q
        self.norm_k = original.norm_k

        # MLP activation (Flux2SwiGLU) is element-wise — no sharding.
        self.mlp_act_fn = original.mlp_act_fn

        # Use the original processor; we patch the few attribute reads
        # below so it sees per-rank shapes where needed.
        self.processor = original.processor
        self._attention_backend = getattr(original, "_attention_backend", None)
        self._parallel_config = getattr(original, "_parallel_config", None)

    # --- Properties the original processor reads ---

    # heads must be the per-rank head count so unflatten gives correct shape.
    # (We override `heads` directly; mlp_hidden_dim and inner_dim are
    # only used by the processor for split sizes.)
    # We expose two views: the global ones (for the diffusers split logic)
    # and per-rank ones (for matrix shapes).
    # Easiest: we re-implement the forward locally, bypassing the original
    # processor's split logic, since the local matmul directly produces
    # the correctly-ordered local QKV+MLP segments.

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[tuple] = None,
        kv_cache=None,
        kv_cache_mode: Optional[str] = None,
        num_txt_tokens: int = 0,
        num_ref_tokens: int = 0,
        **kwargs,
    ) -> torch.Tensor:
        from diffusers.models.embeddings import apply_rotary_emb
        from diffusers.models.attention_dispatch import dispatch_attention_fn

        d_inner_local = self.inner_dim // self.world
        d_mlp_local = self.mlp_hidden_dim // self.world

        # Local fused matmul: [B, T, d_in] -> [B, T, 3*d_inner_local + 2*d_mlp_local]
        proj = F.linear(hidden_states, self.to_qkv_mlp_proj_weight,
                        self.to_qkv_mlp_proj_bias)

        qkv, mlp = torch.split(proj, [3 * d_inner_local, 2 * d_mlp_local], dim=-1)
        query, key, value = qkv.chunk(3, dim=-1)

        # heads are sharded — local heads per rank
        local_heads = self.heads // self.world
        query = query.unflatten(-1, (local_heads, self.head_dim))
        key = key.unflatten(-1, (local_heads, self.head_dim))
        value = value.unflatten(-1, (local_heads, self.head_dim))

        query = self.norm_q(query)
        key = self.norm_k(key)

        if image_rotary_emb is not None:
            # RoPE freqs are typically [B, T, head_dim/2] (real-arith form
            # in the Beta 3 patches we use). Heads sharding doesn't affect
            # the head_dim-only RoPE freqs, so we apply directly.
            query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
            key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)

        attn_output = dispatch_attention_fn(
            query, key, value, attn_mask=attention_mask,
            backend=self._attention_backend,
            parallel_config=self._parallel_config,
        )
        attn_output = attn_output.flatten(2, 3)    # [B, T, local_heads * head_dim]
        attn_output = attn_output.to(query.dtype)

        mlp_hidden = self.mlp_act_fn(mlp)    # SwiGLU collapses 2*d_mlp -> d_mlp

        local_concat = torch.cat([attn_output, mlp_hidden], dim=-1)
        # [B, T, d_inner_local + d_mlp_local]

        local_out = F.linear(local_concat, self.to_out_weight, self.to_out_bias)
        # local_out: [B, T, d_out] with PARTIAL contribution

        if self.world > 1 and self.process_group is not None:
            dist.all_reduce(local_out, op=dist.ReduceOp.SUM, group=self.process_group)

        return local_out


def apply_split_aware_tp(transformer: nn.Module, mesh) -> int:
    """Walk the transformer, replace every Flux2ParallelSelfAttention
    with the split-aware version. Returns the number of modules replaced.
    """
    from diffusers.models.transformers.transformer_flux2 import Flux2SingleTransformerBlock

    rank = mesh.get_local_rank()
    world = mesh.size()
    pg = mesh.get_group()

    replaced = 0
    for block in transformer.modules():
        if isinstance(block, Flux2SingleTransformerBlock):
            original_attn = block.attn
            block.attn = ShardedFlux2ParallelSelfAttention(
                original_attn, rank=rank, world=world, process_group=pg,
            )
            replaced += 1
    return replaced

"""TP plan v3 — shard EVERYTHING correctly: attention + SwiGLU FFN +
single-stream fused blocks.

v2 sharded only the 5 double-stream blocks' attention and left the 20
single-stream blocks + all FFNs replicated. In fp32 the replicated
activation is the OOM driver at >=3MP. v3 splits the fused projections
into separately-shardable linears (Llama-style SwiGLU split + NxDI
separate-out-proj) so the bulk of the model shards and fp32 fits at
high resolution.

Pipeline:
    import flux2_tp_plan_v3 as v3
    v3.restructure_for_tp(inner)                 # split fused linears (CPU)
    parallelize_module(inner, mesh, v3.flux2_tp_plan_v3(world_size))
    v3.apply_tp_fixes_v3(inner, world_size, rank)

restructure_for_tp MUST run before .to(device) and before
parallelize_module. It is weight-preserving (verified by CPU forward
equivalence in flux2_v3_selftest.py).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

N_DOUBLE = 5
N_SINGLE = 20
N_HEADS_FULL = 24
HEAD_DIM = 128
INNER_DIM = N_HEADS_FULL * HEAD_DIM  # 3072


def _split_linear(fused: nn.Linear, sizes, bias=False):
    """Split a fused Linear's out_features into separate Linears, each
    holding a contiguous slice of the original weight rows."""
    outs, off = [], 0
    has_bias = fused.bias is not None
    for s in sizes:
        lin = nn.Linear(fused.in_features, s, bias=has_bias)
        lin.weight.data = fused.weight.data[off:off + s].clone()
        if has_bias:
            lin.bias.data = fused.bias.data[off:off + s].clone()
        lin.to(fused.weight.dtype)
        outs.append(lin)
        off += s
    return outs


# ---------------------------------------------------------------------------
# Double-stream FFN: split fused linear_in [gate;value] -> gate_proj+value_proj
# ---------------------------------------------------------------------------
def _restructure_ffn(ff):
    """Flux2FeedForward: linear_in(dim->2*inner) -> gate_proj+value_proj."""
    if getattr(ff, "_v3_split", False):
        return
    inner = ff.linear_out.in_features          # 9216
    gate_proj, value_proj = _split_linear(ff.linear_in, [inner, inner])
    ff.gate_proj = gate_proj
    ff.value_proj = value_proj
    ff._silu = nn.SiLU()
    del ff.linear_in
    ff._v3_split = True


def _ffn_forward_v3(self, x):
    if getattr(self, "_v3_split", False):
        g = self.gate_proj(x)
        v = self.value_proj(x)
        return self.linear_out(self._silu(g) * v)
    # fallback to original
    x = self.linear_in(x)
    x = self.act_fn(x)
    return self.linear_out(x)


# ---------------------------------------------------------------------------
# Single-stream: split fused to_qkv_mlp_proj + to_out
# ---------------------------------------------------------------------------
def _restructure_single(attn):
    """Flux2ParallelSelfAttention: split the fused QKV+MLP-in and the
    fused attn+mlp out projection."""
    if getattr(attn, "_v3_split", False):
        return
    inner = attn.inner_dim                                  # 3072
    mlp_total = attn.mlp_hidden_dim * attn.mlp_mult_factor  # gate+value
    mlp_half = mlp_total // 2                               # mlp_hidden_dim
    # to_qkv_mlp_proj out layout: [q(inner), k(inner), v(inner),
    #                              mlp_gate(mlp_half), mlp_value(mlp_half)]
    q, k, v, mg, mv = _split_linear(
        attn.to_qkv_mlp_proj, [inner, inner, inner, mlp_half, mlp_half]
    )
    attn.to_q_s, attn.to_k_s, attn.to_v_s = q, k, v
    attn.mlp_gate, attn.mlp_value = mg, mv
    attn._silu = nn.SiLU()

    # to_out: Linear(inner + mlp_half -> dim). Split input dim into the
    # attn part (inner) and mlp part (mlp_half) -> two RowwiseParallel
    # linears whose (replicated) outputs we sum.
    dim = attn.to_out.out_features
    w = attn.to_out.weight.data           # [dim, inner+mlp_half]
    attn_out_proj = nn.Linear(inner, dim, bias=attn.to_out.bias is not None)
    mlp_out_proj = nn.Linear(mlp_half, dim, bias=False)
    attn_out_proj.weight.data = w[:, :inner].clone()
    mlp_out_proj.weight.data = w[:, inner:inner + mlp_half].clone()
    if attn.to_out.bias is not None:
        attn_out_proj.bias.data = attn.to_out.bias.data.clone()
    attn_out_proj.to(w.dtype)
    mlp_out_proj.to(w.dtype)
    attn.attn_out_proj = attn_out_proj
    attn.mlp_out_proj = mlp_out_proj
    del attn.to_qkv_mlp_proj
    del attn.to_out
    attn._v3_split = True


def _install_single_processor_v3():
    """Patch Flux2ParallelSelfAttnProcessor.__call__ to use split linears."""
    from diffusers.models.transformers.transformer_flux2 import (
        Flux2ParallelSelfAttnProcessor,
    )
    from diffusers.models.embeddings import apply_rotary_emb

    if getattr(Flux2ParallelSelfAttnProcessor, "_v3_installed", False):
        return

    def call_v3(self, attn, hidden_states, attention_mask=None,
                image_rotary_emb=None):
        if not getattr(attn, "_v3_split", False):
            raise RuntimeError("v3 processor used on unsplit attn")

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

        # attention via the installed flash/CTE path or SDPA fallback
        from flux2_attention_manual_flash import manual_flash_attention
        attn_out = manual_flash_attention(q, k, v, attn_mask=attention_mask)
        attn_out = attn_out.flatten(2, 3).to(q.dtype)

        mlp_out = attn._silu(g) * m

        # separate out-projections, summed (each Rowwise all-reduces)
        out = attn.attn_out_proj(attn_out) + attn.mlp_out_proj(mlp_out)
        return out

    Flux2ParallelSelfAttnProcessor.__call__ = call_v3
    Flux2ParallelSelfAttnProcessor._v3_installed = True


def restructure_for_tp(model, rank: int = 0):
    """Split fused linears on all blocks. Run on CPU before parallelize."""
    from diffusers.models.transformers.transformer_flux2 import (
        Flux2FeedForward, Flux2ParallelSelfAttention, Flux2TransformerBlock,
        Flux2SingleTransformerBlock,
    )
    # patch FFN forward at class level (idempotent)
    if not getattr(Flux2FeedForward, "_v3_fwd", False):
        Flux2FeedForward.forward = _ffn_forward_v3
        Flux2FeedForward._v3_fwd = True
    _install_single_processor_v3()

    n_ff = n_single = 0
    for m in model.modules():
        if isinstance(m, Flux2FeedForward):
            _restructure_ffn(m); n_ff += 1
        elif isinstance(m, Flux2ParallelSelfAttention):
            _restructure_single(m); n_single += 1
    if rank == 0:
        print(f"[v3] restructured {n_ff} FFNs + {n_single} single-stream "
              f"attns for sharding", flush=True)


def flux2_tp_plan_v3(world_size: int) -> dict:
    from torch.distributed.tensor.parallel import (
        ColwiseParallel, RowwiseParallel,
    )
    plan = {}
    # Double-stream: attention + split FFN
    for i in range(N_DOUBLE):
        p = f"transformer_blocks.{i}"
        for proj in ["to_q", "to_k", "to_v", "add_q_proj", "add_k_proj",
                     "add_v_proj"]:
            plan[f"{p}.attn.{proj}"] = ColwiseParallel()
        plan[f"{p}.attn.to_out.0"] = RowwiseParallel()
        plan[f"{p}.attn.to_add_out"] = RowwiseParallel()
        # split FFN (img + context)
        for ff in ["ff", "ff_context"]:
            plan[f"{p}.{ff}.gate_proj"] = ColwiseParallel()
            plan[f"{p}.{ff}.value_proj"] = ColwiseParallel()
            plan[f"{p}.{ff}.linear_out"] = RowwiseParallel()

    # Single-stream: split QKV + MLP + dual out-proj
    for i in range(N_SINGLE):
        p = f"single_transformer_blocks.{i}.attn"
        for proj in ["to_q_s", "to_k_s", "to_v_s", "mlp_gate", "mlp_value"]:
            plan[f"{p}.{proj}"] = ColwiseParallel()
        plan[f"{p}.attn_out_proj"] = RowwiseParallel()
        plan[f"{p}.mlp_out_proj"] = RowwiseParallel()

    return plan


def _is_sharded(linear) -> bool:
    if linear is None or not hasattr(linear, "weight"):
        return False
    try:
        from torch.distributed.tensor import DTensor
        if isinstance(linear.weight, DTensor):
            return any(pl.__class__.__name__ == "Shard"
                       for pl in linear.weight.placements)
    except ImportError:
        pass
    return False


def apply_tp_fixes_v3(model, world_size: int, rank: int) -> None:
    new_heads = N_HEADS_FULL // world_size
    patched = 0
    for i in range(N_DOUBLE):
        attn = model.transformer_blocks[i].attn
        if _is_sharded(getattr(attn, "to_q", None)):
            attn.heads = new_heads; patched += 1
    for i in range(N_SINGLE):
        attn = model.single_transformer_blocks[i].attn
        if _is_sharded(getattr(attn, "to_q_s", None)):
            attn.heads = new_heads; patched += 1
    if rank == 0:
        print(f"[v3] patched attn.heads -> {new_heads} on {patched} blocks "
              f"(double + single)", flush=True)

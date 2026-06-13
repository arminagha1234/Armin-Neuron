#!/usr/bin/env python3
"""Smoke test for ShardedFlux2ParallelSelfAttention.

Constructs a tiny FLUX-style parallel self-attention block, runs the
unsharded forward, then runs the sharded forward across N=2 simulated
ranks on CPU and verifies the outputs match within numerical tolerance.

Tests run on CPU (no Neuron required) so this is the right thing to run
in CI before any Trainium-side debugging.

Usage
-----
    python src/tp_split_aware_smoke.py

The script prints per-rank shapes plus a max-abs diff between the
reference output and the all-gather'd sharded output. Pass criterion:
diff < 1e-3 in fp32.
"""
from __future__ import annotations

import os
import sys

import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)


def make_tiny_attn(d_in: int, heads: int, head_dim: int, mlp_ratio: float):
    """Build a tiny standalone module that has the same shape contract
    as Flux2ParallelSelfAttention.

    We don't import the diffusers class because it requires a full
    diffusers install plus the parallel self-attn processor. Instead we
    construct a stub with the exact attribute names the sharded version
    needs to copy weights from.
    """
    from diffusers.models.transformers.transformer_flux2 import Flux2SwiGLU

    class TinyAttn(nn.Module):
        def __init__(self):
            super().__init__()
            self.head_dim = head_dim
            self.inner_dim = heads * head_dim
            self.query_dim = d_in
            self.out_dim = d_in
            self.use_bias = False
            self.dropout = 0.0
            self.mlp_ratio = mlp_ratio
            self.mlp_hidden_dim = int(d_in * mlp_ratio)
            self.mlp_mult_factor = 2
            self.heads = heads

            self.to_qkv_mlp_proj = nn.Linear(
                d_in,
                3 * self.inner_dim + self.mlp_hidden_dim * self.mlp_mult_factor,
                bias=False,
            )
            self.mlp_act_fn = Flux2SwiGLU()
            self.norm_q = nn.RMSNorm(head_dim, eps=1e-6)
            self.norm_k = nn.RMSNorm(head_dim, eps=1e-6)
            self.to_out = nn.Linear(
                self.inner_dim + self.mlp_hidden_dim, d_in, bias=False,
            )
            self.processor = None
            self._attention_backend = None
            self._parallel_config = None

        def forward(self, x, image_rotary_emb=None):
            """Reference forward — same shape contract as the real Flux2 attn."""
            from diffusers.models.attention_dispatch import dispatch_attention_fn

            proj = self.to_qkv_mlp_proj(x)
            qkv, mlp = torch.split(
                proj,
                [3 * self.inner_dim, self.mlp_mult_factor * self.mlp_hidden_dim],
                dim=-1,
            )
            q, k, v = qkv.chunk(3, dim=-1)
            q = q.unflatten(-1, (self.heads, -1))
            k = k.unflatten(-1, (self.heads, -1))
            v = v.unflatten(-1, (self.heads, -1))
            q = self.norm_q(q)
            k = self.norm_k(k)
            attn = dispatch_attention_fn(q, k, v, attn_mask=None,
                                         backend=None, parallel_config=None)
            attn = attn.flatten(2, 3).to(q.dtype)
            mlp_h = self.mlp_act_fn(mlp)
            return self.to_out(torch.cat([attn, mlp_h], dim=-1))

    return TinyAttn()


def all_gather_local_outs(local_outs):
    """Sum local outputs across simulated ranks (this is the math an
    `all_reduce(SUM)` would produce in a real distributed run).
    """
    out = torch.zeros_like(local_outs[0])
    for lo in local_outs:
        out = out + lo
    return out


def main():
    torch.manual_seed(0)
    d_in = 512
    heads = 8
    head_dim = 64
    mlp_ratio = 3.0
    world = 2

    print(f"=== ShardedFlux2ParallelSelfAttention smoke test ===")
    print(f"d_in={d_in}, heads={heads}, head_dim={head_dim}, mlp_ratio={mlp_ratio}, world={world}")

    # Build the reference module
    ref = make_tiny_attn(d_in, heads, head_dim, mlp_ratio).eval()

    # Forward at fp32 with random input
    x = torch.randn(2, 16, d_in)
    with torch.no_grad():
        ref_out = ref(x)
    print(f"reference output shape: {tuple(ref_out.shape)}")

    # Build per-rank sharded versions
    from tp_split_aware import ShardedFlux2ParallelSelfAttention

    locals_ = []
    for rank in range(world):
        sh = ShardedFlux2ParallelSelfAttention(
            ref, rank=rank, world=world, process_group=None,
        ).eval()
        with torch.no_grad():
            local = sh(x)
        print(f"rank {rank} local output shape: {tuple(local.shape)}")
        locals_.append(local)

    summed = all_gather_local_outs(locals_)

    diff = (summed - ref_out).abs()
    max_d = diff.max().item()
    mean_d = diff.mean().item()
    print(f"\nmax abs diff:  {max_d:.3e}")
    print(f"mean abs diff: {mean_d:.3e}")
    print(f"ref output norm: {ref_out.norm().item():.3e}")
    print(f"summed norm:     {summed.norm().item():.3e}")

    tol = 1e-3
    if max_d < tol:
        print(f"\nPASS (diff < {tol})")
        sys.exit(0)
    else:
        print(f"\nFAIL (diff >= {tol})")
        sys.exit(1)


if __name__ == "__main__":
    main()

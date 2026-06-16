#!/usr/bin/env python3
"""CPU self-test: verify v3 fused-linear splits are weight-preserving.

Runs entirely on CPU, no Neuron, no compile. Catches the SwiGLU
chunk-scramble class of bug before any expensive device run.
"""
import torch
import torch.nn as nn

torch.manual_seed(0)
from diffusers.models.transformers.transformer_flux2 import (
    Flux2FeedForward, Flux2ParallelSelfAttention, Flux2ParallelSelfAttnProcessor,
)
import flux2_tp_plan_v3 as v3


def check(name, a, b, atol=1e-4):
    d = (a - b).abs().max().item()
    ok = d < atol
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: max|Δ|={d:.2e}")
    return ok


def test_ffn():
    print("FFN split equivalence:")
    dim = 3072
    ff = Flux2FeedForward(dim=dim, dim_out=dim, mult=3.0, bias=False).eval()
    x = torch.randn(1, 64, dim)
    with torch.no_grad():
        y_orig = ff(x)
    v3._restructure_ffn(ff)
    Flux2FeedForward.forward = v3._ffn_forward_v3
    with torch.no_grad():
        y_v3 = ff(x)
    return check("Flux2FeedForward", y_orig, y_v3)


def test_single():
    print("Single-stream split equivalence:")
    dim = 3072
    attn = Flux2ParallelSelfAttention(
        query_dim=dim, dim_head=128, heads=24, out_dim=dim, bias=False,
        out_bias=False, eps=1e-6, mlp_ratio=3.0, mlp_mult_factor=2,
        processor=Flux2ParallelSelfAttnProcessor(),
    ).eval()
    x = torch.randn(1, 64, dim)

    # original fused projection slices
    with torch.no_grad():
        fused = attn.to_qkv_mlp_proj(x)
        inner = attn.inner_dim
        mlp_total = attn.mlp_hidden_dim * attn.mlp_mult_factor
        mlp_half = mlp_total // 2
        qkv, mlp = torch.split(fused, [3 * inner, mlp_total], dim=-1)
        q0, k0, v0 = qkv.chunk(3, dim=-1)
        g0, m0 = mlp[..., :mlp_half], mlp[..., mlp_half:]
        # original to_out on concat [silu(g)*m mapped]... we test the proj
        attn_part = torch.randn(1, 64, inner)
        mlp_part = torch.randn(1, 64, mlp_half)
        out0 = attn.to_out(torch.cat([attn_part, mlp_part], dim=-1))

    v3._restructure_single(attn)
    with torch.no_grad():
        q1 = attn.to_q_s(x); k1 = attn.to_k_s(x); v1 = attn.to_v_s(x)
        g1 = attn.mlp_gate(x); m1 = attn.mlp_value(x)
        out1 = attn.attn_out_proj(attn_part) + attn.mlp_out_proj(mlp_part)

    ok = True
    ok &= check("to_q", q0, q1)
    ok &= check("to_k", k0, k1)
    ok &= check("to_v", v0, v1)
    ok &= check("mlp_gate", g0, g1)
    ok &= check("mlp_value", m0, m1)
    ok &= check("to_out (dual-proj sum)", out0, out1)
    return ok


if __name__ == "__main__":
    a = test_ffn()
    b = test_single()
    print(f"\n{'ALL PASS' if (a and b) else 'FAILURES PRESENT'}")

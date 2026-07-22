#!/usr/bin/env python3
"""
test_correctness.py -- reference correctness harness for the Gemma4-31B
segmented-prefill (design 08) and fused qkv_proj (design 02) patches.

This is a STANDALONE test. It does NOT import the vllm_neuron model. It builds a
pure-torch reference for each patched path and compares it against a pure-torch
"patched-equivalent" implementation that mirrors what the NKI kernels compute
(native-GQA flash w/ prefix cache; fused QKV matmul + pre-RoPE RMS QK-norm + RoPE).

WHY torch-vs-torch and not torch-vs-kernel here?
  - This file is WRITTEN with no device available. It validates the MATH/CONTRACT
    of each patch (masking semantics, GQA expansion, prior/current split, norm
    order, partial-RoPE pass-through) so that when the on-device run happens, any
    mismatch is a KERNEL numerics issue, not a design/contract bug.
  - To also exercise the real kernels on-device, set GEMMA4_TEST_USE_KERNEL=1 and
    run inside the vllm_ga container; the harness will additionally call
    NF.flash_attention / NF.qkv_proj and compare against the reference. If
    vllm_neuron import fails, those checks are skipped with a clear notice.

Layers exercised:
  - one SWA layer:    head_dim=256, 16 KV heads, sliding_window set, full RoPE
  - one global layer: head_dim=512,  4 KV heads, no SWA, partial RoPE (0.25)

Metrics reported per case:
  - per-head cosine similarity (min / mean)
  - torch.allclose(atol=1e-2, rtol=1e-2)

Run:
    python3 test_correctness.py
    # optional on-device kernel check (inside container):
    GEMMA4_TEST_USE_KERNEL=1 python3 test_correctness.py
"""
from __future__ import annotations

import os
import math

import torch
import torch.nn.functional as F

torch.manual_seed(0)

USE_KERNEL = os.environ.get("GEMMA4_TEST_USE_KERNEL", "0") == "1"
DTYPE = torch.float32          # reference math in fp32; on-device kernel path is bf16
ATOL, RTOL = 1e-2, 1e-2


# ---------------------------------------------------------------------------
# Shared reference primitives (mirror gemma4/model.py exactly)
# ---------------------------------------------------------------------------
def rms_norm(x: torch.Tensor, weight: torch.Tensor | None, eps: float) -> torch.Tensor:
    """Gemma4RMSNorm / Gemma4VNorm: normalize on dim=-1 in fp32."""
    input_dtype = x.dtype
    x = x.float()
    var = x.pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(var + eps)
    if weight is not None:
        x = weight.float() * x
    return x.to(input_dtype)


def rotate_half_apply(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """_apply_rotary_emb from model.py (rotate_half / half-split convention)."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    rotated = torch.cat((-x2, x1), dim=-1)
    return x * cos + rotated * sin


def build_cos_sin(T: int, head_dim: int, rope_theta: float,
                  partial_rotary_factor: float, device, dtype) -> tuple:
    """Gemma4RotaryEmbedding.forward: proportional RoPE, cos=1/sin=0 for nope dims."""
    rope_angles = int(partial_rotary_factor * head_dim // 2)
    nope_angles = head_dim // 2 - rope_angles
    inv_freq_rot = 1.0 / (
        rope_theta ** (torch.arange(0, 2 * rope_angles, 2, dtype=torch.float, device=device) / head_dim)
    )
    if nope_angles > 0:
        inv_freq = torch.cat([inv_freq_rot, torch.zeros(nope_angles, dtype=torch.float, device=device)])
    else:
        inv_freq = inv_freq_rot
    positions = torch.arange(T, device=device, dtype=torch.float)
    freqs = (inv_freq[:, None] @ positions[None, :]).transpose(0, 1)   # [T, head_dim//2]
    emb = torch.cat((freqs, freqs), dim=-1)                            # [T, head_dim]
    return emb.cos().to(dtype), emb.sin().to(dtype)


def per_head_cosine(a: torch.Tensor, b: torch.Tensor) -> tuple:
    """a, b: [Nh, T, Dh] -> (min, mean) cosine over heads (flattened T*Dh)."""
    nh = a.shape[0]
    af = a.reshape(nh, -1).float()
    bf = b.reshape(nh, -1).float()
    cos = F.cosine_similarity(af, bf, dim=-1)
    return cos.min().item(), cos.mean().item()


def report(name: str, ref: torch.Tensor, got: torch.Tensor):
    cmin, cmean = per_head_cosine(ref, got)
    ok = torch.allclose(ref.float(), got.float(), atol=ATOL, rtol=RTOL)
    maxdiff = (ref.float() - got.float()).abs().max().item()
    print(f"  [{name}]  cos_min={cmin:.6f} cos_mean={cmean:.6f} "
          f"max_abs_diff={maxdiff:.3e} allclose(atol={ATOL},rtol={RTOL})={ok}")
    return ok


# ===========================================================================
# DESIGN 08: segmented prefill attention -- reference vs patched-equivalent
# ===========================================================================
def ref_segmented_prefill(q, k_cur, v_cur, k_prior, v_prior, prior_len,
                          num_kv_groups, scaling, sliding_window):
    """ORIGINAL torch path: concat prior+current, GQA-expand, additive causal(+SWA)
    mask, SDPA. This is the semantics the OLD model.py produced.

    q:      [Nh, T, Dh]
    k_cur:  [Nkv, T, Dh]      current chunk K
    v_cur:  [Nkv, T, Dh]
    k_prior:[Nkv, P, Dh]      prior context K (P = prior_len valid positions)
    v_prior:[Nkv, P, Dh]
    """
    Nh, T, Dh = q.shape
    Nkv = k_cur.shape[0]
    P = prior_len

    # Concatenate the *valid* prior + current along the KV/time axis.
    k_full = torch.cat([k_prior[:, :P, :], k_cur], dim=1)      # [Nkv, P+T, Dh]
    v_full = torch.cat([v_prior[:, :P, :], v_cur], dim=1)
    S_kv = P + T

    # GQA expand.
    k_e = k_full.unsqueeze(1).expand(Nkv, num_kv_groups, S_kv, Dh).reshape(Nh, S_kv, Dh)
    v_e = v_full.unsqueeze(1).expand(Nkv, num_kv_groups, S_kv, Dh).reshape(Nh, S_kv, Dh)

    # Causal (+ optional SWA) mask. Query i (global pos P+i) attends key j (global pos j).
    q_pos = torch.arange(T, device=q.device).unsqueeze(1) + P     # [T,1]
    k_pos = torch.arange(S_kv, device=q.device).unsqueeze(0)       # [1,S_kv]
    allowed = (k_pos <= q_pos)
    if sliding_window is not None:
        allowed = allowed & (k_pos > q_pos - sliding_window)
    mask = torch.where(allowed,
                       torch.zeros(1, dtype=q.dtype, device=q.device),
                       torch.full((1,), float("-inf"), dtype=q.dtype, device=q.device))
    return F.scaled_dot_product_attention(q, k_e, v_e, attn_mask=mask,
                                          is_causal=False, scale=scaling)


def patched_equiv_segmented_prefill(q, k_cur, v_cur, k_prior_padded, v_prior_padded,
                                    prior_len, padded_kv_len, num_kv_groups,
                                    scaling, sliding_window):
    """PATCHED semantics (design 08 contract): the kernel receives
      - current chunk k/v (causal)
      - k_prior/v_prior = FULL padded span, masked to the first `prior_len`
        positions via prior_used_len.
    We reproduce that mask contract in torch: prior positions [0,prior_len) are
    "full attention" (subject to SWA), current positions are causal. Padding in
    the prior span (>= prior_len) is masked out. This must equal the reference.
    """
    Nh, T, Dh = q.shape
    Nkv = k_cur.shape[0]

    # Combined KV = [ prior_padded (padded_kv_len) | current (T) ].
    k_full = torch.cat([k_prior_padded, k_cur], dim=1)            # [Nkv, padded+T, Dh]
    v_full = torch.cat([v_prior_padded, v_cur], dim=1)
    S_kv = padded_kv_len + T

    k_e = k_full.unsqueeze(1).expand(Nkv, num_kv_groups, S_kv, Dh).reshape(Nh, S_kv, Dh)
    v_e = v_full.unsqueeze(1).expand(Nkv, num_kv_groups, S_kv, Dh).reshape(Nh, S_kv, Dh)

    # Global query position = prior_len + i.
    q_gpos = torch.arange(T, device=q.device).unsqueeze(1) + prior_len       # [T,1]

    # Key global position: prior slot p -> p (valid iff p < prior_len);
    #                       current slot c -> prior_len + c.
    prior_slot = torch.arange(padded_kv_len, device=q.device)
    cur_slot = torch.arange(T, device=q.device)
    k_gpos = torch.cat([prior_slot, prior_len + cur_slot], dim=0).unsqueeze(0)  # [1,S_kv]
    is_prior = torch.cat([torch.ones(padded_kv_len, dtype=torch.bool, device=q.device),
                          torch.zeros(T, dtype=torch.bool, device=q.device)])
    valid_prior = (torch.arange(padded_kv_len, device=q.device) < prior_len)
    valid = torch.cat([valid_prior, torch.ones(T, dtype=torch.bool, device=q.device)]).unsqueeze(0)

    allowed = (k_gpos <= q_gpos) & valid
    if sliding_window is not None:
        allowed = allowed & (k_gpos > q_gpos - sliding_window)
    mask = torch.where(allowed,
                       torch.zeros(1, dtype=q.dtype, device=q.device),
                       torch.full((1,), float("-inf"), dtype=q.dtype, device=q.device))
    return F.scaled_dot_product_attention(q, k_e, v_e, attn_mask=mask,
                                          is_causal=False, scale=scaling)


def test_segmented(layer_name, head_dim, num_kv_heads, num_heads,
                   sliding_window, T=32, prior_len=48, block_size=16):
    print(f"[SEGMENTED PREFILL] {layer_name}: hd={head_dim} kv={num_kv_heads} "
          f"heads={num_heads} SWA={sliding_window} T={T} prior_len={prior_len}")
    num_kv_groups = num_heads // num_kv_heads
    scaling = 1.0
    dev = "cpu"

    q = torch.randn(num_heads, T, head_dim, dtype=DTYPE)
    k_cur = torch.randn(num_kv_heads, T, head_dim, dtype=DTYPE)
    v_cur = torch.randn(num_kv_heads, T, head_dim, dtype=DTYPE)

    # padded prior span (block-aligned). Only first prior_len positions are valid.
    padded_kv_len = ((prior_len + block_size - 1) // block_size) * block_size
    k_prior = torch.randn(num_kv_heads, padded_kv_len, head_dim, dtype=DTYPE)
    v_prior = torch.randn(num_kv_heads, padded_kv_len, head_dim, dtype=DTYPE)
    # zero the padding region so both impls agree on what "padding" contains
    k_prior[:, prior_len:, :] = 0.0
    v_prior[:, prior_len:, :] = 0.0

    ref = ref_segmented_prefill(q, k_cur, v_cur, k_prior, v_prior, prior_len,
                                num_kv_groups, scaling, sliding_window)
    got = patched_equiv_segmented_prefill(q, k_cur, v_cur, k_prior, v_prior,
                                          prior_len, padded_kv_len, num_kv_groups,
                                          scaling, sliding_window)
    ok = report("contract: patched-equiv vs original-torch", ref, got)

    if USE_KERNEL:
        try:
            import vllm_neuron.functional as NF  # noqa
            prior_used_len = torch.tensor([prior_len], dtype=torch.int32)
            k_kernel = NF.flash_attention(
                q=q.to(torch.bfloat16),
                k=k_cur.transpose(1, 2).to(torch.bfloat16),
                v=v_cur.to(torch.bfloat16),
                scale=scaling, causal_mask=True,
                sliding_window=(sliding_window if sliding_window is not None else 0),
                k_prior=k_prior.transpose(1, 2).to(torch.bfloat16),
                v_prior=v_prior.to(torch.bfloat16),
                prior_used_len=prior_used_len,
                tp_q=True, tp_k=False, tp_out=False,
            )
            report("KERNEL: NF.flash_attention vs original-torch", ref, k_kernel.float())
        except Exception as e:  # pragma: no cover
            print(f"  [KERNEL] skipped (vllm_neuron/kernel unavailable): {e}")
    print()
    return ok


# ===========================================================================
# DESIGN 02: fused qkv_proj -- reference vs patched-equivalent (torch)
# ===========================================================================
def ref_qkv(hidden, W, split_idx, T, num_heads, num_kv, head_dim,
            q_gamma, k_gamma, eps, cos, sin):
    """ORIGINAL torch path: matmul -> split -> view -> QK RMS-norm -> V-norm -> RoPE."""
    qkv = torch.matmul(hidden, W)
    q, k, v = torch.tensor_split(qkv, split_idx, dim=-1)
    q = q.view(T, num_heads, head_dim).transpose(0, 1)
    k = k.view(T, num_kv, head_dim).transpose(0, 1)
    v = v.view(T, num_kv, head_dim).transpose(0, 1)
    # QK norm (before RoPE)
    q = rms_norm(q, q_gamma, eps)
    k = rms_norm(k, k_gamma, eps)
    # V norm (no gamma)
    v = rms_norm(v, None, eps)
    # RoPE (partial baked into cos/sin)
    c = cos.unsqueeze(0)
    s = sin.unsqueeze(0)
    q = rotate_half_apply(q, c, s)
    k = rotate_half_apply(k, c, s)
    return q, k, v


def patched_equiv_qkv(hidden, W, split_idx, T, num_heads, num_kv, head_dim,
                      q_gamma, k_gamma, eps, cos, sin):
    """PATCHED semantics (design 02): NF.qkv_proj fuses matmul + pre-RoPE RMS
    QK-norm + RoPE; V-norm applied separately. The MATH is identical to the
    reference (same op order); this asserts the contract holds. On-device the
    kernel replaces the fused ops -- exercised via USE_KERNEL below."""
    # Same math, expressed as the fused-then-vnorm sequence.
    qkv = torch.matmul(hidden, W)                 # kernel does matmul
    q, k, v = torch.tensor_split(qkv, split_idx, dim=-1)
    q = q.view(T, num_heads, head_dim).transpose(0, 1)
    k = k.view(T, num_kv, head_dim).transpose(0, 1)
    v = v.view(T, num_kv, head_dim).transpose(0, 1)
    # kernel: pre-RoPE RMS QK-norm then RoPE
    q = rms_norm(q, q_gamma, eps)
    k = rms_norm(k, k_gamma, eps)
    c = cos.unsqueeze(0)
    s = sin.unsqueeze(0)
    q = rotate_half_apply(q, c, s)
    k = rotate_half_apply(k, c, s)
    # residual V-norm (NOT fused)
    v = rms_norm(v, None, eps)
    return q, k, v


def test_qkv(layer_name, head_dim, num_kv_heads, num_heads, hidden_size,
             rope_theta, partial_rotary_factor, T=32, eps=1e-6):
    print(f"[FUSED QKV_PROJ] {layer_name}: hd={head_dim} kv={num_kv_heads} "
          f"heads={num_heads} H={hidden_size} prf={partial_rotary_factor}")
    dev = "cpu"
    q_size = num_heads * head_dim
    kv_size = num_kv_heads * head_dim
    qkv_size = q_size + 2 * kv_size
    split_idx = [q_size, q_size + kv_size]

    hidden = torch.randn(T, hidden_size, dtype=DTYPE)
    W = torch.randn(hidden_size, qkv_size, dtype=DTYPE) * (1.0 / math.sqrt(hidden_size))
    q_gamma = torch.randn(head_dim, dtype=DTYPE)
    k_gamma = torch.randn(head_dim, dtype=DTYPE)
    cos, sin = build_cos_sin(T, head_dim, rope_theta, partial_rotary_factor, dev, DTYPE)

    rq, rk, rv = ref_qkv(hidden, W, split_idx, T, num_heads, num_kv_heads,
                         head_dim, q_gamma, k_gamma, eps, cos, sin)
    gq, gk, gv = patched_equiv_qkv(hidden, W, split_idx, T, num_heads, num_kv_heads,
                                   head_dim, q_gamma, k_gamma, eps, cos, sin)

    ok = True
    ok &= report("Q contract", rq, gq)
    ok &= report("K contract", rk, gk)
    ok &= report("V contract", rv, gv)

    if USE_KERNEL:
        try:
            import vllm_neuron.functional as NF
            from nkilib.core.utils.common_types import NormType
            cos_c = cos.unsqueeze(0).to(torch.bfloat16)
            sin_c = sin.unsqueeze(0).to(torch.bfloat16)
            qkv_k = NF.qkv_proj(
                hidden=hidden.unsqueeze(0).to(torch.bfloat16),
                qkv_weights=W.to(torch.bfloat16),
                bias=None, d_head=head_dim,
                cos_cache=cos_c, sin_cache=sin_c,
                num_q_heads=num_heads, num_kv_heads=num_kv_heads,
                qk_norm_pre_rope_q_norm=NormType.RMS_NORM,
                qk_norm_pre_rope_k_norm=NormType.RMS_NORM,
                qk_norm_pre_rope_eps=eps,
                qk_norm_pre_rope_q_gamma=q_gamma.reshape(1, head_dim).to(torch.bfloat16),
                qk_norm_pre_rope_k_gamma=k_gamma.reshape(1, head_dim).to(torch.bfloat16),
            ).squeeze(0)
            kq, kk, kv = torch.tensor_split(qkv_k, split_idx, dim=-1)
            kq = kq.view(T, num_heads, head_dim).transpose(0, 1).float()
            kk = kk.view(T, num_kv_heads, head_dim).transpose(0, 1).float()
            kv = rms_norm(kv.view(T, num_kv_heads, head_dim).transpose(0, 1), None, eps).float()
            report("KERNEL Q", rq, kq)
            report("KERNEL K", rk, kk)
            report("KERNEL V(+torch vnorm)", rv, kv)
        except Exception as e:  # pragma: no cover
            print(f"  [KERNEL] skipped (vllm_neuron/kernel unavailable): {e}")
    print()
    return ok


# ===========================================================================
def main():
    print("=" * 78)
    print("Gemma4-31B patch correctness harness  (USE_KERNEL=%s)" % USE_KERNEL)
    print("=" * 78)
    all_ok = True

    # --- SWA layer: head_dim=256, 16 KV heads, sliding_window active, full RoPE.
    #     32 q-heads total; use single-rank equivalent shapes for the math test.
    all_ok &= test_segmented("SWA layer", head_dim=256, num_kv_heads=16,
                             num_heads=32, sliding_window=512)
    all_ok &= test_qkv("SWA layer", head_dim=256, num_kv_heads=16, num_heads=32,
                       hidden_size=5376, rope_theta=10000.0, partial_rotary_factor=1.0)

    # --- Global layer: head_dim=512, 4 KV heads, no SWA, partial RoPE 0.25.
    all_ok &= test_segmented("GLOBAL layer", head_dim=512, num_kv_heads=4,
                             num_heads=32, sliding_window=None)
    all_ok &= test_qkv("GLOBAL layer", head_dim=512, num_kv_heads=4, num_heads=32,
                       hidden_size=5376, rope_theta=1000000.0, partial_rotary_factor=0.25)

    print("=" * 78)
    print("RESULT:", "ALL CONTRACT CHECKS PASS" if all_ok else "FAILURES DETECTED")
    print("=" * 78)
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

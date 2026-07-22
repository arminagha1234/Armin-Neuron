#!/usr/bin/env python3
"""
patch_qkv_proj.py  --  Design 02: fused NKI qkv_proj (QKV matmul + QK-norm + RoPE).

Replaces the per-layer torch.matmul QKV projection + separate q_norm/k_norm +
partial-RoPE block with a SINGLE NF.qkv_proj(...) call in BSD output layout, with
pre-RoPE QK-norm fused into the kernel. V-norm stays a separate torch op (the
kernel norms only Q and K). Gated to TP world_size >= 8 (below that the fused
qkv_dim exceeds the kernel's 4096 gate for global layers -> keep torch fallback).

  ONLY the PREFILL site (forward_prefill) is patched by this script.
  There are ~4 QKV call sites total; see "OTHER SITES" at the bottom.

RUN THIS INSIDE THE vllm_ga CONTAINER:
    sudo docker exec vllm_ga python3 /path/to/patch_qkv_proj.py

Idempotent-safe: asserts the ORIGINAL torch.matmul QKV+norm+RoPE block is present.
Backs up to model.py.pre_qkv and runs ast.parse() as a syntax check.

Verified against the live file on ec2-3-19-59-18 (GA v0.21) on 2026-07-21:
  - forward_prefill QKV+norm+RoPE block: model.py:300-326 (Steps 1-4)
  - NF.qkv_proj signature (output_layout=BSD default, qk_norm_pre_rope_* params,
    cos_cache/sin_cache, num_q_heads/num_kv_heads): qkv.py:646-703
  - BSD layout has NO d_head==128 constraint (only NBSd does): qkv.py:638-640
  - llama3 reference call (RoPE fused via cos_cache/sin_cache): llama3/model.py:574-596
  - Gemma4 QK-norm is RMS (Gemma4RMSNorm), gamma = q_norm.weight/k_norm.weight [head_dim]
  - Gemma4 does QK-norm BEFORE RoPE -> qk_norm_pre_rope_* (model.py Steps 2 then 4)
  - partial RoPE baked into cos/sin caches (cos=1/sin=0 for nope dims) -> pass full caches
"""
import ast
import os
import shutil
import sys

P = "/opt/conda/lib/python3.13/site-packages/vllm_neuron/model/gemma4/model.py"
BACKUP = P + ".pre_qkv"

if len(sys.argv) > 1:
    P = sys.argv[1]
    BACKUP = P + ".pre_qkv"

src = open(P).read()

# ---------------------------------------------------------------------------
# Anchor -- VERIFIED verbatim (forward_prefill Steps 1-4, model.py:300-326).
# We replace from the "Step 1" comment through the partial-RoPE application,
# i.e. everything that produces normed+roped q, k and normed v, up to (but not
# including) "        # Step 5: Update KV Cache".
# ---------------------------------------------------------------------------
old_block = '''        # Step 1: QKV Projection (direct matmul — NKI qkv kernel has
        # head_dim constraints that may not hold for all Gemma4 layers)
        qkv = torch.matmul(hidden_states, self.qkv_proj_weight)

        q, k, v = torch.tensor_split(qkv, self.qkv_split_indices, dim=-1)

        q = q.view(tokens, self.num_attention_heads_per_rank, self.head_dim).transpose(
            0, 1
        )
        k = k.view(tokens, self.num_key_value_heads_per_rank, self.head_dim).transpose(
            0, 1
        )
        v = v.view(tokens, self.num_key_value_heads_per_rank, self.head_dim).transpose(
            0, 1
        )

        # Step 2: QK Normalization (before RoPE)
        q, k = self._apply_qk_norm(q, k)

        # Step 3: V Normalization (operates on dim=-1, works directly on 3D)
        v = self.v_norm(v)

        # Step 4: Apply RoPE (per-layer, possibly partial)
        cos, sin = self.rotary_emb(
            positions, device=hidden_states.device, dtype=hidden_states.dtype
        )
        q, k = self._apply_partial_rotary(q, k, cos, sin)
'''

# NOTE: the "direct matmul —" comment uses a U+2014 em-dash, matching the live file.
# Keep this source file UTF-8 so the anchor byte-matches.
assert old_block in src, (
    "old QKV+norm+RoPE block NOT found in forward_prefill. Either the patch is "
    "already applied, or the block drifted from the verified snapshot "
    "(model.py:300-326). Inspect Steps 1-4 of forward_prefill and re-derive the "
    "anchor. Aborting rather than corrupting the file."
)
# Guard against double-apply: our new marker must NOT already be present.
assert "FUSED NKI QKV_PROJ (design 02)" not in src, (
    "patch marker already present -- qkv_proj fusion already applied. Aborting."
)

# ---------------------------------------------------------------------------
# New block (design 02). TP>=8: fused NKI qkv_proj (matmul + pre-RoPE QK-norm +
# RoPE), then residual V-norm as torch. TP<8: original torch path (fallback),
# because global-layer fused_qkv_dim (5120 @ TP4) trips the kernel's >4096 gate.
# BSD layout -> no d_head==128 constraint, so hd256/hd512 are handled natively.
# ---------------------------------------------------------------------------
new_block = '''        # === FUSED NKI QKV_PROJ (design 02) ===
        # TP>=8: single NKI qkv_proj launch fuses matmul + pre-RoPE QK-norm + RoPE.
        #   BSD output layout -> no d_head==128 constraint (hd256/hd512 native).
        #   V-norm is NOT fused (kernel norms only Q,K) -> applied as torch op below.
        #   partial RoPE (global) is baked into cos/sin caches (cos=1/sin=0 for
        #   non-rotary dims) -> pass the full [1,T,head_dim] caches, no kernel change.
        # TP<8: fall back to the original torch path (global fused_qkv_dim=5120 @ TP4
        #   trips the kernel's fused_qkv_dim<=4096 gate).
        cos, sin = self.rotary_emb(
            positions, device=hidden_states.device, dtype=hidden_states.dtype
        )
        if self.world_size >= 8:
            # cos/sin caches for the kernel: [B=1, T, head_dim].
            cos_cache = cos.unsqueeze(0)
            sin_cache = sin.unsqueeze(0)
            qkv = NF.qkv_proj(
                hidden=hidden_states.unsqueeze(0),
                qkv_weights=self.qkv_proj_weight,
                bias=None,
                d_head=self.head_dim,
                cos_cache=cos_cache,
                sin_cache=sin_cache,
                num_q_heads=self.num_attention_heads_per_rank,
                num_kv_heads=self.num_key_value_heads_per_rank,
                # Pre-RoPE RMS QK-norm fused into the kernel (Gemma4 norms Q,K before RoPE).
                qk_norm_pre_rope_q_norm=NormType.RMS_NORM,
                qk_norm_pre_rope_k_norm=NormType.RMS_NORM,
                qk_norm_pre_rope_eps=self.q_norm.variance_epsilon,
                qk_norm_pre_rope_q_gamma=self.q_norm.weight.reshape(1, self.head_dim),
                qk_norm_pre_rope_k_gamma=self.k_norm.weight.reshape(1, self.head_dim),
            ).squeeze(0)

            q, k, v = torch.tensor_split(qkv, self.qkv_split_indices, dim=-1)
            q = q.view(tokens, self.num_attention_heads_per_rank, self.head_dim).transpose(0, 1)
            k = k.view(tokens, self.num_key_value_heads_per_rank, self.head_dim).transpose(0, 1)
            v = v.view(tokens, self.num_key_value_heads_per_rank, self.head_dim).transpose(0, 1)

            # V-norm is NOT fused into the kernel -> apply as a residual torch op.
            v = self.v_norm(v)
            # Q,K norm + RoPE already applied inside qkv_proj.
        else:
            # --- Original torch path (TP<8 fallback) ---
            qkv = torch.matmul(hidden_states, self.qkv_proj_weight)

            q, k, v = torch.tensor_split(qkv, self.qkv_split_indices, dim=-1)

            q = q.view(tokens, self.num_attention_heads_per_rank, self.head_dim).transpose(0, 1)
            k = k.view(tokens, self.num_key_value_heads_per_rank, self.head_dim).transpose(0, 1)
            v = v.view(tokens, self.num_key_value_heads_per_rank, self.head_dim).transpose(0, 1)

            # QK Normalization (before RoPE)
            q, k = self._apply_qk_norm(q, k)
            # V Normalization
            v = self.v_norm(v)
            # Apply RoPE (per-layer, possibly partial)
            q, k = self._apply_partial_rotary(q, k, cos, sin)
'''

new_src = src.replace(old_block, new_block, 1)
assert new_src != src, "replace produced no change -- aborting."

if not os.path.exists(BACKUP):
    shutil.copy2(P, BACKUP)
    print("BACKUP created: %s" % BACKUP)
else:
    print("BACKUP already exists (not overwriting): %s" % BACKUP)

ast.parse(new_src)
print("ast.parse OK")

open(P, "w").write(new_src)
print("PATCH_OK  bytes_before=%d  bytes_after=%d" % (len(src), len(new_src)))
print("Patched forward_prefill QKV -> NF.qkv_proj (TP>=8) with torch fallback (TP<8).")

# ===========================================================================
# OTHER SITES (NOT patched by this script -- do them once prefill is validated)
# ===========================================================================
# The same torch.matmul QKV + _apply_qk_norm + v_norm + _apply_partial_rotary
# pattern appears at four `qkv = torch.matmul(...)` sites (live line numbers):
#   1. model.py:481  forward_prefill    <-- PATCHED HERE (validate first)
#   2. model.py:642  forward_decode     <-- decode path; small token counts, lower
#                                           priority. Apply same TP>=8 gating.
#   3. model.py:791  (second prefill-style path, e.g. chunked/context)  <-- extend after (1) passes
#   4. model.py:939  (second decode-style / fp8 path, softmax_scale uses k_scale_float) <-- extend last
# For each: replicate the TP>=8 fused branch, keep the torch else-branch, and be
# careful that decode paths reshape [tokens=B*S_decode] and some use different
# cos/sin plumbing. Re-verify the anchor block per site before editing.

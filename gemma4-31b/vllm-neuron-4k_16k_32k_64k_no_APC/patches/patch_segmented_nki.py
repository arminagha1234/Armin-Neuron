#!/usr/bin/env python3
"""
patch_segmented_nki.py  --  Design 08: route segmented prefill through NKI flash.

Replaces the torch-SDPA body of Gemma4Attention._segmented_prefill_attention with
an NF.flash_attention call that feeds prior cached context via the kernel's NATIVE
prefix-cache args (k_prior / v_prior / prior_used_len) instead of a manual
gather+GQA-expand+additive-mask+SDPA. No kernel changes; reuses the attention_cte
CTE kernel already proven at hd512 (model.py global-layer single-shot path).

RUN THIS INSIDE THE vllm_ga CONTAINER:
    sudo docker exec vllm_ga python3 /path/to/patch_segmented_nki.py

Idempotent-safe: asserts the ORIGINAL torch-SDPA block is still present before
editing (re-running after a successful patch fails loudly with a clear message).
Makes a backup at model.py.pre_segnki and runs ast.parse() as a syntax check.

Verified against the live file on ec2-3-19-59-18 (GA v0.21) on 2026-07-21:
  - _segmented_prefill_attention body: model.py:195-252 (STATIC-SHAPE rewrite already applied)
  - NF.flash_attention prefix args k_prior/v_prior/prior_used_len: attention_cte.py:201-203
  - global-layer flash call layout (q=q, k=k.transpose(1,2), v=v, tp_q/tp_k/tp_out): model.py:383-388
"""
import ast
import os
import shutil
import sys

P = "/opt/conda/lib/python3.13/site-packages/vllm_neuron/model/gemma4/model.py"
BACKUP = P + ".pre_segnki"

if len(sys.argv) > 1:  # allow overriding the target path for dry-runs / testing
    P = sys.argv[1]
    BACKUP = P + ".pre_segnki"

src = open(P).read()

# ---------------------------------------------------------------------------
# Anchors -- VERIFIED verbatim against the live file (model.py:195 / 246-252).
# start_marker: first line of the STATIC-SHAPE body (right after the docstring).
# end_marker:   the method's terminal `return attn_output` immediately preceding
#               the `# -- Forward dispatch --` section comment. This disambiguates
#               it from the other `return attn_output` occurrences in the file.
# ---------------------------------------------------------------------------
start_marker = (
    "        # STATIC-SHAPE segmented prefill (trace-safe: no data-dependent shapes)."
)
end_marker = "        return attn_output\n\n    # -- Forward dispatch --"

# Sanity anchor: this SDPA call must be present in the block we are replacing.
# If it is missing, either the file already got patched or it drifted from the
# verified snapshot -- fail rather than silently corrupt the file.
sdpa_anchor = "        attn_output = F.scaled_dot_product_attention(\n            q, k_full, v_full,"

si = src.find(start_marker)
ei = src.find(end_marker)

assert si != -1, (
    "start marker NOT found -- the original STATIC-SHAPE segmented block is "
    "absent. Either patch already applied (check for NF.flash_attention with "
    "k_prior= inside _segmented_prefill_attention) or the file drifted. Aborting."
)
assert ei != -1, "end marker NOT found ('        return attn_output\\n\\n    # -- Forward dispatch --')."
assert si < ei, "markers out of order -- file layout unexpected, aborting."
assert sdpa_anchor in src[si:ei], (
    "SDPA anchor NOT found inside the block -- refusing to patch a body that no "
    "longer matches the verified torch-SDPA snapshot."
)

# ---------------------------------------------------------------------------
# New body (design 08). Keeps the prior gather + FP8 dequant, but repurposes it
# as the k_prior/v_prior prefix tensors and hands prior/current to the kernel:
#   * k_prior/v_prior = full padded block-table span (prior + current chunk).
#   * prior_used_len   = cached_seq_len -> kernel masks the prior tensor to EXACTLY
#                        the prior tokens (positions [0, cached_seq_len)); the
#                        current chunk (positions >= cached_seq_len) is masked OUT
#                        of the prior tensor, so it is NOT double-counted.
#   * current chunk k/v = the method args (causal via causal_mask=True).
#   * native GQA        = pass nkh-batched K/V + Nh-batched Q (no manual expand).
#   * SWA               = sliding_window flag (0 for global layers).
# NF.flash_attention has an automatic PyTorch fallback if kernel constraints are
# violated, so correctness is preserved even if the CTE kernel is unavailable.
# ---------------------------------------------------------------------------
new_body = '''        # === NKI FLASH-ATTENTION SEGMENTED PREFILL (design 08) ===
        # Route segmented prefill through the attention_cte NKI flash kernel,
        # feeding prior cached context via the kernel's NATIVE prefix-cache args
        # (k_prior / v_prior / prior_used_len) instead of pre-concatenating.
        # Kernel semantics: prior = full attention dynamically masked by
        # prior_used_len; current chunk = causal; SWA applied to both.
        nkh = self.num_key_value_heads_per_rank
        max_blocks = block_table.shape[1]
        padded_kv_len = max_blocks * block_size            # static at trace time

        # Gather the FULL block-table span (static shape) as the PRIOR context.
        # The current chunk's K/V was already written to the cache, so it occupies
        # positions [cached_seq_len, cached_seq_len + tokens). prior_used_len below
        # is set to cached_seq_len, so the kernel masks the prior tensor to exactly
        # the prior tokens (positions [0, cached_seq_len)); the current chunk is
        # supplied separately via k/v (causal), avoiding double-counting.
        bt = block_table[0].clamp_min(0).to(torch.int64)               # [max_blocks]
        k_blocks = torch.index_select(self.k_cache, 0, bt)             # [max_blocks, nkh, block_size, Dh]
        v_blocks = torch.index_select(self.v_cache, 0, bt)
        k_prior = k_blocks.permute(1, 0, 2, 3).reshape(nkh, padded_kv_len, self.head_dim)
        v_prior = v_blocks.permute(1, 0, 2, 3).reshape(nkh, padded_kv_len, self.head_dim)

        # Dequantize FP8 cache values back to compute dtype.
        if self.k_cache.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
            k_prior = k_prior.to(self.dtype) / self.k_scale_float
            v_prior = v_prior.to(self.dtype) / self.v_scale_float

        # prior_used_len masks k_prior/v_prior to the first cached_seq_len positions.
        prior_used_len = cached_seq_len.reshape(1).to(torch.int32)

        # NKI flash attention with native GQA + prefix cache. No manual GQA expand,
        # no additive mask, no O(T*S_kv) fp32 score materialization.
        #   current chunk k/v (method args) -> causal; k_prior/v_prior -> full (masked)
        # Kernel layout: q=[Nh,T,D] (tp_q), k=[Nkv,D,T] (tp_k=False), v=[Nkv,T,D].
        attn_output = NF.flash_attention(
            q=q,
            k=k.transpose(1, 2),
            v=v,
            scale=self.scaling,
            causal_mask=True,
            sliding_window=(self.sliding_window if self.sliding_window is not None else 0),
            k_prior=k_prior.transpose(1, 2),
            v_prior=v_prior,
            prior_used_len=prior_used_len,
            tp_q=True, tp_k=False, tp_out=False,
        )
        return attn_output'''

# Splice: [start .. start_of_return] replaced, keeping the trailing
# "\n\n    # -- Forward dispatch --" that end_marker begins with.
tail_keep = "\n\n    # -- Forward dispatch --"
new_src = src[:si] + new_body + tail_keep + src[ei + len(end_marker):]

# Backup then write.
if not os.path.exists(BACKUP):
    shutil.copy2(P, BACKUP)
    print("BACKUP created: %s" % BACKUP)
else:
    print("BACKUP already exists (not overwriting): %s" % BACKUP)

# Syntax check BEFORE writing.
ast.parse(new_src)
print("ast.parse OK")

open(P, "w").write(new_src)
print("PATCH_OK  bytes_before=%d  bytes_after=%d" % (len(src), len(new_src)))
print("Patched _segmented_prefill_attention -> NF.flash_attention (k_prior/v_prior/prior_used_len).")

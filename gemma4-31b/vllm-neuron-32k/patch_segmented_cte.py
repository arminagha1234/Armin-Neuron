#!/usr/bin/env python3
"""explore_v2 edit A: route head_dim>128 in segmented_attention() to the
trace-safe torch fallback instead of raising. Enables chunked prefill for
gemma4 (head_dim 256/512). Mirrors ../vllm_32k_working README edit A.
Idempotent + backs up the original."""
import os, shutil

P = "/opt/conda/lib/python3.12/site-packages/vllm_neuron/functional/attention/attention_segmented_cte.py"
BAK = P + ".bak_armin"

src = open(P).read()
if "explore_v2 edit A" in src:
    print("already patched")
    raise SystemExit(0)

old = '''    if d_head > _MAX_HEAD_DIM:
        raise ValueError(
            f"head_dim={d_head} exceeds maximum supported head dimension "
            f"({_MAX_HEAD_DIM}). The segmented attention kernel requires "
            f"head_dim <= {_MAX_HEAD_DIM}."
        )'''

new = '''    if d_head > _MAX_HEAD_DIM:
        # explore_v2 edit A: gemma4 head_dim 256/512 > 128 -> route to the
        # trace-safe static-shape torch fallback instead of raising. This is
        # what enables chunked/segmented prefill above 16K for gemma4.
        if scale is None:
            scale = 1.0 / (d_head**0.5)
        return _torch_segmented_attention_impl(
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            block_tables=block_tables,
            prior_tokens=prior_tokens,
            block_size=block_size,
            kv_segment_size=kv_segment_size,
            scale=scale,
            tp_q=tp_q,
            tp_out=tp_out,
            sliding_window=sliding_window,
            sink=sink,
        )'''

if old not in src:
    raise SystemExit("ANCHOR NOT FOUND — aborting (container file differs)")

if not os.path.exists(BAK):
    shutil.copy(P, BAK)
open(P, "w").write(src.replace(old, new, 1))
print("patched:", P)
print("backup :", BAK)

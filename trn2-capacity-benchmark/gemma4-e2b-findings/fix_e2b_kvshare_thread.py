#!/usr/bin/env python3
"""THREADING variant: source the shared-layer prefill K/V from the donor's
freshly-computed (post-norm/post-RoPE) tensors, threaded through the already-
threaded `attn_metadata` dict, instead of gathering from the aliased paged cache.

Runs AFTER fix_e2b_kvshare.py + fix_e2b_kvshare_attn.py. Rationale (from research):
- Ordering of an in-place index_put_ write followed by a read on Neuron is
  order-safe via functionalization, so the cache-alias prefill *should* be
  correct -- this variant isolates the prefill KV-sourcing by removing the cache
  read from the prefill path entirely (reference-faithful to HF's shared_kv_states).
- Threading a static-keyed Python dict through the unrolled forward is a proven
  trace-safe pattern on this stack (attn_metadata, aux_hidden_states).
- DECODE is left on the cache-alias path (history is committed; order-safe).

Mechanism (no forward-signature changes): the donor layer (is_kv_donor) stashes
its post-RoPE (k, v) into attn_metadata["_shared_kv"][layer_type] in the NORMAL
forward_prefill; the shared layer reads it in _shared_prefill_attn. Donor index <
shared index, so the stash is present when the shared layer runs (unrolled loop).

Dry-run: run the base+attn selftests first, then this with --selftest.
"""
from __future__ import annotations
import os, sys, ast
FAILS = []
def edit(text, old, new, label, count=1):
    n = text.count(old)
    if n != count:
        FAILS.append(f"{label}: found {n} expected {count}"); return text
    return text.replace(old, new, count)

def patch_model(path):
    s = open(path).read(); orig = s
    # 1. attention __init__: record layer_type + donor flag (config helpers exist)
    s = edit(
        s,
        "        qkv_size = q_size if self.is_kv_shared_layer else q_size + 2 * kv_size",
        "        self.layer_type = config.layer_types[layer_idx]\n"
        "        self.is_kv_donor = config.is_kv_donor_layer(layer_idx)\n"
        "        qkv_size = q_size if self.is_kv_shared_layer else q_size + 2 * kv_size",
        "init: layer_type + is_kv_donor",
    )
    # 2. donor stashes post-RoPE (k, v) in the NORMAL forward_prefill (unique comment)
    s = edit(
        s,
        "        # Step 5: Update KV Cache\n",
        "        # KV-share threading: a donor stashes its post-norm/post-RoPE K/V\n"
        "        # so shared layers of the same type read it in-memory (no cache dep).\n"
        "        if getattr(self, \"is_kv_donor\", False):\n"
        "            _sk = attn_metadata.setdefault(\"_shared_kv\", {})\n"
        "            _sk[self.layer_type] = (k, v)\n"
        "        # Step 5: Update KV Cache\n",
        "prefill: donor stash into attn_metadata",
    )
    # 3. shared prefill reads the threaded donor K/V instead of the cache gather
    s = edit(
        s,
        "        k, v = self._shared_kv_from_cache_prefill(attn_metadata)",
        "        # THREADED: read the donor's freshly-computed post-RoPE K/V\n"
        "        # (falls back to the cache gather if the donor stash is absent).\n"
        "        _sk = attn_metadata.get(\"_shared_kv\", {})\n"
        "        if self.layer_type in _sk:\n"
        "            k, v = _sk[self.layer_type]\n"
        "        else:\n"
        "            k, v = self._shared_kv_from_cache_prefill(attn_metadata)",
        "shared prefill: read threaded donor K/V",
    )
    if s != orig and not FAILS:
        open(path, "w").write(s)
    ast.parse(open(path).read())

def main():
    if "--selftest" in sys.argv:
        mdl = "/tmp/e2bfix/model.py"
        assert os.path.exists(mdl), "run fix_e2b_kvshare.py + fix_e2b_kvshare_attn.py --selftest FIRST"
    else:
        model_dir = None
        for base in sys.path:
            c = os.path.join(base, "vllm_neuron", "model")
            if os.path.isdir(c): model_dir = c; break
        assert model_dir, "vllm_neuron/model not found"
        mdl = os.path.join(model_dir, "gemma4", "model.py")
    print("[thread] model:", mdl)
    patch_model(mdl)
    if FAILS:
        print("[thread] FAILED:"); [print("  -", f) for f in FAILS]; return 1
    print("[thread] SUCCESS - prefill K/V now threaded from donor")
    return 0

if __name__ == "__main__":
    sys.exit(main())

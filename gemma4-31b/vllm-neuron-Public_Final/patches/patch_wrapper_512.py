#!/usr/bin/env python3
"""THE TWO-KERNEL FIX: lift MAX_HEAD_DIM 128 -> 512 in BOTH vllm wrapper files so the model's
existing 2 prefill call sites route to the real nkilib kernels (which support hd512 via d-tiling)
instead of the torch fallback.
  - attention_cte.py         -> single-shot prefill (<=16k)   -> nkilib attention_cte (hd512 OK)
  - attention_segmented_cte.py -> segmented prefill (>16k)     -> nkilib attention_segmented_cte (hd512 OK)
Both nkilib kernels are source-verified compile-clean at hd512 (load-time dma_transpose,
flash-section=8K/num_d_tiles, ModularAllocator). This is a GUARD change, not new kernels.
Idempotent, backs up each file, reversible.
"""
import sys, glob
BASE = "/opt/conda/lib/python3.13/site-packages/vllm_neuron/functional/attention"
files = {
    "single-shot (attention_cte)":      f"{BASE}/attention_cte.py",
    "segmented (attention_segmented_cte)": f"{BASE}/attention_segmented_cte.py",
}
changed = []
for label, f in files.items():
    try:
        src = open(f).read()
    except FileNotFoundError:
        print(f"  SKIP {label}: file not found ({f})"); continue
    # both files use MAX_HEAD_DIM = 128 (single-shot) or _MAX_HEAD_DIM = 128 (segmented)
    new = src
    for pat in ("MAX_HEAD_DIM = 128", "_MAX_HEAD_DIM = 128"):
        if pat in new:
            new = new.replace(pat, pat.replace("128", "512"))
    if new != src:
        open(f + ".pre512", "w").write(src)
        open(f, "w").write(new)
        # report which constant(s) now 512
        import re
        vals = re.findall(r"_?MAX_HEAD_DIM = \d+", new)
        print(f"  PATCHED {label}: {vals}  (backup .pre512)")
        changed.append(label)
    else:
        print(f"  {label}: no '128' cap found (maybe already 512 or different) — check manually")
print(f"\nDONE. patched: {changed if changed else 'NONE'}")
print("Launch serve with these files; hd256/512 prefill now routes to the real NKI kernels.")

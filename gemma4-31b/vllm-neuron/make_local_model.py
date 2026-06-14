#!/usr/bin/env python3
"""Build a local gemma-4-31b-it dir (symlinked blobs + patched tokenizer config).

Serving from a local dir avoids vLLM's hub revision re-validation, which was
reverting our in-place tokenizer_config.json fix. The gemma-4 checkpoint ships
extra_special_tokens as a LIST; transformers 4.57 expects a dict. For a
text-only TTFT bench we drop the vision/audio extra tokens (set to {}).
"""
import json
import os

SNAP = "/root/.cache/huggingface/hub/models--google--gemma-4-31b-it/snapshots/fb9ae262347c3945692f09a612f8bb189def854f"
DST = "/root/models/gemma-4-31b-it"

os.makedirs(DST, exist_ok=True)
for name in os.listdir(SNAP):
    src = os.path.join(SNAP, name)
    real = os.path.realpath(src)
    d = os.path.join(DST, name)
    if os.path.lexists(d):
        os.remove(d)
    if name == "tokenizer_config.json":
        c = json.load(open(real))
        est = c.get("extra_special_tokens")
        if isinstance(est, list):
            c["extra_special_tokens"] = {}
        json.dump(c, open(d, "w"), indent=2)
        print("patched tokenizer_config.json (extra_special_tokens -> {})")
    else:
        os.symlink(real, d)
print("local model dir ready:", DST, "files:", len(os.listdir(DST)))

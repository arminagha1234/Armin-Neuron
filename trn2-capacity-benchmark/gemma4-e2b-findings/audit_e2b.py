#!/usr/bin/env python3
"""DIAGNOSTIC ONLY: prove which E2B parameters never get loaded.

Applies exactly one change to the installed gemma4 model: a loaded-key audit
immediately before `load_state_dict(..., strict=False)`. No behavioural change.

Hypothesis under test
---------------------
E2B sets num_kv_shared_layers=20 and use_double_wide_mlp=True (both verified in
its config.json). Per the HF reference, layers 15..34 are KV-shared and carry NO
k_proj/v_proj/k_norm/v_norm in the checkpoint, and those same 20 layers use a
double-width MLP (intermediate 12288, not 6144). The port maps k_proj/v_proj for
every layer and sizes every MLP at 6144, so those lookups miss; `strict=False`
plus `torch.empty` then leaves the parameters UNINITIALISED.

Prediction if the hypothesis is right
-------------------------------------
  * ~20 x qkv_proj_weight missing  (layers 15..34)
  * ~20 x k_norm.weight    missing (layers 15..34)
  * ~60 x mlp.{gate,up,down}_proj_weight missing (shape mismatch on 15..34)
  * layers 0..14 fully loaded
Anything else falsifies it.
"""
from __future__ import annotations

import ast
import os
import sys

AUDIT = '''        # ---- DIAGNOSTIC AUDIT (no behavioural change) --------------------
        _expected = set(self.state_dict().keys())
        _loaded = set(rank_sharded.keys())
        _missing = sorted(_expected - _loaded)
        print(f"WEIGHT_AUDIT total_expected={len(_expected)} "
              f"loaded={len(_loaded)} missing={len(_missing)}", flush=True)
        if _missing:
            import re as _re
            from collections import Counter as _C
            _byparam = _C()
            _bylayer = set()
            for _k in _missing:
                _byparam[_re.sub(r"layers\\.\\d+\\.", "layers.N.", _k)] += 1
                _m = _re.search(r"layers\\.(\\d+)\\.", _k)
                if _m:
                    _bylayer.add(int(_m.group(1)))
            print("WEIGHT_AUDIT_BY_PARAM:", flush=True)
            for _k, _n in sorted(_byparam.items(), key=lambda x: -x[1]):
                print(f"    {_n:>4}x  {_k}", flush=True)
            _ls = sorted(_bylayer)
            print(f"WEIGHT_AUDIT_LAYERS_AFFECTED n={len(_ls)} {_ls}", flush=True)
            print("WEIGHT_AUDIT_SAMPLE:", flush=True)
            for _k in _missing[:20]:
                _shp = tuple(self.state_dict()[_k].shape)
                print(f"    {_k}  shape={_shp}", flush=True)
        # ------------------------------------------------------------------
'''


def main() -> int:
    model_dir = None
    for base in sys.path:
        c = os.path.join(base, "vllm_neuron", "model")
        if os.path.isdir(c):
            model_dir = c
            break
    assert model_dir, "vllm_neuron/model not found"
    p = os.path.join(model_dir, "gemma4", "model.py")
    s = open(p).read()
    anchor = "        self.load_state_dict(rank_sharded, strict=False, assign=True)"
    if "WEIGHT_AUDIT total_expected" in s:
        print("[audit] already installed")
        return 0
    if anchor not in s:
        print("[audit] *** ANCHOR NOT FOUND ***")
        return 1
    if s.count(anchor) != 1:
        print(f"[audit] *** expected 1 anchor, found {s.count(anchor)} ***")
        return 1
    s = s.replace(anchor, AUDIT + anchor, 1)
    open(p, "w").write(s)
    ast.parse(open(p).read())
    print(f"[audit] installed into {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

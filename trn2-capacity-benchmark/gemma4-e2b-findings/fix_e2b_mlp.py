#!/usr/bin/env python3
"""Fix Gemma-4 E2B: implement use_double_wide_mlp.

EVIDENCE (read directly from google/gemma-4-E2B-it model.safetensors header via
an HTTP range request, so this is the checkpoint's own ground truth):

    gate_proj shapes: {(6144, 1536): 15,  (12288, 1536): 20}
    layer  5 (normal) mlp.gate_proj.weight  [ 6144, 1536]
    layer 20 (shared) mlp.gate_proj.weight  [12288, 1536]
    layers WITHOUT k_proj: NONE       <- KV-sharing theory FALSIFIED
    all 35 layers have identical 17-tensor signatures

E2B sets use_double_wide_mlp=True and num_kv_shared_layers=20, so per HF
(modeling_gemma4_unified.py:469-480) the last 20 of 35 layers use
intermediate_size*2 = 12288. The port sizes EVERY layer at 6144.

Why this passed the weight audit with missing=0: the MLP weight loader is
`sharding_weight_loader(shard_dim=..., shard_size=intermediate_size_per_rank)`.
Given a [12288, 1536] source and shard_size=6144 it does not raise -- it just
takes a 6144-wide slice. So HALF of every gate/up/down projection was silently
discarded on 20 of 35 layers (57% of the network). Silent truncation, not a
missing key, which is why `missing=0` was true and still misleading.

This patch does ONE thing: make the MLP width per-layer. It deliberately does
NOT touch KV sharing (falsified) or attention scaling (untested), so the result
is attributable to a single change.
"""
from __future__ import annotations

import ast
import os
import shutil
import sys

FAILS: list[str] = []


def edit(text: str, old: str, new: str, label: str, count: int = 1) -> str:
    if old not in text:
        FAILS.append(f"{label}: PATTERN NOT FOUND")
        return text
    if text.count(old) != count:
        FAILS.append(f"{label}: expected {count}, found {text.count(old)}")
        return text
    print(f"  [ok] {label}")
    return text.replace(old, new, count)


def patch_config(path: str) -> None:
    s = open(path).read()
    orig = s
    s = edit(
        s,
        "    num_kv_shared_layers: int | None = 0",
        "    num_kv_shared_layers: int | None = 0\n"
        "    # E2B sets this True: the last num_kv_shared_layers layers use\n"
        "    # intermediate_size*2 (HF modeling_gemma4_unified.py:469-480).\n"
        "    # Verified in the checkpoint: 15 layers at 6144, 20 at 12288.\n"
        "    use_double_wide_mlp: bool = False",
        "config: add use_double_wide_mlp",
    )
    s = edit(
        s,
        "\n    def is_global_layer(self, layer_idx: int) -> bool:",
        '''
    def get_layer_intermediate_size(self, layer_idx: int) -> int:
        """Per-layer MLP width.

        Double-wide applies to the KV-shared tail: layer_idx >=
        num_hidden_layers - num_kv_shared_layers. For E2B that is layers
        15..34 -> 12288, layers 0..14 -> 6144.
        """
        shared = self.num_kv_shared_layers or 0
        if self.use_double_wide_mlp and shared > 0:
            if layer_idx >= self.num_hidden_layers - shared:
                return self.intermediate_size * 2
        return self.intermediate_size

    def is_global_layer(self, layer_idx: int) -> bool:''',
        "config: add get_layer_intermediate_size",
    )
    if s != orig:
        open(path, "w").write(s)
    ast.parse(open(path).read())


def patch_model(path: str) -> None:
    s = open(path).read()
    orig = s
    s = edit(
        s,
        "    def __init__(self, config: Gemma4Config):\n"
        "        super().__init__()\n\n"
        "        self.tp_group = get_tp_group()\n"
        "        self.world_size = self.tp_group.world_size\n"
        "        self.rank = self.tp_group.rank_in_group\n\n"
        "        self.hidden_size = config.hidden_size\n"
        "        self.intermediate_size_per_rank = config.intermediate_size // self.world_size",
        "    def __init__(self, config: Gemma4Config, layer_idx: int = 0):\n"
        "        super().__init__()\n\n"
        "        self.tp_group = get_tp_group()\n"
        "        self.world_size = self.tp_group.world_size\n"
        "        self.rank = self.tp_group.rank_in_group\n\n"
        "        self.hidden_size = config.hidden_size\n"
        "        # Per-layer MLP width. Sizing every layer at config.intermediate_size\n"
        "        # made the loader silently slice 12288 -> 6144 on E2B layers 15..34,\n"
        "        # discarding half of every gate/up/down projection.\n"
        "        self.layer_idx = layer_idx\n"
        "        self.intermediate_size = config.get_layer_intermediate_size(layer_idx)\n"
        "        self.intermediate_size_per_rank = (\n"
        "            self.intermediate_size // self.world_size\n"
        "        )\n"
        "        print(\n"
        "            f\"MLP_WIDTH layer={layer_idx} intermediate={self.intermediate_size} \"\n"
        "            f\"per_rank={self.intermediate_size_per_rank}\",\n"
        "            flush=True,\n"
        "        )",
        "mlp: per-layer intermediate size",
    )
    s = edit(
        s,
        "        self.mlp = Gemma4MLP(config)",
        "        self.mlp = Gemma4MLP(config, layer_idx=layer_idx)",
        "decoder: pass layer_idx to MLP",
    )
    if s != orig:
        open(path, "w").write(s)
    ast.parse(open(path).read())


def main() -> int:
    if "--selftest" in sys.argv:
        # Point this at a checkout of gemma4-e4b/vllm-neuron/src to dry-run
        # the patches without a device.
        src = os.environ.get("GEMMA4_E4B_SRC", "./gemma4-e4b/vllm-neuron/src")
        dst = "/tmp/e2bmlp"
        shutil.rmtree(dst, ignore_errors=True)
        os.makedirs(dst)
        shutil.copyfile(f"{src}/model_e4b_ple.py", f"{dst}/model.py")
        shutil.copyfile(f"{src}/config_e4b_ple.py", f"{dst}/config.py")
        cfg, mdl = f"{dst}/config.py", f"{dst}/model.py"
    else:
        md = None
        for base in sys.path:
            c = os.path.join(base, "vllm_neuron", "model")
            if os.path.isdir(c):
                md = c
                break
        assert md, "vllm_neuron/model not found"
        cfg = os.path.join(md, "gemma4", "config.py")
        mdl = os.path.join(md, "gemma4", "model.py")

    print("[fix_mlp] config:", cfg)
    patch_config(cfg)
    print("[fix_mlp] model:", mdl)
    patch_model(mdl)
    if FAILS:
        print("\n[fix_mlp] FAILED:")
        for f in FAILS:
            print("   -", f)
        return 1
    print("[fix_mlp] SUCCESS")
    return 0


if __name__ == "__main__":
    sys.exit(main())

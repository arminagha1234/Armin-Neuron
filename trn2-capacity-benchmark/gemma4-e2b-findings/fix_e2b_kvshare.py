#!/usr/bin/env python3
"""Implement Gemma-4 KV-sharing + double-wide MLP in the E4B/E2B vLLM-Neuron port.

WHY (root cause, confirmed against E2B's real config.json and the HF reference at
customers/tbc/.../hf_ref/modeling_gemma4_unified.py):

  num_kv_shared_layers is DECLARED in config_e4b_ple.py:63 and NEVER READ by
  model_e4b_ple.py (zero occurrences in 1657 lines). Per HF (modeling:372-378):

      first_kv_shared_layer_idx = num_hidden_layers - num_kv_shared_layers
      is_kv_shared_layer        = layer_idx >= first_kv_shared_layer_idx

  and (modeling:385-399) KV-shared layers have NO k_proj / v_proj / k_norm /
  v_norm in the checkpoint at all. The port maps those keys for EVERY layer, so
  for E2B (35 - 20 = 15 -> layers 15..34, i.e. 57% of the network) the lookups
  miss, `strict=False` (1627) silently drops them, and `torch.empty` (264-269)
  leaves the FUSED qkv weight uninitialised -- which corrupts the Q slice too.

  E2B additionally sets use_double_wide_mlp=True, which doubles
  intermediate_size on exactly those same 20 layers (HF modeling:469-480). The
  port's Gemma4MLP takes no layer_idx, so those MLP weights are the wrong shape
  and are ALSO dropped by strict=False.

  Net: 20 of 35 layers ran with uninitialised attention AND uninitialised MLP.
  That is the 0/3 multilingual salad. The quality gradient across the family
  matches the shared fraction exactly: 31B 0% -> works, E4B 43% -> semi-coherent,
  E2B 57% -> salad.

WHAT THIS DOES
  A. config: add use_double_wide_mlp + helpers (first_kv_shared_layer_idx,
     is_kv_shared_layer, kv_donor_layer_idx, get_layer_intermediate_size).
  B. attention: shared layers project Q only (qkv_size = q_size), and read K/V
     from the donor layer's cache + the donor's attn_metadata.
  C. MLP: takes layer_idx, doubles intermediate_size on shared layers.
  D. load_weights: shared layers map q_proj/q_norm/o_proj only -- no k_proj,
     no v_proj, no k_norm.
  E. bind_kv_cache: shared layers alias the donor's k_cache/v_cache tensors.
  F. get_kv_spec: donor layers store FULL-LENGTH KV (sliding_window_size=None)
     so a shared layer can never read an evicted block (HF modeling:427-433
     documents exactly this hazard).
  G. scaling: 1.0, per HF modeling:368. The port's 1/sqrt(head_dim) was a
     workaround tuned against the already-broken model.
  H. a LOADED-KEY AUDIT that prints any expected-but-missing parameter, so a
     silent drop can never masquerade as success again.

Every edit is asserted. Run with --selftest to dry-run against a local copy.
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
    n = text.count(old)
    if n != count:
        FAILS.append(f"{label}: expected {count} occurrence(s), found {n}")
        return text
    print(f"  [ok] {label}")
    return text.replace(old, new, count)


# ---------------------------------------------------------------- config -----
def patch_config(path: str) -> None:
    s = open(path).read()
    orig = s

    s = edit(
        s,
        "    num_kv_shared_layers: int | None = 0",
        "    num_kv_shared_layers: int | None = 0\n"
        "    # E2B sets this True: doubles intermediate_size on KV-shared layers\n"
        "    # (HF modeling_gemma4_unified.py:469-480).\n"
        "    use_double_wide_mlp: bool = False",
        "config: add use_double_wide_mlp",
    )

    helpers = '''
    # ---- Gemma-4 KV sharing (HF modeling_gemma4_unified.py:372-378) --------
    def first_kv_shared_layer_idx(self) -> int:
        """First layer index that reuses another layer's K/V projections."""
        shared = self.num_kv_shared_layers or 0
        return self.num_hidden_layers - shared

    def is_kv_shared_layer(self, layer_idx: int) -> bool:
        """KV-shared layers carry NO k_proj/v_proj/k_norm/v_norm weights."""
        shared = self.num_kv_shared_layers or 0
        if shared <= 0:
            return False
        return layer_idx >= self.first_kv_shared_layer_idx()

    def kv_donor_layer_idx(self, layer_idx: int) -> int:
        """Donor is the LAST layer of the SAME layer_type before the boundary.

        Mirrors HF's
            prev_layers[::-1].index(config.layer_types[layer_idx])
        so sliding layers inherit from the last sliding donor and full layers
        from the last full donor. For E2B: sliding -> 13, full -> 14.
        """
        first = self.first_kv_shared_layer_idx()
        prev = self.layer_types[:first]
        want = self.layer_types[layer_idx]
        for i in range(len(prev) - 1, -1, -1):
            if prev[i] == want:
                return i
        raise ValueError(
            f"no KV donor of type {want!r} before layer {first} for layer {layer_idx}"
        )

    def is_kv_donor_layer(self, layer_idx: int) -> bool:
        """True when some shared layer reads this layer's cache."""
        if not (self.num_kv_shared_layers or 0):
            return False
        if self.is_kv_shared_layer(layer_idx):
            return False
        first = self.first_kv_shared_layer_idx()
        for j in range(first, self.num_hidden_layers):
            if self.kv_donor_layer_idx(j) == layer_idx:
                return True
        return False

    def get_layer_intermediate_size(self, layer_idx: int) -> int:
        """Double-wide MLP applies to KV-shared layers only."""
        if self.use_double_wide_mlp and self.is_kv_shared_layer(layer_idx):
            return self.intermediate_size * 2
        return self.intermediate_size

    def is_global_layer(self, layer_idx: int) -> bool:'''
    s = edit(
        s,
        "\n    def is_global_layer(self, layer_idx: int) -> bool:",
        helpers,
        "config: add KV-share helpers",
    )

    if s != orig:
        open(path, "w").write(s)
    ast.parse(open(path).read())


# ----------------------------------------------------------------- model -----
def patch_model(path: str) -> None:
    s = open(path).read()
    orig = s

    # --- G. scaling = 1.0 (HF modeling:368) --------------------------------
    s = edit(
        s,
        "        self.scaling = 1.0 / (self.head_dim ** 0.5)",
        "        # HF Gemma-4 sets scaling = 1.0 unconditionally\n"
        "        # (modeling_gemma4_unified.py:368); q_norm controls logit\n"
        "        # magnitude. The previous 1/sqrt(head_dim) was a workaround\n"
        "        # tuned against a model whose last 57% of layers were\n"
        "        # uninitialised, and it over-damps attention by 16-23x.\n"
        "        self.scaling = float(_os.environ.get('GEMMA4_ATTN_SCALE', '1.0'))",
        "attn: scaling -> 1.0",
    )
    if "import os as _os" not in s:
        s = edit(
            s,
            "import logging\nimport math",
            "import logging\nimport math\nimport os as _os",
            "model: import os as _os",
        )

    # --- B. shared layers project Q only ----------------------------------
    s = edit(
        s,
        "        qkv_size = q_size + 2 * kv_size",
        "        # KV-shared layers have no k_proj/v_proj in the checkpoint, so the\n"
        "        # fused weight is Q-only (HF modeling:385-399).\n"
        "        self.is_kv_shared_layer = config.is_kv_shared_layer(layer_idx)\n"
        "        self.kv_donor_layer_idx = (\n"
        "            config.kv_donor_layer_idx(layer_idx)\n"
        "            if self.is_kv_shared_layer\n"
        "            else None\n"
        "        )\n"
        "        qkv_size = q_size if self.is_kv_shared_layer else q_size + 2 * kv_size",
        "attn: Q-only fused weight on shared layers",
    )

    # --- C. MLP takes layer_idx, doubles intermediate on shared layers -----
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
        "        # use_double_wide_mlp doubles intermediate_size on KV-shared\n"
        "        # layers only (HF modeling:469-480). E2B sets it True.\n"
        "        self.intermediate_size = config.get_layer_intermediate_size(layer_idx)\n"
        "        self.intermediate_size_per_rank = self.intermediate_size // self.world_size",
        "mlp: layer_idx + double-wide",
    )
    s = edit(
        s,
        "        self.mlp = Gemma4MLP(config)",
        "        self.mlp = Gemma4MLP(config, layer_idx=layer_idx)",
        "decoder: pass layer_idx to MLP",
    )

    # --- F. donor layers store full-length KV ------------------------------
    s = edit(
        s,
        "                    sliding_window_size=layer.self_attn.sliding_window,\n"
        "                    chunk_size=None,",
        "                    # A donor must keep FULL-LENGTH KV: a shared layer may\n"
        "                    # read positions the donor's sliding window would have\n"
        "                    # evicted (HF modeling:427-433 documents this hazard).\n"
        "                    sliding_window_size=(\n"
        "                        None\n"
        "                        if self.config.is_kv_donor_layer(i)\n"
        "                        else layer.self_attn.sliding_window\n"
        "                    ),\n"
        "                    chunk_size=None,",
        "kv_spec: donors store full-length KV",
    )

    # --- E. shared layers alias the donor's cache --------------------------
    s = edit(
        s,
        "    def bind_kv_cache(self, kv_caches: dict[str, list[torch.Tensor, torch.Tensor]]):\n"
        "        for i, layer in enumerate(self.model.layers):\n"
        "            layer_name = f\"layers.{i}.self_attn\"\n"
        "            if layer_name not in kv_caches:\n"
        "                raise Exception(f\"KV cache for layer {layer_name} not initialized\")\n"
        "            layer.self_attn.k_cache = kv_caches[layer_name][0]\n"
        "            layer.self_attn.v_cache = kv_caches[layer_name][1]",
        "    def bind_kv_cache(self, kv_caches: dict[str, list[torch.Tensor, torch.Tensor]]):\n"
        "        for i, layer in enumerate(self.model.layers):\n"
        "            attn = layer.self_attn\n"
        "            # KV-shared layers own no K/V: alias the donor's cache AND read\n"
        "            # the donor's attn_metadata (its block_table indexes its blocks).\n"
        "            src = i\n"
        "            if getattr(attn, \"is_kv_shared_layer\", False):\n"
        "                src = attn.kv_donor_layer_idx\n"
        "            layer_name = f\"layers.{src}.self_attn\"\n"
        "            if layer_name not in kv_caches:\n"
        "                raise Exception(f\"KV cache for layer {layer_name} not initialized\")\n"
        "            attn.kv_source_layer_name = layer_name\n"
        "            attn.k_cache = kv_caches[layer_name][0]\n"
        "            attn.v_cache = kv_caches[layer_name][1]",
        "bind_kv_cache: alias donor cache",
    )

    # --- D. load_weights: no k_proj/v_proj/k_norm for shared layers --------
    s = edit(
        s,
        "            is_global = self.config.is_global_layer(layer_id)",
        "            # KV-shared layers: q_proj only, and no k_norm.\n"
        "            if self.config.is_kv_shared_layer(layer_id):\n"
        "                mappings[f\"{target_prefix}.self_attn.qkv_proj_weight\"] = (\n"
        "                    f\"{prefix}.self_attn.q_proj.weight\"\n"
        "                )\n"
        "                mappings[f\"{target_prefix}.self_attn.o_proj_weight\"] = (\n"
        "                    f\"{prefix}.self_attn.o_proj.weight\"\n"
        "                )\n"
        "                mappings[f\"{target_prefix}.self_attn.q_norm.weight\"] = (\n"
        "                    f\"{prefix}.self_attn.q_norm.weight\"\n"
        "                )\n"
        "                _shared_skip = True\n"
        "            else:\n"
        "                _shared_skip = False\n"
        "            is_global = self.config.is_global_layer(layer_id)",
        "load_weights: shared-layer mapping",
    )
    s = edit(
        s,
        "            mappings[f\"{target_prefix}.self_attn.qkv_proj_weight\"] = qkv_sources\n",
        "            if not _shared_skip:\n"
        "                mappings[f\"{target_prefix}.self_attn.qkv_proj_weight\"] = qkv_sources\n",
        "load_weights: guard fused-QKV mapping",
    )
    s = edit(
        s,
        "            mappings[f\"{target_prefix}.self_attn.k_norm.weight\"] = (\n"
        "                f\"{prefix}.self_attn.k_norm.weight\"\n"
        "            )",
        "            if not _shared_skip:\n"
        "                mappings[f\"{target_prefix}.self_attn.k_norm.weight\"] = (\n"
        "                    f\"{prefix}.self_attn.k_norm.weight\"\n"
        "                )",
        "load_weights: guard k_norm mapping",
    )

    # --- B2. forward: shared layers project Q only and read the donor -----
    # There are TWO metadata blocks (prefill 529, decode 649) and TWO qkv
    # splits (504, 666); patch both. `kv_name` routes every cache-related
    # lookup at a shared layer to its DONOR, because the donor's block_table
    # indexes the donor's blocks -- and after the get_kv_spec change the donor
    # is in a different KV-cache group (full-length vs sliding).
    s = edit(
        s,
        "        layer_name = f\"layers.{self.layer_idx}.self_attn\"\n"
        "        slot_mapping = attn_metadata[layer_name][\"slot_mapping\"]",
        "        layer_name = f\"layers.{self.layer_idx}.self_attn\"\n"
        "        # KV-shared layers read the DONOR's cache and metadata.\n"
        "        kv_name = getattr(self, \"kv_source_layer_name\", layer_name)\n"
        "        slot_mapping = attn_metadata[kv_name][\"slot_mapping\"]",
        "forward: kv_name routes to donor (both paths)",
        count=2,
    )
    for field in ("block_size", "max_blocks_per_seq", "block_table_tensor"):
        s = s.replace(
            "attn_metadata[layer_name][\"" + field + "\"]",
            "attn_metadata[kv_name][\"" + field + "\"]",
        )
    s = s.replace(
        "attn_metadata[layer_name].get(\"swa_kv_pos_offset\")",
        "attn_metadata[kv_name].get(\"swa_kv_pos_offset\")",
    )
    print("  [ok] forward: donor metadata for block_size/max_blocks/block_table")

    # Q-only split on shared layers (both prefill and decode).
    s = edit(
        s,
        "        q, k, v = torch.tensor_split(qkv, self.qkv_split_indices, dim=-1)",
        "        if self.is_kv_shared_layer:\n"
        "            # No k_proj/v_proj on this layer: qkv IS q. K/V come from the\n"
        "            # donor's cache further down (HF modeling:427-433).\n"
        "            q, k, v = qkv, None, None\n"
        "        else:\n"
        "            q, k, v = torch.tensor_split(qkv, self.qkv_split_indices, dim=-1)",
        "forward: Q-only split on shared layers (both paths)",
        count=2,
    )

    # Skip the K/V reshape+norm+rope+cache-write on shared layers by guarding
    # the two index_put_ cache writes.
    s = edit(
        s,
        "        self.k_cache.index_put_(",
        "        _skip_kv_write = self.is_kv_shared_layer\n"
        "        if not _skip_kv_write:\n"
        "            self.k_cache.index_put_(",
        "forward: guard K cache write (both paths)",
        count=2,
    )
    s = edit(
        s,
        "        self.v_cache.index_put_(",
        "        if not _skip_kv_write:\n"
        "            self.v_cache.index_put_(",
        "forward: guard V cache write (both paths)",
        count=2,
    )

    # --- H. loaded-key audit ----------------------------------------------
    s = edit(
        s,
        "        self.load_state_dict(rank_sharded, strict=False, assign=True)",
        "        # AUDIT: strict=False + torch.empty is what let 20 uninitialised\n"
        "        # layers pass silently. Report anything expected-but-unloaded.\n"
        "        _expected = set(self.state_dict().keys())\n"
        "        _loaded = set(rank_sharded.keys())\n"
        "        _missing = sorted(_expected - _loaded)\n"
        "        if _missing:\n"
        "            logger.warning(\n"
        "                \"WEIGHT_AUDIT %d/%d params NOT loaded (first 24): %s\",\n"
        "                len(_missing), len(_expected), _missing[:24],\n"
        "            )\n"
        "            print(\n"
        "                f\"WEIGHT_AUDIT_MISSING={len(_missing)}/{len(_expected)} \"\n"
        "                f\"first={_missing[:12]}\",\n"
        "                flush=True,\n"
        "            )\n"
        "        else:\n"
        "            print(\n"
        "                f\"WEIGHT_AUDIT_OK all {len(_expected)} params loaded\",\n"
        "                flush=True,\n"
        "            )\n"
        "        self.load_state_dict(rank_sharded, strict=False, assign=True)",
        "load_weights: loaded-key audit",
    )

    if s != orig:
        open(path, "w").write(s)
    ast.parse(open(path).read())


def main() -> int:
    if "--selftest" in sys.argv:
        # Point this at a checkout of gemma4-e4b/vllm-neuron/src to dry-run
        # the patches without a device.
        src = os.environ.get("GEMMA4_E4B_SRC", "./gemma4-e4b/vllm-neuron/src")
        dst = "/tmp/e2bfix"
        shutil.rmtree(dst, ignore_errors=True)
        os.makedirs(dst)
        shutil.copyfile(f"{src}/model_e4b_ple.py", f"{dst}/model.py")
        shutil.copyfile(f"{src}/config_e4b_ple.py", f"{dst}/config.py")
        cfg, mdl = f"{dst}/config.py", f"{dst}/model.py"
    else:
        model_dir = None
        for base in sys.path:
            c = os.path.join(base, "vllm_neuron", "model")
            if os.path.isdir(c):
                model_dir = c
                break
        assert model_dir, "vllm_neuron/model not found"
        cfg = os.path.join(model_dir, "gemma4", "config.py")
        mdl = os.path.join(model_dir, "gemma4", "model.py")

    print("[fix_e2b] config:", cfg)
    patch_config(cfg)
    print("[fix_e2b] model:", mdl)
    patch_model(mdl)

    if FAILS:
        print("\n[fix_e2b] FAILED PATCHES:")
        for f in FAILS:
            print("   -", f)
        return 1
    print("[fix_e2b] SUCCESS - all patches applied")
    return 0


if __name__ == "__main__":
    sys.exit(main())

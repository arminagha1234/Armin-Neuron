#!/usr/bin/env python3
"""Complete the Gemma-4 E2B KV-sharing port: the shared-layer ATTENTION.

Runs AFTER fix_e2b_kvshare.py (which wires config/load/bind/kv_spec/Q-only-split/
cache-write-skip/donor-metadata). That leaves ONE gap: a shared layer projects Q
only, so `k = k.view(...)` crashes on None. HF reuses the donor's K/V; under
vLLM-Neuron's paged cache the donor's cache is aliased into the shared layer
(bind_kv_cache), and the donor (lower index) runs first, so its K/V are already
in the shared layer's k_cache/v_cache.

This patch adds two dedicated shared-layer attention methods and routes the two
forward paths to them, and reverts the scaling to the per-layer 1/sqrt(head_dim)
that 31B is coherent with on trn2 (HF's 1.0 is the inf2 path; drop patch G here).

  PREFILL: gather the donor's current chunk from the aliased cache at slot_mapping
           (already normed+RoPE'd by the donor), Q from own proj+norm+rope,
           GQA-expand + causal/sliding mask + _manual_sdpa.
  DECODE : same Q path, then reuse the existing block-table gather over full
           history (the donor wrote the current token before this layer runs).

Dry-run: GEMMA4_E4B_SRC=<repo>/gemma4-e4b/vllm-neuron/src python3 this --selftest
"""
from __future__ import annotations
import os, sys, shutil, ast

FAILS = []
def edit(text, old, new, label, count=1):
    n = text.count(old)
    if n != count:
        FAILS.append(f"{label}: found {n} expected {count}")
        return text
    return text.replace(old, new, count)

SHARED_METHODS = '''
    def _shared_kv_from_cache_prefill(self, attn_metadata):
        """Gather the donor's current-chunk K/V from the aliased cache.

        The donor wrote them at slot_mapping already QK/V-normed and RoPE'd, so
        we do NOT re-apply norm or rope. Cache: [num_blocks, nkh, block_size, Dh].
        """
        layer_name = f"layers.{self.layer_idx}.self_attn"
        kv_name = getattr(self, "kv_source_layer_name", layer_name)
        slot_mapping = attn_metadata[kv_name]["slot_mapping"]
        block_size = attn_metadata[kv_name]["block_size"]
        bi = slot_mapping // block_size
        pi = slot_mapping % block_size
        # advanced index (bi over blocks, pi over positions) + slice over heads
        # -> [tokens, nkh, Dh]
        k = self.k_cache[bi, :, pi]
        v = self.v_cache[bi, :, pi]
        if self.k_cache.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
            k = k.to(self.dtype) / self.k_scale_float
            v = v.to(self.dtype) / self.v_scale_float
        else:
            k = k.to(self.dtype)
            v = v.to(self.dtype)
        # [tokens, nkh, Dh] -> [nkh, tokens, Dh]
        return k.transpose(0, 1).contiguous(), v.transpose(0, 1).contiguous()

    def _shared_prefill_attn(self, qkv, positions, attn_metadata, hidden_states, tokens):
        nkh = self.num_key_value_heads_per_rank
        q = qkv.view(tokens, self.num_attention_heads_per_rank, self.head_dim).transpose(0, 1)
        q = self.q_norm(q)
        cos, sin = self.rotary_emb(positions, device=hidden_states.device, dtype=hidden_states.dtype)
        q, _ = self._apply_partial_rotary(q, q, cos, sin)
        k, v = self._shared_kv_from_cache_prefill(attn_metadata)
        # GQA expand (no alloc)
        k = k.unsqueeze(1).expand(nkh, self.num_key_value_groups, tokens, self.head_dim).reshape(self.num_attention_heads_per_rank, tokens, self.head_dim)
        v = v.unsqueeze(1).expand(nkh, self.num_key_value_groups, tokens, self.head_dim).reshape(self.num_attention_heads_per_rank, tokens, self.head_dim)
        row_idx = torch.arange(tokens, device=hidden_states.device).unsqueeze(1)
        col_idx = torch.arange(tokens, device=hidden_states.device).unsqueeze(0)
        causal = col_idx <= row_idx
        if self.sliding_window is not None:
            causal = causal & ((row_idx - col_idx) < self.sliding_window)
        attn_mask = torch.where(
            causal,
            torch.zeros(1, dtype=self.dtype, device=hidden_states.device),
            torch.full((1,), float("-inf"), dtype=self.dtype, device=hidden_states.device),
        )
        attn_output = self._manual_sdpa(q, k, v, attn_mask)
        attn_output = attn_output.transpose(0, 1).contiguous().view(
            tokens, self.num_attention_heads_per_rank * self.head_dim
        )
        attn_output = torch.matmul(attn_output, self.o_proj_weight)
        if self.world_size > 1:
            attn_output = self.tp_group.reduce_scatter(attn_output, dim=0)
        return attn_output

    def _shared_decode_attn(self, qkv, positions, attn_metadata, hidden_states, tokens):
        nkh = self.num_key_value_heads_per_rank
        layer_name = f"layers.{self.layer_idx}.self_attn"
        kv_name = getattr(self, "kv_source_layer_name", layer_name)
        block_size = attn_metadata[kv_name]["block_size"]
        max_blocks_per_seq = attn_metadata[kv_name]["max_blocks_per_seq"]
        block_table = attn_metadata[kv_name]["block_table_tensor"]
        swa_kv_pos_offset = attn_metadata[kv_name].get("swa_kv_pos_offset")
        B = block_table.shape[0]
        S_decode = tokens // B
        q = qkv.view(tokens, self.num_attention_heads_per_rank, self.head_dim).transpose(0, 1)
        q = self.q_norm(q)
        cos, sin = self.rotary_emb(positions, device=hidden_states.device, dtype=hidden_states.dtype)
        q, _ = self._apply_partial_rotary(q, q, cos, sin)
        # Gather full history from the aliased donor cache (donor wrote current tok)
        S_ctx = max_blocks_per_seq * block_size
        flat_indices = torch.clamp(block_table.reshape(-1), min=0)
        k_blocks = torch.index_select(self.k_cache, 0, flat_indices)
        v_blocks = torch.index_select(self.v_cache, 0, flat_indices)
        k_gathered = k_blocks.view(B, max_blocks_per_seq, nkh, block_size, self.head_dim).permute(0, 2, 1, 3, 4).reshape(B, nkh, S_ctx, self.head_dim)
        v_gathered = v_blocks.view(B, max_blocks_per_seq, nkh, block_size, self.head_dim).permute(0, 2, 1, 3, 4).reshape(B, nkh, S_ctx, self.head_dim)
        if self.k_cache.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
            k_gathered = k_gathered.to(self.dtype) / self.k_scale_float
            v_gathered = v_gathered.to(self.dtype) / self.v_scale_float
        k_gathered = k_gathered.unsqueeze(2).expand(B, nkh, self.num_key_value_groups, S_ctx, self.head_dim).reshape(B, self.num_attention_heads_per_rank, S_ctx, self.head_dim)
        v_gathered = v_gathered.unsqueeze(2).expand(B, nkh, self.num_key_value_groups, S_ctx, self.head_dim).reshape(B, self.num_attention_heads_per_rank, S_ctx, self.head_dim)
        q = q.view(self.num_attention_heads_per_rank, B, S_decode, self.head_dim).permute(1, 0, 2, 3)
        pos = positions.view(B, S_decode)
        ctx_pos = torch.arange(S_ctx, device=positions.device)
        if swa_kv_pos_offset is not None:
            ctx_pos = ctx_pos.view(1, S_ctx) + swa_kv_pos_offset.view(B, 1)
            causal_mask = ctx_pos.unsqueeze(1) <= pos.unsqueeze(-1)
        else:
            causal_mask = ctx_pos.view(1, 1, S_ctx) <= pos.unsqueeze(-1)
        if self.sliding_window is not None:
            window_start = torch.clamp(pos - self.sliding_window + 1, min=0)
            if swa_kv_pos_offset is not None:
                in_window = ctx_pos.unsqueeze(1) >= window_start.unsqueeze(-1)
            else:
                in_window = ctx_pos.view(1, 1, S_ctx) >= window_start.unsqueeze(-1)
            mask = causal_mask & in_window
        else:
            mask = causal_mask
        attn_mask = torch.where(
            mask.view(B, 1, S_decode, S_ctx),
            torch.zeros(1, dtype=self.dtype, device=positions.device),
            torch.full((1,), float("-inf"), dtype=self.dtype, device=positions.device),
        )
        scores = torch.matmul(q.float(), k_gathered.float().transpose(-2, -1)) * self.scaling
        scores = scores + attn_mask.float()
        attn_weights = torch.nn.functional.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v_gathered.float()).to(q.dtype)
        attn_output = attn_output.permute(0, 2, 1, 3).reshape(tokens, self.num_attention_heads_per_rank * self.head_dim)
        output = torch.matmul(attn_output, self.o_proj_weight)
        if self.world_size > 1:
            self.tp_group.all_reduce(output)
        return output
'''

def patch_model(path):
    s = open(path).read()
    orig = s
    # 1. revert patch-G scaling to per-layer 1/sqrt(head_dim) (trn2 correct)
    s = edit(
        s,
        "        self.scaling = float(_os.environ.get('GEMMA4_ATTN_SCALE', '1.0'))",
        "        self.scaling = 1.0 if _os.environ.get('GEMMA4_HF_SCALE','1')=='1' else (1.0 / (self.head_dim ** 0.5))",
        "revert patch-G scaling",
    )
    # shared layers own no k_norm (K/V arrive pre-normed from the donor cache);
    # HF guards its creation, and load_weights (patch D) does not map it -> a
    # bare k_norm.weight param with no checkpoint source fails the strict loader.
    s = edit(
        s,
        "        self.k_norm = Gemma4RMSNorm(self.head_dim, config.rms_norm_eps, self.dtype)",
        "        if not self.is_kv_shared_layer:\n"
        "            self.k_norm = Gemma4RMSNorm(self.head_dim, config.rms_norm_eps, self.dtype)",
        "init: shared layers own no k_norm",
    )
    s = edit(
        s,
        "        self.v_norm = Gemma4VNorm(self.head_dim, config.rms_norm_eps)",
        "        if not self.is_kv_shared_layer:\n"
        "            self.v_norm = Gemma4VNorm(self.head_dim, config.rms_norm_eps)",
        "init: shared layers own no v_norm",
    )
    # Q-only fused weight on shared layers needs a PLAIN sharding loader, not the
    # [Q,K,V] fused loader (which raises "expects [Q, K, V] slices in order").
    s = edit(
        s,
        "        set_weight_loader(\n"
        "            self.qkv_proj_weight,\n"
        "            fused_qkv_weight_loader(\n"
        "                q_size=self.q_size,\n"
        "                kv_size=self.kv_size,\n"
        "                shard_dim=1,\n"
        "                num_shards=self.world_size,\n"
        "                is_storage_transposed=True,\n"
        "                num_kv_replicas=self.num_kv_replicas,\n"
        "            ),\n"
        "        )",
        "        if self.is_kv_shared_layer:\n"
        "            set_weight_loader(\n"
        "                self.qkv_proj_weight,\n"
        "                sharding_weight_loader(\n"
        "                    shard_dim=1,\n"
        "                    shard_size=self.q_size // self.world_size,\n"
        "                    num_shards=self.world_size,\n"
        "                    is_storage_transposed=True,\n"
        "                ),\n"
        "            )\n"
        "        else:\n"
        "            set_weight_loader(\n"
        "                self.qkv_proj_weight,\n"
        "                fused_qkv_weight_loader(\n"
        "                    q_size=self.q_size,\n"
        "                    kv_size=self.kv_size,\n"
        "                    shard_dim=1,\n"
        "                    num_shards=self.world_size,\n"
        "                    is_storage_transposed=True,\n"
        "                    num_kv_replicas=self.num_kv_replicas,\n"
        "                ),\n"
        "            )",
        "setup: Q-only sharding loader on shared layers",
    )
    # 2. inject the shared-layer methods right before `def forward(`
    s = edit(
        s,
        "    def forward(\n        self,\n        hidden_states: torch.Tensor,\n        positions: torch.LongTensor | None,",
        SHARED_METHODS + "\n    def forward(\n        self,\n        hidden_states: torch.Tensor,\n        positions: torch.LongTensor | None,",
        "inject shared-layer methods",
    )
    # 3. route PREFILL shared branch (distinguished by preceding comment)
    s = edit(
        s,
        "        # head_dim constraints that may not hold for all Gemma4 layers)\n"
        "        qkv = torch.matmul(hidden_states, self.qkv_proj_weight)\n"
        "\n"
        "        if self.is_kv_shared_layer:\n"
        "            # No k_proj/v_proj on this layer: qkv IS q. K/V come from the\n"
        "            # donor's cache further down (HF modeling:427-433).\n"
        "            q, k, v = qkv, None, None\n"
        "        else:\n"
        "            q, k, v = torch.tensor_split(qkv, self.qkv_split_indices, dim=-1)",
        "        # head_dim constraints that may not hold for all Gemma4 layers)\n"
        "        qkv = torch.matmul(hidden_states, self.qkv_proj_weight)\n"
        "\n"
        "        if self.is_kv_shared_layer:\n"
        "            return self._shared_prefill_attn(qkv, positions, attn_metadata, hidden_states, tokens)\n"
        "        q, k, v = torch.tensor_split(qkv, self.qkv_split_indices, dim=-1)",
        "route prefill shared branch",
    )
    # 4. route DECODE shared branch (distinguished by preceding comment)
    s = edit(
        s,
        "        # Step 1: QKV Projection (manual, not fused megakernel)\n"
        "        qkv = torch.matmul(hidden_states, self.qkv_proj_weight)\n"
        "        if self.is_kv_shared_layer:\n"
        "            # No k_proj/v_proj on this layer: qkv IS q. K/V come from the\n"
        "            # donor's cache further down (HF modeling:427-433).\n"
        "            q, k, v = qkv, None, None\n"
        "        else:\n"
        "            q, k, v = torch.tensor_split(qkv, self.qkv_split_indices, dim=-1)",
        "        # Step 1: QKV Projection (manual, not fused megakernel)\n"
        "        qkv = torch.matmul(hidden_states, self.qkv_proj_weight)\n"
        "        if self.is_kv_shared_layer:\n"
        "            return self._shared_decode_attn(qkv, positions, attn_metadata, hidden_states, tokens)\n"
        "        q, k, v = torch.tensor_split(qkv, self.qkv_split_indices, dim=-1)",
        "route decode shared branch",
    )
    if s != orig and not FAILS:
        open(path, "w").write(s)
    ast.parse(open(path).read())

def main():
    if "--selftest" in sys.argv:
        src = os.environ.get("GEMMA4_E4B_SRC", "./gemma4-e4b/vllm-neuron/src")
        dst = "/tmp/e2bfix"  # reuse the dir fix_e2b_kvshare wrote to
        mdl = f"{dst}/model.py"
        assert os.path.exists(mdl), "run fix_e2b_kvshare.py --selftest FIRST"
    else:
        model_dir = None
        for base in sys.path:
            c = os.path.join(base, "vllm_neuron", "model")
            if os.path.isdir(c):
                model_dir = c; break
        assert model_dir, "vllm_neuron/model not found"
        mdl = os.path.join(model_dir, "gemma4", "model.py")
    print("[attn] model:", mdl)
    patch_model(mdl)
    if FAILS:
        print("[attn] FAILED:"); [print("  -", f) for f in FAILS]; return 1
    print("[attn] SUCCESS - shared-layer attention wired")
    return 0

if __name__ == "__main__":
    sys.exit(main())

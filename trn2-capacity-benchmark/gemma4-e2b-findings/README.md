# Gemma-4-E2B-it on vLLM-Neuron: one real bug, two falsified theories

**Status: still incoherent (0/3).** This is published as a negative result,
because the two dead ends below are expensive to rediscover and the one real bug
is easy to hit on any Gemma-4 model with `use_double_wide_mlp`.

The port under test is the E4B PLE model (`gemma4-e4b/vllm-neuron/src/`) served
against E2B weights.

## Configuration that matters

```
num_hidden_layers        35
num_kv_shared_layers     20      -> layers 15..34 are "KV-shared"
use_double_wide_mlp      True
num_key_value_heads      1
head_dim                 256  (sliding) / global_head_dim 512
hidden_size              1536
intermediate_size        6144
layer_types              28 sliding_attention + 7 full_attention
```

For comparison, 31B has `num_kv_shared_layers=0` and `use_double_wide_mlp=False`
— which is why the same code serves 31B correctly at 3/3.

## THE REAL BUG: `use_double_wide_mlp` silently truncates half of every MLP

Read straight from the checkpoint's safetensors header (an HTTP range request for
the header, no 11 GB download):

```
gate_proj shapes: {(6144, 1536): 15,  (12288, 1536): 20}

layer  5 (normal)  mlp.gate_proj.weight  [ 6144, 1536]
layer 20 (shared)  mlp.gate_proj.weight  [12288, 1536]
```

Exactly 15 normal + 20 double-wide, matching
`first_kv_shared_layer_idx = 35 - 20 = 15`. Per HF, `use_double_wide_mlp` doubles
`intermediate_size` on the KV-shared tail.

The port sizes **every** MLP at `config.intermediate_size // world_size` = 6144.
The weight loader is
`sharding_weight_loader(shard_dim=..., shard_size=intermediate_size_per_rank)`,
and given a `[12288, 1536]` source with `shard_size=6144` **it does not raise** —
it takes a 6144-wide slice. So half of every gate/up/down projection was
discarded on 20 of 35 layers.

`fix_e2b_mlp.py` makes the width per-layer. Verified in the run log:
`20 x 12288, 15 x 6144`.

**This did not restore coherence** — but it is unambiguously a bug, and it will
silently degrade any Gemma-4 checkpoint that sets the flag.

### Why the weight audit said everything was fine

`audit_e2b.py` diffs expected against loaded parameter keys. It reported:

```
WEIGHT_AUDIT total_expected=531 loaded=531 missing=0
```

Zero missing. True, and misleading: this is **silent truncation**, not a missing
key. A key-presence audit cannot see it. Only a shape comparison can.

## FALSIFIED #1: "KV-shared layers have no k_proj/v_proj"

The HF reference (`modeling_gemma4_unified.py:385-399`) does not instantiate
`k_proj`, `v_proj`, `k_norm` or `v_norm` for KV-shared layers, which suggests
those tensors are absent from the checkpoint and would therefore load as
uninitialised `torch.empty` under `strict=False`. That theory predicts ~20
missing `qkv_proj_weight` entries.

Both diagnostics refute it:

```
WEIGHT_AUDIT ... missing=0
layers WITHOUT k_proj: NONE
layers WITHOUT v_proj: NONE
distinct per-layer signatures: 1   (all 35 layers have identical 17 tensors)
```

Google exports all projections for every layer. Nothing is uninitialised.

## FALSIFIED #2: attention scaling

The port sets `self.scaling = 1.0 / sqrt(head_dim)` with a comment describing it
as an inf2 precision workaround, while HF Gemma-4 sets `self.scaling = 1.0`
unconditionally (`modeling_gemma4_unified.py:368`) and relies on `q_norm` to
control logit magnitude. For E2B this is two different wrong temperatures — 1/16
on sliding layers, 1/22.6 on global — and E2B is the most exposed config, since
`num_key_value_heads=1` with 8 query heads leaves no head diversity to absorb a
flattened softmax.

Tested `scaling=1.0` on top of the MLP fix: **still 0/3.**

## Remaining prime suspect: KV-sharing semantics

The checkpoint *contains* `k_proj`/`v_proj` for layers 15..34, but HF never
instantiates them and instead reuses the donor layer's K/V via
`shared_kv_states[layer_type]` (`modeling_gemma4_unified.py:427-446`). Donors are
the last layer of the **same** `layer_type` before the boundary — for E2B,
sliding -> layer 13 and full -> layer 14. Shared layers also must **not** write
to the cache.

So the port computing K/V from tensors HF ignores is a genuine semantic
divergence, and it is the last untested candidate.

`fix_e2b_kvshare.py` is a complete 17-patch implementation of this
(per-layer-type donor resolution, Q-only fused weight on shared layers, donor
cache aliasing in `bind_kv_cache`, donors forced to full-length KV so a shared
layer cannot read an evicted block, guarded cache writes). Its config helpers are
unit-verified against E2B's real numbers:

```
shared layers: 15..34 (20 of 35)
donors: [13 sliding, 14 full]   type-mismatched donors: NONE
intermediate: layer 0 -> 6144, layer 20 -> 12288
31B regression: no shared layers, no donors, intermediate 21504 unchanged
```

**It is deliberately not wired in.** The forward path needs `shared_kv_states`
plumbed through `Gemma4Model -> Gemma4DecoderLayer -> Gemma4Attention` under
Dynamo; with the current patch set `k`/`v` become `None` and crash the reshape
immediately after the QKV split. That plumbing is the next step and it is real
design work, not a text edit.

## Files

| File | Purpose |
|---|---|
| `fix_e2b_mlp.py` | **the real fix** — per-layer MLP width. Dry-runnable with `--selftest`. |
| `audit_e2b.py` | key-presence audit. Falsified theory #1. Keep it: it turns silent drops into loud ones. |
| `fix_e2b_kvshare.py` | complete KV-sharing implementation, config helpers verified, forward plumbing incomplete. |

## Practical advice

E2B already **exceeds** a 50 RPS target through the native path (9,688 tok/s
prefill, ~177 RPS/box extrapolated), so the vLLM path is not on the critical
path for capacity. Fix it for correctness, not for throughput.

And when a model produces token salad, check **shapes**, not just key presence.
A loader that slices instead of raising will pass every audit you write.

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

## Update — root cause corrected, and the exact remaining work

Revisited against the canonical HF source
(`transformers/models/gemma4_unified/modeling_gemma4_unified.py`), which resolves
the contradiction between this file's earlier notes and the weight audit.

**The premise "shared layers have no k/v in the checkpoint" was wrong.** HF does
not *instantiate* `k_proj/v_proj/k_norm/v_norm` for a shared layer
(`if not self.is_kv_shared_layer:` guards their construction), but the checkpoint
still *ships* those tensors for all 35 layers — Google exported them. So they load
(audit `missing=0`) and are simply unused. Both observations are now consistent.

**The real contract** (`Gemma4UnifiedTextAttention.forward`, lines ~427-446):
- A **donor** is the last non-shared layer of each `layer_type`. It computes K/V,
  writes the cache, and stashes its K/V in `shared_kv_states[layer_type]`
  (`store_full_length_kv`). For E2B (35 layers, 20 shared → first shared = 15):
  sliding donor = 13, full donor = 14.
- A **shared layer** reads `shared_kv_states[layer_type]`, projects **Q only**, and
  does **not** write the cache.

### What is actually wired

`fix_e2b_kvshare.py` is ~90% of the port, not "not wired in": config helpers,
`load_weights` (Q-only mapping), `bind_kv_cache` (alias the donor's cache),
`get_kv_spec` (donor stores full-length KV), Q-only fused weight, cache-write skip,
and donor-metadata routing (`kv_name`) are all applied and dry-run clean. The **one**
remaining gap is the shared-layer *attention computation*: after the Q-only split sets
`k = v = None`, the code still runs `k.view(...)` and crashes.

### The exact remaining work (turnkey)

Under vLLM-Neuron's **paged** attention no HF-style `shared_kv_states` threading is
needed — the donor's cache alias already carries the K/V, and the donor (lower index)
runs first in the unrolled layer loop:

- **Decode** is nearly free. `forward_decode` already gathers the full history from the
  (aliased donor) cache in Step 6. Guard the k/v reshape + QK-norm + RoPE so a shared
  layer computes Q only, then let Step 6 run unchanged.
- **Prefill** needs a gather. The full-prefill branch attends *in-memory* K/V. For a
  shared layer, gather the donor's current chunk from the aliased cache at `slot_mapping`
  (the donor wrote it there already normed and RoPE'd), skip re-norm/re-RoPE, then run
  the existing GQA-expand + mask + `_manual_sdpa`.

### Do NOT set the attention scale to 1.0 on trn2

`fix_e2b_kvshare.py`'s patch G rewrites `scaling` to `1.0` (HF canonical). The evidence
says that is wrong for this stack: **31B — the same model code, no KV sharing — is
coherent 3/3 on trn2 with the per-layer `1/sqrt(head_dim)`** the port already uses, and
the `_manual_sdpa` fallback for head_dim 256/512 is the *inf2* path that degrades, not
the trn2 one. Keep the per-layer `1/sqrt(head_dim)`; drop patch G on trn2.

**Status:** root cause corrected and the port wired to the attention boundary; the
shared-layer gather above is dry-run-designed but **not yet device-validated**. E2B
vLLM is a correctness gap, not a capacity one (native already clears 50 RPS), so it is
not on the capacity critical path — which is why it is documented to this precision
rather than rushed onto a device.

## Device validation of the attention completion (`fix_e2b_kvshare_attn.py`)

The shared-layer attention was implemented (`fix_e2b_kvshare_attn.py`) and run on a
`trn2.48xlarge` via Kaizen. It uses the **cache-alias** path (no forward-signature
threading): a shared layer projects Q only, and reads K/V from the donor's aliased
paged cache — decode via the existing full-history block-table gather, prefill via a
gather of the donor's current chunk at `slot_mapping` (the donor wrote it already
QK/V-normed and RoPE'd). Scaling is kept at the per-layer `1/sqrt(head_dim)` that 31B
is coherent with on trn2 (patch G's `1.0` is reverted).

Five device iterations, each failing fast at load (cheap to iterate):

| # | Result | Fix |
|---|---|---|
| 1 | `k_norm.weight` for layer 15 unmapped | shared layers don't create `k_norm` (HF guards it) |
| 2 | `fused_qkv_weight_loader expects [Q,K,V]` | Q-only fused weight gets a plain sharding loader + guard `v_norm` |
| 3 | `NCC_EOOM002` 42.8 GB > 24 GB (TP=1) | E2B can't use TP≥2 (`num_kv_heads=1` → broken replica path); cap the KV pool |
| 4 | still OOM (I/O ~fixed vs LEN/mns) | `--num-gpu-blocks-override 128`, `LEN=128`, `max_num_seqs=1` |
| 5 | **compiles + serves** | — |

**Status: compiles and serves; not yet coherent (0/3).** This is real progress — the
approach *traces on Neuron* (the aliased-cache write-then-read and the
`k_cache[bi, :, pi]` gather both compile; the gather index math is verified correct
against the donor's `index_put_`), and the failure mode changed from the pre-fix
**multilingual salad** to **degenerate English** ("you are you are you are…"). That
shift means the KV-share path is partially correct, not inert.

### The remaining issue and the robust fix

The most likely residual cause is the **prefill cache write-then-read ordering**. In
the unrolled layer loop the donor (lower index) writes its K/V to the shared cache
in-place, and the shared layer reads it back; a graph compiler is free to reorder two
in-place ops it deems independent, so a shared layer may attend to **stale** K/V.
Small per-token errors over a 57%-shared network compound into repetition.

The robust fix is HF's approach: **thread the donor's freshly-computed (post-RoPE) K/V
through the forward** (`Gemma4Model → DecoderLayer → Attention`) in a
`shared_kv_states[layer_type]` dict — a donor stashes, a shared layer reads it directly
(in-memory, no cache-ordering dependency) for prefill; decode can keep the cache path
since prior history is already committed. Under the *unrolled* loop this is
compile-time tensor routing, so it should trace. That is the next step; the
cache-alias path here is the systems-level de-risking that had to come first.

## Threading variant — and the finding that ordering was never the bug

`fix_e2b_kvshare_thread.py` implements the HF-faithful alternative: the donor stashes
its post-norm/post-RoPE K/V into the already-threaded `attn_metadata` dict, and the
shared layer reads it **in-memory** for prefill (no cache read at all), keeping decode
on the cache-alias path. Threading a static-keyed dict through the unrolled layer loop
is a proven trace-safe pattern here (`attn_metadata`, `aux_hidden_states`).

It compiles, serves, and produces output **byte-identical** to the cache-alias variant
("Please explain you are you are…"). That is a decisive negative result: the two K/V
sourcing paths are numerically equivalent, so **the prefill write-then-read ordering
was never the bug** — consistent with the fact that torch-neuronx functionalization
(`InPlaceToOutOfPlacePass`) rewrites an in-place `index_put_` write and its later read
to preserve read-after-write within one compiled graph. The degeneration is present
from the **first generated token**, so it originates in **prefill**, not decode.

Ruled out as the cause (via an HF-reference audit): the **double-wide MLP** (fixed by
`fix_e2b_kvshare.py` patch C — per-layer intermediate size + sharding), **MQA**
(`num_global_key_value_heads=None`→1 via the E4B config's None-handling), and the
**sliding-window mask** (math verified `[i-511, i]`). The 31B port already exercises
global head_dim 512, partial rotary, q/k/v-norm and logit softcap coherently, so those
are exonerated too.

**Live suspects, in order:** (1) Per-Layer Embeddings (PLE) correctness — engaged
(`PLE_ENGAGED dim=256`) but never validated numerically against HF; (2) a residual
shared-layer attention-math issue. The decisive next step is a **per-layer hidden-state
parity harness vs HF-CPU** to find the first divergent layer (embedding/PLE → pre-0;
global-attn → a full_attention layer <15; KV-share → layer 15). That localizes the bug
instead of guessing.

## Research synthesis — what's exonerated, and the real shape of the problem

Three parallel investigations (KV-sharing precedent/mechanism, Neuron ordering
guarantees, and an HF-reference correctness audit) converged:

- **Ordering was never the bug.** torch-neuronx functionalization
  (`InPlaceToOutOfPlacePass`) preserves read-after-write for a native `index_put_`
  write and a later read in one compiled graph. Proven empirically: the cache-alias
  and the in-memory-threading variants produce byte-identical output.
- **PLE is correct.** A full op-by-op audit against HF `gemma3n` and upstream vLLM
  `gemma4` (embed-scale `sqrt(256)`, projection-scale `1/sqrt(hidden)`, combine
  `1/sqrt(2)`, tanh-GELU gate, RMSNorm-without-`+1`, per-layer slice, `layer_scalar`)
  found no discrepancy. Exonerated.
- **Double-wide MLP, MQA load, sliding-window** are exonerated: patch C fixes the
  per-layer MLP width + sharding; the E4B config's `num_global_key_value_heads=None`
  resolves KV heads to 1; the sliding mask is `[i-511, i]` correct.
- **head_dim 256/512 attention** is fine on trn2 (the 31B port is coherent 3/3 with
  the same head dims and the same f32 SDPA fallback).

**The uncomfortable finding:** the E4B PLE port this builds on was **never fully
coherent even for E4B** — its own `STATUS.md` records "PLE implemented, k_eq_v fixed,
still garbage," and its README notes "some prompts still produce garbage at TP=2." So
E2B coherence is not a single KV-sharing fix; it is compounded by (a) the base E4B
port's pre-existing quality gap and (b) E2B's MQA (`num_key_value_heads=1`), which —
with 8 query heads and no KV-head diversity — is the config most exposed to any
residual attention-precision or masking imprecision. My KV-share fix moved the failure
from *multilingual salad* to *degenerate English*, i.e. it helped but sits on top of a
base that isn't clean.

**Decisive next step (tooling now identified):** a per-layer hidden-state parity
harness using the vLLM-Neuron staging comparators (`TensorCaptureModel` for per-layer
capture, `setup_reference_model` for the HF-CPU reference, `assert_close_three_way`),
to find the first layer that diverges from HF. That converts this from "guess the bug"
to "read the layer index." It requires making the E2B port importable/runnable in CPU
mode (the model isn't registered in the current staging tree).

**Pragmatic call:** E2B is correctness-only — the native-PyTorch path already meets the
50 RPS target — so this vLLM cell is not on the capacity critical path. The compiles-
and-serves milestone + the exonerations above are the bankable result; full coherence
is a multi-issue debugging project best gated behind the parity harness.

## RESOLVED — the per-layer parity harness found the real bug (it was NOT KV-sharing)

`e2b_parity_harness.py` captures every decoder layer's output from the port (via
vLLM-Neuron's built-in `tensor_capture`) and compares it, cosine per layer, against
an HF-CPU reference — with an **fp32 vs bf16 noise floor** so real divergence is
distinguishable from dtype noise. It runs the whole thing (HF bf16 + HF fp32 + port
capture + compare) in one Kaizen job.

**What it showed (scaling = `1/sqrt(head_dim)`, the shipped port):**
- HF bf16-vs-fp32 noise floor ≈ **1.0000 at every layer** — bf16 is noise-free here,
  so any port cosine < 0.999 is a *real* divergence.
- The port diverged from HF **starting at layer 0** (cos 0.9755) — a **non-shared**
  SWA layer. That alone proves the primary bug was **not KV-sharing**.
- Divergence tracked `head_dim`: worst at the global/full-attention layers
  (layer 11, head_dim 512, cos **0.669**). That fingerprint points at the attention
  **scale**.

**Root cause: the attention scale.** The port used `scaling = 1/sqrt(head_dim)` — a
comment in the source calls it an *inf2 bf16 workaround*. HF Gemma-4 uses
`scaling = 1.0` and lets **q_norm** control logit magnitude, so `1/sqrt(head_dim)`
is an *extra* scale on top of q_norm. On trn2 that is simply wrong. 31B tolerated it
(16 KV heads absorb the flattened softmax); E2B's **MQA (1 KV head)** did not, which
is why E2B degenerated while 31B stayed coherent.

**Fix + confirmation (scaling = `1.0`):**
- Every layer now matches HF at **cos ≈ 0.9999–1.0** (== the noise floor); the
  port's greedy next token becomes **identical to HF's**.
- End-to-end coherence gate: **3/3** — *"The capital of France is **Paris**"*,
  *"...**Jupiter**..."*, *"4"*.

So the working E2B vLLM config is **the KV-share port (donor reuse, from
`fix_e2b_kvshare.py` + `fix_e2b_kvshare_attn.py`) *and* `scaling=1.0`**. The
KV-share work was necessary for the shared layers to match HF; the scale was the
dominant bug that corrupted even the non-shared layers and hid everything else.
`fix_e2b_kvshare_attn.py` now defaults to `scaling=1.0`.

**Lesson:** a per-layer parity harness with a dtype noise floor localizes a bug in a
few runs. Weeks of reasoning pointed at KV-sharing; one aligned measurement pointed
at layer 0 and the head-dim fingerprint, and the one-line scale change fixed it.

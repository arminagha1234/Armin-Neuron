# E2B: KV sharing implemented. Necessary, but not sufficient.

`num_kv_shared_layers` is now implemented in the Public_Final gemma4 port
(prefill path, gated by `GEMMA4_KV_SHARE=1`, default off). It is a genuine
missing-feature fix verified against HuggingFace. It does **not** on its own make
E2B coherent, and neither does stacking it with the double-wide MLP fix.

## Why the port needed this

gemma-4-E2B sets `num_kv_shared_layers=20` over 35 layers, so layers 15-34 must
**reuse** the K/V of the last pre-boundary layer of the same attention type
rather than projecting their own. Before this change the port parsed the field
and never used it (`kv_shared` appeared exactly twice in the whole package, both
in `config.py`), so it computed fresh K/V on all 35 layers — a numerically
different model. 31B is unaffected because it sets `num_kv_shared_layers=0`,
which is why the same code serves 31B correctly at 3/3.

## Implementation

Mirrors `transformers/models/gemma4/modeling_gemma4.py`:

```python
first_kv_shared_layer_idx = num_hidden_layers - num_kv_shared_layers   # 35-20 = 15
is_kv_shared_layer = layer_idx >= first_kv_shared_layer_idx >= 0
prev = layer_types[:first_kv_shared_layer_idx]
store_full_length_kv = (not is_kv_shared_layer and
    layer_idx == len(prev) - 1 - prev[::-1].index(layer_types[layer_idx]))
```

Donors publish K/V **after** qk-norm and RoPE; only K/V are shared and Q stays
per-layer. For E2B this yields donors **13** (sliding) and **14** (full), with
layers 15-34 consuming them. The donor map was verified before compiling.

The substitution sits immediately after RoPE in `forward_prefill`. That point is
efficient in this port because prefill computes local k/v, writes them to the
paged cache, then attends — and both the NKI v2 path and the SDPA fallback
consume the locals. So one substitution covers attention *and* leaves the paged
cache holding the donor's K/V, which keeps the decode step consistent for free.

Donor layers already store full-length K/V here, which HF requires: the sliding
window is applied as an attention-time mask rather than by truncating the cache.

No weight-loader change is needed. The checkpoint carries `k_proj`/`v_proj` for
all 35 layers even though HF ignores them on shared layers, so the fix is simply
to stop using them.

## Result: still incoherent

Every row below was measured on one node with a same-node control, greedy,
1-12 tokens (capital of France, boiling point, mitochondrion, 2+2, largest
planet).

| configuration | score | output character |
|---|---|---|
| neither fix | 0/5 | multilingual token soup |
| KV sharing only | 0/5 | *different* soup — proves the patch engages |
| double-wide MLP only | 0/5 | latin script + markdown artifacts |
| MLP + KV sharing | 0/5 | + `<unused####>` tokens |
| MLP + KV sharing + V2 NKI prefill | 0/5 | byte-identical to previous, so V2 did not engage |

The failure *character* changes at each step, confirming each fix takes effect.
There is at least one further defect.

## What is now ruled out

- **Weight loading** — checkpoint has k/v for all 35 layers; nothing missing.
- **AltUp / LAUREL** — zero tensors for either in the checkpoint.
- **Per-layer embeddings** — implemented in the port, present in the checkpoint
  (108 tensors).
- **MLP width** — now correct: `{6144: 15 layers, 12288: 20 layers}`, matching
  the checkpoint's `gate_proj` shapes.

## Next lead

`<unused####>` are Gemma vocab padding tokens. Emitting them as greedy top-1
points at the LM head, final norm, or embedding scale rather than at attention.
Worth also finding out why `_v2_can_run(q)` declines the V2 prefill kernel,
because the port's own comment says the SDPA fallback for `head_dim>128` only
ever produced "partially coherent English":

```python
# WORKAROUND for inf2: 1/sqrt(d) compensates for bf16 precision in QK-norm +
# attention. Produces partially coherent English. Proper fix requires NKI
# prefill kernel for head_dim>128
self.scaling = 1.0
```

E2B is `head_dim=256`, squarely inside that unsupported regime.

The recommended next step is a **layer-by-layer logit comparison against HF on
CPU**, not more speculative patches. E2B via vLLM-Neuron is an unfinished port
with multiple independent defects; two are now closed and at least one remains.

The only credible E2B throughput number remains the native single-core proxy:
2.77 RPS/chip (9,688 tok/s at 3500 tokens, batch 1, forward pass only). Any
number measured from the incoherent vLLM build is invalid **and inflated**,
because the MLP truncation removed real work.

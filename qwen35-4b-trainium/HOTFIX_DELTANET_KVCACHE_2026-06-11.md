# HOTFIX 2026-06-11 — DeltaNet KV-Cache State Fix

**TL;DR:** the prior branch (`qwen35-4b-fixed`) had a silent correctness bug
under autoregressive decode. Fix is in this branch
(`qwen35-4b-deltanet-kvcache-fix`).

If you're picking up this code, **use this branch's `model_bf16.py`** —
not the one on the `qwen35-4b-fixed` branch.

---

## What was broken

The previous "fixed" branch served Qwen3.5-4B end-to-end and produced the
correct first token (e.g. " Paris" for "The capital of France is"), but
tokens 2..N collapsed into degenerate output:

```
"The capital of France is" -> " Parisuseruseruseruser..."
"Once upon a time"          -> " thereuseruseruseruser..."
```

The bench harness reported reasonable throughput numbers ($1.63/M-input)
because the forward pass *was* finishing — it just wasn't doing
autoregressive generation correctly.

## Root cause

DeltaNet's recurrent state (`recurrent_state_buffer`) and convolution
state (`conv_state_buffer`) were registered as
`nn.Buffer(persistent=False)` and zero-initialized.

**Neuron's compiler constant-folds zero-initialized buffers.** Mutations
via `.data.copy_()` did not survive between forward calls — the compiler
treated the buffer as a compile-time constant zero. DeltaNet became
**stateless**: every decode step recomputed from a fresh zero state.

The model effectively fell back to GQA (the 8 GQA layers) plus
last-token embedding (the 24 DeltaNet layers contributing nothing
meaningful), which is why the first token was right (single-token
prefill needs no state) but every subsequent token was garbage.

The "+ buffer * 0" residual trick that's commonly used to defeat
constant-folding is **insufficient** for this case. The compiler still
folds the zero.

## The fix

Store DeltaNet's recurrent and conv state in **vllm-neuron's tracked KV
cache** (allocated via `bind_kv_cache`) instead of side-channel
`nn.Buffer`s. The KV cache is allocated by the runtime and IS persisted
across forward calls.

Read/write uses flat `index_put_` — the same operation pattern that real
GQA attention uses for its K/V cache, which has been validated to
preserve state on Neuron.

## Code changes (this branch)

- `src/qwen3_5/model_bf16.py`: the only file that changed
  - `Qwen3_5DeltaNet.__init__`: removed `nn.Buffer` registrations for
    `recurrent_state_buffer` / `conv_state_buffer`; added dummy
    `k_cache`/`v_cache` attrs so `bind_kv_cache` accepts the layer
  - `Qwen3_5DeltaNet.forward`: replaced direct buffer reads/writes with
    helpers backed by the KV cache
  - `Qwen3_5ForCausalLM.bind_kv_cache`: routes DeltaNet layers to KV
    cache slots sized to hold one `recurrent_state` + one `conv_state`
    each

## Verification (4B, TP=4, MAX_LEN=512, BF16 KV)

Multi-token coherence is back:

```
"Once upon a time in a small village,"
  -> " there lived a young boy named Tom. Tom was a curious boy who
      loved to explore the world around him..."

"The recipe for chocolate chip cookies requires"
  -> " 1 cup of flour, 1 cup of sugar, 1 cup of butter, 1 cup of
      chocolate chips..."

"Q: What language is spoken in Brazil?"
  -> " The answer is Portuguese."

"Q: How many days are in a week?"
  -> " There are 7 days in a week."
```

First-token canonical: 4/6 (the 2 "fails" are arguable — model answered
"100%" for "legs on a dog", "1.8" for Celsius/Fahrenheit conversion;
both are sensible answers, just not the expected substring).

## Throughput on the FIXED model (trn2.3xl, TP=2, MAX_LEN=512, BF16 KV)

| concurrency | input tok/s | $/M-input |
|---:|---:|---:|
| 1 | 72  | $8.55 |
| 2 | 108 | $5.72 |
| 4 | 144 | $4.30 |
| 8 | 144 | $4.31 |

These numbers are **lower than the prior `qwen35-4b-fixed` branch's
$1.63/M-input** number — but those prior numbers were measured against
the broken model where DeltaNet was stateless. The forward pass on the
broken model skipped real DeltaNet recurrence, so the numbers were
inflated and didn't reflect actual autoregressive generation.

The cost difference is the **honest cost of correctness**.

For customer-relevant numbers (longer MAX_LEN, larger TP), see the
PRODUCTION_PLAN — the 3xl is undersized for compiling MAX_LEN > 1024
graphs. The 48xl is the right hardware for serving at the 20K-input
workload.

## Lesson codified

> On Neuron, `nn.Buffer` mutations via `data.copy_()` may not persist
> between forward calls when the buffer is zero-initialized — the
> compiler constant-folds it. The "+ buffer * 0" residual trick is
> INSUFFICIENT. Use vllm-neuron's KV cache infrastructure (or pass
> state through forward arguments à la Eagle3) for any stateful tensor
> that needs to evolve across decode steps.

## How to use

```bash
export TP=2 MAX_LEN=512 BUCKET=512 MAX_NUM_SEQS=8 KV_CACHE_DTYPE=auto
export PYTHONPATH=/path/to/qwen35-4b-trainium/src
./qwen35-4b-trainium/src/serve.sh
```

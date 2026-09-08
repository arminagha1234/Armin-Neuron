# Qwen3-8B: passes the target at 1 output token, misses it 3.2x at 50

The headline number for this model (4.13 RPS/replica, 66 RPS/box, 131% of a
50 RPS target) is measured at **one output token**. Re-measured with 50 output
tokens under sustained load, it delivers **15.4 RPS/box — 31% of target**.

Decode costs **259 ms/token**, which is 98% of end-to-end wall time and roughly
**47x** the memory-bandwidth floor for an 8B bf16 model at TP=4. This is almost
certainly a defect rather than a hardware limit, and it is not yet root-caused.

## Measurements

`trn2.48xlarge`, TP=4 (4 logical cores = 1 Trainium2 chip at LNC=2), bf16,
`max_model_len=4096`, `max_num_seqs=16`, `max_num_batched_tokens=4096`, prefix
caching off, greedy. Prompt 3,460 tokens. Sustained 45 s load windows, single
server, nothing else on the box.

Coherence 3/3 with `enable_thinking: false` — "The capital of France is Paris.",
"...**Jupiter**...", "2 + 2 equals 4."

### The published shape reproduces

| | published | this run |
|---|---:|---:|
| prefill-only p50 | — | 0.259 s |
| prefill throughput | 13,579 tok/s | **13,355 tok/s** |
| peak RPS/replica (1 out tok) | 4.13 | **4.108** |

Within 1.7% on throughput and 0.5% on RPS. Note the published concurrency
numbers came from one-shot batches (`ok: 1`, `ok: 8` in
`results/raw/vllm-qwen3-8b-bench.json`, i.e. batch size / wall time); this run
used sustained 45 s windows and agrees anyway, which is a useful validation of
both.

### Sustained sweep, 1 output token

| conc | 1 | 4 | 8 | 16 |
|---|---:|---:|---:|---:|
| RPS | 3.907 | 4.107 | **4.108** | 4.106 |

Flat from concurrency 4 on, zero errors. Confirms the existing finding that a
3.5K prefill already saturates the engines, so co-scheduling more prefills adds
nothing. **65.7 RPS/box = 131% of target.**

### Sustained sweep, 50 output tokens

| conc | 1 | 4 | 8 | 16 |
|---|---:|---:|---:|---:|
| RPS | 0.077 | 0.292 | 0.545 | **0.963** |

Still climbing at 16, zero errors. **15.4 RPS/box = 31% of target.** Unlike
prefill, decode does scale with concurrency (0.077 x 16 = 1.23 vs 0.963 measured,
~78% scaling efficiency), so the decode path batches correctly — each token is
just extraordinarily expensive.

```
e2e p50 (50 tok)  12.957 s
prefill p50        0.259 s
=> decode          259.15 ms/token   (98% of e2e)
```

## Why 259 ms/token is anomalous

- **Bandwidth floor:** 8B bf16 is ~16 GB of weights. At TP=4 on one chip
  (~2.9 TB/s HBM) a memory-bound decode step should floor around **5.5 ms**.
  We are ~47x above that.
- **Against a smaller sibling on the same box:** Qwen3.5-4B decodes at
  **15.6 ms/token**. Qwen3-8B has 2x the parameters and is **16.6x slower per
  decode token**. Parameter count does not explain it.
- **Suspicious coincidence:** prefill of 3,460 tokens takes 0.259 s, and one
  decode token takes 0.25915 s. Identical to three digits. That is the signature
  of decode re-executing prefill-shaped work every step.

## Hypothesis tested and FALSIFIED: token bucketing

The obvious reading of that coincidence is that the config declares a single
`num_batched_tokens_buckets: [4096]`, so a 1-token decode step has no small
bucket to land in and pads to 4096. Tested by adding a small bucket:

```
--additional-config '{"neuron_config":{"num_batched_tokens_buckets":[16,4096], ...}}'
```

| | baseline `[4096]` | fix `[16,4096]` |
|---|---:|---:|
| NEFFs compiled | 2 | **3** |
| prefill p50 | 0.259 s | 0.257 s |
| decode | 259.15 ms/tok | **258.57 ms/tok** |
| speedup | — | **1.00x** |
| peak RPS/chip (50 out) | 0.963 | 0.954 |

The bucket list was accepted and the graph genuinely changed (an extra NEFF
compiled), and decode did not move at all. **Token-bucket padding is not the
cause.** Recording this so the obvious fix is not retried.

## Remaining hypotheses, in order

1. **Decode re-runs the full-context forward.** The exact prefill-time match
   points here. Diagnostic: watch the engine's own metrics during a pure decode
   phase — if `Avg prompt throughput` stays non-zero while only generating, the
   prefill path is executing per token. Cheap and decisive.
2. **mRoPE per-token cost.** Because the port derives from `qwen3_vl`,
   `mk_qwen3.py` synthesises `mrope_section` and sets `mrope_interleaved: True`
   on a dense text model purely to satisfy an assert. If the rotary path expands
   or recomputes over the full position range each step rather than the single
   new position, that is an O(context) per-token cost. `mk_qwen3.py` itself
   carries the comment "the rotary module expands 1D [T] internally", which is
   worth reading closely.
3. **KV cache not being reused**, i.e. re-encoding the prefix. Would present
   identically to (1).
4. **No dedicated decode graph for this architecture** — the derived model may
   only implement a fused full-sequence forward, with `qwen3_vl`'s decode path
   inactive.

Next step is diagnostic (1), then inspect whether a small decode NEFF exists and
is actually dispatched.

## Consequence for the capacity study

Qwen3-8B was the only one of the four models meeting its Indeed target. At a
realistic output length it does not:

| Model | target | RPS/box | % of target |
|---|---:|---:|---:|
| Qwen3-8B @ 1 out tok | 50 | 65.7 | 131% |
| **Qwen3-8B @ 50 out tok** | 50 | **15.4** | **31%** |
| Qwen3.5-4B @ 50 out tok | 500 | 12.4 | 2.5% |
| Gemma-4-31B @ 50 out tok | 50 | 7.66 | 15% |
| Gemma-4-E2B | 50 | — | incoherent |

**At 50 output tokens, none of the four models currently meets its target.** Any
capacity plan built on the 1-output-token figures is optimistic by ~4x for
Qwen3-8B, and the E2B row (also published at 1 output token) should be assumed to
carry the same problem until re-measured.

The upside: a 259 ms/token decode that should floor near 5.5 ms is a large,
well-localised defect rather than a tuning gap. If it is (1) or (2), the fix is
likely worth most of the missing 3.2x on its own.

## Reproducing

Weights are cached on FSX at `$FSX/models/Qwen3-8B` (16 GB, pulled in 37 s).

```bash
python3 mk_qwen3.py            # 14 asserted edits, registers Qwen3ForCausalLM
vllm serve $FSX/models/Qwen3-8B --served-model-name q38b \
  --tensor-parallel-size 4 --max-model-len 4096 --max-num-seqs 16 \
  --max-num-batched-tokens 4096 --no-enable-prefix-caching \
  --additional-config '{"neuron_config":{"num_batched_tokens_buckets":[4096],
    "num_seqs_buckets":[16],"on_device_sampling_config":{"all_greedy":true}}}'
```

Ready in ~200 s from a cold cache. Measure with `max_tokens=min_tokens=50` and
`ignore_eos: true` so the output length is exact, and use sustained load windows
rather than one-shot batches.

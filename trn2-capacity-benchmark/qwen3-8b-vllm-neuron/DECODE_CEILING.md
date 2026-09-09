# Qwen3-8B: passes the target at 1 output token, misses it 3.2x at 50

The headline number for this model (4.13 RPS/replica, 66 RPS/box, 131% of a
50 RPS target) is measured at **one output token**. Re-measured with 50 output
tokens under sustained load, it delivers **15.4 RPS/box — 31% of target**.

Decode costs **259 ms/token** — 98% of end-to-end wall time and ~47x the
memory-bandwidth floor for an 8B bf16 model at TP=4.

**Root cause: the decode NEFF executes the full static `max_num_seqs` x context
shape on every step**, regardless of how many sequences are actually active. The
published run paid a 16-slot decode step to serve one sequence. Setting
`max_num_seqs=1` drops decode to **11.27 ms/token (23x)**, and sweeping the knob
for throughput lands on **`max_num_seqs=8`: 23.0 RPS/box, 1.49x the published
config from one flag**. That is still only 46% of target, so the model needs
~2.2 boxes rather than the ~0.8 implied by the 1-output-token number.

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

## ROOT CAUSE: decode executes the full static `max_num_seqs` x context shape every step

Neuron requires static shapes, so the decode NEFF is compiled for the worst case
and runs **all** sequence slots and the **whole** context window on every step,
regardless of how many sequences are actually active or how long the real context
is. Two isolating measurements:

| config | `max_num_seqs` | `max_model_len` | decode ms/token | vs baseline |
|---|---:|---:|---:|---:|
| baseline | 16 | 4096 | 259.15 | 1.00x |
| shrink the seq bucket | **1** | 4096 | **11.27** | **23.0x** |
| shrink the context 4x | 16 | 1024 | 49.81 | 5.20x |

Decode cost is proportional to `max_num_seqs x max_model_len`. At
`max_num_seqs=1` decode is 11.27 ms/token — within 2x of the ~5.5 ms bandwidth
floor, i.e. finally sane. The published run paid the full 16-slot cost while
serving **one** sequence.

This also explains why the defect was invisible before: at 1 output token you pay
one decode step and prefill dominates, so a 16x-oversized decode step never shows
up.

### `max_num_seqs` is an unswept throughput knob

Smaller buckets are cheaper per step but batch less, so there is an optimum.
Sustained 40 s windows, 50 output tokens, 3,460-token prompt:

| `max_num_seqs` | decode ms/tok | ms/tok **per seq** | peak RPS/chip | RPS/box | % of 50 RPS target |
|---:|---:|---:|---:|---:|---:|
| 1 | 11.27 | 11.27 | 1.252 | 20.0 | 40% |
| 2 | 24.47 | 12.24 | 1.188 | 19.0 | 38% |
| 4 | 40.48 | 10.12 | 1.353 | 21.6 | 43% |
| **8** | **72.31** | **9.04** | **1.438** | **23.0** | **46%** |
| 16 *(published)* | 259.15 | 16.20 | 0.963 | 15.4 | 31% |

**`max_num_seqs=8` is optimal and gives 1.49x the published configuration**
(0.963 -> 1.438 RPS/chip) from a single flag, no code change. Boxes for 50 RPS:
3.2 -> **2.2**.

There is a **superlinear cliff between 8 and 16**: doubling the bucket multiplies
decode by 3.58x, and per-sequence efficiency degrades from 9.04 to 16.20 ms/token
having improved monotonically up to that point. Something changes qualitatively at
16 — plausibly an SBUF/tiling capacity threshold that pushes the decode working
set into spilling. Worth a profile at MNS=8 vs 16 to confirm, since the cliff is
where the remaining easy factor of ~1.8x on per-sequence efficiency lives.

### Why the profile looked the way it did

The 82% GpSimd / 98.6% dynamic-DMA / 15M-packet picture is the *symptom* of
executing a 16x-oversized decode step, not an independent bug: the graph moves
21.6 GB of HBM per token against a ~4 GB weight shard because it is scanning 16
sequence slots x 4096 positions x 36 layers every time.

## Hypotheses tested and falsified

Three plausible causes were measured and ruled out. All three left decode within
3% of baseline, so none of them should be retried:

| hypothesis | test | result |
|---|---|---|
| token-bucket padding | `num_batched_tokens_buckets [16,4096]` | 258.57 ms/tok — **1.00x** |
| oversized KV block pool | `--num-gpu-blocks-override 512` (auto-sized was 192,928 tokens = 47x one request) | 258.42 ms/tok — **1.00x** |
| per-layer KV `index_put_` scatter | disabled the scatter entirely (breaks correctness) | 250.73 ms/tok — **1.03x**, so the scatter is only **3%** |

The KV-scatter result is worth keeping in mind: `NF.attention_decode`'s own
`update_cache=True` path performs the same `index_put_` pattern internally
(`attention_decode.py` ~797-800), so moving the scatter into the kernel would not
have helped either.

## Does it generalize? Tested on Qwen3.5-4B: NO

The obvious inference is that every model on this stack should have `max_num_seqs`
swept. We tested that on Qwen3.5-4B — the model that dominates the fleet estimate
— and **it does not transfer**. Same box, same method, blocked inverse + NKI
decode, `LEN=2048`, blocks=200, 1,811-token prompt, 50 output tokens:

| `max_num_seqs` | decode ms/tok | peak RPS/chip | RPS/box | boxes for 500 RPS |
|---:|---:|---:|---:|---:|
| 4 | 23.02 | **0.779** | 12.5 | 40 |
| 8 | 34.17 | 0.679 | 10.9 | 46 |
| 16 *(as published)* | **15.60** | 0.775 | 12.4 | 40 |

Two differences from Qwen3-8B:

1. **Decode does not scale with the bucket** — it is *best* at MNS=16 (15.60 ms)
   and worse at 4 and 8, non-monotonically. Qwen3.5 uses a different decode path
   (the recurrent DeltaNet step plus the `head_dim=256` kernel) than Qwen3-8B's
   `NF.attention_decode`, so the static-shape cost model does not apply.
2. **Throughput is flat regardless** — 0.779 vs 0.775 RPS/chip between the best
   and the published setting, inside noise. Because Qwen3.5 is **prefill-bound**
   (0.906 s prefill out of ~1.28 s of service time), even a 1.5x decode regression
   does not move the peak. That is the same conclusion the transposed-state
   experiment reached from the other direction.

So the correct statement is narrower: **`max_num_seqs` matters when decode is on
the critical path, and is worth checking rather than assumed.** For Qwen3-8B at
50 output tokens decode is 98% of e2e and the knob is worth 1.49x. For Qwen3.5 it
is worth nothing.

**Gemma-4-31B remains untested.** It was measured at `max_num_seqs=32` with 50
output tokens, and the `gemma4-31b/` README independently found MNS=16 optimal
with 32 regressing — but on a different shape (in=1024/out=256, TP=32), so it is
not evidence for our configuration. Whether 7.66 RPS/box is understated is an open
question, not a claim.

## Superseded hypotheses (kept for the record)

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
| Qwen3-8B @ 50 out tok, `MNS=16` (published cfg) | 50 | 15.4 | 31% |
| **Qwen3-8B @ 50 out tok, `MNS=8` (tuned)** | 50 | **23.0** | **46%** |
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
# max_num_seqs=8 is the throughput optimum -- NOT 16. See the sweep above.
vllm serve $FSX/models/Qwen3-8B --served-model-name q38b \
  --tensor-parallel-size 4 --max-model-len 4096 --max-num-seqs 8 \
  --max-num-batched-tokens 4096 --no-enable-prefix-caching \
  --additional-config '{"neuron_config":{"num_batched_tokens_buckets":[4096],
    "num_seqs_buckets":[8],"on_device_sampling_config":{"all_greedy":true}}}'
```

Keep `num_seqs_buckets` in step with `--max-num-seqs`; the bucket is what the
decode graph is compiled against, and it is what costs you.

Ready in ~200 s from a cold cache. Measure with `max_tokens=min_tokens=50` and
`ignore_eos: true` so the output length is exact, and use sustained load windows
rather than one-shot batches.

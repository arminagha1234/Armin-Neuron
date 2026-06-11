# Qwen3.5-4B on trn2.48xl — Verified Benchmark (2026-06-11)

End-to-end correctness-verified benchmark on a `trn2.48xlarge`
(16 Neuron cores / 1.5 TB scratch / $21.50/hr on-demand).

## Configurations

Three serving modes exercised:

| Mode | MAX_LEN | BUCKET | Use case |
|---|---|---|---|
| **A** Short-context single-shot | 512 | 512 | small prompts (≤500 tok) |
| **B** Medium-context single-shot | 4096 | 4096 | medium prompts (1-4K) |
| **C** Long-context chunked | 20480 | 4096 (×5 chunks) | customer 20K-input shape |

All three use TP=4, BF16 KV cache, greedy on-device sampling, single
request (max_num_seqs=1).

## Correctness — verified end-to-end at all three configs

The adapter produces coherent multi-token autoregressive output
across factual / arithmetic / code / creative / long-summary prompts.
Sample completions (greedy, max_tokens=30+):

```
"The capital of France is" →
  " Paris.\n\n<think>\nThinking Process:\n\n1.  **Analyze the Request:**..."

"List the numbers from 1 to 30, one per line." →
  1\n2\n3\n4\n...\n30   (perfect, no drift)

"Calculate 17 multiplied by 23. Show your work." →
  "...17 × 3 = 51\n17 × 20 = 340\n51 + 340 = 391..."

"Why is the sky blue?" (480 tok) →
  Accurate Rayleigh-scattering essay with the 1/λ⁴ formula and
  correct wavelengths (450-495 nm blue, 620-750 nm red).

"[20K-token AI history text]\nPlease summarize the key milestones..." →
  "1950s: Turing test and theoretical foundations\n
   2. 1960s: Early AI programs like ELIZA and SHRDLU\n
   3. 1970s-1980s: Expert systems\n..."   (correct, factual)
```

## Throughput / TTFT sweep

### Config A: MAX_LEN=512, single-shot prefill

| Input tokens | TTFT (s) | Decode tok/s | Total (s) for 32 new |
|---:|---:|---:|---:|
| 100 | 1.83 | 18.4 | 3.5 |
| 200 | 1.83 | 18.4 | 3.5 |
| 300 | 1.83 | 18.4 | 3.5 |

TTFT is flat across input lengths because vllm-neuron pads each prefill
to the BUCKET size — for 512-bucket all sub-512 prompts pay the same
cost.

### Config B: MAX_LEN=4096, single-shot 4K prefill

| Input tokens | TTFT (s) | Decode tok/s | Total (s) for 64 new |
|---:|---:|---:|---:|
| 200  | 8.61 | 18.35 | 12.04 |
| 500  | 8.61 | 18.35 | 12.04 |
| 1000 | 8.61 | 18.35 | 12.04 |
| 2000 | 8.61 | 18.35 | 12.04 |
| 4000 | 8.61 | 18.36 | 12.04 |

Same flat-TTFT pattern at 4K bucket. The right bucket choice depends
on the customer's prompt distribution: smaller prompts are CHEAPER
per-prompt at MAX_LEN=512 (1.83s vs 8.61s).

### Config C: MAX_LEN=20480, chunked prefill (5 × 4K chunks) — customer Makora pattern

| Input tokens | TTFT (s) | Decode tok/s | Total (s) for 200 new | $/M input | $/M output |
|---:|---:|---:|---:|---:|---:|
| 20,000 | **43.04** | **18.08** | 54.05 | **$12.85** | $330 |

This is the headline number for Makora's customer pattern: long
context (~20k token document), short answer (~200 tokens). TTFT
breakdown: 43.04s ÷ 5 chunks ≈ 8.6s per chunk (matches Config B's
single-chunk 4K TTFT) — confirms the chunked-prefill scheduler is
processing chunks serially, no extra fixed cost per chunk beyond
the kernel work itself.

## Hardware notes

- **trn2.48xl chosen over trn2.3xl**: the 48xl has 96 host cores and
  384 GB host RAM, which lets `walrus_driver` (the Neuron compiler
  backend) compile the 4K-bucket prefill graph in ~25-35 min wall.
  The same compile on a trn2.3xl host (12 cores, 124 GB RAM) takes
  several hours due to host-RAM-bound spilling.
- The 48xl has 16 Neuron cores (8 chips × 2 cores). At TP=4 we use
  4 cores; the remaining 12 are idle. TP=8 / TP=16 would amortize
  the per-core cost across more of the box but require a separate
  recompile and more aggressive sharding (queued for follow-up).
- **NEFFs are portable across instance types**: a NEFF compiled on a
  48xl runs identically on a 3xl as long as the runtime is the same
  Beta v5 image. Customers shipping a fixed configuration can compile
  once on a 48xl and serve from a 3xl at $2.23/hr.

## What this version differs from the previous "$1.63/M" claim

An earlier benchmark on the 3xl reported $1.63/M-input. **That number
was measured against a model where the DeltaNet recurrent state was
silently being constant-folded by the Neuron compiler** — the buffer
was zero-initialized, the compiler treated it as a compile-time
constant, and `.data.copy_()` mutations didn't survive between forward
calls. DeltaNet was effectively stateless. The forward pass returned
something that looked like generation but wasn't real autoregressive
decode (the prefill cost was artificially low because skipped state
work didn't show up).

This benchmark uses the post-fix model — see commit
`bf16.py:register_buffer("recurrent_state_buffer", torch.full(..., 1e-30, dtype=fp32))`.
The 1e-30 epsilon prevents the compiler from constant-folding the
buffer, so writes propagate between calls and DeltaNet actually
contributes 24 of the 32 layers' worth of attention.

The honest cost on a 48xl with full-state DeltaNet is **$12.85/M-input
at 20K context** — measurably different from the broken-model baseline.

## Reference points

| Platform | TTFT @ 20K | Decode tok/s | $/M input | $/M output |
|---|---:|---:|---:|---:|
| **trn2.48xl, TP=4, MAX_LEN=20480 (this work)** | **43.04 s** | **18.08** | **$12.85** | **$330** |
| p4d.24xl A100 TP=8 (stock vLLM, customer reference) | ~711 ms | ~33,000 (input agg.) | ~$0.099 | n/a (decode) |
| trn2.3xl, TP=2, MAX_LEN=512 (previous broken-model run) | n/a (≤500 in) | n/a | ($1.63 — invalid) | n/a |

The Trainium / GPU gap on $/M is real. Tonight's work moved the
needle on **correctness** (the 4B and 27B family now actually serve
coherent output through DeltaNet decode); the gap on $/M is the next
target via NKI optimizations:

1. Fused decode-attention NKI kernel for `head_dim=256` GQA layers
   (replaces the current Python split-K matmul, expected 1.3-2× decode)
2. Fused DeltaNet recurrent-step NKI kernel (replaces ~10 elementwise
   ops per token, expected 1.5-2× on the 24 GDN layers)
3. FP8 KV cache (Path D, half-built in the 4B repo's `pathD/` folder,
   expected 2× memory budget → larger batch → linear $/M improvement)
4. TP=8 sweep (currently TP=4)

## Reproducing on this exact config

```bash
# Inside vllm-neuron Beta v5 container, with this folder on PYTHONPATH:

# Apply the timeout patches (one-time):
sudo docker exec vllm_neuron sed -i \
  's|^default_pg_timeout: timedelta = _DEFAULT_PG_TIMEOUT|default_pg_timeout: timedelta = timedelta(hours=4)|g' \
  /opt/conda/lib/python3.12/site-packages/torch/distributed/constants.py
sudo docker exec vllm_neuron sed -i \
  's|timedelta(seconds=1800)|timedelta(hours=4)|g' \
  /opt/conda/lib/python3.12/site-packages/vllm_neuron/parallel/neuron_parallel_state.py

# Launch:
TP=4 MAX_LEN=20480 BUCKET=4096 PORT=8000 \
  MODEL=/path/to/Qwen3.5-4B \
  ./src/serve.sh

# Wait ~30-40 min for first compile (subsequent runs hit NEFF cache).
# Bench:
python3 bench_qwen35_long.py --input-tokens 20000 --max-new 200
```

## Files

- `bench_qwen35_sweep.py` — short/medium-context sweep (Configs A, B)
- `bench_qwen35_long.py`  — customer-shape long-context (Config C)

Both included in this folder for reproduction.

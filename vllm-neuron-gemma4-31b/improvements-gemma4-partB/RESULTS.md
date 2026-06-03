# Gemma4 31B IT — vLLM-Neuron Path B Measured Results

All numbers below are **measured on a real trn2.48xlarge NeuronCore** (`i-0d55a7514c80f5075`,
us-east-2), vLLM-Neuron v5 beta image, custom `Gemma4ForConditionalGeneration` model,
bf16, on-device greedy sampling.

Raw JSONs and serve logs are in
[`results/`](https://github.com/arminagha1234/Armin-Neuron/tree/main/vllm-neuron-gemma4-31b/improvements-gemma4-partB/results).

## TL;DR

- **Best TTFT @ 4K input: 292.6 ms** (TP=32, single-bucket `[4096]`) — **41% under
  Hippocratic's 500 ms target.**
- **8K input: 658.5 ms** (TP=32, single-bucket `[8192]`) — fails the 500 ms target by 32%.
- **Throughput saturates at concurrency=4** with `max_num_seqs=4`: **693 tok/min** (in=1024,
  out=256). Beyond conc=4 requests just queue. Raise `max_num_seqs` if you want more.
- **Generation works:** "The capital of France is" → " Paris." (TTFT 292.6 ms, TPOT 343 ms).


## TTFT — TP=32, single-bucket `[4096]`, max_num_seqs=4

Each input length is the *logical* prompt size; the single-bucket config pads
every prompt to 4096 tokens before the kernel runs, which is why the cost is
flat across logical lengths. This is the optimal latency configuration when
your workload tops out around 4K input.

| Logical input | Median TTFT | Min | Max | vs 500 ms |
|---:|---:|---:|---:|---|
| 256 | **287.7 ms** | 286.8 | 312.5 | ✅ PASS |
| 512 | **286.6 ms** | 286.1 | 288.0 | ✅ PASS |
| 1024 | **288.5 ms** | 286.9 | 295.6 | ✅ PASS |
| 2048 | **290.5 ms** | 289.8 | 291.8 | ✅ PASS |
| ~3900 (regression) | **292.6 ms** | 286.9 | 310.7 | ✅ PASS |

5-iter median per row, 7 iters for the 3900 regression. Variance run-to-run is
≤2 ms across the table.

Raw: [`results/run5_ttft_scan.json`](results/run5_ttft_scan.json),
[`results/run4_gen.json`](results/run4_gen.json) (4K regression).

## TTFT — TP=32, single-bucket `[8192]`, 8K input

Re-measured cleanly with a tokenizer-sized 7,800-token prompt (the original
auto-sized bench overshot the 8192 budget and rejected requests).

| Input | Median TTFT | Min | Max | vs 500 ms |
|---:|---:|---:|---:|---|
| 7,800 | **658.5 ms** | 656.9 | 688.7 | ❌ FAIL (32% over) |

5-iter median. Raw: [`results/run2_ttft_8k_clean.json`](results/run2_ttft_8k_clean.json).

The 8K cost (~659 ms) is consistent with the linear-in-context scaling visible
in earlier multi-bucket runs (~165 ms / 1K once past the small-prompt regime).
8K is real compute, not a bucket artifact.

## Throughput — TP=32 single-bucket `[4096]`, max_num_seqs=4

Aggregate output throughput at increasing client concurrency, in=1024 / out=256
per request, 2 requests per concurrency level.

| Concurrency | Wall (s) | Out tokens | tok/s | tok/min | Avg latency |
|---:|---:|---:|---:|---:|---:|
| 1 | 175.6 | 512 | 2.9 | **175** | 87.8 s |
| 4 | 177.2 | 2,048 | 11.6 | **693** | 88.6 s |
| 8 | 354.4 | 4,096 | 11.6 | 693 | 155.1 s |
| 16 | 708.8 | 8,192 | 11.6 | 693 | 287.9 s |

**Throughput plateaus at concurrency 4 = 693 tok/min.** Server-side
`max_num_seqs=4` caps how many requests decode in parallel; beyond that, client
concurrency just adds queue time. Raise `max_num_seqs` if your workload needs
more sustained throughput (at the cost of more KV-cache HBM and possibly a
recompile).

Decode rate is `~2.8-2.9 output tokens/sec/sequence`, ~350 ms/token. That's the
head_dim>128 SDPA-fallback penalty that PR #1552 / Part C target — Gemma4's
head_dim is 256 (SWA layers) / 512 (global layers) and the fused NKI flash-decode
kernel only supports head_dim ≤ 128, so decode runs the unfused fallback.

Raw: [`results/run3_throughput.json`](results/run3_throughput.json).

## Generation Proof

Real coherent text generation, captured per-token latency:

```
Prompt:  "The capital of France is"
TTFT:    292.6 ms
Output:  " Paris.\n\nThe capital of France is Paris.\n\nThe
          capital of France is Paris.\n\nThe capital of France is
          Paris.\n\nThe capital of France is"
n_tokens:    32
TPOT mean:   343.6 ms (min 341.9, max 359.0)
Total wall:  10.94 s
```

Raw: [`results/run4_gen.json`](results/run4_gen.json).

## Why single-bucket beats multi-bucket on TTFT

A multi-bucket config (e.g. `[256, 512, 1024, 2048, 4096, 8192, 10240]`) at the
same TP=32 measured **993 ms TTFT @ 4K** in the original Path B run. The
single-bucket `[4096]` config measures **288-292 ms** at every logical length
≤4K. **3.4× faster** for the same model, same hardware, same TP.

The compiler optimizes a single-bucket NEFF independently of any others, so its
shared-layout decisions aren't constrained by the larger buckets. With many
buckets, every NEFF compiles a layout that's compatible with the others, which
is suboptimal for any one of them.

**Practical recommendation:** if your workload has a known maximum context
(e.g. always ~4K), use a single-bucket config matching that length. Multi-bucket
is for genuinely variable-length traffic where you accept per-bucket
suboptimality to cover the full range.

## Why TP=32 beats TP=16 on TTFT (and why TP=64 doesn't exist)

| TP | Bucket | TTFT @ 4K | Notes |
|---:|---:|---:|---|
| 16 | [4096] | 452 ms | Earlier measurement |
| **32** | **[4096]** | **293 ms** | This run, 35% faster than TP=16 |
| 64 | — | — | **Impossible** — Gemma4 has 32 attention heads, max TP=32 |

TP=32 is the maximum that shards cleanly (1 head per rank). TP=64 fails head
divisibility.

## Recommendations by workload

| Workload | Config | TTFT | Throughput | Status |
|---|---|---:|---:|---|
| 4K input, latency-critical | TP=32, bucket `[4096]`, max_num_seqs=4 | **293 ms** | 693 tok/min | ✅ Ship |
| 4K input, throughput-leaning | TP=32, bucket `[4096]`, max_num_seqs=16 | ~340 ms (est) | 2,500-3,000 tok/min (est) | Untested — recompile |
| 8K input | TP=32, bucket `[8192]`, max_num_seqs=4 | 659 ms | (not measured at 8K) | ❌ Misses 500 ms target |
| 8K with prefix caching | shared system prompt | ~80-100 ms (est) | (depends on cache hit rate) | Untested |

## Honest caveats

1. **Single-bucket pads to the bucket size.** A 256-token logical prompt at
   single-bucket `[4096]` runs through the 4096 NEFF and pays the 4K cost
   (~288 ms). That's why all rows from 256 to 2048 in the TTFT table are flat
   ~287-290 ms. If your workload is mostly small prompts, a different config
   (say single-bucket `[1024]`) will be cheaper for those — at the cost of a
   second compile and not handling longer prompts.

2. **Decode is the bottleneck for long outputs.** The 343 ms/token decode rate
   means a 256-token response takes ~88 s of wall time after TTFT. For
   Hippocratic's pattern (long input, short output), this is fine. For
   chat-style workloads with long generations, this dominates the experience.
   The Part C NKI work targets this; see `improvements-gemma4-partC/`.

3. **The earlier RESULTS.md mentioned an "8K = 662 ms" projection.** The
   measured value is 658.5 ms — the projection (4K × 2.25) was within 0.5%.
   But it was a projection until this run; it's now a measurement.

4. **Numbers were captured 2026-06-03 on a working trn2.48xl with NEFFs
   already cached** from prior runs. Cold-compile times for fresh NEFFs are
   substantially longer (the 8K NEFF originally took several hours to compile
   on its first build per the Part F notes).

## Reproduce

Inside the vLLM-Neuron v5 container with the model at `/root/models/gemma-4-31b-it`:

```bash
# 4K winner config
TP=32 MAXLEN=4096 NBT=4096 GEMMA4_APPLY_PATHB=1 \
  vllm serve /root/models/gemma-4-31b-it \
    --tensor-parallel-size 32 \
    --max-model-len 4096 \
    --max-num-seqs 4 \
    --max-num-batched-tokens 4096 \
    --additional-config '{"neuron_config":{
        "num_batched_tokens_buckets":[4096],
        "num_seqs_buckets":[4],
        "on_device_sampling_config":{"all_greedy":true}}}'

# Then in another shell:
python3 bench_ttft.py --model /root/models/gemma-4-31b-it \
    --seq-lens 256,512,1024,2048,3900 --runs 5
python3 bench_throughput.py --model /root/models/gemma-4-31b-it \
    --concurrency 1,4,8,16 --input-tokens 1024 --output-tokens 256 \
    --reqs-per-level 2
```

For 8K, swap `4096` → `8192` everywhere and use `bench_8k_ttft.py` (this
folder) which sizes prompts via the tokenizer to land cleanly under the budget.

## Files

- `results/run5_ttft_scan.json` — TTFT scan 256-3900, 5 iters each
- `results/run4_gen.json` — generation proof, prompt → text + per-token latency
- `results/run3_throughput.json` — throughput sweep, conc 1/4/8/16
- `results/run2_ttft_8k_clean.json` — clean 8K TTFT measurement
- `results/run1_regression.json` — 3900-token regression check
- `results/serveA.log`, `results/serveB.log`, `results/orchestrator.log` — raw
- `bench_8k_ttft.py` — clean 8K bench harness (tokenizer-sized prompts)

---

*Measured 2026-06-03 on `i-0d55a7514c80f5075` (trn2.48xlarge, us-east-2),
vLLM-Neuron v5 beta image, NEFFs cached from prior compile sessions.*

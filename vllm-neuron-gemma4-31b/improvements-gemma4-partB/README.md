# Gemma4 31B — Part B: Throughput & Max Tokens/Min

**Goal:** maximize **tokens/minute** and **max sustained throughput** for
Gemma4 31B IT on trn2.48xlarge — the metrics the customer cares about (Part A
covered single-request TTFT).

Part A result (for reference): **292 ms TTFT @ 4K, TP=32, single bucket.**

## What's different in Part B

Part A optimized **latency** (one request, time-to-first-token). Part B optimizes
**throughput** (many requests, total tokens/min). Different knobs:

| Lever | Part A (latency) | Part B (throughput) |
|---|---|---|
| Batch size / concurrency | 1 (lowest latency) | **high** (saturate the cores) |
| TP | 32 (max sharding) | 16 vs 32 — measure both (more cores ≠ more aggregate tput) |
| Metric | TTFT (ms) | tokens/s, tokens/min, max batch before OOM |
| Bottleneck | prefill compute | **decode** (~2.8 tok/s/seq at batch=1) |

## Key facts pulled from PR #1552 (`feat/gemma4-model`)

PR #1552's `experiments/gemma4_optimizations/OPTIMIZATION_TRACKER.md` measured
(TP=16, batch=1):

| Scenario | Throughput | Notes |
|---|---:|---|
| Prefill s4096 | **7,795 tok/s** | NF.mlp kernel active (peak prefill) |
| Prefill s2048 | 7,355 tok/s | |
| Prefill s8192 | 6,688 tok/s | falls off past 4K |
| Decode s1024+o256 | 14.3 tok/s total | ~2.8 output tok/s/seq |
| Decode s4096+o256 | 47.8 tok/s total | ~2.8 output tok/s/seq |

**Decode bottleneck root cause (from #1552):** Gemma4 head_dim=256 (SWA) / 512
(Global) exceeds the NKI decode megakernel limit of 128, so decode falls back to
PyTorch SDPA. There is **no NKI decode kernel for head_dim>128 today**, so the
only throughput lever on decode is **batching** — run many sequences concurrently
so the per-step cost amortizes across the batch.

**Two #1552 optimizations already in the model code:**
- Opt-1: fuse `pre_feedforward_layernorm` into the NKI MLP (`NormType.RMS_NORM` + `ln_w`) — saves 1 kernel launch + 1 HBM round-trip per layer × 60 layers
- Opt-2: GQA KV replication via `expand().reshape()` instead of `repeat_interleave()` — avoids a memory copy

Both are in the `gemma4/model.py` we serve, so Part B inherits them.

## Strategy for max tokens/min

1. **Batch sweep** — the single biggest lever. Decode at batch=1 is ~2.8 tok/s/seq;
   at batch=N it's ~N× aggregate until the cores saturate. Sweep
   `max-num-seqs` = 1, 4, 8, 16, 32, 64 and find the max before OOM.
2. **TP comparison** — TP=16 frees cores for more KV cache (bigger batch) vs TP=32
   (faster per-seq but less KV headroom). Measure aggregate tok/min for both.
3. **Use the standard `vllm bench throughput` tool** (same as #1552) for
   apples-to-apples numbers, plus our concurrent-client bench for serving-style
   tokens/min.
4. **Output-length realism** — report tokens/min at a realistic output length
   (e.g. 256 tok) since decode dominates long generations.

## Files

- `bench_throughput.py` — concurrent-client throughput sweep (batch 1→max), reports tokens/min
- `run_vllm_bench.sh` — wraps the standard `vllm bench throughput` (matches #1552 methodology)
- `RESULTS.md` — measured numbers (filled in as we run)

## Sources pulled

- Part A example: `../` (this repo)
- PR #1552: `aws-neuron/private-vllm-neuron#1552` (`feat/gemma4-model`) —
  optimization tracker + decode/prefill baselines + `vllm bench throughput` method
- NxDI sweep: `customers/Hippocratic/gemma4_31b_it_results/`

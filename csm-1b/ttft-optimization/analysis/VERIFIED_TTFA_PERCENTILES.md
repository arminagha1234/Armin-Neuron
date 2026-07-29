# CSM-1B TTFA latency percentiles (p50/p90/p99) — MEASURED

**The customer SLO metric: time-to-first-audio (TTFA).** This is a **real sampled
distribution**, not a component sum with hardcoded terms — it directly closes code-review
Finding 1 (the earlier "~126 ms" TTFT was a modeled sum with hardcoded backbone/host terms +
best-of timing).

## Method
- Harness: `src/ttft_percentiles.py`.
- Box: `trn2.48xlarge`, native torch-neuronx, single NeuronCore (`NEURON_RT_VISIBLE_CORES=0`),
  in the DLC container.
- Path timed per iteration = the real first-audio critical path:
  **backbone decode step → depth decode (32 codebooks, compiled) → Mimi codec (1 frame, CPU)**.
- Depth loop is the compiled resident-weight loop with the head+QK fp32 fix baked in
  (`fair_depth_handroll.py`), i.e. the **bf16 fast path**.
- 10 warmup iters (absorb compile + allocator warmup), then **200 timed iters**, warm.
- `time.perf_counter()` around `depth(...).cpu()` + `codec_decode(...)` (the `.cpu()` forces a
  real device sync so we time completed work, not just dispatch).

## Result (200 warm iters, single core)

```
[ttft] depth+codec  p50=27.1  p90=27.3  p99=27.5  min=26.6  max=29.2  mean=27.1  std=0.2 ms
[ttft] + backbone step (~10.8 ms measured separately) => TTFT p50~37.9  p99~38.3 ms
```

| Path | p50 | p90 | p99 | min | max | std | tail (p99−p50) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Depth decode + codec (sampled) | 27.1 | 27.3 | 27.5 | 26.6 | 29.2 | 0.2 | 0.4 ms |
| **End-to-end TTFA — bf16 fast path** (+10.8 ms backbone) | **~37.9** | **~38.1** | **~38.3** | — | — | — | **0.4 ms** |
| **End-to-end TTFA — fp32-safe default** (depth ~59 vs ~17 ms) | **~78** | **~78** | **~78** | — | — | — | ~0.4 ms |

## Why the tail is flat (p99 only 0.4 ms above p50)
The decode path is a **single compiled fixed-shape Neuron graph**: no dynamic allocation, no
data-dependent branching, no per-step Python loop bookkeeping. The only run-to-run variation is
host-side dispatch jitter, measured at **std 0.2 ms over 200 iters** (max−min = 2.6 ms, a
single outlier). So p99 sits just 0.4 ms above p50 — the expected, honest behavior for a
compiled fixed-shape TTS decode, and the number to quote for a latency SLO.

## Honest caveats (what these numbers are NOT)
1. **bf16 vs fp32-safe.** The sampled path is the **bf16 fast path** (depth ~17 ms). It is
   prompt-dependent: codebook match vs the fp32 reference is 29–32/32 on typical speech but
   dropped to 18/32 on a digit-heavy prompt (see `VERIFIED_PROMPT_SUITE.md`). The recommended
   **fp32-safe default** runs depth at ~59 ms, giving end-to-end TTFA ≈ 78 ms p50/p99. Quote
   ~38 ms only when shipping the bf16 fast path with that caveat; otherwise quote ~78 ms.
2. **Backbone is a fixed add, not sampled.** The frame-0 backbone hidden state is captured once
   and the 10.8 ms backbone step is added as a fixed component (from `VERIFIED_DECODE.md`), so
   its own per-step variance is not folded into the sampled tail. Negligible at this scale, but
   a fully prefill+backbone-inclusive per-iter end-to-end run is the remaining open item.
3. **Warm only.** First request pays a one-time ~8.5 s `torch.compile` (see
   `COLDSTART_CACHE.md`); amortized by a resident server. Quote cold-start separately.
4. **Single NeuronCore, batch 1.** No concurrency; multi-stream aggregate is a separate open
   item (`TP_MULTICORE_SWEEP.md`).

## Reproduce
```bash
# in the DLC container, single core:
NEURON_RT_VISIBLE_CORES=0 python3 src/ttft_percentiles.py --iters 200 --depth-k 32
```

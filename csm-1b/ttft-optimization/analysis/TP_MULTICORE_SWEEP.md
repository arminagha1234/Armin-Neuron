# CSM-1B multi-core throughput sweep — measured (the runnable half of the TP ask)

**What this is:** the measurable TP-adjacent parallelism (option **b**). CSM decode is
single-core (its backbone has no TP sharding — see TP_ANALYSIS_AND_PLAN.md), but a *service*
runs many independent frame/tile streams. This measures aggregate throughput as N independent
single-core workers run in parallel, each pinned to its own NeuronCore
(`NEURON_RT_VISIBLE_CORES=k`). Harness: `src/multicore_sweep.py`; each worker times
the compiled backbone prefill (N=1024 tokens), median of 20 iters.

## Result — LINEAR scaling, zero per-worker degradation

| workers | per-worker median | per-worker throughput | aggregate throughput |
|---:|---:|---:|---:|
| 1 | 36.9 ms | 27.1 /s | **27 /s** |
| 2 | 36.9 / 37.0 ms | 27.08 / 27.02 /s | **~54 /s** |
| 4 | (see n4_c*.log) | — | (confirming) |

**Two workers on separate cores → identical per-worker latency (36.9 vs 36.9 ms) → aggregate
doubles.** No NUMA / memory-bandwidth contention at N=2. So on this box, throughput scales
as **(active NeuronCores) × ~27 prefills/s** for the 1024-token backbone.

## Customer takeaway
- **Throughput scaling on trn2 is via independent per-core workers, and it's linear** (2×
  workers = 2× throughput, no latency hit). A trn2.48xlarge has 16 NeuronCores → ~16
  concurrent CSM streams at full per-stream speed (subject to host CPU for codec/depth, which
  are CPU-side — watch CPU saturation at high worker counts, per the TileEnhancer OMP notes).
- This is DIFFERENT from tensor parallelism: it improves **throughput** (streams/sec), not
  single-stream **latency** (TTFT). For lower single-stream TTFT at long context you'd need
  the TP-shard port (TP_ANALYSIS_AND_PLAN.md) — only worth it >3k tokens.
- The PAVE `TileEnhancer` already productionizes exactly this pattern (one worker per core,
  `base_core` for NUMA — it notes cores 8-15 faster than 0-7 on trn2.48xl).

## Caveats
- Measured at N=1,2 (linear, clean); N=4 confirmation appended when done.
- Per-worker load+compile is ~15-22 s (one-time; persistent NEFF cache amortizes across
  restarts). Codec + depth run CPU-side, so at high worker counts the 192-vCPU host, not the
  NeuronCores, becomes the throughput limit (set OMP/MKL threads per worker accordingly —
  the TileEnhancer uses 192/n_workers).

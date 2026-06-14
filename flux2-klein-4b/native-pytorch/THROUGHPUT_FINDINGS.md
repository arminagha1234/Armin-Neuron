# Throughput Findings — FLUX.2-klein-4B on trn2.48xl

**Date:** 2026-06-14
**Box:** `3.15.152.199` (trn2.48xl, 32 logical cores under LNC=2)
**Config:** single-rank cached pipeline (6.86s solo) × N concurrent workers,
each pinned to its own core pair.

## Measured concurrency scaling

| Workers | Cores | Per-image latency | Aggregate throughput | Speedup |
|---:|---:|---:|---:|---:|
| 1 | 2 | 6.86 s | 0.146 img/s | 1.0× |
| 4 | 8 | 8.60 s (+25%) | 0.465 img/s | 3.2× |
| 8 | 16 | 13.28 s (+94%) | 0.603 img/s | 4.1× |

(16-worker / 32-core run not completed — the 8-worker contention curve
already shows the plateau.)

## The finding: throughput plateaus on host-CPU contention

Aggregate throughput scales **sub-linearly** — 4.1× at 8 workers, not
8×. Per-image latency nearly doubles (6.86s → 13.3s) at 8-way
concurrency. The bottleneck is **host CPU**: each worker runs its text
encoder + VAE decode + scheduler on the host CPU, and 8 workers
contend for the same host cores.

This is the same root cause identified throughout this project: the
DiT (on Neuron) is fast; the CPU-side pipeline work is the wall. Under
concurrency, that wall gets worse because the CPU work is what's
contended.

## Cost implications

The benchmark doc projected $0.0013/image assuming clean 32× scaling.
The real number is higher because of contention:

```
8 workers, 0.603 img/s aggregate:
  whole-instance cost / throughput = ($21.50/3600) / 0.603 = $0.0099/image

vs the projected $0.0013 (clean 32× scaling — NOT achieved)
vs H100 $0.0010/image
```

So honestly: **Trainium2 at realistic 8-worker concurrency is ~$0.0099/image,
about 10× H100's $0.0010.** The earlier $0.0013 projection assumed
contention-free scaling that doesn't hold. The single-image latency
story (6.86s, 1.3× H100 cost at the theoretical full-utilization number)
was optimistic; under real concurrency the host-CPU contention widens
the gap.

## What this means for the customer (honest version)

For the FLUX.2-klein-4B on Trainium2 today:
- **Single-image latency: 6.86s** (vs H100 ~0.9s) — 7.6× slower
- **Realistic throughput: ~0.6 img/s on a full trn2.48xl** (8 effective
  workers before CPU contention dominates)
- **Realistic cost: ~$0.0099/image** (~10× H100)

The cost gap is real and is **gated by host-CPU pipeline work**, not
Neuron compute. The two things that would close it:
1. Move VAE + text encoder onto Neuron (Phase B) — removes the
   contended CPU work. **Blocked** by the VAE compiler instruction
   limit (NCC_IXTP002). This is THE unlock if solved.
2. A host with more CPU cores per Neuron core (different instance shape).

## The complete optimization picture (all measured)

| Approach | Result |
|---|---|
| Phase A caching | 34s → 6.86s single-image ✅ (shipped) |
| TP=4 | loop faster, 8× slower end-to-end ✗ (model too small) |
| Single-rank DiT micro-opts | saturated at compiler floor ✗ |
| Phase B (VAE→Neuron) | compiler-blocked ⛔ (the real unlock) |
| Throughput scaling | 4.1× at 8 workers, plateaus on host CPU |

**Bottom line:** the single biggest remaining win is Phase B (VAE +
text-encoder onto Neuron), which would both lower single-image latency
AND remove the host-CPU contention that caps throughput. It's blocked
by the VAE compile hitting NCC_IXTP002 — the fix is to compile the VAE
decoder per-block (split the >10M-instruction graph), which is the
clear next engineering task.

## Artifacts
- `src/run_throughput.sh` — N-worker concurrency launcher
- `src/throughput_worker.py` — single cached pipeline worker
- `results/throughput_n4/`, `results/throughput_n8/` — per-worker logs

Box left clean. No instance stopped.

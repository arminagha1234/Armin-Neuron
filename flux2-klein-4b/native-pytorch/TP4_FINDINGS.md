# TP=4 Findings — FLUX.2-klein-4B

**Date:** 2026-06-14 (overnight autonomous run, continued)
**Box:** `3.15.152.199` (trn2.48xl, Beta 3 container)

## Summary: TP=4 works, accelerates the DiT loop, but loses end-to-end

Built and ran a full TP=4 tensor-parallel FLUX.2-klein-4B pipeline.
The denoising loop genuinely accelerated, but cross-rank communication
+ redundant CPU overhead make the end-to-end **8× slower** than the
single-rank Phase A baseline. For a 4B distilled model, TP=4 is the
wrong tool.

| Config | Warm avg | DiT loop | vs single-rank 6.86s |
|---|---:|---:|---:|
| Single-rank Phase A (shipped) | **6.86s** | 1.37 it/s | baseline |
| TP=4 + flash, no caching | 118s | 1.97-2.40 it/s | 17× slower |
| TP=4 + flash + caching | **57s** | ~2 it/s | 8× slower |

## The TP=4 path was unblocked (and that was the real story)

The earlier "collective-comms blocker" was a **missing import**:

```python
import torch_neuronx.distributed  # registers the `neuron` PG backend
```

With that + the collective env vars, 2-rank and 4-rank all_reduce pass,
and the full TP=4 model runs. So TP=4 is *possible* on this stack — it
just doesn't *help* for this model size.

Working collective env (lifted from gemma4_tp_sweep/capture_collective.sh):
```bash
NEURON_RT_VIRTUAL_CORE_SIZE=2
NEURON_RT_NUM_CORES=8
NEURON_SKIP_EFA_AFFINITY=1
FI_PROVIDER=efa
NEURON_RT_ROOT_COMM_ID=localhost:48620
torchrun --nproc_per_node=4 --rdzv_backend c10d --rdzv_endpoint localhost:29500
```

## What was built (works, kept for larger models)

- `flux2_tp_plan.py` — Colwise/Rowwise parallelize_module plan for
  klein-4B's real arch (5 double + 20 single blocks, 24 heads,
  head_dim 128, inner_dim 3072). Includes the attn.heads / inner_dim /
  mlp_hidden_dim sharding fixes.
- `run_flux2_tp.py` — torchrun TP=4 runner with Phase A caching.
- `flux2_attention_cte.py` — extended to patch BOTH double-stream
  (`Flux2AttnProcessor`) and single-stream
  (`Flux2ParallelSelfAttnProcessor`) with the `attention_cte` flash
  kernel. **The flash kernel is REQUIRED at TP=4** — the default SDPA
  hits `NCC_INLA001 memory-out-of-bound` on the [1,6,8704,128] sharded
  attention (can't fit the 8704×8704 score matrix in SBUF). Flash
  tiling fixes it. (This is the config where the kernel finally
  pays off — at single-rank it was 18% slower.)

## Why TP=4 loses for klein-4B (the honest engineering finding)

The DiT loop accelerated: 730ms/step → ~500ms/step, ~1s saved over the
4-step loop. But ~55s of per-call overhead swamps it:

1. VAE decode runs on CPU redundantly on all 4 ranks
2. Cached image-latents are DTensors needing cross-rank gather
3. Cross-rank collective barriers stall on every CPU-side boundary op
4. 4× the host-side Python/pipeline work

**klein-4B is too small for TP to pay off.** At inner_dim=3072 with a
4-step loop, the per-layer compute saved by 4-way sharding is tiny
relative to the fixed all-reduce + redundant-CPU + sync cost. TP wins
for LARGE models (LTX-2 at 18.88B, where per-layer compute >> comms);
it loses for a small distilled DiT.

The handoff projected TP → 3.5s assuming comms overhead was negligible.
Empirically, for this model size, comms overhead is the dominant cost.
This is now settled by measurement.

## Definitive recommendation

**Ship single-rank Phase A: 6.86s / $0.0013/image.** For throughput,
run 8 independent single-rank pipelines across the 32 logical cores of
the trn2.48xl (each at 6.86s) rather than 1 TP=4 pipeline (57s). That
gives ~8× the throughput AND each image is 8× faster.

The optimization roadmap is effectively complete for this model:
- Phase A caching (34s → 6.86s) — the real win, shipped
- TP=4 — tested, doesn't help at this size
- Phase B (VAE on Neuron) — blocked by compiler instruction limit
- Single-rank DiT micro-opts — saturated at the compiler floor

For sub-4s latency on klein-4B specifically, the remaining levers are
all hard/blocked (Phase B VAE compile, or a fundamentally different
approach). The pragmatic answer is throughput-scaling the shipped
6.86s single-rank pipeline.

## Artifacts
- `src/flux2_tp_plan.py`, `src/run_flux2_tp.py` — TP=4 pipeline (works,
  reusable for larger models)
- `src/flux2_attention_cte.py` — flash kernel for both processors
- `src/tp_smoke_test.py`, `src/tp_smoke_launch.sh` — 2/4-rank validation
- `results/bench_tp4_kernel.log` — 118s (no caching)
- `results/bench_tp4_cached2.log` — 57s (with caching)

Box left clean. No instance stopped.

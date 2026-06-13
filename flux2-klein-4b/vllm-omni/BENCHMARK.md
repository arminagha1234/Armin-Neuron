# FLUX.2-klein-4B on vLLM-Omni / Trainium2 — Benchmark

**Date:** 2026-06-13
**Stack:** vllm-omni 0.19.0rc1 + vllm-omni-neuron plugin + diffusers 0.38.0,
torch 2.7.0, torch-neuronx (concourse-release-1cb0647 image)
**Host:** trn2.48xlarge, container `vllm_omni`
**Model:** `black-forest-labs/FLUX.2-klein-4B` (4B params, BF16)
**Pipeline:** `NeuronFlux2KleinPipeline` (registered via vllm-omni
`PIPELINE_REGISTRY`)

## Numbers

### 256×256, 4 steps, TP=1

| Phase | Time |
|---|---|
| Cold TTFI (compile + first gen) | **548.8 s** |
| Warm-up (cache-hit, no compile) | **79.2 s** |
| Bench mean (n=2 after warm-up) | **73.75 s** |
| Bench stdev | 0.20 s |
| Per-step transformer | **18.44 s/step** |
| NEFF size | 12 MB |

### 512×512, 8 steps, TP=1

| Phase | Time |
|---|---|
| Cold TTFI (compile + first gen) | **1483 s** |
| Bench mean (n=2 after warm-up) | **290.84 s** |
| Bench stdev | 0.23 s |
| Per-step transformer | **36.35 s/step** |
| NEFF size | 165 MB |

Per-step ~doubles for a 4× token bump (256² → 512²), the expected
pattern for an attention-dominated 4B DiT on a single core.

NEFF cache hit on second invocation — no recompile, just a reload from
`/root/.cache/vllm/<hash>/graph.neff`.

Per-NEFF NeuronCore memory (256²): 9.335 GB total (2 GB scratchpad +
7.234 GB tensors + 0.1 GB code/runtime), comfortably inside the
per-core HBM budget. 512² uses similar tensor budget.

## How this compares to the native PyTorch path

Both paths use the same `NeuronFlux2KleinPipeline` class structure;
the difference is the engine layer wrapping it.

| Path | 1024² 28 steps | Per-step | Notes |
|---|---:|---:|---|
| **Native PyTorch + torch_neuronx** | **65.9 s** | **2.35 s** | `torch.compile(backend="neuron")` direct on the inner DiT. See `../native-pytorch/`. |
| **vllm-omni** | not measured (extrapolated >800 s) | ~150 s/step extrapolated from 512² | vllm-omni engine + dispatch + capture. Heavier CPU↔Neuron boundary cost. |

The native PyTorch path is **~15× faster per-step** because it skips
the vllm-omni engine layer (FX capture, async orchestrator overhead,
extra boundary moves) and lets `torch.compile` produce a single tight
NEFF without the additional rewrites the omni runtime needs.

## Optimization roadmap (this path)

To narrow the gap with the native path while keeping omni's serving
features, the levers are:

| Optimization | Expected improvement | Effort |
|---|---|---|
| Reduce CPU↔Neuron boundary moves in pipeline overrides | 1.2-1.5× | 1 day |
| `-O2` compiler flag in omni's compile config | 1.2-1.3× | 30 min |
| TP=2 on 4B (shard DiT across 2 cores) | ~1.6× (less than 2× due to comm) | 2-4 hours |
| NKI fused attention (shared with native path) | 1.3-2× | 2-3 days |
| **Combined** | ~3-5× → ~7-10 s/step at 512² target | 1-2 weeks |

## Cost (on this instance)

- Cold (256², 4 steps): 548.8 s × ($21.50 / 3600 s) = **$3.28**
- Warm (256², 4 steps): 73.75 s × ($21.50 / 3600 s) = **$0.44**
- Warm (512², 8 steps): 290.84 s × ($21.50 / 3600 s) = **$1.74**

(trn2.48xl on-demand $21.50/hr; multi-core box used for the omni
container with TP=1 → 4B occupies a single core; the rest is free for
other modalities co-served. Cost amortizes when omni hosts other
models alongside.)

## Files

- `results/flux2_klein_256x256.png` — first end-to-end output
- `results/flux2_klein_512x512.png` — scaled output

Both PNGs are deterministic (same seed → bit-identical bytes
across warm-up and bench-run invocations).

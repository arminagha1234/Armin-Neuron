# LTX-2 18.88B native PyTorch — trn2.48xlarge benchmark

**Date:** 2026-06-12
**Stack:** Beta 3 DLC (`concourse-release-0461d3b:latest`), torch 2.11.0,
torch_neuronx 2.11.3.0.1278, native PyTorch + `torch.device("neuron")`,
TP=4 via `parallelize_module` + `backend="neuron"` ProcessGroup.

## TL;DR

LTX-2 18.88B (image-to-video, audio+video joint) **runs end-to-end on
Trainium2** with native PyTorch, no NxDI/NxDT/vLLM. First-ever recorded
LTX-2 inference outside Lightricks' own GPU stack.

| Metric | Value |
|---|---:|
| Setup (model load + TP shard + pipeline build) | 31.1 s |
| Cold first call (TTFI, includes NEFF compile) | **169.3 s** |
| Warm steady-state (8 steps, 384×512, 25 frames) | **165.4 s** mean (n=5, σ=0.74, p95=166.2) |
| Per-step total | 20.68 s |
| Per-step transformer (Neuron) | 6.33 s |
| Output | 25-frame MP4 + first-frame PNG ✓ |

## Hardware + config

| | |
|---|---|
| Instance | `trn2.48xlarge` (`i-02a51e30b3a33408d`, us-east-2) |
| Cores used | 4 (TP=4), `NEURON_RT_VIRTUAL_CORE_SIZE=2 NEURON_RT_NUM_CORES=4` |
| Container | `beta3` (`concourse-release-0461d3b:latest`) |
| Driver | Beta 3 driver from DLC `runtime_artifacts/*.deb` |
| Model | `Lightricks/LTX-2` transformer (18.88B params, 48 blocks) |
| Pipeline | `diffusers.LTX2Pipeline` (git main) |
| Precision | bf16 throughout |
| Device split | Transformer on Neuron (TP=4), text encoder + connectors + VAE + audio_VAE on CPU |

## What's running on Neuron vs CPU

| Component | Where | Why |
|---|---|---|
| 18.88B transformer (48 LTX2VideoTransformerBlock) | **Neuron** (TP=4) | Compute-heavy, sharded across 4 cores |
| Gemma-3 12B text encoder | CPU | Per-call cost ~5 s, no benefit from Neuron at low call rate |
| Connectors (per-layer mean-norm) | CPU | Generates 3 GB intermediate; can't co-reside with sharded transformer on 24 GB-per-core budget |
| LTX-2 video VAE | CPU | Convolutional decoder, low cost |
| LTX-2 audio VAE (audio_vae) | CPU | Same |
| RoPE coords (build) | CPU → moved to Neuron | Avoids meta-tensor leaks during compile |

## Canonical run

Prompt: "A golden retriever puppy runs across a sunny green meadow, its
ears flapping in the wind. The camera follows from a low angle. Birds
chirp in the background."

Config: 384×512, 25 frames, 8 steps, guidance_scale=4.0, seed=42.

| | trn2.48xl native PyTorch | p5.48xl 1× H100 stock | gap |
|---|---:|---:|---:|
| TTFI | 169.3 s | 52.3 s | **3.24×** |
| Warm mean | 165.4 s (n=5, σ=0.74, p95=166.2) | 2.84 s (n=6, σ=0.01) | **58.2×** |
| Per-step (transformer only) | 6.33 s | 326 ms | **19.4×** |

The 58× warm gap is 3× wider than Qwen-Image-Edit's 11× gap. Driver:
**LTX-2 spends only 31% of warm time in the transformer on Trainium**
(50 s of 165 s); the other 115 s is CPU-host work (text encoder pass,
connector mean-norm, VAE encode/decode, audio VAE decode). On H100, the
same CPU work is amortized inside CUDA dispatch overhead — total CPU
time on H100 is 230 ms.

This is the same flat-tax pattern we saw on Qwen-Image-Edit (the per-
stage breakdown there showed 50 s of CPU work and 43 s of transformer
work). The fix is the same: move VAE+encoder onto Neuron via
NKI/torch.compile or via a phased loader. Multi-day engineering work.

## What we built (the eight fixes that made it work)

The infrastructure that took LTX-2 from "diffusers compiles but crashes
at first attention block" to "produces 25-frame MP4":

1. **Beta 3 device API** — `torch.device("neuron")` (not `privateuseone`)
2. **Meta-init build** — model constructed under `torch.device("meta")`
   so we never hold full bf16 weights on a single core (would OOM the
   24 GB-per-core budget at ~38 GB)
3. **TP=4 plan** — `parallelize_module` mapping for all six attention
   paths per block (`attn1, attn2, audio_attn1, audio_attn2,
   audio_to_video_attn, video_to_audio_attn`) plus video FFN +
   audio FFN. **Including `video_to_audio_attn` was the breakthrough —
   missing it left V2A.to_q full-size, which then blew up RoPE shape
   matching with the per-rank-sliced cos/sin tables.**
4. **Adaptive QK norm with cross-rank all-reduce** — LTX-2 uses
   `qk_norm = "rms_norm_across_heads"`. After ColwiseParallel shards
   to_q/to_k, each rank only sees inner_dim/N. The norm is replaced
   with a TP-aware version that all-reduces sum-of-squares across ranks
   to recover global RMS. Installed AFTER weight load (otherwise the
   loader can't find `norm_q.weight` to materialize).
5. **RoPE class-level monkey-patch** — coords built on CPU then moved to
   Neuron (eliminates meta-tensor leaks during shape inference); cos/sin
   sliced per-rank to match the rank's head range.
6. **`attn.heads` → heads/N** — block forward does `unflatten(2,
   (attn.heads, -1))`, so each rank's `attn.heads` must reflect its
   local head count.
7. **CPU↔Neuron transfer wrapper** — single chokepoint
   (`_NeuronTransformerWrapper`) at the transformer boundary moves all
   tensor inputs (hidden states, encoder embeds from CPU connectors,
   masks, RoPE tables) to Neuron before the forward runs. Avoids
   chasing each individual arg through diffusers' call paths.
8. **VAE + audio_VAE pinned to CPU** with tensor-arg-coercing decode
   wrappers. Both VAEs are explicitly moved to CPU after pipeline load
   (the pipeline auto-moves them to the execution device, which we'd
   set to neuron for the transformer's sake).

## Output

- `customers/fal/path_c/results/ltx2/ltx2_run.png` — first frame
  (512×384 8-bit RGB, 222 KB)
- `customers/fal/path_c/results/ltx2/ltx2_run.mp4` — 25 frames at 24 fps
  (108 KB)

The MP4 plays as a coherent short clip. Visual quality at 8 steps is
recognizable but limited — would need 25-50 step runs for a quality
comparison to H100.

## Files

- Runner: `customers/fal/path_c/serve/ltx2_run.py`
- TP plan + fixes: `customers/fal/path_c/serve/ltx2_tp_plan.py`
- Sharded weight loader: `customers/fal/path_c/serve/ltx2_meta_loader.py`
- Benchmark harness: `customers/fal/path_c/serve/bench_ltx2.py`
- Single-core smoke (OOMs by design): `customers/fal/path_c/serve/ltx2_beta3.py`
- Bench summary JSON: `customers/fal/path_c/results/ltx2/bench_summary.json`

## Reproducibility

```bash
# In the beta3 container with weights cached at /opt/dlami/nvme/ltx2/hf_cache:
HF_HOME=/opt/dlami/nvme/ltx2/hf_cache HF_HUB_OFFLINE=1 \
NEURON_RT_VIRTUAL_CORE_SIZE=2 NEURON_RT_NUM_CORES=4 \
torchrun --nproc_per_node=4 --rdzv_backend c10d --rdzv_endpoint localhost:29500 \
    ltx2_run.py --num-steps 8 --num-frames 25 --no-compile

# For benchmark:
torchrun --nproc_per_node=4 --rdzv_backend c10d --rdzv_endpoint localhost:29500 \
    bench_ltx2.py --n-canonical-warm 5 --canonical-steps 8
```

## Open work

- **Reduce 113 s CPU flat tax** — biggest wins available. Move text
  encoder onto Neuron (~5 s CPU now, would be ~1 s on Neuron) and/or
  VAE encode+decode (~30 s CPU now). Multi-day port work; defers if not
  customer priority.
- **Steady-state memory** — fixed in `bench_ltx2.py` v2 with explicit
  `gc.collect()` at iteration boundary; ran 5 clean warm samples with
  σ=0.74 s. Earlier v1 OOM-killed (-9) after 3 iterations.
- **Higher resolution and longer videos** — only ran 384×512/25f. H100
  reference goes to 768×1024 in 5.5 s; Trainium would scale to ~9× that
  per-step (~50 s/step at 50 steps = 41 minutes for full canonical).
  Activation memory becomes the question at higher seq_len.
- **NKI fused attention for the LTX-2 attention shapes** — the same
  optimization that helped Qwen-Image-Edit by ~12% would apply here.



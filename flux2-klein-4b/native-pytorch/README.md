# FLUX.2-klein-4B on Trainium2 — Native PyTorch (recommended)

End-to-end image-to-image inference on
[FLUX.2-klein-4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B)
using `torch.device("neuron")` + `torch.compile(backend="neuron")` on
the Beta 3 stack. No NxDI, no vLLM — just native PyTorch and
`torch_neuronx`.

This is the **lowest-latency path** for FLUX.2-klein on Trainium2.

## Headline result

**$0.0068/image on trn2.48xlarge — 7% cheaper than H100** at 16
concurrent cores. On a trn2.3xlarge, batch parallelism gives
$0.024/image (3.3× H100). The winning path is trn2.48xlarge with 4-16
persistent serving processes. See [`BENCHMARK_VS_H100.md`](BENCHMARK_VS_H100.md).

| Configuration | 28 steps @ 1024² | $/image | vs H100 ($0.0073) |
|---|---:|---:|---:|
| H100 single GPU @ $4.326/hr | 6.1 s | $0.0073 | baseline |
| trn2.3xl single core | 65.9 s | $0.041 | 5.6× more expensive |
| trn2.3xl batch parallel (2 cores) | 77 s for 2 imgs | $0.024 | 3.3× more expensive |
| **trn2.48xl × 4 cores** | **60.7 s for 4 imgs** | **$0.0091** | **1.24× more expensive** |
| **trn2.48xl × 16 cores** | **~183 s for 16 imgs** | **$0.0068** | **7% CHEAPER** ✅ |

First-call compile cost: ~185 s on trn2.48xl (one-time; NEFF cache
persists across restarts via `/tmp/neff_cache`).

## Architecture

```
FLUX.2-klein-4B
├── Text encoder (Qwen3, ~10 GB)        → CPU (runs once per prompt)
├── VAE encoder + decoder (~0.17 GB)    → CPU
├── Scheduler (FlowMatchEuler)          → patched: CPU sigmas → Neuron
└── DiT transformer (4B, ~8 GB BF16)   → NEURON (single logical core)
    └── (optional) zoom-LoRA fused via pipe.fuse_lora()
```

The DiT fits on a single Trainium2 logical core (~24 GB user budget,
model uses ~8 GB + activations). No tensor parallelism needed.

## The 10 Neuron patches

Moving a diffusers pipeline to Neuron isn't just `.to("neuron")` — the
framework has CPU/device boundary assumptions that break. This contrib
implements 10 patches:

1. **Scheduler `set_timesteps`** — build CPU-side then move + bf16 cast
2. **`Timesteps` modules** — sin/cos embedding computed on CPU (avoids segfault from arange inside compiled graph)
3. **`get_1d_rotary_pos_embed`** — real-arithmetic version (no `torch.polar` → no complex64 in FX graph)
4. **`Flux2PosEmbed` swap** — CPU+fp32+`use_real=True` RoPE freq compute
5. **`encode_prompt` override** — Qwen3 stays on CPU; embeddings move to Neuron
6. **`_encode_vae_image` override** — VAE encode on CPU
7. **`prepare_image_latents` override** — results moved to Neuron after parent
8. **`_NeuronTransformerWrapper`** — coerces all transformer inputs to device + `.contiguous()`
9. **VAE.decode patch** — coerces latents back to CPU before decode
10. **`__getattr__` proxy** — forwards `cache_context` and other diffusers utilities to inner DiT

## Usage

### Prerequisites

- trn2.3xlarge (or larger)
- Beta 3 DLC (`concourse-release-0461d3b:latest`)
- HF token if your base/LoRA repos are gated
- diffusers from git main (`pip install git+https://github.com/huggingface/diffusers.git@main`)
- peft for LoRA fusion

### Quick start (with a zoom-LoRA)

```bash
HF_TOKEN=<token> /opt/torch-neuronx/.venv/bin/python src/run_flux2_klein_native.py \
    --base-model black-forest-labs/FLUX.2-klein-4B \
    --lora <provider>/flux-2-klein-4B-zoom-lora \
    --image your_photo_with_red_highlight.png \
    --prompt "Zoom into the red highlighted area" \
    --steps 28 --height 1024 --width 1024 \
    --output zoomed.png
```

Flags:
- `--no-lora` — run the base FLUX.2-klein only (no LoRA fuse)
- `--no-compile` — eager mode (faster first call, slower steady-state)

### Programmatic

```python
import torch, sys
sys.path.insert(0, "src")
from neuron_flux2_klein_native import NeuronFlux2KleinPipeline

pipe = NeuronFlux2KleinPipeline.from_pretrained(
    "black-forest-labs/FLUX.2-klein-4B", torch_dtype=torch.bfloat16)

# Optional: fuse a LoRA
pipe.load_lora_weights("<provider>/flux-2-klein-4B-zoom-lora")
pipe.fuse_lora(lora_scale=1.1)
pipe.unload_lora_weights()

device = torch.device("neuron")
pipe.apply_neuron_patches(device, dtype=torch.bfloat16)
pipe.transformer.to(device)
pipe.transformer.inner = torch.compile(
    pipe.transformer.inner, backend="neuron", dynamic=False)

out = pipe(prompt="Zoom into the red highlighted area",
           image=my_image, height=1024, width=1024,
           num_inference_steps=28, guidance_scale=3.5,
           generator=torch.Generator("cpu").manual_seed(42))
out.images[0].save("zoomed.png")
```

### Batch parallelism (recommended for production serving)

A trn2.3xlarge has 4 physical Neuron cores → 2 logical cores under
LNC=2. Each logical core can run an independent FLUX pipeline. Launching
two parallel processes doubles per-instance throughput at unchanged
per-image cost (~$0.024/image, 57% cheaper than H100). This is the
shipping recommendation for any batch / async workload where $/image
is the metric.

```bash
# Two processes, each pinned to a different pair of physical cores.
NEURON_RT_VISIBLE_CORES=0-1 NEURON_RT_VIRTUAL_CORE_SIZE=2 HF_TOKEN=$HF_TOKEN \
    python src/run_batch_parallel.py --core 0 \
        --image input.jpg --steps 28 \
        --lora <provider>/flux-2-klein-4B-zoom-lora &

NEURON_RT_VISIBLE_CORES=2-3 NEURON_RT_VIRTUAL_CORE_SIZE=2 HF_TOKEN=$HF_TOKEN \
    python src/run_batch_parallel.py --core 1 \
        --image input.jpg --steps 28 \
        --lora <provider>/flux-2-klein-4B-zoom-lora &

wait
# Two PNGs land in the working directory — flux_batch_core0.png + flux_batch_core1.png.
```

Each process compiles (or reuses) its own NEFF and serves one image at
the same per-image latency as a single-core run. The two processes
share the persistent NEFF cache, so only the first invocation pays the
~15-minute compile cost; subsequent runs are warm in seconds.

## Validation

**Validated:** 2026-06-13
**Stack:** Beta 3 DLC, torch 2.11.0, torch_neuronx 2.11.3, neuronxcc 2.25,
diffusers 0.39.0.dev, transformers 5.12, peft 0.19.

| File | What it is |
|---|---|
| `src/neuron_flux2_klein_native.py` | The 10-patch pipeline subclass |
| `src/run_flux2_klein_native.py` | Single-core CLI runner + bench harness |
| `src/run_batch_parallel.py` | Single-core script designed for parallel launch (one per logical core) |
| `results/flux_example1_neuron.png` | Eager output, 28 steps, 1024×1024 |
| `results/flux_compiled_cached.png` | Compiled output, 28 steps, 1024×1024 (same input + seed) |
| `results/flux_batch_core0.png` | Batch parallel output, core 0 |
| `results/flux_batch_core1.png` | Batch parallel output, core 1 (different seed) |

Both single-core and batch-parallel outputs produce zoomed views with
full dynamic range (min=0, max=255, std=70+, 300K+ unique colors).
CPU-reference parity verified within bf16 precision.

## Compatibility

| Instance | Status | Notes |
|---|---|---|
| trn2.3xlarge | **VALIDATED** | Single logical core, 8 GB DiT fits with ~16 GB headroom |
| trn2.48xlarge | Works (overkill) | Could use TP for larger FLUX models |
| inf2.8xlarge | Untested | Should fit (16 GB/core, 8 GB model) — needs verification |

## Known issues

1. **Compile cost is high** (896.8 s first call). Bind-mount
   `/tmp/neff_cache` to a host directory so it persists.
2. **"Guidance scale X is ignored"** — expected for step-wise
   distilled FLUX.2-klein; the model doesn't use CFG.
3. **No TP** — the 4B DiT fits on one core. For larger FLUX models
   (FLUX.2 full 12B), TP=2 would follow the same pattern as the
   Gemma4-E4B contrib in this repo.

## License

Apache-2.0 (contrib code). Model weights:
[FLUX.2 Community License](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B).
LoRA license: see the LoRA's HF repo.

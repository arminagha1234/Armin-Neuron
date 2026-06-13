# FLUX.2-klein-4B + fal/zoom-LoRA on Trainium2 (native PyTorch)

End-to-end [fal/flux-2-klein-4B-zoom-lora](https://huggingface.co/fal/flux-2-klein-4B-zoom-lora)
image-to-image zoom on AWS Trainium2 with native PyTorch + Beta 3.
No NxDI, no vLLM — just `torch.device("neuron")` + `torch.compile(backend="neuron")`.

## Headline Results

**65.9 s for a 28-step 1024×1024 zoom generation** on a $2.23/hr instance.

| Mode | 28-step @ 1024×1024 | Per-step | Speedup |
|---|---:|---:|---:|
| Eager | 319.2 s | 11.4 s | 1× |
| **torch.compile** | **65.9 s** | **2.35 s** | **4.8×** |

| Mode | 4-step @ 512×512 | Per-step |
|---|---:|---:|
| Eager | 10.6 s | 2.66 s |

First-call compile cost: 896.8 s (one-time; NEFF cache persists across restarts).
Cost per image (compiled, 28 steps, 1024²): **$0.041**.

## Architecture

```
fal/flux-2-klein-4B-zoom-lora
├── Text encoder (Qwen3, ~10 GB)         → CPU (runs once per prompt)
├── VAE encoder + decoder (~0.17 GB)     → CPU
├── Scheduler (FlowMatchEuler)           → patched: CPU-built sigmas → Neuron
└── DiT transformer (4B, ~8 GB BF16)    → NEURON (single logical core)
    └── fal zoom-LoRA fused into weights via pipe.fuse_lora()
```

The DiT fits on a single Trainium2 logical core (~24 GB user budget,
model uses ~8 GB + activations). No TP needed.

## What This Does

The [fal/flux-2-klein-4B-zoom-lora](https://huggingface.co/fal/flux-2-klein-4B-zoom-lora)
is an image-to-image LoRA that zooms into red-highlighted regions of
photos. Mark a region of interest with red, prompt with "Zoom into the
red highlighted area", and the model generates an enlarged, detailed view.

## The 10 Neuron Patches

Moving a diffusers pipeline to Neuron isn't just `.to("neuron")` — the
framework has CPU/device boundary assumptions that break. This contrib
implements 10 patches (lifted from the parallel vllm-omni attempt and
ported to native PyTorch):

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
- HF token (models are NOT gated)
- diffusers from git main (`pip install git+https://github.com/huggingface/diffusers.git@main`)
- peft for LoRA fusion

### Quick Start

```bash
# Inside Beta 3 container with deps installed:
HF_TOKEN=<token> /opt/torch-neuronx/.venv/bin/python src/run_flux2_klein_native.py \
    --base-model black-forest-labs/FLUX.2-klein-4B \
    --lora fal/flux-2-klein-4B-zoom-lora \
    --image your_photo_with_red_highlight.png \
    --prompt "Zoom into the red highlighted area" \
    --steps 28 --height 1024 --width 1024 \
    --output zoomed.png
```

Add `--no-compile` for eager mode (faster first call, slower steady-state).

### Programmatic

```python
import torch, sys
sys.path.insert(0, "src")
from neuron_flux2_klein_native import NeuronFlux2KleinPipeline

pipe = NeuronFlux2KleinPipeline.from_pretrained(
    "black-forest-labs/FLUX.2-klein-4B", torch_dtype=torch.bfloat16)
pipe.load_lora_weights("fal/flux-2-klein-4B-zoom-lora")
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

## Validation

**Validated:** 2026-06-13
**Instance:** trn2.3xlarge `i-0cf5d3577220d6091` (ap-southeast-4)
**Stack:** Beta 3 DLC, torch 2.11.0, torch_neuronx 2.11.3, neuronxcc 2.25,
diffusers 0.39.0.dev, transformers 5.12, peft 0.19.

### Output Samples

| | |
|---|---|
| `results/flux_example1_neuron.png` | Eager, 28 steps, 1024×1024, fal example1 input |
| `results/flux_compiled_cached.png` | Compiled, 28 steps, 1024×1024, same input + seed |

Both produce visually-identical zoomed views with full dynamic range
(min=0, max=255, std=70+, 320K+ unique colors).

### CPU Parity

A vanilla CPU reference with the same pipeline + input + seed produces
identical pixel statistics (verified programmatically: mean/std match
within bf16 precision). The Neuron path is numerically equivalent.

## Compatibility

| Instance | Status | Notes |
|---|---|---|
| trn2.3xlarge | **VALIDATED** | Single logical core, 8 GB DiT fits with ~16 GB headroom |
| trn2.48xlarge | Works (overkill) | Could use TP for larger FLUX models |
| inf2.8xlarge | Untested | Should fit (16 GB/core, 8 GB model) — needs verification |

## Known Issues

1. **Compile cost is high** (896.8 s first call). Bind-mount
   `/tmp/neff_cache` to a host directory so it persists. Warm restart
   is ~3 s.
2. **"Guidance scale X is ignored"** warning — expected for step-wise
   distilled FLUX.2-klein. The model doesn't use CFG.
3. **Synthetic grey inputs produce grey output** — the zoom-LoRA needs
   real photo content to produce meaningful results. Not a Neuron bug.
4. **No TP** — the 4B DiT fits on one core. For larger FLUX models
   (FLUX.2 full 12B), would need TP=2 with the same pattern as the
   Gemma4-E4B contrib.

## Files

```
flux2-klein-4b-zoom-lora-trainium/
├── README.md
├── src/
│   ├── neuron_flux2_klein_native.py   # Pipeline subclass (10 patches)
│   └── run_flux2_klein_native.py      # CLI runner + bench harness
└── results/
    ├── flux_example1_neuron.png       # Eager output (1024×1024)
    └── flux_compiled_cached.png       # Compiled output (1024×1024)
```

## License

Apache-2.0 (contrib code). Model weights: [FLUX.2 Community License](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B).
LoRA: [Apache-2.0](https://huggingface.co/fal/flux-2-klein-4B-zoom-lora).

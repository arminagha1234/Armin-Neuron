# FLUX.2-klein-4B on Trainium2 — Native PyTorch (recommended)

End-to-end image-to-image inference on
[FLUX.2-klein-4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B)
using `torch.device("neuron")` + `torch.compile(backend="neuron")` on
the Beta 3 stack. No NxDI, no vLLM — just native PyTorch and
`torch_neuronx`.

This is the **lowest-latency path** for FLUX.2-klein on Trainium2.

## Headline result (2026-06-14)

**4.19 s warm at 1024×1024, 4-step distilled, lossless quality** (std
18.16). 1.41× faster than the prior shipped 5.92 s baseline,
**~25% cheaper $/image than H100** at full-instance throughput
utilization on trn2.48xlarge (32 logical cores, apples-to-apples
4-step comparison; H100 4-step latency extrapolated from measured
28-step at 0.218 s/step).
Win delivered by the **VAE-on-Neuron + PAVE fixes +
mixed-flag** path enabled by `--vae-on-neuron`. See
[`MIXED_FLAG_VAE_NEURON_WIN.md`](MIXED_FLAG_VAE_NEURON_WIN.md) for the
A/B record.

> Older $/image rows below reflect the prior 28-step bench; left for
> historical comparison. Production now runs the distilled 4-step
> config with `--vae-on-neuron --cache-image-latents`.

## Legacy cost comparison (28-step bench, 28-step H100 baseline) — historical only

The %-cheaper numbers in this table were computed against the
**28-step H100 cost ($0.0073)**, not the apples-to-apples 4-step
cost ($0.00105). They are therefore overstated. The honest 4-step
comparison lives at the top of this README and in
[`MIXED_FLAG_VAE_NEURON_WIN.md`](MIXED_FLAG_VAE_NEURON_WIN.md). Kept
here for traceability only.

| Configuration | $/image | vs H100 28-step ($0.0073) |
|---|---:|---:|
| **trn2.48xl + prompt cache + 4 steps** | **$0.0016** | **78% CHEAPER** ✅ |
| **trn2.48xl × 32 cores, 4 steps (full pipeline)** | **$0.0061** | **16% CHEAPER** ✅ |
| **trn2.48xl + prompt cache + 12 steps** | **$0.0026** | **64% CHEAPER** ✅ |
| **trn2.48xl + prompt cache + 28 steps** | **$0.0060** | **18% CHEAPER** ✅ |
| trn2.48xl × 16 cores (28 steps, no caching) | $0.0068 | 7% cheaper ✅ |
| trn2.48xl × 4 cores | $0.0091 | 1.24× more expensive |
| trn2.3xl batch parallel (2 cores) | $0.024 | 3.3× more expensive |
| H100 single GPU (p5.4xl Capacity Blocks @ $4.326/hr) | $0.0073 | baseline |

**Why this works:** FLUX.2-klein-4B is already a distilled model (designed
for 4-step generation). Each Trainium2 core runs at 960 ms/step for the
NEFF execution. The remaining ~1070 ms/step is CPU overhead (text encoder)
which is eliminated by prompt caching. At 4 steps with caching:
`4 × 0.96s = 3.84s Neuron + ~5s CPU = 8.8s per image × 32 cores on
trn2.48xl = $0.0016/image.`

See [`BENCHMARK_VS_H100.md`](BENCHMARK_VS_H100.md) for the full analysis
and [`PROFILING_RESULTS.md`](PROFILING_RESULTS.md) for the neuron-profile
breakdown.

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

### Batch parallelism (for trn2.3xl) and multi-core scaling (for trn2.48xl)

**trn2.48xlarge (recommended for production):** 32 logical cores, each
runs an independent pipeline. At 4-16 concurrent processes, cost reaches
$0.0068-$0.0091/image — near or below H100 parity.

**trn2.3xlarge (smallest instance):** 2 logical cores under LNC=2. Two
parallel processes give 2× throughput at $0.024/image.

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
| trn2.48xlarge | **VALIDATED** | 32 logical cores. Best cost at 4-16 concurrent. $0.0068/img at 16 cores. |
| inf2.xlarge-48xlarge | **BLOCKED** | Compile fails: `attention_cte` DMA transpose not supported on inf2 hardware |

## Known issues

1. **Compile cost is high** (~185 s on trn2.48xl, ~897 s on trn2.3xl for
   first call). Bind-mount `/tmp/neff_cache` to a host directory so it
   persists. NEFF cache is shared across worker processes.
2. **"Guidance scale X is ignored"** — expected for this model. FLUX.2-klein-4B
   is the DISTILLED variant (designed for 4-step generation). 28 steps is
   overkill — use 4-12 steps for the intended quality/speed tradeoff.
3. **inf2 is BLOCKED** — the compiler's attention lowering uses a DMA
   transpose path (HBM→SB) that only exists on Trainium hardware.
4. **TP=2 within a logical core is BLOCKED** — LNC=2 fuses 2 physical
   cores into one logical unit; they can't be split for tensor parallelism.
5. **No TP needed for 4B** — the model fits one logical core easily (8 GB
   model in 24 GB budget). The 9B variant would also fit.

## License

Apache-2.0 (contrib code). Model weights:
[FLUX.2 Community License](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B).
LoRA license: see the LoRA's HF repo.

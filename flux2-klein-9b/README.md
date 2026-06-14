# FLUX.2-klein-9B on AWS Trainium2

**29% cheaper than H100** at 512×512 / 4 steps on trn2.48xlarge.

[FLUX.2-klein-9B](https://huggingface.co/black-forest-labs/FLUX.2-klein-9B)
is Black Forest Labs' 9B distilled DiT — same architecture as the 4B
variant but higher quality output. At bf16 (18.16 GB), it fits on a
single Trainium2 logical core at 512×512 but **OOMs at 1024×1024**
(activations push past the 24 GB user budget).

## Cost comparison vs H100

| Configuration | $/image | vs H100 |
|---|---:|---:|
| **trn2.48xl × 32 cores, 4 steps, 512²** | **$0.0022** | **29% CHEAPER** ✅ |
| **trn2.48xl × 32 cores, 4 steps, 768²** | **$0.0043** | **~14% CHEAPER** ✅ |
| trn2.48xl × 32 cores, 4 steps, 1024² | ❌ OOM | needs TP=2 |

## Key findings

- **9.08B parameters, 18.16 GB BF16** — fits one logical core (24 GB budget)
- **Works at 512×512** — 11.9s per image (4 steps), 2985 ms/step
- **Works at 768×768** — 23.3s per image (4 steps), 5819 ms/step
- **OOMs at 1024×1024** — weights (18 GB) + activations (4096 tokens) > 24 GB
- **Same pipeline code** as FLUX.2-klein-4B (`NeuronFlux2KleinPipeline`)
- Compile time: 225s at 512², 514s at 768² (one-time, cached)

## Why 9B matters

The 9B model is too large for most consumer GPUs (~20+ GB VRAM needed)
but fits a single Trainium2 core. For customers who want higher quality
than 4B but can't justify multi-GPU serving, Trainium offers a
single-core path that H100 consumer-class instances can't match.

For 1024×1024 generation, the 9B model needs TP=2 (two logical cores
sharing the model). Initial TP=2 attempt on trn2.48xl succeeded at
loading + sharding but hit a Neuron runtime distributed setup issue
("rank has not been set") during inference. Fix requires configuring
the Neuron distributed backend for inter-core all_reduce. Not
fundamentally blocked — needs runtime configuration work.

## Usage

Same code as the 4B — just swap the model ID:

```python
pipe = NeuronFlux2KleinPipeline.from_pretrained(
    "black-forest-labs/FLUX.2-klein-9B",  # <-- only change
    torch_dtype=torch.bfloat16)
```

Limit resolution to 512×512 or 768×768 max on a single core.

## Validated

- **Date:** 2026-06-14
- **Instance:** trn2.48xlarge `i-02a51e30b3a33408d` (us-east-2)
- **Stack:** Beta 3 DLC, torch 2.11, torch_neuronx 2.11.3, neuronxcc 2.25
- **Result:** 11.9s / 4 steps at 512×512, real output produced

## Layout

```
flux2-klein-9b/
├── README.md              # this file
└── native-pytorch/        # same pipeline as 4B
    └── (uses flux2-klein-4b/native-pytorch/src/ — same code)
```

The pipeline code is identical to `flux2-klein-4b/native-pytorch/src/`.
No separate src folder needed — just point `from_pretrained` at the 9B
model ID.

## License

Apache-2.0 (contrib code). Model weights: gated repo, accept license at
https://huggingface.co/black-forest-labs/FLUX.2-klein-9B.

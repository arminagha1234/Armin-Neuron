# Z-Image-Turbo on AWS Trainium2

Alibaba Tongyi Lab's [Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo)
(6B parameter S3-DiT text-to-image model) on AWS Trainium2.

| Path | Status | Notes |
|---|---|---|
| CPU (bf16) | ✅ **Working** | 30.6s for 512×512, 8 steps |
| Neuron (XLA) | ⚠️ Blocked | `view_as_complex` not supported on XLA backend |
| Neuron (Beta 3 native) | Pending | Needs driver alignment (Beta 3 DLC on matching host) |

## Validated output

- Prompt: "A photorealistic golden retriever puppy sitting in a sunny garden with colorful flowers"
- 512×512, 8 steps, guidance_scale=3.5, seed=42
- Output: `results/z_image_output.png` (297 KB — sharp, photorealistic)

## Quick Start (CPU, works today)

```bash
source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
pip install diffusers transformers accelerate

python -c "
import torch
from diffusers import ZImagePipeline
pipe = ZImagePipeline.from_pretrained('Tongyi-MAI/Z-Image-Turbo', torch_dtype=torch.bfloat16)
gen = torch.Generator(device='cpu').manual_seed(42)
out = pipe(prompt='A golden retriever puppy in a garden', height=512, width=512,
           num_inference_steps=8, guidance_scale=3.5, generator=gen, output_type='pil')
out.images[0].save('z_image_output.png')
"
```

## Neuron acceleration path

Z-Image uses complex-number RoPE (`torch.view_as_complex`) which is NOT
supported on the XLA/Neuron backend. The fix:

1. Replace `apply_rotary_emb` with a real-arithmetic version (sin/cos
   decomposition instead of complex multiplication)
2. Same pattern as FLUX.2-klein's RoPE patch

This is ~1 hour of work (write the patch, monkey-patch before pipeline load).
Once patched, Z-Image (6B) fits on a single Neuron core and should run at
~2-5s per image with `torch.compile(backend="neuron")`.

## Model details

| | |
|---|---|
| Model | `Tongyi-MAI/Z-Image-Turbo` |
| Params | 6B (S3-DiT — Scalable Single-Stream DiT) |
| Scheduler | FlowMatchEulerDiscreteScheduler |
| Text encoder | Bundled (AutoModel) |
| VAE | AutoencoderKL |
| Inference | 8 steps (distilled turbo) |
| License | Apache-2.0 |

## Validation

- Date: 2026-06-14
- Instance: trn2.48xlarge (`i-0c2806a95b490e26e`, us-east-2)
- Venv: `/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/`
- diffusers 0.39.0.dev0

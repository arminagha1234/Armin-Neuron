# CLIP — Trainium Validation Results

Compiled with `torch_neuronx.trace` and run on a NeuronCore, compared against a CPU
reference forward.

| Checkpoint | Box | Output | max_abs_diff | cosine | Status |
|---|---|---|---|---|---|
| `openai/clip-vit-base-patch32` | trn2.48xlarge | logits_per_image | 1.7e-05 | 1.0000 | ✅ PASS |
| | | image_embeds | 3.3e-07 | 1.0000 | ✅ PASS |
| | | text_embeds | 2.7e-07 | 1.0000 | ✅ PASS |

- **Neuron SDK:** runtime 2.32, `torch-neuronx` on PyTorch 2.9
- **transformers:** 4.57
- No code surgery — standard ViT + text transformer attention.

Reproduce: `python src/port_clip.py --model openai/clip-vit-base-patch32`

# SigLIP — Trainium Validation Results

Compiled with `torch_neuronx.trace` and run on a NeuronCore, compared against a CPU
reference forward.

| Checkpoint | Box | Output | max_abs_diff | cosine | Status |
|---|---|---|---|---|---|
| `google/siglip-base-patch16-224` | trn2.48xlarge | logits_per_image | 6.7e-06 | 1.0000 | ✅ PASS |
| | | image_embeds | 6.3e-07 | 1.0000 | ✅ PASS |
| | | text_embeds | 6.1e-08 | 1.0000 | ✅ PASS |

- **Neuron SDK:** runtime 2.32, `torch-neuronx` on PyTorch 2.9
- **transformers:** 4.57 (+ `sentencepiece` for the tokenizer)
- No code surgery — standard ViT + text transformer attention.

Reproduce: `python src/port_siglip.py --model google/siglip-base-patch16-224`

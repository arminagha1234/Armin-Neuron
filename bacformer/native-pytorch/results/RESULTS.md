# Bacformer — Trainium Validation Results

Stage-2 genome encoder compiled with `torch_neuronx.trace` and run on a NeuronCore,
compared against a CPU reference forward. Stage-1 protein embeddings computed with
the plain HuggingFace `EsmModel` (no flash-attn).

| Checkpoint | Box | Output shape | max_abs_diff | cosine | Status |
|---|---|---|---|---|---|
| `macwiatrak/bacformer-masked-MAG` (26M) | trn2.48xlarge | (1, 18, 480) | 6.4e-06 | 1.0000 | ✅ PASS |

- **Neuron SDK:** runtime 2.32, `torch-neuronx` on PyTorch 2.9
- **transformers:** 4.57
- The encoder uses PyTorch SDPA (not flash-attn); only the ESM preprocessing helper
  needed de-CUDA-ing, done by using the plain HF `EsmModel`.

Reproduce: `python src/port_bacformer.py --n-proteins 16`

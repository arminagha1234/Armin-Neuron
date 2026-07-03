# ESM-2 — Trainium Validation Results

Compiled with `torch_neuronx.trace` and run on a NeuronCore, compared against a CPU
reference forward.

| Checkpoint | Box | Output shape | max_abs_diff | cosine | Status |
|---|---|---|---|---|---|
| `facebook/esm2_t6_8M_UR50D` | trn1.2xlarge | (2, 128, 320) | 1.6e-05 | 1.0000 | ✅ PASS |

- **Neuron SDK:** runtime 2.32, `torch-neuronx` on PyTorch 2.9
- **transformers:** 4.57
- No code surgery required — HuggingFace `EsmModel` uses an eager attention path.

Reproduce: `python src/port_esm2.py --model facebook/esm2_t6_8M_UR50D --seqlen 128`

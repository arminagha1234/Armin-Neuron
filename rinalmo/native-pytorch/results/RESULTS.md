# RiNALMo — Trainium Validation Results

Compiled with `torch_neuronx.trace` and run on a NeuronCore, compared against a CPU
reference forward.

| Checkpoint | Box | Output shape | max_abs_diff | cosine | Status |
|---|---|---|---|---|---|
| `multimolecule/rinalmo-micro` (30M) | trn2.48xlarge | (2, 128, 480) | 2.3e-05 | 1.0000 | ✅ PASS |
| `multimolecule/rinalmo-giga` (650M) | trn2.48xlarge | (2, 128, 1280) | 6.0e-05 | 1.0000 | ✅ PASS |

- **Neuron SDK:** runtime 2.32, `torch-neuronx` on PyTorch 2.9
- **transformers:** 5.12 (multimolecule needs `transformers.initialization`, 5.x only)
- Upstream flash-attn / `torch.cuda.amp` dependency avoided via the multimolecule
  eager/SDPA re-implementation.

Reproduce: `python src/port_rinalmo.py --model multimolecule/rinalmo-giga --seqlen 128`

# Chai-1 on Trainium — Native PyTorch (Beta 3) port

Porting [chai-lab](https://github.com/chaidiscovery/chai-lab) (an AlphaFold3-class
protein structure predictor) to AWS Trainium using **Native PyTorch** (the
`torch.device("neuron")` eager backend — no torch-xla, no tracing).

## Status

- **Working end-to-end on Trainium** via a hybrid: 5 of 6 model components run
  natively on the NeuronCore (including the compute-dominant diffusion module);
  the trunk runs on CPU due to a Beta 3 runtime bug. Output scores match the
  pure-CPU baseline. See `run_hybrid_neuron.py`.
- **Full native run of the small + diffusion components** — validated vs CPU
  (exact / bf16-clean). See `NATIVE_NEURON_RESULTS.md`.

## Key finding

chai-lab ships its network as **pre-traced TorchScript artifacts** (no eager
source), so this is a pure device-placement port — no model rewrite. 5/6
components "just work" on `torch.device("neuron")`. The **trunk** hits a runtime
assertion (`tensor_set_slice`, a 32-bit size overflow, `mac_count = 2^32`) that
we traced to a **fused NEFF** — see `FINE_GRAIN_FALLBACK_FINDINGS.md`. A minimal
repro for the Neuron team is in `trunk_repro/`.

## Files

| File | Purpose |
|---|---|
| `PORT_PLAN.md` | Architecture analysis & portability assessment |
| `NATIVE_NEURON_RESULTS.md` | Per-component results + parity vs CPU |
| `FINE_GRAIN_FALLBACK_FINDINGS.md` | Fine-grained CPU-fallback investigation (fused-NEFF root cause) |
| `run_native_neuron.py` | Full pipeline on `device="neuron"` |
| `run_hybrid_neuron.py` | **Working** hybrid: trunk on CPU, rest on Neuron |
| `capture_all.py` | Capture per-component inputs + CPU reference outputs |
| `test_component_neuron.py` | Run one component on Neuron, compare to CPU |
| `test_trunk_*.py` | Trunk debugging (MSA truncation, freeze, fallback) |
| `trunk_repro/` | Minimal repro of the trunk runtime bug for the Neuron team |

## Environment

Native PyTorch Beta 3 — DLC `concourse-release-0461d3b`, driver
`aws-neuronx-dkms 2.28`, torch 2.11 + native `torch_neuronx`, neuronx-cc
2.0.253257, on trn2.48xlarge.

## Note

The trunk repro's input fixture (`trunk_inputs.pt`, ~581 MB) is **not committed**
(too large for git). Regenerate it with `capture_all.py`, or synthesize inputs
at the shapes documented in `trunk_repro/README.md`.

# Chai-1 trunk — Native PyTorch (Beta 3) runtime failure repro

## Summary

The chai-1 **trunk** TorchScript component fails on the Native PyTorch Beta 3
`neuron` device with:

```
python: /opt/workspace/KaenaRuntime/tdrv/tensor.c:185:
  tensor_set_slice: Assertion `(tensor_source->_size) >= (offset + size)' failed.
```

The **same module + same inputs run correctly on CPU**. The other 5 chai-1
components (feature_embedding, bond_loss_input_proj, token_embedder,
diffusion_module, confidence_head) all run correctly on the `neuron` device.

## Environment

| Item | Version |
|---|---|
| DLC | `421672808698.dkr.ecr.us-east-1.amazonaws.com/concourse-release-0461d3b` |
| driver | `aws-neuronx-dkms 2.28.0.0` |
| runtime-lib | `2.32.19.0` |
| torch | `2.11.0` |
| torch_neuronx | `2.11.3.0.1254+1dc9304c.dev` (native, PrivateUse1 "neuron") |
| neuronx-cc | `2.0.253257.0a0+fd6c623c` |
| nki | `0.4.0b4` |
| instance | trn2.48xlarge (us-east-2) |

## Files

- `reproduce.py` — minimal standalone reproducer.
- `trunk_inputs.pt` — captured `forward_256` inputs from a real chai-1 CPU
  inference (token bucket 256). Deterministic.
- `trunk_neuron_debug.log` — runtime INFO log of the failing run (shows the
  NEFF load + `mac_count: 4294967296` right before the abort).

## Reproduce

```bash
# works
DEVICE=cpu    python reproduce.py

# fails with the assertion
DEVICE=neuron python reproduce.py
```

Set `TRUNK_PT=/path/to/trunk.pt` if the artifact lives elsewhere. The trunk is
downloaded by chai to `<site-packages>/downloads/models_v2/trunk.pt` on first
inference (chai ships no eager source — TorchScript only).

## Key diagnostic signal

The failing NEFF logs `mac_count: 4294967296` = exactly **2^32**. The trunk's
triangle ops at token bucket N=256 compute 256^4 = 2^32, pointing at a **32-bit
size/descriptor overflow** in the runtime's `tensor_set_slice` path for that op.

## What was ruled out (all reproduce the same assertion)

- `NEURON_CC_FLAGS="--hbm-scratchpad-page-size=1024"` + `NEURON_SCRATCHPAD_PAGE_SIZE=1024`
- `TORCH_NEURONX_ENABLE_ASYNC_NRT=0`
- `torch.use_deterministic_algorithms(True)` (unfused eager)
- MSA depth truncation 16384 → 512 (rules out tensor-size-driven overflow)
- `torch.jit.freeze` (device-first) — **moved the abort deeper** (compiled more
  blocks) but same assertion → confirms a specific deep op, not setup
- `torch.compile(backend="neuron")` — ScriptModule is opaque to dynamo → eager
  fallback → same assertion
- forcing `.contiguous()` on all inputs

## Impact / ask

Blocks a fully-on-Neuron chai-1 (AlphaFold3-class) inference. A CPU-trunk /
Neuron-rest hybrid works end-to-end and matches CPU scores, but the trunk is the
compute-heavy pairformer we want on the accelerator. Requesting a fix to the
runtime `tensor_set_slice` / 32-bit descriptor path for this op.

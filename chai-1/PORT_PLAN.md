# Chai-1 → Neuron Port Assessment

_Assessed 2026-07-02 on trn2.48xlarge `i-038786075e02b9697` (us-east-2), Neuron
DLAMI venv `/opt/aws_neuronx_venv_pytorch_2_9` (torch 2.9.1 + torch_neuronx)._

## TL;DR — how much can we port?

**The compute-heavy core (~1.2 GB of model, ~all the FLOPs) is a tractable
Neuron port. The surrounding pipeline stays on CPU as-is.** This is a
better-than-average porting candidate because the model uses a tiny, standard
op vocabulary, has **no data-dependent control flow**, and already runs at
**fixed bucketed shapes**.

## What chai-1 actually is (the key structural fact)

The open-source `chai_lab` repo contains **zero `nn.Module` definitions**. The
network ships as **pre-traced TorchScript `.pt` artifacts** downloaded at
runtime. The Python repo is only: input featurization (CPU), the orchestration
loop (`chai1.py`), and output writing (mmCIF).

So we do **not** rewrite the model in eager PyTorch. The port path is:
**`torch_neuronx.trace()` each downloaded TorchScript component with
representative fixed-shape example inputs → NEFF per component → swap
`load_exported()` to load the Neuron-compiled module.**

## Component inventory (downloaded TorchScript)

| Component | Size | prim::If | prim::Loop | Role | Port |
|---|---|---|---|---|---|
| `trunk.pt` | 633 MB | 0 | 0 | Pairformer trunk (recycled Nx) | **primary target** |
| `diffusion_module.pt` | 477 MB | 0 | 0 | Denoiser (called per diffusion step) | **primary target** |
| `confidence_head.pt` | 55 MB | 0 | 0 | pTM/ipTM/pLDDT | easy |
| `token_embedder.pt` | 6.3 MB | 0 | 0 | token input embed | easy |
| `feature_embedding.pt` | 4.6 MB | 0 | 0 | feature embed | easy |
| `bond_loss_input_proj.pt` | ~0 MB | 0 | 0 | bond feature proj | trivial |

Total ~1.2 GB — fits comfortably in a single NeuronCore's 96 GB.

## Op vocabulary (from TorchScript graph introspection)

- **trunk**: `einsum` (heavy — triangle attention/multiplication), `layer_norm`,
  `linear`, `silu`, `chunk`, `cat`, `to`, `mul`. 20 unique op types.
- **diffusion_module**: `linear` (dominant), `layer_norm`, `sigmoid`, `silu`,
  `einsum`, `reshape`, `chunk`. 24 unique op types.
- **confidence_head**: `einsum`, `linear`, `layer_norm`. 7 unique op types.

**No custom CUDA ops, no Triton, no custom flash-attention, no exotic aten
ops.** Everything here is supported by torch-xla / Neuron. `einsum` is the op to
watch for lowering efficiency (it's the AF3-style triangle ops), but it is
supported.

## Why this is Neuron-friendly

1. **Fixed shapes.** Inputs are padded to buckets
   `[256, 384, 512, 768, 1024, 1536, 2048]` tokens
   (`chai_lab/data/collate/utils.py`). Static shapes = one NEFF per bucket, no
   dynamic-shape recompiles.
2. **No control flow.** `prim::If = 0`, `prim::Loop = 0` across all components →
   straight-line graphs, exactly what XLA/Neuron trace wants.
3. **Recycling / diffusion loops live in Python** (`chai1.py`), calling the
   compiled module repeatedly → compile once, run many. The diffusion module is
   invoked ~200× per prediction, so accelerating it has the biggest payoff.

## What stays on CPU (no port needed, ~free)

- Input parsing / featurization (`chai_lab/data/**`): numpy/rdkit/torch on CPU.
- mmCIF output writing.
- **ESM embeddings** (`use_esm_embeddings`): a separate HuggingFace ESM2 model,
  hardcoded to `cuda:0` in `esm.py`. Either run on CPU, disable, or port
  separately. Not part of the chai core.

## Known edges / risks

- **`einsum` lowering**: functionally supported; watch performance on the
  triangle-multiplication paths in the trunk. May want to rewrite hot einsums as
  explicit matmuls if the compiler doesn't fuse well.
- **`.cuda()` hardcoding** in `chai_lab/tools/rigid.py` (templates path) and
  `cuda:0` in `chai1.py`/`esm.py` → patch device plumbing to `xm.xla_device()`.
- **bf16/precision**: trunk runs bf16, diffusion casts to fp32 (`.float()`).
  Keep the same cast points; validate numerics vs CPU reference.
- **TorchScript → Neuron trace**: `torch_neuronx.trace` should accept the
  ScriptModule + example inputs, but if a specific submodule op refuses to
  lower, that submodule is the fallback-to-CPU boundary (partitioned execution).

## Baseline already established

Native PyTorch chai-1 **runs end-to-end on this box's CPU** (peptide `GAAL`,
1 recycle / 2 diffusion steps → 5 `.cif` outputs, DONE_OK). That's the
functional reference to validate the Neuron-compiled version against.

## Proposed port order

1. **`confidence_head.pt`** — smallest, cleanest (7 ops) → prove the
   `torch_neuronx.trace(scripted, example_inputs)` flow works on a real chai
   component. Capture the example input shapes from a CPU run.
2. **`diffusion_module.pt`** — highest ROI (called ~200×/prediction). Trace at
   one bucket (e.g. 384), validate output vs CPU, then benchmark.
3. **`trunk.pt`** — biggest/most einsum-heavy; trace + validate, watch einsum
   perf.
4. Wire compiled components into `load_exported()` behind a `device="xla"` /
   `NEURON=1` switch; keep CPU fallback.
5. Extend to all buckets; end-to-end parity check on the example FASTA against
   the CPU baseline (RMSD / score match).

## Instance state

- chai-lab installed in the Neuron venv; TorchScript artifacts cached at
  `/opt/aws_neuronx_venv_pytorch_2_9/lib/python3.12/site-packages/downloads/models_v2/`.
- CPU baseline outputs at `/home/ubuntu/chai_out/`.
- Instance left running for iteration.

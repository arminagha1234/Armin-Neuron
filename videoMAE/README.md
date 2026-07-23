# VideoMAE v2 on AWS Trainium2 — native PyTorch (TorchNeuron)

Working reference for training **`OpenGVLab/VideoMAEv2-Base`** on Trainium2 using the
**native PyTorch backend (TorchNeuron)** — `torch.device("neuron")` eager mode and
`torch.compile(backend="neuron")`, **no `torch_xla` / no `xm.mark_step`**, and *not*
NxD Inference / NxD Training.

VideoMAE v2 ships on HuggingFace as `custom_code` (a raw ViT-B/16 video encoder behind
`trust_remote_code`, not a stock `transformers` architecture) and the real training code
lives in the [OpenGVLab/VideoMAEv2](https://github.com/OpenGVLab/VideoMAEv2) repo
(DeepSpeed + decord). This example vendors a **self-contained, dependency-light** version
(no timm/DeepSpeed/flash-attn/custom-CUDA) and demonstrates the full ladder on-device:

1. **Forward** parity vs CPU,
2. **Fine-tune** step (classifier head on the pretrained backbone),
3. **Full self-supervised pretraining** — the real objective (encoder-on-visible +
   decoder + tube masking + per-patch-normalized pixel reconstruction), **all params**,
4. **Multi-core** versions of (2) and (3) via PyTorch **FSDP2** (`fully_shard`).

## Status

Validated on `trn2.3xlarge` (1 Neuron device / 4 NeuronCores, LNC2) under the
Native PyTorch **Beta 3** DLC (torch 2.11 + torch-neuronx 2.11.3 + neuronx-cc 2.25).

| Capability | Result |
|---|---|
| Forward (neuron eager) vs CPU | max abs diff **8.3e-7** (fp32 — numerically exact) |
| Conv3d tubelet patch-embed | **lowers natively** — no rewrite needed |
| Fine-tune step (single core) | loss **6.10 → 0.91** / 10 steps; all grads flow |
| Fine-tune FSDP2 (2 cores) | params sharded **43.3M/rank**; loss 6.12 → 3.38 |
| **Pretraining** (single core, real objective, 101.3M params) | loss **1.327 → 1.001** / 40 steps; neuron matches CPU to ~6 decimals |
| **Pretraining FSDP2** (2 cores) | params sharded **50.65M/rank**; loss 1.328 → 1.007 |

> **What this proves:** the full VideoMAE v2 training *path* runs correctly and stably on
> Trainium (compiles, all params update, loss falls monotonically, numerics match CPU).
> It is **not** a converged/pretrained checkpoint — runs use synthetic structured video
> for a small number of steps. Real pretraining needs a real video dataset + dataloader
> and many thousands of steps. See [Caveats](#caveats).

## The one Neuron-specific change

Fine-tuning needs **zero** model changes. Pretraining needs exactly one: the reference
selects visible/masked tokens with **boolean `masked_select`** (`x[~mask]`), which is a
data-dependent (dynamic-shape) op that compiled accelerators (Neuron, and XLA/TPU) don't
support. We replace it with **static `torch.gather`** on integer keep/mask indices
(`ids_keep` / `ids_mask`) computed host-side. Tube masking keeps the visible/masked counts
identical across samples, so this is exact and fully static — the standard "MAE-on-XLA"
trick, ~15 lines. Everything else (Conv3d, attention, decoder, LayerNorm) is unchanged.

## Performance

Full throughput sweep (eager vs `torch.compile`, fp32 vs bf16, batch 1–8) with methodology
is in [BENCHMARK_TRN2_3XL.md](./BENCHMARK_TRN2_3XL.md). Headline for the pretraining step,
single NeuronCore:

- **`torch.compile(backend="neuron")` beats eager 1.45–1.77×.**
- Best: **bf16 + batch 8 + compile = 15.57 videos/s**, i.e. **~3.4×** over the eager/fp32/
  batch-1 baseline (4.54) — from dtype + batch + compile alone, **no custom kernels**.
- bf16 roughly halves memory vs fp32; compile keeps scaling with batch where eager plateaued.

Beta note: compiling more than one input shape in a single process fails (dynamic-shape
recompile is rejected) — use `dynamic=False` + one process per shape, or fixed-shape buckets
in a real training loop. See the benchmark doc's caveat.

## Quickstart

Requires the **Native PyTorch Beta DLC** (private beta; obtain via your AWS account team).
Set up the `native_venv` per the beta guide, then:

```bash
source ~/workspace/native_venv/bin/activate
pip install einops safetensors huggingface_hub   # numpy/torch already in the beta venv
cd src

# 1. Forward parity (neuron vs CPU) on the real pretrained weights
python forward_neuron.py

# 2. Fine-tune step (single NeuronCore)
python train_step_neuron.py

# 3. Full self-supervised pretraining (single NeuronCore, all 101M params)
python pretrain_neuron.py --device neuron --steps 40

# 4. Multi-core FSDP2 (2 logical cores under LNC2)
bash run_fsdp.sh            # fine-tune, FSDP2
bash run_fsdp_pretrain.sh   # pretraining, FSDP2

# 5. Throughput benchmark (batch x dtype; add --compile for torch.compile)
python bench_pretrain.py --device neuron --batches 1,2,4,8 --dtypes fp32,bf16
python bench_pretrain.py --device neuron --batches 4,8 --dtypes bf16 --compile
```

First step of any run includes NEFF compilation (single-core pretraining step 0 ≈ 97 s);
subsequent steps hit the persistent NEFF cache (~0.5 s/step).

## Layout

```
src/
  modeling_videomaev2_native.py   # vendored ViT-B/16 encoder (loads model.safetensors, strict)
  modeling_pretrain_native.py     # pretraining model: encoder+decoder+mask token+tube masking (gather-based)
  download_vmae2.py               # pull OpenGVLab/VideoMAEv2-Base weights from HF
  forward_neuron.py               # forward parity: neuron vs CPU
  train_step_neuron.py            # single-core fine-tune step
  train_fsdp_neuron.py            # multi-core FSDP2 fine-tune
  pretrain_neuron.py              # single-core REAL pretraining (structured synthetic video)
  train_fsdp_pretrain.py          # multi-core FSDP2 pretraining
  bench_pretrain.py               # throughput bench (batch x dtype x eager/compile)
  run_fsdp.sh / run_fsdp_pretrain.sh
logs/                             # captured run + benchmark logs (evidence)
BENCHMARK_TRN2_3XL.md
NOTICE
```

## Caveats

- **Private beta software.** Uses the Native PyTorch (TorchNeuron) Beta DLC and a
  pre-release Neuron driver. Not GA; share beta feedback with the Neuron team only.
- **Synthetic data.** Training runs use generated structured video (drifting sinusoidal
  gratings) so a falling loss reflects genuine optimization rather than memorization. This
  is a *training-path* validation, not a trained model. Wire a real decord dataloader +
  dataset for actual pretraining/fine-tuning.
- **Dual masking off.** We run standard full-decode MAE. VideoMAE v2's dual masking (a
  decoder-side efficiency trick) is an available extension, not required for correctness.
- **`cc-by-nc-4.0`** on the released `VideoMAEv2-Base` weights (non-commercial). Fine for
  a PoC; flag for production. Pretraining from scratch avoids the weight license.
- Benign beta notes: int64→int32 label autocast warning; fp64→fp32 downcast.

## Roadmap

- Real video dataloader (decord) + a downstream fine-tune with accuracy numbers.
- bf16 as the default training dtype (measured ~half memory, faster at batch ≥ 4).
- VideoMAE v2 dual masking (algorithmically shrinks the decoder token count).
- **NKI decoder-attention kernel** (see BENCHMARK) — attention is ~29% of training FLOPs
  (decoder-heavy, 1568 tokens). A *flash-style fwd+bwd* kernel could help, but note the
  sibling qwen result where a naive NKI attention kernel was **20% slower** than the
  eager path `neuronx-cc` auto-fuses — profile first, kernel second.

## License

Apache 2.0 (this adapter). VideoMAE v2 weights © OpenGVLab, `cc-by-nc-4.0`. See `NOTICE`.

# Qwen3.5-4B training on Trainium2 — native PyTorch (TorchNeuron beta)

End-to-end **training** of Qwen3.5-4B (hybrid **GatedDeltaNet + GQA**, 32 layers = 24 GDN + 8 full-attention,
4.21B params) on a single `trn2.48xlarge` in **native PyTorch** — `torch.device("neuron")`, eager →
`torch.compile(backend="neuron")`, `loss.backward()`, `optimizer.step()`. **No XLA, no NxDI, no vLLM.** bf16.
Both **LoRA** and **full fine-tune** (via FSDP) train on device.

Ships a **custom trainable GatedDeltaNet (GDN) NKI kernel** (forward + backward autograd) — the GDN
linear-attention layers are the hard part of this model on any accelerator, and the stock implementation
cannot lower under `torch.compile` on Neuron. Ours can.

## Results (measured on trn2.48xlarge, native-PyTorch Neuron beta DLC, bf16)
Full detail in [`docs/THROUGHPUT_FINAL.md`](docs/THROUGHPUT_FINAL.md) and
[`docs/QWEN35_4B_TRAINING_MEASURED.md`](docs/QWEN35_4B_TRAINING_MEASURED.md).

- ✅ **Trains end-to-end:** eager + LoRA + full fine-tune, loss decreasing, fwd + bwd + AdamW on device
  (full-FT via FSDP: loss 14.2 → 10.7).
- ✅ **Custom GDN kernel:** ~**2.5× faster** step (2.50 s vs 6.2 s), compiles under `torch.compile` where the
  stock kernel cannot, numerically correct at full 32L (forward parity cos ≈ 1.0 vs HF).
- ✅ **Multi-core FSDP:** 4 ranks (LNC2) → 8 ranks (LNC1, FSDP2 `fully_shard`) + activation checkpointing.
- ✅ **Headline: 883.8 tok/s** (8-rank + NKI kernel + activation-ckpt, full-32L, finite).

**Read the headline honestly — it's two stacked levers, not one:**
| Lever | Factor | From → to |
|---|---|---|
| GDN NKI kernel (single core) | ~2.5× | 83 → 205 tok/s |
| 8-core FSDP scaling (sub-linear) | ~4.3× | 205 → 883.8 tok/s |
| **Total vs single-core stock** | **~10.6×** | 83 → 883.8 tok/s |

Most of the 10.6× is using 8 cores instead of 1; the **software/kernel win is the ~2.5×**.

### Honest caveat — MFU ~3.5%
Peak MFU is **~3.5%** (8-rank), well below the ~20–25% good-Trn2 target. Root causes, all documented:
`bs=1` starves the tensor engine (the real MFU lever), but bigger batch is **memory-gated** at LNC1's
24 GB/core; GDN linear-attention is inherently low arithmetic intensity; and this is a beta stack. There is
real headroom — see the batch/MFU sweep in `docs/THROUGHPUT_FINAL.md`. Separately, 16/32-rank scaling is
currently blocked by a Neuron collective-topology constraint (AWS-side, pre-GA).

## Layout
- **`bench/`** — training + benchmark scripts: `qwen35_train_bench2.py` (core eager/compile training bench),
  `qwen35_fsdp_bench.py` (multi-core FSDP), `nki_full32_val.py` (full-32L NKI-kernel run), `integrated3.py`.
- **`gdn_kernel/`** — the kernel IP: `chunked_gdn.py` (chunked forward + block inverse), `gdn_nki_fwd.py` /
  `gdn_nki_bwd.py` / `chunked_gdn_nki.py` (NKI fwd/bwd autograd), `chunked_gdn_bwd_ref.py` (VJP oracle), and
  parity / device-probe tests.
- **`docs/`** — measured results (`THROUGHPUT_FINAL.md`, `QWEN35_4B_TRAINING_MEASURED.md`) and the multi-core
  reproduction recipe (`FSDP_RECIPE.md`).

## Run (inside the native-PyTorch Neuron beta DLC container, model at `/work/Qwen3.5-4B`)
```bash
# single-core, full-32L LoRA with the NKI GDN kernel (fast + finite, ~2.5 s/step)
export QWEN35_GDN_CHUNKED=1 NEURON_CC_FLAGS="--model-type=transformer --auto-cast=none"
python3 bench/nki_full32_val.py

# core training bench (eager; add --compile for torch.compile(backend=neuron))
python3 bench/qwen35_train_bench2.py --model /work/Qwen3.5-4B --seq 512 --bs 1 --steps 6 --lora --lora_r 16

# 8-rank FSDP (headline throughput) — see docs/FSDP_RECIPE.md
NEURON_RT_VIRTUAL_CORE_SIZE=1 torchrun --nproc_per_node 8 bench/qwen35_fsdp_bench.py --lora --seq 512
```

## Stack
Native-PyTorch Neuron beta DLC (PyTorch 2.11 + torch-neuronx + neuronx-cc + nki), bf16,
`attn_implementation="eager"`, transformers 5.14.1, peft 0.19.1. `torch.device("neuron")` — no `torch_xla`,
no `neuronx_distributed`.

## License
Apache-2.0. Qwen3.5-4B weights © Alibaba (Apache-2.0).

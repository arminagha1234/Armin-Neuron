# Llama on AWS Trainium — Native PyTorch (TorchNeuron)

Student-friendly, **validated** examples for running Meta's Llama models on AWS
Trainium using the **native PyTorch backend (TorchNeuron)** — plain
`torch.nn.Module`, `device="neuron"`, eager + `torch.compile`, **no XLA**, no
`mark_step`. The model stays a normal PyTorch model; the dispatcher routes ops
to Trainium.

If you've used a GPU in PyTorch, this will feel familiar — the only change is
`torch.device("neuron")` instead of `"cuda"`.

> **Native TorchNeuron is currently a closed beta.** These scripts run inside the
> TorchNeuron container. Request access through your AWS account team. See
> [`SETUP.md`](./SETUP.md) and the official docs:
> https://awsdocs-neuron.readthedocs-hosted.com/en/latest/frameworks/torch/pytorch-native-overview.html

## What's here

| Example | Model | Status | Notes |
|---|---|---|---|
| [`training-smoke-test/`](./training-smoke-test/) | tiny GPT (from scratch) | ✅ works | 30-line training loop — verify your Trainium can train before you touch Llama |
| [`llama-7b/`](./llama-7b/) | LLaMA-1 7B (`huggyllama/llama-7b`) | ✅ **validated** | Single NeuronCore inference; **100%** token agreement vs CPU fp32 |
| [`llama-3.1-8b/`](./llama-3.1-8b/) | Llama-3.1-8B (`meta-llama/Llama-3.1-8B`) | ✅ **validated** (TP=2) | 2-core tensor parallelism; **100%** token agreement vs CPU fp32 |
| [`llama-2-13b-chat/`](./llama-2-13b-chat/) | Llama-2-13b-chat (`meta-llama/Llama-2-13b-chat-hf`) | ⚠️ needs Trn2 | 13B needs TP≥4 (cross-chip); OOMs at TP=2, and cross-chip TP is blocked on the Trn1 beta — run on Trn2 |

## The one thing to know about memory (read this first)

Each Trainium1 (Trn1) NeuronCore has **16 GB of HBM** (~14 GB usable after runtime
overhead). That determines what fits on a single core:

| Model | bf16 weights | Fits on 1 Trn1 core? |
|---|---:|---|
| LLaMA-1 **7B** | ~13.5 GB | ✅ yes (tight) → run single-core |
| Llama-3.1 **8B** | ~16 GB | ❌ no → needs **tensor parallelism** across ≥2 cores (or a Trn2, which has more HBM/core) |

This is why `llama-7b/` is a simple single-core script and `llama-3.1-8b/` is a
`torchrun` tensor-parallel script. Quantization (int8/fp8) isn't available in the
current beta, so the fix for big models is more cores, not smaller weights.

## Quickstart

1. Get a Trainium instance and the TorchNeuron container running — see [`SETUP.md`](./SETUP.md).
2. Sanity-check training works on your box:
   ```bash
   python3 training-smoke-test/train_smoke.py
   ```
3. Run Llama-1 7B inference on a single core:
   ```bash
   python3 llama-7b/run_native.py --model huggyllama/llama-7b --prompt "The capital of France is"
   ```
4. (Optional) Validate numerics vs a CPU reference:
   ```bash
   python3 llama-7b/validate.py huggyllama/llama-7b
   ```

The **first run compiles a NEFF** (tens of seconds to a few minutes for these
models). Subsequent runs hit the persistent NEFF cache and are fast.

## Model access

- `huggyllama/llama-7b` is a public re-upload of the original LLaMA-1 7B weights — no gating.
- `meta-llama/Llama-3.1-8B` is **gated**: accept the license on its Hugging Face page and
  log in (`hf auth login`) before downloading.

## Validation approach

Each port is checked by a **teacher-forced, per-position agreement** test: feed a fixed
prompt, compare the argmax (top-1 next-token) of `neuron bf16` logits against a
`cpu fp32` reference at every position. ≥95% agreement = port is faithful. LLaMA-1 7B
scores **100%**. (See [`neuron-framework-equivalence`] methodology for the rigorous
R-ratio version.)

## License

Example code here is Apache 2.0. Llama weights are © Meta under their respective
community licenses — you are responsible for accepting them.

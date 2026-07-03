# Running RiNALMo on AWS Trainium — Step-by-Step Guide

Copy-paste guide to run **RiNALMo** (the 650M RNA language model) on **AWS Trainium**
in native PyTorch. RiNALMo produces per-nucleotide embeddings used for secondary-
structure prediction, splice-site prediction, and mean-ribosome-loading tasks.

> **Status:** ✅ **RiNALMo working on Trainium** — cosine **1.0000** at both the 30M
> "micro" and the **650M "giga"** tiers vs a CPU reference.

---

## The porting problem

The upstream repo requires `pip install flash-attn==2.3.2` (CUDA-only) and its
inference path uses `torch.cuda.amp.autocast()` on `cuda:0`. Neither runs on Neuron.
This port loads RiNALMo through the
[`multimolecule`](https://huggingface.co/multimolecule/rinalmo-giga) re-implementation
— a BERT-style encoder (33 layers / 1280 hidden / 20 heads at the 650M tier) with
eager / SDPA attention, no flash-attn.

## 1. Instance

650M ≈ 1.3 GB bf16 — a **single NeuronCore** (any Trn2 slice).

## 2. Environment

```bash
source /opt/aws_neuronx_venv_pytorch_2_9/bin/activate
pip install -r requirements.txt
```

> **Dependency note:** `multimolecule` imports `transformers.initialization`, which
> only exists in **transformers >= 5.x**. The other models in this repo were
> validated on transformers 4.57; RiNALMo needs 5.x. Use a dedicated venv if you want
> to keep 4.57 for the others.

## 3. Run

```bash
python src/port_rinalmo.py --model multimolecule/rinalmo-giga --seqlen 128
# smaller: multimolecule/rinalmo-mega (150M), multimolecule/rinalmo-micro (30M)
```

## What the script does

1. Loads the model + RNA tokenizer, pads sequences to a fixed length.
2. Runs a CPU reference forward.
3. `torch_neuronx.trace(...)` compiles to a NeuronCore.
4. Reports max-abs-diff + cosine vs CPU on `last_hidden_state`.
5. Saves `rinalmo_neuron.pt`.

See [results/RESULTS.md](results/RESULTS.md).

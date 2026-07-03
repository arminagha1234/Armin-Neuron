# Running ESM-2 on AWS Trainium — Step-by-Step Guide

Copy-paste guide to run **ESM-2** (Meta AI's protein language model) on **AWS
Trainium** in native PyTorch. ESM-2 produces per-residue embeddings used for
structure prediction, variant-effect scoring, and downstream protein classifiers.

**You do not need to know anything about Trainium or Neuron.** Every command is
copy-paste.

> **Status:** ✅ **ESM-2 working on Trainium** — cosine **1.0000**, max-abs-diff
> ~1.6e-05 vs a CPU reference. The 8M–650M tiers fit on a single Trainium chip.

---

## 1. Instance

ESM-2 encoder tiers (8M–650M) run on a **single Trainium chip**, so the smallest
slice is enough — a **trn1.2xlarge** or **trn2.3xlarge** (not a 48xlarge). The 3B/15B
tiers also fit on one Trn2 chip (~96 GB HBM) but compile longer.

## 2. Environment

Use the Neuron DLAMI PyTorch venv:

```bash
source /opt/aws_neuronx_venv_pytorch_2_9/bin/activate
pip install -r requirements.txt      # transformers
```

## 3. Run

```bash
python src/port_esm2.py --model facebook/esm2_t6_8M_UR50D --seqlen 128
```

Swap `--model` for any tier:

| Checkpoint | Params |
|---|---|
| `facebook/esm2_t6_8M_UR50D` | 8M |
| `facebook/esm2_t12_35M_UR50D` | 35M |
| `facebook/esm2_t30_150M_UR50D` | 150M |
| `facebook/esm2_t33_650M_UR50D` | 650M |

## What the script does

1. Loads the model + tokenizer, pads a batch of protein sequences to a fixed length
   (Neuron needs static shapes).
2. Runs a CPU reference forward.
3. `torch_neuronx.trace(...)` compiles the graph to a NeuronCore.
4. Runs on Neuron and reports max-abs-diff + cosine vs CPU.
5. Saves `esm2_neuron.pt`.

See [results/RESULTS.md](results/RESULTS.md) for validated numbers.

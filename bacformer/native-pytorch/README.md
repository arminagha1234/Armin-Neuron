# Running Bacformer on AWS Trainium — Step-by-Step Guide

Copy-paste guide to run **Bacformer** (a bacterial-genome foundation model) on **AWS
Trainium** in native PyTorch. Bacformer contextualises a genome's proteins for
strain clustering, essential-gene prediction, operon identification, and more.

> **Status:** ✅ **Bacformer working on Trainium** — Stage-2 encoder cosine
> **1.0000** vs a CPU reference.

---

## Two stages

1. **Protein embedding (ESM-2):** each protein is embedded to one vector. Bacformer
   *base* uses `esm2_t12_35M_UR50D` (480-d).
2. **Genome transformer (Bacformer encoder):** contextualises the sequence of
   per-protein embeddings along the chromosome/plasmid.

## The porting problem

The upstream `protein_seqs_to_bacformer_inputs` helper computes Stage-1 embeddings
through [`faesm`](https://github.com/pengzhangzhi/faplm), which requires CUDA
`flash-attn`. This port **bypasses** it and runs Stage 1 with the plain HuggingFace
`EsmModel` (mean-pooled per protein). The Bacformer encoder's own `trust_remote_code`
modeling file already uses `torch.nn.functional.scaled_dot_product_attention`, so
Stage 2 traces directly.

## 1. Instance

Base model is 26M — a **single NeuronCore** (any Trn2 slice).

## 2. Environment

```bash
source /opt/aws_neuronx_venv_pytorch_2_9/bin/activate
pip install -r requirements.txt      # transformers
```

## 3. Run

```bash
python src/port_bacformer.py --n-proteins 16
```

The script embeds a toy genome with ESM-2, builds the Bacformer input sequence
(`[CLS] prot... [END]`, special-token ids from `configuration_bacformer.py`),
compiles the Stage-2 encoder, and compares CPU vs Neuron `last_hidden_state`.

See [results/RESULTS.md](results/RESULTS.md).

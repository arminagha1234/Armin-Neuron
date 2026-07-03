# ESM-2 on Trainium

[ESM-2](https://github.com/facebookresearch/esm) (Meta AI) — the protein language
model family — running on AWS Trainium in native PyTorch.

- **[native-pytorch/](native-pytorch/)** — script + guide to compile ESM-2 with
  `torch_neuronx.trace` and validate it. Working with **cosine 1.0000** vs a CPU
  reference. Produces per-residue embeddings for protein sequences.

ESM-2 is a BERT-style encoder (`EsmModel` in HuggingFace `transformers`) with rotary
attention and no CUDA-specific code, so it traces to a NeuronCore with zero surgery.
Sizes 8M → 15B; the 8M–650M tiers run comfortably on a single chip.

> **Status:** ✅ validated on **trn1.2xlarge** (Trainium1) — the smallest/cheapest
> box is plenty for the encoder tiers.

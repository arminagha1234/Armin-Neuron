# Bacformer on Trainium

[Bacformer](https://github.com/macwiatrak/Bacformer) — a prokaryotic foundation model
that treats a whole bacterial genome as an ordered sequence of proteins — running on
AWS Trainium in native PyTorch.

- **[native-pytorch/](native-pytorch/)** — script + guide for the two-stage pipeline
  (ESM-2 protein embeddings → Bacformer genome transformer). Working with **cosine
  1.0000** vs a CPU reference on the Stage-2 encoder.

> **Status:** ✅ validated on **trn2.48xlarge** — base model is 26M, one NeuronCore.

## Porting note
The upstream `protein_seqs_to_bacformer_inputs` helper computes Stage-1 embeddings
through `faesm`, which needs CUDA `flash-attn`. This port bypasses it and runs Stage 1
with the plain HuggingFace `EsmModel` (see [`../esm2`](../esm2)). The Bacformer encoder
itself already uses `torch.nn.functional.scaled_dot_product_attention` (not
flash-attn), so Stage 2 traces to a NeuronCore directly.

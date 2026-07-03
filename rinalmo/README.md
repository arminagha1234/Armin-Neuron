# RiNALMo on Trainium

[RiNALMo](https://github.com/lbcb-sci/RiNALMo) — the largest RNA language model
(650M) — running on AWS Trainium in native PyTorch.

- **[native-pytorch/](native-pytorch/)** — script + guide to compile RiNALMo with
  `torch_neuronx.trace`. Working with **cosine 1.0000** vs a CPU reference at both
  the 30M and **650M** tiers. Produces per-nucleotide RNA embeddings.

> **Status:** ✅ validated on **trn2.48xlarge** — 30M and 650M tiers, one NeuronCore.

## Porting note
Upstream RiNALMo hard-requires `flash-attn==2.3.2` + `torch.cuda.amp` (CUDA-only).
This port loads RiNALMo through the [`multimolecule`](https://huggingface.co/multimolecule)
re-implementation — a BERT-style encoder with eager / SDPA attention, no flash-attn —
so it traces to a NeuronCore directly.

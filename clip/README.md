# CLIP on Trainium

[CLIP](https://github.com/openai/CLIP) (OpenAI) — the original contrastive
vision-language model — running on AWS Trainium in native PyTorch.

- **[native-pytorch/](native-pytorch/)** — script + guide to compile `CLIPModel`
  with `torch_neuronx.trace`. Working with **cosine 1.0000** vs a CPU reference for
  image embeds, text embeds, and logits.

Uses the HuggingFace `CLIPModel` implementation (ViT image encoder + text
transformer, eager attention), so it traces to a NeuronCore with zero surgery.

> **Status:** ✅ validated on **trn2.48xlarge** — one NeuronCore is enough.

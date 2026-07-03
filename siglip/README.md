# SigLIP / SigLIP2 on Trainium

[SigLIP](https://github.com/merveenoyan/siglip) (Google's sigmoid-loss CLIP) —
a vision-language model — running on AWS Trainium in native PyTorch.

- **[native-pytorch/](native-pytorch/)** — script + guide to compile `SiglipModel`
  with `torch_neuronx.trace`. Working with **cosine 1.0000** vs a CPU reference for
  image embeds, text embeds, and logits. Zero-shot image classification / retrieval.

The `merveenoyan/siglip` repo is a demo/search project wrapping the `google/siglip-*`
checkpoints in HuggingFace `transformers`; this port compiles the underlying model.
It's a ViT image encoder + text transformer with standard attention (the sigmoid is
only in the training loss), so inference traces directly.

> **Status:** ✅ validated on **trn2.48xlarge** — one NeuronCore is enough.

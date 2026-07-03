# Running SigLIP on AWS Trainium — Step-by-Step Guide

Copy-paste guide to run **SigLIP / SigLIP2** (Google's sigmoid-loss vision-language
model) on **AWS Trainium** in native PyTorch. SigLIP does zero-shot image
classification and image-text retrieval.

> **Status:** ✅ **SigLIP working on Trainium** — cosine **1.0000** on image
> embeds, text embeds, and logits vs a CPU reference.

---

## 1. Instance

A ViT-base + text transformer runs on a **single NeuronCore** — any Trn2 slice works
(this was validated on a trn2.48xlarge but does not need one).

## 2. Environment

```bash
source /opt/aws_neuronx_venv_pytorch_2_9/bin/activate
pip install -r requirements.txt      # transformers, sentencepiece, pillow, numpy
```

## 3. Run

```bash
python src/port_siglip.py --model google/siglip-base-patch16-224
# larger: google/siglip-so400m-patch14-384, google/siglip2-base-patch16-224
```

## What the script does

1. Loads `SiglipModel` + processor; builds a fixed-shape batch (text padded to 64
   tokens — the SigLIP default — image 224x224).
2. Traces a wrapper returning `logits_per_image`, `image_embeds`, `text_embeds`.
3. `torch_neuronx.trace(...)` compiles to a NeuronCore.
4. Reports max-abs-diff + cosine vs CPU for all three outputs.
5. Saves `siglip_neuron.pt`.

See [results/RESULTS.md](results/RESULTS.md).

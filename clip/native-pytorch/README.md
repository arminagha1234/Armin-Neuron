# Running CLIP on AWS Trainium — Step-by-Step Guide

Copy-paste guide to run **CLIP** (OpenAI's contrastive vision-language model) on
**AWS Trainium** in native PyTorch. CLIP does zero-shot image classification and
image-text retrieval.

> **Status:** ✅ **CLIP working on Trainium** — cosine **1.0000** on image embeds,
> text embeds, and logits vs a CPU reference.

---

## 1. Instance

ViT-B/32 (or L/14) runs on a **single NeuronCore** — any Trn2 slice works.

## 2. Environment

```bash
source /opt/aws_neuronx_venv_pytorch_2_9/bin/activate
pip install -r requirements.txt      # transformers, pillow, numpy
```

## 3. Run

```bash
python src/port_clip.py --model openai/clip-vit-base-patch32
# larger: openai/clip-vit-large-patch14
```

## What the script does

1. Loads `CLIPModel` + processor; builds a fixed-shape batch (text padded to 77
   tokens — the CLIP default — image 224x224).
2. Traces a wrapper returning `logits_per_image`, `image_embeds`, `text_embeds`.
3. `torch_neuronx.trace(...)` compiles to a NeuronCore.
4. Reports max-abs-diff + cosine vs CPU for all three outputs.
5. Saves `clip_neuron.pt`.

See [results/RESULTS.md](results/RESULTS.md).

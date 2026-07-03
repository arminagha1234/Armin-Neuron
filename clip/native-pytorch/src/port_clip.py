#!/usr/bin/env python
"""Port CLIP (vision-language) to AWS Trainium via torch-neuronx.

OpenAI CLIP as implemented in HuggingFace `transformers` (`CLIPModel`): a ViT
image encoder + text transformer, both eager attention. Traces cleanly.

Run on a Neuron box:
    source /opt/aws_neuronx_venv_pytorch_2_9/bin/activate
    pip install -r requirements.txt
    python port_clip.py --model openai/clip-vit-base-patch32
"""
import argparse
import torch
import torch_neuronx
from transformers import AutoModel, AutoProcessor


def cosine(a, b):
    a, b = a.flatten().float(), b.flatten().float()
    return torch.nn.functional.cosine_similarity(a, b, dim=0).item()


class ClipWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask, pixel_values):
        out = self.model(input_ids=input_ids, attention_mask=attention_mask,
                         pixel_values=pixel_values)
        return out.logits_per_image, out.image_embeds, out.text_embeds


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="openai/clip-vit-base-patch32")
    p.add_argument("--seqlen", type=int, default=77)
    p.add_argument("--out", default="clip_neuron.pt")
    args = p.parse_args()

    print(f"[clip] loading {args.model}")
    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model).eval()
    wrapper = ClipWrapper(model).eval()

    from PIL import Image
    import numpy as np
    img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    texts = ["a photo of a cat", "a photo of a dog"]
    inputs = processor(text=texts, images=[img, img], return_tensors="pt",
                       padding="max_length", max_length=args.seqlen, truncation=True)
    example = (inputs["input_ids"], inputs["attention_mask"], inputs["pixel_values"])

    print("[clip] CPU reference forward")
    with torch.no_grad():
        cpu_logits, cpu_img, cpu_txt = wrapper(*example)

    print("[clip] tracing with torch_neuronx (compiling)...")
    neuron_model = torch_neuronx.trace(wrapper, example)

    print("[clip] Neuron forward")
    neu_logits, neu_img, neu_txt = neuron_model(*example)

    for name, c, n in [("logits_per_image", cpu_logits, neu_logits),
                       ("image_embeds", cpu_img, neu_img),
                       ("text_embeds", cpu_txt, neu_txt)]:
        print(f"[clip] {name}: max_abs={float((c-n).abs().max()):.3e} "
              f"cosine={cosine(c, n):.6f}")

    torch.jit.save(neuron_model, args.out)
    print(f"[clip] saved compiled model -> {args.out}")
    assert cosine(cpu_img, neu_img) > 0.99, "image embed cosine too low"
    print("[clip] PASS")


if __name__ == "__main__":
    main()

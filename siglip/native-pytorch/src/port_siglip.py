#!/usr/bin/env python
"""Port SigLIP / SigLIP2 (vision-language) to AWS Trainium via torch-neuronx.

SigLIP is CLIP with a sigmoid loss; the model itself is a ViT image encoder + a
text transformer, both eager-attention in HuggingFace `transformers`
(`SiglipModel`). The sigmoid is only in the loss, so inference traces cleanly.

Run on a Neuron box:
    source /opt/aws_neuronx_venv_pytorch_2_9/bin/activate
    pip install -r requirements.txt
    python port_siglip.py --model google/siglip-base-patch16-224
"""
import argparse
import torch
import torch_neuronx
from transformers import AutoModel, AutoProcessor


def cosine(a, b):
    a, b = a.flatten().float(), b.flatten().float()
    return torch.nn.functional.cosine_similarity(a, b, dim=0).item()


class SiglipWrapper(torch.nn.Module):
    """Return only tensors (logits + embeds) so the graph traces cleanly."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, pixel_values):
        out = self.model(input_ids=input_ids, pixel_values=pixel_values)
        return out.logits_per_image, out.image_embeds, out.text_embeds


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="google/siglip-base-patch16-224")
    p.add_argument("--out", default="siglip_neuron.pt")
    args = p.parse_args()

    print(f"[siglip] loading {args.model}")
    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model).eval()
    wrapper = SiglipWrapper(model).eval()

    # SigLIP text pads to a fixed 64 tokens; image is 224x224.
    from PIL import Image
    import numpy as np
    img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    texts = ["a photo of a cat", "a photo of a dog"]
    inputs = processor(text=texts, images=[img, img], return_tensors="pt",
                       padding="max_length", max_length=64)
    example = (inputs["input_ids"], inputs["pixel_values"])

    print("[siglip] CPU reference forward")
    with torch.no_grad():
        cpu_logits, cpu_img, cpu_txt = wrapper(*example)

    print("[siglip] tracing with torch_neuronx (compiling)...")
    neuron_model = torch_neuronx.trace(wrapper, example)

    print("[siglip] Neuron forward")
    neu_logits, neu_img, neu_txt = neuron_model(*example)

    for name, c, n in [("logits_per_image", cpu_logits, neu_logits),
                       ("image_embeds", cpu_img, neu_img),
                       ("text_embeds", cpu_txt, neu_txt)]:
        print(f"[siglip] {name}: max_abs={float((c-n).abs().max()):.3e} "
              f"cosine={cosine(c, n):.6f}")

    torch.jit.save(neuron_model, args.out)
    print(f"[siglip] saved compiled model -> {args.out}")
    assert cosine(cpu_img, neu_img) > 0.99, "image embed cosine too low"
    print("[siglip] PASS")


if __name__ == "__main__":
    main()

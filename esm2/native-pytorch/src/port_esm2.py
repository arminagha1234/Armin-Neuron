#!/usr/bin/env python
"""Port ESM-2 (protein language model) to AWS Trainium via torch-neuronx.

ESM-2 is a BERT-style encoder in HuggingFace `transformers` (`EsmModel`) with an
eager attention path, so it traces cleanly with no CUDA / flash-attn.

Run on a Neuron box:
    source /opt/aws_neuronx_venv_pytorch_2_9/bin/activate
    pip install -r requirements.txt
    python port_esm2.py --model facebook/esm2_t6_8M_UR50D --seqlen 128
"""
import argparse
import torch
import torch_neuronx
from transformers import AutoTokenizer, EsmModel


def cosine(a, b):
    a, b = a.flatten().float(), b.flatten().float()
    return torch.nn.functional.cosine_similarity(a, b, dim=0).item()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="facebook/esm2_t6_8M_UR50D",
                   help="e.g. esm2_t6_8M / t12_35M / t30_150M / t33_650M_UR50D")
    p.add_argument("--seqlen", type=int, default=128)
    p.add_argument("--out", default="esm2_neuron.pt")
    args = p.parse_args()

    print(f"[esm2] loading {args.model}")
    tok = AutoTokenizer.from_pretrained(args.model)
    model = EsmModel.from_pretrained(args.model, torchscript=True).eval()

    seqs = [
        "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKR",
        "MSTNPKPQRKTKRNTNRRPQDVKFPGG",
    ]
    enc = tok(seqs, return_tensors="pt", padding="max_length",
              truncation=True, max_length=args.seqlen)
    example = (enc["input_ids"], enc["attention_mask"])

    print("[esm2] CPU reference forward")
    with torch.no_grad():
        cpu_out = model(*example)
    cpu_repr = cpu_out[0]  # last_hidden_state

    print("[esm2] tracing with torch_neuronx (compiling to NeuronCore)...")
    neuron_model = torch_neuronx.trace(model, example)

    print("[esm2] Neuron forward")
    neu_out = neuron_model(*example)
    neu_repr = neu_out[0]

    max_abs = (cpu_repr - neu_repr).abs().max().item()
    cos = cosine(cpu_repr, neu_repr)
    print(f"[esm2] last_hidden_state shape={tuple(cpu_repr.shape)}")
    print(f"[esm2] max_abs_diff={max_abs:.3e}  cosine={cos:.6f}")

    torch.jit.save(neuron_model, args.out)
    print(f"[esm2] saved compiled model -> {args.out}")
    assert cos > 0.99, "cosine similarity too low — port failed"
    print("[esm2] PASS")


if __name__ == "__main__":
    main()

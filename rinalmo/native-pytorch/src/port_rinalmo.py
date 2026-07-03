#!/usr/bin/env python
"""Port RiNALMo (RNA language model) to AWS Trainium via torch-neuronx.

The original lbcb-sci/RiNALMo repo hard-requires `flash-attn==2.3.2` and
`torch.cuda.amp`, which are CUDA-only and cannot run on Neuron. This port uses the
`multimolecule` HuggingFace re-implementation of RiNALMo, which is a standard
BERT-style encoder (rotary attention, eager / SDPA — no flash-attn), so it traces
to a NeuronCore directly.

Run on a Neuron box:
    source /opt/aws_neuronx_venv_pytorch_2_9/bin/activate
    pip install -r requirements.txt
    python port_rinalmo.py --model multimolecule/rinalmo --seqlen 128
"""
import argparse
import torch
import torch_neuronx


def cosine(a, b):
    a, b = a.flatten().float(), b.flatten().float()
    return torch.nn.functional.cosine_similarity(a, b, dim=0).item()


class RiNALMoWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        out = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return out.last_hidden_state


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="multimolecule/rinalmo-giga",
                   help="rinalmo-giga (650M) / rinalmo-mega (150M) / rinalmo-micro (30M)")
    p.add_argument("--seqlen", type=int, default=128)
    p.add_argument("--out", default="rinalmo_neuron.pt")
    args = p.parse_args()

    from multimolecule import RnaTokenizer, RiNALMoModel

    print(f"[rinalmo] loading {args.model}")
    tok = RnaTokenizer.from_pretrained(args.model)
    model = RiNALMoModel.from_pretrained(args.model).eval()
    wrapper = RiNALMoWrapper(model).eval()

    seqs = ["ACUUUGGCCAUGCUAGCUAGCUAGCUAGCUGACUACGUAGCUAGC",
            "CCCGGUACGUACGUACGUACGU"]
    enc = tok(seqs, return_tensors="pt", padding="max_length",
              truncation=True, max_length=args.seqlen)
    example = (enc["input_ids"], enc["attention_mask"])

    print("[rinalmo] CPU reference forward")
    with torch.no_grad():
        cpu_repr = wrapper(*example)

    print("[rinalmo] tracing with torch_neuronx (compiling)...")
    neuron_model = torch_neuronx.trace(wrapper, example)

    print("[rinalmo] Neuron forward")
    neu_repr = neuron_model(*example)

    print(f"[rinalmo] last_hidden_state shape={tuple(cpu_repr.shape)}")
    print(f"[rinalmo] max_abs_diff={float((cpu_repr-neu_repr).abs().max()):.3e} "
          f"cosine={cosine(cpu_repr, neu_repr):.6f}")

    torch.jit.save(neuron_model, args.out)
    print(f"[rinalmo] saved compiled model -> {args.out}")
    assert cosine(cpu_repr, neu_repr) > 0.99, "cosine too low — port failed"
    print("[rinalmo] PASS")


if __name__ == "__main__":
    main()

"""AlphaGenome on Trainium — simple inference wrapper.

For a researcher who just wants to run the model on a Trainium box. Handles the
two Neuron-specific details for you:
  1. Sets NEURON_CC_FLAGS=--optlevel=1 BEFORE the compiler runs (the default
     optimizer level crashes neuronx-cc at long sequence lengths via an internal
     polyhedral-analysis error; optlevel=1 avoids it).
  2. Skips the `splice_junctions` head, which needs a `sort` op that trn2 does
     not support. All other heads run on-device and match CPU to ~6 decimals.

Usage:
    # demo on a random 131,072-bp sequence
    python predict_alphagenome.py

    # your own one-hot input saved as .npy with shape (1, S, 4), S multiple of 128
    python predict_alphagenome.py --input my_seq.npy --organism 0 --out preds.pt

Programmatic:
    from predict_alphagenome import load_model, predict
    model = load_model()                       # loads + moves to NeuronCore
    out = predict(model, dna_onehot)            # dict of {head: {res: tensor}}
"""
import os
# MUST be set before torch_xla / neuronx-cc import so the compiler picks it up.
os.environ.setdefault("NEURON_CC_FLAGS", "--optlevel=1")

import argparse
import numpy as np
import torch
import torch_xla.core.xla_model as xm
from alphagenome_pytorch import AlphaGenome

WEIGHTS = os.environ.get("AG_WEIGHTS", "/scratch/alphagenome/hf/model_fold_0.safetensors")

# Heads that run on Neuron (everything except splice_junctions; see module docstring).
NEURON_HEADS = (
    "atac", "dnase", "procap", "cage", "rna_seq", "chip_tf", "chip_histone",
    "contact_maps", "splice_sites", "splice_site_usage",
)


def load_model(weights: str = WEIGHTS):
    dev = xm.xla_device()
    model = AlphaGenome.from_pretrained(weights, device="cpu").eval().to(dev)
    return model


def predict(model, dna_onehot: torch.Tensor, organism_index: int = 0, heads=NEURON_HEADS):
    """dna_onehot: (1, S, 4) float tensor (A=0,C=1,G=2,T=3). Returns CPU output dict."""
    dev = xm.xla_device()
    x = dna_onehot.to(dev)
    with torch.no_grad():
        out = model.predict(x, organism_index=organism_index, heads=heads)
    xm.mark_step()
    # Move tensors to CPU for downstream use.
    def to_cpu(v):
        if torch.is_tensor(v):
            return v.detach().to("cpu")
        if isinstance(v, dict):
            return {k: to_cpu(sv) for k, sv in v.items()}
        return v
    return {k: to_cpu(v) for k, v in out.items()}


def _demo_input(seq_len=131072, seed=1234):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, 4, size=(1, seq_len))
    return torch.from_numpy(np.eye(4, dtype=np.float32)[idx])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", help="path to .npy one-hot (1,S,4); omit for random demo")
    ap.add_argument("--organism", type=int, default=0, help="0=human, 1=mouse")
    ap.add_argument("--seq-len", type=int, default=131072, help="demo length (multiple of 128)")
    ap.add_argument("--out", default=None, help="optional .pt path to save predictions")
    args = ap.parse_args()

    if args.input:
        arr = np.load(args.input)
        x = torch.as_tensor(arr, dtype=torch.float32)
    else:
        x = _demo_input(args.seq_len)
    print(f"[ag] input {tuple(x.shape)} organism={args.organism}")

    model = load_model()
    print("[ag] first call compiles (minutes); subsequent calls are fast...")
    out = predict(model, x, organism_index=args.organism)

    print(f"[ag] done. {len(out)} heads:")
    for head, val in out.items():
        if isinstance(val, dict):
            for res, t in val.items():
                if torch.is_tensor(t):
                    print(f"    {head}@{res}: {tuple(t.shape)}")
        elif torch.is_tensor(val):
            print(f"    {head}: {tuple(val.shape)}")

    if args.out:
        torch.save(out, args.out)
        print(f"[ag] saved -> {args.out}")


if __name__ == "__main__":
    main()

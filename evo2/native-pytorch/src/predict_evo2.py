"""Evo2 (StripedHyena2) on Trainium — simple inference wrapper.

For a researcher who just wants to run Evo2 on a Trainium box. Handles the
Neuron-specific details automatically:
  1. use_fp8_input_projections=False  -> pure-PyTorch projections (no TransformerEngine).
  2. attn_implementation="eager"      -> pure-PyTorch attention (no flash-attn).
  3. FFT-conv -> conv1d patch          -> removes complex ops neuronx-cc can't compile.
  4. model.float()                     -> fp32, avoids a bf16 norm-collapse on the
                                          model's massive activations.
  5. use_cache=False                   -> static prefill graph.

Usage:
    # embeddings for a DNA string
    python predict_evo2.py --seq ACGTACGTACGT --mode embed

    # next-token logits
    python predict_evo2.py --seq ACGT --mode logits --out logits.pt

Programmatic:
    from predict_evo2 import load_model, embed, logits
    tok, model = load_model()
    emb = embed(tok, model, "ACGTACGT")        # (1, T, hidden)
"""
import os
import argparse
import torch

# Apply the FFT->conv1d patches as soon as the model module is imported.
import torch_xla.core.xla_model as xm
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import evo2_neuron_patch as P

MODEL = os.environ.get("EVO2_MODEL", os.path.expanduser("~/evo2/evo2_1b_8k"))


def load_model(model_path: str = MODEL):
    cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    cfg.use_fp8_input_projections = False
    P.patch_config(cfg, seqlen=0)
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True, config=cfg, attn_implementation="eager"
    ).eval().float()
    P.apply_evo2_neuron_patches()          # FFT -> conv1d (FIR + IIR)
    model = model.to(xm.xla_device())
    return tok, model


def _ids(tok, seq):
    return tok([seq], return_tensors="pt")["input_ids"].to(xm.xla_device())


def logits(tok, model, seq):
    with torch.no_grad():
        out = model(input_ids=_ids(tok, seq), use_cache=False)
    xm.mark_step()
    return out.logits.detach().float().cpu()


def embed(tok, model, seq):
    """Last hidden state (1, T, hidden). Evo2 embeddings are used for downstream tasks."""
    with torch.no_grad():
        out = model(input_ids=_ids(tok, seq), use_cache=False, output_hidden_states=True)
    xm.mark_step()
    return out.hidden_states[-1].detach().float().cpu()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", default="ACGTACGTACGTACGT", help="DNA string (A/C/G/T)")
    ap.add_argument("--mode", choices=["embed", "logits"], default="embed")
    ap.add_argument("--out", default=None, help="optional .pt to save the result")
    args = ap.parse_args()

    print(f"[evo2] loading model (first call compiles, ~30s)...")
    tok, model = load_model()
    print(f"[evo2] seq len {len(args.seq)} bp, mode={args.mode}")

    res = embed(tok, model, args.seq) if args.mode == "embed" else logits(tok, model, args.seq)
    print(f"[evo2] {args.mode} shape {tuple(res.shape)} finite={torch.isfinite(res).all().item()}")
    if args.out:
        torch.save(res, args.out)
        print(f"[evo2] saved -> {args.out}")


if __name__ == "__main__":
    main()

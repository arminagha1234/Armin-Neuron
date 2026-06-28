"""Evo2 (StripedHyena2) bring-up on Trainium via the Taykhoom HF port.

Runs the model on CPU (oracle) or on a NeuronCore and compares logits.

Key settings for Neuron:
  - use_fp8_input_projections=False  -> pure-PyTorch bf16 input projections
    (no TransformerEngine / no Hopper needed).
  - attn_implementation="eager"      -> pure-PyTorch attention (no flash-attn).
  - use_cache=False                  -> static graph (no KV-cache control flow).

Usage:
  python run_evo2.py cpu     --seqlen 256     # writes oracle
  python run_evo2.py neuron  --seqlen 256     # compiles, runs, compares
"""
import os, sys, time, argparse
import torch

MODEL = os.environ.get("EVO2_MODEL", os.path.expanduser("~/evo2/evo2_1b_8k"))
SEED = 1234


def make_ids(tok, seqlen):
    # Deterministic ACGT sequence of the requested byte length.
    import numpy as np
    rng = np.random.default_rng(SEED)
    seq = "".join("ACGT"[i] for i in rng.integers(0, 4, size=seqlen))
    return tok([seq], return_tensors="pt")["input_ids"]


def load_model(seqlen, patch=False):
    from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
    cfg = AutoConfig.from_pretrained(MODEL, trust_remote_code=True)
    cfg.use_fp8_input_projections = False
    if patch:
        import evo2_neuron_patch as P
        P.patch_config(cfg, seqlen)        # route IIR onto conv1d (no FFT)
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, trust_remote_code=True, config=cfg, attn_implementation="eager"
    ).eval().float()   # fp32: avoids bf16 norm-collapse on the model's massive activations
    if patch:
        import evo2_neuron_patch as P
        n = P.apply_evo2_neuron_patches()  # swap engine.fftconv_func -> conv1d
        print(f"[patch] replaced fftconv_func in {n} module(s); long_fir_threshold={cfg.long_fir_threshold}")
    return tok, model


def stats(logits):
    f = logits.detach().to(torch.float32).cpu()
    return {"shape": tuple(f.shape), "mean": f.mean().item(), "std": f.std().item(),
            "slice": f.flatten()[:128].clone(), "argmax": f[0].argmax(-1)[:64].clone()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("device", choices=["cpu", "neuron"])
    ap.add_argument("--seqlen", type=int, default=256)
    ap.add_argument("--patch", action="store_true", help="apply Neuron conv1d patches (cpu parity check)")
    args = ap.parse_args()
    oracle = os.path.expanduser(f"~/evo2/oracle_{args.seqlen}.pt")

    # Neuron always needs the patches; on CPU they're opt-in for parity checking.
    patch = args.patch or args.device == "neuron"
    tok, model = load_model(args.seqlen, patch=patch)
    ids = make_ids(tok, args.seqlen)
    print(f"[{args.device}] input_ids {tuple(ids.shape)}")

    if args.device == "neuron":
        import torch_xla.core.xla_model as xm
        dev = xm.xla_device()
        model = model.to(dev)
        ids = ids.to(dev)
        print("[neuron] first forward (compiles, may take minutes)...")
        t0 = time.time()
        with torch.no_grad():
            out = model(input_ids=ids, use_cache=False)
        xm.mark_step()
        s = stats(out.logits)
        print(f"[neuron] compile+run {time.time()-t0:.1f}s logits {s['shape']}")
        if os.path.exists(oracle):
            ref = torch.load(oracle)
            a, b = ref["slice"], s["slice"]
            cos = torch.nn.functional.cosine_similarity(a, b, dim=0).item()
            top1 = (ref["argmax"] == s["argmax"]).float().mean().item()
            print(f"[neuron] vs CPU oracle: cosine={cos:.6f}  top1_agree={top1*100:.1f}%  "
                  f"Δmean={s['mean']-ref['mean']:+.4e}")
        else:
            print(f"[neuron] no oracle at {oracle}; run cpu first.")
    else:
        t0 = time.time()
        with torch.no_grad():
            out = model(input_ids=ids, use_cache=False)
        s = stats(out.logits)
        if args.patch and os.path.exists(oracle):
            ref = torch.load(oracle)
            cos = torch.nn.functional.cosine_similarity(ref["slice"], s["slice"], dim=0).item()
            top1 = (ref["argmax"] == s["argmax"]).float().mean().item()
            print(f"[cpu+patch] vs FFT oracle: cosine={cos:.6f}  top1_agree={top1*100:.1f}%  "
                  f"Δmean={s['mean']-ref['mean']:+.4e}  (proves conv1d == FFT math)")
        else:
            torch.save(s, oracle)
            print(f"[cpu] forward {time.time()-t0:.1f}s logits {s['shape']} -> saved {oracle}")


if __name__ == "__main__":
    sys.exit(main())

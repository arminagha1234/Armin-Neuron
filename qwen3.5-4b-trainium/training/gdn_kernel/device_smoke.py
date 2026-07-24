"""
Device smoke test for the chunked-GDN monkeypatch on Trainium2 (box1).

Two things it proves:
  1. TRAIN: 2-layer Qwen3.5 (both linear/GDN layers) does forward+backward+AdamW
     on torch.device('neuron') with our chunked GDN, and loss decreases.
  2. COMPILE (the key test): torch.compile(backend='neuron', dynamic=False) on
     the 2-layer model SUCCEEDS where the stock torch-fallback GDN failed
     (32L -> SBUF overflow; 2L -> neuronx-cc "Can only vectorize loop or free axes").

Usage:
    QWEN35_GDN_CHUNKED=1 python3 device_smoke.py --mode train
    QWEN35_GDN_CHUNKED=1 python3 device_smoke.py --mode compile
    (set QWEN35_GDN_CHUNKED=0 to run the stock fallback for A/B comparison)
"""
from __future__ import annotations

import os, sys, time, argparse, traceback
import torch

# make chunked_gdn importable (same dir)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from chunked_gdn import patch_qwen35_gdn


def build_model(model_path, layers, vocab=None):
    from transformers import AutoConfig
    import transformers.models.qwen3_5.modeling_qwen3_5 as M
    full = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    # Qwen3.5-4B is a VL wrapper; the LLM config (incl. GDN dims) is under text_config.
    cfg = full.text_config
    if layers:
        cfg.num_hidden_layers = layers
        if hasattr(cfg, "layer_types") and isinstance(cfg.layer_types, (list, tuple)):
            cfg.layer_types = list(cfg.layer_types[:layers])
    if vocab:
        # Shrink token embedding + lm_head so a single-core FULL-FT smoke fits 24GB HBM.
        # Does NOT change GDN core dims (hidden_size=2560, 32 v-heads, K=V=128).
        cfg.vocab_size = vocab
    cfg._attn_implementation = "eager"
    model = M.Qwen3_5ForCausalLM(cfg)
    model = model.to(torch.bfloat16)
    # report which layers are linear/GDN
    try:
        lts = model.config.layer_types
        print(f"[cfg] layer_types(first {layers})={lts}", flush=True)
    except Exception:
        pass
    return model, cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/work/Qwen3.5-4B")
    ap.add_argument("--seq", type=int, default=256)
    ap.add_argument("--bs", type=int, default=1)
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--mode", choices=["train", "compile"], default="train")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--vocab", type=int, default=4096)
    a = ap.parse_args()

    patched = patch_qwen35_gdn(chunk_size=128, verbose=True)
    print(f"[cfg] QWEN35_GDN_CHUNKED={os.environ.get('QWEN35_GDN_CHUNKED','0')} patched={patched} "
          f"mode={a.mode} layers={a.layers} seq={a.seq} bs={a.bs}", flush=True)

    dev = torch.device("neuron")
    t0 = time.time()
    model, cfg = build_model(a.model, a.layers, vocab=(a.vocab or None))
    model = model.to(dev)
    model.train()
    print(f"[load] built+to-device in {time.time()-t0:.1f}s "
          f"params={sum(p.numel() for p in model.parameters())/1e6:.1f}M", flush=True)

    if a.mode == "compile":
        print("[compile] torch.compile(backend='neuron', dynamic=False) ...", flush=True)
        model = torch.compile(model, backend="neuron", dynamic=False)

    V = getattr(cfg, "vocab_size", None) or cfg.text_config.vocab_size
    torch.manual_seed(0)
    ids = torch.randint(0, V, (a.bs, a.seq), device=dev)
    labels = ids.clone()

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr)

    losses = []
    ok_compile = None
    try:
        for step in range(a.steps):
            t = time.time()
            opt.zero_grad()
            out = model(input_ids=ids, labels=labels, use_cache=False)
            loss = out.loss
            loss.backward()
            opt.step()
            l = float(loss.detach().float().cpu())
            losses.append(l)
            tag = "WARMUP/compile" if step == 0 else "warm"
            print(f"[step {step}] {tag} time={time.time()-t:.2f}s loss={l:.4f}", flush=True)
        ok_compile = True
    except Exception as e:
        ok_compile = False
        print(f"[ERROR] step failed: {e!r}", flush=True)
        traceback.print_exc()

    if losses:
        dec = losses[-1] < losses[0]
        print(f"\n=== RESULT mode={a.mode} ===", flush=True)
        print(f"loss_curve={['%.4f'%x for x in losses]}", flush=True)
        print(f"loss_decreased={dec}  first={losses[0]:.4f} last={losses[-1]:.4f}", flush=True)
    if a.mode == "compile":
        print(f"COMPILE_RESULT: success={ok_compile}", flush=True)
    print("SMOKE_DONE", flush=True)


if __name__ == "__main__":
    main()

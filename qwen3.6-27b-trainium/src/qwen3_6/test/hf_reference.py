# SPDX-License-Identifier: Apache-2.0
"""HF reference forward for Qwen3.6-27B — ground truth for parity debugging.

Loads the model via HuggingFace transformers on CPU and captures:
  - input token IDs
  - hidden state after embedding
  - hidden state after layer 0 (a DeltaNet/linear-attn layer)
  - hidden state after layer 3 (first full-attn GQA layer)
  - final logits + top-5 tokens for the next position

Writes everything to /tmp/hf_ref.pt for the adapter-side comparison.

Run with the server STOPPED (needs ~54 GB RAM for the model):
    python -m qwen3_6.test.hf_reference --model /root/models/Qwen3.6-27B
"""

import argparse
import torch


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/root/models/Qwen3.6-27B")
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--out", default="/tmp/hf_ref.pt")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    ids = tok(args.prompt, return_tensors="pt").input_ids
    print(f"[hf] prompt={args.prompt!r}  ids={ids.tolist()}")

    print("[hf] loading model on CPU (bf16)...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True,
        attn_implementation="eager",
    )
    model.eval()

    captured = {}

    # Find the decoder layer list. For the multimodal wrapper it's under
    # model.model.language_model.layers (or model.language_model.model.layers).
    def find_layers(m):
        for path in [
            "model.language_model.layers",
            "language_model.model.layers",
            "model.model.language_model.layers",
            "model.layers",
        ]:
            obj = m
            ok = True
            for part in path.split("."):
                if hasattr(obj, part):
                    obj = getattr(obj, part)
                else:
                    ok = False
                    break
            if ok:
                return obj, path
        return None, None

    layers, layers_path = find_layers(model)
    print(f"[hf] found layers at: {layers_path}  (n={len(layers) if layers is not None else '?'})")

    # Hook layer 0 and layer 3 outputs
    hooks = []
    def mk_hook(name):
        def hook(mod, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            captured[name] = h.detach().float().cpu()
        return hook
    if layers is not None:
        hooks.append(layers[0].register_forward_hook(mk_hook("layer0_out")))
        hooks.append(layers[3].register_forward_hook(mk_hook("layer3_out")))

    print("[hf] forward...", flush=True)
    with torch.no_grad():
        out = model(ids, output_hidden_states=True)

    logits = out.logits[0, -1].float()  # last position
    top5 = torch.topk(logits, 5)
    print("[hf] top-5 next-token:")
    for i in range(5):
        tid = top5.indices[i].item()
        print(f"   [{tid}] logit={top5.values[i].item():.4f}  {tok.decode([tid])!r}")

    hs = out.hidden_states  # tuple: embeddings + each layer
    captured["ids"] = ids
    captured["embed_out"] = hs[0][0].float().cpu()      # after embedding
    captured["final_hidden"] = hs[-1][0].float().cpu()  # after final norm-ish
    captured["logits_last"] = logits
    captured["top5_ids"] = top5.indices.tolist()

    for h in hooks:
        h.remove()

    torch.save(captured, args.out)
    print(f"[hf] saved → {args.out}")
    print(f"[hf] embed_out  shape={tuple(captured['embed_out'].shape)}  "
          f"mean={captured['embed_out'].mean():.5f}  std={captured['embed_out'].std():.5f}")
    if "layer0_out" in captured:
        print(f"[hf] layer0_out shape={tuple(captured['layer0_out'].shape)}  "
              f"mean={captured['layer0_out'].mean():.5f}  std={captured['layer0_out'].std():.5f}")
    if "layer3_out" in captured:
        print(f"[hf] layer3_out shape={tuple(captured['layer3_out'].shape)}  "
              f"mean={captured['layer3_out'].mean():.5f}  std={captured['layer3_out'].std():.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

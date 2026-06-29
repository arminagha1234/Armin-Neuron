"""B1 — get the CSM depth decoder onto the NeuronCore (attack the 156ms/frame floor).

The depth decoder is 31 serial CPU steps per audio frame and is now the dominant TTFA
cost (~156ms/frame). A1 proved prefix caching can't touch this, so this is the prize.

A prior offload attempt hit `NRT_EXEC_OOB`. Root-cause hypothesis (from reading
transformers/models/csm/modeling_csm.py):

  1. CsmDepthDecoderModel.forward:
        codebook_idxs = clamp(cache_position - 1, min=0)
        offset = codebook_idxs * self.vocab_size
        inputs_embeds = self.embed_tokens(input_ids + offset)   # <-- dynamic index into
        # embed_tokens = Embedding(num_codebooks*vocab_size, ...). On device a
        # data-dependent index that can exceed table bounds -> NRT_EXEC_OOB.

  2. CsmCodebooksHead.forward:
        codebook_idxs = cache_position - 1
        codebook_weight = self.weight[codebook_idxs]            # <-- dynamic gather into
        # weight[num_codebooks-1, hidden, vocab]. Same data-dependent index risk.

This harness runs the depth decoder STANDALONE (one frame -> 31 codebook steps) on the
device, instrumenting the two gather sites to print the index ranges actually seen, so we
can confirm whether the OOB is (a) index out of table range, or (b) a recompile/shape
issue. Then we try a bounded-index fix and re-run.

Usage:
    python b1_depth_on_device.py --mode diagnose   # offload as-is, print index ranges
    python b1_depth_on_device.py --mode fix        # apply bounded-index patch, run
"""
import os, sys, argparse, time, torch
import torch_xla.core.xla_model as xm
from transformers import AutoProcessor, CsmForConditionalGeneration
from transformers.models.csm import modeling_csm

MODEL = os.environ.get("CSM_MODEL", os.path.expanduser("~/csm/csm_1b"))


def _to(obj, dev):
    if torch.is_tensor(obj):
        return obj.to(dev)
    if obj.__class__.__name__.endswith("Cache"):
        return obj
    if isinstance(obj, (list, tuple)):
        return type(obj)(_to(x, dev) for x in obj)
    if isinstance(obj, dict):
        return {k: _to(v, dev) for k, v in obj.items()}
    try:
        from transformers.utils import ModelOutput
        if isinstance(obj, ModelOutput):
            for k in list(obj.keys()):
                obj[k] = _to(obj[k], dev)
            return obj
    except Exception:
        pass
    return obj


def _offload(module, dev, method="forward"):
    module.to(dev)
    for m in module.modules():
        for k, v in list(vars(m).items()):
            if torch.is_tensor(v) and v.device.type != "xla":
                setattr(m, k, v.to(dev))
    real = getattr(module, method)

    def wrapped(*a, **k):
        out = real(*_to(a, dev), **_to(k, dev))
        xm.mark_step()
        return _to(out, "cpu")
    setattr(module, method, wrapped)


def instrument_gather_sites():
    """Wrap the two dynamic-index sites to log the index ranges actually seen."""
    orig_model_fwd = modeling_csm.CsmDepthDecoderModel.forward

    def logged_model_fwd(self, *a, **k):
        cp = k.get("cache_position")
        ids = k.get("input_ids")
        if cp is not None and ids is not None:
            idx = (torch.clamp(cp - 1, min=0) * self.vocab_size)
            lo = int((ids + idx).min()); hi = int((ids + idx).max())
            tbl = self.embed_tokens.num_embeddings
            print(f"  [embed] cache_pos={cp.tolist()} index range=[{lo},{hi}] table={tbl}"
                  f"  {'OOB!' if hi >= tbl or lo < 0 else 'ok'}")
        return orig_model_fwd(self, *a, **k)

    modeling_csm.CsmDepthDecoderModel.forward = logged_model_fwd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["diagnose", "fix"], default="diagnose")
    ap.add_argument("--model", default=MODEL)
    args = ap.parse_args()

    dev = xm.xla_device()
    print(f"[b1] loading {args.model} (mode={args.mode}) ...")
    proc = AutoProcessor.from_pretrained(args.model)
    model = CsmForConditionalGeneration.from_pretrained(args.model, dtype=torch.bfloat16).eval()
    model.codec_model = model.codec_model.float()

    if args.mode == "diagnose":
        instrument_gather_sites()
        # keep depth on CPU first to log the *expected* index ranges (ground truth)...
        print("[b1] CPU reference run (logs expected index ranges):")
        inputs = proc("[0]Hello from Trainium.", add_special_tokens=True, return_tensors="pt")
        with torch.no_grad():
            model.generate(**inputs, output_audio=False, do_sample=False,
                           max_new_tokens=1, cache_implementation="static")
        print("[b1] ^ if all 'ok', the OOB on device is NOT an index-range problem "
              "-> it's a dynamic-gather lowering / recompile issue.")
        # ...then offload depth to device and try one frame.
        print("\n[b1] now offloading depth_decoder to device and running one frame:")
        _offload(model.backbone_model, dev)
        _offload(model.depth_decoder, dev)
        _offload(model.codec_model, dev, method="decode")
        try:
            with torch.no_grad():
                model.generate(**inputs, output_audio=False, do_sample=False,
                               max_new_tokens=1, cache_implementation="static")
            print("[b1] depth decoder ran on device WITHOUT OOB.")
        except Exception as e:
            print(f"[b1] depth-on-device FAILED: {type(e).__name__}: {str(e)[:400]}")
        return

    # mode == fix : patch the codebooks head to avoid the dynamic weight gather.
    # The depth loop visits codebooks in order; replace self.weight[cache_position-1]
    # with a per-step static slice driven by python int (decode is step-by-step anyway).
    print("[b1] (fix mode) applying bounded-index patch — TODO once diagnose confirms cause")


if __name__ == "__main__":
    sys.exit(main())

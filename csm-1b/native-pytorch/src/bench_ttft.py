"""CSM-1B Stage-0 latency harness — warm per-component breakdown + TTFA estimate.

Instruments the offloaded backbone (Neuron), the depth decoder (CPU), and the Mimi
codec (Neuron) to report where per-frame time goes, after warming the NEFF cache.

Reports:
  T_prefill          - backbone forward over the text prompt (once)
  T_backbone_step    - per-frame backbone decode step (median)
  T_depth_total      - per-frame depth decoder (31 codebook steps) (median)
  T_codec_1frame     - Mimi decode of a single frame
  TTFA_est           - estimated time-to-first-audio (streaming):
                       T_prefill + T_backbone_step + T_depth_total + T_codec_1frame
"""
import os, sys, time, argparse, statistics, torch
import torch_xla.core.xla_model as xm
from transformers import AutoProcessor, CsmForConditionalGeneration

MODEL = os.environ.get("CSM_MODEL", "/scratch/csm/csm_1b")
TEXT = "[0]Hello from Trainium, this is a latency benchmark."

T = {"backbone": [], "depth": [], "codec": []}


def _to(obj, dev):
    if torch.is_tensor(obj):
        return obj.to(dev)
    try:
        from transformers.utils import ModelOutput
        if isinstance(obj, ModelOutput):
            for k in list(obj.keys()):
                obj[k] = _to(obj[k], dev)
            return obj
    except Exception:
        pass
    if obj.__class__.__name__.endswith("Cache"):
        return obj
    if isinstance(obj, (list, tuple)):
        return type(obj)(_to(x, dev) for x in obj)
    if isinstance(obj, dict):
        return {k: _to(v, dev) for k, v in obj.items()}
    return obj


def _offload_timed(module, dev, bucket, method="forward"):
    module.to(dev)
    for m in module.modules():
        for k, v in list(vars(m).items()):
            if torch.is_tensor(v) and v.device.type != "xla":
                setattr(m, k, v.to(dev))
    real = getattr(module, method)

    def wrapped(*args, **kwargs):
        t0 = time.perf_counter()
        out = real(*_to(args, dev), **_to(kwargs, dev))
        xm.mark_step()
        out = _to(out, "cpu")           # forces device completion (sync)
        T[bucket].append((time.perf_counter() - t0) * 1000.0)
        return out
    setattr(module, method, wrapped)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=8, help="max_new_tokens (audio frames)")
    args = ap.parse_args()

    dev = xm.xla_device()
    proc = AutoProcessor.from_pretrained(MODEL)
    # Full bf16 — validated clean (cosine 0.999968, argmax 100%, no collapse; CsmRMSNorm
    # self-upcasts variance to fp32). ~1.4x on the backbone, no dtype boundaries.
    model = CsmForConditionalGeneration.from_pretrained(MODEL, dtype=torch.bfloat16).eval()
    model.codec_model = model.codec_model.float()  # codec fp32 (bf16 breaks its convs); fed int codes
    _offload_timed(model.backbone_model, dev, "backbone")
    _offload_timed(model.codec_model, dev, "codec", method="decode")

    # time depth decoder generate (runs on CPU; its 31 codebook steps)
    real_depth = model.depth_decoder.generate
    def depth_timed(*a, **k):
        t0 = time.perf_counter()
        r = real_depth(*a, **k)
        T["depth"].append((time.perf_counter() - t0) * 1000.0)
        return r
    model.depth_decoder.generate = depth_timed

    inputs = proc(TEXT, add_special_tokens=True, return_tensors="pt")
    gkw = dict(output_audio=True, do_sample=False, max_new_tokens=args.frames)

    def run():
        for k in T: T[k].clear()
        with torch.no_grad():
            model.generate(**inputs, **gkw)

    print(f"[bench] WARM pass (compiles all per-step shapes, frames={args.frames})...")
    t0 = time.time(); run(); print(f"[bench] warm pass done in {time.time()-t0:.1f}s")

    print("[bench] MEASURE pass (cached)...")
    t0 = time.time(); run(); wall = time.time() - t0

    bb = T["backbone"]; dp = T["depth"]; cd = T["codec"]
    # backbone: first call = prefill, rest = per-frame decode steps
    t_prefill = bb[0] if bb else float("nan")
    bb_steps = bb[1:] if len(bb) > 1 else bb
    med = lambda x: statistics.median(x) if x else float("nan")

    # codec on a single frame (separate, clean measurement)
    nq = model.config.num_codebooks
    codes1 = torch.randint(0, 2048, (1, nq, 1))
    T["codec"].clear()
    with torch.no_grad():
        model.codec_model.decode(codes1)
    t_codec_1f = T["codec"][0] if T["codec"] else float("nan")

    t_bb_step = med(bb_steps)
    t_depth = med(dp)
    ttfa = t_prefill + t_bb_step + t_depth + t_codec_1f

    print("\n========== CSM-1B WARM LATENCY (single NeuronCore, fp32) ==========")
    print(f"  T_prefill (backbone over prompt)   : {t_prefill:8.1f} ms")
    print(f"  T_backbone_step (per frame, median): {t_bb_step:8.1f} ms   (n={len(bb_steps)})")
    print(f"  T_depth_total  (per frame, median) : {t_depth:8.1f} ms   (31 codebook steps, n={len(dp)})")
    print(f"  T_codec_1frame                     : {t_codec_1f:8.1f} ms")
    print(f"  ---------------------------------------------------")
    print(f"  TTFA_est (streaming first audio)   : {ttfa:8.1f} ms   <-- vs 100ms / 500ms target")
    print(f"  full {args.frames}-frame wall (non-stream)   : {wall*1000:8.1f} ms")
    print(f"  per-frame budget (real-time @12.5fps): 80.0 ms (must beat for streaming)")
    print("===================================================================")


if __name__ == "__main__":
    sys.exit(main())

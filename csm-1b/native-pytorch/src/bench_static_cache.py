"""Test: StaticCache to kill per-frame recompiles on the offload path.

DynamicCache grows each decode frame -> new shapes -> neuronx-cc recompiles every
frame (the latency ceiling). StaticCache pre-allocates a fixed max length -> every
frame has identical shapes -> compile once. Measures per-frame latency stability.
"""
import os, time, argparse, statistics, torch
import torch_xla.core.xla_model as xm
from transformers import AutoProcessor, CsmForConditionalGeneration

MODEL = os.environ.get("CSM_MODEL", "/scratch/csm/csm_1b")


def _to(obj, dev):
    if torch.is_tensor(obj): return obj.to(dev)
    try:
        from transformers.utils import ModelOutput
        if isinstance(obj, ModelOutput):
            for k in list(obj.keys()): obj[k] = _to(obj[k], dev)
            return obj
    except Exception: pass
    if obj.__class__.__name__.endswith("Cache"): return obj
    if isinstance(obj, (list, tuple)): return type(obj)(_to(x, dev) for x in obj)
    if isinstance(obj, dict): return {k: _to(v, dev) for k, v in obj.items()}
    return obj


def _offload(module, dev, bucket, T, method="forward"):
    module.to(dev)
    for m in module.modules():
        for k, v in list(vars(m).items()):
            if torch.is_tensor(v) and v.device.type != "xla":
                setattr(m, k, v.to(dev))
    real = getattr(module, method)
    def wrapped(*a, **k):
        t0 = time.perf_counter()
        out = real(*_to(a, dev), **_to(k, dev)); xm.mark_step(); out = _to(out, "cpu")
        T[bucket].append((time.perf_counter()-t0)*1000)
        return out
    setattr(module, method, wrapped)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=10)
    ap.add_argument("--cache", default="static", choices=["static", "dynamic"])
    args = ap.parse_args()

    dev = xm.xla_device()
    T = {"backbone": []}
    proc = AutoProcessor.from_pretrained(MODEL)
    model = CsmForConditionalGeneration.from_pretrained(MODEL, dtype=torch.bfloat16).eval()
    model.codec_model = model.codec_model.float()
    _offload(model.backbone_model, dev, "backbone", T)

    inputs = proc("[0]Hello from Trainium static cache test.", add_special_tokens=True, return_tensors="pt")
    gkw = dict(output_audio=False, do_sample=False, max_new_tokens=args.frames)
    if args.cache == "static":
        gkw["cache_implementation"] = "static"

    def run(label):
        T["backbone"].clear()
        t0 = time.time()
        with torch.no_grad():
            model.generate(**inputs, **gkw)
        return time.time()-t0

    print(f"[{args.cache}] warm pass..."); run("warm")
    print(f"[{args.cache}] measure pass..."); wall = run("measure")
    bb = T["backbone"]
    steps = bb[1:] if len(bb) > 1 else bb
    print(f"\n  cache={args.cache}  total={wall*1000:.0f}ms  backbone calls={len(bb)}")
    print(f"  backbone prefill={bb[0]:.1f}ms  decode-step median={statistics.median(steps):.1f}ms "
          f"min={min(steps):.1f} max={max(steps):.1f}")
    print(f"  (stable per-step = StaticCache killed recompiles; high variance = still recompiling)")


if __name__ == "__main__":
    main()

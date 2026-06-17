#!/usr/bin/env python3
"""Cosmos-Predict2-2B Video2World — DiT on Trainium (native PyTorch, Beta 3).

Generates a short video clip conditioned on an input image + prompt, with
the CosmosTransformer3DModel running on `torch.device("neuron")`. Same
patches as the text2image runner (dummy safety, transforms shim, DiT
forward wrapper). Start small (few frames, low res) to prove the path;
video activation memory is the wall to watch.

Env: H, W, FRAMES, STEPS, HF_HOME, HF_TOKEN.
"""
import os, time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image

MODEL = "nvidia/Cosmos-Predict2-2B-Video2World"
PROMPT = ("A nighttime city street in the rain, neon signs reflecting on "
          "wet asphalt, gentle camera push-in, cinematic")
COND_IMG = "/mnt/data/cosmos_work/cosmos_neuron_2b.png"


class DummySafetyChecker(nn.Module):
    def __init__(self):
        super().__init__()
        self._p = nn.Parameter(torch.zeros(1), requires_grad=False)
    @property
    def device(self): return self._p.device
    @property
    def dtype(self): return self._p.dtype
    def check_text_safety(self, prompt): return True
    def check_video_safety(self, frames): return frames


def install_transforms_shim():
    import types
    import diffusers.models.transformers.transformer_cosmos as tc
    if getattr(tc, "_transforms_shim", False):
        return
    class _IM:
        NEAREST = "nearest"; BILINEAR = "bilinear"
    def _resize(img, size, interpolation="nearest", **kw):
        mode = interpolation if isinstance(interpolation, str) else "nearest"
        return F.interpolate(img, size=list(size), mode=mode)
    tc.transforms = types.SimpleNamespace(
        functional=types.SimpleNamespace(resize=_resize), InterpolationMode=_IM)
    tc._transforms_shim = True


def neuron_sync():
    if hasattr(torch, "neuron") and hasattr(torch.neuron, "synchronize"):
        torch.neuron.synchronize()


def _to_dev(x, dev):
    if isinstance(x, torch.Tensor): return x.to(dev)
    if isinstance(x, (list, tuple)): return type(x)(_to_dev(v, dev) for v in x)
    if isinstance(x, dict): return {k: _to_dev(v, dev) for k, v in x.items()}
    return x


def wrap_transformer_on_neuron(transformer, device):
    transformer.to(device)
    orig = transformer.forward
    stats = {"calls": 0, "total": 0.0, "first": None, "last": None}
    def fwd(*args, **kwargs):
        args = _to_dev(args, device); kwargs = _to_dev(kwargs, device)
        t = time.time()
        out = orig(*args, **kwargs); neuron_sync()
        dt = time.time() - t
        stats["calls"] += 1; stats["total"] += dt
        if stats["first"] is None: stats["first"] = dt
        stats["last"] = dt
        return _to_dev(out, "cpu")
    transformer.forward = fwd
    transformer._fwd_stats = stats
    return transformer


def main():
    from diffusers import Cosmos2VideoToWorldPipeline
    from diffusers.utils import export_to_video
    install_transforms_shim()
    import torch_neuronx  # noqa
    device = torch.device("neuron")

    h = int(os.environ.get("H", 256)); w = int(os.environ.get("W", 256))
    frames = int(os.environ.get("FRAMES", 17))
    steps = int(os.environ.get("STEPS", 12))

    t0 = time.time()
    pipe = Cosmos2VideoToWorldPipeline.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, token=os.environ.get("HF_TOKEN"),
        safety_checker=DummySafetyChecker())
    print(f"[vid] pipeline loaded {time.time()-t0:.1f}s", flush=True)

    wrap_transformer_on_neuron(pipe.transformer, device)
    print("[vid] transformer on neuron (T5+VAE on CPU)", flush=True)

    if os.path.exists(COND_IMG):
        cond = Image.open(COND_IMG).convert("RGB").resize((w, h))
    else:
        cond = Image.new("RGB", (w, h), (90, 90, 120))

    def run(tag):
        s = pipe.transformer._fwd_stats
        s["calls"] = 0; s["total"] = 0.0; s["first"] = None; s["last"] = None
        t = time.time()
        gen = torch.Generator(device="cpu").manual_seed(42)
        out = pipe(image=cond, prompt=PROMPT, height=h, width=w,
                   num_frames=frames, num_inference_steps=steps, generator=gen)
        dt = time.time() - t
        vid = out.frames[0]
        arr = np.array([np.array(f) for f in vid])
        avg = s["total"] / max(s["calls"], 1)
        print(f"[vid] {tag}: {dt:.1f}s | frames={len(vid)} "
              f"std={arr.std():.2f} mean={arr.mean():.1f}", flush=True)
        print(f"[vid] {tag} DiT: calls={s['calls']} total={s['total']:.1f}s "
              f"avg={avg:.2f}s first={s['first']:.2f}s last={s['last']:.2f}s "
              f"(other={dt - s['total']:.1f}s)", flush=True)
        return out

    print(f"[vid] === first call (compile) {h}x{w} x{frames}f ===", flush=True)
    out = run("cold")
    out = run("warm0")
    from diffusers.utils import export_to_video
    export_to_video(out.frames[0], "/mnt/data/cosmos_work/cosmos_video_2b.mp4",
                    fps=int(os.environ.get("FPS", 12)))
    print("[vid] saved cosmos_video_2b.mp4", flush=True)


if __name__ == "__main__":
    main()

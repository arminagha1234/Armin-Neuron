#!/usr/bin/env python3
"""Cosmos-Predict2-2B Text2Image — DiT on Trainium (native PyTorch, Beta 3).

Runs the CosmosTransformer3DModel (the hot path) on
`torch.device("neuron")` while T5 + WAN-VAE stay on CPU. The transformer's
forward is wrapped to shuttle tensors to/from Neuron, so the diffusers
pipeline orchestrates on CPU unchanged.

Env: H, W, STEPS, HF_HOME, HF_TOKEN.
"""
import os, time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

MODEL = "nvidia/Cosmos-Predict2-2B-Text2Image"
PROMPT = ("A nighttime city street in the rain, neon signs reflecting on "
          "wet asphalt, cinematic, highly detailed")


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
    if isinstance(x, torch.Tensor):
        return x.to(dev)
    if isinstance(x, (list, tuple)):
        return type(x)(_to_dev(v, dev) for v in x)
    if isinstance(x, dict):
        return {k: _to_dev(v, dev) for k, v in x.items()}
    return x


def wrap_transformer_on_neuron(transformer, device):
    """Move transformer to neuron; shuttle its forward I/O CPU<->neuron."""
    transformer.to(device)
    orig = transformer.forward

    def fwd(*args, **kwargs):
        args = _to_dev(args, device)
        kwargs = _to_dev(kwargs, device)
        out = orig(*args, **kwargs)
        neuron_sync()
        return _to_dev(out, "cpu")

    transformer.forward = fwd
    return transformer


def main():
    from diffusers import Cosmos2TextToImagePipeline
    install_transforms_shim()
    import torch_neuronx  # noqa
    device = torch.device("neuron")

    t0 = time.time()
    pipe = Cosmos2TextToImagePipeline.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, token=os.environ.get("HF_TOKEN"),
        safety_checker=DummySafetyChecker())
    print(f"[neuron] pipeline loaded {time.time()-t0:.1f}s", flush=True)

    # DiT -> neuron; T5 + VAE stay on CPU.
    wrap_transformer_on_neuron(pipe.transformer, device)
    print("[neuron] transformer moved to neuron (T5+VAE on CPU)", flush=True)

    h = int(os.environ.get("H", 512)); w = int(os.environ.get("W", 512))
    steps = int(os.environ.get("STEPS", 12))

    def run(tag):
        t = time.time()
        gen = torch.Generator(device="cpu").manual_seed(42)
        out = pipe(prompt=PROMPT, height=h, width=w,
                   num_inference_steps=steps, generator=gen)
        dt = time.time() - t
        a = np.array(out.images[0])
        print(f"[neuron] {tag}: {dt:.1f}s | std={a.std():.2f} "
              f"mean={a.mean():.1f} uniq={len(np.unique(a))}", flush=True)
        return out

    print("[neuron] === first call (compiles DiT) ===", flush=True)
    out = run("cold")
    out = run("warm0")
    out = run("warm1")
    out.images[0].save("/mnt/data/cosmos_work/cosmos_neuron_2b.png")
    print("[neuron] saved cosmos_neuron_2b.png", flush=True)


if __name__ == "__main__":
    main()

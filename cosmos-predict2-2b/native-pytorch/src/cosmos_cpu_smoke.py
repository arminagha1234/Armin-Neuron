#!/usr/bin/env python3
"""Cosmos-Predict2-2B Text2Image — CPU smoke test.

Confirms the diffusers pipeline loads and generates a correct image on
CPU (our reference) before we port to Neuron.
"""
import os, time, sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def install_transforms_shim():
    """Cosmos transformer uses torchvision transforms.functional.resize to
    resize the padding mask. torchvision has no ABI-matching build for the
    Neuron torch 2.11, so inject a tiny shim backed by F.interpolate."""
    import types
    import diffusers.models.transformers.transformer_cosmos as tc
    if getattr(tc, "_transforms_shim", False):
        return

    class _IM:
        NEAREST = "nearest"
        BILINEAR = "bilinear"

    def _resize(img, size, interpolation="nearest", **kw):
        mode = interpolation if isinstance(interpolation, str) else "nearest"
        return F.interpolate(img, size=list(size), mode=mode)

    tc.transforms = types.SimpleNamespace(
        functional=types.SimpleNamespace(resize=_resize),
        InterpolationMode=_IM,
    )
    tc._transforms_shim = True


MODEL = "nvidia/Cosmos-Predict2-2B-Text2Image"
PROMPT = ("A nighttime city street in the rain, neon signs reflecting on "
          "wet asphalt, cinematic, highly detailed")


class DummySafetyChecker(nn.Module):
    """No-op stand-in so we don't need the heavy cosmos_guardrail models.
    The pipeline's `.device` property reads safety_checker.device first
    (alphabetical), so we expose a real device + param."""
    def __init__(self):
        super().__init__()
        self._p = nn.Parameter(torch.zeros(1), requires_grad=False)
    @property
    def device(self):
        return self._p.device
    @property
    def dtype(self):
        return self._p.dtype
    def check_text_safety(self, prompt):
        return True
    def check_video_safety(self, frames):
        return frames


def main():
    from diffusers import Cosmos2TextToImagePipeline
    install_transforms_shim()
    t0 = time.time()
    pipe = Cosmos2TextToImagePipeline.from_pretrained(
        MODEL, torch_dtype=torch.float32, token=os.environ.get("HF_TOKEN"),
        safety_checker=DummySafetyChecker())
    print(f"[cpu] pipeline loaded in {time.time()-t0:.1f}s", flush=True)
    print(f"[cpu] transformer={type(pipe.transformer).__name__} "
          f"vae={type(pipe.vae).__name__} "
          f"text_encoder={type(pipe.text_encoder).__name__}", flush=True)

    h = int(os.environ.get("H", 512)); w = int(os.environ.get("W", 512))
    steps = int(os.environ.get("STEPS", 12))
    t0 = time.time()
    gen = torch.Generator(device="cpu").manual_seed(42)
    out = pipe(prompt=PROMPT, height=h, width=w,
               num_inference_steps=steps, generator=gen)
    dt = time.time() - t0
    img = out.images[0]
    img.save("/mnt/data/cosmos_work/cosmos_cpu_2b.png")
    a = np.array(img)
    print(f"[cpu] gen {h}x{w} steps={steps} in {dt:.1f}s | "
          f"img std={a.std():.2f} mean={a.mean():.1f} uniq={len(np.unique(a))}",
          flush=True)
    print("[cpu] saved /mnt/data/cosmos_work/cosmos_cpu_2b.png", flush=True)


if __name__ == "__main__":
    main()

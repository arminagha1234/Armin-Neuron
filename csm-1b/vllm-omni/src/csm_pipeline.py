# SPDX-License-Identifier: Apache-2.0
"""CsmPipeline — Sesame CSM-1B text-to-speech pipeline for vLLM-Omni on Neuron.

Registers a TTS pipeline alongside Wan22Pipeline/HelloWorldPipeline in the
`vllm_omni_neuron` plugin. CSM is a dual-decoder model (Llama backbone +
depth decoder) + Mimi codec; its `generate` loop cannot be lowered to Neuron
(int64 dynamic control flow), so we keep the generate loop on host and OFFLOAD the
heavy modules (backbone transformer + Mimi codec) to the NeuronCore via
forward/method wrappers. Validated: backbone cb0 logits cosine 1.0 (teacher-forced,
argmax 100%) and Mimi decode cosine 1.0 vs CPU.

forward(request) -> DiffusionOutput(output=<audio waveform tensor>).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.utils import get_local_device

PIPELINE_REGISTRY = [
    {"model_arch": "CsmForConditionalGeneration", "class_name": "CsmPipeline"},
    {"model_arch": "CsmPipeline", "class_name": "CsmPipeline"},
]

_DEFAULT_TEXT = "[0]Hello from Trainium."


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


def _move_stray(module, dev):
    """Move plain tensor attributes (e.g. Mimi RVQ self.embed) that .to() misses."""
    for m in module.modules():
        for k, v in list(vars(m).items()):
            if torch.is_tensor(v) and v.device.type != "xla":
                setattr(m, k, v.to(dev))


def _offload(module, dev, method="forward"):
    """Move `module` to `dev`; wrap `method` to accept/return CPU tensors so the
    host-side generate loop is unaffected while compute runs on the NeuronCore."""
    import torch_xla.core.xla_model as xm
    module.to(dev)
    _move_stray(module, dev)
    real = getattr(module, method)

    def wrapped(*args, **kwargs):
        args = _to(args, dev); kwargs = _to(kwargs, dev)
        out = real(*args, **kwargs)
        xm.mark_step()
        return _to(out, "cpu")

    setattr(module, method, wrapped)


def _extract_prompt(request) -> str:
    for attr in ("prompt", "text", "input"):
        v = getattr(request, attr, None)
        if isinstance(v, str) and v:
            return v
    for attr in ("prompts", "texts"):
        v = getattr(request, attr, None)
        if isinstance(v, (list, tuple)) and v and isinstance(v[0], str):
            return v[0]
    if isinstance(request, str):
        return request
    if isinstance(request, dict):
        for k in ("prompt", "text", "input"):
            if isinstance(request.get(k), str):
                return request[k]
    return _DEFAULT_TEXT


class CsmPipeline(nn.Module):
    weights_sources: list = []
    vae = None  # satisfies vllm-omni vae_use_slicing/tiling checks

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = ""):
        super().__init__()
        from transformers import AutoProcessor, CsmForConditionalGeneration
        self.od_config = od_config
        self.device = get_local_device()
        model = od_config.model
        self.max_new_tokens = int(getattr(od_config, "max_new_tokens", 0) or 256)

        self.processor = AutoProcessor.from_pretrained(model)
        self.model = CsmForConditionalGeneration.from_pretrained(
            model, dtype=torch.float32
        ).eval()
        # Offload the heavy compute to the NeuronCore; depth decoder stays on host.
        _offload(self.model.backbone_model, self.device)
        _offload(self.model.codec_model, self.device, method="decode")

    def to(self, *args, **kwargs):
        return self  # submodules already placed (backbone/codec on device)

    def compile(self, *args, **kwargs):
        pipeline = self

        class _CompiledPipeline:
            def forward(self, request):
                return pipeline.forward(request)

            def __call__(self, request):
                return self.forward(request)

            def __getattr__(self, name):
                return getattr(pipeline, name)

        return _CompiledPipeline()

    def load_weights(self, weights: object) -> set[str]:
        return set()  # weights loaded via from_pretrained in __init__

    @torch.no_grad()
    def forward(self, request) -> DiffusionOutput:
        text = _extract_prompt(request)
        inputs = self.processor(text, add_special_tokens=True, return_tensors="pt")
        # transformers>=4.57 masking does attention_mask.to(device=xla, dtype=bool);
        # torch_xla rejects a simultaneous device+dtype cast, so pre-cast to bool.
        if "attention_mask" in inputs and inputs["attention_mask"] is not None:
            inputs["attention_mask"] = inputs["attention_mask"].bool()
        audio = self.model.generate(
            **inputs, output_audio=True, do_sample=False,
            max_new_tokens=self.max_new_tokens,
        )
        wav = (audio[0] if isinstance(audio, (list, tuple)) else audio)
        wav = wav.detach().float().cpu()
        return DiffusionOutput(output=wav)

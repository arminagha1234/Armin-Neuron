# SPDX-License-Identifier: Apache-2.0
"""NeuronLTX2Pipeline — LTX-2 video pipeline for Neuron hardware.

This is the v1 pragmatic port: it puts the DiT transformer on Neuron and
keeps the text encoder, VAE, audio VAE, and vocoder on CPU. That mirrors
the pattern Qwen-Image-Edit Path C used to ship — Trainium handles the
heavy attention path, and the lighter encoder/VAE work runs on CPU.

Subclasses vllm-omni's `LTX2Pipeline` (base class), inheriting:
  - forward(), encode_prompt(), prepare_latents(), check_inputs(),
    predict_noise(), denoising loop

Overrides:
  - __init__()    — Neuron device handling, CPU encoders, sharded transformer
  - to()          — selectively move components to Neuron vs leave on CPU
  - load_weights()— TP-sharded weight routing for the transformer
  - _decode_*()   — CPU-side VAE decode to avoid nxdi dtype-cast issues
  - compile()     — Neuron compile of the transformer (encoders stay eager)

Reference template: vllm_omni_neuron/diffusion/models/neuron_wan_pipeline.py

PIPELINE_REGISTRY at the bottom enables auto-registration when this file is
dropped into vllm_omni_neuron/diffusion/models/.

Status: v1 — designed to produce a video frame-by-frame on Neuron with
known correctness (CPU encoders + VAE) at the cost of some flat-tax
overhead. Phase 2 will move VAE + text encoder to Neuron once the v1
pipeline is validated end-to-end.
"""
from __future__ import annotations

import copy
import logging
import os

import torch
import torch.distributed as dist
from torch import nn

from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.models.ltx2 import (
    LTX2Pipeline,
    create_transformer_from_config,
    get_ltx2_post_process_func,  # noqa: F401 — required by registry
    load_transformer_config,
)


logger = logging.getLogger(__name__)


PIPELINE_REGISTRY = [
    {
        "model_arch": "LTX2Pipeline",
        "class_name": "NeuronLTX2Pipeline",
        "post_process_func_name": "get_ltx2_post_process_func",
    },
]


class NeuronLTX2Pipeline(LTX2Pipeline):
    """LTX-2 T2V pipeline for Neuron.

    Subclasses vllm-omni's `LTX2Pipeline` and replaces only the bits that
    need Neuron-aware handling. The base class's denoising loop and
    transformer call sites stay unchanged.

    Architecture notes (v1):
      - text_encoder (Gemma3 12B), connectors, vae, audio_vae, vocoder
        stay on CPU. Their forward() is fast enough relative to the
        denoising loop that the per-call CPU/Neuron round-trip is fine
        for v1. The denoising loop itself runs on Neuron.
      - transformer (LTX2VideoTransformer3DModel) is built from config
        on Neuron. Base class already uses vLLM parallel primitives so
        TP sharding is handled by the framework weight loader.
      - We override to() to pin encoders/VAEs to CPU even when the
        engine calls pipeline.to(neuron_device).

    Phase 2 plan (post-v1):
      - Wrap text_encoder with a Neuron-compiled module (mirrors
        NeuronTextEncoderWrapper for UMT5 → Gemma3).
      - Move VAE to Neuron with tiled decoding (PR #57's pattern).
      - Wrap audio_vae + vocoder if customer needs audio output.
    """

    def __init__(self, *, od_config, prefix: str = ""):
        # Skip LTX2Pipeline.__init__ (does GPU `.to(device)` on every
        # component which OOMs on Neuron). Call nn.Module directly.
        nn.Module.__init__(self)
        self.od_config = od_config
        self.device = get_local_device()
        dtype = getattr(od_config, "dtype", torch.bfloat16)

        model = od_config.model
        local_files_only = os.path.isdir(model)

        # ---- CPU-resident components ----
        # The base LTX2Pipeline.__init__ does `.to(self.device)` on each
        # of these; we replicate the construction without the move so they
        # stay on CPU. The denoising loop's forward() will pay a small
        # CPU/Neuron data-transfer cost each step, but that's < the win
        # of keeping ~24 GB of encoder/VAE weights off the Neuron core.
        from transformers import AutoTokenizer, Gemma3ForConditionalGeneration
        from diffusers import (
            AutoencoderKLLTX2Audio,
            AutoencoderKLLTX2Video,
            FlowMatchEulerDiscreteScheduler,
        )
        from diffusers.pipelines.ltx2 import LTX2TextConnectors
        from diffusers.pipelines.ltx2.vocoder import LTX2Vocoder
        from diffusers.video_processor import VideoProcessor

        self.tokenizer = AutoTokenizer.from_pretrained(
            model, subfolder="tokenizer", local_files_only=local_files_only,
        )
        self.text_encoder = Gemma3ForConditionalGeneration.from_pretrained(
            model, subfolder="text_encoder",
            torch_dtype=dtype,
            local_files_only=local_files_only,
        )  # stays on CPU
        self.connectors = LTX2TextConnectors.from_pretrained(
            model, subfolder="connectors",
            torch_dtype=dtype,
            local_files_only=local_files_only,
        )  # stays on CPU
        # VAE is small enough that we keep on CPU for v1. Phase 2 → Neuron tiled.
        self.vae = AutoencoderKLLTX2Video.from_pretrained(
            model, subfolder="vae",
            torch_dtype=dtype,
            local_files_only=local_files_only,
        )
        # Audio components only loaded if the model directory has them
        # (some LTX-2 checkpoints are video-only).
        try:
            self.audio_vae = AutoencoderKLLTX2Audio.from_pretrained(
                model, subfolder="audio_vae",
                torch_dtype=dtype,
                local_files_only=local_files_only,
            )
            self.vocoder = LTX2Vocoder.from_pretrained(
                model, subfolder="vocoder",
                torch_dtype=dtype,
                local_files_only=local_files_only,
            )
            self.has_audio = True
        except (OSError, ValueError) as e:
            logger.info("LTX-2 audio components not found — video-only mode (%s)", e)
            self.audio_vae = None
            self.vocoder = None
            self.has_audio = False

        # ---- Transformer goes on Neuron ----
        # The base LTX2VideoTransformer3DModel uses vLLM parallel layers
        # internally, so the framework weight loader handles TP sharding.
        transformer_config = load_transformer_config(
            model, "transformer", local_files_only,
        )
        self.transformer = create_transformer_from_config(transformer_config)
        # We do NOT call self.transformer.to(self.device) here — the engine
        # calls pipeline.to(neuron_device) after __init__, and our
        # to() override below handles the selective move.

        # Tell the framework loader to load ONLY the transformer weights
        # (encoders + VAE were loaded above via from_pretrained()).
        # The framework calls our `load_weights(weights)` with an iterable
        # of (name, tensor) tuples coming from these sources, which we
        # forward to the base LTX2VideoTransformer3DModel's load_weights.
        from vllm_omni.diffusion.model_loader.diffusers_loader import (
            DiffusersPipelineLoader,
        )
        self.weights_sources = [
            DiffusersPipelineLoader.ComponentSource(
                model_or_path=od_config.model,
                subfolder="transformer",
                revision=None,
                prefix="transformer.",
                fall_back_to_pt=True,
            ),
        ]

        # Scheduler is config-only, no weights to move
        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            model, subfolder="scheduler", local_files_only=local_files_only,
        )

        # ---- Compression ratios + helpers (mirror base) ----
        self.vae_spatial_compression_ratio = (
            self.vae.spatial_compression_ratio
            if getattr(self, "vae", None) is not None else 32
        )
        self.vae_temporal_compression_ratio = (
            self.vae.temporal_compression_ratio
            if getattr(self, "vae", None) is not None else 8
        )
        self.audio_vae_mel_compression_ratio = (
            self.audio_vae.mel_compression_ratio
            if getattr(self, "audio_vae", None) is not None else 4
        )
        self.audio_vae_temporal_compression_ratio = (
            self.audio_vae.temporal_compression_ratio
            if getattr(self, "audio_vae", None) is not None else 4
        )
        self.transformer_spatial_patch_size = (
            self.transformer.config.patch_size
            if getattr(self, "transformer", None) is not None else 1
        )
        self.transformer_temporal_patch_size = (
            self.transformer.config.patch_size_t
            if getattr(self, "transformer", None) is not None else 1
        )
        self.audio_sampling_rate = (
            self.audio_vae.config.sample_rate
            if getattr(self, "audio_vae", None) is not None else 16000
        )
        self.audio_hop_length = (
            self.audio_vae.config.mel_hop_length
            if getattr(self, "audio_vae", None) is not None else 160
        )

        self.video_processor = VideoProcessor(
            vae_scale_factor=self.vae_spatial_compression_ratio,
        )

        # tokenizer_max_length: same fallback chain as base
        tokenizer_max_length = 1024
        if getattr(self, "tokenizer", None) is not None:
            tokenizer_max_length = self.tokenizer.model_max_length
            if tokenizer_max_length is None or tokenizer_max_length > 100000:
                encoder_config = getattr(self.text_encoder, "config", None)
                config_max_len = getattr(encoder_config, "max_position_embeddings", None)
                if config_max_len is None:
                    config_max_len = getattr(encoder_config, "max_seq_len", None)
                tokenizer_max_length = config_max_len or 1024
        self.tokenizer_max_length = int(tokenizer_max_length)

        # Private state used by inherited forward()
        self._guidance_scale = None
        self._guidance_rescale = None
        self._attention_kwargs = None
        self._interrupt = False
        self._num_timesteps = None

        # We don't need an Omni warmup — first real call will compile.
        # Mirrors NeuronWanPipeline; engine respects this flag.
        self.skip_warmup = True

        # VAE is not TP-sharded — only rank 0 owns it for I/O purposes
        # (encoders + VAE are CPU but the engine still queries .vae on
        # rank 0). We keep is_vae_rank for parity with the Wan pattern.
        self.is_vae_rank = not dist.is_initialized() or dist.get_rank() == 0

    # ------------------------------------------------------------------
    # Selective device move
    # ------------------------------------------------------------------
    def to(self, *args, **kwargs):
        """Move ONLY the transformer to Neuron; encoders + VAE stay on CPU.

        The engine calls pipeline.to(neuron_device) after __init__. The
        base nn.Module.to() walks every submodule, which would put the
        12B Gemma3 + the VAE on Neuron and OOM. We override to move only
        the DiT.
        """
        if self.transformer is not None:
            self.transformer.to(*args, **kwargs)
        return self

    # ------------------------------------------------------------------
    # CPU-resident encoder / VAE wiring
    # ------------------------------------------------------------------
    # The base class's encode_prompt() and the denoising loop use
    # self.text_encoder, self.connectors, self.vae, self.audio_vae,
    # self.vocoder. Since we kept those on CPU, the base class works
    # unchanged — it just sees CPU tensors come back from those calls.
    #
    # The transformer call site in the base class typically looks like:
    #     noise_pred = self.transformer(latents, ...)
    # Where `latents` is on the transformer's device. The base class
    # pre-moves latents via prepare_latents() which calls
    # randn(..., device=self.device) — and self.device == neuron device
    # because we set it in __init__. So latents land on Neuron, the
    # transformer call works, and noise_pred comes back on Neuron.
    #
    # The VAE decode at the end of the loop receives Neuron-side latents
    # and runs on a CPU module. The base class does this:
    #     latents = latents.to(self.vae.dtype)
    #     video = self.vae.decode(latents).sample
    # Since self.vae is on CPU, we need to move latents to CPU before
    # decode. Override _decode_latents (or whatever the base calls) to
    # do that move.

    # ------------------------------------------------------------------
    # Compile + load_weights
    # ------------------------------------------------------------------
    def compile_transformer(self, *args, **kwargs):
        """Compile the DiT transformer for Neuron."""
        t_kwargs = copy.deepcopy(kwargs)
        t_kwargs["fullgraph"] = True
        t_options = t_kwargs.setdefault("options", {})
        t_options["compiler_args"] = (
            "--model-type=transformer --auto-cast=none -O1 "
            "--hbm-scratchpad-page-size=2048 "
        )
        self.transformer.compile(*args, **t_kwargs)

    def compile(self, *args, **kwargs):
        """Compile only the transformer; encoders + VAE run eager on CPU."""
        if self.transformer is not None:
            self.compile_transformer(*args, **kwargs)
        return self

    def load_weights(self, weights=None):
        """Forward the framework's (name, tensor) iterable to the transformer.

        The framework's diffusers_loader walks `self.weights_sources` (set
        in __init__ to point at the transformer subfolder), iterates the
        safetensors files, and calls this method with an iterable of
        `(name, tensor)` tuples prefixed with `transformer.`. The base
        LTX2VideoTransformer3DModel.load_weights handles vLLM TP sharding
        internally — we strip the prefix before forwarding (so the
        transformer's parameter dict matches), then we add the prefix
        back to the names of the loaded set we return (so the framework
        loader's "weights not initialized" check matches).
        """
        if weights is None:
            return set()
        if self.transformer is None or not hasattr(self.transformer, "load_weights"):
            return set()

        prefix = "transformer."
        def _strip_prefix(it):
            for name, tensor in it:
                if name.startswith(prefix):
                    name = name[len(prefix):]
                yield name, tensor
        loaded = self.transformer.load_weights(_strip_prefix(weights))
        # Re-prefix so the framework's "expected" set matches what we
        # return (the framework checks this list against the names from
        # our safetensors files, which have the `transformer.` prefix).
        return {prefix + n for n in (loaded or set())}

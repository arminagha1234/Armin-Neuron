# SPDX-License-Identifier: Apache-2.0
"""NeuronFlux2KleinPipeline — FLUX.2-klein 4B on Neuron via vllm-omni.

Mirrors the structure of `neuron_ltx2_pipeline.py` and `neuron_wan_pipeline.py`:
the heavy DiT transformer runs on Neuron, the lighter components (Qwen3 text
encoder + AutoencoderKLFlux2 VAE) stay on CPU.

Subclasses vllm-omni's `Flux2KleinPipeline`, inheriting:
  - encode_prompt()
  - prepare_latents() / prepare_image_latents() / _encode_vae_image()
  - check_inputs(), property accessors, scheduler glue
  - forward()  (the denoising loop)

Overrides:
  - __init__()    — Neuron device handling, CPU encoders/VAE, sharded transformer
  - to()          — selectively move only the transformer to Neuron
  - encode_prompt — keep Qwen3 forward on CPU, then move embeds to Neuron
  - prepare_latents — force bf16 + CPU generator (Neuron has no Generator)
  - prepare_image_latents — same; VAE encode runs on CPU
  - load_weights  — TP-sharded weight routing for the transformer
  - compile()     — Neuron compile of the transformer (encoders stay eager)
  - forward()     — short-circuit Omni's dummy_run; otherwise delegate to base

Reference templates:
  - vllm_omni_neuron/diffusion/models/neuron_ltx2_pipeline.py
  - vllm_omni_neuron/diffusion/models/neuron_wan_pipeline.py

PIPELINE_REGISTRY at the bottom enables auto-registration when this file is
dropped into the vllm_omni_neuron plugin's diffusion/models/ folder.

Customer driver: fal/flux-2-klein-4B-zoom-lora — image-to-image editing LoRA
on top of FLUX.2-klein 4B base. The LoRA is a transformer-only adapter
(~76 MB safetensors) that fuses into the DiT before serving.
"""
from __future__ import annotations

import copy
import logging
import math
import os

import torch
import torch.distributed as dist
from torch import nn

from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.models.flux2_klein import (
    Flux2KleinPipeline,
    Flux2Transformer2DModel,
    get_flux2_klein_post_process_func,  # noqa: F401 — required by registry
)


logger = logging.getLogger(__name__)


PIPELINE_REGISTRY = [
    {
        "model_arch": "Flux2KleinPipeline",
        "class_name": "NeuronFlux2KleinPipeline",
        "post_process_func_name": "get_flux2_klein_post_process_func",
    },
]


# ---------------------------------------------------------------------------
# Helper: scheduler patch (CPU build + bf16 pre-cast + move)
# ---------------------------------------------------------------------------
def _make_neuron_scheduler(scheduler, target_dtype):
    """Patch a scheduler instance to keep timesteps CPU-built then moved.

    Mirrors the LTX-2 pattern but applied to whatever scheduler class
    Flux2-klein uses (FlowMatchEulerDiscreteScheduler at the time of
    writing). We monkey-patch `.set_timesteps` on the instance to pre-
    build on CPU + cast to `target_dtype` + move to the requested device.

    The pipeline at flux2_klein/pipeline_flux2_klein.py:922 does:
        timestep = t.expand(latents.shape[0]).to(latents.dtype)
    On Neuron a real bf16 cast on a Neuron-resident timestep tensor
    trips `Expected self.dtype() == dst.dtype()`. By pre-casting on
    CPU here, the later `.to()` becomes a no-op.
    """
    base_set = scheduler.set_timesteps

    def patched_set_timesteps(num_inference_steps=None, device=None,
                              sigmas=None, mu=None, **kwargs):
        target_device = device
        cpu_kwargs = dict(kwargs)
        if sigmas is not None:
            cpu_kwargs["sigmas"] = sigmas
        if mu is not None:
            cpu_kwargs["mu"] = mu
        cpu_kwargs["device"] = torch.device("cpu")
        result = (
            base_set(num_inference_steps=num_inference_steps, **cpu_kwargs)
            if num_inference_steps is not None
            else base_set(**cpu_kwargs)
        )
        if hasattr(scheduler, "timesteps") and scheduler.timesteps is not None:
            if target_dtype is not None:
                scheduler.timesteps = scheduler.timesteps.to(dtype=target_dtype)
            if target_device is not None:
                scheduler.timesteps = scheduler.timesteps.to(device=target_device)
        if hasattr(scheduler, "sigmas") and scheduler.sigmas is not None:
            if target_dtype is not None:
                scheduler.sigmas = scheduler.sigmas.to(dtype=target_dtype)
            if target_device is not None:
                scheduler.sigmas = scheduler.sigmas.to(device=target_device)
        return result

    scheduler.set_timesteps = patched_set_timesteps
    return scheduler


# ---------------------------------------------------------------------------
# Helper: Patch Timesteps modules to compute embedding on CPU
# ---------------------------------------------------------------------------
def _patch_timesteps_to_cpu(transformer):
    """Force the diffusers `Timesteps` modules' embedding to be built on CPU.

    The base Flux2 transformer has:
        self.time_proj = Timesteps(num_channels=in_channels, ...)
        self.timestep_embedder = TimestepEmbedding(...)

    `Timesteps.forward()` calls `get_timestep_embedding(timesteps, ...)`
    which does:
        torch.arange(start=0, end=half_dim, dtype=torch.float32,
                     device=timesteps.device)

    When `timesteps` is on Neuron (which it is during the warm-up call),
    that arange lands on Neuron INSIDE the compiled graph. The XLA
    backend then segfaults during `PjRtComputationClient::ExecuteComputation`
    when the resulting graph runs.

    Fix: wrap each `Timesteps` instance's forward to take its input on
    CPU, do the sin/cos table build on CPU, then move the result back to
    the original device. Embedding is a small per-step op (length 256)
    so the round-trip is cheap.

    The Flux2 transformer's path is:
        transformer.time_guidance_embed.time_proj   (Timesteps)
        transformer.time_guidance_embed.timestep_embedder  (TimestepEmbedding)
        transformer.time_guidance_embed.guidance_embedder  (TimestepEmbedding, optional)
    so we walk recursively rather than hardcoding the path.
    """
    # Recursively find every Timesteps instance under the transformer.
    from diffusers.models.embeddings import Timesteps
    targets = []
    for name, mod in transformer.named_modules():
        if isinstance(mod, Timesteps):
            targets.append((name, mod))

    for name, sub in targets:
        orig_forward = sub.forward

        def _make_wrapped(_orig):
            def _wrapped(timesteps):
                orig_device = timesteps.device
                ts_cpu = timesteps.to(device="cpu")
                with torch.no_grad():
                    out = _orig(ts_cpu)
                return out.to(device=orig_device)
            return _wrapped

        sub.forward = _make_wrapped(orig_forward)
        logger.info("[neuron-flux2] patched Timesteps at %s to run on CPU", name)

    # Belt-and-suspenders: also monkey-patch the class-level
    # get_timestep_embedding to force its torch.arange off the input
    # device (i.e. always CPU). Some Dynamo trace paths unwrap our
    # instance-method patch back to the class implementation; this
    # ensures the freq table is built on CPU regardless.
    try:
        import diffusers.models.embeddings as _emb_mod
        _orig_gte = _emb_mod.get_timestep_embedding

        def _gte_cpu_arange(timesteps, embedding_dim, *args, **kwargs):
            # Run the whole function on CPU then move back.
            orig_device = timesteps.device
            ts_cpu = timesteps.to(device="cpu")
            with torch.no_grad():
                out = _orig_gte(ts_cpu, embedding_dim, *args, **kwargs)
            return out.to(device=orig_device)

        if not getattr(_emb_mod, "_neuron_flux2_patched", False):
            _emb_mod.get_timestep_embedding = _gte_cpu_arange
            _emb_mod._neuron_flux2_patched = True
            logger.info("[neuron-flux2] class-level get_timestep_embedding patched (CPU arange)")
    except Exception as _e:
        logger.debug("[neuron-flux2] class-level patch skipped: %s", _e)


def _patch_pos_embed_to_cpu(transformer):
    """Force RoPE freq computation on CPU + float32.

    `Flux2PosEmbed.forward()` does:
        is_npu = ids.device.type == "npu"   # False for Neuron (type="xla")
        freqs_dtype = float32 if (is_mps or is_npu) else float64
        for axis: get_1d_rotary_pos_embed(..., freqs_dtype=freqs_dtype)

    Two problems on Neuron:
      1. device.type for Neuron is "xla" → falls through to float64.
         Neuron doesn't support float64 — segfaults during execute.
      2. `get_1d_rotary_pos_embed` does torch.arange(..., device=pos.device).
         If pos is on Neuron, arange runs on Neuron INSIDE the compile
         boundary and trips the lazy backend.

    Fix: replace `Flux2PosEmbed.forward()` (CLASS-level — Dynamo
    unwraps instance patches) with a CPU+float32 implementation. The
    position IDs are small (img seq + txt seq tokens); CPU compute
    is cheap and predictable. We move ids to CPU, compute cos/sin in
    float32, and move the result back to the requested device.

    NOTE: instance-level patches via `pe.forward = ...` get unwrapped
    by Dynamo back to the class definition, which is why we need the
    class-level monkey-patch.
    """
    pe = getattr(transformer, "pos_embed", None)
    if pe is None or not hasattr(pe, "forward"):
        return

    pe_class = type(pe)
    if getattr(pe_class, "_neuron_flux2_patched", False):
        # Already patched at class level; ensure pe_replacement also installed
        pass

    from diffusers.models.embeddings import get_1d_rotary_pos_embed

    # Replacement module: same axes_dim/theta config but CPU-only forward
    # that returns real cos/sin tensors. We swap the WHOLE submodule
    # (`transformer.pos_embed = NeuronFluxPosEmbed(...)`) so Dynamo traces
    # this class instead of the original (which contains torch.polar
    # → complex64, Neuron-fatal).
    class _NeuronFluxPosEmbed(nn.Module):
        def __init__(self, axes_dim, theta):
            super().__init__()
            self.axes_dim = list(axes_dim)
            self.theta = theta

        def forward(self, ids):
            orig_device = ids.device
            ids_cpu = ids.to(device="cpu")
            cos_out = []
            sin_out = []
            pos = ids_cpu.float()
            with torch.no_grad():
                for i in range(len(self.axes_dim)):
                    # use_real=False produces a complex tensor of shape
                    # (seq, dim/2). The original Flux2PosEmbed splits
                    # it into .real and .imag for concat — we replicate
                    # that exactly so downstream rope sees the same
                    # shapes. The complex compute happens on CPU so
                    # `torch.polar` never enters the FX graph.
                    freqs_cis = get_1d_rotary_pos_embed(
                        self.axes_dim[i],
                        pos[..., i],
                        theta=self.theta,
                        use_real=False,
                        freqs_dtype=torch.float32,
                    )
                    cos_out.append(freqs_cis.real.contiguous())
                    sin_out.append(freqs_cis.imag.contiguous())
            freqs_cos = torch.cat(cos_out, dim=-1).to(device=orig_device)
            freqs_sin = torch.cat(sin_out, dim=-1).to(device=orig_device)
            return freqs_cos, freqs_sin

    new_pe = _NeuronFluxPosEmbed(pe.axes_dim, pe.theta)
    # Replace in-place. Subsequent transformer.forward calls see the
    # new module.
    transformer.pos_embed = new_pe
    # Some references may have been captured (e.g. rope_prepare.pos_embed).
    if hasattr(transformer, "rope_prepare") and hasattr(transformer.rope_prepare, "pos_embed"):
        transformer.rope_prepare.pos_embed = new_pe

    # Belt-and-suspenders: Dynamo has been observed to unwrap submodule
    # swaps and trace the ORIGINAL Flux2PosEmbed.forward via the class
    # MRO. Overwrite that class's forward in place so even if Dynamo
    # re-resolves through the type, it sees the CPU-fp32 version.
    pe_class.forward = _NeuronFluxPosEmbed.forward
    pe_class._neuron_flux2_patched = True
    logger.info("[neuron-flux2] swapped Flux2PosEmbed for _NeuronFluxPosEmbed (CPU+fp32+real, no torch.polar)")
    return  # do NOT also class-patch — the swap is the canonical fix.

    # ---- (unreachable) class-level patch kept for reference ----
    def _patched_forward(self, ids):
        orig_device = ids.device
        ids_cpu = ids.to(device="cpu")
        cos_out = []
        sin_out = []
        pos = ids_cpu.float()
        with torch.no_grad():
            for i in range(len(self.axes_dim)):
                freqs_cis = get_1d_rotary_pos_embed(
                    self.axes_dim[i],
                    pos[..., i],
                    theta=self.theta,
                    use_real=False,
                    freqs_dtype=torch.float32,
                )
                cos_out.append(freqs_cis.real)
                sin_out.append(freqs_cis.imag)
        freqs_cos = torch.cat(cos_out, dim=-1).to(device=orig_device)
        freqs_sin = torch.cat(sin_out, dim=-1).to(device=orig_device)
        return freqs_cos, freqs_sin

    pe_class.forward = _patched_forward
    pe_class._neuron_flux2_patched = True
    logger.info("[neuron-flux2] CLASS-level Flux2PosEmbed.forward patched (CPU+fp32, no torch.polar in graph)")

    # Belt #3: monkey-patch the module-level `get_1d_rotary_pos_embed`
    # so even if Dynamo unwraps our class-level patch and re-traces the
    # original `Flux2PosEmbed.forward`, the inner function still
    # short-circuits to a CPU+fp32+real-valued (cos/sin) implementation
    # that doesn't emit `torch.polar` (complex64 — Neuron-unsupported)
    # in the FX graph.
    try:
        import diffusers.models.embeddings as _emb_mod
        _orig_g1d = _emb_mod.get_1d_rotary_pos_embed

        def _g1d_cpu_real(dim, pos, *args, **kwargs):
            # Force CPU execution and use_real=True to emit a (cos, sin)
            # tuple instead of a complex tensor. Neuron has no complex
            # dtype support, so torch.polar is fatal.
            kwargs["use_real"] = True
            kwargs["freqs_dtype"] = torch.float32
            orig_device = pos.device if torch.is_tensor(pos) else None
            pos_cpu = pos.to(device="cpu") if torch.is_tensor(pos) else pos
            with torch.no_grad():
                out = _orig_g1d(dim, pos_cpu, *args, **kwargs)
            # use_real=True → (cos, sin) tuple of real tensors
            if isinstance(out, tuple) and len(out) == 2 and orig_device is not None:
                return tuple(t.to(device=orig_device) for t in out)
            return out

        if not getattr(_emb_mod, "_neuron_flux2_g1d_patched", False):
            _emb_mod.get_1d_rotary_pos_embed = _g1d_cpu_real
            _emb_mod._neuron_flux2_g1d_patched = True
            logger.info("[neuron-flux2] module-level get_1d_rotary_pos_embed patched (CPU+fp32+use_real=True)")
    except Exception as _e:
        logger.debug("[neuron-flux2] g1d patch skipped: %s", _e)


# ---------------------------------------------------------------------------
# Eager wrapper around the Flux2 DiT
# ---------------------------------------------------------------------------
class _NeuronTransformerWrapper(nn.Module):
    """Eager wrapper around the Flux2-klein DiT.

    Sits OUTSIDE the torch.compile boundary so it can do data-movement
    that the compiled graph can't tolerate (CPU→Neuron tensor moves,
    contiguity coercion, dtype coercion). All Neuron-specific data-prep
    happens in this wrapper's `forward()`; the inner DiT stays pure
    tensor math that compiles cleanly.

    Same pattern used in `neuron_ltx2_pipeline._NeuronTransformerWrapper`,
    minus the LTX-specific RoPE precompute.
    """

    def __init__(self, inner):
        super().__init__()
        self.inner = inner
        self._target_device = None

    @property
    def config(self):
        return self.inner.config

    @property
    def dtype(self):
        if hasattr(self.inner, "dtype"):
            return self.inner.dtype
        for p in self.inner.parameters():
            return p.dtype
        return torch.bfloat16

    def _resolve_target_device(self):
        if self._target_device is not None:
            return self._target_device
        for p in self.inner.parameters():
            self._target_device = p.device
            return self._target_device
        self._target_device = get_local_device()
        return self._target_device

    def forward(self, *args, **kwargs):
        target = self._resolve_target_device()
        for key in (
            "hidden_states",
            "timestep",
            "guidance",
            "encoder_hidden_states",
            "txt_ids",
            "img_ids",
        ):
            v = kwargs.get(key)
            if v is not None and torch.is_tensor(v) and v.device != target:
                kwargs[key] = v.contiguous().to(device=target)
        return self.inner(*args, **kwargs)

    def load_weights(self, weights):
        if hasattr(self.inner, "load_weights"):
            return self.inner.load_weights(weights)
        return set()

    def compile(self, *args, **kwargs):
        if hasattr(self.inner, "compile"):
            self.inner.compile(*args, **kwargs)
        return self

    def to(self, *args, **kwargs):
        self._target_device = None
        self.inner.to(*args, **kwargs)
        return self


# ---------------------------------------------------------------------------
# The pipeline
# ---------------------------------------------------------------------------
class NeuronFlux2KleinPipeline(Flux2KleinPipeline):
    """FLUX.2-klein 4B image / image-to-image pipeline for Neuron."""

    def __init__(self, *, od_config, prefix: str = "", is_distilled: bool = False):
        # Skip Flux2KleinPipeline.__init__ — it puts encoders + VAE on
        # Neuron. Construct via nn.Module directly + load components on CPU.
        nn.Module.__init__(self)
        self.od_config = od_config
        self.is_distilled = is_distilled
        self._execution_device = get_local_device()
        self.device = self._execution_device
        dtype = getattr(od_config, "dtype", torch.bfloat16)

        model = od_config.model
        local_files_only = os.path.exists(model)

        # ---- CPU-resident components ----
        from diffusers.models.autoencoders.autoencoder_kl_flux2 import AutoencoderKLFlux2
        from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
        from transformers import Qwen2TokenizerFast, Qwen3ForCausalLM

        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            model, subfolder="scheduler", local_files_only=local_files_only,
        )
        # Patch scheduler to build timesteps on CPU + pre-cast to bf16.
        self.scheduler = _make_neuron_scheduler(self.scheduler, dtype)

        self.text_encoder = Qwen3ForCausalLM.from_pretrained(
            model, subfolder="text_encoder",
            torch_dtype=dtype,
            local_files_only=local_files_only,
        )  # stays on CPU
        self.tokenizer = Qwen2TokenizerFast.from_pretrained(
            model, subfolder="tokenizer", local_files_only=local_files_only,
        )
        self.vae = AutoencoderKLFlux2.from_pretrained(
            model, subfolder="vae",
            torch_dtype=dtype,
            local_files_only=local_files_only,
        )  # stays on CPU

        # ---- Transformer goes on Neuron ----
        from vllm_omni.diffusion.utils.tf_utils import get_transformer_config_kwargs
        transformer_kwargs = get_transformer_config_kwargs(
            od_config.tf_model_config, Flux2Transformer2DModel,
        )
        inner_transformer = Flux2Transformer2DModel(
            quant_config=od_config.quantization_config,
            **transformer_kwargs,
        )
        # Patch the Timesteps module so the sinusoidal embedding is
        # computed on CPU then moved. Keeps the freq-table arange off
        # the compiled device subgraph (which segfaults).
        _patch_timesteps_to_cpu(inner_transformer)
        # Patch Flux2PosEmbed (rope freq compute) to run on CPU + float32.
        # The base does float64 compute when device.type isn't "npu" or
        # "mps" — but Neuron device.type is "xla", so it falls through
        # to float64 which Neuron rejects.
        _patch_pos_embed_to_cpu(inner_transformer)
        self.transformer = _NeuronTransformerWrapper(inner_transformer)

        # ---- Image processor + scale factor ----
        from vllm_omni.diffusion.models.flux2_klein.pipeline_flux2_klein import (
            Flux2ImageProcessor,
        )
        self.vae_scale_factor = (
            2 ** (len(self.vae.config.block_out_channels) - 1)
            if getattr(self, "vae", None) else 8
        )
        self.image_processor = Flux2ImageProcessor(vae_scale_factor=self.vae_scale_factor * 2)
        self.tokenizer_max_length = 512
        self.default_sample_size = 128

        # ---- Weight-source for the framework loader ----
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

        # Private state used by inherited methods
        self._guidance_scale = None
        self._attention_kwargs = None
        self._num_timesteps = None
        self._current_timestep = None
        self._interrupt = False

        try:
            self.setup_diffusion_pipeline_profiler(
                enable_diffusion_pipeline_profiler=getattr(
                    od_config, "enable_diffusion_pipeline_profiler", False,
                )
            )
        except Exception:
            logger.debug("setup_diffusion_pipeline_profiler skipped (optional)")

        self.skip_warmup = True
        self.is_vae_rank = not dist.is_initialized() or dist.get_rank() == 0

    # ------------------------------------------------------------------
    # Selective device move
    # ------------------------------------------------------------------
    def to(self, *args, **kwargs):
        """Move ONLY the transformer to Neuron; encoder + VAE stay on CPU."""
        if self.transformer is not None:
            self.transformer.to(*args, **kwargs)
        return self

    # ------------------------------------------------------------------
    # Encode prompt — Qwen3 stays on CPU
    # ------------------------------------------------------------------
    def encode_prompt(
        self,
        prompt,
        device=None,
        num_images_per_prompt: int = 1,
        prompt_embeds=None,
        max_sequence_length: int = 512,
        text_encoder_out_layers=(9, 18, 27),
    ):
        target_device = device or self._execution_device

        if prompt is None:
            prompt = ""
        prompt = [prompt] if isinstance(prompt, str) else prompt

        if prompt_embeds is None:
            prompt_embeds = self._get_qwen3_prompt_embeds(
                text_encoder=self.text_encoder,
                tokenizer=self.tokenizer,
                prompt=prompt,
                # Force CPU — encoder lives on CPU.
                device=torch.device("cpu"),
                max_sequence_length=max_sequence_length,
                hidden_states_layers=text_encoder_out_layers,
            )

        batch_size, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        text_ids = self._prepare_text_ids(prompt_embeds)

        encoder_dtype = getattr(self.text_encoder, "dtype", torch.bfloat16)
        prompt_embeds = prompt_embeds.to(dtype=encoder_dtype)
        prompt_embeds = prompt_embeds.to(device=target_device)
        text_ids = text_ids.to(device=target_device)
        return prompt_embeds, text_ids

    # ------------------------------------------------------------------
    # VAE encode — runs on CPU
    # ------------------------------------------------------------------
    def _encode_vae_image(self, image, generator):
        if image.ndim != 4:
            raise ValueError(f"Expected image dims 4, got {image.ndim}.")
        image_cpu = image.to(device="cpu")
        from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img import (
            retrieve_latents,
        )
        image_latents = retrieve_latents(
            self.vae.encode(image_cpu), generator=generator, sample_mode="argmax",
        )
        image_latents = self._patchify_latents(image_latents)
        latents_bn_mean = self.vae.bn.running_mean.view(1, -1, 1, 1).to(
            image_latents.device, image_latents.dtype,
        )
        latents_bn_std = torch.sqrt(
            self.vae.bn.running_var.view(1, -1, 1, 1) + self.vae.config.batch_norm_eps,
        )
        image_latents = (image_latents - latents_bn_mean) / latents_bn_std
        return image_latents

    # ------------------------------------------------------------------
    # prepare_latents — force bf16 + CPU generator
    # ------------------------------------------------------------------
    def prepare_latents(
        self,
        batch_size,
        num_latents_channels,
        height,
        width,
        dtype,
        device,
        generator,
        latents=None,
    ):
        transformer_dtype = (
            self.transformer.dtype
            if (self.transformer is not None and hasattr(self.transformer, "dtype"))
            else torch.bfloat16
        )
        dtype = transformer_dtype
        if isinstance(generator, torch.Generator) and generator.device.type != "cpu":
            generator = torch.Generator(device="cpu").manual_seed(generator.initial_seed())
        return super().prepare_latents(
            batch_size, num_latents_channels, height, width,
            dtype, device, generator, latents,
        )

    def prepare_image_latents(self, images, batch_size, generator, device, dtype):
        transformer_dtype = (
            self.transformer.dtype
            if (self.transformer is not None and hasattr(self.transformer, "dtype"))
            else torch.bfloat16
        )
        dtype = transformer_dtype
        if isinstance(generator, torch.Generator) and generator.device.type != "cpu":
            generator = torch.Generator(device="cpu").manual_seed(generator.initial_seed())
        return super().prepare_image_latents(
            images, batch_size, generator, device, dtype,
        )

    # ------------------------------------------------------------------
    # Compile + load_weights
    # ------------------------------------------------------------------
    def compile_transformer(self, *args, **kwargs):
        t_kwargs = copy.deepcopy(kwargs)
        t_kwargs["fullgraph"] = True
        t_options = t_kwargs.setdefault("options", {})
        t_options["compiler_args"] = (
            "--model-type=transformer --auto-cast=none -O1 "
            "--hbm-scratchpad-page-size=2048 "
        )
        self.transformer.compile(*args, **t_kwargs)

    def compile(self, *args, **kwargs):
        if self.transformer is not None:
            self.compile_transformer(*args, **kwargs)
        return self

    def forward(self, req):
        from vllm_omni.diffusion.data import DiffusionOutput
        if (
            getattr(self, "skip_warmup", False)
            and getattr(req, "request_ids", None) == ["dummy_req_id"]
        ):
            prompt_field = req.prompts[0] if req.prompts else None
            if isinstance(prompt_field, str):
                prompt = prompt_field
            elif isinstance(prompt_field, dict):
                prompt = prompt_field.get("prompt")
            else:
                prompt = None
            if prompt == "dummy run":
                logger.info("Skipping warmup request on Neuron Flux2-klein pipeline")
                return DiffusionOutput(output=None)
        return super().forward(req)

    def load_weights(self, weights=None):
        if weights is None:
            return set()
        if self.transformer is None or not hasattr(self.transformer, "load_weights"):
            return set()

        in_prefix = "transformer."
        out_prefix = "transformer.inner."

        def _strip_prefix(it):
            for name, tensor in it:
                if name.startswith(in_prefix):
                    name = name[len(in_prefix):]
                yield name, tensor

        loaded = self.transformer.load_weights(_strip_prefix(weights))
        return {out_prefix + n for n in (loaded or set())}

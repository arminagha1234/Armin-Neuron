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


def _patch_coord_prep_to_cpu(transformer):
    """Force the transformer's RoPE coordinate-prep methods to RETURN CPU tensors.

    `prepare_video_coords` / `prepare_audio_coords` live on the RoPE
    submodules (`transformer.rope` and `transformer.audio_rope`), and the
    pipeline calls them with `device` as a POSITIONAL argument:
        transformer.rope.prepare_video_coords(B, F, H, W, latents.device, fps=...)
        transformer.audio_rope.prepare_audio_coords(B, F, audio.device)

    They build grid coordinates with meshgrid → stack → flatten → repeat
    + in-place index assignment. On Neuron those produce non-contiguous
    tensors and use in-place ops the lazy backend rejects
    (`Expected self.is_contiguous() to be true`).

    Critical detail: pipeline_ltx2.py:1090 then does
        video_coords = video_coords.repeat((2,) + (1,)*(coords.ndim-1))
    which, if coords are on Neuron, also trips the same contiguity error
    (this is the catch-22 that defeated previous attempts).

    Fix: leave coords on CPU here. They're tiny, data-independent, and
    the pipeline's `.repeat(...)` is a CPU op. The
    `_NeuronTransformerWrapper.forward()` then moves them to the Neuron
    device + `.contiguous()` right before calling the compiled inner
    DiT — outside the compile boundary, so it's a no-op for compiled
    inputs. See ltx2-omni-DECISIONS.md.
    """
    import torch as _torch

    # (submodule_attr, method_name, positional index of the `device` arg)
    targets = [
        ("rope", "prepare_video_coords", 4),
        ("audio_rope", "prepare_audio_coords", 2),
    ]

    for sub_attr, meth_name, dev_pos in targets:
        sub = getattr(transformer, sub_attr, None)
        if sub is None:
            continue
        orig = getattr(sub, meth_name, None)
        if orig is None:
            continue

        def _make_wrapper(_orig, _dev_pos):
            def _wrapped(*args, **kwargs):
                new_args = list(args)
                # Force coord prep to run on CPU and KEEP coords on CPU.
                # Pipeline_ltx2's `.repeat()` after this call is a CPU op
                # if coords are CPU. The Neuron move happens later in
                # `_NeuronTransformerWrapper.forward()` (eager, outside
                # the compile boundary) so the compiled inner DiT still
                # sees Neuron-resident coords.
                if "device" in kwargs:
                    kwargs["device"] = _torch.device("cpu")
                elif len(new_args) > _dev_pos and isinstance(
                    new_args[_dev_pos], (_torch.device, str)
                ):
                    new_args[_dev_pos] = _torch.device("cpu")
                out = _orig(*new_args, **kwargs)
                if hasattr(out, "contiguous"):
                    out = out.contiguous()
                return out
            return _wrapped

        setattr(sub, meth_name, _make_wrapper(orig, dev_pos))


def _patch_sdpa_data_dependent_branch():
    """No-op stub: the in-container `sdpa.py` already has the correct
    inline patch for the data-dependent branch.

    Earlier we tried to monkey-patch `_maybe_reshape_attn_mask` to return
    the mask unchanged. That was WRONG: the original function does a
    necessary 2D→4D reshape (for `broadcast_k` it goes
    `[B, S_k] → [B, 1, 1, S_k]`). Skipping that reshape feeds SDPA the
    raw 2D mask which then can't broadcast to `[B, H, S_q, S_k]` —
    surface error: "Attempting to broadcast a dimension of length 128
    at -1".

    The actual fix lives directly in
    `/opt/conda/lib/python3.12/site-packages/vllm_omni/diffusion/
    attention/backends/sdpa.py`: the `if torch.all(...)` block was
    replaced with `pass` (we keep the original `.bak`). That's the only
    change needed — remove the data-dependent branch but keep the
    reshape that follows.

    This stub is kept so the pipeline file documents the situation and
    is easy to extend if a future container layout needs a runtime
    patch instead. See ltx2-omni-DECISIONS.md.
    """
    return


# Apply the SDPA patch at import time (idempotent no-op today; container-side
# inline patch handles the real fix).
_patch_sdpa_data_dependent_branch()


class _NeuronTransformerWrapper(nn.Module):
    """Eager wrapper around the LTX-2 DiT transformer.

    Sits OUTSIDE the torch.compile boundary so it can do data-movement
    that the compiled graph can't tolerate (CPU→Neuron coord move,
    contiguity, dtype coercion). All Neuron-specific data-prep happens
    in this wrapper's `forward()`; the inner DiT stays pure tensor math
    that compiles cleanly.

    This pattern is borrowed from Jim Burtoft's NxDI LTX-2 port
    (`neuron/external/pr-117-nxdi-diffusion-models/contrib/models/
    ltx2-video-audio/src/pipeline.py::NeuronTransformerWrapper`) which
    solved the same catch-22 in the NxDI flow. The shape difference:
    NxDI compiles a separate "backbone" (blocks-only); we compile the
    full DiT and just wrap it with a coord-mover.

    Pipeline-visible interface (the base `LTX2Pipeline.forward()`
    accesses these attributes on `self.transformer`, so the wrapper
    proxies them):
        .config           — DiT config
        .dtype            — for `latents.to(prompt_embeds.dtype)`
        .rope             — for `transformer.rope.prepare_video_coords`
        .audio_rope       — for `transformer.audio_rope.prepare_audio_coords`
        .cache_context    — optional, for `_transformer_cache_context`
        .__call__         — forward()
    """

    def __init__(self, inner: nn.Module):
        super().__init__()
        # Store under a non-`_modules` name so torch's auto-wrap doesn't
        # try to compile it via `self.compile()`. We DO want it tracked
        # for `.parameters()`, `.to()`, etc., so register as a child.
        self.inner = inner
        # Cached target device — lazily resolved on first forward().
        self._target_device = None
        # Cache for precomputed RoPE outputs (filled in by
        # `_precompute_rope_outputs` on first forward). Keyed on
        # video_coords/audio_coords shape so cache invalidates if the
        # video shape changes between requests. Same shape = no
        # recompute, just constant tensor reuse.
        self._rope_cache: dict = {}

    # --- attribute proxies (the base pipeline reads these) -----------
    @property
    def config(self):
        return self.inner.config

    @property
    def dtype(self):
        return self.inner.dtype if hasattr(self.inner, "dtype") else torch.bfloat16

    @property
    def rope(self):
        return self.inner.rope

    @property
    def audio_rope(self):
        return self.inner.audio_rope

    @property
    def cache_context(self):
        # Optional; pipeline's `_transformer_cache_context` checks
        # `callable(...)` so we forward only when the inner has it.
        return getattr(self.inner, "cache_context", None)

    # --- forward (data-prep + compiled call) -------------------------
    def _resolve_target_device(self):
        """Find the device the inner is on (its first parameter)."""
        if self._target_device is not None:
            return self._target_device
        for p in self.inner.parameters():
            self._target_device = p.device
            return self._target_device
        # Fallback: the local Neuron device.
        self._target_device = get_local_device()
        return self._target_device

    def _precompute_rope_outputs(self, video_coords, audio_coords, target_device):
        """Compute the four RoPE outputs on CPU and cache as Neuron tensors.

        The transformer's `forward()` calls four rope modules INSIDE the
        compile boundary:
            self.rope(video_coords, device=hidden_states.device)
            self.audio_rope(audio_coords, device=audio_hidden_states.device)
            self.cross_attn_rope(video_coords[:, 0:1, :], device=hs.device)
            self.cross_attn_audio_rope(audio_coords[:, 0:1, :], device=ahs.device)

        Each rope's `forward()` does
            torch.linspace(..., device=device)  # device = hidden_states.device
            ...stack/repeat ops...
            grid = stack.to(device)             # ← FAILS

        Inside the FX trace, `hidden_states.device` resolves to a virtual
        XLA device and the final `.to(neuron:N)` becomes an
        `unimplemented _copy_from xla:0neuron:N` error. The rope outputs
        are deterministic functions of (coords, batch_size, seq_len) —
        same every diffusion step — so we precompute them on CPU here
        (eagerly, outside compile), move to the Neuron device, and
        monkey-patch the four rope modules to return these precomputed
        tensors. The compiled graph then just sees "constant tensor"
        values for cos/sin and never tries to build them.
        """
        import torch as _torch

        cache = self._rope_cache
        v_key = tuple(int(x) for x in video_coords.shape)
        a_key = tuple(int(x) for x in audio_coords.shape)
        if cache.get("video_key") == v_key and cache.get("audio_key") == a_key:
            return  # already cached

        v_cpu = video_coords.detach().to(device="cpu")
        a_cpu = audio_coords.detach().to(device="cpu")
        cpu = _torch.device("cpu")

        with _torch.no_grad():
            video_rope = self.inner.rope(v_cpu, device=cpu)
            audio_rope = self.inner.audio_rope(a_cpu, device=cpu)
            video_xrope = self.inner.cross_attn_rope(v_cpu[:, 0:1, :], device=cpu)
            audio_xrope = self.inner.cross_attn_audio_rope(
                a_cpu[:, 0:1, :], device=cpu)

        def _to_target(rope_out):
            # rope_out is (cos, sin)
            cos, sin = rope_out
            return (
                cos.contiguous().to(device=target_device),
                sin.contiguous().to(device=target_device),
            )

        cache["video"] = _to_target(video_rope)
        cache["audio"] = _to_target(audio_rope)
        cache["video_x"] = _to_target(video_xrope)
        cache["audio_x"] = _to_target(audio_xrope)
        cache["video_key"] = v_key
        cache["audio_key"] = a_key

        # Monkey-patch the four rope forwards to return cached values.
        # We patch on the INSTANCE (not the class) so we don't pollute
        # other LTX2 models in this process.
        wrapper_self = self

        def _make_const_forward(cache_key):
            def _forward(coords, device=None):
                return wrapper_self._rope_cache[cache_key]
            return _forward

        # Each rope is called with positional `coords` and either
        # positional or keyword `device`. We accept both via *args,
        # **kwargs to be safe.
        def _make_robust_forward(cache_key):
            def _forward(*args, **kwargs):
                return wrapper_self._rope_cache[cache_key]
            return _forward

        # Only install the forward overrides ONCE — installing every
        # call would compound the bound-method dispatch.
        if not cache.get("installed", False):
            self.inner.rope.forward = _make_robust_forward("video")
            self.inner.audio_rope.forward = _make_robust_forward("audio")
            self.inner.cross_attn_rope.forward = _make_robust_forward("video_x")
            self.inner.cross_attn_audio_rope.forward = _make_robust_forward("audio_x")
            cache["installed"] = True

    def forward(self, *args, **kwargs):
        target = self._resolve_target_device()
        # Coords come from the pipeline as CPU tensors (we patched the
        # coord-prep). Move them to the inner's device with .contiguous()
        # right before the compiled call, eagerly — outside compile.
        for key in ("video_coords", "audio_coords"):
            v = kwargs.get(key)
            if v is not None and hasattr(v, "to") and v.device != target:
                kwargs[key] = v.contiguous().to(device=target)

        # Precompute the four RoPE outputs on CPU and cache them on the
        # wrapper. The cache only invalidates on shape changes (which
        # don't happen across diffusion steps for a fixed video shape),
        # so this is a no-op after the first step. See docstring above.
        v = kwargs.get("video_coords")
        a = kwargs.get("audio_coords")
        if v is not None and a is not None:
            # Use the original CPU coords for hashing/computing. The
            # `kwargs[key]` were already moved to Neuron above; rebuild
            # CPU views for the rope precompute. The coords are tiny
            # and contiguous, so .cpu() round-trip is safe.
            self._precompute_rope_outputs(v.cpu(), a.cpu(), target)

        return self.inner(*args, **kwargs)

    # --- weight-loading + compile passthroughs -----------------------
    def load_weights(self, weights):
        """Forward the framework's (name, tensor) iterable to the inner."""
        if hasattr(self.inner, "load_weights"):
            return self.inner.load_weights(weights)
        return set()

    def compile(self, *args, **kwargs):
        """Compile the INNER DiT, not the wrapper.

        We want the data-prep in `_NeuronTransformerWrapper.forward()` to
        stay eager (so coord moves happen outside the compile boundary).
        Calling `self.inner.compile(...)` swaps the inner's `forward` /
        `__call__` with the dynamo-wrapped version; the outer wrapper's
        `forward` just dispatches to `self.inner(...)` which goes through
        the compiled path.
        """
        if hasattr(self.inner, "compile"):
            self.inner.compile(*args, **kwargs)
        return self

    def to(self, *args, **kwargs):
        # Reset cached device — the inner may move.
        self._target_device = None
        self.inner.to(*args, **kwargs)
        return self


class _ConnectorsCompatWrapper(nn.Module):
    """Adapter for the vLLM-Omni-Beta-1 ↔ diffusers-0.38 connectors API gap.

    vLLM-Omni's pipeline_ltx2.forward() calls:
        connectors(prompt_embeds, additive_attention_mask, additive_mask=True)

    but diffusers 0.38 LTX2TextConnectors.forward() signature is:
        forward(text_encoder_hidden_states, attention_mask,
                padding_side="left", scale_factor=8)

    Two differences:
      1. No `additive_mask` kwarg in diffusers — we drop it.
      2. vLLM-Omni passes an *additive* mask: `(1 - binary_mask) * -1e6`
         (0.0 where keep, -1e6 where drop). diffusers wants a
         *multiplicative binary* mask (1 where keep, 0 where drop).
         We convert: binary = (additive >= -0.5) → 1.0, else 0.0.
         (additive is 0.0 on kept positions, -1e6 on dropped — so a
         threshold near 0 recovers the binary mask.)

    This wrapper lets us keep vLLM-Omni's `forward()` intact (it has all
    the audio/CFG/two-stage logic) while fixing only the broken call.
    Delete this wrapper when a Beta 2/3 Omni image fixes the connectors
    API. See ltx2-omni-DECISIONS.md.
    """

    def __init__(self, inner):
        super().__init__()
        self.inner = inner

    def forward(self, hidden_states, mask, additive_mask=False, **kwargs):
        import torch as _torch

        # The connectors module lives on CPU (we kept the lighter
        # components off the Neuron core). The vLLM-Omni forward() passes
        # Neuron-resident tensors here, so move inputs to CPU, run, and
        # move the outputs back to the original device. Two-step dtype/
        # device moves throughout to dodge Neuron's combined-.to() trap.
        orig_device = hidden_states.device
        inner_dtype = next(self.inner.parameters()).dtype

        hs_cpu = hidden_states.to(device="cpu")
        hs_cpu = hs_cpu.to(dtype=inner_dtype)
        mask_cpu = mask.to(device="cpu")

        if additive_mask:
            # Convert additive (0 / -1e6) → multiplicative binary (1 / 0).
            binary_mask = (mask_cpu >= -0.5).to(inner_dtype)
        else:
            binary_mask = mask_cpu.to(inner_dtype)

        padding_side = kwargs.pop("padding_side", "left")
        scale_factor = kwargs.pop("scale_factor", 8)

        with _torch.no_grad():
            out = self.inner(
                hs_cpu,
                binary_mask,
                padding_side=padding_side,
                scale_factor=scale_factor,
            )

        # out is a tuple (connector_prompt_embeds,
        # connector_audio_prompt_embeds, connector_attention_mask).
        # Move each back to the Neuron device in two steps.
        def _back(t):
            if t is None or not hasattr(t, "to"):
                return t
            return t.to(device=orig_device)

        if isinstance(out, tuple):
            return tuple(_back(t) for t in out)
        return _back(out)

    def to(self, *args, **kwargs):
        self.inner.to(*args, **kwargs)
        return self


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
        # vLLM-Omni Beta 1's pipeline_ltx2.forward() calls
        #   self.connectors(embeds, additive_attention_mask, additive_mask=True)
        # but the diffusers 0.38 LTX2TextConnectors.forward() signature is
        #   forward(text_encoder_hidden_states, attention_mask,
        #           padding_side="left", scale_factor=8)
        # — it has no `additive_mask` kwarg, and it wants a *multiplicative
        # binary* mask (1s=keep, 0s=drop), not the additive (1-mask)*-1e6
        # form vllm-omni passes. This is a vendor API mismatch in Beta 1.
        # We wrap the connectors so the vllm-omni-style call is translated
        # to the diffusers-style call. See ltx2-omni-DECISIONS.md.
        self.connectors = _ConnectorsCompatWrapper(self.connectors)
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
        inner_transformer = create_transformer_from_config(transformer_config)
        # Patch the transformer's coordinate-prep methods to RETURN CPU
        # tensors. `prepare_video_coords` (and its audio sibling) build
        # RoPE grid coordinates with meshgrid/stack/flatten/repeat +
        # in-place index assignment — on Neuron these trip
        # `Expected self.is_contiguous() to be true`. The coords are
        # tiny and data-independent, so we run them on CPU. The
        # pipeline then does `coords.repeat(...)` (CPU op, fine), and
        # the `_NeuronTransformerWrapper.forward()` below moves coords
        # to Neuron+contiguous right before the compiled call — outside
        # the compile boundary. See ltx2-omni-DECISIONS.md.
        _patch_coord_prep_to_cpu(inner_transformer)
        # Wrap with `_NeuronTransformerWrapper` so the data-movement
        # (CPU→Neuron coord move, contiguity) stays eager (outside
        # `torch.compile`). The wrapper proxies `.config`, `.dtype`,
        # `.rope`, `.audio_rope`, and `.cache_context` so the base
        # `LTX2Pipeline.forward()` sees an LTX2-compatible object.
        # Calling `.compile()` on the wrapper compiles the INNER DiT,
        # leaving the wrapper's data-prep eager.
        self.transformer = _NeuronTransformerWrapper(inner_transformer)
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
    # Encode prompt — text encoder stays on CPU
    # ------------------------------------------------------------------
    def _get_gemma_prompt_embeds(
        self,
        prompt,
        num_videos_per_prompt: int = 1,
        max_sequence_length: int = 1024,
        scale_factor: int = 8,
        device=None,
        dtype=None,
    ):
        """Same as base but runs the Gemma3 encoder on CPU.

        The base class moves the tokenized input_ids to `self.device`
        (which is the Neuron device) and then calls `self.text_encoder`.
        Since our text_encoder is on CPU (we kept it there for v1),
        the dtype/device mismatch causes
        `RuntimeError: Expected self.dtype() == dst.dtype()`.

        Fix: feed the encoder CPU tensors, then move the output
        embeddings to the requested device. Mirrors NeuronWanPipeline's
        _encode_prompt() pattern (see neuron_wan_pipeline.py).
        """
        import torch as _torch

        target_device = device or self.device
        target_dtype = dtype or self.text_encoder.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt

        if getattr(self, "tokenizer", None) is not None:
            self.tokenizer.padding_side = "left"
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

        prompt = [p.strip() for p in prompt]
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        # Keep on CPU — the encoder is on CPU.
        text_input_ids = text_inputs.input_ids
        prompt_attention_mask = text_inputs.attention_mask

        with _torch.no_grad():
            text_encoder_outputs = self.text_encoder(
                input_ids=text_input_ids,
                attention_mask=prompt_attention_mask,
                output_hidden_states=True,
            )
        text_encoder_hidden_states = text_encoder_outputs.hidden_states
        text_encoder_hidden_states = _torch.stack(text_encoder_hidden_states, dim=-1)
        sequence_lengths = prompt_attention_mask.sum(dim=-1)

        # _pack_text_embeds is base-class, expects to be called with
        # device kwarg — pass CPU so the packing happens on CPU, then
        # we move the final embeds to the target device below.
        prompt_embeds = self._pack_text_embeds(
            text_encoder_hidden_states,
            sequence_lengths,
            device=_torch.device("cpu"),
            padding_side=self.tokenizer.padding_side,
            scale_factor=scale_factor,
        )

        # Two-step: cast dtype on CPU first (cheap; small tensor), then
        # move to Neuron. Avoids the combined .to(device, dtype) trap.
        prompt_embeds = prompt_embeds.to(dtype=target_dtype)
        prompt_embeds = prompt_embeds.to(device=target_device)

        # The base class `forward()` does `(1 - prompt_attention_mask.to(prompt_embeds.dtype)) * -1e6`.
        # That `.to(...)` on a Neuron tensor with int→fp cast is itself
        # the dtype trap. Pre-cast the mask to the prompt-embeds dtype
        # on CPU so the base class's .to() becomes a no-op.
        attention_mask = prompt_attention_mask.to(dtype=target_dtype)
        attention_mask = attention_mask.to(device=target_device)
        return prompt_embeds, attention_mask

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
    # prepare_latents — force bf16 + CPU generator
    # ------------------------------------------------------------------
    def prepare_latents(self, *args, **kwargs):
        """Force latents to the transformer dtype (bf16) and use a CPU
        generator.

        The base `forward()` does `latent_model_input.to(prompt_embeds.dtype)`
        in the denoising loop (pipeline_ltx2.py:1162). If latents come back
        in fp32 (the base default) and prompt_embeds is bf16, that `.to()`
        is a real dtype cast on a Neuron tensor and trips
        `RuntimeError: Expected self.dtype() == dst.dtype()`. By forcing
        latents to bf16 here, the later `.to()` becomes a no-op.

        Also: Neuron has no Generator — fall back to a CPU generator with
        the same seed. Mirrors NeuronWanPipeline.prepare_latents.

        The base `forward()` calls this positionally:
            prepare_latents(batch_size, num_channels, height, width,
                            num_frames, noise_scale, dtype, device,
                            generator, latents)
        so `dtype` is positional arg index 6 and `generator` is index 8.
        We rewrite those in-place rather than passing kwargs (which would
        collide with the positional values).
        """
        transformer_dtype = (
            self.transformer.dtype
            if (self.transformer is not None and hasattr(self.transformer, "dtype"))
            else torch.bfloat16
        )
        args = list(args)
        # dtype @ positional index 6
        if len(args) > 6:
            args[6] = transformer_dtype
        elif "dtype" in kwargs:
            kwargs["dtype"] = transformer_dtype
        else:
            kwargs["dtype"] = transformer_dtype
        # generator @ positional index 8 — force to CPU
        if len(args) > 8 and isinstance(args[8], torch.Generator) \
                and args[8].device.type != "cpu":
            args[8] = torch.Generator(device="cpu").manual_seed(args[8].initial_seed())
        elif isinstance(kwargs.get("generator"), torch.Generator) \
                and kwargs["generator"].device.type != "cpu":
            g = kwargs["generator"]
            kwargs["generator"] = torch.Generator(device="cpu").manual_seed(g.initial_seed())
        return super().prepare_latents(*args, **kwargs)

    # ------------------------------------------------------------------
    # Compile + load_weights
    # ------------------------------------------------------------------
    def prepare_audio_latents(self, *args, **kwargs):
        """Force audio latents to the transformer dtype + CPU generator.

        Same dtype trap as video latents: the denoising loop does
        `audio_latent_model_input.to(prompt_embeds.dtype)`
        (pipeline_ltx2.py:1166). The base is called with `dtype=fp32`
        as a kwarg here (unlike video which is positional), so we just
        override the kwarg.
        """
        transformer_dtype = (
            self.transformer.dtype
            if (self.transformer is not None and hasattr(self.transformer, "dtype"))
            else torch.bfloat16
        )
        kwargs["dtype"] = transformer_dtype
        g = kwargs.get("generator")
        if isinstance(g, torch.Generator) and g.device.type != "cpu":
            kwargs["generator"] = torch.Generator(device="cpu").manual_seed(g.initial_seed())
        return super().prepare_audio_latents(*args, **kwargs)

    @staticmethod
    def _pack_audio_latents(
        latents: torch.Tensor,
        patch_size: int | None = None,
        patch_size_t: int | None = None,
    ) -> torch.Tensor:
        """Override base `_pack_audio_latents` to do the transpose/permute
        + flatten on CPU.

        The base implementation does:
            latents = latents.transpose(1, 2).flatten(2, 3)        # else branch
            latents = latents.permute(0,2,4,1,3,5).flatten(3,5).flatten(1,2)  # patched branch

        Both transpose and permute produce non-contiguous tensors. On
        Neuron, neither `.contiguous()` nor `.flatten()` on a
        non-contiguous device tensor work — the lazy backend rejects
        with "Expected self.is_contiguous() to be true". The fix is to
        do this layout reorganization on CPU and move the result back.
        Latents are small (audio mel-spec latent), so this round-trip
        is cheap. See ltx2-omni-DECISIONS.md.
        """
        orig_device = latents.device
        latents = latents.to(device="cpu")
        if patch_size is not None and patch_size_t is not None:
            batch_size, num_channels, latent_length, latent_mel_bins = latents.shape
            post_patch_latent_length = latent_length // patch_size_t
            post_patch_mel_bins = latent_mel_bins // patch_size
            latents = latents.reshape(
                batch_size, -1, post_patch_latent_length, patch_size_t,
                post_patch_mel_bins, patch_size,
            )
            latents = latents.permute(0, 2, 4, 1, 3, 5).contiguous()
            latents = latents.flatten(3, 5).flatten(1, 2)
        else:
            latents = latents.transpose(1, 2).contiguous()
            latents = latents.flatten(2, 3)
        return latents.to(device=orig_device)

    @staticmethod
    def _pack_video_latents(
        latents: torch.Tensor,
        patch_size: int = 1,
        patch_size_t: int = 1,
    ) -> torch.Tensor:
        """Override base `_pack_video_latents` to do the layout ops on CPU.

        Same Neuron contiguity issue as `_pack_audio_latents`. The base
        does a permute+flatten that produces a non-contiguous source
        for `flatten`, which Neuron's lazy backend rejects. Round-trip
        via CPU; latents are small.
        """
        orig_device = latents.device
        latents = latents.to(device="cpu")
        batch_size, num_channels, num_frames, height, width = latents.shape
        post_patch_num_frames = num_frames // patch_size_t
        post_patch_height = height // patch_size
        post_patch_width = width // patch_size
        latents = latents.reshape(
            batch_size, -1,
            post_patch_num_frames, patch_size_t,
            post_patch_height, patch_size,
            post_patch_width, patch_size,
        )
        latents = latents.permute(0, 2, 4, 6, 1, 3, 5, 7).contiguous()
        latents = latents.flatten(4, 7).flatten(1, 3)
        return latents.to(device=orig_device)

    def compile_transformer(self, *args, **kwargs):
        """Compile the DiT transformer for Neuron.

        Use `fullgraph=False` to let Dynamo split the giant LTX-2 DiT
        forward into smaller subgraphs. With `fullgraph=True`, the
        single big NEFF compile of one of the transformer blocks
        triggers what looks like a SIGKILL (silent worker death after
        ~1-2 minutes of `Acquired compilation lock`). Splitting the
        graph keeps each NEFF small enough to compile reliably.

        Trade-off: more graph breaks means more eager-mode dispatch
        overhead, slightly slower per-step. But correctness over
        performance for v1.
        """
        t_kwargs = copy.deepcopy(kwargs)
        t_kwargs.setdefault("fullgraph", False)
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

    def forward(self, req):
        """Skip the engine's dummy_run, otherwise pass to the base
        pipeline's forward.

        The vLLM-Omni `DiffusionEngine.__init__` always calls
        `_dummy_run()` even when the pipeline sets `skip_warmup=True`.
        That dummy run sends a request with `prompt="dummy run"` and
        `request_ids=["dummy_req_id"]`, which our pipeline can identify
        and short-circuit. The base LTX2 pipeline's forward expects
        coords/encoders to be fully wired; running it in dummy_run mode
        with our CPU-resident encoders + Neuron transformer surfaces
        the "non-contiguous Device Tensor" error from
        `vllm_neuron.compile.backend` (some intermediate tensor produced
        by the dummy-shaped pipeline doesn't make the contiguity
        contract). Skipping the dummy run avoids that — real requests
        from `omni.generate(...)` go through the same path but with
        properly-shaped inputs and the issue doesn't reproduce. The
        Wan and Helios Neuron pipelines do the same.

        For real requests, delegate to the base `LTX2Pipeline.forward`.
        """
        from vllm_omni.diffusion.models.wan2_2.pipeline_wan2_2 import DiffusionOutput
        if getattr(self, "skip_warmup", False) and getattr(req, "request_ids", None) == ["dummy_req_id"]:
            prompt = req.prompts[0] if isinstance(req.prompts[0], str) else req.prompts[0].get("prompt")
            if prompt == "dummy run":
                logger.info("Skipping warmup request on Neuron LTX-2 pipeline")
                return DiffusionOutput(output=None)
        return super().forward(req)

    def load_weights(self, weights=None):
        """Forward the framework's (name, tensor) iterable to the inner DiT.

        The framework's diffusers_loader walks `self.weights_sources` (set
        in __init__ to point at the transformer subfolder), iterates the
        safetensors files, and calls this method with an iterable of
        `(name, tensor)` tuples prefixed with `transformer.`. The base
        LTX2VideoTransformer3DModel.load_weights handles vLLM TP sharding
        internally — we strip the prefix before forwarding (so the
        transformer's parameter dict matches), then we add the prefix
        back to the names of the loaded set we return.

        Naming subtlety with the `_NeuronTransformerWrapper`: because we
        wrap the DiT in a wrapper named `inner`, the framework's
        `named_parameters()` sees `transformer.inner.<param>`. So we
        return names with the `transformer.inner.` prefix to match what
        the framework's strict-load check expects, while feeding the
        inner DiT's `load_weights()` an iterable with the leading
        `transformer.` stripped (matching its own parameter dict).
        """
        if weights is None:
            return set()
        if self.transformer is None or not hasattr(self.transformer, "load_weights"):
            return set()

        in_prefix = "transformer."           # what the framework feeds us
        out_prefix = "transformer.inner."    # what named_parameters() shows

        def _strip_prefix(it):
            for name, tensor in it:
                if name.startswith(in_prefix):
                    name = name[len(in_prefix):]
                yield name, tensor

        # `self.transformer` is `_NeuronTransformerWrapper` whose
        # `.load_weights()` proxies to `inner.load_weights()`.
        loaded = self.transformer.load_weights(_strip_prefix(weights))
        # Re-prefix with `transformer.inner.` so the framework's
        # "expected vs loaded" set diff matches `named_parameters()`.
        return {out_prefix + n for n in (loaded or set())}

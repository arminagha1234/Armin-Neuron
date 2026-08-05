"""Device-agnostic patches for FireRedTTS v1.

The upstream repo hardcodes CUDA in two places, which breaks CPU / Neuron runs:

  1. ``FireRedTTS.__init__(..., device="cuda")`` default.
  2. ``FireRedTTS.extract_spk_embeddings`` calls ``audio_resampled.to(device="cuda")``
     regardless of the model's actual device.

Importing this module and calling :func:`apply_device_patches` rebinds
``extract_spk_embeddings`` so the reference audio is sent to ``self.device``
instead of a hardcoded ``"cuda"``. Everything else in the class already threads
``self.device`` through correctly.

Import this BEFORE constructing ``FireRedTTS`` and pass an explicit ``device``.
"""
from __future__ import annotations

import torch


def patch_torch_load_cpu():
    """Default ``torch.load(..., map_location="cpu")`` when CUDA is unavailable.

    Several checkpoints (e.g. fireredtts_speaker.bin) were saved from CUDA tensors, and
    upstream ``speaker.py`` calls ``torch.load(ckpt_path)`` with no map_location, which
    raises on a CPU/Neuron box. Wrapping torch.load to fall back to CPU fixes every such
    call without editing the vendored source.
    """
    import torch

    if getattr(torch.load, "_firered_patched", False):
        return
    _orig_load = torch.load

    def _load(*args, **kwargs):
        if kwargs.get("map_location") is None and not torch.cuda.is_available():
            kwargs["map_location"] = "cpu"
        return _orig_load(*args, **kwargs)

    _load._firered_patched = True
    torch.load = _load


def patch_torchaudio_io():
    """Route ``torchaudio.load``/``save`` through soundfile.

    torchaudio >= 2.9 delegates I/O to ``torchcodec`` (which needs ffmpeg libs). The
    container ships torchaudio 2.11 without torchcodec, so ``torchaudio.load`` raises.
    soundfile (already a dependency) reads/writes WAV fine; ``torchaudio.functional``
    DSP ops (resample, mel) are pure and unaffected.
    """
    import numpy as np
    import torch
    import torchaudio

    if getattr(torchaudio.load, "_firered_patched", False):
        return

    def _load(path, *a, **k):
        import soundfile as sf

        data, sr = sf.read(str(path), dtype="float32", always_2d=True)  # [T, C]
        return torch.from_numpy(data.T.copy()), sr  # -> [C, T], sr

    def _save(path, wav, sr, *a, **k):
        import soundfile as sf

        arr = wav.detach().cpu().numpy()
        if arr.ndim == 2:  # [C, T] -> [T, C]
            arr = arr.T
        sf.write(str(path), arr, int(sr))

    _load._firered_patched = True
    torchaudio.load = _load
    torchaudio.save = _save


def patch_flow_conformer_contiguous():
    """Make the conformer's relative-position attention lower on Neuron.

    ``RelPositionMultiHeadedAttention.rel_shift`` returns a heavily strided view
    (from cat/view/slice/view_as). Adding that non-contiguous ``matrix_bd`` to the
    contiguous ``matrix_ac`` crashes neuronx-cc (internal error on the strided add).
    Forcing the shift result contiguous materializes it first, which lowers fine.
    """
    from fireredtts.modules.flow.conformer import RelPositionMultiHeadedAttention

    if getattr(RelPositionMultiHeadedAttention.rel_shift, "_firered_patched", False):
        return
    _orig = RelPositionMultiHeadedAttention.rel_shift

    def rel_shift(self, x):
        return _orig(self, x).contiguous()

    rel_shift._firered_patched = True
    RelPositionMultiHeadedAttention.rel_shift = rel_shift


def _to_cpu(obj):
    """Recursively move tensors in obj (incl. HF ModelOutput / tuples) to CPU."""
    if torch.is_tensor(obj):
        return obj.to("cpu")
    try:
        from transformers.utils import ModelOutput

        if isinstance(obj, ModelOutput):
            for k in list(obj.keys()):
                obj[k] = _to_cpu(obj[k])
            return obj
    except Exception:
        pass
    if isinstance(obj, (list, tuple)):
        return type(obj)(_to_cpu(x) for x in obj)
    return obj


def patch_gpt_fixed_shape(tts, device="neuron", bucket=128, num_return_sequences=7):
    """Run the 30-layer GPT-2 transformer on ``device`` with FIXED (bucketed) shapes.

    The HF ``.generate()`` loop normally grows the KV cache one token per step, so the
    transformer forward changes shape every step → a recompile every step on Neuron. We
    instead:
      * force ``use_cache=False`` so every step re-runs the FULL sequence (the prefill
        path in GPT2InferenceModel), and
      * left-pad ``inputs_embeds`` + ``attention_mask`` up to a multiple of ``bucket`` so
        only a few distinct shapes are ever compiled. Real tokens stay right-aligned, so
        HF's ``logits[:, -1]`` is still the true last token, and the pad positions are
        masked out (validated: last-position hidden is bit-identical vs unpadded).

    ``gpt.wpe`` is a no-op (positions are baked into ``emb`` by GPT2InferenceModel), so
    left-padding the already-embedded sequence is safe.
    """
    import types
    import torch.nn.functional as F

    gpt2 = tts.gpt.gpt
    gpt2.to(device)
    for m in gpt2.modules():  # move stray registered tensors that .to() misses
        for k, v in list(vars(m).items()):
            if torch.is_tensor(v) and v.device.type != device:
                setattr(m, k, v.to(device))
    # gpt.wte IS the shared mel_embedding (init_gpt_for_inference sets gpt.wte =
    # mel_embedding). GPT2InferenceModel builds `emb` from it ON CPU, and the transformer
    # never uses wte because we always pass inputs_embeds — so keep it on CPU.
    if hasattr(gpt2, "wte") and hasattr(gpt2.wte, "to"):
        gpt2.wte.to("cpu")

    real_forward = gpt2.forward

    def forward(*args, **kwargs):
        emb = kwargs.get("inputs_embeds")
        am = kwargs.get("attention_mask")
        if emb is not None:
            s = emb.shape[1]
            p = ((s + bucket - 1) // bucket) * bucket  # round up to bucket multiple
            padn = p - s
            if padn > 0:
                emb = F.pad(emb, (0, 0, padn, 0))  # left-pad time dim (right-align real)
                if am is not None:
                    am = F.pad(am, (padn, 0), value=0)  # mask the pad positions
            kwargs["inputs_embeds"] = emb.to(device)
            kwargs["attention_mask"] = am.to(device) if am is not None else None
            kwargs["position_ids"] = None  # gpt.wpe is null; position_ids unused
        kwargs["use_cache"] = False
        args = tuple(a.to(device) if torch.is_tensor(a) else a for a in args)
        return _to_cpu(real_forward(*args, **kwargs))

    gpt2.forward = forward

    # Force generate down the full-sequence (use_cache=False) path, make the candidate
    # count configurable, and tolerate sequences with no EOS.
    stop = tts.config["gpt"]["gpt_stop_audio_token"]

    def do_gpt_inference(self, spk_gpt, text_tokens):
        with torch.no_grad():
            gpt_codes = self.gpt.generate(
                cond_latents=spk_gpt,
                text_inputs=text_tokens,
                input_tokens=None,
                do_sample=True,
                top_p=0.85,
                top_k=30,
                temperature=0.75,
                num_return_sequences=num_return_sequences,
                num_beams=1,
                length_penalty=1.0,
                repetition_penalty=2.0,
                output_attentions=False,
                use_cache=False,
            )
        seqs = []
        for seq in gpt_codes:
            idx = (seq == stop).nonzero(as_tuple=True)[0]
            seqs.append(seq[: idx[0]] if len(idx) > 0 else seq)
        sorted_seqs = sorted(seqs, key=len)
        pick = sorted_seqs[1] if len(sorted_seqs) > 1 else sorted_seqs[0]
        return pick.unsqueeze(0)

    tts.do_gpt_inference = types.MethodType(do_gpt_inference, tts)


def patch_gpt_kv_cache_bucketed(tts, device="neuron", bucket=256, num_return_sequences=7,
                                prefill_bucket=64, compile_fwd=False):
    """Fast GPT AR decode on Neuron: a fixed-length KV cache resident on the device.

    Unlike ``patch_gpt_fixed_shape`` (which sets use_cache=False and re-runs the WHOLE
    sequence every step, O(n^2)), this keeps HF's incremental KV cache (use_cache=True) so
    each decode step processes only the ONE new token, and:

      * the cache tensors stay on the NeuronCore across steps (never marshalled to CPU),
      * the past is left-padded to a multiple of ``bucket`` before each transformer call so
        only a few fixed shapes are ever compiled, then the returned present is un-padded
        back to its true length for HF's bookkeeping,
      * pad positions are masked out via a 2D attention_mask (real keys right-aligned).

    GPT2 uses the legacy tuple cache: ``past_key_values`` is a per-layer ``(key, value)``,
    each ``[B, H, L, d]``, concatenated along dim=-2. ``gpt.wte`` (the shared mel_embedding)
    stays on CPU; the transformer only ever sees ``inputs_embeds``.

    ``prefill_bucket`` pads the (short) prompt to its own small multiple so the prefill
    forward — the time-to-first-token (TTFT) critical path — stays cheap instead of being
    padded up to the (large) decode ``bucket``. TTFT for the run is recorded on
    ``tts._gpt_stats``.

    NOTE: this is a per-request, in-process cache — NOT cross-request prefix caching (APC).
    """
    import time
    import types
    import torch.nn.functional as F

    stats = {"ttft": None, "prefill_s": None, "decode_steps": 0, "decode_s": 0.0}
    tts._gpt_stats = stats
    _t = {"prefill_start": None}

    gpt2 = tts.gpt.gpt
    gpt2.to(device)
    for mod in gpt2.modules():
        for k, v in list(vars(mod).items()):
            if torch.is_tensor(v) and v.device.type != device:
                setattr(mod, k, v.to(device))
    if hasattr(gpt2, "wte") and hasattr(gpt2.wte, "to"):
        gpt2.wte.to("cpu")

    def _strip(past, padn):
        # un-pad each layer's (k, v) along the sequence dim (dim=-2); keep on device.
        if padn <= 0 or past is None:
            return past
        return tuple(
            (k[..., padn:, :].contiguous(), v[..., padn:, :].contiguous()) for (k, v) in past
        )

    real_forward = gpt2.forward
    if compile_fwd:
        # Fuse the ~300 eager op-dispatches of the 30-layer forward into ONE Neuron graph.
        # Eager mode dispatches each aten op to the device individually (~80 ms/step of
        # launch overhead for a 1-token decode); torch.compile(backend="neuron") traces the
        # whole forward to a single NEFF -> one launch per fixed shape.
        real_forward = torch.compile(real_forward, backend="neuron", dynamic=False)

    def forward(*args, **kwargs):
        emb = kwargs.get("inputs_embeds")
        past = kwargs.get("past_key_values")
        b = emb.shape[0]
        kwargs["position_ids"] = None  # gpt.wpe is null; positions baked into emb
        kwargs["use_cache"] = True

        if past is None:
            # PREFILL: pad the (short) prompt to its own small bucket -> cheap TTFT.
            _t["prefill_start"] = time.time()
            s = emb.shape[1]
            p = ((s + prefill_bucket - 1) // prefill_bucket) * prefill_bucket
            padn = p - s
            emb_d = F.pad(emb, (0, 0, padn, 0)).to(device) if padn > 0 else emb.to(device)
            # arange comparison (no dynamic in-place slice-assign): valid where idx >= padn
            idx = torch.arange(p, device=device)
            am = (idx >= padn).unsqueeze(0).expand(b, -1).to(torch.long)
            kwargs["inputs_embeds"] = emb_d
            kwargs["attention_mask"] = am
            out = real_forward(*args, **kwargs)
            hs = out.last_hidden_state
            out.last_hidden_state = (hs[:, padn:, :] if padn > 0 else hs).to("cpu")
            out.past_key_values = _strip(out.past_key_values, padn)  # stays on device
            stats["prefill_s"] = time.time() - _t["prefill_start"]
            return out

        # DECODE: 1 new token; past = per-layer (k, v) each [B, H, L, d] on device.
        length = past[0][0].shape[-2]
        p = ((length + bucket - 1) // bucket) * bucket
        padn = p - length
        if padn > 0:
            past = tuple((F.pad(k, (0, 0, padn, 0)), F.pad(v, (0, 0, padn, 0))) for (k, v) in past)
        # arange comparison over past(p)+new(1): valid where idx >= padn (real past + new)
        idx = torch.arange(p + 1, device=device)
        am = (idx >= padn).unsqueeze(0).expand(b, -1).to(torch.long)
        kwargs["past_key_values"] = past
        kwargs["attention_mask"] = am
        kwargs["inputs_embeds"] = emb.to(device)
        _t0 = time.time()
        out = real_forward(*args, **kwargs)
        out.last_hidden_state = out.last_hidden_state.to("cpu")
        out.past_key_values = _strip(out.past_key_values, padn)  # -> true length L+1, on device
        dt = time.time() - _t0
        stats["decode_steps"] += 1
        stats["decode_s"] += dt
        if stats["ttft"] is None:  # first decoded token = time from prefill start
            stats["ttft"] = time.time() - _t["prefill_start"]
        return out

    gpt2.forward = forward

    stop = tts.config["gpt"]["gpt_stop_audio_token"]

    def do_gpt_inference(self, spk_gpt, text_tokens):
        with torch.no_grad():
            gpt_codes = self.gpt.generate(
                cond_latents=spk_gpt, text_inputs=text_tokens, input_tokens=None,
                do_sample=True, top_p=0.85, top_k=30, temperature=0.75,
                num_return_sequences=num_return_sequences, num_beams=1,
                length_penalty=1.0, repetition_penalty=2.0, output_attentions=False,
            )
        seqs = []
        for seq in gpt_codes:
            idx = (seq == stop).nonzero(as_tuple=True)[0]
            seqs.append(seq[: idx[0]] if len(idx) > 0 else seq)
        s = sorted(seqs, key=len)
        return (s[1] if len(s) > 1 else s[0]).unsqueeze(0)

    tts.do_gpt_inference = types.MethodType(do_gpt_inference, tts)


def apply_device_patches():
    """Monkeypatch FireRedTTS to honor self.device instead of hardcoded cuda."""
    patch_torch_load_cpu()
    patch_torchaudio_io()
    from fireredtts.fireredtts import FireRedTTS
    from fireredtts.utils.utils import load_audio

    def extract_spk_embeddings(self, prompt_wav):
        # 16 kHz mono for the ECAPA-TDNN speaker encoder.
        _, _, audio_resampled = load_audio(audiopath=prompt_wav, sampling_rate=16000)
        # Original hardcodes .to(device="cuda"); use the model's device instead.
        spk_embeddings = self.speaker_extractor(
            audio_resampled.to(device=self.device)
        ).unsqueeze(0)
        return spk_embeddings

    FireRedTTS.extract_spk_embeddings = extract_spk_embeddings
    return FireRedTTS


def bypass_text_normalizer():
    """Fallback when WeTextProcessing / pynini can't be installed (e.g. Python 3.12,
    where pynini has no wheel and building from source fails).

    ``fireredtts/modules/text_normalizer/normalize.py`` does ``from tn.chinese.normalizer
    import Normalizer`` at IMPORT time, and that module is imported transitively when you
    ``import fireredtts.fireredtts``. So a runtime monkeypatch alone is too late — we must
    stub the ``tn`` package in ``sys.modules`` BEFORE any fireredtts import happens, then
    swap ``VoiceBpeTokenizer``'s normalizer for a light-weight one.

    Call this FIRST, before ``apply_device_patches()`` or importing ``FireRedTTS``.

    The lite normalizer does basic cleanup + zh/en detection but skips number/date
    expansion, so spell out numbers in the input text when using this path.
    """
    import re
    import sys
    import types

    # 1) Stub the `tn` package so `from tn.chinese.normalizer import Normalizer` resolves.
    class _DummyNormalizer:
        def __init__(self, *a, **k):
            pass

        def normalize(self, text):
            return text

    for name in ("tn", "tn.chinese", "tn.chinese.normalizer", "tn.english", "tn.english.normalizer"):
        if name not in sys.modules:
            mod = types.ModuleType(name)
            if name.endswith("normalizer"):
                mod.Normalizer = _DummyNormalizer
            sys.modules[name] = mod

    # 2) Now it's safe to import the tokenizer module; patch it to skip the real normalizer.
    from fireredtts.modules.tokenizer import tokenizer as tok_mod

    def _detect_lang(text: str) -> str:
        return "zh" if re.search(r"[\u4e00-\u9fff]", text) else "en"

    class _LiteNormalizer:
        def tn(self, text):
            text = text.strip()
            lang = _detect_lang(text)
            if lang == "en":
                text = text.lower()
            text = re.sub(r"\s+", " ", text).strip()
            if len(text) > 0 and text[-1] not in ".?！？。":
                text = text + "."
            return text, lang

    def patched_init(self):
        from fireredtts.modules.tokenizer.whisper_tokenizer import get_tokenizer

        self.tokenizer = get_tokenizer(multilingual=True)
        self.tn_engine = _LiteNormalizer()

    tok_mod.VoiceBpeTokenizer.__init__ = patched_init
    return tok_mod.VoiceBpeTokenizer

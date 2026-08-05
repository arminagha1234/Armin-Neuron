"""FireRedTTS v1 — sentence-streaming synthesis on AWS Trainium (native PyTorch).

FireRedTTS v1 is NOT a natively streaming model: its flow-matching decoder (a full-context
conformer + GroupNorm over the whole time axis + a CFM ODE over the whole mel) and the
BigVGAN vocoder are non-causal, whole-utterance. True sub-sentence streaming would need the
separate **FireRedTTS-1S** model (a causal semantic->acoustic decoder) — see the README's
"Streaming" section.

BUT v1's own ``synthesize()`` already splits text into sentence-sized chunks (``text_split``,
merged to >~30 chars) and synthesizes each chunk INDEPENDENTLY, then concatenates. This
script exposes that as a generator that yields each chunk's waveform the moment it is ready:

  - **time-to-first-audio (TTFA)** drops from "the whole paragraph" to "the first chunk", and
  - synthesis of later chunks overlaps playback of earlier ones.

The concatenated output is identical to non-streaming ``synthesize()`` (same per-chunk
``synthesize_base``); only the *delivery* is incremental. GPT stays compiled on the
NeuronCore; flow+vocoder run on CPU (the recommended hybrid). For a single short sentence
there is only one chunk, so streaming == non-streaming — the win is for multi-sentence text.

Usage (inside the native-PyTorch Neuron DLC container):
    PYTHONPATH=$PWD/FireRedTTS FIRERED_MODEL=$PWD/pretrained_models \
    python stream_fireredtts.py --prompt-wav FireRedTTS/examples/prompt_1.wav \
        --offload gpt --gpt-compile --warmup --no-tn --out stream.wav
"""
import argparse
import os
import time

import torch
import torch_neuronx  # noqa: F401  (registers the native "neuron" device)

from firered_patch import (
    apply_device_patches,
    bypass_text_normalizer,
    patch_gpt_kv_cache_bucketed,
)
from run_fireredtts_neuron import find_config

NEURON = "neuron"

DEFAULT_TEXT = (
    "Hello from Trainium. This is a longer streaming synthesis test running on the NeuronCore. "
    "Each sentence is generated independently and delivered the moment it is ready. "
    "That means you hear the beginning while the rest is still being computed. "
    "The time to first audio no longer depends on the length of the whole passage. "
    "It depends only on the first sentence. "
    "That is the core benefit of streaming synthesis."
)


def _gpt_time(tts):
    """GPT prefill+decode seconds for the last chunk, from the KV-cache patch's stats
    (0.0 if unavailable). Reset per chunk by the caller."""
    st = getattr(tts, "_gpt_stats", None)
    if not st:
        return 0.0
    return (st.get("prefill_s") or 0.0) + (st.get("decode_s") or 0.0)


def _gpt_reset(tts):
    st = getattr(tts, "_gpt_stats", None)
    if st:
        st.update({"ttft": None, "prefill_s": None, "decode_steps": 0, "decode_s": 0.0})


def stream_chunks(tts, prompt_wav, text, lang):
    """Yield (idx, n_chunks, text_chunk, wav_cpu, gpt_s) per sentence chunk, in order. Uses
    the stock per-chunk ``synthesize_base`` so the concatenation matches non-streaming
    ``synthesize()`` exactly — only the delivery is incremental. ``gpt_s`` is the GPT
    prefill+decode time so the caller can split GPT (NeuronCore) from token2wav (CPU)."""
    from fireredtts.modules.text_normalizer.utils import text_split

    chunks = text_split(text=text)
    for i, sub in enumerate(chunks):
        _gpt_reset(tts)
        wav = tts.synthesize_base(prompt_wav=prompt_wav, text=sub, lang=lang)
        yield i, len(chunks), sub, wav.detach().cpu(), _gpt_time(tts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("FIRERED_MODEL", "./pretrained_models"))
    ap.add_argument("--prompt-wav", required=True, help="Reference voice .wav to clone.")
    ap.add_argument("--text", default=DEFAULT_TEXT, help="Multi-sentence text (streaming helps most here).")
    ap.add_argument("--lang", default="en", choices=["zh", "en", "auto"])
    ap.add_argument("--offload", default="gpt", help="'gpt' (recommended hybrid) or '' for all-CPU.")
    ap.add_argument("--gpt-bucket", type=int, default=256)
    ap.add_argument("--gpt-seqs", type=int, default=7)
    ap.add_argument("--gpt-prefill-bucket", type=int, default=64)
    ap.add_argument("--gpt-compile", action="store_true", help="torch.compile the GPT decode step (fast).")
    ap.add_argument("--warmup", action="store_true", help="Compile/cache graphs on a throwaway chunk first.")
    ap.add_argument("--no-tn", action="store_true", help="Bypass the WeTextProcessing/pynini normalizer.")
    ap.add_argument("--out", default="stream.wav")
    args = ap.parse_args()

    def _tn_available():
        try:
            import tn  # noqa: F401
            return True
        except Exception:
            return False

    if args.no_tn or not _tn_available():
        bypass_text_normalizer()
    args.prompt_wav = os.path.abspath(args.prompt_wav)
    args.out = os.path.abspath(args.out)
    args.model = os.path.abspath(args.model)

    apply_device_patches()
    from fireredtts.fireredtts import FireRedTTS
    import fireredtts as _f

    os.chdir(os.path.dirname(os.path.abspath(next(iter(_f.__path__)))))  # repo root (asset paths)

    cfg = find_config(args.model)
    print(f"[stream] loading FireRedTTS on CPU (config={cfg})...")
    tts = FireRedTTS(config_path=cfg, pretrained_path=args.model, device="cpu")

    if "gpt" in args.offload:
        print(f"[stream] offloading GPT -> NeuronCore (compile={args.gpt_compile})")
        patch_gpt_kv_cache_bucketed(
            tts, NEURON, bucket=args.gpt_bucket, num_return_sequences=args.gpt_seqs,
            prefill_bucket=args.gpt_prefill_bucket, compile_fwd=args.gpt_compile,
        )

    if args.warmup:
        # Warm up on a FULL-LENGTH sentence, not a short phrase: this compiles the GPT
        # graphs AND — critically — primes the CPU flow+vocoder oneDNN conv primitives for a
        # representative mel length. Without this, the first real chunk pays a large one-time
        # primitive-selection cost (~20-50s) that dominates TTFA.
        print("[stream] warmup (compiling GPT graphs + priming CPU flow/vocoder primitives)...")
        warm_text = ("This is a warmup sentence that is about as long as a typical sentence "
                     "in the passage so the vocoder primitives are primed.")
        with torch.no_grad():
            for _ in stream_chunks(tts, args.prompt_wav, warm_text, args.lang):
                pass

    from fireredtts.modules.text_normalizer.utils import text_split
    n = len(text_split(text=args.text))
    print(f"[stream] synthesizing {n} chunk(s), streaming as each is ready...\n")

    pieces, ttfa, t0, prev = [], None, time.time(), time.time()
    with torch.no_grad():
        for i, ntot, sub, wav, gpt_s in stream_chunks(tts, args.prompt_wav, args.text, args.lang):
            now = time.time()
            ready, synth = now - t0, now - prev
            prev = now
            if ttfa is None:
                ttfa = ready
            dur = wav.shape[-1] / 24000
            t2w_s = max(synth - gpt_s, 0.0)  # flow + vocoder (CPU) + spk/mel extraction
            pieces.append(wav)
            print(f"[stream]   chunk {i+1}/{ntot}: ready@{ready:5.2f}s  synth={synth:5.2f}s "
                  f"(gpt {gpt_s:4.2f}s / token2wav {t2w_s:4.2f}s)  audio={dur:5.2f}s  "
                  f"rtf={synth / max(dur, 1e-9):.2f}  | {sub[:40]!r}")
    total = time.time() - t0

    wav = torch.concat(pieces, axis=-1)
    aud = wav.shape[-1] / 24000
    print(f"\n[stream] TTFA (time-to-first-audio):   {ttfa * 1000:6.0f} ms   <- wait before playback starts")
    print(f"[stream] non-streaming wait would be:  {total * 1000:6.0f} ms   <- you wait for the whole clip today")
    if ttfa:
        print(f"[stream] => first-audio ~{total / ttfa:.1f}x sooner")
    print(f"[stream] total synth {total:.2f}s for {aud:.2f}s of audio (overall rtf {total / aud:.2f})")

    import torchaudio

    torchaudio.save(args.out, wav, 24000)
    print(f"[stream] wrote {args.out}")


if __name__ == "__main__":
    main()

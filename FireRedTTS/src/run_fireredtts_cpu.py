"""FireRedTTS v1 — CPU reference run.

The correctness oracle: runs the full pipeline (GPT AR decode -> flow-matching ->
BigVGAN vocoder) on CPU and writes a 24 kHz .wav. Use this to (a) confirm the model
and checkpoints load and synthesize at all, and (b) produce a reference waveform to
compare the Neuron offload against.

Usage:
    python run_fireredtts_cpu.py \
        --model ./pretrained_models \
        --prompt-wav ./FireRedTTS/examples/prompt_1.wav \
        --text "Hello from Trainium." \
        --lang en \
        --out cpu.wav

Notes:
  - CPU synthesis is slow (the 30-layer GPT decodes 7 candidate sequences); expect
    tens of seconds to a few minutes depending on text length and core count.
  - --prompt-wav is the reference voice to clone (3-10 s of clean speech works best).
"""
import argparse
import os
import sys
import time

import torch

# Make the hardcoded-cuda paths device-agnostic before constructing the model.
from firered_patch import apply_device_patches, bypass_text_normalizer


def find_config(model_dir: str) -> str:
    """Locate the model config. The HF repo's config.json is empty (0 bytes); the real
    config is the repo's configs/config_24k.json. Prefer a non-empty model/config.json,
    else fall back to the cloned repo copy."""
    cand = os.path.join(model_dir, "config.json")
    if os.path.exists(cand) and os.path.getsize(cand) > 0:
        return cand
    repo_cfg = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "FireRedTTS",
        "configs",
        "config_24k.json",
    )
    if os.path.exists(repo_cfg):
        return repo_cfg
    sys.exit(f"Could not find a valid config in {model_dir} or the repo configs/.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("FIRERED_MODEL", "./pretrained_models"))
    ap.add_argument("--prompt-wav", required=True, help="Reference voice .wav to clone.")
    ap.add_argument("--text", default="Hello from Trainium.")
    ap.add_argument("--lang", default="auto", choices=["zh", "en", "auto"])
    ap.add_argument("--out", default="cpu.wav")
    ap.add_argument(
        "--no-tn",
        action="store_true",
        help="Bypass WeTextProcessing/pynini text normalizer (use if it won't install).",
    )
    args = ap.parse_args()

    # WeTextProcessing (`tn`) is imported at fireredtts import time; bypass it BEFORE
    # any fireredtts import if requested or if it isn't installed.
    def _tn_available():
        try:
            import tn  # noqa: F401
            return True
        except Exception:
            return False

    if args.no_tn or not _tn_available():
        print("[cpu] using lite text normalizer (WeTextProcessing/pynini not in use)")
        bypass_text_normalizer()
    # Resolve user paths to absolute BEFORE we chdir into the repo root below.
    args.prompt_wav = os.path.abspath(args.prompt_wav)
    args.out = os.path.abspath(args.out)
    args.model = os.path.abspath(args.model)

    apply_device_patches()
    from fireredtts.fireredtts import FireRedTTS

    # Upstream config uses repo-relative asset paths (e.g. the flow codebook.npy), so run
    # from the repo root. Derive it from the fireredtts package location.
    import fireredtts as _f

    repo_root = os.path.dirname(os.path.abspath(next(iter(_f.__path__))))
    os.chdir(repo_root)

    cfg = find_config(args.model)
    print(f"[cpu] loading FireRedTTS (config={cfg}, model={args.model}) on CPU (cwd={repo_root})...")
    tts = FireRedTTS(config_path=cfg, pretrained_path=args.model, device="cpu")

    print(f"[cpu] synthesizing: {args.text!r} (lang={args.lang})")
    t0 = time.time()
    with torch.no_grad():
        wav = tts.synthesize(prompt_wav=args.prompt_wav, text=args.text, lang=args.lang)
    dt = time.time() - t0

    wav = wav.detach().cpu()
    dur = wav.shape[-1] / 24000
    print(f"[cpu] {dt:.1f}s -> {wav.shape[-1]} samples (~{dur:.1f}s @ 24kHz, RTF {dt/max(dur,1e-6):.2f})")

    import torchaudio

    torchaudio.save(args.out, wav, 24000)
    print(f"[cpu] wrote {args.out}")


if __name__ == "__main__":
    main()

"""Download FireRedTTS v1 checkpoints from the Hugging Face Hub.

Pulls the four files the model needs (~7.3 GB total) into a local directory:
  - config.json                 (model config; also shipped in the repo as configs/config_24k.json)
  - fireredtts_gpt.pt           (GPT-2 based autoregressive acoustic decoder)
  - fireredtts_speaker.bin      (ECAPA-TDNN speaker encoder, for zero-shot voice cloning)
  - fireredtts_token2wav.pt     (flow-matching decoder + BigVGAN vocoder)

Usage:
    python download_model.py                       # -> ./pretrained_models
    python download_model.py --out /data/firered   # custom location
"""
import argparse
import os
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out",
        default=os.environ.get("FIRERED_MODEL", "./pretrained_models"),
        help="Directory to download the checkpoints into.",
    )
    ap.add_argument(
        "--repo", default="FireRedTeam/FireRedTTS", help="HF model repo id."
    )
    args = ap.parse_args()

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        sys.exit("huggingface_hub not installed. Run: pip install huggingface_hub")

    os.makedirs(args.out, exist_ok=True)
    print(f"[download] {args.repo} -> {args.out}")
    path = snapshot_download(
        repo_id=args.repo,
        local_dir=args.out,
        allow_patterns=[
            "config.json",
            "fireredtts_gpt.pt",
            "fireredtts_speaker.bin",
            "fireredtts_token2wav.pt",
        ],
    )
    print(f"[download] done -> {path}")
    for f in sorted(os.listdir(path)):
        fp = os.path.join(path, f)
        if os.path.isfile(fp):
            print(f"           {f:32s} {os.path.getsize(fp)/1e6:8.1f} MB")


if __name__ == "__main__":
    main()

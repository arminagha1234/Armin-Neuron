#!/usr/bin/env python3
"""Materialize a local Gemma 4 E4B-it dir with a patched tokenizer config.

The published checkpoint ships ``tokenizer_config.json`` with
``extra_special_tokens`` as a *list*. transformers 4.x / 5.x expects a
*dict*. Loading the tokenizer directly from the HF cache snapshot raises
``ValueError: extra_special_tokens has to be a dict``.

This script symlinks every blob from the local HF snapshot into a
target dir and rewrites just the tokenizer config in place. It does NOT
copy the safetensors — they stay in the HF cache and the local dir
points to them via symlink, so disk overhead is negligible.

Run inside the Beta 3 container after
``huggingface-cli download google/gemma-4-E4B-it`` has populated the
cache:

    python build_local_model.py --dst /root/models/gemma-4-E4B-it
"""
from __future__ import annotations

import argparse
import glob
import json
import os


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="google/gemma-4-E4B-it")
    ap.add_argument("--dst", default="/root/models/gemma-4-E4B-it")
    ap.add_argument(
        "--cache-dir", default=None,
        help="HF cache root; defaults to $HF_HOME/.cache or "
             "/root/.cache/huggingface",
    )
    args = ap.parse_args()

    cache_root = (
        args.cache_dir
        or os.environ.get("HF_HOME")
        or "/root/.cache/huggingface"
    )
    repo_underscore = args.repo.replace("/", "--")
    snapshot_glob = os.path.join(
        cache_root, "hub", f"models--{repo_underscore}", "snapshots", "*",
    )
    snaps = glob.glob(snapshot_glob)
    if not snaps:
        raise SystemExit(
            f"No HF snapshot found under {snapshot_glob}. "
            f"Run `huggingface-cli download {args.repo}` first."
        )
    snap = snaps[0].rstrip("/")
    print(f"snapshot: {snap}")

    os.makedirs(args.dst, exist_ok=True)
    for name in os.listdir(snap):
        src_path = os.path.join(snap, name)
        real = os.path.realpath(src_path)
        d = os.path.join(args.dst, name)
        if os.path.lexists(d):
            os.remove(d)
        if name == "tokenizer_config.json":
            with open(real) as f:
                cfg = json.load(f)
            est = cfg.get("extra_special_tokens")
            if isinstance(est, list):
                cfg["extra_special_tokens"] = {}
                with open(d, "w") as f:
                    json.dump(cfg, f, indent=2)
                print(f"  patched tokenizer_config.json (was list)")
            else:
                os.symlink(real, d)
                print(f"  symlinked tokenizer_config.json (already dict)")
        else:
            os.symlink(real, d)
    print(f"local model dir ready: {args.dst}")
    print(f"contents: {sorted(os.listdir(args.dst))}")


if __name__ == "__main__":
    main()

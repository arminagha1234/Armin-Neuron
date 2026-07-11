"""
Sample prompts from OpenThoughts3-1.2M into a JSONL the generator consumes.

Mirrors the repo's scripts/prepare_sft_prompts.py: pull N prompts, write one
{"prompt": <str>} per line. Uses streaming so it never downloads the full 1.2M
rows when you only want 300k (or 64 for a smoke).

Usage:
  python prepare_prompts.py --num-samples 300000 --seed 42 \
      --output data/openthoughts3_300000.jsonl
  python prepare_prompts.py --num-samples 64 --output data/prompts_smoke.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--hf-dataset", default="open-thoughts/OpenThoughts3-1.2M")
    p.add_argument("--split", default="train")
    p.add_argument("--num-samples", type=int, default=300000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", required=True)
    return p.parse_args()


def extract_prompt(row):
    """OpenThoughts3 rows carry a conversation; grab the user question text."""
    for key in ("prompt", "question", "instruction"):
        if isinstance(row.get(key), str) and row[key]:
            return row[key]
    convo = row.get("conversations") or row.get("messages")
    if isinstance(convo, list):
        for turn in convo:
            if isinstance(turn, dict):
                role = turn.get("role") or turn.get("from")
                content = turn.get("content") or turn.get("value")
                if role in ("user", "human") and content:
                    return content
    return None


def main():
    args = parse_args()
    from datasets import load_dataset

    print(f"[data] streaming {args.hf_dataset} [{args.split}] for {args.num_samples} prompts")
    ds = load_dataset(args.hf_dataset, split=args.split, streaming=True)
    ds = ds.shuffle(seed=args.seed, buffer_size=10000)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with open(args.output, "w", encoding="utf-8") as fh:
        for row in ds:
            prompt = extract_prompt(row)
            if not prompt:
                continue
            fh.write(json.dumps({"prompt": prompt}) + "\n")
            written += 1
            if written >= args.num_samples:
                break
            if written % 10000 == 0:
                print(f"  ... {written} prompts", flush=True)
    print(f"[data] wrote {written} prompts -> {args.output}")


if __name__ == "__main__":
    main()

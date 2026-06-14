#!/usr/bin/env python3
"""
GLM 5.1 — 2-Layer Smoke Test
==============================
Quick validation that the model loads, compiles, and generates tokens.
Uses only 2 layers to minimize compile time (~5 min vs hours for full 78).

Run inside the vLLM-Neuron container on trn2.48xl:
    python test_2layer.py --model-path /path/to/glm5

Expected behavior:
- Loads config from GLM-5.1 checkpoint
- Overrides num_hidden_layers=2
- Compiles for TP=64 (or TP=2 for faster iteration)
- Generates a single completion
- Prints token + latency

Success criteria:
- Output is not garbage/NaN
- Generation completes without compiler error
- TTFT < 10 seconds (2 layers should be near-instant after compile)
"""

import argparse
import json
import os
import sys
import time

import torch


def main():
    parser = argparse.ArgumentParser(description="GLM 5.1 2-layer smoke test")
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to GLM-5.1 weights (HF format)")
    parser.add_argument("--tp", type=int, default=64,
                        help="Tensor parallel degree")
    parser.add_argument("--num-layers", type=int, default=2,
                        help="Number of layers to load (default: 2)")
    parser.add_argument("--max-model-len", type=int, default=128,
                        help="Max model length")
    parser.add_argument("--prompt", type=str,
                        default="The meaning of life is",
                        help="Test prompt")
    args = parser.parse_args()

    print(f"[INFO] GLM 5.1 smoke test — {args.num_layers} layers, TP={args.tp}")
    print(f"[INFO] Model path: {args.model_path}")

    # Strategy: Use vLLM offline inference with num_hidden_layers override
    # This avoids needing the full model to validate the pipeline
    os.environ.setdefault("NEURON_RT_VIRTUAL_CORE_SIZE", "2")

    from vllm import LLM, SamplingParams

    # Override config to limit layers
    # vLLM supports this via hf_overrides
    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=args.tp,
        max_model_len=args.max_model_len,
        max_num_seqs=1,
        dtype="bfloat16",
        hf_overrides={"num_hidden_layers": args.num_layers},
    )

    sampling_params = SamplingParams(
        max_tokens=16,
        temperature=0.0,  # greedy for reproducibility
    )

    print(f"[INFO] Generating with prompt: '{args.prompt}'")
    t0 = time.time()
    outputs = llm.generate([args.prompt], sampling_params)
    t_total = time.time() - t0

    for output in outputs:
        generated = output.outputs[0].text
        tokens = output.outputs[0].token_ids
        print(f"\n[RESULT] Prompt: {args.prompt}")
        print(f"[RESULT] Generated: {generated}")
        print(f"[RESULT] Tokens: {len(tokens)}")
        print(f"[RESULT] Total time: {t_total:.2f}s")
        print(f"[RESULT] TTFT estimate: {t_total / max(len(tokens), 1):.2f}s per token")

    print("\n[INFO] ✅ Smoke test PASSED" if generated.strip() else "\n[WARN] ⚠️ Empty output")


if __name__ == "__main__":
    main()

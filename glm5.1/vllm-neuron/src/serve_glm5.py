#!/usr/bin/env python3
"""
GLM 5.1 — vLLM-Neuron serve / smoke runner

Runs GLM 5.1 inference on Trainium2 via vLLM-Neuron with the patches
applied (see patch_registry.py and patch_get_tensor_names.py).

Examples:

    # 30-layer smoke (TP=32) — known working
    NEURON_RT_VIRTUAL_CORE_SIZE=2 NEURON_SKIP_EFA_AFFINITY=1 \\
        python serve_glm5.py --num-hidden-layers 30 --tp 32

    # Full 78 layers, TP=32 (currently OOMs at layer ~45 — needs FP8 on-device)
    NEURON_RT_VIRTUAL_CORE_SIZE=2 NEURON_SKIP_EFA_AFFINITY=1 \\
        python serve_glm5.py --tp 32

Required env vars:
    NEURON_RT_VIRTUAL_CORE_SIZE=2     # standard for trn2
    NEURON_SKIP_EFA_AFFINITY=1        # bypass EFA setup on single-host
"""

import argparse
import os
import sys
import time


DEFAULT_MODEL = (
    "/mnt/data/hf_cache/models--mconcat--GLM-5.1-FP8-Dynamic/"
    "snapshots/3e613be45ea079bfc2e8e9141ce6f4338d6c35e4"
)


def main():
    parser = argparse.ArgumentParser(description="Serve GLM 5.1 on vLLM-Neuron")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help="Path to GLM 5.1 weights (HF format)")
    parser.add_argument("--tp", type=int, default=32,
                        help="Tensor parallel degree (max 32 due to index_n_heads)")
    parser.add_argument("--max-model-len", type=int, default=128)
    parser.add_argument("--max-num-seqs", type=int, default=1)
    parser.add_argument("--num-hidden-layers", type=int, default=None,
                        help="Override num_hidden_layers (use 30 for smoke; full=78)")
    parser.add_argument("--prompt", type=str, default="The future of AI is")
    parser.add_argument("--max-tokens", type=int, default=16)
    args = parser.parse_args()

    # Required environment
    os.environ.setdefault("NEURON_RT_VIRTUAL_CORE_SIZE", "2")
    os.environ.setdefault("NEURON_SKIP_EFA_AFFINITY", "1")

    print(f"[INFO] GLM 5.1 — vLLM-Neuron")
    print(f"[INFO] Model: {args.model}")
    print(f"[INFO] TP={args.tp}, max_model_len={args.max_model_len}, "
          f"layers={args.num_hidden_layers or 'full (78)'}")

    from vllm import LLM, SamplingParams

    hf_overrides = {}
    if args.num_hidden_layers is not None:
        hf_overrides["num_hidden_layers"] = args.num_hidden_layers

    t0 = time.time()
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tp,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        dtype="bfloat16",
        hf_overrides=hf_overrides,
    )
    load_min = (time.time() - t0) / 60
    print(f"[INFO] Loaded in {load_min:.1f} min")

    params = SamplingParams(max_tokens=args.max_tokens, temperature=0.0)

    print(f"[INFO] Generating: '{args.prompt}'")
    t1 = time.time()
    outputs = llm.generate([args.prompt], params)
    ttft_ms = (time.time() - t1) * 1000
    print(f"[INFO] TTFT = {ttft_ms:.1f} ms")

    for o in outputs:
        print(f"[OUT] {o.outputs[0].text}")

    print("[DONE]")


if __name__ == "__main__":
    main()

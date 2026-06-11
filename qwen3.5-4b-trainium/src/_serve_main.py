# SPDX-License-Identifier: Apache-2.0
"""Tiny wrapper to apply the Qwen3.5 registry patch and exec `vllm serve`.

Configured entirely via env vars to keep the launcher shell simple.
Called from `serve.sh`.

================================================================
ENV VARS
================================================================

- MODEL: path to local HF weights (default: /root/models/Qwen3.5-4B)
- TP: tensor parallel size (default: 4). Hard caps: 16 query heads on
  Qwen3.5-4B → max TP=16. KV heads = 4 → above TP=4, KV is replicated.
- MAX_LEN: max model context window (default: 4096). This is the BIG number
  — it caps `--max-model-len` in vLLM. For 50K input you need MAX_LEN=51200
  or above.
- BUCKET: comma-separated list of prefill bucket sizes for
  `num_batched_tokens_buckets`. Each request is padded UP to the smallest
  bucket that fits its prompt+output, then routed to the matching NEFF.
  More buckets = better fit / less wasted compute, but longer compile time.
  Examples:
    BUCKET=4096                       # single 4K bucket
    BUCKET=8192,20480,51200           # 8K, 20K, 50K (customer sweep)
- KV_SEGMENT: chunked-prefill window size (default: 4096). MUST be one of
  {512, 1024, 2048, 4096} — vllm_neuron's segmented attention kernel only
  ships these segment sizes. This is NOT the full KV size, it's the chunk
  size at which prefill is broken up.
- PORT: HTTP port (default: 8000)

================================================================
WHY THERE ARE TWO BUCKET KNOBS (BUCKET vs KV_SEGMENT)
================================================================

`num_batched_tokens_buckets` (BUCKET) and `kv_segment_size_buckets`
(KV_SEGMENT) sound similar but mean different things on Neuron — and on
the v2 vllm_neuron container they have a strict relationship that's
easy to get wrong:

- num_batched_tokens_buckets — how many tokens the prefill graph
  processes per forward pass. Compiles one NEFF per bucket.

- kv_segment_size_buckets — chunked-prefill stride. The kernel processes
  prefill in chunks of this size. Capped at 4096 by the segmented
  attention kernel that ships in v2/v3.

THE STRICT RULE on v2/v3:
  When kv_segment_size_buckets is set, num_batched_tokens_buckets MUST
  EQUAL kv_segment_size_buckets. Both must use values from
  {512, 1024, 2048, 4096}.

Setting num_batched_tokens_buckets larger than the kv segment crashes
worker init with `ValueError: prefill bucket length must equal segment
size`. There is NO way to pre-compile a single 50K-prefill graph on
this build — long prefills MUST be chunked through the 4K kernel by
vLLM's chunked-prefill scheduler at runtime.

So for ANY long-context serve (20K, 50K, 200K all the same):
  BUCKET=512,1024,2048,4096
  KV_SEGMENT=4096   # or omit and rely on default
  MAX_LEN=<whatever>  # this is what caps the customer's input length

The chunked-prefill scheduler in vLLM handles streaming the long input
through the 4K kernel iteratively. TTFT scales roughly linearly with
sequence length because of this chunking — that IS the architectural
trade-off in this build.

(See live run log gates 14b/14c in README for the discovery trail.)

================================================================
SWEEPING SEQ LENGTHS — WHAT TO SET
================================================================

Customer wants to test 8K, 20K, 50K, 100K, 200K? On v2/v3 vllm_neuron
the answer is the SAME config for all of them — because the prefill
bucket list is locked to {512, 1024, 2048, 4096}, ALL long prefills
go through chunked-prefill at runtime regardless of input size:

  TP=8 MAX_LEN=204800 BUCKET=512,1024,2048,4096 ./serve.sh

MAX_LEN sets the largest input the customer can send. The `BUCKET`
list configures the chunked-prefill kernel's prefill segment buckets.
The bench script just sends prompts of different lengths to the same
server; vLLM's scheduler handles streaming each one through the 4K
kernel.

KEY CONSEQUENCE: TTFT scales ~linearly with sequence length on this
build, because doubling the input doubles the number of 4K chunks
processed. This is an architectural property of v2/v3, not a bug in
our model.

================================================================
"""

import json
import os
import sys


def main() -> int:
    # 1. Patch the registry FIRST, before vllm sees the model.
    from qwen3_5.register import register
    register()

    # 2. Build sys.argv from env vars.
    model = os.environ.get("MODEL", "/root/models/Qwen3.5-4B")
    tp = os.environ.get("TP", "4")
    max_len = os.environ.get("MAX_LEN", "4096")
    port = os.environ.get("PORT", "8000")
    bucket = os.environ.get("BUCKET", "").strip()
    max_num_seqs = int(os.environ.get("MAX_NUM_SEQS", "1"))

    # PATH D: KV cache dtype. "auto" = match model dtype (BF16). "fp8" /
    # "fp8_e4m3" enables FP8-quantized KV cache, which the model code
    # detects via `self.k_cache.dtype` and switches the write/read path
    # accordingly. Run-time scale calibration happens elsewhere.
    kv_cache_dtype = os.environ.get("KV_CACHE_DTYPE", "auto").strip()

    # additional_config JSON
    if bucket:
        # bucket can be a single int or comma-separated list, e.g. "4096" or "512,1024,2048,4096"
        bucket_list = [int(b.strip()) for b in bucket.split(",") if b.strip()]
        # Chunked prefill mode — for >4K inputs.
        # NOTE: on v2/v3 vllm_neuron, num_batched_tokens_buckets and
        # kv_segment_size_buckets MUST BE THE SAME LIST. The validator at
        # vllm_neuron/utils/bucket_utils.py:325 raises ValueError otherwise.
        # Both must use values from {512, 1024, 2048, 4096}. To process
        # longer inputs (8K, 20K, ..., 200K), MAX_LEN sets the limit and
        # vLLM's chunked-prefill scheduler streams them through the 4K
        # kernel iteratively.
        addl = {
            "neuron_config": {
                "num_batched_tokens_buckets": bucket_list,
                "num_seqs_buckets": [max_num_seqs],
                "kv_segment_size_buckets": bucket_list,  # MUST equal num_batched_tokens_buckets
                "on_device_sampling_config": {"all_greedy": True},
            }
        }
        max_batched_tokens = max(bucket_list)
    else:
        addl = {
            "neuron_config": {
                "num_seqs_buckets": [max_num_seqs],
                "on_device_sampling_config": {"all_greedy": True},
            }
        }
        max_batched_tokens = max_len

    sys.argv = [
        "vllm",
        "serve",
        model,
        "--tensor-parallel-size", str(tp),
        "--max-model-len", str(max_len),
        "--max-num-batched-tokens", str(max_batched_tokens),
        "--max-num-seqs", str(max_num_seqs),
        "--kv-cache-dtype", kv_cache_dtype,
        "--port", str(port),
        "--additional-config", json.dumps(addl),
    ]

    print("[serve] launching vllm with argv:")
    for arg in sys.argv:
        print(f"  {arg}")

    from vllm.entrypoints.cli.main import main as vllm_main
    vllm_main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

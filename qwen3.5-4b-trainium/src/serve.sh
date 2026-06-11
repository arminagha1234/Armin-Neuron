#!/usr/bin/env bash
# Launch vllm serve for Qwen3.5-4B with the Path B registry patch.
#
# Three usage modes:
#
#   ./serve.sh                                 # 4K context, single-shot prefill
#   MAX_LEN=16384 ./serve.sh                   # 16K context, single-shot prefill
#   MAX_LEN=32768 BUCKET=4096 ./serve.sh       # 32K context, chunked prefill (4K chunks) — customer 20K-input shape
#
# Required: run from inside the vllm_neuron container with this folder
# (qwen3_5 package) on PYTHONPATH. e.g.:
#
#   docker exec -e PYTHONPATH=/path/to/pathB/vllm_neuron_native_qwen35 vllm_neuron \
#       /path/to/serve.sh

set -euo pipefail

export MODEL=${MODEL:-/root/models/Qwen3.5-4B}
export TP=${TP:-4}
export MAX_LEN=${MAX_LEN:-4096}
export PORT=${PORT:-8000}
export BUCKET=${BUCKET:-}
export MAX_NUM_SEQS=${MAX_NUM_SEQS:-1}
export KV_CACHE_DTYPE=${KV_CACHE_DTYPE:-auto}
export NEURON_SKIP_EFA_AFFINITY=${NEURON_SKIP_EFA_AFFINITY:-1}

HERE="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${HERE}:${PYTHONPATH:-}"

echo "==============================="
echo "Path D Qwen3.5 vLLM serve (Path C + FP8 KV)"
echo "  Model:           ${MODEL}"
echo "  TP:              ${TP}"
echo "  max_len:         ${MAX_LEN}"
echo "  bucket:          ${BUCKET:-<none, single-shot prefill>}"
echo "  max_num_seqs:    ${MAX_NUM_SEQS}"
echo "  kv_cache_dtype:  ${KV_CACHE_DTYPE}"
echo "  port:            ${PORT}"
echo "  PYTHONPATH:      ${PYTHONPATH}"
echo "==============================="

exec python "${HERE}/_serve_main.py"

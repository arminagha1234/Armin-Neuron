#!/usr/bin/env bash
# Launch vllm serve for Qwen3.6-27B with the registry patch.
#
# Three usage modes:
#
#   ./serve.sh                                 # 4K context, single-shot prefill
#   MAX_LEN=16384 ./serve.sh                   # 16K context, single-shot prefill
#   MAX_LEN=32768 BUCKET=4096 ./serve.sh       # 32K context, chunked prefill (4K chunks)
#
# Required: run from inside the vllm_neuron container with this folder
# (qwen3_6 package) on PYTHONPATH. e.g.:
#
#   docker exec -e PYTHONPATH=/path/to/Makora_27B/qwen36_27b/src vllm_neuron \
#       /path/to/serve.sh

set -euo pipefail

export MODEL=${MODEL:-/root/models/Qwen3.6-27B}
# Qwen3.6-27B has 24 Q heads. Allowed TP: divisors of 24 (1,2,3,4,6,8,12,24).
# DeltaNet has 48 v-heads, divides cleanly through TP=24. Above TP=4 the
# 4 KV heads get replicated. Default TP=8 is the sweet spot on trn2.48xl.
export TP=${TP:-8}
export MAX_LEN=${MAX_LEN:-4096}
export PORT=${PORT:-8000}
export BUCKET=${BUCKET:-}
export MAX_NUM_SEQS=${MAX_NUM_SEQS:-1}
export KV_CACHE_DTYPE=${KV_CACHE_DTYPE:-auto}
export NEURON_SKIP_EFA_AFFINITY=${NEURON_SKIP_EFA_AFFINITY:-1}

HERE="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${HERE}:${PYTHONPATH:-}"

echo "==============================="
echo "Qwen3.6-27B vLLM serve"
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

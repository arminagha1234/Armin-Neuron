#!/usr/bin/env bash
# Standard `vllm bench throughput` for Gemma4 31B — matches PR #1552 methodology.
# Run INSIDE the container (no separate server needed; this spins up its own).
#
# Usage: bash run_vllm_bench.sh <input_len> <output_len> <tp> <max_num_seqs>
set -euo pipefail

MODEL="${MODEL:-/root/models/gemma-4-31b-it}"
IN="${1:-1024}"
OUT="${2:-256}"
TP="${3:-16}"
NSEQ="${4:-16}"
MAXLEN=$(( IN + OUT + 16 ))
OUTDIR="/work/results/vllm_bench_in${IN}_out${OUT}_tp${TP}_b${NSEQ}"
mkdir -p "$OUTDIR"

export NEURON_SKIP_EFA_AFFINITY=1
export PYTHONPATH="/work/pkg:/work:${PYTHONPATH:-}"
export VLLM_NEURON_COMPILATION_TIMEOUT=2400

echo "[vllm bench] in=$IN out=$OUT tp=$TP max_num_seqs=$NSEQ maxlen=$MAXLEN"
vllm bench throughput \
  --model "$MODEL" \
  --tokenizer "$MODEL" \
  --dtype bfloat16 \
  --tensor-parallel-size "$TP" \
  --max-model-len "$MAXLEN" \
  --max-num-seqs "$NSEQ" \
  --num-prompts $(( NSEQ * 4 )) \
  --input-len "$IN" \
  --output-len "$OUT" \
  --additional-config "{\"neuron_config\": {\"num_batched_tokens_buckets\": [$MAXLEN], \"num_seqs_buckets\": [$NSEQ], \"on_device_sampling_config\": {\"all_greedy\": true}}}" \
  --output-json "$OUTDIR/bench.json" 2>&1 | tee "$OUTDIR/bench.log"

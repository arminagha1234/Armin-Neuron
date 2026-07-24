#!/bin/bash
# Gemma4-31B TTFT sweep runner (public image). Full sweep or any subset via env vars.
#   MODEL=/root/models/gemma-4-31b-text bash run_benchmark.sh          # full sweep
#   ONLY=16k LEVELS=1,2,4 TP=16 MODEL=... bash run_benchmark.sh        # subset
set -euo pipefail
MODEL="${MODEL:?set MODEL=/root/models/gemma-4-31b-text}"
ONLY="${ONLY:-4k,8k,16k,32k,64k}"; LEVELS="${LEVELS:-1,2,4,8,16,32}"
TP="${TP:-32}"; GEN="${GEN:-40}"; APC="${APC:-0}"
OUT="results_$(date +%Y%m%d_%H%M%S)"; mkdir -p "$OUT"
declare -A CTX=( [4k]=4096 [8k]=8192 [16k]=15000 [32k]=31000 [64k]=63000 )
apc_flag="--no-enable-prefix-caching"; [ "$APC" = "1" ] && apc_flag=""
echo "size,conc,tp,in_tok,ttft_med_s,ttft_p99_s,tpot_ms,status" > "$OUT/summary.csv"
for sz in ${ONLY//,/ }; do
  ctx=${CTX[$sz]}; maxlen=16384; batched=16384; buckets='[256,512,1024,2048,4096,8192,16384]'
  # >16k => segmented serve (SEG=2048)
  if [ "$ctx" -gt 16384 ] 2>/dev/null || [ "$sz" = "32k" ] || [ "$sz" = "64k" ]; then
    maxlen=$([ "$sz" = "64k" ] && echo 65536 || echo 32768); batched=2048; buckets='[2048]'
  fi
  lv="$LEVELS"; { [ "$sz" = "32k" ] || [ "$sz" = "64k" ]; } && lv="1,2,4"
  echo "[serve] $sz TP=$TP maxlen=$maxlen seg=$batched"
  VLLM_CACHE_ROOT=/root/.cache/bench_${TP}_${sz} GEMMA4_CTE_PREFILL=1 GEMMA4_BF16_FALLBACK=1 \
    vllm serve "$MODEL" --served-model-name gemma4 --tensor-parallel-size "$TP" \
    --max-model-len "$maxlen" --max-num-seqs 32 --max-num-batched-tokens "$batched" \
    $apc_flag --async-scheduling \
    --additional-config "{\"neuron_config\":{\"num_batched_tokens_buckets\":$buckets,\"num_seqs_buckets\":[32],\"on_device_sampling_config\":{\"all_greedy\":true}}}" \
    --port 8000 --host 0.0.0.0 > "$OUT/serve_$sz.log" 2>&1 &
  # wait for startup
  for i in $(seq 1 60); do grep -qa "Application startup complete" "$OUT/serve_$sz.log" && break; sleep 20; done
  # warmup + bench
  for w in 1 2 3; do curl -s -m180 http://localhost:8000/v1/completions -H 'Content-Type: application/json' \
    -d "{\"model\":\"gemma4\",\"prompt\":\"$(python3 -c "import random;random.seed($w);print(' '.join(str(random.randint(0,9)) for _ in range($((ctx/2)))))")\",\"max_tokens\":8}" >/dev/null 2>&1; done
  python3 bench_random16.py --ctx-tokens "$ctx" --gen "$GEN" --levels "$lv" --range-ratio 0 \
    --out "$OUT/rand_${sz}.json" 2>&1 | grep -aE '^ *[0-9]' | while read c ok err intok tt p99 tpot rest; do
      echo "$sz,$c,$TP,$intok,$tt,$p99,$tpot,ok" >> "$OUT/summary.csv"; done
  pkill -f "vllm serve"; sleep 8   # NOTE: if next serve fails to init cores, restart the container (orphan-worker guard)
done
echo "=== done -> $OUT/summary.csv ==="; cat "$OUT/summary.csv"

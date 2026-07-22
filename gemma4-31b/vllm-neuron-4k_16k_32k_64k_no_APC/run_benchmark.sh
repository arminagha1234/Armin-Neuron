#!/bin/bash
# =============================================================================
# Gemma4-31B benchmark — NO-APC / honest cold-prefill edition.
#
# Difference vs the sibling `vllm-neuron-4k_16k_32k_64k/`:
#   * NO prefix caching (APC OFF) and NO FP8 — pure bf16.
#   * Every request gets a UNIQUE random prompt (bench_random.py) so prefix
#     caching CANNOT serve it from cache — this is the honest, apples-to-apples
#     COLD-prefill number. (The sibling folder's headline uses APC + FP8-KV on a
#     repeated-context / cache-hit workload, which is NOT comparable to cold.)
#   * ≤16k uses SINGLE-SHOT prefill (max_num_batched_tokens == max_model_len) —
#     the fastest path. >16k uses SEGMENTED prefill (single-shot caps at 16k).
#
# Measures TTFT / TPOT / E2E at concurrency 1,2,4,8,16,32 for input sizes
# 4k / 8k / 16k / 32k / 64k, 40 output tokens.
#
#   bash run_benchmark.sh                         # everything
#   ONLY=4k,8k,16k bash run_benchmark.sh          # subset
#   SKIP_LAUNCH=1 BASE_URL=http://host:8000 bash run_benchmark.sh   # existing server
#
# Run on the Trainium instance, inside your vLLM-Neuron container.
# =============================================================================
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"

GEN=${GEN:-40}                              # output tokens
LEVELS=${LEVELS:-1,2,4,8,16,32}             # concurrency levels (≤16k). Long ctx uses LEVELS_LONG.
LEVELS_LONG=${LEVELS_LONG:-1,2,4}           # 32k/64k: high conc is KV-capacity-bound (queues) — keep low
TP=${TP:-32}
MNS=${MNS:-32}
MODEL=${MODEL:-google/gemma-4-31B-it}
SERVING_PKG=${SERVING_PKG:-}
BASE_URL=${BASE_URL:-http://localhost:8000}
MODEL_NAME=${MODEL_NAME:-gemma4}
SKIP_LAUNCH=${SKIP_LAUNCH:-0}
ONLY=${ONLY:-4k,8k,16k,32k,64k}
# Reduced parallel trace/compile workers avoid a trace-worker race that can
# silently kill workers mid-compile at TP32 (see optimizations/SESSION_LEARNINGS).
export VLLM_NEURON_PARALLEL_COMPILE_WORKERS=${VLLM_NEURON_PARALLEL_COMPILE_WORKERS:-4}
export VLLM_NEURON_PARALLEL_TRACE_WORKERS=${VLLM_NEURON_PARALLEL_TRACE_WORKERS:-4}
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
OUT="$HERE/results_${STAMP}"
mkdir -p "$OUT"

echo "Gemma4-31B NO-APC benchmark :: $STAMP"
echo "  output_tokens=$GEN  conc=$LEVELS (long:$LEVELS_LONG)  TP=$TP  MNS=$MNS  bf16, APC OFF, unique random prompts"
echo "  results -> $OUT"

# bench one input size (ctx tokens) against whatever server is up
bench_size () {
  local name=$1 ctx=$2 levels=$3
  case ",$ONLY," in *",$name,"*) : ;; *) echo "-- skip $name"; return 0 ;; esac
  echo ""
  echo ">>> INPUT $name  (ctx≈$ctx tok, cold random, APC OFF) <<<"
  python3 "$HERE/bench_random.py" --base-url "$BASE_URL" --model "$MODEL_NAME" \
    --ctx-tokens "$ctx" --gen "$GEN" --levels "$levels" --range-ratio 0 \
    --out "$OUT/$name.json" 2>&1 | tee "$OUT/$name.log"
}

launch () {  # LEN SEG BUCKETS  -> single-shot when SEG==LEN, segmented when SEG<LEN
  local len=$1 seg=$2 buckets=$3
  [ "$SKIP_LAUNCH" = "1" ] && return 0
  MODEL="$MODEL" TP="$TP" LEN="$len" SEG="$seg" BUCKETS="$buckets" MNS="$MNS" \
    KV_CACHE_DTYPE=auto APC=0 SERVING_PKG="$SERVING_PKG" LOGDIR="$OUT" SERVED_NAME="$MODEL_NAME" \
    GEMMA4_DECODE_BACKEND=sdpa GEMMA4_SWA_DECODE_BACKEND=sdpa \
    bash "$HERE/launch_serve.sh" || { echo "!! serve failed to start"; return 1; }
}

# ---- Pool A: SINGLE-SHOT prefill for ≤16k (max_num_batched_tokens == max_model_len) ----
# One serve (LEN=SEG=16384) with fine buckets covers 4k/8k/16k; each prompt routes to the
# smallest bucket that fits (less padding waste on short prompts).
case ",$ONLY," in *4k*|*8k*|*16k*)
  echo ""; echo "########## POOL A: single-shot (bf16, no APC), LEN=16384 ##########"
  launch 16384 16384 "256,512,1024,2048,4096,8192,16384" && {
    bench_size 4k  4000  "$LEVELS"
    bench_size 8k  8000  "$LEVELS"
    bench_size 16k 15800 "$LEVELS"
  } ;;
esac

# ---- Pool B: SEGMENTED prefill for 32k/64k (single-shot caps at 16k) ----
# One serve (LEN=66560, SEG=8192) covers both. seg=8192 = fewest chunks (fastest).
case ",$ONLY," in *32k*|*64k*)
  echo ""; echo "########## POOL B: segmented seg=8192 (bf16, no APC), LEN=66560 ##########"
  launch 66560 8192 8192 && {
    bench_size 32k 32000 "$LEVELS_LONG"
    bench_size 64k 63000 "$LEVELS_LONG"
  } ;;
esac

echo ""; echo "=================== SUMMARY ==================="
python3 "$HERE/summarize.py" --results "$OUT" | tee "$OUT/summary.txt"
echo ""; echo "Done. Results in: $OUT"

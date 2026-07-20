#!/bin/bash
# =============================================================================
# Gemma4-31B benchmark — one command, all configs.
# Measures TTFT / TPOT / E2E at concurrency 1,2,4,8,16,32 for input sizes
# 4k / 16k / 32k / 64k, with 40 output tokens. Writes results/ + a summary.
#
#   bash run_benchmark.sh                    # run everything (launches a server per input size)
#   ONLY=4k,16k bash run_benchmark.sh         # subset of input sizes
#   SKIP_LAUNCH=1 BASE_URL=http://host:8000 bash run_benchmark.sh   # bench an EXISTING server
#   KV_CACHE_DTYPE=fp8_e4m3 bash run_benchmark.sh                    # fp8 KV cache
#
# Run this on the Trainium instance (inside your vLLM-Neuron environment).
# =============================================================================
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"

# ------------------------------- settings ----------------------------------
GEN=${GEN:-40}                              # output tokens (fixed per the benchmark spec)
LEVELS=${LEVELS:-1,2,4,8,16,32}             # concurrency levels
TP=${TP:-32}                                # tensor-parallel size
MNS=${MNS:-32}                              # max-num-seqs (>= max concurrency)
KV_CACHE_DTYPE=${KV_CACHE_DTYPE:-auto}      # 'auto' = bf16; 'fp8_e4m3' for fp8 KV cache
MODEL=${MODEL:-/root/models/gemma-4-31b-it} # Gemma4-31B checkpoint (path or HF id)
SERVING_PKG=${SERVING_PKG:-}                # optional custom vLLM-Neuron PYTHONPATH
BASE_URL=${BASE_URL:-http://localhost:8000}
MODEL_NAME=${MODEL_NAME:-gemma4}
SKIP_LAUNCH=${SKIP_LAUNCH:-0}               # 1 = don't launch; benchmark the server at BASE_URL
ONLY=${ONLY:-4k,16k,32k,64k}
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
OUT="$HERE/results_${STAMP}"
mkdir -p "$OUT"
# ---------------------------------------------------------------------------

echo "Gemma4-31B benchmark :: $STAMP"
echo "  output_tokens=$GEN  concurrency=$LEVELS  TP=$TP  MNS=$MNS  KV=$KV_CACHE_DTYPE"
echo "  base_url=$BASE_URL  skip_launch=$SKIP_LAUNCH  results -> $OUT"
echo

# name  LEN     CTX_TOKENS  SEG    BUCKETS
run_one () {
  local name=$1 len=$2 ctx=$3 seg=$4 buckets=$5
  case ",$ONLY," in *",$name,"*) : ;; *) echo "-- skip $name"; return 0 ;; esac
  echo ""
  echo "=================== INPUT $name  (LEN=$len, CTX=$ctx) ==================="
  if [ "$SKIP_LAUNCH" != "1" ]; then
    MODEL="$MODEL" TP="$TP" LEN="$len" SEG="$seg" BUCKETS="$buckets" MNS="$MNS" \
      KV_CACHE_DTYPE="$KV_CACHE_DTYPE" SERVING_PKG="$SERVING_PKG" LOGDIR="$OUT" SERVED_NAME="$MODEL_NAME" \
      bash "$HERE/launch_serve.sh" || { echo "!! $name serve failed to start — skipping"; return 0; }
  fi
  python3 "$HERE/bench.py" --base-url "$BASE_URL" --model "$MODEL_NAME" \
    --ctx-tokens "$ctx" --gen "$GEN" --levels "$LEVELS" --out "$OUT/$name.json" 2>&1 | tee "$OUT/$name.log"
}

run_one 4k   5120   4096   4096  512,1024,2048,4096
run_one 16k  20480  16384  4096  4096
run_one 32k  36864  32768  4096  4096
run_one 64k  69632  65536  4096  4096

echo ""
echo "=================== SUMMARY ==================="
python3 "$HERE/summarize.py" --results "$OUT" | tee "$OUT/summary.txt"
echo ""
echo "Done. All results in: $OUT"

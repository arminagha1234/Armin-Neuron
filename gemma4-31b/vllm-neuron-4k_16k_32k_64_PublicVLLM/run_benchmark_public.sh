#!/bin/bash
# =============================================================================
# Gemma4-31B benchmark on the PUBLIC vLLM-Neuron v0.21 DLC — one command, all configs.
# Measures TTFT / TPOT / E2E at concurrency 1,2,4,8,16,32 for input sizes
# 4k / 16k / 32k / 64k, with 40 output tokens. Writes results/ + a summary.
#
#   bash run_benchmark_public.sh                     # run everything (server per input size)
#   ONLY=4k,16k bash run_benchmark_public.sh          # subset of input sizes
#   SKIP_LAUNCH=1 BASE_URL=http://host:8000 bash run_benchmark_public.sh   # bench an EXISTING server
#
# Run INSIDE the public DLC container (after install_public.sh + make_local_model.py).
# =============================================================================
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"

GEN=${GEN:-40}                                   # output tokens (benchmark spec)
LEVELS=${LEVELS:-1,2,4,8,16,32}                  # concurrency levels
TP=${TP:-32}
MNS=${MNS:-32}                                   # max-num-seqs (>= max concurrency)
MODEL=${MODEL:-/root/models/gemma-4-31b-it}      # local text-only model dir (make_local_model.py)
BASE_URL=${BASE_URL:-http://localhost:8000}
MODEL_NAME=${MODEL_NAME:-gemma4}
SKIP_LAUNCH=${SKIP_LAUNCH:-0}
ONLY=${ONLY:-4k,16k,32k,64k}
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
OUT="$HERE/results_${STAMP}"
mkdir -p "$OUT"

echo "Gemma4-31B PUBLIC benchmark :: $STAMP"
echo "  output_tokens=$GEN  concurrency=$LEVELS  TP=$TP  MNS=$MNS  base_url=$BASE_URL  -> $OUT"

# name  LEN     CTX     SEG  BUCKETS  KV_DTYPE
run_one () {
  local name=$1 len=$2 ctx=$3 seg=$4 buckets=$5 kv=$6
  case ",$ONLY," in *",$name,"*) : ;; *) echo "-- skip $name"; return 0 ;; esac
  echo ""
  echo "=================== INPUT $name  (LEN=$len CTX=$ctx SEG=$seg KV=$kv APC=on) ==================="
  if [ "$SKIP_LAUNCH" != "1" ]; then
    MODEL="$MODEL" TP="$TP" LEN="$len" SEG="$seg" BUCKETS="$buckets" MNS="$MNS" \
      KV_CACHE_DTYPE="$kv" APC=1 LOGDIR="$OUT" SERVED_NAME="$MODEL_NAME" \
      bash "$HERE/launch_serve_public.sh" || { echo "!! $name serve failed — skipping"; return 0; }
  fi
  python3 "$HERE/bench.py" --base-url "$BASE_URL" --model "$MODEL_NAME" \
    --ctx-tokens "$ctx" --gen "$GEN" --levels "$LEVELS" --out "$OUT/$name.json" 2>&1 | tee "$OUT/$name.log"
}

# Optimized config: seg=512 + prefix caching (APC) + fp8-KV (>=16k) + right-sized max-model-len.
# name   LEN    CTX     SEG  BUCKETS  KV
run_one 4k    5120   4096   512  512   auto
run_one 16k   17408  16384  512  512   fp8_e4m3
run_one 32k   33792  32768  512  512   fp8_e4m3
run_one 64k   66560  65536  512  512   fp8_e4m3

echo ""
echo "=================== SUMMARY ==================="
python3 "$HERE/summarize.py" --results "$OUT" | tee "$OUT/summary.txt"
echo ""
echo "Done. All results in: $OUT"

#!/usr/bin/env bash
# Throughput-vs-TTFT sweep for Gemma4-31B at TP=32, multi-bucket prefill.
# Sweeps max_num_seqs to lift the decode-batch ceiling while holding the
# weighted-avg TTFT under the 174 ms target. Runs INSIDE the container.
set -u
cd /work
export PYTHONPATH=/work
export NEURON_SKIP_EFA_AFFINITY=1

SNAP=/root/.cache/huggingface/hub/models--google--gemma-4-31b-it/snapshots/3548789868c5356dbf307c98e6f609007b82b3eb
PROG=/work/logs/sweep_progress.log
mkdir -p /work/logs /work/sweep_results
: > "$PROG"

log(){ echo "[$(date +%T)] $*" | tee -a "$PROG"; }

free_cores(){
  pkill -f "vllm serve" 2>/dev/null
  for i in $(seq 1 40); do
    busy=$(neuron-ls 2>/dev/null | grep -c "VLLM::Worker")
    [ "$busy" = "0" ] && { sleep 3; return 0; }
    sleep 3
  done
  for p in $(neuron-ls 2>/dev/null | grep "VLLM::Worker" | grep -oE '[0-9]{3,7}' | sort -u); do kill -9 "$p" 2>/dev/null; done
  sleep 5
}

wait_ready(){
  # poll up to 900s for /v1/models
  for i in $(seq 1 180); do
    if python3 - <<'PY' 2>/dev/null
import urllib.request,sys
try:
    d=urllib.request.urlopen("http://127.0.0.1:8000/v1/models",timeout=3).read().decode()
    sys.exit(0 if '"object"' in d else 1)
except Exception:
    sys.exit(1)
PY
    then return 0; fi
    sleep 5
  done
  return 1
}

for MNS in 4 8 16 32; do
  log "===== CONFIG max_num_seqs=$MNS (TP=32) ====="
  free_cores
  CFG="{\"neuron_config\":{\"quantization\":\"bf16\",\"on_device_sampling_config\":{\"all_greedy\":true},\"num_batched_tokens_buckets\":[512,1024,2048,4096],\"num_seqs_buckets\":[$MNS]}}"
  SLOG=/work/logs/serve_mns${MNS}.log
  log "launching server (log: $SLOG)"
  nohup vllm serve "$SNAP" \
    --tensor-parallel-size 32 \
    --max-model-len 4096 \
    --max-num-seqs $MNS \
    --max-num-batched-tokens 4096 \
    --additional-config "$CFG" > "$SLOG" 2>&1 &
  if ! wait_ready; then
    log "MNS=$MNS: SERVER FAILED TO BECOME READY — tail:"
    tail -8 "$SLOG" | tee -a "$PROG"
    log "MNS=$MNS: skipping benches (likely OOM at this batch). Continuing."
    continue
  fi
  log "MNS=$MNS: server READY. Running distribution TTFT bench..."
  python3 /work/bench_distribution.py --model "$SNAP" --runs 5 \
    --out /work/sweep_results/dist_mns${MNS}.json >> "$PROG" 2>&1
  log "MNS=$MNS: running throughput bench (in=1024 out=256, conc=4,$MNS)..."
  python3 /work/bench_throughput.py --model "$SNAP" \
    --concurrency "4,$MNS" --input-tokens 1024 --output-tokens 256 \
    --reqs-per-level 2 \
    --out /work/sweep_results/thru_mns${MNS}.json >> "$PROG" 2>&1
  log "MNS=$MNS: DONE"
done

log "===== SWEEP COMPLETE ====="
pkill -f "vllm serve" 2>/dev/null
log "ALL_DONE"

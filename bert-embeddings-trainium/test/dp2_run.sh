#!/bin/bash
# DP=2 on native+compile, MiniLM. Two parallel workers, ~2,048 prompts each.
set -e
echo "=== killing prior ==="
pkill -9 -f dp2_native_worker 2>/dev/null || true
sleep 4

MODEL="${MODEL:-sentence-transformers/all-MiniLM-L6-v2}"
N=2048
T0=$(date +%s.%N)

WORKER_ID=A NEURON_RT_VISIBLE_CORES=0-1 N_PROMPTS=$N MODEL="$MODEL" \
  OUT_FILE=/tmp/dp_native_A.json \
  python3 -u /tmp/dp2_native_worker.py > /tmp/dp_native_log_A.txt 2>&1 &
PA=$!
WORKER_ID=B NEURON_RT_VISIBLE_CORES=2-3 N_PROMPTS=$N MODEL="$MODEL" \
  OUT_FILE=/tmp/dp_native_B.json \
  python3 -u /tmp/dp2_native_worker.py > /tmp/dp_native_log_B.txt 2>&1 &
PB=$!

wait $PA; RC_A=$?
wait $PB; RC_B=$?
T1=$(date +%s.%N)

echo "=== exit: A=$RC_A B=$RC_B ==="
echo "=== A: ==="; cat /tmp/dp_native_A.json 2>/dev/null || tail -10 /tmp/dp_native_log_A.txt
echo "=== B: ==="; cat /tmp/dp_native_B.json 2>/dev/null || tail -10 /tmp/dp_native_log_B.txt
python3 - <<PY
import json
try:
    A = json.load(open("/tmp/dp_native_A.json"))
    B = json.load(open("/tmp/dp_native_B.json"))
    total = A["n_prompts"] + B["n_prompts"]
    slowest = max(A["run_s"], B["run_s"])
    combined = total / slowest
    print(f"\n=== DP=2 native+compile summary ===")
    print(f"total prompts: {total}")
    print(f"slowest worker run_s: {slowest:.4f}")
    print(f"combined seq_per_s: {combined:.1f}")
    print(f"per-worker A: {A['seq_per_s']}, B: {B['seq_per_s']}")
    json.dump({
      "n_total": total, "wall_s": round(slowest,4),
      "dp2_seq_per_s": round(combined,1),
      "per_worker_A": A["seq_per_s"], "per_worker_B": B["seq_per_s"],
      "model": A["model"],
    }, open("/tmp/dp2_native_summary.json","w"), indent=2)
    print("saved /tmp/dp2_native_summary.json")
except Exception as e:
    print(f"summary failed: {e}")
PY

#!/usr/bin/env bash
# Single-core memory-ceiling sweep for PixelDiT-XL (1B) on one NeuronCore.
# Runs each (batch,image,patch) config for 1 step; records FIT / OOM.
set -u
CORE=${CORE:-6}
OUT=/work/sweep_results.txt
: > "$OUT"
run() {
  B=$1; IMG=$2; P=$3
  echo "=== config B=$B IMG=$IMG P=$P ===" | tee -a "$OUT"
  log=/work/sweep_B${B}_IMG${IMG}_P${P}.log
  timeout 600 env NEURON_RT_VISIBLE_CORES=$CORE python3 -u /work/pixeldit_xl_train.py \
      --hidden 1408 --depth 28 --heads 16 --steps 1 \
      --batch $B --image-size $IMG --patch $P > "$log" 2>&1
  rc=$?
  if grep -q "\[done\]" "$log"; then
    ms=$(grep "first" "$log" | sed -E 's/.*= ([0-9.]+) ms/\1/')
    echo "RESULT B=$B IMG=$IMG P=$P -> FIT (compile+step ${ms} ms)" | tee -a "$OUT"
  elif grep -qiE "Failed to allocate|out of memory|OOM|bad_alloc|RESOURCE" "$log"; then
    echo "RESULT B=$B IMG=$IMG P=$P -> OOM/alloc-fail (rc=$rc)" | tee -a "$OUT"
  else
    echo "RESULT B=$B IMG=$IMG P=$P -> OTHER-FAIL (rc=$rc) see $log" | tee -a "$OUT"
  fi
}
run 1 256 16
run 2 256 16
run 4 256 16
run 8 256 16
run 1 512 16
echo "=== SWEEP DONE ===" | tee -a "$OUT"

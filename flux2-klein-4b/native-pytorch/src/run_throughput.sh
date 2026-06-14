#!/bin/bash
# Throughput harness — launch N independent single-rank FLUX.2-klein-4B
# pipelines, each pinned to its own pair of Neuron cores (LNC=2), and
# measure aggregate images/sec. This validates the "$/image at full
# instance utilization" claim that the benchmark doc projects but never
# measured concurrently.
#
# Each worker runs the cached single-rank pipeline (6.86s/image warm).
# On a trn2.48xl with 32 logical cores, N can go up to 16 (each worker
# uses 2 physical cores under LNC=2).
#
# Usage (inside container):
#   bash run_throughput.sh <N_WORKERS> <IMAGES_PER_WORKER>
set -e

N=${1:-4}
IMGS=${2:-3}
WORKDIR=/mnt/data/work/flux2
OUT=$WORKDIR/results/throughput_n${N}
mkdir -p $OUT

echo "=== Throughput test: $N workers x $IMGS images each ==="
echo "    (each worker = 1 single-rank pipeline on 2 cores, LNC=2)"

START=$(date +%s.%N)
pids=()
for w in $(seq 0 $((N-1))); do
  c0=$((w*2))
  c1=$((w*2+1))
  NEURON_RT_VISIBLE_CORES="${c0}-${c1}" \
  NEURON_RT_VIRTUAL_CORE_SIZE=2 \
  HF_HOME=/mnt/data/hf_cache \
  HF_TOKEN=$HF_TOKEN \
  NEURONX_DUMP_TO=$WORKDIR/neff_cache_4step \
  /opt/torch-neuronx/.venv/bin/python $WORKDIR/throughput_worker.py \
    --worker $w --images $IMGS \
    > $OUT/worker_${w}.log 2>&1 &
  pids+=($!)
done

echo "  launched $N workers, waiting..."
fail=0
for p in "${pids[@]}"; do
  if ! wait $p; then fail=$((fail+1)); fi
done
END=$(date +%s.%N)

WALL=$(echo "$END - $START" | bc)
TOTAL_IMGS=$((N * IMGS))
echo ""
echo "=== RESULTS ==="
echo "  workers:        $N"
echo "  images/worker:  $IMGS"
echo "  total images:   $TOTAL_IMGS"
echo "  wall-clock:     ${WALL}s"
echo "  failed workers: $fail"
python3 -c "print(f'  throughput:     {$TOTAL_IMGS/$WALL:.3f} images/sec')"
python3 -c "print(f'  per-image cost: \${$WALL/$TOTAL_IMGS * 21.50/3600:.4f} (at trn2.48xl \$21.50/hr, this {$N}-worker slice)')"
echo ""
echo "=== per-worker warm times ==="
for w in $(seq 0 $((N-1))); do
  echo "  worker $w:"; grep -E "warm|avg" $OUT/worker_${w}.log | tail -2 | sed 's/^/    /'
done

#!/bin/bash
# FLUX.2-klein-4B TP launcher — works with Beta 3 distributed.
#
# Usage:
#   ./launch_tp.sh 2 1024 1024 4    # TP=2, 1024x1024, 4 steps
#   ./launch_tp.sh 4 2048 2048 4    # TP=4, 2048x2048 (4 MP), 4 steps
#   ./launch_tp.sh 4 1920 1088 4    # TP=4, 1080p widescreen, 4 steps
#
# Required env (set already if running inside the beta3 container):
#   NEURON_RT_VIRTUAL_CORE_SIZE=2
#   NEURON_LOGICAL_NC_CONFIG=2          # required to match prebuilt NEFFs
#   NEURON_SKIP_EFA_AFFINITY=1          # bypass EFA on hosts/containers without EFA
#   HF_TOKEN=...                         # for HF model download

set -u

TP=${1:-2}
H=${2:-1024}
W=${3:-1024}
STEPS=${4:-4}
RUNS=${RUNS:-2}

OUTDIR=${OUTDIR:-/mnt/data/flux2_tp${TP}_${H}x${W}}
NEFF=${NEFF:-/mnt/data/flux2_tp${TP}_neff_${H}x${W}}

mkdir -p "$OUTDIR"
export NEURON_COMPILE_CACHE_URL="$NEFF"
export NEURON_RT_VIRTUAL_CORE_SIZE=2
export NEURON_LOGICAL_NC_CONFIG=2
export NEURON_SKIP_EFA_AFFINITY=1

cd /mnt/data/work/flux2_latest

torchrun \
    --nproc_per_node=$TP \
    --rdzv_backend c10d \
    --rdzv_endpoint localhost:29500 \
    --redirects 3 \
    --log-dir "$OUTDIR/torchrun_log" \
    run_flux2_tp.py \
        --height $H --width $W \
        --steps $STEPS \
        --runs $RUNS \
        --output "$OUTDIR/out.png" \
    2>&1 | tee "$OUTDIR/launch.log"

echo
echo "=== Logs ==="
echo "Per-rank: $OUTDIR/torchrun_log/"
echo "Main: $OUTDIR/launch.log"
echo "Output PNG: $OUTDIR/out.png"

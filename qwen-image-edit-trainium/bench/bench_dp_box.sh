#!/bin/bash
# Launch 4 worker replicas + 4 servers on disjoint Neuron core sets,
# then run bench_dp_box.py against all four to measure full-box DP
# throughput on a trn2.48xlarge (16 cores total, TP=4 per worker).
#
# Each worker uses NEURON_RT_VISIBLE_CORES to claim 4 cores. Each
# worker writes to its own Unix socket. Each FastAPI server listens on
# a different port and connects to the corresponding socket.
#
# WARNING: This burns ~22 GB user budget × 4 cores × 4 workers = 16
# core-equivalents, which IS the full trn2.48xl. Don't run alongside
# the single-worker setup.
set -euo pipefail

cd /work/path_c

# Cleanup
pkill -9 -f worker.py 2>/dev/null || true
pkill -9 -f uvicorn 2>/dev/null || true
pkill -9 -f torchrun 2>/dev/null || true
rm -f /tmp/fal_pipeline_*.sock /tmp/fal_request_*.json
sleep 3

source /opt/torch-neuronx/.venv/bin/activate
export HF_HOME=/root/.cache/huggingface
export TORCH_NEURONX_FALLBACK_ONLY_FOR_UNIMPLEMENTED_OPS=0
export TOKENIZERS_PARALLELISM=false

# Logs go here
mkdir -p /work/path_c/logs/dp_bench

for i in 0 1 2 3; do
    LO=$((i * 4))
    HI=$((LO + 3))
    SOCK="/tmp/fal_pipeline_${i}.sock"
    PORT=$((8000 + i))
    LOG="/work/path_c/logs/dp_bench/worker_${i}.log"

    echo "[dp] launching worker ${i}: cores ${LO}-${HI}, sock ${SOCK}, port ${PORT}"
    NEURON_RT_VISIBLE_CORES="${LO}-${HI}" \
        torchrun \
            --nproc_per_node=4 --standalone \
            --master_port=$((29500 + i)) \
            serve/worker.py \
            --base-model-path /root/.cache/huggingface/models--Qwen--Qwen-Image-Edit-2511/snapshots/6f3ccc0b56e431dc6a0c2b2039706d7d26f22cb9 \
            --merged-transformer /opt/dlami/nvme/fal/merged_lora/transformer \
            --tp 4 \
            --socket-path "${SOCK}" >"${LOG}" 2>&1 &

    sleep 2
done

# Wait for all 4 workers to finish weight load + first compile (~5-15 min cold).
# Tail logs and look for "pipeline ready, entering serve loop".
echo "[dp] waiting for 4 workers to come up; check /work/path_c/logs/dp_bench/*.log"
echo "[dp] expect 5-15 min on cold start"

# Now spin up 4 FastAPI servers (in same shell, backgrounded)
for i in 0 1 2 3; do
    SOCK="/tmp/fal_pipeline_${i}.sock"
    PORT=$((8000 + i))
    LOG="/work/path_c/logs/dp_bench/server_${i}.log"

    # Wait for the corresponding socket to appear
    while [ ! -S "${SOCK}" ]; do
        sleep 5
    done
    echo "[dp] socket ${SOCK} ready, starting server on port ${PORT}"

    FAL_SOCKET_PATH="${SOCK}" \
        uvicorn serve.server:app --host 0.0.0.0 --port "${PORT}" \
        --workers 1 --log-level warning >"${LOG}" 2>&1 &
done

sleep 10
echo "[dp] all up; ports 8000-8003"
echo "[dp] sample bench:"
echo "  python serve/bench_dp_box.py \\"
echo "      --hosts http://localhost:8000,http://localhost:8001,http://localhost:8002,http://localhost:8003 \\"
echo "      --concurrency 4 --duration 600"

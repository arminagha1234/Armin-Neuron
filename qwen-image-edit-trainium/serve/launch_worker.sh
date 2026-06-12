#!/bin/bash
# Launch the persistent torchrun worker. Mirrors run_compiled_28step.sh
# env so the worker bakes the same NEFF cache as the production runner.
#
# Usage:
#   bash /work/path_c/serve/launch_worker.sh
#
# After this comes up (logs "pipeline ready, entering serve loop"),
# launch the FastAPI server with launch_server.sh.
set -euo pipefail

pkill -9 -f run_simple.py 2>/dev/null || true
pkill -9 -f run_compiled.py 2>/dev/null || true
pkill -9 -f run_real_rope.py 2>/dev/null || true
pkill -9 -f serve/worker.py 2>/dev/null || true
pkill -9 -f torchrun 2>/dev/null || true
sleep 3

source /opt/torch-neuronx/.venv/bin/activate
export HF_HOME=/root/.cache/huggingface
export TORCH_NEURONX_FALLBACK_ONLY_FOR_UNIMPLEMENTED_OPS=0
export TOKENIZERS_PARALLELISM=false
cd /work/path_c

NEURON_RT_NUM_CORES=4 torchrun --nproc_per_node=4 --standalone serve/worker.py \
    --base-model-path /root/.cache/huggingface/models--Qwen--Qwen-Image-Edit-2511/snapshots/6f3ccc0b56e431dc6a0c2b2039706d7d26f22cb9 \
    --merged-transformer /opt/dlami/nvme/fal/merged_lora/transformer \
    --tp 4 \
    --socket-path /tmp/fal_pipeline.sock 2>&1

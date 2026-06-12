#!/bin/bash
# 28-step quality run
pkill -9 -f run_simple.py 2>/dev/null || true
pkill -9 -f torchrun 2>/dev/null || true
sleep 3

source /opt/torch-neuronx/.venv/bin/activate
export HF_HOME=/root/.cache/huggingface
export TORCH_NEURONX_FALLBACK_ONLY_FOR_UNIMPLEMENTED_OPS=0
export NEURON_LAUNCH_BLOCKING=1
export TOKENIZERS_PARALLELISM=false
cd /work/path_c

NEURON_RT_NUM_CORES=4 torchrun --nproc_per_node=4 --standalone run_simple.py \
    --base-model-path /root/.cache/huggingface/models--Qwen--Qwen-Image-Edit-2511/snapshots/6f3ccc0b56e431dc6a0c2b2039706d7d26f22cb9 \
    --merged-transformer /opt/dlami/nvme/fal/merged_lora/transformer \
    --images /work/path_c/results/test_input.png \
    --prompt show_from_a_different_camera_angle \
    --num-steps 28 \
    --height 512 --width 512 \
    --output /work/path_c/results/run_28step/output.png 2>&1

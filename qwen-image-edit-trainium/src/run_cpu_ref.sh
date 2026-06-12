#!/bin/bash
# CPU reference run (no torchrun, no Neuron). Run inside fal_beta2
# container so we use the same diffusers/transformers as the Trainium
# pipeline.
source /opt/torch-neuronx/.venv/bin/activate
export HF_HOME=/root/.cache/huggingface
export TOKENIZERS_PARALLELISM=false
cd /work/path_c

python run_cpu_ref.py \
    --base-model-path /root/.cache/huggingface/models--Qwen--Qwen-Image-Edit-2511/snapshots/6f3ccc0b56e431dc6a0c2b2039706d7d26f22cb9 \
    --merged-transformer /opt/dlami/nvme/fal/merged_lora/transformer \
    --image /work/path_c/results/test_input.png \
    --prompt show_from_a_different_camera_angle \
    --num-steps 28 \
    --height 512 --width 512 \
    --output /work/path_c/results/output_cpu_ref.png 2>&1

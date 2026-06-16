#!/bin/bash
# Launch full 64-layer Qwen3.6-27B LoRA SFT via accelerate FSDP=16 on Neuron.
#
# Validated: 2026-06-16 on trn2.48xlarge (16 cores, ~2TB RAM).
# Result:   loss 3.91 -> 0.32, 10 steps, ~49s/step, ACC_EXIT=0.
#
# Run inside the Beta 3 DLC container:
#   sudo docker exec neuron_grpo bash -lc 'bash /mnt/data/launch_full_sft.sh'

set -e

# --- Required Neuron environment ---
export ON_NEURON=1
export ACCELERATE_TORCH_DEVICE=neuron
export ACCELERATE_USE_FSDP=1
export NEURON_RT_VISIBLE_CORES=0-15
export NEURON_RT_VIRTUAL_CORE_SIZE=2
export TORCH_NEURONX_NEFF_CACHE_DIR=/mnt/data/neff_cache
export TORCH_NEURONX_NEFF_LOCAL_CACHE_DIR=/mnt/data/neff_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TORCH_NEURONX_LOG_LEVEL=2

# --- Training config (override as needed) ---
export MODEL=/mnt/data/models/Qwen3.6-27B
export SEQ=500
export STEPS=10

# --- FSDP config derivation ---
# Start from czkkkkkk's deepmath fsdp16.yaml and apply 4 fixes for the 27B hybrid.
#
# Why NOT ram_efficient: rank0 broadcasts each full layer to the other 15 ranks
# on-device; the repeated full-layer alloc/free fragments HBM (saw 16GB free
# but largest contiguous chunk 58MB -> neuron::alloc::lazy OOM). With ~2TB host
# RAM, every rank loads full bf16 to CPU and FSDP2 slices directly to device.
#
# Why NOT offload: grad-norm clip all_reduce runs on CPU DTensors, but the
# Neuron PG has no CPU/gloo backend ("No backend type associated with device
# type cpu").
FSDP_CFG=/mnt/data/fsdp16_ram.yaml
cp /mnt/data/rl_examples/deepmath/accelerate_configs/fsdp16.yaml $FSDP_CFG

# Fix 3: keep ram_efficient OFF
sed -i 's/fsdp_cpu_ram_efficient_loading: true/fsdp_cpu_ram_efficient_loading: false/' $FSDP_CFG

# Fix 4: keep offload OFF
sed -i 's/fsdp_offload_params: true/fsdp_offload_params: false/' $FSDP_CFG

# Fix 2: pin wrap class to the text decoder layer (avoids VisionBlock lookup error)
grep -q 'fsdp_transformer_layer_cls_to_wrap' $FSDP_CFG \
  || sed -i '/fsdp_auto_wrap_policy:/a\  fsdp_transformer_layer_cls_to_wrap: Qwen3_5DecoderLayer' $FSDP_CFG

# --- Launch ---
cd /mnt/data
/mnt/data/miniconda3/envs/test_eager/bin/accelerate launch \
  --config_file $FSDP_CFG \
  /mnt/data/full_sft.py > /mnt/data/acc.log 2>&1
echo "ACC_EXIT=$?" >> /mnt/data/acc.log

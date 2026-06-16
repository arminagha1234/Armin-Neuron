#!/bin/bash
# Sequence-length scan: same FSDP config as launch_full_sft.sh, but points
# at the long-content scan script (phase3_long_seq_scan.py renamed to
# /mnt/data/long_sft.py on the box) and writes /mnt/data/long.log.
# See SEQ_SCALING.md for the validated rungs and where it walls.
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
export MODEL=/mnt/data/models/Qwen3.6-27B
export SEQ=1024
export STEPS=5

# --- FSDP config derivation (see qwen3.6/SETUP.md for the why) ---
# Host is a trn2.48xl with ~2TB RAM, so we do NOT use ram_efficient loading
# or CPU param offload. Both caused failures on the 27B hybrid model:
#   * fsdp_cpu_ram_efficient_loading=true  -> rank0 broadcasts each full layer
#     to the other 15 ranks on-device; the repeated full-layer alloc/free
#     fragments HBM (saw 16GB free but largest contiguous chunk 58MB ->
#     neuron::alloc::lazy NRT_RESOURCE OOM at FSDP prepare).
#   * fsdp_offload_params=true             -> grad-norm clip all_reduce then
#     runs on CPU-resident DTensors, but the Neuron PG has no CPU backend
#     ("No backend type associated with device type cpu").
# With 2TB RAM every rank loads the full bf16 model to CPU (16 x 54GB < 2TB)
# and FSDP2 slices each shard straight to device -- no transient device
# gather (no fragmentation) and no CPU-side collectives.
cp /mnt/data/rl_examples/deepmath/accelerate_configs/fsdp16.yaml /mnt/data/fsdp16_ram.yaml
# Keep ram_efficient + offload OFF (defaults in the deepmath yaml are false).
sed -i 's/fsdp_cpu_ram_efficient_loading: true/fsdp_cpu_ram_efficient_loading: false/' /mnt/data/fsdp16_ram.yaml
sed -i 's/fsdp_offload_params: true/fsdp_offload_params: false/' /mnt/data/fsdp16_ram.yaml
# Pin the wrap class to the text decoder layer. The model's _no_split_modules
# also lists Qwen3_5VisionBlock, which isn't instantiated in a text-only
# AutoModelForCausalLM load, so TRANSFORMER_BASED_WRAP fallback errors trying
# to find it. Setting the class explicitly avoids the vision block lookup.
grep -q 'fsdp_transformer_layer_cls_to_wrap' /mnt/data/fsdp16_ram.yaml \
  || sed -i '/fsdp_auto_wrap_policy:/a\  fsdp_transformer_layer_cls_to_wrap: Qwen3_5DecoderLayer' /mnt/data/fsdp16_ram.yaml

cd /mnt/data
/mnt/data/miniconda3/envs/test_eager/bin/accelerate launch \
  --config_file /mnt/data/fsdp16_ram.yaml \
  /mnt/data/long_sft.py > /mnt/data/long.log 2>&1
echo "ACC_EXIT=$?" >> /mnt/data/long.log

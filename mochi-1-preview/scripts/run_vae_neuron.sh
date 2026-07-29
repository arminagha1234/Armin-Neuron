#!/bin/bash
export HF_HOME=/host/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export NEURON_RT_NUM_CORES=4 TORCH_NEURONX_ENABLE_HOST_CC=1 TORCH_NEURONX_ENABLE_ASYNC_NRT=1
export OMP_NUM_THREADS=48 MKL_NUM_THREADS=48 TOKENIZERS_PARALLELISM=false
cd /host/Mochi
torchrun --nnodes 1 --nproc_per_node 4 --rdzv_backend c10d --rdzv_endpoint localhost:29500   src/run_mochi_native.py --num-frames 19 --num-steps 8 --guidance-scale 1.0   --vae-device neuron --output results/mochi_vae_neuron.mp4 --report results/report_vae_neuron.json

#!/usr/bin/env bash
# Launch native-PyTorch FSDP2 training of VideoMAEv2 across 2 NeuronCores on trn2.3xlarge.
set -e
source "$HOME/workspace/native_venv/bin/activate"
cd "$HOME/vmae"
# 2 logical cores under default LNC2. torchrun sets rank/world/master env.
torchrun --standalone --nproc_per_node=2 train_fsdp_neuron.py

#!/usr/bin/env bash
# Multi-core FSDP2 pretraining of VideoMAE v2 across 2 NeuronCores on trn2.3xlarge.
set -e
source "$HOME/workspace/native_venv/bin/activate"
cd "$HOME/vmae"
torchrun --standalone --nproc_per_node=2 train_fsdp_pretrain.py

#!/bin/bash
#SBATCH --job-name=deepmath
#SBATCH --partition=compute1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=192
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Create logs directory and snapshot scripts/configs for reproducibility
mkdir -p logs
SCRIPT_DIR=logs/job-${SLURM_JOB_ID}-scripts
mkdir -p "$SCRIPT_DIR"
cp run.slurm "$SCRIPT_DIR/"
cp main.py default_main.py "$SCRIPT_DIR/"
cp -r grpo_configs accelerate_configs "$SCRIPT_DIR/"

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate grpo_baseline

# Set working directory
cd /path/to/deepmath

export SLURM_TB_LOG_DIR=output/runs/tb-${SLURM_JOB_ID}
export VLLM_COLOCATE_DTYPE="bfloat16"

accelerate launch \
    --config_file "$SCRIPT_DIR/accelerate_configs/ddp.yaml" \
    "$SCRIPT_DIR/main.py" \
    --config "$SCRIPT_DIR/grpo_configs/grpo.yaml"

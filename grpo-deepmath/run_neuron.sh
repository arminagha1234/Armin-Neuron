
export SLURM_TB_LOG_DIR=output/runs/tb-${SLURM_JOB_ID}

export TORCH_NEURONX_NEFF_CACHE_DIR="/path/to/neff_cache/"
export TORCH_NEURONX_NEFF_LOCAL_CACHE_DIR="/path/to/neff_local_cache/"
export ON_NEURON=1
export ACCELERATE_TORCH_DEVICE=neuron
export TORCH_NEURONX_LOG_LEVEL=2
# export ACCELERATE_USE_FSDP=true

source /path/to/miniconda3/bin/activate eager
SCRIPT_DIR=$(dirname "$(realpath "$0")")

# CONFIG_FILE="$SCRIPT_DIR/accelerate_configs/ddp.yaml"
CONFIG_FILE="$SCRIPT_DIR/accelerate_configs/fsdp.yaml"
accelerate launch \
    --config_file "$CONFIG_FILE" \
    "$SCRIPT_DIR/main.py" \
    --config "$SCRIPT_DIR/grpo_configs/grpo.yaml"
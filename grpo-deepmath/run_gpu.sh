
# export ACCELERATE_USE_FSDP=true

source /path/to/miniconda3/bin/activate eager
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
SCRIPT_DIR=$(dirname "$(realpath "$0")")

# CONFIG_FILE="$SCRIPT_DIR/accelerate_configs/ddp.yaml"
CONFIG_FILE="$SCRIPT_DIR/accelerate_configs/ddp.yaml"
accelerate launch \
    --config_file "$CONFIG_FILE" \
    "$SCRIPT_DIR/main.py" \
    --config "$SCRIPT_DIR/grpo_configs/grpo_gpu.yaml"

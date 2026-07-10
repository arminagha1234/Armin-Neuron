# source /path/to/miniconda3/bin/activate grpo_baseline
source /path/to/miniconda3/bin/activate eager

export TORCH_DEVICE_BACKEND_AUTOLOAD=0
PORT=8001

# model_id="Qwen/Qwen2-0.5B-Instruct"
model_id="Qwen/Qwen3-0.6B"

CUDA_VISIBLE_DEVICES=7 trl vllm-serve \
    --model "$model_id" \
    --vllm_model_impl transformers \
    --port "$PORT"
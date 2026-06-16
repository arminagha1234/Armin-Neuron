"""Full 64-layer Qwen3.6-27B LoRA SFT via FSDP across 16 Neuron cores.

Launched with `accelerate launch --config_file fsdp16.yaml`. Loads the
real 27B weights (FSDP shards the frozen base across 16 ranks), attaches
PEFT-LoRA, runs trl SFTTrainer on a tiny real dataset.

Validated: 2026-06-16, trn2.48xlarge, loss 3.91 -> 0.32 over 10 steps.

Requirements:
  - czkkkkkk/transformers (neuron branch) — has differentiable DeltaNet
  - czkkkkkk/trl (neuron branch)
  - accelerate (main)
  - peft, datasets
  - torch_neuronx 2.11.3.0.19138 wheel
  - Set: ON_NEURON=1, ACCELERATE_TORCH_DEVICE=neuron, ACCELERATE_USE_FSDP=1
"""
import os, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig
from datasets import Dataset
try:
    from trl import SFTConfig, SFTTrainer
except ImportError:
    from trl.trainer.sft_config import SFTConfig
    from trl.trainer.sft_trainer import SFTTrainer

# Accelerate gather patch for Neuron (0-d tensors) — harmless for SFT.
if os.environ.get("ON_NEURON") == "1":
    try:
        import accelerate.utils.operations as _ops
        _orig = _ops._gpu_gather
        def _patched(tensor):
            import torch as _t
            def _flat(t):
                return t.view(-1) if isinstance(t, _t.Tensor) and t.dim() == 0 else t
            from accelerate.utils.operations import recursively_apply
            return _orig(recursively_apply(_flat, tensor))
        _ops._gpu_gather = _patched
    except Exception as e:
        print("gather patch skipped:", e, flush=True)

# --- Configuration (override via environment) ---
MODEL = os.environ.get("MODEL", "/mnt/data/models/Qwen3.6-27B")
SEQ = int(os.environ.get("SEQ", "500"))
STEPS = int(os.environ.get("STEPS", "10"))
LR = float(os.environ.get("LR", "5e-4"))
rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))

# --- Load model ---
if rank == 0:
    print("loading FULL 64-layer model (FSDP will shard)", flush=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, torch_dtype=torch.bfloat16, trust_remote_code=True,
    attn_implementation="eager", low_cpu_mem_usage=True,
)
model.config.use_cache = False

# --- Tokenizer ---
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

# --- Dataset (replace with your own for real training) ---
base = [
    "### Instruction:\nWhat is the capital of France?\n### Response:\nThe capital of France is Paris.",
    "### Instruction:\nName a primary color.\n### Response:\nRed is a primary color.",
    "### Instruction:\nWhat is 2 plus 2?\n### Response:\n2 plus 2 equals 4.",
]
ds = Dataset.from_list([{"text": base[i % len(base)]} for i in range(48)])

# --- LoRA config ---
# Targets both standard transformer projections AND DeltaNet-specific ones.
lora = LoraConfig(
    r=8, lora_alpha=16, lora_dropout=0.0, bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj",
                    "in_proj_qkvz", "in_proj_ba", "out_proj"],
)

# --- SFT config ---
# NOTE: gradient_checkpointing MUST be False when FSDP config has
# fsdp_activation_checkpointing=true — the two can't both be on.
sft = SFTConfig(
    output_dir="/mnt/data/qwen36_full_sft_out",
    max_length=SEQ,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    max_steps=STEPS,
    learning_rate=LR,
    logging_steps=1,
    bf16=True,
    gradient_checkpointing=False,
    report_to=[],
    save_strategy="no",
)

# --- Train ---
trainer = SFTTrainer(
    model=model, args=sft, train_dataset=ds,
    processing_class=tok, peft_config=lora,
)
if rank == 0:
    print("SFTTrainer built; training full 27B...", flush=True)
trainer.train()

# --- Report ---
hist = [h["loss"] for h in trainer.state.log_history if "loss" in h]
if rank == 0 and hist:
    print("FULL_SFT_RESULT first=%.4f last=%.4f min=%.4f decreased=%s" % (
        hist[0], hist[-1], min(hist), hist[-1] < hist[0]), flush=True)

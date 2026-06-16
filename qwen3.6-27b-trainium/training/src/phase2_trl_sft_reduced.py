"""Reduced-layer (8) TRL SFT validation on Neuron — quick smoke test.

Runs the same training pipeline as the full 64-layer script but on an
8-layer reduced model (~5.6B) that fits on a single Neuron core pair.
Use this to validate the env + training pipeline before committing to the
full 16-core FSDP run.

Usage (in-container, single core pair):
  sudo docker exec -e NEURON_RT_VISIBLE_CORES=4-5 -e NEURON_RT_VIRTUAL_CORE_SIZE=2 \
    -e ACCELERATE_TORCH_DEVICE=neuron -e ON_NEURON=1 \
    -e MODEL=/mnt/data/models/Qwen3.6-27B -e SEQ=500 -e NLAYERS=8 -e STEPS=30 \
    -e TORCH_NEURONX_NEFF_CACHE_DIR=/mnt/data/neff_cache \
    neuron_grpo /mnt/data/miniconda3/envs/test_eager/bin/python -u phase2_trl_sft_reduced.py

Expected: loss decreases over 30 steps (e.g. 13.02 -> 0.53).
"""
import os, torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig
from datasets import Dataset
try:
    from trl import SFTConfig, SFTTrainer
except ImportError:
    from trl.trainer.sft_config import SFTConfig
    from trl.trainer.sft_trainer import SFTTrainer

MODEL = os.environ.get("MODEL", "/mnt/data/models/Qwen3.6-27B")
SEQ = int(os.environ.get("SEQ", "500"))
NLAYERS = int(os.environ.get("NLAYERS", "8"))
STEPS = int(os.environ.get("STEPS", "30"))

# Build reduced-layer config from the full model config.
cfg = AutoConfig.from_pretrained(MODEL, trust_remote_code=True)
text_cfg = getattr(cfg, "text_config", None)
_cfgs = [cfg] + ([text_cfg] if text_cfg is not None and text_cfg is not cfg else [])
for c in _cfgs:
    if hasattr(c, "num_hidden_layers"):
        c.num_hidden_layers = NLAYERS
    lt = getattr(c, "layer_types", None)
    if isinstance(lt, list):
        c.layer_types = lt[:NLAYERS]
if text_cfg is not None and text_cfg is not cfg:
    for k, v in vars(text_cfg).items():
        if not k.startswith("_") and not hasattr(cfg, k):
            setattr(cfg, k, v)

print("building reduced model %d layers" % NLAYERS, flush=True)
model = AutoModelForCausalLM.from_config(
    cfg, trust_remote_code=True, attn_implementation="eager"
).to(torch.bfloat16)
model.config.use_cache = False

tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

# Tiny learnable dataset.
base = [
    "### Instruction:\nWhat is the capital of France?\n### Response:\nThe capital of France is Paris.",
    "### Instruction:\nName a primary color.\n### Response:\nRed is a primary color.",
    "### Instruction:\nWhat is 2 plus 2?\n### Response:\n2 plus 2 equals 4.",
    "### Instruction:\nWhat language do Trainium kernels use?\n### Response:\nNKI, the Neuron Kernel Interface.",
]
ds = Dataset.from_list([{"text": base[i % len(base)]} for i in range(64)])

lora = LoraConfig(
    r=8, lora_alpha=16, lora_dropout=0.0, bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj",
                    "in_proj_qkvz", "in_proj_ba", "out_proj"],
)

sft = SFTConfig(
    output_dir="/mnt/data/qwen36_sft_out",
    max_length=SEQ, per_device_train_batch_size=1,
    gradient_accumulation_steps=1, max_steps=STEPS,
    learning_rate=5e-4, logging_steps=1, bf16=True,
    report_to=[], save_strategy="no",
)

trainer = SFTTrainer(
    model=model, args=sft, train_dataset=ds,
    processing_class=tok, peft_config=lora,
)
print("SFTTrainer built; training...", flush=True)
trainer.train()

hist = [h["loss"] for h in trainer.state.log_history if "loss" in h]
if hist:
    print("TRL_SFT_RESULT first=%.4f last=%.4f min=%.4f decreased=%s" % (
        hist[0], hist[-1], min(hist), hist[-1] < hist[0]), flush=True)

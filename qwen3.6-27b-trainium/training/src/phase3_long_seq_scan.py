"""Sequence-length scan with REAL long content (not padded toy data).

Builds each training example by concatenating base examples until the
tokenized sequence reaches ~SEQ tokens. This way per-step time actually
reflects long-context compute, not padding.
"""
import os, time, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig
from datasets import Dataset
try:
    from trl import SFTConfig, SFTTrainer
except ImportError:
    from trl.trainer.sft_config import SFTConfig
    from trl.trainer.sft_trainer import SFTTrainer

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

MODEL = os.environ.get("MODEL", "/mnt/data/models/Qwen3.6-27B")
SEQ = int(os.environ.get("SEQ", "8192"))
STEPS = int(os.environ.get("STEPS", "5"))
rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))

if rank == 0:
    print(f"loading model for seq={SEQ}", flush=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, torch_dtype=torch.bfloat16, trust_remote_code=True,
    attn_implementation="eager", low_cpu_mem_usage=True,
)
model.config.use_cache = False

tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

# Build LONG examples: concat base passages until we reach ~SEQ tokens.
PASSAGE = (
    "The Trainium2 chip from AWS provides high throughput for transformer "
    "training and inference. It uses systolic arrays organized into Neuron "
    "cores, with logical neuron cores that can be combined for larger "
    "workloads. The NKI programming model lets developers write custom "
    "kernels at a high level while still mapping efficiently to the "
    "hardware. Tensor parallelism shards weights across cores, while "
    "fully sharded data parallelism shards optimizer state and gradients. "
    "Linear attention layers like GatedDeltaNet scale better than standard "
    "attention on long sequences because their compute grows linearly "
    "rather than quadratically with sequence length. "
)

# Tokenize once to figure out how many repeats we need.
passage_tok_count = len(tok.encode(PASSAGE, add_special_tokens=False))
target_tokens = int(SEQ * 0.85)  # 85% real tokens, 15% padding headroom
repeats = max(1, target_tokens // passage_tok_count)
long_text = (PASSAGE * repeats).strip()
real_tokens = len(tok.encode(long_text, add_special_tokens=False))
if rank == 0:
    print(f"per-passage tokens={passage_tok_count} repeats={repeats} "
          f"per-example real tokens={real_tokens} (target {target_tokens})", flush=True)

ds = Dataset.from_list([{"text": long_text} for _ in range(48)])

lora = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.0, bias="none",
                  task_type="CAUSAL_LM",
                  target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                  "gate_proj", "up_proj", "down_proj",
                                  "in_proj_qkvz", "in_proj_ba", "out_proj"])

sft = SFTConfig(output_dir=f"/mnt/data/qwen36_seq{SEQ}_out",
                max_length=SEQ, per_device_train_batch_size=1,
                gradient_accumulation_steps=1, max_steps=STEPS,
                learning_rate=5e-4, logging_steps=1, bf16=True,
                gradient_checkpointing=False, report_to=[],
                save_strategy="no")

trainer = SFTTrainer(model=model, args=sft, train_dataset=ds,
                     processing_class=tok, peft_config=lora)
if rank == 0:
    print(f"SFTTrainer built; training seq={SEQ}", flush=True)
t0 = time.time()
trainer.train()
elapsed = time.time() - t0

hist = [h["loss"] for h in trainer.state.log_history if "loss" in h]
toks = [h.get("num_tokens") for h in trainer.state.log_history if "num_tokens" in h]
if rank == 0:
    print(f"SCAN_RESULT seq={SEQ} steps={STEPS} elapsed={elapsed:.1f}s "
          f"first_loss={hist[0]:.3f} last_loss={hist[-1]:.3f} "
          f"first_num_tokens={toks[0] if toks else 'NA'} "
          f"last_num_tokens={toks[-1] if toks else 'NA'}", flush=True)

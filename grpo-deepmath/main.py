# train_grpo.py
import os
from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer, TrlParser
from trl.rewards import accuracy_reward
from math_verify import parse
from transformers import TrainerCallback
if os.environ.get("ON_NEURON") == "1":
    import torch_neuronx._C as _C

class NeuronCacheDiagnosticsCallback(TrainerCallback):
    """Logs Neuron cache sizes per training step to diagnose OOM / recompilation."""

    def on_step_end(self, args, state, control, **kwargs):
        sizes = _C._get_all_cache_sizes()
        print(
            f"[NEURON_CACHE] step={state.global_step} "
            f"compilation={sizes['compilation_cache_entries']} "
            f"compilation_mem_MB={sizes['compilation_cache_memory_bytes'] / 1e6:.1f} "
            f"model_handles={sizes['model_handle_cache_entries']} "
            f"neff_loaded_MB={sizes['model_handle_total_neff_bytes'] / 1e6:.1f} "
            f"merged_ops={sizes['merged_operation_cache_entries']}"
        )

parser = TrlParser(GRPOConfig)
(config,) = parser.parse_args_and_config()

if os.environ.get("SLURM_TB_LOG_DIR"):
    config.logging_dir = os.environ["SLURM_TB_LOG_DIR"]

SYSTEM_PROMPT = "Solve the math problem. Put your final answer in \\boxed{}."

dataset = load_dataset("trl-lib/DeepMath-103K", split="train")
dataset = dataset.filter(lambda x: len(parse(x["solution"])) > 0)
dataset = dataset.map(
    lambda x: {"prompt": [{"role": "system", "content": SYSTEM_PROMPT}] + x["prompt"]}
)

trainer = GRPOTrainer(
    args=config,
    # model="Qwen/Qwen2-0.5B-Instruct",
    model="Qwen/Qwen3-0.6B",
    reward_funcs=accuracy_reward,
    train_dataset=dataset,
    callbacks=[NeuronCacheDiagnosticsCallback()] if os.environ.get("ON_NEURON") == "1" else [],
)
trainer.train()
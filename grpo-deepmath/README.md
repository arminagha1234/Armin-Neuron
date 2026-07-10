# DeepMath GRPO Training on AWS Trainium

A [TRL](https://github.com/huggingface/trl) **GRPO** example for mathematical
reasoning, with the training loop running on **AWS Trainium (Neuron eager mode)**.

- **Model:** `Qwen/Qwen3-0.6B`
- **Dataset:** `trl-lib/DeepMath-103K`
- **Algorithm:** GRPO with accuracy reward
- **Training config:** 2000 steps, per-device batch 1 × grad-accum 8, lr 5e-7
  (cosine), beta 0.001, 8 generations/prompt, max completion 1024
- **Parallelism:** FSDP2, 4 NeuronCores (`dp_shard=4`), bf16

> Cluster-specific paths (conda env, NEFF cache, project dir) are placeholders
> (`/path/to/...`, `/fsx/USER`); the vLLM host is `<VLLM_SERVER_HOST>`. Set these
> for your environment. The `gitlab.aws.dev` dependency branches are **internal AWS**
> repos carrying the Neuron-eager patches to `trl`/`transformers`/`accelerate`.

## Architecture (why there are two processes)

GRPO alternates **generation** (sample completions) and **training** (policy update).
This example runs them as two processes connected over TCP:

```
   ┌─────────────────────────┐        rollouts / weights        ┌──────────────────────┐
   │  vLLM generation server  │  <───────────────────────────>  │  Trainium training    │
   │  (trl vllm-serve)        │      port 8001 / group 51217     │  (run_neuron.sh, FSDP)│
   └─────────────────────────┘                                  └──────────────────────┘
```

The **training loop is Trainium-only** (`run_neuron.sh`). The generation server
(`vllm_serve/run_vllm.sh`) is a separate process — as written it targets a GPU
(`CUDA_VISIBLE_DEVICES`); it can also be pointed at a vLLM-Neuron instance (see
note in Step 3).

## Files
```
grpo-deepmath/
├── main.py                       # GRPO training script (+ Neuron cache diagnostics)
├── run_neuron.sh                 # Trainium launcher (FSDP, server-mode vLLM)
├── accelerate_configs/fsdp.yaml  # FSDP2 config, 4 NeuronCores
├── grpo_configs/grpo.yaml        # GRPO hyperparameters (Neuron)
└── vllm_serve/run_vllm.sh        # vLLM generation server launcher
```

## Prerequisites
- A Trainium instance (e.g. trn2) with the **Neuron eager** stack installed and a
  working conda/venv (`accelerate`, `trl`, `transformers`, `torch-neuronx`).
- A reachable **vLLM generation server** (GPU box or vLLM-Neuron instance) on the
  same network as the Trainium box.

## Setup

Install the Neuron-eager-patched libraries (editable) from the internal branches:
```bash
git clone https://gitlab.aws.dev/scale/scale-fte/hf-post-training/accelerate   && pip install -e ./accelerate
git clone https://gitlab.aws.dev/scale/scale-fte/hf-post-training/transformers && pip install -e ./transformers
git clone https://gitlab.aws.dev/scale/scale-fte/hf-post-training/trl          && pip install -e ./trl
pip install math_verify datasets
```

## Step-by-step

### Step 1 — edit the two paths for your environment
- `run_neuron.sh`: set the conda activate path and the two `TORCH_NEURONX_NEFF*_CACHE_DIR`.
- `grpo_configs/grpo.yaml`: set `vllm_server_host` to your generation server's IP and
  confirm `vllm_server_port: 8001` / `vllm_group_port: 51217`.

### Step 2 — start the generation server (separate box)
```bash
bash vllm_serve/run_vllm.sh        # serves Qwen/Qwen3-0.6B via `trl vllm-serve` on :8001
```
> To generate on **Trainium** instead of GPU, run a vLLM-Neuron server for
> `Qwen/Qwen3-0.6B` on :8001 in place of this script (drop `CUDA_VISIBLE_DEVICES`).
Ensure the Trainium and server boxes can reach each other on ports **8001** and **51217**.

### Step 3 — launch GRPO training on Trainium
```bash
bash run_neuron.sh
```
This sets `ON_NEURON=1`, `ACCELERATE_TORCH_DEVICE=neuron`, and
`accelerate launch --config_file accelerate_configs/fsdp.yaml main.py
--config grpo_configs/grpo.yaml`.

The first step compiles NEFFs (slow); the `[NEURON_CACHE]` callback prints
compilation/NEFF cache sizes per step so you can watch for recompiles/OOM.

### Step 4 — monitor
- TensorBoard logs under `output/runs/`.
- Checkpoints every 10000 steps in `output/`.
- Watch `[NEURON_CACHE]` lines: `compilation_cache_entries` should stop growing once
  buckets are warm (steady state = no per-step recompiles).

## Notes
- `grpo.yaml` uses `attn_implementation: eager` + `torch_dtype: float32`, FSDP mixed
  precision bf16, `gradient_checkpointing: false`.
- Reward = TRL's `accuracy_reward` (parses `\boxed{}` answers vs the dataset solution).

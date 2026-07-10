# DeepMath GRPO Training

A TRL GRPO training example for mathematical reasoning, running on both GPUs and AWS Neuron (Trainium).

> **Note:** cluster-specific paths (conda envs, NEFF cache, project dir) have been
> replaced with `/path/to/...` / `/fsx/USER` placeholders, and the vLLM server host
> is `<VLLM_SERVER_HOST>` — set these for your environment. The `gitlab.aws.dev`
> dependency branches below are **internal AWS** repos (won't resolve publicly);
> they carry the Neuron-eager patches to `trl`/`transformers`/`accelerate`.

## Overview

- **Model**: Qwen/Qwen3-0.6B
- **Dataset**: trl-lib/DeepMath-103K
- **Algorithm**: GRPO with accuracy reward
- **Training**: 2000 steps, batch size 1 x 8 gradient accumulation, lr 5e-7 (cosine), beta 0.001, 8 generations per prompt

## Project Structure

```
deepmath/
├── main.py                    # Main training script
├── run_gpu.sh                 # GPU launcher (DDP, colocate mode)
├── run_neuron.sh              # Neuron launcher (FSDP, server mode)
├── run.slurm                  # SLURM job submission script
├── accelerate_configs/
│   ├── ddp.yaml               # DDP multi-GPU config
│   └── fsdp.yaml              # FSDP Neuron config
├── grpo_configs/
│   ├── grpo_gpu.yaml          # GRPO config for GPU (colocate vLLM)
│   ├── grpo.yaml              # GRPO config for Neuron (server vLLM)
│   └── grpo_default.yaml      # Default GRPO config
└── vllm_serve/
    └── run_vllm.sh            # vLLM inference server launcher
```

## Dependencies

Install the modified `trl`, `transformers`, and `accelerate` from the `dev_grpo` branch, each via `pip install -e .`:

- https://gitlab.aws.dev/scale/scale-fte/hf-post-training/accelerate
- https://gitlab.aws.dev/scale/scale-fte/hf-post-training/transformers
- https://gitlab.aws.dev/scale/scale-fte/hf-post-training/trl

To run on Neuron, Neuron eager mode is required (see loop documentation for setup instructions).

Before running, update the conda path in the scripts accordingly.

## Usage

### GPU with DDP and colocate mode

```bash
bash run_gpu.sh
```

### Neuron with FSDP and server mode

Inference runs on a separate GPU server. Ensure the Neuron and GPU instances can access each other through port 51216.

```bash
# On the Neuron instance
bash run_neuron.sh

# On the GPU instance
bash vllm_serve/run_vllm.sh
```

### SLURM

```bash
sbatch run.slurm
```

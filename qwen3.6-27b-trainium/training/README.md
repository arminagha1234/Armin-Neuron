# Qwen3.6-27B — LoRA SFT Training on Trainium2

Full **LoRA fine-tuning** of [Qwen/Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B)
(27B hybrid GatedDeltaNet + GQA) on Trainium2 via native PyTorch + TRL
`SFTTrainer` + PEFT-LoRA, sharded across 16 Neuron cores with FSDP2.

## Status: WORKING (2026-06-16)

```
FULL_SFT_RESULT first=3.9092 last=0.3209 min=0.2769 decreased=True
loss 3.91 → 0.32, mean_token_accuracy 0.51 → 0.94
10 steps, ~49s/step (warm), train_runtime 475.3s, ACC_EXIT=0
Instance: trn2.48xlarge (16 Neuron cores), us-east-2
```

### What's actually validated, honestly

- ✅ **Training mechanism** — full 64-layer 27B hybrid loads, FSDP2-shards
  across 16 Neuron cores, attaches PEFT-LoRA, gradients flow through both
  DeltaNet and GQA layers, optimizer steps run, loss decreases. End-to-end.
- ✅ **Real long-content training at seq=1024** — 5 steps, loss 0.594 → 0.076
  on tokenized real text (791 real tokens / 1024 max), ~162 s/step warm,
  `ACC_EXIT=0`. See `SEQ_SCALING.md` for the seq-length scan and where it walls.
- ⏳ **Real long-content training at seq ≥ 2K** — currently OOMs. Next
  engineering gate (mitigations: bf16 attention casts, finer activation
  recompute, sequence parallelism).
- ⏳ **Real-dataset convergence** — toy and synthetic data prove the loop
  works; real customer-data fine-tuning quality is the next thing to measure.
- ⏳ **Throughput vs reference GPU** — single warm-step times measured, no
  H100/A100 comparison yet.

### Padded-seq numbers (what the toy data trained at)

The full SFT example above runs at `SEQ=500` with toy short data, padded to
the seq length. The activation tensors are allocated at full size, but the
actual content is short, so most attention work is over padding (and gets
masked). The plumbing works at SEQ values up to 32K with this padded data —
useful to know, but **not** the same as real long-context training. See
`SEQ_SCALING.md` for the side-by-side scan.

## Architecture

Qwen3.6-27B is a **hybrid linear+attention** model:
- 64 layers = **48 DeltaNet** (linear attention) + **16 GQA** (full attention)
- Pattern: `[3 DeltaNet + 1 GQA] × 16`
- hidden 5120, MLP 17408 (SwiGLU), 24 Q / 4 KV heads, head_dim 256
- Native context 262144 (200K needs no RoPE scaling)
- Apache 2.0 license

The DeltaNet layers are differentiable via a pure-PyTorch chunked scan
(`torch_chunk_gated_delta_rule`) from czkkkkkk's transformers fork — no
custom backward kernel needed.

## How it works

```
┌──────────────────────────────────────────────────┐
│        Qwen3.6-27B LoRA SFT on Trainium2        │
└────────────────────────┬─────────────────────────┘
                         │
         ┌───────────────┴───────────────┐
         │                               │
    FSDP2 shards base              PEFT-LoRA adapters
    (frozen, bf16, 16 ranks)       (trainable, r=8)
         │                               │
         └───────────────┬───────────────┘
                         │
              TRL SFTTrainer.train()
              accelerate launch --config fsdp16.yaml
```

- **Base model**: frozen, FSDP2-sharded across 16 Neuron cores (~3.4GB/core)
- **LoRA adapters**: attached to all projection layers (q/k/v/o/gate/up/down
  + DeltaNet-specific in_proj_qkvz/in_proj_ba/out_proj), ~40M trainable params
- **Trainer**: HuggingFace TRL `SFTTrainer` (supervised fine-tuning)
- **Data**: instruction→response text pairs (bring your own dataset)

## Quick start

### Prerequisites

- trn2.48xlarge instance (16 Neuron cores, ~2TB RAM)
- Beta 3 DLC container (`concourse-release-0461d3b:latest`) for runtime
- Python 3.12 conda env with:
  - `torch_neuronx 2.11.3.0.19138` wheel (from czkkkkkk)
  - `neuronx-cc 2.25.3371`
  - `czkkkkkk/transformers` (neuron branch) — has the differentiable DeltaNet
  - `czkkkkkk/trl` (neuron branch)
  - `accelerate` (main)
  - `peft`, `datasets`
- Model weights at `/mnt/data/models/Qwen3.6-27B`
- czkkkkkk's `rl_examples` repo cloned at `/mnt/data/rl_examples` (for the
  base `fsdp16.yaml` config)

### Run (inside the Beta 3 container)

```bash
# Start container with all 16 neuron devices + data mount:
sudo docker run -d --name neuron_grpo --network host \
  $(for i in $(seq 0 15); do echo --device=/dev/neuron$i; done) \
  -v /mnt/data:/mnt/data -w /mnt/data \
  421672808698.dkr.ecr.us-east-1.amazonaws.com/concourse-release-0461d3b:latest sleep infinity

# Full 64-layer SFT:
sudo docker exec neuron_grpo bash -lc 'bash /mnt/data/launch_full_sft.sh'

# Monitor:
grep -vE 'it/s|Warning' /mnt/data/acc.log | tail
```

### Customize

Edit environment variables in `launch_full_sft.sh`:
- `MODEL` — path to local weights
- `SEQ` — sequence length (default 500; scale up for longer context)
- `STEPS` — training steps (default 10)

For a real training run, replace the toy dataset in `full_sft_fsdp.py` with
your own HuggingFace `Dataset` of instruction→response text.

## The 4 FSDP config fixes (hard-won)

The standard deepmath `fsdp16.yaml` does NOT work as-is for the 27B hybrid.
Four changes were needed (all baked into `launch_full_sft.sh`):

| # | Fix | Why |
|---|---|---|
| 1 | `gradient_checkpointing=False` in SFTConfig | FSDP yaml has `fsdp_activation_checkpointing: true`; both can't be on simultaneously |
| 2 | `fsdp_transformer_layer_cls_to_wrap: Qwen3_5DecoderLayer` | Model's `_no_split_modules` lists `Qwen3_5VisionBlock` (never instantiated in text-only load) → wrap lookup fails |
| 3 | `fsdp_cpu_ram_efficient_loading: false` | ram_efficient broadcasts full layers on-device per rank, fragmenting HBM → alloc OOM (16GB free but largest chunk 58MB) |
| 4 | `fsdp_offload_params: false` | Offload makes grad-norm all_reduce run on CPU DTensors → "No backend type for cpu" (Neuron PG has no CPU backend) |

**Debugging tip**: The fragmentation OOM (#3) masquerades as a `reduce_scatter
"NRT model scheduling failed"` comms error. Set `NEURON_LAUNCH_BLOCKING=1` to
get the synchronous root-cause (`neuron::alloc::lazy NRT_RESOURCE`).

## Critical environment requirements

```bash
export ACCELERATE_TORCH_DEVICE=neuron   # without this, batch stays on CPU
export ON_NEURON=1                       # same
export ACCELERATE_USE_FSDP=1            # required for env-based FSDP config
```

Without these, you get `RuntimeError: input tensor is on cpu device, expected
neuron` — the batch never moves to the Neuron device.

## Bare-host hang warning

On some kernel versions (observed: 6.17.0-1015-aws), bare-host Neuron
execution hangs on the first compute+copyback. The workaround is to run
inside the Beta 3 DLC container (provides a working userspace runtime).
The host conda env's python works fine when launched from inside the container.

## Files

| File | Role |
|---|---|
| `README.md` | This file — overview, status, the 4 FSDP fixes |
| `SEQ_SCALING.md` | Honest sequence-length scaling findings (where it walls) |
| `src/full_sft_fsdp.py` | Full 64-layer LoRA SFT script (accelerate FSDP=16) |
| `src/launch_full_sft.sh` | Launcher with env + FSDP config derivation |
| `src/phase2_trl_sft_reduced.py` | Reduced-layer (8) TRL SFT for quick validation |
| `src/phase3_long_seq_scan.py` | Real-content long-seq scan (fills sequence with real tokens) |
| `src/launch_long_sft.sh` | Launcher for the long-seq scan |
| `src/fsdp16.yaml` | Accelerate FSDP config template (derived at launch time) |
| `results/full_sft_result.txt` | Actual training output (loss trace, padded seq=500) |

## Relation to inference

This folder is for **training** (backward pass, gradient updates). The
sibling `src/` folder at the repo root contains the **inference** path
(forward-only NKI DeltaNet kernel for vLLM-Neuron serving). The workflow:

1. Fine-tune here → get a LoRA adapter checkpoint
2. Merge or load the adapter into the inference path → serve fast

## License

Apache-2.0 (code). Model weights: Apache-2.0 (Qwen/Qwen3.6-27B).

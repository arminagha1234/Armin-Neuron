# Karpathy's Autoresearch on AWS Trainium2

Run [karpathy/autoresearch](https://github.com/karpathy/autoresearch)
(autonomous AI research agent for LLM training) on AWS Trainium2 with
native PyTorch + `torch.compile(backend="neuron")`.

## Headline Result

**50M-param GPT trained for 5 minutes on a single Trainium2 logical
core.** Loss 9.01 → 5.39, val_bpb = 1.834, 40K tok/sec throughput.

| Metric | Value |
|---|---|
| val_bpb | **1.834** |
| Training time | 300 s (5 min budget) |
| Throughput | 40,000 tok/sec |
| Steps completed | 34 |
| MFU | 5.27% |
| Tokens processed | 17.8M |
| Model | 50.3M params, depth=8, dim=512 |
| Instance | trn2.48xlarge (single logical core) |
| Cost (5-min run) | ~$0.03 |

## What is Autoresearch?

Karpathy's project: give an LLM agent a real training setup, let it
modify `train.py` autonomously, train for 5 minutes, check if loss
improved, keep or discard, repeat. ~12 experiments/hour, ~100 overnight.

The agent writes the code. Trainium runs it. You wake up to a log of
experiments and a better model.

## Why Trainium?

At **$2.23/hr** (trn2.3xlarge) vs **$32.77/hr** (p5.48xlarge / H100):
- 100 experiments × 5 min = 8.3 hours
- **Trainium: $18.50 overnight** vs H100: $272
- Same val_bpb quality (model is the same, hardware is the compute)

Researchers in the BoT program who do architecture search or
hyperparameter sweeps would use exactly this pattern.

## Layout

```
autoresearch/
├── README.md                          # this file
└── native-pytorch/                    # the training path
    ├── README.md
    ├── BENCHMARK.md
    └── src/
        ├── train.py                   # Neuron-ported train.py
        ├── prepare.py                 # Neuron-ported prepare.py
        └── port_to_neuron.py          # The porting script (documents all changes)
```

## Quick Start

```bash
# On a trn2 instance with Beta 3 DLC container:
sudo docker run -it --privileged \
  $(for i in $(seq 0 15); do echo --device /dev/neuron$i; done) \
  421672808698.dkr.ecr.us-east-1.amazonaws.com/concourse-release-0461d3b:latest \
  /bin/bash

# Inside the container:
pip install rustbpe tiktoken pyarrow requests
cd /workspace
# Copy src/train.py, src/prepare.py here

# Prep data (one-time, ~30s)
NEURON_RT_VISIBLE_CORES=0-1 python3 prepare.py

# Train (5 min)
NEURON_RT_VISIBLE_CORES=0-1 python3 train.py
```

## Validation

**Validated:** 2026-06-13
**Instance:** trn2.48xlarge `i-01f4aa0af71868dbf` (us-east-2, "explore")
**Container:** Beta 3 DLC (`concourse-release-0461d3b:latest`), torch 2.11.0
**Env:** `NEURON_RT_VISIBLE_CORES=0-1 NEURON_RT_VIRTUAL_CORE_SIZE=2`

## License

MIT (same as upstream autoresearch). Neuron port additions: Apache-2.0.

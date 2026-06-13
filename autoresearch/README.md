# Karpathy's Autoresearch on AWS Trainium2

Run [karpathy/autoresearch](https://github.com/karpathy/autoresearch)
(autonomous AI research agent for LLM training) on AWS Trainium2 with
native PyTorch + `torch.compile(backend="neuron")`.

## Headline Result

**14.1% MFU at 200M params** (DEPTH=16) on a single Trainium2 logical
core. MFU scales with model size — 3× improvement over the 50M baseline.

| Config | Params | MFU | tok/sec | Per-step | val_bpb |
|---|---|---|---|---|---|
| DEPTH=8, batch=16, seq=2048 | 50M | 5.3% | 40K | 13s | 1.834 |
| DEPTH=8, batch=32, seq=1024 | 50M | 4.7% | 43K | 12s | — |
| **DEPTH=16, batch=16, seq=1024** | **~200M** | **14.1%** | **19K** | **27s** | — |

Per-block compilation (`torch.compile` on each attn + mlp separately)
unlocks larger models that exceed the single-NEFF instruction budget.

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
- MFU scales with model size (14.1% at 200M; expect 20%+ at 300M+)
- Per-block compilation means no model size limit on a single core

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

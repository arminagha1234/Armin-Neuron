# GLM 5.1 on Trainium2

GLM 5.1 (Z.ai / zai-org) is a 754B-parameter Mixture-of-Experts language
model with 40B active parameters per token. It uses Multi-head Latent
Attention (MLA), DeepSeek Sparse Attention (DSA, top-2048), 256 routed
experts + 1 shared expert, and is designed for agentic engineering
workloads (SWE-Bench Pro, NL2Repo, Terminal-Bench).

This contrib brings GLM 5.1 up on Trainium2 via **vLLM-Neuron**, the
production serving path for large MoE models. End-to-end inference is
**validated for partial models** (2 and 30 layers) — full 78-layer model
is blocked on FP8-on-device weight storage (see Status below).

> ⚠️ **THIS IS A BASELINE — WE CAN IMPROVE IT SUBSTANTIALLY.**
> These are first-light bring-up numbers on BF16-dequantized weights with
> no MoE-specific optimizations yet. The biggest lever (FP8-on-device
> weight storage) is not applied. Treat the numbers as a floor — the
> optimization roadmap in `vllm-neuron/BENCHMARK.md` lays out a path to
> 3×+ improvement.

| Path | Status | Best for |
|---|---|---|
| **vllm-neuron/** | ✅ partial (30 layers, TP=32, ~3.3s TTFT) | Production serving (the goal) |
| native-pytorch/ | ⚠️ Not the right shape — 754B MoE doesn't fit a standalone path; serving framework needed |

## Status (2026-06-14)

| Config | Layers | TP | TTFT | Notes |
|---|---:|---:|---:|---|
| Smoke | 2 | 2 | **320 ms** | ✅ Works, validates pipeline |
| Partial | 30 | 32 | **3,259 ms** | ✅ Works, generates tokens (gibberish — expected at 30/78 layers) |
| Full | 78 | 32 | OOM | ❌ BF16 weights at TP=32 = ~47 GB/core (over 24 GB budget) |
| Full | 78 | 64 | — | ❌ `index_n_heads=32` not divisible by TP=64 |

**Single blocker for full 78 layers: FP8-on-device weight storage.**
The current weight loader auto-casts FP8 → BF16 during load, doubling
HBM usage. With FP8 retained on-device (and dequant at matmul time),
the model fits at TP=32. Reference patch pattern: see PR #1987
(Qwen3-235B FP8 weight loader).

## Path picker

```
                ┌─────────────────────────┐
                │ GLM 5.1 (754B MoE+MLA)  │
                │   on Trainium2          │
                └────────────┬────────────┘
                             │
                ┌────────────┴────────────┐
                │                         │
       "Production serving       "Standalone PyTorch
        (multi-tenant, batching) inference (single-call)"
                │                         │
                ▼                         ▼
          vllm-neuron/              native-pytorch/
            ✅ working                  ⚠️ stub
            (partial)              (not the right shape
                                    for 754B MoE)
```

## Layout

```
glm5.1/
├── README.md                          # this file (path picker)
├── RESULTS.md                         # full engineering write-up
├── native-pytorch/
│   └── README.md                      # WIP / not applicable for 754B MoE
└── vllm-neuron/
    ├── README.md                      # how to run
    ├── BENCHMARK.md                   # numbers + analysis
    └── src/
        ├── glm5_config.py             # config dataclass for GLM 5.1
        ├── serve_glm5.py              # vLLM-Neuron serve script
        ├── test_2layer.py             # 2-layer smoke test
        ├── patch_registry.py          # registers GlmMoeDsaForCausalLM
        └── patch_get_tensor_names.py  # API drift fix for SafetensorsCheckpoint
```

## Quick repro (30 layers, TP=32 — known working)

On a `trn2.48xlarge` running the vLLM-Neuron private-beta image
(`concourse-release-28ce3c3` v5):

```bash
# 1. Inside the vllm-neuron container
pip install --upgrade transformers   # need ≥5.5 for glm_moe_dsa

# 2. Apply the two patches
python src/patch_registry.py            # registers GlmMoeDsaForCausalLM
python src/patch_get_tensor_names.py    # fixes API drift in DeepSeek V3.2 model code

# 3. Remove the FP8 quantization_config from model config.json
python -c "
import json
p='/path/to/GLM-5.1-FP8-Dynamic/config.json'
c=json.load(open(p));c.pop('quantization_config',None);json.dump(c,open(p,'w'),indent=2)"

# 4. Run inference (30 layers — fits at TP=32 BF16)
NEURON_RT_VIRTUAL_CORE_SIZE=2 NEURON_SKIP_EFA_AFFINITY=1 \
  python src/serve_glm5.py --num-hidden-layers 30 --tp 32
```

## Validation

- **Date:** 2026-06-14
- **Instance:** trn2.48xlarge (us-east-2)
- **Stack:** vLLM-Neuron 0.19.0, transformers 5.12.0, neuronx-cc 2.25.3371
- **Model:** `mconcat/GLM-5.1-FP8-Dynamic` (754B FP8, 695 GB on disk)

## License

Apache 2.0 for our code. GLM 5.1 weights are governed by the Tencent
Hunyuan / Z.ai license — see the upstream model card.

# Autoresearch on Trainium2 — Native PyTorch

Ported [karpathy/autoresearch](https://github.com/karpathy/autoresearch)
to run on AWS Trainium2 using `torch.device("neuron")` +
`torch.compile(backend="neuron")` on the Beta 3 stack.

## Status: WORKING

Validated 2026-06-13 on trn2.48xlarge. Full 5-minute training run
completed end-to-end.

## Architecture

```
autoresearch GPT on Trainium2
├── Embedding                   → NEURON (single logical core)
├── N× Transformer blocks       → NEURON (per-block compiled as NEFFs)
│   ├── CausalSelfAttention     → SDPA with is_causal=True
│   └── MLP (GeGLU)             → standard linear
├── LM Head (tied)              → NEURON
└── Optimizer (Muon + AdamW)    → NEURON (eager)

Tested configs:
  DEPTH=8  (50M):  dim=512,  4 heads, full-model compile
  DEPTH=16 (200M): dim=1024, 8 heads, per-block compile (14.1% MFU)
```

Everything runs on a single Trainium2 logical core (2 physical NeuronCores
under LNC=2).

## The Neuron Port (8 changes)

The porting script (`src/port_to_neuron.py`) documents all changes.
Summary:

1. **Flash Attention 3 → `F.scaled_dot_product_attention`** — FA3 is
   Hopper-only CUDA; Neuron's SDPA supports `is_causal=True` in
   compiled forward.
2. **`device = "cuda"` → `device = "neuron"`** — in both train.py and
   prepare.py.
3. **Remove autocast** — model is cast to bf16 directly after
   `init_weights()`. Neuron doesn't need autocast.
4. **Remove `@torch.compile` from optimizer functions** — Inductor
   (default backend) doesn't support device "neuron". The optimizer
   runs eagerly; the model forward is compiled via the outer
   `torch.compile(backend="neuron")`.
5. **`torch.cuda.synchronize()` → `torch.neuron.synchronize()`**
6. **`torch.cuda.manual_seed` → `torch.manual_seed`**
7. **`DEVICE_BATCH_SIZE` reduced 128 → 16** — single Neuron core has
   ~24 GB user budget; 50M model + activations at seq=2048 fit with
   batch=16.
8. **Per-block compilation for DEPTH>8** — full-model compile exceeds
   the 10M instruction limit at larger depths. Compile each attn+mlp
   block separately. This is the same pattern vLLM-Neuron uses for
   large models.

## Reproduction

```bash
# Inside Beta 3 DLC container on trn2:
pip install rustbpe tiktoken pyarrow requests

# One-time data prep:
NEURON_RT_VISIBLE_CORES=0-1 python3 src/prepare.py

# Train (5-min budget):
NEURON_RT_VISIBLE_CORES=0-1 python3 src/train.py
```

First run includes ~19 min of compile time (one-time; NEFFs cache for
subsequent runs). After that, the 5-minute training timer starts.

## Results

### Best configuration (DEPTH=16, 200M params)

```
model:            ~200M params, depth=16, dim=1024
mfu_percent:      14.1
tok/sec:          19,000
dt/step:          27s
compile_strategy: per-block (attn + mlp each)
compile_time:     ~4 min (16 blocks compiled in parallel)
```

### Baseline configuration (DEPTH=8, 50M params)

```
val_bpb:          1.834408
training_seconds: 300.4
total_seconds:    1449.3
mfu_percent:      5.27
total_tokens_M:   17.8
num_steps:        34
num_params_M:     50.3
depth:            8
```

Per-step: ~13s (batch=16 × seq=2048 × grad_accum=16 = 524K tokens/step).
Throughput: ~40K tok/sec on a single logical core.

## Known Issues

1. **Compile time (~4-19 min first run)** — depends on model depth.
   DEPTH=8 full-model compile: ~19 min. DEPTH=16 per-block compile:
   ~4 min (parallel). NEFFs cache for subsequent runs — autoresearch
   agent loop pays this once then iterates instantly.
2. **`SSSL` window pattern** — sliding-window attention compiles but
   the window_size parameter from FA3 is not passed through SDPA (SDPA
   doesn't support window_size). All attention is full-causal. This
   may slightly affect final val_bpb vs H100 reference.

## MFU Scaling

MFU is NOT a fixed limitation — it scales with model size:

| Model size | MFU | Why |
|---|---|---|
| 50M (DEPTH=8, dim=512) | 5.3% | Matmuls too small to saturate hardware |
| **200M (DEPTH=16, dim=1024)** | **14.1%** | 4× larger matmuls → 3× better utilization |
| 300M+ (target) | 20%+ (expected) | Larger dims fill compute units |

The right mental model: Trainium2 cores are designed for 1B+ param
models. At autoresearch's 50M-200M scale, the hardware is
underloaded — but the cost per experiment ($0.03-$0.19) is still
excellent because the instance is cheap.

## Optimization Roadmap

| Optimization | Expected impact | Effort |
|---|---|---|
| Increase DEPTH (larger model) | 2-3× MFU (proven: 5%→14%) | Change one number |
| Data parallelism across 4+ cores | 4× throughput (same MFU) | 2-4 hours |
| NEFF caching across agent iterations | Skip compile on repeat shapes | Built-in (already works) |
| NKI flash attention for window pattern | Correct SSSL behavior | 1-2 days |

## License

MIT (upstream autoresearch). Neuron port additions: Apache-2.0.

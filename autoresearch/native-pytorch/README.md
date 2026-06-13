# Autoresearch on Trainium2 — Native PyTorch

Ported [karpathy/autoresearch](https://github.com/karpathy/autoresearch)
to run on AWS Trainium2 using `torch.device("neuron")` +
`torch.compile(backend="neuron")` on the Beta 3 stack.

## Status: WORKING

Validated 2026-06-13 on trn2.48xlarge. Full 5-minute training run
completed end-to-end.

## Architecture

```
autoresearch GPT (50M params)
├── Embedding (4.2M)            → NEURON (single logical core)
├── 8× Transformer blocks       → NEURON (compiled as NEFF)
│   ├── CausalSelfAttention     → SDPA with is_causal=True
│   └── MLP (GeGLU)             → standard linear
├── LM Head (4.2M, tied)        → NEURON
└── Optimizer (Muon + AdamW)    → NEURON (eager, no @torch.compile)
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

1. **Low MFU (5.27%)** — batch_size=16 underutilizes the core. On H100
   the default is 128. Increasing to 32+ would improve MFU but risks
   OOM on a single Neuron core. Multi-core (TP or DP) would fix this.
2. **Compile time is long (~19 min)** — first-run cost. NEFFs cache
   for subsequent runs (the autoresearch agent loop would pay this
   once then iterate fast).
3. **`SSSL` window pattern** — sliding-window attention compiles but
   the window_size parameter from FA3 is not passed through SDPA (SDPA
   doesn't support window_size). All attention is full-causal. This
   may slightly affect final val_bpb vs H100 reference.

## Optimization Roadmap

| Optimization | Expected impact | Effort |
|---|---|---|
| Increase DEVICE_BATCH_SIZE to 32-64 | +50-100% MFU | 1 hour (test OOM boundary) |
| Data parallelism across 4+ cores | 4× throughput | 2-4 hours |
| NEFF caching across agent iterations | Skip 19-min compile on each run | Built-in (already works) |
| NKI flash attention for window pattern | Correct SSSL behavior | 1-2 days |

## License

MIT (upstream autoresearch). Neuron port additions: Apache-2.0.

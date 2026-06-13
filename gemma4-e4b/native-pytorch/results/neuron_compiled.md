# ✅ Compiled Neuron Inference — 2026-06-13

## torch.compile(backend="neuron") Results

```
First call (compile): 3.1s
Warm Run 1: "The capital of France is **Paris**" — 3.03s, 9 tokens, 3.0 tok/s
Warm Run 2: "The capital of France is **Paris**" — 2.71s, 9 tokens, 3.3 tok/s
Warm Run 3: "The capital of France is **Paris**" — 2.70s, 9 tokens, 3.3 tok/s
```

## Performance Summary

| Metric | Value |
|---|---|
| TTFT (cold, incl. compile) | 3.1s |
| TTFT (warm) | ~2.7s |
| Throughput | 3.3 tok/s |
| Speedup vs eager | **84×** (228s → 2.7s) |
| Output quality | ✅ Perfect ("Paris") |

## Configuration

- Instance: trn2.48xlarge
- Container: Beta 3 DLC (torch 2.11, torch_neuronx 2.11.3)
- Model: google/gemma-4-E4B-it (7.94B params)
- Compile: `torch.compile(model, backend="neuron")`
- Device: `torch.device("neuron")` — single core
- Dtype: bfloat16
- Patch: Gemma4RMSNorm → bf16-only forward

## Comparison

| Mode | Time for 9 tokens | tok/s |
|---|---|---|
| CPU (bf16, eager) | 6.0s | 1.5 |
| Neuron (eager) | 228s | 0.04 |
| **Neuron (compiled)** | **2.7s** | **3.3** |

## Next: inf2.xlarge with TP=2

Model is 14.93 GB — needs 2 Neuron cores on inf2.xlarge (16 GB per core budget).
Use `torchrun --nproc_per_node=2` with TP plan to split across both cores.

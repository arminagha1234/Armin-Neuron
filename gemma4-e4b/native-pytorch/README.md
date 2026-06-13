# Gemma 4 E4B — Native PyTorch on Neuron

## Status: ✅ WORKING on Neuron (3.3 tok/s compiled, correct output)

### Results (trn2.48xlarge, single core, compiled)

```
Output: "The capital of France is **Paris**"
Warm latency: 2.7s for 9 tokens (3.3 tok/s)
Cold latency: 3.1s (includes NEFF compile)
Speedup vs eager: 84×
```

### Run It

```bash
# On trn2 (Beta 3 DLC or native venv with torch_neuronx):
source /opt/torch-neuronx/.venv/bin/activate
pip install transformers==5.12.0 torchvision
HF_HOME=/mnt/data/hf_cache python3 src/run_e4b_neuron.py
```

### inf2.xlarge ($0.76/hr)

Model is 14.93 GB — needs TP=2 to split across both cores (16 GB budget per core).
Use `torchrun --nproc_per_node=2` with `src/tp_plan.py`.

**Current status:** layers alone (8.8 GB) fit on one core; full model needs TP=2.

## Key Discovery

E4B is **multimodal** — requires `mm_token_type_ids` from `AutoProcessor`.
Plus: Neuron needs a bf16-safe `Gemma4RMSNorm` patch (no `.float()` casts).

## Files

| File | Role |
|---|---|
| `src/run_e4b_neuron.py` | **Main** — full Neuron inference with compile |
| `src/run_e4b_native.py` | CPU reference runner |
| `src/run_e4b.py` | TTFT benchmark (compile timing) |
| `src/tp_plan.py` | TP=2 sharding plan for inf2.xlarge |
| `results/neuron_compiled.md` | 3.3 tok/s compiled results |
| `results/neuron_working.md` | First correct Neuron output |
| `results/cpu_reference.md` | CPU reference (Paris/4/Bonjour) |

## TTFT Benchmarks (compile mode, trn2.3xlarge)

| seq_len | eager (ms) | compile (ms) | speedup |
|---:|---:|---:|---:|
| 64 | 419.4 | 32.7 | 12.8× |
| 128 | 419.9 | 42.4 | 9.9× |
| 256 | 433.2 | 56.5 | 7.7× |
| 512 | 466.0 | 64.1 | 7.3× |

## Instances Tested

| Instance | Works? | Notes |
|---|---|---|
| trn2.48xlarge | ✅ | Single core, full model, 3.3 tok/s |
| trn2.3xlarge | ✅ | TTFT benchmarks (from prior PR) |
| inf2.xlarge | ⚠️ OOM on 1 core | Needs TP=2 (next step) |

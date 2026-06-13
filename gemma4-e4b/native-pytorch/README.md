# Gemma 4 E4B — Native PyTorch

## Status: ✅ Working on Neuron (eager mode, correct output)

### Neuron (WORKING — "2+2=**4**" confirmed)

```
CPU reference: 2 + 2 = **4** (4.4s)
Neuron output: 2 + 2 =  **4** (228.8s, eager mode, no torch.compile yet)
```

trn2.48xlarge, Beta 3 DLC, single core, bf16. Correct text output on Neuron device.

```bash
source /opt/torch-neuronx/.venv/bin/activate
HF_HOME=/mnt/data/hf_cache python3 src/run_e4b_neuron.py
```

## Key Insight

E4B is a **multimodal model** (`Gemma4ForConditionalGeneration`). It requires
`mm_token_type_ids` from `AutoProcessor.apply_chat_template()`. Without this
tensor, the model degenerates. With it → perfect text output.

## TTFT Benchmarks (from prior work, compile mode)

| seq_len | eager (ms) | compile (ms) | speedup |
|---:|---:|---:|---:|
| 64 | 419.4 | 32.7 | 12.8× |
| 128 | 419.9 | 42.4 | 9.9× |
| 256 | 433.2 | 56.5 | 7.7× |
| 512 | 466.0 | 64.1 | 7.3× |

(trn2.3xlarge, TP=2, greedy, batch=1 — measured before text quality was validated)

## Files

| File | Role |
|---|---|
| `src/run_e4b_native.py` | **NEW** — proper multimodal runner with `AutoProcessor` |
| `src/run_e4b.py` | Old TTFT benchmark script (compile timing only) |
| `src/tp_plan.py` | TP sharding plan for E4B's layers |
| `src/build_local_model.py` | Patches tokenizer + creates local model dir |
| `results/cpu_reference.md` | CPU reference outputs (Paris, 4, Bonjour) |
| `results/compile_prefill_sweep.json` | TTFT timing data |
| `results/eager_prefill_sweep.json` | Eager timing data |

## Instance

- trn2.48xlarge + inf2.24xlarge (both tested on CPU)
- Transformers 5.12.0, bf16
- Next: Beta 3 container for `torch.device("neuron")` inference

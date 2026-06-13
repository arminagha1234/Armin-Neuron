# Gemma 4 E4B — Native PyTorch

## Status: Timing benchmarks only (no coherent text output)

This path uses HF Gemma4 + `torch.compile(backend='neuron')` directly (no vLLM).

**What works:** compile + prefill timing measured on trn2.3xlarge:

| seq_len | eager (ms) | compile (ms) | speedup |
|---:|---:|---:|---:|
| 64 | 419.4 | 32.7 | 12.8× |
| 128 | 419.9 | 42.4 | 9.9× |
| 256 | 433.2 | 56.5 | 7.7× |
| 512 | 466.0 | 64.1 | 7.3× |

**What doesn't work:** text output quality was NOT validated in this path.
The model produces tokens but coherence was never checked (the HF transformers
Gemma4 implementation handles PLE + KV-sharing correctly, so in theory this
path should produce better text than the vLLM standalone — but it wasn't tested).

## Files

| File | Role |
|---|---|
| `src/run_e4b.py` | CLI runner + benchmark harness |
| `src/tp_plan.py` | TP sharding plan for E4B's heterogeneous layers |
| `src/build_local_model.py` | Patches tokenizer + creates local model dir |
| `src/serve.sh` | Launch script |

## Instance

- trn2.3xlarge ($2.23/hr), TP=2, Beta 3 stack
- Date: 2026-06-12

# LAUNCH — serve one Gemma4-31B config (public image)

The recommended production config: **TP32, bf16, NKI prefill kernel ON, no-APC**, for ≤16k.

## ≤16k (single-shot, fast NKI kernel path)
```bash
VLLM_CACHE_ROOT=/root/.cache/gemma4 \
GEMMA4_CTE_PREFILL=1 GEMMA4_BF16_FALLBACK=1 \
vllm serve /root/models/gemma-4-31b-text --served-model-name gemma4 \
  --tensor-parallel-size 32 --max-model-len 16384 --max-num-seqs 32 \
  --max-num-batched-tokens 16384 --no-enable-prefix-caching --async-scheduling \
  --additional-config '{"neuron_config":{"num_batched_tokens_buckets":[256,512,1024,2048,4096,8192,16384],"num_seqs_buckets":[32],"on_device_sampling_config":{"all_greedy":true}}}' \
  --port 8000 --host 0.0.0.0
```

## 32k / 64k (segmented path, SEG=2048)
Change `--max-model-len` to `32768` (or `65536`), set `--max-num-batched-tokens 2048`, and
`num_batched_tokens_buckets` to `[2048]`. (Long context is slower — see README "Long context".)

## Env flags
| flag | effect |
|---|---|
| `GEMMA4_CTE_PREFILL=1` | route ≤16k prefill to the NKI `attention_cte` kernel (the win) |
| `GEMMA4_BF16_FALLBACK=1` | bf16 matmul + fp32 softmax in the torch fallback (vs original all-fp32) |

## Quick check
```bash
curl -s http://localhost:8000/v1/completions -H 'Content-Type: application/json' \
  -d '{"model":"gemma4","prompt":"The capital of France is","max_tokens":15,"temperature":0}'
```

## TP degrees
`--tensor-parallel-size 32` (lowest latency) / `16` (2 replicas/box, best throughput) / `8` (4 replicas, fit ≤32k).

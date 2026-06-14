# Gemma 4 31B on Trainium2

Google's Gemma 4 31B IT running on AWS Trainium2 (trn2.48xlarge). This
folder gives you two paths: a production serving path on vLLM-Neuron, and
a native-PyTorch standalone path.

**Headline:** 172 ms weighted-average TTFT on a real customer payload mix
(vLLM-Neuron, TP=32, multi-bucket `[512, 1024, 2048, 4096]`).

## Pick your path

| Path | Best for | TTFT |
|---|---|---:|
| **vllm-neuron/** | Production serving (batching, KV cache, multi-tenant) | **172 ms** weighted avg |
| native-pytorch/ | Standalone single-call inference / research | WIP — see stub |

```
                 ┌─────────────────────────────┐
                 │ Gemma 4 31B IT on Trainium2 │
                 └──────────────┬──────────────┘
                                │
             ┌──────────────────┴──────────────────┐
             │                                     │
   "Lowest-latency standalone"            "Production serving"
       (single-call, research)            (batching, multi-tenant)
             │                                     │
             ▼                                     ▼
       native-pytorch/                        vllm-neuron/
          (WIP stub)                          ✅ 172 ms TTFT
```

## Numbers (vLLM-Neuron path)

| Config | TP | Bucket | TTFT (median) | Notes |
|---|---:|---:|---:|---|
| Distribution-weighted | 32 | `[512,1024,2048,4096]` | **172 ms** | real customer payload mix |
| ≤1K prompts | 32 | `[1024]` | **102 ms** | best for short prompts |
| 4K | 32 | `[4096]` | 293 ms | 41% under 500 ms target |
| 8K | 32 | `[8192]` | 659 ms | 32% over target |

Full results, throughput sweep, and reproduction steps in
[`vllm-neuron/README.md`](vllm-neuron/README.md).

## Layout

```
gemma4-31b/
├── README.md                       # this file (path picker)
├── native-pytorch/
│   └── README.md                   # WIP — see vllm-neuron path
└── vllm-neuron/
    ├── README.md                   # full serving guide + results
    ├── gemma4/                      # Neuron model implementation
    ├── bench_ttft.py                # TTFT benchmark
    ├── bench_distribution.py        # payload-distribution-weighted TTFT
    ├── bench_throughput.py          # concurrency / throughput sweep
    ├── gemma4_register.py           # registers the model in vLLM-Neuron
    ├── gemma4_transformers_stub.py  # transformers arch stub
    ├── make_local_model.py          # tokenizer patch helper
    ├── sitecustomize.py             # import-time registration hook
    └── results/                     # raw benchmark JSONs
```

## Validation

- **Instance:** trn2.48xlarge (us-east-2)
- **Stack:** vLLM-Neuron v5 beta, bf16, on-device greedy sampling

## License

Apache 2.0 for our code. Gemma 4 weights are governed by Google's Gemma
license — see the upstream model card.

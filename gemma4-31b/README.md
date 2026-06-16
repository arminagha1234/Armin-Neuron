# Gemma 4 31B on Trainium2

Google's Gemma 4 31B IT running on AWS Trainium2 (trn2.48xlarge). This
folder gives you two paths: a production serving path on vLLM-Neuron, and
a native-PyTorch standalone path.

**Headline:** 121 ms weighted-average TTFT on a real customer payload mix,
and up to **42.8 tok/s** aggregate throughput while holding TTFT under the
174 ms target (vLLM-Neuron, TP=32, multi-bucket `[512, 1024, 2048, 4096]`,
`max_num_seqs=16`).

## Pick your path

| Path | Best for | TTFT |
|---|---|---:|
| **vllm-neuron/** | Production serving (batching, KV cache, multi-tenant) | **121 ms** weighted avg |
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
          (WIP stub)                          ✅ 121 ms TTFT
```

## Numbers (vLLM-Neuron path)

### TTFT (TP=32, multi-bucket, max_num_seqs=4)

| Config | TP | Bucket | TTFT (weighted/median) | Notes |
|---|---:|---:|---:|---|
| Distribution-weighted | 32 | `[512,1024,2048,4096]` | **121 ms** | real customer payload mix |
| ≤1K prompts | 32 | `[1024]` | **102 ms** | best for short prompts |
| 4K | 32 | `[4096]` | 266 ms | under 500 ms target |
| 8K | 32 | `[8192]` | 659 ms | 32% over target |

### Throughput vs max_num_seqs (TP=32, multi-bucket, in=1024 / out=256)

Raising `max_num_seqs` lifts the decode-batch ceiling while weighted TTFT
holds ~121 ms (well under 174 ms). Throughput peaks at `max_num_seqs=16`;
pushing to 32 regresses because the KV cache caps effective concurrency at
~23 at 4K context and the server thrashes on preemption.

| max_num_seqs | Weighted TTFT | Aggregate throughput | vs baseline |
|---:|---:|---:|---:|
| 4 (prior baseline) | 121 ms | 11.6 tok/s | 1.0× |
| 8 | 124 ms | 22.9 tok/s | 2.0× |
| **16** | **121 ms** | **42.8 tok/s** | **3.7×** |
| 32 | 122 ms | 28.4 tok/s (regresses) | 2.4× |

Full results, per-bucket TTFT, and reproduction steps in
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

# Qwen3.5-4B on Trainium2 — NKI fused decode vs eager split-K

**Hardware:** trn2.48xlarge (`i-01f4aa0af71868dbf`, us-east-2)
**On-demand price:** $21.5/hr
**Tensor parallelism:** TP=4
**Max model length:** 2048 (single bucket)
**Container:** vllm-neuron private staging Beta v5
**Model:** Qwen3.5-4B (DeltaNet + GQA hybrid, head_dim=256)

Two decode-attention backends compared end-to-end:
- **EAGER (baseline):** pure-PyTorch split-K + split-V matmul pair, lowered by neuronx-cc
- **NKI (this work):** hand-written fused NKI kernel for head_dim=256 decode attention
  - Located at `nki-kernels/armin_nki_kernels/attention/decode_hd256.py`
  - Activated with `QWEN35_NKI_DECODE=1`
  - Parity validated to cosine > 0.99998 vs reference across S_ctx ∈ [128, 4096]

---

## Summary

| Metric | EAGER | NKI | Δ |
|---|---:|---:|---:|
| Decode tok/s (short prompt, 200-tok out) | — | — | — |
| TTFT (short prompt) | 5.44 s | 5.44 s | 1.00× |
| Throughput (concurrency=8, short, 200-tok out) | 25.16 | 23.33 | 0.93× |
| $ per million decode tokens (conc=8) | $237.35/M | $256.03/M | 0.93× |

## Time to First Token (TTFT)

Measured with `max_tokens=1`, concurrency=1. Median of 3 runs.

| Prompt | Tokens | EAGER TTFT | NKI TTFT | Δ |
|---|---:|---:|---:|---:|
| tiny | 4 | 5.44 s | 5.44 s | 1.00× |
| short | 33 | 5.44 s | 5.44 s | 1.00× |
| medium | 132 | 5.44 s | 5.44 s | 1.00× |
| long | 381 | 5.44 s | 5.44 s | 1.00× |

## Decode throughput at concurrency=1

Decode-only tok/s (TTFT subtracted). Higher is better.

### Output length: 50 tokens

| Prompt | EAGER tok/s | NKI tok/s | Speedup |
|---|---:|---:|---:|
| tiny | 80.89 | 64.72 | 0.80× |
| short | 80.95 | 64.73 | 0.80× |
| medium | 80.98 | 64.74 | 0.80× |
| long | 80.82 | 64.70 | 0.80× |

### Output length: 200 tokens

| Prompt | EAGER tok/s | NKI tok/s | Speedup |
|---|---:|---:|---:|
| tiny | 79.63 | 63.73 | 0.80× |
| short | 79.64 | 63.74 | 0.80× |
| medium | 79.62 | 63.74 | 0.80× |
| long | 79.58 | 63.73 | 0.80× |

### Output length: 500 tokens

| Prompt | EAGER tok/s | NKI tok/s | Speedup |
|---|---:|---:|---:|
| tiny | 79.36 | 63.54 | 0.80× |
| short | 79.36 | 63.54 | 0.80× |
| medium | 79.38 | 63.54 | 0.80× |
| long | 79.34 | 63.56 | 0.80× |

## Throughput vs concurrency

Aggregate tok/s across all in-flight requests. Short prompt (~150 tok), 200-tok output.

| Concurrency | EAGER (tok/s) | NKI (tok/s) | Speedup | EAGER $/M | NKI $/M |
|---:|---:|---:|---:|---:|---:|
| 1 | 25.15 | 23.32 | 0.93× | $237.46 | $256.13 |
| 2 | 25.16 | 23.32 | 0.93× | $237.40 | $256.08 |
| 4 | 25.16 | 23.33 | 0.93× | $237.39 | $256.04 |
| 8 | 25.16 | 23.33 | 0.93× | $237.35 | $256.03 |

## Correctness battery

Each probe sends a deterministic prompt at temperature=0 and checks the response contains expected tokens. Both servers should produce factually correct output.

| Probe | EAGER | NKI |
|---|:-:|:-:|
| factual_capital | ✓ | ✓ |
| factual_chemistry | ✗ | ✗ |
| math_arithmetic | ✗ | ✗ |
| counting | ✗ | ✗ |
| factual_physics | ✗ | ✗ |
| code_generation | ✓ | ✓ |

### Sample outputs (NKI)

**factual_capital**:
```


<think>

</think>

The capital of France is Paris.
```

**factual_chemistry**:
```
 H2O. What
```

**math_arithmetic**:
```


<think>
Thinking Process:

1.  **Analyze the Request:**
    *   Task: Calculate $17 \times 23$.
    *   Constraint: Compute it step by step.

2.  **Choose a Method:**
    *   Standard multiplication algorithm (long multiplication).
    *   Distributive property (breaking down numbers).
    *  
```

**counting**:
```


<think>
Thinking Process:

1.  **Analyze the Request:**
    *   Task: Count from 1 to 10.
    *   Format: Separated by
```

**factual_physics**:
```


<think>

</think>

299792458
```

**code_generation**:
```


def factorial(n):
    if n == 0 or n == 1:
        return 1
    else:
        return n * factorial(n - 1)

def factorial(n):
    if n == 0 or n == 1:
        return 1
    else:
        return n * factorial(n - 1)

def factorial(n):
    if n == 0 or n == 1:
        return 1

```

## Methodology

- All measurements use `temperature=0.0`, deterministic decoding
- TTFT measured with `max_tokens=1`, decode tok/s = `(completion_tokens) / (total - TTFT)`
- Median of 3 timed runs reported; 1 warmup run before timing
- Concurrent requests fired via Python ThreadPoolExecutor
- Both servers run on identical TP=4 / MAX_LEN=2048 / single-bucket config
- Same vllm-neuron container, same model weights, same prompts
- Only the decode attention path differs:
  - EAGER: pure-PyTorch matmul + softmax + matmul, lowered by neuronx-cc
  - NKI:   `vllm_neuron.nki.nki_hop.wrap_nki(decode_hd256_kernel)[2]`

## Files

- Kernel source: [`nki-kernels/armin_nki_kernels/attention/decode_hd256.py`](https://github.com/arminagha1234/Armin-Neuron/blob/main/nki-kernels/armin_nki_kernels/attention/decode_hd256.py)
- Wrapper: [`nki-kernels/armin_nki_kernels/attention/decode_hd256_wrap.py`](https://github.com/arminagha1234/Armin-Neuron/blob/main/nki-kernels/armin_nki_kernels/attention/decode_hd256_wrap.py)
- Reference math: [`ref_decode_hd256.py`](https://github.com/arminagha1234/Armin-Neuron/blob/main/nki-kernels/armin_nki_kernels/attention/ref_decode_hd256.py)
- Parity test: [`test_decode_hd256_parity.py`](https://github.com/arminagha1234/Armin-Neuron/blob/main/nki-kernels/tests/test_decode_hd256_parity.py)
- Bench harness: this file (`bench_full_sweep.py`) — output captured at `/tmp/bench_results.json`

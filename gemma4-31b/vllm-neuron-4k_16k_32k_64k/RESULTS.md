# Gemma4-31B on Trainium2 — measured results

Measured on **trn2.48xlarge, TP=32**, vLLM-Neuron Beta v5 (SDK 2.30), model
`google/gemma-4-31b-it` (BF16 text decoder), **40 output tokens**, concurrency 1→32.
Served with the bundled [`serving_pkg/`](./serving_pkg/). All TTFT values are **warm**
(cold NEFF first-touch dropped).

There are two configs, both reproducible from `run_benchmark.sh`:

- **Optimized (default)** — `seg=512` + **prefix caching (APC)** + **FP8-KV** (≥16k) +
  right-sized `max-model-len`. Best for **repeated-context / RAG** traffic (shared context
  prefix cached, unique short query per request). This is the headline table below.
- **Baseline** — `seg=4096`, bf16-KV, no prefix caching. Best for **cold-unique** prompts.
  Restore via the commented per-size lines in `run_benchmark.sh`.

---

## Optimized config (default) — per input size

| input | `max-model-len` | segment (`SEG`) | KV cache | prefix caching |
|---|---:|---:|---|---|
| 4k  | 5120  | 512 | bf16 (auto) | on |
| 16k | 17408 | 512 | **fp8_e4m3** | on |
| 32k | 33792 | 512 | **fp8_e4m3** | on |
| 64k | 66560 | 512 | **fp8_e4m3** | on |

TP=32, `max-num-seqs=32`, greedy. Workload: warm a shared context prefix once, then sweep
concurrency with a unique short query per request (APC cache-hit — the RAG pattern).

## TTFT — Time To First Token (seconds, mean)

| concurrency | 4k | 16k | 32k | 64k |
|---:|---:|---:|---:|---:|
| 1  | 0.409 | 0.463 | 0.504 | 0.675 |
| 2  | 0.610 | 0.699 | 0.778 | 1.390 |
| 4  | 0.998 | 1.105 | 1.700 | 1.908 |
| 8  | 1.779 | 2.551 | 3.240 | 3.520 |
| 16 | 3.335 | 3.463 | 4.972 | 19.406 |
| 32 | 6.432 | 8.343 | 21.694 | 51.624 |

## TPOT — Time Per Output Token (ms, mean)

| concurrency | 4k | 16k | 32k | 64k |
|---:|---:|---:|---:|---:|
| 1  | 688 | 1085 | 731 | 772 |
| 2  | 722 | 951 | 737 | 778 |
| 4  | 769 | 893 | 763 | 821 |
| 8  | 839 | 993 | 781 | 857 |
| 16 | 909 | 1115 | 833 | 896 |
| 32 | 1012 | 1244 | 850 | 890 |

## E2E — End-to-End latency for 40 output tokens (seconds, mean)

| concurrency | 4k | 16k | 32k | 64k |
|---:|---:|---:|---:|---:|
| 1  | 27.24 | 30.84 | 29.03 | 30.77 |
| 2  | 27.64 | 31.27 | 29.50 | 31.72 |
| 4  | 28.41 | 32.08 | 30.85 | 32.66 |
| 8  | 29.97 | 34.34 | 33.21 | 35.14 |
| 16 | 33.08 | 36.83 | 36.57 | 51.45 |
| 32 | 39.29 | 44.52 | 53.77 | 83.93 |

## Aggregate output throughput (tokens/sec)

| concurrency | 4k | 16k | 32k | 64k |
|---:|---:|---:|---:|---:|
| 1  | 1.5 | 0.9 | 1.4 | 1.3 |
| 2  | 2.8 | 2.1 | 2.7 | 2.5 |
| 4  | 5.2 | 4.5 | 5.1 | 4.7 |
| 8  | 9.3 | 7.8 | 9.4 | 8.6 |
| 16 | 16.4 | 13.5 | 17.0 | 8.3 |
| 32 | 27.3 | 13.0 | 16.9 | 8.4 |

---

## Baseline config (cold-unique, seg=4096, bf16-KV, no APC) — TTFT (s)

For traffic where prompts are unique each time (no shared prefix to cache). Single-stream
TTFT is a bit higher and high-concurrency TTFT climbs faster (bf16-KV concurrency ceiling).

| concurrency | 4k | 16k | 32k | 64k |
|---:|---:|---:|---:|---:|
| 1  | 0.723 | 0.809 | 0.927 | 1.188 |
| 2  | 1.072 | 1.237 | 1.412 | 2.235 |
| 4  | 1.790 | 2.078 | 2.376 | 3.295 |
| 8  | 3.158 | 3.538 | 11.311 | 20.987 |
| 16 | 6.586 | 17.807 | 35.047 | 59.604 |
| 32 | 23.240 | 45.148 | 82.302 | 136.997 |

**Optimized vs baseline, single-stream TTFT:** 4k 0.409 vs 0.723, 16k 0.463 vs 0.809,
32k 0.504 vs 0.927, 64k 0.675 vs 1.188 — roughly **1.7–1.8× faster**, and much larger at
high concurrency (e.g. 32k C=16: 4.97 s vs 35.0 s).

---

## How to reproduce

```bash
# optimized (default) — repeated-context / RAG
bash run_benchmark.sh

# baseline — cold-unique prompts (restore the seg=4096 lines noted in run_benchmark.sh),
# or just force bf16 + one size:
KV_CACHE_DTYPE=auto ONLY=4k bash run_benchmark.sh
```

## Reading the numbers
- **TTFT rises with context** at C=1 (0.41 → 0.68 s for 4k → 64k) — the first-decode step
  over a longer cached KV. It stays flat with input length because the SWA windowed-gather
  keeps attention cheap.
- **TTFT climbs with concurrency** once offered load exceeds the KV concurrency ceiling
  (requests queue). FP8-KV raises that ceiling for 16k/32k/64k; the knee comes earlier for
  longer contexts (64k saturates ~C=8, 4k scales to C=32).
- **TPOT** (~0.7–1.2 s/token) is decode, which is KV-cache-I/O-bound — batching amortizes
  it (throughput scales to ~C=16). See `optimizations/` for the decode-throughput levers.
- These are **cache-hit** numbers. Cold-unique long prompts at seg=512 pay a one-time
  per-prefix cold cost; use the baseline (seg=4096) config if prompts don't repeat.

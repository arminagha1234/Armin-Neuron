# Gemma4-31B on Trainium2 — measured results (PUBLIC GA, vLLM-Neuron v0.21)

Measured on **trn2.48xlarge, TP=32**, the **public GA** vLLM NeuronX DLC
`public.ecr.aws/neuron/pytorch-inference-vllm-neuronx:0.21.0.1.0.0-neuronx-py313-sdk2.31.0-ubuntu24.04`
(vLLM 0.21, Neuron SDK 2.31, Transformers V5), model `google/gemma-4-31b-it`
(BF16 text decoder), **40 output tokens**, concurrency 1→32. Served with the bundled
[`serving_pkg/gemma4`](./serving_pkg/gemma4) + `install_public.sh`. All TTFT values are
**warm** (shared context prefix cached via APC; unique short query per request — the RAG pattern).

> This is the **public / GA** counterpart of the private-beta
> [`vllm-neuron-4k_16k_32k_64k`](../vllm-neuron-4k_16k_32k_64k) example. Same model code,
> same benchmark, **no private-beta ECR image** — it runs on the publicly available DLC.

## Config — per input size (optimized)

| input | `max-model-len` | segment (`SEG`) | KV cache | prefix caching |
|---|---:|---:|---|---|
| 4k  | 5120  | 512 | bf16 (auto) | on |
| 16k | 17408 | 512 | **fp8_e4m3** | on |
| 32k | 33792 | 512 | **fp8_e4m3** | on |
| 64k | 66560 | 512 | **fp8_e4m3** | on |

TP=32, `max-num-seqs=32`, greedy, `seg=512` (segmented/chunked prefill auto-enabled).
Reproduce with `bash run_benchmark_public.sh`.

## TTFT — Time To First Token (seconds, mean)

| concurrency | 4k | 16k | 32k | 64k |
|---:|---:|---:|---:|---:|
| 1  | 0.123 | 0.227 | 0.362 | 0.620 |
| 2  | 0.184 | 0.338 | 0.809 | 0.948 |
| 4  | 0.302 | 0.950 | 1.000 | 1.379 |
| 8  | 0.507 | 0.831 | 1.411 | 2.710 |
| 16 | 0.917 | 1.595 | 3.724 | 16.658 |
| 32 | 1.754 | 4.307 | 14.961 | 40.990 |

## TPOT — Time Per Output Token (ms, mean)

| concurrency | 4k | 16k | 32k | 64k |
|---:|---:|---:|---:|---:|
| 1  | 460 | 573 | 588 | 639 |
| 2  | 455 | 560 | 583 | 645 |
| 4  | 461 | 573 | 580 | 676 |
| 8  | 469 | 586 | 618 | 681 |
| 16 | 479 | 604 | 607 | 687 |
| 32 | 489 | 638 | 609 | 707 |

## E2E — End-to-End latency for 40 output tokens (seconds, mean)

| concurrency | 4k | 16k | 32k | 64k |
|---:|---:|---:|---:|---:|
| 1  | 16.21 | 20.29 | 19.76 | 23.63 |
| 2  | 16.33 | 20.48 | 20.35 | 24.17 |
| 4  | 16.55 | 21.28 | 20.79 | 24.99 |
| 8  | 16.96 | 21.48 | 21.71 | 27.16 |
| 16 | 17.79 | 22.92 | 25.07 | 41.36 |
| 32 | 19.47 | 26.96 | 36.61 | 66.24 |

## Aggregate output throughput (tokens/sec)

| concurrency | 4k | 16k | 32k | 64k |
|---:|---:|---:|---:|---:|
| 1  | 2.2 | 1.8 | 1.7 | 1.6 |
| 2  | 4.5 | 3.6 | 3.3 | 3.1 |
| 4  | 8.8 | 6.8 | 6.8 | 5.8 |
| 8  | 17.0 | 13.5 | 12.6 | 10.8 |
| 16 | 32.6 | 25.3 | 23.2 | 10.5 |
| 32 | 61.4 | 43.2 | 23.7 | 10.6 |

---

## Public (v0.21) vs private-beta reference

The public stack matches or beats the private-beta
[`vllm-neuron-4k_16k_32k_64k/RESULTS.md`](../vllm-neuron-4k_16k_32k_64k/RESULTS.md):

- **TPOT is lower on public** across the board: ~460–707 ms (public) vs ~688–1244 ms (beta).
- **Single-stream TTFT** is comparable/better: 0.123/0.227/0.362/0.620 s (public, 4k/16k/32k/64k)
  vs 0.409/0.463/0.504/0.675 s (beta).
- **High-concurrency long-context TTFT** climbs once offered load exceeds the KV concurrency
  ceiling and requests queue. The public scheduler reports the effective cap:
  32k caps at ~18 concurrent, 64k at ~9 (fp8-KV at these lengths). This mirrors the beta's
  saturation behavior (64k saturates ~C=8; 4k scales to C=32).

## Reading the numbers
- **TTFT rises with context** at C=1 (0.12 → 0.62 s for 4k → 64k) — the first-decode step
  over a longer cached KV. It stays low with input length thanks to segmented (windowed)
  attention over the cached prefix.
- **TPOT** (~0.46–0.71 s/token) is decode, KV-cache-I/O-bound; batching amortizes it
  (aggregate throughput scales to ~C=16–32 at short context).
- These are **cache-hit** numbers (APC). Cold-unique long prompts pay a one-time per-prefix
  prefill cost.

## Provenance
Raw per-size JSON + CSV in [`results/`](./results). Serve/bench logs are produced under
`results_<timestamp>/` when you run `run_benchmark_public.sh`.

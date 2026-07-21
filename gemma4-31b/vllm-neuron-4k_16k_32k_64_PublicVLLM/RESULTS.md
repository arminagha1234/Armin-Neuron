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

## KV cache: bf16 and fp8_e4m3, side by side

Each long-context size (16k/32k/64k) was measured with **both** KV-cache dtypes so you can pick
the tradeoff. 4k uses bf16 only (its KV is tiny). Common config: TP=32, `max-num-seqs=32`,
greedy, `seg=512` (segmented/chunked prefill auto-enabled), APC on, `max-model-len` =
5120/17408/33792/66560.

**Effective concurrency ceiling** (worst-case, no prefix-cache hit) — fp8 KV is half the size,
so it fits ~2× the concurrent requests before queueing:

| input | bf16 KV cap | fp8_e4m3 KV cap |
|---|---:|---:|
| 4k  | 32+ | — |
| 16k | 16  | 32+ |
| 32k | 9   | 18  |
| 64k | 4   | 9   |

**Takeaway:** at **low concurrency the two are within noise** (TTFT is prefill-bound, TPOT is
similar). fp8's advantage is **headroom** — at long context + high concurrency, bf16 hits its
KV ceiling sooner and TTFT balloons from queueing. Use **fp8 for long-context / high-concurrency**
serving; **bf16 is fine** for low-concurrency or short context and avoids KV quantization.

## 4k input (bf16 KV) — `max-model-len=5120`

| concurrency | TTFT (s) | TPOT (ms) | E2E (s) | agg tok/s |
|---:|---:|---:|---:|---:|
| 1  | 0.123 | 460 | 16.21 | 2.2 |
| 2  | 0.184 | 455 | 16.33 | 4.5 |
| 4  | 0.302 | 461 | 16.55 | 8.8 |
| 8  | 0.507 | 469 | 16.96 | 17.0 |
| 16 | 0.917 | 479 | 17.79 | 32.6 |
| 32 | 1.754 | 489 | 19.47 | 61.4 |

## 16k input — bf16 vs fp8_e4m3 KV — `max-model-len=17408`

| conc | TTFT bf16 | TTFT fp8 | TPOT bf16 | TPOT fp8 | E2E bf16 | E2E fp8 |
|---:|---:|---:|---:|---:|---:|---:|
| 1  | 0.222 | 0.227 | 555 | 573 | 19.10 | 20.29 |
| 2  | 0.355 | 0.338 | 558 | 560 | 19.32 | 20.48 |
| 4  | 0.608 | 0.950 | 563 | 573 | 19.74 | 21.28 |
| 8  | 0.872 | 0.831 | 565 | 586 | 20.34 | 21.48 |
| 16 | 2.421 | 1.595 | 579 | 604 | 22.57 | 22.92 |
| 32 | 13.416 | 4.307 | 585 | 638 | 33.79 | 26.96 |

## 32k input — bf16 vs fp8_e4m3 KV — `max-model-len=33792`

| conc | TTFT bf16 | TTFT fp8 | TPOT bf16 | TPOT fp8 | E2E bf16 | E2E fp8 |
|---:|---:|---:|---:|---:|---:|---:|
| 1  | 0.372 | 0.362 | 513 | 588 | 18.85 | 19.76 |
| 2  | 0.579 | 0.809 | 549 | 583 | 19.19 | 20.35 |
| 4  | 0.846 | 1.000 | 541 | 580 | 19.73 | 20.79 |
| 8  | 1.750 | 1.411 | 538 | 618 | 21.17 | 21.71 |
| 16 | 11.963 | 3.724 | 537 | 607 | 31.54 | 25.07 |
| 32 | 30.987 | 14.961 | 549 | 609 | 50.83 | 36.61 |

## 64k input — bf16 vs fp8_e4m3 KV — `max-model-len=66560`

| conc | TTFT bf16 | TTFT fp8 | TPOT bf16 | TPOT fp8 | E2E bf16 | E2E fp8 |
|---:|---:|---:|---:|---:|---:|---:|
| 1  | 0.637 | 0.620 | 717 | 639 | 25.75 | 23.63 |
| 2  | 0.978 | 0.948 | 687 | 645 | 26.30 | 24.17 |
| 4  | 2.276 | 1.379 | 706 | 676 | 28.00 | 24.99 |
| 8  | 16.082 | 2.710 | 718 | 681 | 41.98 | 27.16 |
| 16 | 44.551 | 16.658 | 717 | 687 | 70.55 | 41.36 |
| 32 | 98.915 | 40.990 | 720 | 707 | 124.95 | 66.24 |

---

## Reading the numbers
- **Low concurrency (C=1–4):** bf16 ≈ fp8. TTFT is prefill-bound and nearly identical; TPOT is
  within ~10%. If you serve one-to-a-few streams, either KV dtype is fine.
- **High concurrency + long context:** fp8 wins by a wide margin — not because a single request
  is faster, but because fp8 KV fits ~2× the concurrent requests, so bf16 queues (its TTFT at
  16k C=32 is 13.4 s vs fp8 4.3 s; 64k C=32 is 98.9 s vs fp8 41.0 s).
- **TTFT rises with context** at C=1 (0.12 → 0.64 s for 4k → 64k) but stays low thanks to
  segmented (windowed) attention over the cached prefix.
- All numbers are **cache-hit** (APC); cold-unique long prompts pay a one-time per-prefix cost.

## Public (v0.21) vs private-beta reference
The public stack matches or beats the private-beta
[`vllm-neuron-4k_16k_32k_64k/RESULTS.md`](../vllm-neuron-4k_16k_32k_64k/RESULTS.md): TPOT ~460–720 ms
(public) vs ~688–1244 ms (beta), and comparable/better single-stream TTFT.

## GPU (H100) comparison
See [`PERF_VS_GPU.md`](./PERF_VS_GPU.md) for the TTFT comparison vs H100 (uses the recommended
optimized config — fp8 KV at ≥16k for concurrency headroom).

## Provenance
Raw per-size JSON in [`results/`](./results): `results/{4k,16k,32k,64k}.json` are the **bf16**
runs; `results/fp8/{16k,32k,64k}.json` are the **fp8_e4m3** runs. Serve/bench logs are produced
under `results_<timestamp>/` when you run `run_benchmark_public.sh`.

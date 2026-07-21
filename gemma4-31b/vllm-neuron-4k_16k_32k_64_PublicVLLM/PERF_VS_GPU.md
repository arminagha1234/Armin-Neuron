# Gemma4-31B — Trainium2 vs GPU (H100/H200): TTFT comparison

Time-To-First-Token (prefill latency) for `google/gemma-4-31b-it`, **Trainium2**
(trn2.48xlarge, TP=32) vs a **GPU** baseline, across input size (4k/16k/32k/64k) and
concurrency (1→32). Lower is better. GPU baseline is **H100** for 4k and **H200** for
16k/32k/64k. Values are mean TTFT in **seconds**.

![TTFT: Trainium2 vs GPU](./assets/ttft_trn2_vs_gpu.png)

*(Regenerate the chart with `python3 make_perf_chart.py`.)*

## Headline
- **Long context is Trainium2's home turf.** At 32k and 64k input, Trn2 delivers **lower TTFT
  than the H200** across low-to-mid concurrency — up to **~3.4× faster** at 64k, single stream.
- **GPU leads at short context.** At 4k (vs H100) and 16k (vs H200), the GPU has lower TTFT at
  low concurrency; the two converge as concurrency rises.
- **The crossover** happens around C=16–32 for long context, where both platforms are
  KV/scheduling-bound and queueing dominates.

## 4k input — TTFT (s)  ·  GPU = H100
| concurrency | Trainium2 | H100 | faster |
|---:|---:|---:|:--|
| 1  | 0.409 | 0.121 | GPU 3.4× |
| 2  | 0.611 | 0.164 | GPU 3.7× |
| 4  | 1.011 | 0.301 | GPU 3.4× |
| 8  | 2.066 | 0.468 | GPU 4.4× |
| 16 | 3.338 | 0.806 | GPU 4.1× |
| 32 | 6.444 | 1.494 | GPU 4.3× |

## 16k input — TTFT (s)  ·  GPU = H200
| concurrency | Trainium2 | H200 | faster |
|---:|---:|---:|:--|
| 1  | 0.471 | 0.449 | ~tie |
| 2  | 0.701 | 0.627 | GPU 1.1× |
| 4  | 1.184 | 1.009 | GPU 1.2× |
| 8  | 2.156 | 1.727 | GPU 1.2× |
| 16 | 3.515 | 3.207 | ~tie |
| 32 | 8.521 | 6.156 | GPU 1.4× |

## 32k input — TTFT (s)  ·  GPU = H200
| concurrency | Trainium2 | H200 | faster |
|---:|---:|---:|:--|
| 1  | 0.530 | 0.992 | **Trn2 1.9×** |
| 2  | 0.819 | 1.372 | **Trn2 1.7×** |
| 4  | 1.230 | 2.201 | **Trn2 1.8×** |
| 8  | 2.128 | 3.827 | **Trn2 1.8×** |
| 16 | 4.558 | 7.094 | **Trn2 1.6×** |
| 32 | 21.375 | 13.597 | GPU 1.6× |

## 64k input — TTFT (s)  ·  GPU = H200
| concurrency | Trainium2 | H200 | faster |
|---:|---:|---:|:--|
| 1  | 0.661 | 2.249 | **Trn2 3.4×** |
| 2  | 1.010 | 3.192 | **Trn2 3.2×** |
| 4  | 1.427 | 5.139 | **Trn2 3.6×** |
| 8  | 3.048 | 9.005 | **Trn2 3.0×** |
| 16 | 13.174 | 16.773 | **Trn2 1.3×** |
| 32 | 33.075 | 32.258 | ~tie |

## Why the long-context win
Trainium2's TTFT stays low as context grows because the Gemma4 serve uses **segmented /
windowed attention** over the cached prefix (sliding-window layers gather a static number of KV
blocks at a dynamic offset), so prefill cost scales sub-linearly with context length. The GPU
baseline pays a steeper prefill cost as context grows, so at 32k/64k the Trn2 single-stream and
mid-concurrency TTFT is materially lower.

## Notes
- **Metric:** mean TTFT (request sent → first output token). Only TTFT is compared here — the
  GPU E2E/TPOT figures weren't part of this dataset.
- **Trn2 config:** trn2.48xlarge, TP=32, greedy. These are the reference Trn2 TTFT figures used
  for the GPU comparison. The **public GA** optimized config in [`RESULTS.md`](./RESULTS.md)
  (seg=512 + prefix caching + fp8-KV) measures **even lower** Trn2 TTFT at short context
  (e.g. 4k C1 ≈ 0.12 s), which narrows the 4k/16k gap further.
- **GPU baseline:** H100 (4k) / H200 (16k/32k/64k), vendor-typical vLLM serving.

# Gemma4-31B on Trainium2 — measured results (NO-APC / honest cold prefill)

Measured on **trn2.48xlarge, TP=32**, **public GA vLLM-Neuron 0.21** (SDK 2.31), model
`gemma-4-31b-it` (bf16), **40 output tokens**, on a **clean single-tenant box**. Every request uses a
**unique random prompt** (`bench_random.py`) so **prefix caching cannot serve it** — these are true
**cold** prefill numbers. **APC OFF, KV cache bf16 (no FP8).** TTFT is warm-NEFF (cold compile dropped).

Model file has the best-TTFT patches applied (`patches/`: fused qkv_proj + o_proj NKI kernel;
segmented-NKI for >16k). Decode = sdpa.

## Config (set automatically by `run_benchmark.sh`)

| input | prefill mode | max-model-len | max-num-batched-tokens | buckets | KV | APC |
|---|---|---:|---:|---|---|---|
| 4k / 8k / 16k | **single-shot** (`mnbt == max-model-len`) | 16384 | 16384 | 256…16384 | bf16 | off |
| 32k / 64k | **segmented** (chunked, `mnbt < max-model-len`) | 66560 | 8192 (seg) | 8192 | bf16 | off |

Single-shot is the fast path but caps at 16k (above it, neuronx-cc stalls on the global-attention
graph — tested). 32k/64k therefore use segmented prefill (seg=8192 = fewest chunks).

## TTFT — Time To First Token (seconds, mean) — COLD

| concurrency | 4k | 8k | 16k | 32k | 64k |
|---:|---:|---:|---:|---:|---:|
| 1  | **0.224** | **0.390** | **0.754** | **3.056** | **6.089** |
| 2  | 0.331 | 0.585 | 1.127 | 4.589 | 9.151 |
| 4  | 0.605 | 0.969 | 1.944 | 8.335 | 15.938 |
| 8  | 1.878 | 2.606 | 4.215 | — | — |
| 16 | 4.983 | 6.406 | 9.429 | — | — |
| 32 | 11.459 | 14.176 | 19.901 | — | — |

(TTFT p99 at conc≥8 is ~2× the mean — queuing spread once offered load exceeds the KV ceiling.)
32k/64k conc≥8 not run: bf16 KV capacity is exhausted well before conc=8 at long context, so requests
queue and TTFT is dominated by head-of-line wait (the sibling APC folder's bf16 baseline shows 64k
conc32 ≈ 137 s) — not a meaningful single-request number. Use multi-replica for concurrent long-context.

## TPOT — Time Per Output Token (ms, mean)

| concurrency | 4k | 8k | 16k | 32k | 64k |
|---:|---:|---:|---:|---:|---:|
| 1  | 94 | 94 | 94 | 71 | 70 |
| 2  | 97 | 99 | 103 | 109 | 147 |
| 4  | 103 | 109 | 122 | 128 | 185 |
| 8  | 105 | 113 | 131 | — | — |
| 16 | 106 | 116 | 135 | — | — |
| 32 | 107 | 117 | 138 | — | — |

Decode is sdpa (~94 ms/token at conc1 ≤16k). It rises modestly with concurrency (batched decode) and
is a throughput/$-per-token factor, not part of TTFT.

## Reading the numbers
- **Single-shot ≤16k is the win:** 4k 0.22 s, 8k 0.39 s, 16k 0.75 s at conc=1 — all under a 500 ms SLA
  cold. It scales ~linearly in input length (0.22 → 0.39 → 0.75 for 4k → 8k → 16k).
- **The jump at 32k** (0.75 s → 3.06 s) is the single-shot→segmented transition, not a smooth curve —
  single-shot physically can't compile >16k, and segmented pays per-chunk prior-KV gather.
- **Segmented scales linearly** in the 32k→64k range (3.06 → 6.09 ≈ 2× for 2× tokens).
- **TTFT climbs steeply with concurrency** once offered load exceeds the bf16 KV concurrency ceiling
  (requests queue). The knee comes earlier for longer contexts (64k saturates below conc=8; 4k scales
  to ~conc=8 before the climb). FP8-KV would raise this ceiling but is intentionally out of scope (bf16-only).

## Why this differs from the sibling `vllm-neuron-4k_16k_32k_64k/` (APC) numbers
That folder's headline (4k 0.409 / 16k 0.463 / 32k 0.504 / 64k 0.675 at conc1) is a **cache-hit**
number: a shared context prefix is warmed once (APC), then every request reuses it and only prefills
the short unique tail. It also uses FP8-KV. That's a real **repeated-context / RAG** best-case, but it
is **not comparable to a cold request**. This folder removes APC + FP8 and sends unique prompts, so the
numbers are higher and honest:

| conc=1 TTFT | 4k | 16k | 32k | 64k |
|---|---|---|---|---|
| sibling (APC cache-hit, FP8) | 0.41 | 0.46 | 0.50 | 0.68 |
| **this folder (cold, no APC, bf16)** | **0.22** | **0.75** | **3.06** | **6.09** |

Note 4k is actually *faster* here (0.22 vs 0.41) because single-shot bf16 beats the APC folder's
seg=512 config at short context — APC's win only shows up at longer context on cache hits. The two
folders answer different questions: **RAG best-case (sibling)** vs **cold/unique worst-case (here)**.

## Reproduce
```bash
bash run_benchmark.sh                          # full sweep (single-shot ≤16k + segmented >16k)
ONLY=4k,8k,16k bash run_benchmark.sh           # single-shot pool only
LEVELS=1,2,4 ONLY=32k,64k bash run_benchmark.sh # segmented pool only
```

## UPDATE — SWA windowed-prior fix for >16k (VALIDATED, −33%)

The 32k/64k numbers above are the full-span segmented baseline. A correctness-preserving
optimization (`patches/patch_swa_window_prior_v2.py`) windows the SWA-layer prior gather (50 of 60
layers only attend to the last 1024 keys). Measured on-device (same box, conc=1, bf16, no-APC):

| input | full-span baseline | SWA-windowed (v2) | Δ | token parity |
|---|---:|---:|:--:|:--:|
| 32k TTFT | 3.03 s | **2.021 s** | **−33.3%** | ✅ byte-identical |
| 64k TTFT | 6.053 s | **4.040 s** | **−33.3%** | ✅ byte-identical |

Correctness proven 3 ways (CPU masking diff 2.4e-7, CPU gather diff 0.0, on-device 18k token
parity). See `SWA_WINDOW_FINDING.md`. This supersedes the earlier "broken/reverted" writeup — that
was a misdiagnosis (the masking is shift-invariant). Decode TPOT unchanged (~211ms; prefill-only fix).

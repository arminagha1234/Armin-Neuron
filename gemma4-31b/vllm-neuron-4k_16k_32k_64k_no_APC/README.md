# Gemma4-31B on AWS Trainium — serving benchmark (NO-APC / honest cold prefill)

A dependency-free benchmark that measures **TTFT / TPOT / E2E** for **Gemma4-31B** on AWS Trainium
(Trn2), across a concurrency sweep and input sizes **4k / 8k / 16k / 32k / 64k**, **40 output tokens**.

**This is the honest, apples-to-apples COLD variant of the sibling
[`vllm-neuron-4k_16k_32k_64k/`](../vllm-neuron-4k_16k_32k_64k/).** The difference is the whole point:

| | sibling folder (APC) | **this folder (no-APC)** |
|---|---|---|
| prefix caching (APC) | **on** | **off** |
| KV cache | fp8_e4m3 (≥16k) | **bf16** |
| prefill | segmented seg=512 | **single-shot ≤16k**, segmented >16k |
| workload | shared prefix + short query → **cache hit** | **unique random prompt every request → cold** |
| what it measures | best-case repeated-context (RAG) TTFT | **true cold-prefill TTFT** |

The sibling's headline numbers are *cache-hit* numbers (a shared context prefix is warmed once, then
every request reuses it). That's a real and useful RAG number, but it is **not comparable to a cold
request** — on a cache hit only the short unique tail is prefilled. This folder removes that advantage:
APC off, and every request gets a **unique random prompt** so nothing can be served from cache. Every
number here is a genuine cold prefill.

It also uses **single-shot prefill** for ≤16k, which is markedly faster than the segmented path there.

---

## Just want to run it? → [LAUNCH.md](./LAUNCH.md)

Same runbook as the sibling folder; the only differences are baked into `run_benchmark.sh` (APC off,
bf16, single-shot ≤16k / segmented >16k, unique-random-prompt client). Start there.

## Reproduce in 3 steps
1. **Set up the container** — [SETUP.md](./SETUP.md).
2. **Get this benchmark (in the container):**
   ```bash
   git clone https://github.com/arminagha1234/Armin-Neuron.git
   cd Armin-Neuron/gemma4-31b/vllm-neuron-4k_16k_32k_64k_no_APC
   ```
3. **Run it:**
   ```bash
   MODEL=/root/models/gemma-4-31b-it bash run_benchmark.sh
   ```
   It launches two servers — a **single-shot** one (LEN=16384) for 4k/8k/16k and a **segmented** one
   (LEN=66560, seg=8192) for 32k/64k — runs the concurrency sweep with unique random prompts, and
   writes `results_<timestamp>/` (per-size JSON + `summary.txt` + `summary.csv`).

   Existing server? `SKIP_LAUNCH=1 BASE_URL=http://localhost:8000 bash run_benchmark.sh`.

## What "single-shot" means (it is not a vLLM flag)
"Single-shot" is the vLLM-Neuron plugin's name for **chunked prefill disabled** — the whole prompt is
prefilled in one pass. It's selected automatically when **`--max-num-batched-tokens == --max-model-len`**
(and `max_model_len ≤ 16384`, the plugin's single-shot cap). When `--max-num-batched-tokens < --max-model-len`,
you get **segmented** prefill (upstream vLLM chunked prefill; chunk ∈ {512,1024,2048,4096,8192}).
`launch_serve.sh` sets `SEG==LEN` for single-shot and `SEG<LEN` for segmented. There is no `--single-shot` argument.

> **Why >16k is segmented:** single-shot above 16k is a hard compiler limit, not a config choice —
> raising the cap and compiling a 32k single-shot graph does **not** OOM but **stalls neuronx-cc**
> (the WalrusDriver backend grinds indefinitely on the hd512 global-attention graph). Tested directly.

## Configuration (defaults)

| Variable | Default | Notes |
|---|---|---|
| `GEN` | `40` | output tokens |
| `LEVELS` | `1,2,4,8,16,32` | concurrency (≤16k) |
| `LEVELS_LONG` | `1,2,4` | 32k/64k — high concurrency is KV-capacity-bound (queues) |
| `ONLY` | `4k,8k,16k,32k,64k` | input sizes |
| `TP` | `32` | tensor-parallel size |
| `MNS` | `32` | max-num-seqs |
| **`APC`** | **`0`** | **prefix caching OFF** (the whole point of this folder) |
| **KV cache** | **bf16** | no FP8 |
| `MODEL` | `google/gemma-4-31B-it` | HF id or local path |

Per-size serve config (set automatically by `run_benchmark.sh`):

| input | prefill | max-model-len | max-num-batched-tokens | buckets | KV |
|---|---|---:|---:|---|---|
| 4k / 8k / 16k | **single-shot** | 16384 | 16384 (==len) | 256…16384 | bf16 |
| 32k / 64k | **segmented** | 66560 | 8192 (seg) | 8192 | bf16 |

## Reference numbers — cold, single-stream (conc=1), TTFT (seconds)

Measured on **trn2.48xlarge, TP=32**, public GA vLLM-Neuron 0.21, clean single-tenant box, bf16, APC
off, unique random prompts, 40 output tokens. (3-sample at ≤16k; variance <5ms.)

| input | 4k | 8k | 16k | 32k | 64k |
|---|---|---|---|---|---|
| **TTFT (conc=1)** | **0.22 s** | **0.39 s** | **0.75 s** | **3.06 s** | **6.09 s** |
| prefill path | single-shot | single-shot | single-shot | segmented | segmented |

≤16k is single-shot and comfortably under a 500 ms SLA at conc=1. The jump at 32k is the single-shot→
segmented transition (single-shot can't compile >16k). Full concurrency tables: **[RESULTS.md](./RESULTS.md)**.

## Files
- `LAUNCH.md` / `SETUP.md` — runbook + container setup
- `RESULTS.md` — measured TTFT / TPOT / E2E tables (no-APC, cold)
- `run_benchmark.sh` — main entry (single-shot ≤16k + segmented >16k, APC off)
- `launch_serve.sh` — launches one server; `SEG==LEN` → single-shot, `SEG<LEN` → segmented
- `bench_random.py` — **unique-random-prompt** concurrency client (defeats prefix caching), stdlib only
- `summarize.py` — combines per-size results into one table + CSV
- `patches/` — the model-side changes that give the best TTFT (from `optimizations/`): `patch_qkv_proj.py`
  (fused QKV+QK-norm+RoPE), `patch_oproj.py` (o_proj NKI kernel), `patch_segmented_nki.py` (routes >16k
  segmented prefill through NKI flash), `test_correctness.py` (cosine gate), `APPLY_AND_TEST.md` (apply/revert)
- `serving_pkg/` — Gemma4-31B model + vLLM-Neuron registration

## Notes
- Every number is **cold** — no prefix-cache reuse. For repeated-context / RAG best-case, see the
  sibling APC folder; the two are complementary, not competing.
- TPOT (~94 ms/token ≤16k, ~70 ms >16k) is sdpa decode; it's a throughput/$-per-token factor, not TTFT.
- High concurrency at 32k/64k is KV-capacity-bound (bf16) — requests queue; that's why `LEVELS_LONG`
  stops at 4. FP8-KV would raise that ceiling but is intentionally out of scope here (bf16-only).

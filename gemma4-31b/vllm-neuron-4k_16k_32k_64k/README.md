# Gemma4-31B on AWS Trainium — serving benchmark

A small, dependency-free benchmark that measures **TTFT**, **TPOT**, and **E2E latency** for
**Gemma4-31B** served with vLLM-Neuron on AWS Trainium (Trn2), across a concurrency sweep and a range
of input lengths.

It runs, for **input sizes 4k / 16k / 32k / 64k tokens**, at **concurrency 1, 2, 4, 8, 16, 32**, with
**40 output tokens** per request, and prints one clean summary table.

Gemma4-31B is not in the base vLLM-Neuron Beta — support is added by the bundled
[`serving_pkg/`](./serving_pkg/), loaded automatically. No vLLM fork, no manual registration steps.

---

## Reproduce in 3 steps

### Step 1 — set up the environment (one time)
Follow **[SETUP.md](./SETUP.md)**: pull the vLLM-Neuron Beta image, install the Neuron driver, and start
the container on your Trn2 instance. (Ends with you `exec`'d into the container.)

### Step 2 — get this benchmark (inside the container)
```bash
git clone https://github.com/arminagha1234/Armin-Neuron.git
cd Armin-Neuron/gemma4-31b/vllm-neuron-4k_16k_32k_64k
```

### Step 3 — run it
```bash
bash run_benchmark.sh
```
That's it. For each input size it launches the Gemma4-31B server (via `launch_serve.sh`, which puts
`serving_pkg/` on `PYTHONPATH` so Gemma4 is recognized), runs the concurrency sweep, and writes
everything to `results_<timestamp>/` (per-size JSON + `summary.txt` + `summary.csv`).

Already have a server running? Skip the launch and point at it:
```bash
SKIP_LAUNCH=1 BASE_URL=http://localhost:8000 bash run_benchmark.sh
```

---

## Requirements
- AWS Trainium2 instance (reference numbers below are from **trn2.48xlarge**, TP=32).
- vLLM-Neuron Beta container (see [SETUP.md](./SETUP.md)).
- Access to the **`google/gemma-4-31B-it`** weights on Hugging Face (gated — accept the license + set
  `HF_TOKEN`).
- **Python 3.8+** — the benchmark client uses only the standard library (no `pip install`).

## What it measures

| Metric | Meaning |
|---|---|
| **TTFT** (Time To First Token, s) | request sent → first output token — prefill latency |
| **TPOT** (Time Per Output Token, ms) | steady-state decode latency per token = `(E2E − TTFT) / (out_tokens − 1)` |
| **E2E** (End-to-End, s) | request sent → last output token = `TTFT + (out_tokens − 1) × TPOT` |
| **throughput** | output tokens/sec (aggregate across the batch, and per request) |

Each level is reported as mean / p50 / p99 for TTFT.

## Configuration

All settings are environment variables (defaults shown). Override any inline, e.g.
`ONLY=4k,16k KV_CACHE_DTYPE=fp8_e4m3 bash run_benchmark.sh`.

| Variable | Default | Notes |
|---|---|---|
| `GEN` | `40` | output tokens per request |
| `LEVELS` | `1,2,4,8,16,32` | concurrency levels |
| `ONLY` | `4k,16k,32k,64k` | which input sizes to run |
| `TP` | `32` | tensor-parallel size |
| `MNS` | `32` | max-num-seqs (must be ≥ max concurrency, else requests queue) |
| `KV_CACHE_DTYPE` | per-size | bf16 at 4k, `fp8_e4m3` at 16k/32k/64k (set per input size). Override to force one dtype everywhere. |
| `APC` | `1` | `1` = enable prefix caching (`--enable-prefix-caching`); the big TTFT win for repeated context. Requires `SEG < max-model-len`. |
| `MODEL` | `google/gemma-4-31B-it` | checkpoint (HF id or local path) |
| `SERVING_PKG` | `./serving_pkg` | Gemma4 registration package (bundled); auto-added to `PYTHONPATH` |
| `BASE_URL` | `http://localhost:8000` | server URL (use with `SKIP_LAUNCH=1`) |
| `SKIP_LAUNCH` | `0` | `1` = benchmark an already-running server at `BASE_URL` |

### Default config: optimized for repeated-context (RAG)

Per input size, `run_benchmark.sh` launches the server with **`seg=512` + prefix caching
(APC) + FP8-KV (≥16k) + right-sized `max-model-len`** — the config that produces the
reference numbers below. This is tuned for **repeated-context / RAG** traffic (a shared
context prefix that gets cached, plus a short unique query per request):

| input | max-model-len | segment (`SEG`) | buckets | KV cache | prefix caching |
|---|---:|---:|---:|---|---|
| 4k  | 5120  | 512 | 512 | bf16 (auto) | on |
| 16k | 17408 | 512 | 512 | **fp8_e4m3** | on |
| 32k | 33792 | 512 | 512 | **fp8_e4m3** | on |
| 64k | 66560 | 512 | 512 | **fp8_e4m3** | on |

> **Cold-unique traffic?** If prompts don't repeat (no shared prefix to cache), use the
> baseline `seg=4096` / bf16-KV config instead — the per-size lines are preserved as a
> comment in `run_benchmark.sh`. See **[RESULTS.md](./RESULTS.md)** for both.

Full measured TTFT / TPOT / E2E / throughput tables for both configs: **[RESULTS.md](./RESULTS.md)**.

## Output

`results_<timestamp>/` contains per-size JSON, `summary.txt` (human-readable table), `summary.csv`, and
server/benchmark logs. Example (16k, optimized config):
```
### 16k input, 40 output tokens
  conc   TTFT_s  TTFT_p99   TPOT_ms    E2E_s    tok/s  tok/s/req
     1    0.463     0.463   1084.74   30.836      0.9       0.94
     8    2.551     4.044    993.45   34.341      7.8       0.97
     ...
```

## Reference numbers (trn2.48xlarge, TP=32, 40 output tokens) — TTFT (seconds)

Measured with the default optimized config (`seg=512` + APC + FP8-KV, cache-hit workload).
For sanity-checking your run — numbers vary with instance, vLLM-Neuron build, and traffic
pattern. Full TTFT / TPOT / E2E / throughput tables (and the baseline config) are in
**[RESULTS.md](./RESULTS.md)**.

| concurrency | 4k | 16k | 32k | 64k |
|---|---|---|---|---|
| 1  | 0.409 | 0.463 | 0.504  | 0.675  |
| 2  | 0.610 | 0.699 | 0.778  | 1.390  |
| 4  | 0.998 | 1.105 | 1.700  | 1.908  |
| 8  | 1.779 | 2.551 | 3.240  | 3.520  |
| 16 | 3.335 | 3.463 | 4.972  | 19.406 |
| 32 | 6.432 | 8.343 | 21.694 | 51.624 |

## Files
- `SETUP.md`          — Step 1: beta container setup
- `RESULTS.md`        — measured TTFT / TPOT / E2E / throughput tables (optimized + baseline)
- `run_benchmark.sh`  — main entry (all input sizes + summary)
- `launch_serve.sh`   — launches a Gemma4-31B server for one config (loads `serving_pkg/`)
- `bench.py`          — concurrency sweep client (TTFT / TPOT / E2E), stdlib only
- `summarize.py`      — combines per-size results into one table + CSV
- `serving_pkg/`      — Gemma4-31B model + vLLM-Neuron registration (see its README)

## Notes
- **TPOT** is per-token and largely length-independent, so it's the most robust cross-run decode metric.
  **E2E** scales with output length — keep `GEN` fixed (default 40) when comparing runs.
- `MNS` below your max concurrency causes queuing and inflates TTFT at high concurrency.

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
| `KV_CACHE_DTYPE` | `auto` | `auto` = bf16 KV cache; `fp8_e4m3` = fp8 KV cache |
| `MODEL` | `google/gemma-4-31B-it` | checkpoint (HF id or local path) |
| `SERVING_PKG` | `./serving_pkg` | Gemma4 registration package (bundled); auto-added to `PYTHONPATH` |
| `BASE_URL` | `http://localhost:8000` | server URL (use with `SKIP_LAUNCH=1`) |
| `SKIP_LAUNCH` | `0` | `1` = benchmark an already-running server at `BASE_URL` |

Per input size, `run_benchmark.sh` launches the server with:

| input | max-model-len | segment | buckets |
|---|---|---|---|
| 4k  | 5120  | 4096 | 512,1024,2048,4096 |
| 16k | 20480 | 4096 | 4096 |
| 32k | 36864 | 4096 | 4096 |
| 64k | 69632 | 4096 | 4096 |

## Output

`results_<timestamp>/` contains per-size JSON, `summary.txt` (human-readable table), `summary.csv`, and
server/benchmark logs. Example:
```
### 16k input, 40 output tokens
  conc   TTFT_s  TTFT_p99   TPOT_ms    E2E_s    tok/s  tok/s/req
     1    0.471     0.471     31.20    1.520      3.0       3.0
     ...
```

## Reference numbers (trn2.48xlarge, TP=32, 40 output tokens) — TTFT (seconds)

For sanity-checking your run. Numbers vary with instance, vLLM-Neuron build, and KV-cache dtype — treat
as a ballpark, not an exact target.

| concurrency | 4k | 16k | 32k | 64k |
|---|---|---|---|---|
| 1  | 0.409 | 0.471 | 0.530  | 0.661  |
| 2  | 0.611 | 0.701 | 0.819  | 1.010  |
| 4  | 1.011 | 1.184 | 1.230  | 1.427  |
| 8  | 2.066 | 2.156 | 2.128  | 3.048  |
| 16 | 3.338 | 3.515 | 4.558  | 13.174 |
| 32 | 6.444 | 8.521 | 21.375 | 33.075 |

## Files
- `SETUP.md`          — Step 1: beta container setup
- `run_benchmark.sh`  — main entry (all input sizes + summary)
- `launch_serve.sh`   — launches a Gemma4-31B server for one config (loads `serving_pkg/`)
- `bench.py`          — concurrency sweep client (TTFT / TPOT / E2E), stdlib only
- `summarize.py`      — combines per-size results into one table + CSV
- `serving_pkg/`      — Gemma4-31B model + vLLM-Neuron registration (see its README)

## Notes
- **TPOT** is per-token and largely length-independent, so it's the most robust cross-run decode metric.
  **E2E** scales with output length — keep `GEN` fixed (default 40) when comparing runs.
- `MNS` below your max concurrency causes queuing and inflates TTFT at high concurrency.

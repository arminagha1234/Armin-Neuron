# Gemma4-31B on AWS Trainium — serving benchmark

A small, dependency-free benchmark that measures **TTFT**, **TPOT**, and **E2E latency** for
**Gemma4-31B** served with vLLM-Neuron on AWS Trainium (Trn2), across a concurrency sweep and a range
of input lengths.

It runs, for **input sizes 4k / 16k / 32k / 64k tokens**, at **concurrency 1, 2, 4, 8, 16, 32**, with
**40 output tokens** per request, and prints one clean summary table.

---

## Quickstart

On your Trainium instance, inside your vLLM-Neuron environment (with the Gemma4-31B checkpoint available):

```bash
git clone <this-repo> && cd gemma4_31_example
bash run_benchmark.sh
```

That's it. It launches a server for each input size, runs the concurrency sweep, and writes everything
to a `results_<timestamp>/` folder (per-size JSON + `summary.txt` + `summary.csv`).

Already have a server running? Point the benchmark at it instead of launching one:

```bash
SKIP_LAUNCH=1 BASE_URL=http://localhost:8000 bash run_benchmark.sh
```

---

## Requirements

- An AWS Trainium instance (results below are from **trn2.48xlarge**, TP=32).
- A vLLM-Neuron environment that serves **Gemma4-31B** (`vllm serve` with the Neuron backend).
- The **Gemma4-31B** checkpoint (local path or HF id) — set `MODEL` (default `/root/models/gemma-4-31b-it`).
- **Python 3.8+** — the benchmark client uses only the standard library (no `pip install`).

---

## What it measures

| Metric | Meaning |
|---|---|
| **TTFT** (Time To First Token, s) | request sent → first output token — prefill latency |
| **TPOT** (Time Per Output Token, ms) | steady-state decode latency per token = `(E2E − TTFT) / (out_tokens − 1)` |
| **E2E** (End-to-End, s) | request sent → last output token = `TTFT + (out_tokens − 1) × TPOT` |
| **throughput** | output tokens/sec (aggregate across the batch, and per request) |

The client warms a shared context prefix, then sends `concurrency` simultaneous requests
(shared prefix + a short unique query) and reports mean / p50 / p99.

---

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
| `MODEL` | `/root/models/gemma-4-31b-it` | checkpoint path or HF id |
| `SERVING_PKG` | *(unset)* | optional `PYTHONPATH` to a custom vLLM-Neuron build |
| `BASE_URL` | `http://localhost:8000` | server URL (use with `SKIP_LAUNCH=1`) |
| `SKIP_LAUNCH` | `0` | `1` = benchmark an already-running server at `BASE_URL` |

Per input size, `run_benchmark.sh` launches the server with (edit the bottom of the script to change):

| input | max-model-len | segment | buckets |
|---|---|---|---|
| 4k  | 5120  | 4096 | 512,1024,2048,4096 |
| 16k | 20480 | 4096 | 4096 |
| 32k | 36864 | 4096 | 4096 |
| 64k | 69632 | 4096 | 4096 |

---

## Output

`results_<timestamp>/` contains:
- `4k.json`, `16k.json`, `32k.json`, `64k.json` — raw per-concurrency metrics
- `summary.txt` — the human-readable table
- `summary.csv` — same data for spreadsheets
- `serve_*.log`, `*.log` — server + benchmark logs

Example `summary.txt` block:
```
### 16k input, 40 output tokens
  conc   TTFT_s  TTFT_p99   TPOT_ms    E2E_s    tok/s  tok/s/req
     1    0.471     0.471     31.20    1.520      3.0       3.0
     ...
```

---

## Reference numbers (AWS Trainium trn2.48xlarge, TP=32, 40 output tokens) — TTFT (seconds)

For sanity-checking your run. Your numbers will vary with instance, vLLM-Neuron build, and KV-cache
dtype, so treat these as a ballpark, not an exact target.

| concurrency | 4k | 16k | 32k | 64k |
|---|---|---|---|---|
| 1  | 0.409 | 0.471 | 0.530  | 0.661  |
| 2  | 0.611 | 0.701 | 0.819  | 1.010  |
| 4  | 1.011 | 1.184 | 1.230  | 1.427  |
| 8  | 2.066 | 2.156 | 2.128  | 3.048  |
| 16 | 3.338 | 3.515 | 4.558  | 13.174 |
| 32 | 6.444 | 8.521 | 21.375 | 33.075 |

---

## Files
- `run_benchmark.sh` — main entry (runs all input sizes + summary)
- `launch_serve.sh`  — launches a Gemma4-31B server for one config
- `bench.py`         — concurrency sweep client (TTFT / TPOT / E2E), stdlib only
- `summarize.py`     — combines per-size results into one table + CSV

## Notes
- TTFT and E2E depend on input/output length and config; **TPOT** is per-token and largely
  length-independent, so it's the most robust cross-run decode metric.
- E2E scales with output length — keep `GEN` fixed (default 40) when comparing runs.
- `MNS` below your max concurrency causes queuing and inflates TTFT at high concurrency.

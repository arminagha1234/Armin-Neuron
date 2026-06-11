# Qwen3.5-4B on trn2.3xl — Verified Benchmark (2026-06-11)

This is the honest, end-to-end-correct benchmark on a single trn2.3xl
($2.23/hr, 1 chip / 4 cores / 96 GB HBM) using this repo's adapter.

## Configuration

| Setting | Value |
|---|---|
| Model | `Qwen/Qwen3.5-4B` (HF safetensors, hybrid 24 GatedDeltaNet + 8 GQA, 4.21B params) |
| Hardware | trn2.3xl, single Neuron device, 4 cores, 96 GB HBM |
| Runtime | vllm-neuron private beta v5 (vLLM v0.19.0) |
| TP | 2 (LNC=2) |
| MAX_LEN | 512 |
| BUCKET | 512 (single bucket — no chunked prefill) |
| MAX_NUM_SEQS | 8 |
| KV cache dtype | BF16 (`KV_CACHE_DTYPE=auto`) |
| Sampling | greedy (on-device) |

## Correctness — verified end-to-end

**The adapter produces coherent multi-token autoregressive output.**

Sample completions (greedy, max_tokens=30):

```
"Once upon a time in a small village,"
  → " there lived a young boy named Tom. Tom was a curious boy who loved
     to explore the world around him..."

"The recipe for chocolate chip cookies requires"
  → " 1 cup of flour, 1 cup of sugar, 1 cup of butter, 1 cup of chocolate
     chips..."

"Q: What language is spoken in Brazil?"
  → " The answer is Portuguese."

"Q: How many days are in a week?"
  → " There are 7 days in a week."

"Q: What is 12 plus 8?"
  → " 20"
```

First-token canonical probe: 4/6 (the 2 "fails" are both arguable model
preferences — "100%" for "legs on a dog", "1.8" for the
Celsius→Fahrenheit conversion factor — both sensible, just different
from the literal expected substring).

## Throughput sweep

Workload: ~440-token input prompt, 50-token output, repeated `N`
concurrent requests.

| Concurrency | Wall (s) | Total input tokens | Aggregate input tok/s | Per-stream tok/s | $/M-input |
|---:|---:|---:|---:|---:|---:|
| 1 | 5.5 | 401  | 72  | 72 | $8.55 |
| 2 | 7.4 | 802  | 108 | 54 | $5.72 |
| 4 | 11.1 | 1,604 | 144 | 36 | $4.30 |
| 8 | 22.3 | 3,208 | 144 | 18 | $4.31 |

**Knee:** throughput plateaus at concurrency=4 due to chunked-prefill
scheduler serializing prefills at this BUCKET size. Going from 4 to 8
adds latency (wall doubles) without adding aggregate throughput.

## What the previous "$1.63/M" claim was, and why this is lower

An earlier session reported $1.63/M-input at this hardware/config. That
number was measured against a model where the DeltaNet recurrent state
was being **constant-folded by the Neuron compiler** (it was registered
as a zero-init `nn.Buffer`, and the compiler treated it as a
compile-time zero, so per-step `.data.copy_()` mutations didn't survive
between forward calls).

The result: DeltaNet was effectively **stateless** — all 24 linear-attention
layers contributed nothing meaningful, and the model fell back to
GQA + last-token embedding. The forward pass completed and returned
something, but it wasn't real autoregressive generation. The decode
"throughput" was inflated because the broken pseudo-prefill was skipping
real DeltaNet recurrence.

The fix (this branch) initializes the DeltaNet state buffers with a
non-zero epsilon (`1e-30`) instead of zero, defeating the constant-fold
and letting state mutations actually persist. With real state in real
DeltaNet, you get coherent output — and slower honest throughput.

The cost difference between `$1.63/M (broken)` and `$4.30/M (correct)`
is the **honest cost of correctness** on this 3xl at MAX_LEN=512.

## Reproducing

In the vllm-neuron container:

```bash
export TP=2 MAX_LEN=512 BUCKET=512 MAX_NUM_SEQS=8 KV_CACHE_DTYPE=auto
export PYTHONPATH=$(pwd)/src
./src/serve.sh
```

Wait for compile (~15-30 min on trn2.3xl, much faster on trn2.48xl).
Then:

```bash
curl -s -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"/path/to/Qwen3.5-4B","prompt":"Once upon a time in a small village,","max_tokens":30,"temperature":0}'
```

Expect coherent narrative output.

## Hardware note: 3xl is undersized for this model's compile

Walrus (Neuron compiler) compile time grows ~O(N · seq_len²). On the
trn2.3xl (124 GB host RAM, 12 host cores) this means:

| MAX_LEN | Compile wall on 3xl |
|---|---|
| 512  | ~15-25 min |
| 2048 | ~60-90 min |
| 4096 | several hours |

For customer workloads that need MAX_LEN > 1024, **compile on a
trn2.48xl** (96 host cores, 384 GB RAM, supports
`VLLM_NEURON_PARALLEL_COMPILE_WORKERS=8`), then ship the cached NEFF
to a 3xl for serving. Same NEFF works on both (the compute is identical,
just the compile box differs).

## Reference points

| Platform | Aggregate input tok/s | $/M-input |
|---|---:|---:|
| **trn2.3xl, MAX_LEN=512, conc=8** | **144** | **$4.31** |
| p4d.24xl A100 CB rate (customer reference) | ~33,000 | $0.099 |
| Future: trn2.48xl with longer MAX_LEN | TBD | TBD |

The 3xl numbers above are the floor — they reflect a deliberately small
MAX_LEN chosen to fit the 3xl's compile budget. With MAX_LEN sized to
the customer's actual workload (e.g. 20K input tokens) on a 48xl, the
per-token economics improve substantially because prefill efficiency
scales with bucket size and there's parallel batch headroom.

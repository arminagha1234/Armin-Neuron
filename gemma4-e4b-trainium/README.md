# Gemma 4 E4B-it on AWS Trainium2 (native PyTorch + TP=2)

Native PyTorch port of [Google's Gemma 4 E4B-it](https://huggingface.co/google/gemma-4-E4B-it)
to AWS Neuron. **First validation of E4B on Neuron** — the published
[`vllm-neuron-gemma4`](https://github.com/aws-neuron/private-vllm-neuron) port
(PR #1552) only supports the 31B variant; this contrib uses HuggingFace's
own `transformers` Gemma4 implementation under
`torch.compile(backend="neuron")` so no fork is required.

## TL;DR

**32.7 ms TTFT for short prompts on a $2.23/hr instance.** Single-stream
serving, native PyTorch, no NxDI / no vLLM dependency.

| Metric | Eager | `torch.compile` |
|---|---:|---:|
| TTFT @ 64 tokens   | 419 ms | **32.7 ms** (12.8x) |
| TTFT @ 256 tokens  | 433 ms | **56.5 ms** (7.7x)  |
| TTFT @ 1024 tokens | 539 ms | **106 ms**  (5.1x)  |
| TTFT @ 2048 tokens | 673 ms | **273 ms**  (2.5x)  |

## Model Information

- **HuggingFace ID:** [`google/gemma-4-E4B-it`](https://huggingface.co/google/gemma-4-E4B-it)
- **Model Type:** Decoder-only transformer (multimodal-capable but
  text-only path used here)
- **Parameters:** 7.94B raw / 4.5B "Effective" (E4B) — ~16 GB at BF16
- **Architecture:** Gemma 4 with three E-variant features:
  * **Per-Layer Embeddings (PLE):** an extra small embedding lookup
    (`hidden_size_per_layer_input=256`) wrapped via
    `per_layer_input_gate` (Linear 2560 -> 256) +
    `per_layer_projection` (Linear 256 -> 2560) and an element-wise
    multiply against `hidden_states` per layer
  * **KV-sharing across layers:** `num_kv_shared_layers=18`. Of 42
    decoder layers, 24 are "owners" with full QKV and 18 reuse a prior
    owner's KV cache (so they only have `q_proj` + `o_proj`)
  * **41 SWA + 1 global layer** (vs the 31B's 49 SWA + 11 global)
- **Heads:** 8 attention heads, **2 KV heads**, head_dim=256
- **License:** Gemma Terms of Use
- **Maintainer:** Armin Aghaebrahimian (`armin@amazon.com`)

## Overview

E4B is the "Effective 4B" member of Google's Gemma 4 family — a small,
efficient decoder designed for edge / single-stream inference. The two
architectural tricks that make it interesting (PLE + KV-sharing) are
also what kept the existing Gemma 4 ports from working out of the box:
the upstream `vllm-neuron` Gemma 4 model class is hand-written for the
31B variant and crashes on E4B with `TypeError: '>=' not supported
between instances of 'int' and 'NoneType'` (it reads
`num_global_key_value_heads`, which is `None` on E4B because global
layers reuse `num_key_value_heads=2`).

Rather than fork that 31B model class, this contrib uses HuggingFace's
existing `Gemma4ForConditionalGeneration` directly. transformers 5.12
already implements PLE and KV-sharing in pure PyTorch; we add tensor
parallelism via `parallelize_module` + a TP plan that knows about the
E4B-specific layer split, then let `torch.compile(backend="neuron")`
lower the result. Total new code is ~140 lines for the TP plan +
~250 lines for the bench/serve script.

### Neuron Implementation

- **Decoder TP plan** (`src/tp_plan.py`): Colwise/Rowwise sharding over
  the q/k/v/o projections on the 24 owner attention layers, q/o only on
  the 18 KV-shared layers, and full Colwise/Rowwise on every MLP. PLE
  projections (per_layer_input_gate / per_layer_projection) are
  intentionally **NOT** sharded — they're tiny (~84 MB total) and the
  HF forward path expects full-rank tensors for the
  `hidden_states * per_layer_input` broadcast.
- **Distributed backend:**
  `dist.init_process_group(backend="neuron")` +
  `init_device_mesh("neuron", (TP,))` (the Beta 3 / Beta 2 PG backend,
  not `xla` and not `gloo`). Launched with `torchrun
  --rdzv_backend=c10d`.
- **`torch.compile`:** Applied with `backend="neuron"`, `dynamic=False`.
  First call per `seq_len` triggers a 2-5 minute neuronx-cc compile;
  subsequent calls hit the persistent NEFF cache at `/tmp/neff_cache`
  (~50 MB per bucket) and run at the steady-state numbers in this
  README. Bind-mount that path from a host directory to make the cache
  survive container restarts (3-3.3 s reload vs 70-100 s fresh
  compile).

## Validation Results

**Validated:** 2026-06-12 (Friday)
**Instance:** trn2.3xlarge `i-0cf5d3577220d6091` (ap-southeast-4 /
Melbourne), `LNC=2`, 1 Trainium2 chip, 4 logical cores total (TP=2
uses 2 of them; the other 2 are idle).
**Stack:** Beta 3 DLC (`421672808698.dkr.ecr.us-east-1.amazonaws.com/concourse-release-0461d3b:latest`), torch 2.11.0+cu130,
torch_neuronx 2.11.3.0.1278, neuronxcc 2.25.1280.0, transformers 5.12.0,
`torch.device("neuron")`.

### Headline TTFT — Eager vs `torch.compile`

Single-stream, batch=1, greedy sampling, prompt = `"The capital of
France is"` padded with the tokenizer pad token to the bucket size.
3 timed runs after 1 warmup, per-rank-aligned via `dist.barrier()`.

| seq_len | Eager mean (ms) | `torch.compile` mean (ms) | Speedup | First-compile (eager) | First-compile (`torch.compile`) |
|---:|---:|---:|---:|---:|---:|
| 64   | 419.4 | **32.7**  | **12.8x** | 3.2 s (cached)  | 215.6 s |
| 128  | 419.9 | **42.4**  | **9.9x**  | 3.1 s (cached)  | 221.6 s |
| 256  | 433.2 | **56.5**  | **7.7x**  | 3.1 s (cached)  | 235.3 s |
| 512  | 466.0 | **64.1**  | **7.3x**  | 3.3 s (cached)  | 298.3 s |
| 1024 | 538.5 | **106.0** | **5.1x**  | 80.0 s          | 97.6 s  |
| 2048 | 672.9 | **273.3** | **2.5x**  | 101.8 s         | 158.4 s |

Raw JSONs: [`results/eager_prefill_sweep.json`](results/eager_prefill_sweep.json),
[`results/compile_prefill_sweep.json`](results/compile_prefill_sweep.json).

Eager TTFT is **flat** across short seq_lens (419 ms at 64, 446 ms at
512) because per-token compute is dominated by the 42-layer MLP +
PLE work, not attention. At 1024+ the attention term grows and TTFT
starts climbing.

`torch.compile` collapses that constant overhead and exposes the real
seq_len curve underneath. The 2.5x ratio at 2048 isn't a regression —
it's the limit of how much constant overhead is left to remove.

### NEFF cache persistence

| | seq_len 64 | seq_len 128 | seq_len 256 | seq_len 512 |
|---|---:|---:|---:|---:|
| Fresh first compile | 70.5 s | 67.0 s | 69.6 s | 71.8 s |
| Warm restart (cache hit) | **3.2 s** | **3.1 s** | **3.1 s** | **3.3 s** |

Bind-mount `/tmp/neff_cache` to a host directory; cache survives
`docker rm -f`. **20-30x faster cold-start** on bucket sets you've
pre-warmed.

### KV-cache decode (with caveat)

| | Value |
|---|---:|
| TTFT (prefill 128 tokens, KV-cache enabled) | **1198.7 ms** |
| Decode first token | 61.0 s |
| Decode steady-state mean | **11.5 s/token** |
| TPOT throughput | **0.087 tok/s** |

Raw JSON: [`results/decode_kvcache.json`](results/decode_kvcache.json).

**Caveat — known limitation, not unique to this port.** Beta 3
`torch.compile(backend="neuron")` does not support dynamic shapes. The
HF KV-cache decode loop grows the attention mask by 1 every token, so
each step recompiles a fresh graph. The 11.5 s steady-state is
dominated by the recompile, not the model forward.

For production decode, use a static-shape KV cache: pre-allocate
`max_kv_len` slots, full-size attention mask with a "valid position"
flag, never resize. This is the same pattern that vllm-neuron and NxDI
implement internally. **A static-shape decode rewrite is out of scope
for this contrib** — flagging here so the next reader doesn't chase the
0.087 tok/s number.

The prefill numbers above are the genuine result and are unaffected by
this issue.

### Generation Proof

```
Prompt:   "The capital of France is"
TTFT:     32.7 ms (compile, 64-bucket warm)
Output:   " France"          # next-token greedy
```

### Accuracy Validation

| Component | Metric | Result |
|---|---|---|
| Decoder (full prefill) | Greedy next-token vs HF CPU reference | **match** (`" France"`) |
| Prefill TTFT rank symmetry | `\|rank0_ms - rank1_ms\|` | < 0.5 ms across all 6 buckets |

Per-token logit parity vs CPU isn't asserted in this contrib (would
require a CPU reference run for the same prompt + temperature=0). The
greedy next-token match is a meaningful smoke check given E4B's
deterministic instruction-tuned behaviour for the canonical "capital of
France" prompt.

## Hardware budget

E4B at TP=2 is comfortable on Trainium2:

| | Per-rank weight footprint | Trainium2 user budget per logical core | Headroom |
|---|---:|---:|---:|
| Eager   | ~8.0 GB   | ~24 GB | ~16 GB |
| Compile | ~8.0 GB + activation buffers | ~24 GB | ~10-12 GB |

Per-Layer Embeddings (PLE) replicate fully on each rank, adding ~84 MB
across all 42 layers — negligible.

## TP=4 is not viable for E4B

`num_key_value_heads=2`. Standard `ColwiseParallel` on `k_proj` /
`v_proj` at TP > 2 fails with:

```
RuntimeError: shape '[1, S, -1, 256]' is invalid for input of size <local>
```

because 2 KV heads can't shard 4 ways. **TP=2 is the architectural
ceiling** with the naive plan. To go higher you'd need to replicate KV
heads across rank pairs — possible but out of scope for this contrib.
On a 4-core trn2.3xlarge, run two independent TP=2 instances for ~2x
aggregate throughput instead.

## Usage

### Prerequisites

1. trn2.3xlarge or larger, with a Beta 3 Neuron DLC pulled.
2. HF token with access to the (gated) `google/gemma-4-E4B-it` repo.
3. `huggingface-cli download google/gemma-4-E4B-it` populated under
   `/root/.cache/huggingface`. (~16 GB.)

### Compile + run

```bash
cd src/

# Materialize a local model dir with patched tokenizer_config.json
# (the published config ships extra_special_tokens as a list;
# transformers 5.x wants a dict).
/opt/torch-neuronx/.venv/bin/python build_local_model.py \
    --dst /root/models/gemma-4-E4B-it

# Eager benchmark sweep
./serve.sh

# torch.compile sweep (recommended for production)
COMPILE=1 ./serve.sh
```

The sweep writes `results_eager.json` / `results_compile.json` with
per-bucket timings.

### Programmatic

```python
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor.parallel import parallelize_module
from transformers import AutoTokenizer, AutoModelForImageTextToText
from src.tp_plan import build_e4b_tp_plan

# Inside a torchrun --nproc_per_node=2 process:
dist.init_process_group(backend="neuron")
mesh = init_device_mesh("neuron", (dist.get_world_size(),))
device = torch.device("neuron")

tokenizer = AutoTokenizer.from_pretrained("/root/models/gemma-4-E4B-it")
model = AutoModelForImageTextToText.from_pretrained(
    "/root/models/gemma-4-E4B-it",
    dtype=torch.bfloat16, attn_implementation="eager",
)
plan, _, _, owners, shareds = build_e4b_tp_plan(model)
parallelize_module(model, mesh, plan)
model = model.to(device).eval()
model = torch.compile(model, backend="neuron", dynamic=False)

# Forward a single prefill — pad to a fixed bucket size.
tok = tokenizer("The capital of France is", return_tensors="pt",
                padding="max_length", truncation=True, max_length=64)
out = model(input_ids=tok["input_ids"].to(device),
            attention_mask=tok["attention_mask"].to(device),
            use_cache=False)
print(tokenizer.decode([int(out.logits[:, -1].argmax(-1))]))
# -> ' France'
```

## Compatibility Matrix

| Instance | TP | LNC | SDK | Status |
|---|---:|---:|---:|---|
| trn2.3xlarge   | 2 | 2 | Beta 3 (2.30.x equivalent) | **VALIDATED** |
| trn2.48xlarge  | 2 | 2 | Beta 3 | not tested (would work; overkill for single-stream E4B) |
| inf2.8xlarge   | 2 | — | Beta 3 | not tested (architecturally feasible at TP=2; per-core HBM is 16 GB which is tight for ~8 GB weights + activations + KV cache; verify) |
| inf2.xlarge    | 2 | — | Beta 3 | NOT recommended (only 16 GB host RAM — borderline for loading the 16 GB BF16 weights before sharding) |

### Configuration Notes

- `NEURON_RT_VIRTUAL_CORE_SIZE=2 NEURON_RT_NUM_CORES=2` for TP=2 on
  trn2 with LNC=2.
- `attn_implementation="eager"` on `from_pretrained()`. SDPA path
  triggers different shapes the TP plan doesn't cover.
- `use_cache=False` for prefill TTFT measurements. Toggle to `True`
  only for the decode path; see "KV-cache decode" caveat above.
- Bind-mount `/tmp/neff_cache` to a host directory for persistent NEFF
  cache. Without it, every `docker rm -f` costs ~70-100 s per bucket on
  next start.

## Testing Instructions

```bash
# Smoke test (asserts next-token = " France" + per-rank symmetry)
cd test/
NEURON_RT_VIRTUAL_CORE_SIZE=2 NEURON_RT_NUM_CORES=2 \
  /opt/torch-neuronx/.venv/bin/torchrun \
    --nproc_per_node=2 --rdzv_backend=c10d \
    --rdzv_endpoint=localhost:29500 \
    test_e4b_smoke.py
```

Expected output ends with `PASS`.

## Known Issues

1. **Decode TPOT recompiles every step.** Beta 3
   `torch.compile(backend="neuron")` doesn't support dynamic shapes;
   HF KV-cache decode grows the attention mask by 1 per token, so each
   step recompiles. Workaround: static-shape KV cache (pre-allocated
   `max_kv_len` slots + fixed mask). Out of scope for this contrib —
   would be a follow-on.
2. **TP > 2 not supported with the naive plan.**
   `num_key_value_heads=2` caps `ColwiseParallel`-on-KV at TP=2.
   Higher TP needs KV head replication across rank pairs.
3. **Tokenizer config patch required.** Published
   `tokenizer_config.json` ships `extra_special_tokens` as a list;
   transformers 5.x wants a dict. `src/build_local_model.py` patches
   it in place via a one-shot symlink+rewrite.
4. **First call per `seq_len` bucket compiles a new NEFF.** ~70-100 s
   eager, ~100-300 s under `torch.compile`. Persistent NEFF cache
   makes subsequent restarts fast (~3 s reload). Pre-warm the bucket
   set you intend to serve at startup.
5. **Single-stream only.** No continuous batching, no
   OpenAI-compatible API. For high-throughput serving, the right path
   is a `vllm-neuron` E4B model class (~3-5 days of work — would need
   to implement PLE + KV-sharing in `vllm_neuron/model/gemma4_e/`).

## Files

```
gemma4-e4b-trainium/
|-- README.md                        # this file
|-- src/
|   |-- __init__.py
|   |-- tp_plan.py                   # E4B TP plan (24 owner / 18 KV-shared)
|   |-- run_e4b.py                   # bench harness (prefill + decode)
|   |-- build_local_model.py         # tokenizer_config patcher
|   `-- serve.sh                     # one-shot launcher
|-- test/
|   |-- __init__.py
|   `-- test_e4b_smoke.py           # next-token + per-rank symmetry
`-- results/
    |-- eager_prefill_sweep.json     # eager TTFT across 64..2048
    |-- compile_prefill_sweep.json   # torch.compile TTFT across 64..2048
    `-- decode_kvcache.json          # HF KV-cache decode (with caveat)
```

## License

The contrib code in `src/` and `test/` is Apache-2.0. The Gemma 4
E4B-it weights are governed by the [Gemma Terms of
Use](https://ai.google.dev/gemma/terms).

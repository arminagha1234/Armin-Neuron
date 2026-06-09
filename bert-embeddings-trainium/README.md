# Contrib Model: BERT Embeddings on Trainium2

Native PyTorch + `torch.compile(backend="neuron")` reference for BERT-family encoder embeddings on AWS Trainium2, plus a vLLM-Neuron variant and a Triton Inference Server wrapper.

## Model Information

- **HuggingFace IDs:**
  - [`sentence-transformers/all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) (22M, 384-dim, BERT-architecture)
  - [`BAAI/bge-base-en-v1.5`](https://huggingface.co/BAAI/bge-base-en-v1.5) (110M, 768-dim, BERT-architecture)
- **Model Type:** Encoder-only transformer (BERT) for sentence embeddings
- **Parameters:** 22M – 110M (BF16)
- **Architecture:** Standard BERT (learned position embeddings, post-LN, mean-pool over real tokens)
- **License:** Apache 2.0 (both checkpoints)
- **Maintainer:** Armin Agha-Ebrahim

## Overview

Reference implementation for serving BERT-style encoder embeddings on Trainium2 via three serving paths, with reproducible benchmarks and correctness checks for each. The custom `BertEncoder` matches HuggingFace BERT numerically (cosine ≥ 0.9999 vs CPU reference) and is compatible with both `torch.compile(backend="neuron")` and the vllm-neuron private-beta runner.

The three paths in this contrib:

1. **Native PyTorch + torch.compile** (`src/`, `test/`) — fastest. 7,150 seq/s @ N=512, 1.10 ms p50 on a single trn2.3xlarge logical core.
2. **Triton Inference Server + native+compile** (`triton/`) — deployable HTTP/gRPC wrapper. 2,104 seq/s through the full Triton serving stack.
3. **vLLM-Neuron + custom `BertModel` class** (`vllm_path/`) — for users committed to vLLM-on-Triton today. Requires three small in-image patches to the `vllm-neuron` runner to wire up the pooling output path; full patcher script included. 590 seq/s.

### Neuron Implementation

- **Custom `BertEncoder`** (`src/native_bert_model.py`) — bidirectional attention with explicit `matmul → softmax → matmul`, masked mean-pool, additive attention mask from `(input_ids != PAD)`. Loads HF `bert.*` weight keys with prefix-stripping fallback for `sentence-transformers` checkpoints.
- **Native + torch.compile** — uses `torch.device("neuron")` and compiles per static batch size (Beta 3 doesn't support dynamic shapes in `torch.compile`).
- **vLLM-Neuron path** — adds a `BertModel` Neuron class that satisfies the runner's `from_configs` contract and hand-attaches `vllm.model_executor.layers.pooler.DispatchPooler.for_embedding` so the engine's pooling task is recognized. A 3-edit patch to `neuron_model_runner.py` adds `is_pooling_model` derivation, advertises `embed`/`encode` in `get_supported_tasks`, and returns a `pooler_output`-bearing `ModelRunnerOutput` from `execute_model` when pooling is active. Full patcher with `.orig` backup at `vllm_path/patch_runner_inimage.py`.
- **NKI fused attention** (`src/native_nki_attention.py`, `src/NKI_KERNEL_NOTES.md`) — partial fused-attention kernel with notes on 8 NKI ISA gotchas hit during bring-up. Compiles up through the softmax broadcast step. Documented for future iteration; **not** required by the recommended path.

## Validation Results

**Validated:** 2026-06-08
**Instance:** trn2.3xlarge (LNC=2, single logical core / 2 physical cores active)
**Containers:**
- Native path: Beta 3 native DLC (`torch_neuronx 2.11.3.0.1278+5013c208`, `nki 0.4.0+25940409122`, `neuronx-cc 2.25.3371.0+f524f7f8`)
- vLLM path: vllm-neuron private-beta DLC (`vllm-neuron 0.19.0.0`, torch 2.10, transformers 4.57.6)

### Benchmark Results — single trn2.3xlarge logical core, MiniLM-L6, seq_len=128, bf16

| Path | Throughput @ N=128 | Throughput @ N=512 | Latency P50 | Latency P99 | Cosine vs HF |
|---|---|---|---|---|---|
| vLLM-Neuron (Path A) | 590.9 seq/s | 585.6 seq/s | 3.36 ms | 3.93 ms | 0.99990 |
| vLLM-Neuron + DP=2 | 1,074.9 seq/s | — | — | — | 0.99990 |
| **Native + torch.compile** | **2,675 seq/s** | **7,150 seq/s** | **1.10 ms** | **1.20 ms** | **0.99991** |
| **Native + torch.compile + DP=2** | — | **14,375 seq/s** | (per-worker ~1.10) | — | 0.99991 |
| Triton Server + native+compile | 2,104 seq/s | (TBD) | 4.15 ms (warm) | (TBD) | 0.99988 |

Native + compile is **12× faster on throughput** and **3× faster on latency** than vLLM-Neuron at the same shape, with numerically equivalent embeddings.

### bge-base (110M) for reference

| Path | Throughput @ N=128 | Latency P50 | Cosine vs HF |
|---|---|---|---|
| Native eager (no compile) | 849 seq/s | 35.3 ms | (matches HF) |
| **Native + torch.compile** | **1,985 seq/s** | **2.67 ms** | **0.99994** |

bge-base + native+compile beats Path A's MiniLM serving on both throughput and latency.

### Sequence length sweep — native + torch.compile

| Model | Seq=128 | Seq=256 | Seq=512 |
|---|---|---|---|
| MiniLM-L6 | 2,684 seq/s / 1.21 ms | 2,662 seq/s / 1.17 ms | 1,038 seq/s / 1.68 ms |
| bge-base | 1,990 seq/s / 2.83 ms | 449 seq/s / 43.1 ms ⚠ | 182 seq/s / 37.7 ms ⚠ |

MiniLM scales gracefully through 512. **bge-base hits a graph-compile regime change at ≥256** (4.4× throughput drop, 15× latency increase) — the `[256×256]×12-head` score matrix appears to cross a PSUM tile threshold and the compiler falls back to a much-slower tiled implementation. Recommendation: chunk to ≤128 for bge-base.

### Accuracy Validation

| Component | Metric | Result |
|---|---|---|
| Native + compile (MiniLM) | Cosine vs HF reference, masked-mean | **0.99991** |
| Native + compile (bge-base) | Cosine vs HF reference, masked-mean | **0.99994** |
| vLLM-Neuron (MiniLM) | Cosine vs HF reference, masked-mean | **0.99990** |
| Triton Server (MiniLM, batch consistency) | Cosine(prompt[0] in N=3 vs prompt[0] in N=128) | **0.99988** |

The remaining 0.01% gap is bf16 vs fp32 precision. No path leaks numerical error.

## Usage

### Prerequisites

```bash
# Beta 3 native DLC (Trainium native PyTorch)
docker pull 421672808698.dkr.ecr.us-east-1.amazonaws.com/concourse-release-0461d3b:latest

# Inside the container, install dependencies (already present in Beta 3):
pip install transformers safetensors huggingface_hub
```

### Native + torch.compile (recommended)

```python
import torch
import torch_neuronx
from transformers import AutoModel, AutoTokenizer
from src.native_bert_model import BertEncoder, load_from_hf

MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MAX_LEN = 128
dtype = torch.bfloat16
dev = torch.device("neuron")

tok = AutoTokenizer.from_pretrained(MODEL)
hf = AutoModel.from_pretrained(MODEL, return_dict=False).eval()
ours = BertEncoder(hf.config, dtype=dtype).eval()
load_from_hf(hf, ours)
ours = ours.to(dtype).to(dev)

# Compile per static batch size (Beta 3 requires dynamic=False)
m = torch.compile(ours, backend="neuron", dynamic=False)

# Tokenize and forward
prompts = ["fast embeddings on Trainium", "second sentence"]
enc = tok(prompts, return_tensors="pt", padding="max_length",
          max_length=MAX_LEN, truncation=True)
ids = enc["input_ids"].to(dev)
am = enc["attention_mask"].to(dev).to(dtype)
pos = torch.arange(MAX_LEN, device=dev).unsqueeze(0).expand_as(ids)

with torch.no_grad():
    emb = m(ids, pos, am)            # [B, hidden]
    torch_neuronx.synchronize()
print(emb.shape, emb[0, :5].cpu().tolist())
```

### Triton Inference Server wrapper

```bash
# Drop the model_repo into a Triton container that has Neuron runtime + torch_neuronx
BERT_MODEL=sentence-transformers/all-MiniLM-L6-v2 \
BERT_MAX_LEN=128 \
BERT_BUCKETS=1,8,32,128,512 \
tritonserver --model-repository=/path/to/triton/model_repo
```

Client (HTTP):
```python
import numpy as np
import tritonclient.http as httpclient

c = httpclient.InferenceServerClient(url="localhost:8000")
prompts = ["fast embeddings", "second"]
inp = httpclient.InferInput("PROMPTS", [len(prompts)], "BYTES")
inp.set_data_from_numpy(np.array([p.encode("utf-8") for p in prompts], dtype=object))
result = c.infer(model_name="bert_embed", inputs=[inp])
emb = result.as_numpy("EMBEDDING")     # [N, 384]
```

### vLLM-Neuron path (requires private-beta DLC + in-image runner patch)

Inside the `vllm-neuron` container:
```bash
# 1. Apply the runner patches (writes neuron_model_runner.py.orig backup)
python3 vllm_path/patch_runner_inimage.py

# 2. Run with our custom BertModel class
python3 vllm_path/run_bert_vllm.py
```

The patcher modifies `/opt/conda/lib/python3.12/site-packages/vllm_neuron/vllm/worker/neuron_model_runner.py` in three places:
- `__init__`: derive `is_pooling_model` from `runner_type=="pooling"` / `convert_type=="embed"` instead of hardcoded `False`
- `get_supported_tasks()`: return `("encode", "embed")` when pooling
- `execute_model()`: when pooling, build a `ModelRunnerOutput` with `pooler_output` from the model output and return it (vLLM core consumes `execute_model`'s return value directly for pooling — `sample_tokens` is never called)

## Compatibility Matrix

| Instance | TP | SDK | Path | Status |
|---|---|---|---|---|
| trn2.3xlarge (LNC=2) | 1 | Beta 3 native | Native + torch.compile | VALIDATED |
| trn2.3xlarge (LNC=2) | 1 | Beta 3 native | Triton + native+compile | VALIDATED |
| trn2.3xlarge (LNC=2) | 1 | vllm-neuron private beta | vLLM-Neuron + custom class | VALIDATED |
| trn2.48xlarge | 1×16 (DP) | Beta 3 native | Native + torch.compile (16 instances) | Estimated only |

### Configuration Notes

- **Static shapes required.** Beta 3 `torch.compile(backend="neuron")` doesn't support `dynamic=True`. Compile per batch size and route requests to the nearest bucket. Default buckets: `{1, 8, 32, 128, 512}`.
- **PAD token handling.** The model's masked mean-pool keys off `(input_ids != 0)`. For checkpoints where `pad_token_id != 0`, swap to `(input_ids != tokenizer.pad_token_id)` in `BertEncoder.forward`.
- **DP=2 on trn2.3xlarge** uses `NEURON_RT_VISIBLE_CORES=0-1` for worker A and `=2-3` for worker B (`NEURON_VISIBLE_DEVICES` doesn't fully work because the chip has only one device under LNC=2).
- **vLLM-Neuron position-padding pitfall.** vllm-neuron pads `positions` by repeating the last real position (not `arange`). The custom `BertEncoder` uses the runner-supplied positions and applies an attention mask, which gives bit-correct embeddings under that convention. Diverging from this (using `arange` or skipping the mask) drops cosine similarity to ~0.5.

## Testing Instructions

```bash
# Inside the Beta 3 native container with /workspace/contrib mounted:
cd /workspace/contrib/bert-embeddings-trainium

# 1. Correctness check (cosine vs HF, MiniLM)
python3 test/check_correctness.py

# 2. Benchmark (throughput sweep + latency P50/P99)
USE_COMPILE=1 python3 test/bench_native.py

# 3. DP=2 throughput
bash test/dp2_run.sh

# 4. Sequence length sweep
python3 test/seq_sweep.py
```

For the vLLM-Neuron path inside the `vllm-neuron` container:
```bash
cd /workspace/contrib/bert-embeddings-trainium/vllm_path
python3 patch_runner_inimage.py     # applies the 3 runner edits with .orig backup
python3 run_bert_vllm.py             # smoke-tests llm.embed() through vLLM
```

For the Triton wrapper standalone validation:
```bash
cd /workspace/contrib/bert-embeddings-trainium/triton
python3 standalone_test.py           # mocks tritonserver and exercises model.py
```

## Known Issues

1. **bge-base graph-compile cliff at seq ≥ 256.** Throughput drops 4.4× and latency jumps 15×. Workaround: chunk documents to ≤ 128 tokens before embedding, or use a smaller model (MiniLM holds up through 512).
2. **vLLM-Neuron container lacks `torch_neuronx.nki_hop`.** Customer-written NKI kernels cannot be dispatched from a vLLM-served model. Verified: blocked by torch C++ ABI mismatch + the public Neuron pip channel not yet shipping `torch_neuronx 2.11.x` with `nki_hop`. Full reproduction and root-cause analysis in [`vllm_path/NKI_VLLM_BLOCKER.md`](vllm_path/NKI_VLLM_BLOCKER.md). Workaround: native-DLC + Triton Python backend.
3. **vLLM-Neuron runner patches are throwaway.** They edit shipped beta image internals and won't survive an image update. Long-term fix is the upstream `vllm-neuron` runner adding pooling/embed support natively (see the rationale comments in `vllm_path/patch_runner_inimage.py`).
4. **Cold-start compile time on Beta 3.** Compiling 5 buckets {1, 8, 32, 128, 512} for MiniLM takes ~3 minutes the first time. Cache hits make subsequent boots ~20 s. Persist the NEFF cache directory (`/root/.cache/vllm/neuron/compile_cache/`) on attached storage to avoid recompiles across restarts.
5. **`nl.softmax`, `nisa.tensor_tensor`'s `subtract`/`divide`, `nisa.nc_transpose`'s engine selection** — eight NKI ISA gotchas documented in `src/NKI_KERNEL_NOTES.md` for anyone attempting fused-attention kernels.

## Recommendation

For most embedding workloads on Trainium today: **use Native + `torch.compile(backend="neuron")`**, wrap with Triton's Python backend if you want HTTP/gRPC. Reserve vLLM-Neuron for autoregressive-generation models where its scheduler / paged-KV-cache machinery actually pays off — it's not the right tool for encoders.

---

## PR Checklist

- [x] `src/` — Native BERT encoder + (partial) NKI attention kernel + ISA gotcha notes
- [x] `test/` — Correctness check, benchmark, DP=2 runner, sequence-length sweep
- [x] `triton/` — Triton Inference Server Python backend wrapper + standalone test
- [x] `vllm_path/` — vLLM-Neuron BERT class + runner-patcher script
- [x] `README.md` — This file with benchmark results
- [x] Benchmark JSONs in `benchmarks/`
- [x] Tested on target instance (trn2.3xlarge)
- [x] No hardcoded paths
- [x] License compatible (Apache 2.0)

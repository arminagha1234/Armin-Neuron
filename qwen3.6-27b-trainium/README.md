# Contrib Model: Qwen3.6-27B on AWS Trainium2 (vLLM-Neuron)

Serving Alibaba's **Qwen3.6-27B** dense coding model (hybrid GatedDeltaNet +
GQA attention) via `vllm serve` on AWS Trainium2, with a drop-in
`vllm_neuron` model plugin — no fork of the vllm-neuron container required.

## Model Information

- **HuggingFace ID:** [`Qwen/Qwen3.6-27B`](https://huggingface.co/Qwen/Qwen3.6-27B)
- **Model Type:** Hybrid decoder (linear-attention DeltaNet + full-attention GQA), multimodal-capable (served text-only here)
- **Parameters:** ~27B (BF16, ~54 GB)
- **Architecture:** 64 layers in a `[3 × DeltaNet + 1 × GQA]` repeating block; hidden 5120; 24 Q heads / 4 KV heads; head_dim 256; partial RoPE (25%); SwiGLU MLP (intermediate 17408); DeltaNet with 48 value / 16 key heads; per-layer QK-norm; attention output gate; `(1 + weight)` RMSNorm convention
- **HF arch class:** `Qwen3_5ForConditionalGeneration` (Alibaba kept the class name across the 3.5 → 3.6 bump)
- **License:** Apache 2.0
- **Maintainer:** Armin Agha-Ebrahim

## Overview

Qwen3.6-27B is a dense, Apache-2.0 coding model that Alibaba reports
out-performing its own 397B-A17B MoE on several coding benchmarks. Its
architecture is in the Qwen3.5 / Qwen3-Next family: a hybrid that
interleaves linear-attention (GatedDeltaNet) layers with periodic
full-attention (GQA) layers.

vLLM-Neuron does not support this architecture out of the box. This
contribution is a self-contained `vllm_neuron` model package that
registers the architecture into both vLLM's and vllm-neuron's model
registries at import time (via a `sitecustomize.py` + post-plugin hook),
so the standard `vllm serve` pipeline traces, compiles, and serves the
model on Trainium2.

### Neuron Implementation

- **Full-attention (GQA) layers** (16 of 64): standard vllm-neuron
  primitives — `NF.qkv_proj`, `NF.flash_attention`, `NF.o_proj` — plus
  Qwen3.6-specific partial RoPE, per-head QK-norm, and the attention
  output gate (spliced into `q_proj`'s per-head second half).
- **Linear-attention (DeltaNet) layers** (48 of 64): the fused chunked
  GatedDeltaNet NKI kernel, wrapped through
  `vllm_neuron.nki.nki_hop.wrap_nki`. State carried in side-channel
  buffers (not the paged KV cache).
- **Decode path:** Gemma-style split-K flash attention for `head_dim=256`
  (the fused decode megakernel caps head_dim at 128; split-K does two
  128-wide matmuls accumulated in PSUM).
- **Sampling / embedding / lm_head:** vllm-neuron `Sampler`,
  `VocabDimShardedEmbedding`, `ColumnParallelLinear`.

## Validation Results

**Validated:** 2026-06-10
**Instance:** trn2.48xlarge (LNC=2)
**SDK:** Neuron driver 2.28, tools 2.30, vllm-neuron private beta v5

### Correctness

Numerical parity verified against the HuggingFace reference
(transformers 5.10.2, CPU, BF16) on a per-layer basis:

| Check | Result |
|---|---|
| GQA layer (layer 3) output vs HF, per-token cosine | **1.000000** (max abs err 0.0039) |
| HF reference top-1 for "The capital of France is" | `" Paris"` (logit 15.6) |
| On-device top-1 for "The capital of France is" | `" Paris"` ✅ |
| On-device "…quick brown fox jumps over the lazy" | `" dog"` ✅ |

### Serving

| Property | Value |
|---|---|
| Tensor parallel | TP=8 (also boots at TP=4) |
| Per-core HBM | 15.34 GiB / 24 GiB (8.66 GiB headroom) |
| KV cache @ 4K ctx | 406,000 tokens (~99× concurrency) |
| Weight load time | ~14 s (rank 0), ~5 s (others) |
| Cold compile | ~40 min (6 prefill buckets + decode, 64 layers); cached restarts near-instant |

> Throughput / TTFT / $-per-M-token benchmarks are a follow-up
> (neuronx-benchmark-tool) and will be appended here.

## The key portability fix (read this if you port a sibling model)

Qwen3.6 uses the **`(1 + weight)` RMSNorm convention** (Gemma-style):
weights are stored centered at 0 and the runtime applies
`x_normed * (1.0 + weight)`, NOT the standard `x_normed * weight`. Using
the standard form makes every RMSNorm in the network (input,
post-attention, q-norm, k-norm, final — ×64 layers) wrong by the +1
offset, which uniformly distorts activations and produces degenerate
token loops while still "running." This was THE correctness bug; once
fixed, per-layer cosine vs HF jumped from 0.97 → 1.000000.

Other Qwen3.6-specific details handled in this package:

- `q_proj` ships at `(num_heads × head_dim × 2, hidden)` with the
  attention **output gate interleaved per head** (`[h0_q | h0_gate | h1_q | …]`).
  Loaders split it accordingly; the gate is applied as
  `attn_out * sigmoid(gate)`.
- `tie_word_embeddings = False` — `lm_head.weight` is its own tensor.
- `mtp.*` (multi-token-prediction head) and `model.visual.*` (vision
  tower) are present in the checkpoint but skipped for text-only serving.
- Partial RoPE rotates 25% of head_dim with **NeoX `rotate_half`**
  (the `mrope_interleaved` flag affects only frequency assembly across
  the 3 mRoPE sections, which collapse to standard RoPE for text input).

## Usage

### Prerequisites

```bash
# 1. Download weights
hf download Qwen/Qwen3.6-27B --local-dir /root/models/Qwen3.6-27B

# 2. Pull the vllm-neuron beta container (v5), then start it with this
#    package's src/ mounted on PYTHONPATH (see Serve below).
```

### Serve

```bash
docker run -d --name vllm_neuron --network host \
  $(for i in $(seq 0 15); do echo --device=/dev/neuron$i; done) \
  -v /root/models:/root/models \
  -v $(pwd)/src:/workspace/qwen36_adapter \
  -e PYTHONPATH=/workspace/qwen36_adapter \
  -e NEURON_SKIP_EFA_AFFINITY=1 \
  -w /workspace/qwen36_adapter "$IMAGE" sleep infinity

docker exec -d vllm_neuron bash -c \
  "cd /workspace/qwen36_adapter && TP=8 MAX_LEN=4096 MODEL=/root/models/Qwen3.6-27B ./serve.sh"
```

### Query

```bash
curl -s http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"/root/models/Qwen3.6-27B","prompt":"The capital of France is","max_tokens":20,"temperature":0}'
```

## Compatibility Matrix

| Instance | TP | SDK | Status |
|----------|-----|-----|--------|
| trn2.48xlarge (LNC=2) | 8 | vllm-neuron beta v5 | VALIDATED (correctness) |
| trn2.48xlarge (LNC=2) | 4 | vllm-neuron beta v5 | Boots; correctness same |

### Configuration Notes

- `head_dim=256` exceeds the fused decode megakernel's 128 transpose
  limit → decode uses split-K (two 128-wide matmuls). Handled in the package.
- TP must divide 24 (Q heads): valid TP ∈ {1,2,3,4,6,8,12,24}. TP=8 is
  the recommended default on trn2.48xl.
- For long context use chunked prefill: on the v5 build,
  `num_batched_tokens_buckets` must equal `kv_segment_size_buckets` and
  be a single value from {512,1024,2048,4096} — use `BUCKET=4096`.

## Testing Instructions

```bash
export PYTHONPATH=$(pwd)/src
# Registry + config + weight-mapping coverage (CPU, no device needed):
python -m qwen3_6.test.test_phase1_skeleton
python -m qwen3_6.test.test_paris_smoke

# Per-layer numerical parity vs HF (needs a transformers>=5.10 venv):
python test/parity_layer3.py
```

## Known Issues

1. **Cold compile is ~40 min** for the full 6-bucket / 64-layer build.
   NEFF cache persists, so restarts are fast. Use `MAX_LEN=256` for quick
   correctness iteration.
2. **MTP speculative decoding not wired** — the `mtp.*` head is skipped.
3. **Vision tower skipped** — text-only serving.

---

## PR Checklist

- [x] `src/` — Model implementation (`qwen3_6/` package)
- [x] `test/` — Smoke + parity tests
- [x] `README.md` — This file with validation results
- [x] Tested on target instance (trn2.48xlarge)
- [x] No hardcoded paths (env-var driven launcher)
- [x] License compatible (Apache 2.0)

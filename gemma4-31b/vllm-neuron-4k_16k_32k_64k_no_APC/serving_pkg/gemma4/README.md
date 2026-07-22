# Gemma4 31B

BF16 text decoder for Google's Gemma4 31B multimodal model. Only the text
decoder is implemented; vision encoder weights are skipped during loading.

## Architecture

| Parameter                    | Value                          |
|------------------------------|--------------------------------|
| hidden_size                  | 5376                           |
| num_hidden_layers            | 60                             |
| num_attention_heads          | 32                             |
| vocab_size                   | 262144                         |
| intermediate_size            | 21504                          |
| rms_norm_eps                 | 1e-6                           |
| max_position_embeddings      | 262144                         |
| tie_word_embeddings          | True                           |
| final_logit_softcapping      | 30.0                           |
| Activation                   | GeGLU (gelu_pytorch_tanh)      |
| Embedding scaling            | multiply by sqrt(hidden_size)  |

### Heterogeneous Layer Design

Gemma4 uses two layer types with different head dimensions and KV head counts:

| Property              | SWA (Sliding Window)   | Global (Full Attention) |
|-----------------------|------------------------|-------------------------|
| head_dim              | 256                    | 512                     |
| num_kv_heads          | 16                     | 4                       |
| RoPE theta            | 10,000                 | 1,000,000               |
| partial_rotary_factor | 1.0 (full rotation)    | 0.25 (partial rotation) |
| attention_k_eq_v      | No (separate V proj)   | Yes (V copies K)        |
| sliding_window        | 1024                   | None (full context)     |

Layer pattern (60 layers): mostly SWA with global layers at indices 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55.

### Key Architecture Features

- **QK normalization** — RMSNorm (with learnable scale) applied to Q and K after projection
- **V normalization** — RMSNorm (without learnable scale) applied to V
- **layer_scalar** — learned per-layer multiplicative factor on residual
- **4 norms per layer** — input_layernorm, post_attention_layernorm, pre_feedforward_layernorm, post_feedforward_layernorm
- **Logit softcapping** — `tanh(logits / 30.0) * 30.0` before final softmax
- **No 1/sqrt(head_dim) scaling** — attention uses `scaling = 1.0` (QK norm handles it)

## Running the Model

### Prerequisites

- Trainium2 instance (trn2.48xlarge or trn2p.48xlarge)
- Model weights downloaded locally (e.g., from HuggingFace)
- All dependencies installed (neuronx-cc, libtorch-neuronx-lite, etc.)
- **transformers gemma4 stub** (required until transformers >= 5.0 ships native support)

### Install transformers gemma4 stub

The `gemma4` model type is not yet in the public transformers release (4.x).
A compatibility stub and tokenizer patch must be installed:

```bash
cd examples/vllm_neuron/models/gemma4
./install_transformers_stub.sh /path/to/your/venv /path/to/gemma-4-31b-it
```

This script does two things:
1. Installs `gemma4` config/processor stubs into `transformers/models/gemma4/`
2. Patches `tokenizer_config.json` — the checkpoint ships `extra_special_tokens`
   as a list, but transformers 4.x expects a dict (fixed in transformers 5.x)

Or manually:

```bash
SITE_PACKAGES=$(python3 -c "import site; print(site.getsitepackages()[0])")
cp -r examples/vllm_neuron/models/gemma4/transformers_stub/ \
    "$SITE_PACKAGES/transformers/models/gemma4/"
python examples/vllm_neuron/models/gemma4/patch_tokenizer_config.py /path/to/gemma-4-31b-it
```

Verify: `python3 -c "from transformers.models.gemma4.configuration_gemma4 import Gemma4Config; print('OK')"`

### Quick Start

```bash
python examples/vllm_neuron/models/gemma4/run.py \
    --model-checkpoint /path/to/gemma-4-31b-it \
    --tp-size 8 \
    --max-model-len 4096 \
    --max-num-seqs 4
```

### Online Serving

```bash
vllm serve /path/to/gemma-4-31b-it \
    --tensor-parallel-size 8 \
    --max-model-len 4096 \
    --max-num-seqs 4 \
    --additional-config '{"neuron_config": {"quantization": "bf16", "on_device_sampling_config": {"all_greedy": true}, "num_batched_tokens_buckets": [4096], "num_seqs_buckets": [4]}}'
```

### Configuration Options

| Parameter          | Recommended      | Notes                                        |
|--------------------|------------------|----------------------------------------------|
| tensor_parallel    | 8 or 16          | 8 for single-chip, 16 for multi-chip         |
| max_model_len      | 4096–8192        | Higher values need larger CC buffer           |
| max_num_seqs       | 1–8              | Decode batch size                             |
| quantization       | bf16             | Only BF16 supported currently                 |

For sequence lengths > 4096, set the CC buffer environment variable:

```bash
export NEURON_RT_DBG_INTRA_RDH_CHANNEL_BUFFER_SIZE=$((SEQ_LEN * 5376 * 2))
```

## Module Structure

```text
vllm_neuron/model/gemma4/
├── __init__.py          # Exports Gemma4ForConditionalGeneration
├── README.md            # This file
├── config.py            # Gemma4Config dataclass with per-layer accessors
├── factory.py           # Factory class registered with vLLM ModelRegistry
└── model.py             # Full model: attention, MLP, decoder, weight loading
```

## Test Structure

```text
test/nxdi/model/gemma4/
├── __init__.py
├── test_config.py                    # Unit tests for Gemma4Config
├── test_factory.py                   # Unit tests for factory registration
├── bf16/
│   ├── e2e/
│   │   └── test_logits.py           # Full logit validation (online/offline)
│   └── modules/
│       ├── test_attention.py         # Attention component tests
│       └── test_rope.py             # RoPE component tests
└── equivalence/
    ├── EQUIVALENCE_REPORT.md         # Results summary
    ├── run_all_tests.py              # 19-test validation suite
    ├── test_rope_equivalence.py      # RoPE inv_freq, cos/sin, SWA vs global
    ├── test_rope_with_module.py      # RoPE module forward pass
    ├── test_attention_components.py  # RMSNorm, VNorm, QK norm, K=V, full attn
    ├── test_decoder_layer.py         # Full decoder layer (SWA + global)
    ├── test_model_e2e.py             # Softcapping, embed scaling, 2-layer E2E
    └── test_embed_scale_bf16.py      # BF16 embedding scale precision
```

## Feature Status

| Feature                    | Status | Notes                                              |
|----------------------------|--------|----------------------------------------------------|
| TP (head sharding)         | ✅     | Attention heads + MLP intermediate sharded         |
| SP (sequence parallel)     | ✅     | All-gather/reduce-scatter on prefill               |
| Segmented prefill          | ✅     | Sliding window attention with segment boundaries   |
| On-device sampling         | ✅     | Greedy and top-k/top-p                             |
| Logit softcapping          | ✅     | tanh(x/30)*30 applied before sampling              |
| Heterogeneous KV cache     | ✅     | Per-layer head_dim via LayerSpec                   |
| torch.compile              | ✅     | Full-graph compilation                             |
| Tied embeddings            | ✅     | lm_head shares embed_tokens weights                |
| BF16 inference             | ✅     | All ops in bfloat16                                |
| Equivalence validated      | ✅     | 19/19 tests passing (< 1e-4 max abs diff)          |
| On-device smoke test       | ✅     | Validated on trn2 with TP=8, 4-layer subset        |
| MXFP4 quantization         | ❌     | Not yet implemented                                |
| Vision encoder             | ❌     | Text decoder only (vision weights skipped)         |
| FP8 KV cache               | ❌     | Not yet implemented                                |

## Weight Loading

Gemma4 checkpoint keys are prefixed with `model.language_model.`. The weight
loader handles:

- Stripping the `model.language_model.` prefix
- Fusing Q/K/V into a single QKV tensor (with KV replication for global layers)
- Copying K weights to V for global layers (attention_k_eq_v)
- Loading QK norm weights (q_norm, k_norm)
- Loading per-layer layer_scalar
- Tying lm_head from embed_tokens (tie_word_embeddings)
- Skipping all vision encoder weights (prefixed with `model.vision_tower.`)

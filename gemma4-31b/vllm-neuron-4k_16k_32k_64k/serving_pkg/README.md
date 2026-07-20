# serving_pkg — Gemma4-31B support for vLLM-Neuron

The vLLM-Neuron **Beta** (SDK 2.30) officially serves only Llama3 and GPT-OSS. This package adds
**Gemma4-31B** (`Gemma4ForConditionalGeneration`) support **without forking vLLM** — you just put this
directory on `PYTHONPATH` and `vllm serve google/gemma-4-31B-it` works. `launch_serve.sh` does that for
you automatically (it points `PYTHONPATH` at this folder), so there are no manual steps.

## How registration works (no vLLM fork)

`vllm serve` is its own entrypoint — you can't pass it an "import this module first" flag. Python,
however, **auto-imports a module named `sitecustomize`** at interpreter startup if it's importable. So
dropping `sitecustomize.py` on `PYTHONPATH` makes our registration run inside the vLLM server process
**and every worker subprocess**, with zero changes to vLLM itself.

At startup, `sitecustomize.py` runs three steps:

1. **Teach `transformers` about Gemma4** — `gemma4_transformers_stub.install()` registers the
   `gemma4` / `gemma4_text` config types with `AutoConfig`. vLLM calls `AutoConfig.from_pretrained(...)`
   **before** it ever reaches our model, and stock `transformers` doesn't know `model_type: gemma4`, so
   this must happen first. (It's a config stub only — the actual model is built from our `gemma4/`
   package.)

2. **Register the model** — `gemma4_register.register()` force-inserts
   `Gemma4ForConditionalGeneration` (→ `gemma4.factory.Gemma4ForConditionalGeneration`) into both
   `vllm.model_executor.models.registry.ModelRegistry` and the `vllm_neuron` registry, marked as a
   text-generation model. `install_post_plugin_hook()` wraps vLLM's plugin loader so registration is
   **re-applied after** the `vllm_neuron` plugin loads (the plugin otherwise resets the registry).

3. **(Optional) Path B patches** — if `GEMMA4_APPLY_PATHB=1`, `patch_pathB.apply_pathB_patches()`
   monkey-patches TTFT-optimization hooks onto the loaded model. These are gated behind sub-flags and
   **default OFF** (see below). None of them block startup — every step is wrapped in try/except.

## Contents

```
serving_pkg/
├── sitecustomize.py             # auto-imported entrypoint; runs the 3 steps above
├── gemma4_transformers_stub.py  # AutoConfig stub for model_type gemma4 / gemma4_text
├── gemma4_register.py           # registers Gemma4ForConditionalGeneration into vLLM + vllm_neuron
├── patch_pathB.py               # optional TTFT-optimization patches (gated, default off)
└── gemma4/                      # the Gemma4-31B model implementation
    ├── __init__.py
    ├── config.py                # Gemma4Config.from_configs(hf_config, ...)
    ├── factory.py               # Gemma4ForConditionalGeneration entry class
    ├── model.py                 # the BF16 text decoder (60 layers, SWA + global attention)
    ├── flash_attn_hd256_nki.py  # split-K attention reference for head_dim=256
    ├── fused_embed_scale.py     # embedding * sqrt(hidden_size)
    ├── fused_geglu.py           # GeGLU (gelu_pytorch_tanh) MLP
    ├── fused_logit_softcap.py   # final logit softcap (tanh(logits/30)*30)
    ├── fused_norm_residual.py   # RMSNorm + residual
    ├── fused_qk_norm_rope.py    # QK-norm + RoPE
    └── optimized_forward.py     # optimized forward hooks
```

## Environment flags

| Flag | Default | Effect |
|---|---|---|
| `GEMMA4_APPLY_PATHB` | unset | `1` = apply Path B TTFT-optimization patches (below); else stock model |
| `GEMMA4_PB_FLASH` | `0` | (Path B) route SWA head_dim=256 prefill attention through the split-K path |
| `GEMMA4_PB_SOFTCAP` | `0` | (Path B) fused lm-head + logit softcap |

**Honesty note:** the Path B fused-kernel hooks are currently conservative pass-throughs/placeholders
kept OFF by default — the validated, correctness-first stock model is what serves unless you explicitly
opt in and re-measure. Turn a `GEMMA4_PB_*` flag on, benchmark, and keep only measured wins.

## Manual use (if not using launch_serve.sh)

```bash
export PYTHONPATH=/path/to/serving_pkg:$PYTHONPATH
vllm serve google/gemma-4-31B-it --served-model-name gemma4 --tensor-parallel-size 32 ...
```

Verify registration standalone: `python -m gemma4_register` (prints the registered architectures and
asserts `Gemma4ForConditionalGeneration` is present).

## Model architecture (summary)

Gemma4-31B BF16 text decoder: 60 layers, hidden 5376, 32 Q heads, vocab 262144, GeGLU MLP, tied
embeddings, final logit softcap 30.0. Heterogeneous layers — mostly **SWA** (head_dim 256, 16 KV heads,
sliding window 1024, full RoPE) with **global** layers (head_dim 512, 4 KV heads, full context, partial
RoPE 0.25) at every 5th index. QK-norm (learnable) + V-norm (no scale); attention `scaling = 1.0`.

# serving_pkg — Gemma4-31B on vLLM-Neuron

This package is what makes `vllm serve` recognize and correctly run
**Gemma4-31B** on the vLLM-Neuron beta. The base beta (SDK 2.30) officially
supports only Llama3 + GPT-OSS; Gemma4 comes entirely from the files here.

You do not import anything by hand. `launch_serve.sh` puts this directory on
`PYTHONPATH`, and Python auto-imports `sitecustomize.py` at interpreter startup
inside the `vllm serve` process (and every worker). That one hook wires up
everything below.

## What it does (3 things, in order)

1. **Deploys the patched segmented-attention kernel.**
   `deploy_segmented_cte.py` copies the bundled `attention_segmented_cte.py`
   over the container's installed
   `vllm_neuron/functional/attention/attention_segmented_cte.py`. This is
   required (a `PYTHONPATH` shadow is not enough — `vllm_neuron` imports the
   module by its installed path). The patched copy adds:
   - **edit A** — routes `head_dim > 128` (Gemma4 is 256 for sliding-window
     layers, 512 for global layers) to a trace-safe torch fallback instead of
     raising. This is what enables **chunked / segmented prefill above 16K**.
   - **SWA windowed gather** — for the 49/60 sliding-window layers, gathers a
     static number of KV blocks at a dynamic offset instead of the full padded
     span. Pure PyTorch, and the source of the ~1.9× TTFT improvement.

   The deploy is **idempotent** and backs the original up once to
   `attention_segmented_cte.py.bak_armin` before overwriting.

2. **Installs the transformers gemma4 config stub** (`gemma4_transformers_stub`)
   so `AutoConfig` recognizes `model_type: gemma4` — needed *before* vLLM calls
   `AutoConfig` on the checkpoint (public transformers 4.x has no gemma4 yet).

3. **Registers the model** (`gemma4_register`) — force-registers
   `Gemma4ForConditionalGeneration` into vLLM's `ModelRegistry` and the
   `vllm_neuron` registry, with a post-plugin re-register hook (the vllm_neuron
   plugin resets the registry when it loads).

## Files

| file | what it is |
|---|---|
| `sitecustomize.py` | auto-import entrypoint — runs the 3 steps above |
| `deploy_segmented_cte.py` | copies the patched segmented kernel over the installed vllm_neuron copy (idempotent, backs up original) |
| `attention_segmented_cte.py` | the patched segmented wrapper (edit A + SWA windowed gather) — source for the deploy above |
| `gemma4_register.py` | force-registers `Gemma4ForConditionalGeneration` into vLLM |
| `gemma4_transformers_stub.py` | teaches transformers `AutoConfig` about `model_type: gemma4` |
| `gemma4_flash_prefill_v2.py` | V2 d-tiled flash-prefill NKI kernel (head_dim 256/512). Only used on the **non-chunked** prefill path; see note below |
| `gemma4/` | the model package — `model.py` (heterogeneous SWA/global attention, QK/V norm, GeGLU, logit softcap, tied embeds), `attention_decode_kernel.py`, `config.py`, `factory.py`, `__init__.py` |

## Env flags

| var | default | effect |
|---|---|---|
| `GEMMA4_V2_PREFILL` | `1` (on) | use the V2 flash-prefill NKI kernel on the non-chunked prefill path. Falls back to torch SDPA if the kernel can't load. |

> **Note on V2 prefill vs. the benchmark.** The benchmark in this folder runs
> **chunked** prefill for every input size (`--max-num-batched-tokens` = 4096 <
> `--max-model-len`), so attention always goes through the segmented path in
> step 1 above — the V2 kernel's non-chunked branch is not on that hot path.
> The kernel is bundled for completeness and for single-shot (`SEG >= LEN`,
> ≤16K) serving, where it lowers TTFT further. The validated long-context
> numbers come from the segmented path (edit A + SWA windowed gather), not V2.

## Validated

This is the package that served Gemma4-31B at up to **32K input + 500 output**
on a trn2.48xlarge (TP=32, beta v5 / SDK 2.30), fresh compile: flat TTFT
(~0.83 s @ 4K → ~0.92 s @ 32K) and 7/7 correctness including needle-in-haystack
retrieval across ~27K tokens of context.

The patched `attention_segmented_cte.py` is validated against the vLLM-Neuron
**private beta v5 (SDK 2.30, vLLM 0.19.0)** image. On a different vLLM-Neuron
build the file may not match; if the deploy step warns or the server fails to
start, that version mismatch is the first thing to check.

# Dense Qwen3 on vLLM-Neuron 0.21

`mk_qwen3.py` generates a text-only `vllm_neuron/model/qwen3` package so that a
dense `Qwen3ForCausalLM` checkpoint (e.g. `Qwen/Qwen3-8B`) can be served.

## Why a port is needed

Enumerating the plugin registry inside the 0.21 container returns four
architectures and no dense Qwen3:

```
Eagle3LlamaForCausalLM            vllm_neuron.model.llama3.factory
GptOssForCausalLM                 vllm_neuron.model.gpt_oss.factory
LlamaForCausalLM                  vllm_neuron.model.llama3.factory
Qwen3VLForConditionalGeneration   vllm_neuron.model.qwen3_vl.factory
```

Pointing vLLM at `Qwen/Qwen3-8B` therefore resolves vLLM's own generic class and
fails with `AttributeError: type object 'Qwen3ForCausalLM' has no attribute
'from_configs'`.

## Why derive from `qwen3_vl`, not `llama3`

Qwen3 dense is Llama plus QK-norm. The obvious move is to copy `llama3/model.py`
and add QK-norm — but that means inserting five kwargs into several
`NF.qkv_proj` call sites across the prefill and decode paths.

`qwen3_vl` already has all of it, because its text decoder *is* Qwen3:

- prefill: `qk_norm_pre_rope_q_norm/k_norm/eps/q_gamma/k_gamma`
- decode: `rmsnorm_QK_pre_rope_W_Q` / `W_K`

So the smaller diff is to take a correct Qwen3 decoder and remove the vision
half. That is what this script does.

## What it changes

| # | Edit | Why |
|---|---|---|
| 1 | `HF_TEXT_PREFIX` `model.language_model` -> `model` | dense checkpoints are not nested |
| 2 | drop `SupportsVisionWarmup`, `SupportsMRoPE` bases | text-only |
| 3 | `self.visual = None` | do not build a vision encoder |
| 4 | skip `self.visual.load_weights(...)` | no vision weights in the checkpoint |
| 5 | rename `get_mrope_input_positions`, `build_vision_synthetic_inputs`, `embed_multimodal` | these are `runtime_checkable` Protocols — removing the method removes the capability, so the runner sends plain 1D positions |
| 6 | `rotary_position_ids` optional, defaults to `positions` | M-RoPE degenerates to standard RoPE when all three sections carry the same ramp; the rotary module expands 1D internally |
| 7 | `from_configs` accepts a FLAT dense config and synthesizes `rope_parameters` from `rope_theta` | dense Qwen3 has no nested `text_config` and no `rope_parameters` |

`mrope_section` must sum to `head_dim // 2`. For Qwen3-8B (`head_dim=128`) the
script emits `[22, 21, 21]` and asserts the sum.

## Usage

Run inside the vLLM-Neuron 0.21 container, then serve normally:

```bash
python3 mk_qwen3.py          # writes vllm_neuron/model/qwen3/ + registers it

vllm serve Qwen/Qwen3-8B --served-model-name qwen3_8b \
  --tensor-parallel-size 4 --max-model-len 4096 --max-num-seqs 16 \
  --max-num-batched-tokens 4096 --no-enable-prefix-caching \
  --additional-config '{"neuron_config":{"num_batched_tokens_buckets":[4096],
    "num_seqs_buckets":[16],"on_device_sampling_config":{"all_greedy":true}}}'
```

Every edit is asserted and match-counted, so a pattern that stops matching a
future 0.21 revision fails loudly instead of silently no-opping. Dry-run it
against an extracted copy of the package before spending a device.

On success the engine logs the derived config, which is worth eyeballing:

```
[qwen3] text-only config: layers=36 hidden=4096 heads=32 kv=8 head_dim=128
        vocab=151936 rope_theta=1000000 tie=False
```

## Measured — Qwen3-8B, TP=4, 3288 prompt tokens / 1 output

Coherence 3/3 with `enable_thinking: false`: *"The capital of France is Paris."*,
*"...**Jupiter**..."*, *"4"*.

| conc | RPS | avg latency |
|---:|---:|---:|
| 1 | 3.97 | 0.25 s |
| 8 | 4.10 | 1.10 s |
| 32 | 4.13 | 4.01 s |
| 64 | 4.13 | 7.87 s |

Ready in 197 s. 13,579 tok/s prefill per replica, which is above the
native-PyTorch path's 10,472 tok/s for the same model and TP.

### RPS is flat because prefill already saturates the engines

Raising `max_num_batched_tokens` from 4096 to 16384 changed throughput by **1%**
(4.09 -> 4.13 RPS) while latency stayed linear in concurrency. A 3.3K-token
prefill has enough parallelism to fill the tensor engines on its own, so
co-scheduling more prefills cannot add throughput. Do not spend time tuning the
batch budget for a long-prompt / short-output shape.

Note that vLLM clamps `max_num_batched_tokens` to `max_model_len` on this
platform: passing 16384 against `--max-model-len 4096` is rejected with
`Last bucket in num_batched_tokens_buckets must equal max_num_batched_tokens
(4096), got 16384`. Raise both together.

## Caveat on FP8

This generator produces a **BF16** model (`qwen3_vl` is bf16-only). See the
FP8 section of the top-level README: a dense Qwen3 FP8 port has to start from
`llama3/model_static_fp8.py`, and only `modelopt` / `compressed-tensors`
checkpoints are accepted.

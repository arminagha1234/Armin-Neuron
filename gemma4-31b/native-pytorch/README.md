# Gemma 4 31B — Native PyTorch (planned)

**Status: not yet built.** The validated path today is the production
serving path at [`../vllm-neuron/`](../vllm-neuron/README.md), which hits
**121 ms weighted-avg TTFT** and **42.8 tok/s** aggregate throughput on
trn2.48xlarge.

This folder is reserved for the standalone native-PyTorch inference path
(`torch.device("neuron")` + `torch.compile(backend="neuron")`) — for
single-call latency studies and research where you want to read and
modify the model code directly, without a serving framework.

## Why the vLLM-Neuron path was first

The customer latency target lives on the production serving path
(weighted-avg TTFT under the 174 ms budget on a real payload mix), so
all the validated numbers and graphs are over there:

- TTFT vs target → [`../vllm-neuron/results/ttft_vs_target.png`](../vllm-neuron/results/ttft_vs_target.png)
- Throughput sweep → [`../vllm-neuron/results/throughput_sweep.png`](../vllm-neuron/results/throughput_sweep.png)
- Per-bucket TTFT at the recommended config → [`../vllm-neuron/results/per_bucket_ttft.png`](../vllm-neuron/results/per_bucket_ttft.png)
- Old vs new → [`../vllm-neuron/results/old_vs_new.png`](../vllm-neuron/results/old_vs_new.png)

## Plan when this path is built

1. Load `google/gemma-4-31b-it` with `to(torch.device("neuron"))` in bf16.
2. Wrap the forward with `torch.compile(model, backend="neuron")`.
3. Reuse the head_dim>128 attention handling documented in the vLLM path's
   `gemma4/model.py`. The same SWA (head_dim=256) and Global (head_dim=512)
   layer split applies here.
4. Benchmark single-call TTFT and per-token decode rate, and compare
   against the served numbers from `../vllm-neuron/results/`.

## Target benchmark template (to fill in)

| Config | TP | TTFT | Per-token decode | Notes |
|---|---:|---:|---:|---|
| Native PyTorch, 4K | 32 | _TBD_ | _TBD_ | direct compile |
| Native PyTorch, 1K | 32 | _TBD_ | _TBD_ | direct compile |

## Known constraints (from the vLLM-Neuron path bring-up)

These will carry over verbatim to the native path because they're model
properties, not framework properties:

- Both SWA (head_dim=256) and Global (head_dim=512) layers exceed the
  fused decode megakernel's 128 head_dim cap. Decode runs the decomposed
  manual-attention path and is host-bound at ~2.9 tok/s per request,
  invariant to TP and to the attention sub-kernel choice.
- The bf16 → fp8 KV cache config doesn't currently raise the concurrency
  ceiling because the scheduler's worst-case KV capacity calculation
  doesn't honor the dtype.
- `transformers` requires the `gemma4` config/processor stub from
  [`../vllm-neuron/gemma4_transformers_stub.py`](../vllm-neuron/gemma4_transformers_stub.py)
  until upstream `transformers >= 5.0` ships native gemma4 support.

See [`../vllm-neuron/README.md`](../vllm-neuron/README.md) for the
production-validated serving path, full results, and reproduction steps.

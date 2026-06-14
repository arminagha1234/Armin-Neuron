# Gemma 4 31B — Native PyTorch (WIP)

**Status:** WIP — the validated path today is [`../vllm-neuron/`](../vllm-neuron/README.md).

## What this path will be

A standalone native-PyTorch inference path using
`torch.device("neuron")` + `torch.compile(backend="neuron")` — no serving
framework, for single-call inference and research where you want to read
and modify the model code directly.

## Why it's a stub for now

Gemma 4 31B was brought up first on the production serving path
(vLLM-Neuron) because that's where the customer latency target lives
(weighted-avg TTFT on a real payload mix). The native-PyTorch standalone
path is a straightforward follow-on:

1. Load `google/gemma-4-31b-it` with `device_map="neuron"` in bf16
2. Wrap the forward with `torch.compile(backend="neuron")`
3. Apply the same head_dim>128 SDPA handling noted in the vLLM path
4. Benchmark single-call latency vs the served numbers

## Target benchmark (to fill in)

| Config | TP | TTFT | Per-token | Notes |
|---|---:|---:|---:|---|
| Native PyTorch, 4K | 32 | _TBD_ | _TBD_ | direct compile |

See [`../vllm-neuron/README.md`](../vllm-neuron/README.md) for the working
serving path and full results.

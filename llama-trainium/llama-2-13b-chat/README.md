# Llama-2-13b-chat on Trainium — needs TP≥4 (run on Trn2)

`meta-llama/Llama-2-13b-chat-hf` (~26 GB bf16) needs tensor parallelism across
**≥4 NeuronCores**. On **Trn1** this hits a hardware boundary:

| Config | Result |
|---|---|
| TP=2 (both cores on 1 chip) | ❌ **OOM** — 13B fills 15.4 GB of the 16 GB core |
| TP=4 (spans 2 chips) | ❌ init fails: `Failed to execute the device barrier 1` (cross-chip collective) |

So on Trn1 the 13B is blocked: too big for the 2 intra-chip cores, and cross-chip
TP (TP≥4) fails at the device-barrier init on the current beta backend (tried with
and without `NEURON_RT_ROOT_COMM_ID`).

## Run it on a Trn2

A Trn2 core has more HBM (so TP fits) and supports the cross-chip collective. The
scripts here are **identical** to the validated
[`../llama-3.1-8b/`](../llama-3.1-8b/) example — only the core count and model id
change:

```bash
# gated: accept license + hf auth login first
python3 cpu_ref.py meta-llama/Llama-2-13b-chat-hf ref13b.pt

NEURON_RT_NUM_CORES=4 TORCH_NEURONX_ENABLE_HOST_CC=1 TORCH_NEURONX_ENABLE_ASYNC_NRT=1 \
  torchrun --nnodes 1 --nproc_per_node=4 --rdzv_backend c10d --rdzv_endpoint localhost:29500 \
  tp_validate.py meta-llama/Llama-2-13b-chat-hf llama-2-13b-chat ref13b.pt
```

## Why not smaller weights?

The beta doesn't support int8/fp8, and fp16 is the same 16 GB as bf16, so you
can't shrink a 13B to fit fewer/smaller cores — the answer is more cores (TP≥4)
with enough HBM, i.e. Trn2.

## Files
- `cpu_ref.py`, `tp_validate.py` — same as the 8B example; see
  [`../llama-3.1-8b/README.md`](../llama-3.1-8b/README.md) for the mechanism and the
  essential `TORCH_NEURONX_ENABLE_HOST_CC=1` flag.

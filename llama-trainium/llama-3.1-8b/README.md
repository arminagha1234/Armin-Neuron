# Llama-3.1-8B on Trainium — tensor-parallel (WIP)

Llama-3.1-8B in bf16 is ~16 GB, which does **not** fit on a single 16 GB Trn1
NeuronCore (a 7B does — see [`../llama-7b/`](../llama-7b/)). So this example shards
the model across **2 NeuronCores** with native `torch.distributed`
(`backend="neuron"`) + Hugging Face `tp_plan="auto"`.

## Status: 🚧 partial — sharding works, cross-core collective is blocked on the current beta

| Step | Result |
|---|---|
| Shard 8B across 2 cores (`tp_plan="auto"`) | ✅ loads in ~2.5 s, ~8 GB/core |
| Cross-core all-reduce in the forward | ❌ **hangs** on this beta build |
| End-to-end validated logits | ⏳ blocked on the above |

The forward stalls on the tensor-parallel collective with:

```
TDRV:exec_request_check_proxy_done_state  Timeout exceeded: Waiting on barrier proxy task: 120 sec
```

preceded upstream by `aws-ofi-nccl initialization failed ... is EFA enabled?`.
In other words: the model and the sharding are fine, but the **collective
transport between NeuronCores** isn't initializing in the current native-beta
container, so the all-reduce never completes. This is a beta-backend issue, not a
model-port bug — report it through your AWS account team.

## Run (once the collective path works)

```bash
NEURON_RT_NUM_CORES=2 torchrun --nproc_per_node=2 \
    tp_validate.py meta-llama/Llama-3.1-8B
```

(`meta-llama/Llama-3.1-8B` is gated — accept its license and `hf auth login` first.)

## Workarounds

- **Use a Trn2.** Trainium2 has more HBM per core, so an 8B fits on a single core
  and you can use the simple, working single-core pattern from
  [`../llama-7b/validate.py`](../llama-7b/validate.py) with no tensor parallelism
  or collectives at all.
- **Wait for the collective fix** in a newer native-beta build, then re-run the
  `torchrun` command above.

## Files

- `tp_validate.py` — 2-core TP validation. Initializes the `neuron` process group,
  loads the model sharded via `tp_plan="auto"`, runs a forward, and (on rank 0)
  compares per-position top-1 predictions against a CPU fp32 reference.

## Why tensor parallelism (and not quantization)

The current beta doesn't support int8/fp8, and fp16 is the same 16 GB as bf16, so
you can't shrink the weights to fit one core. The fix for a model bigger than one
core's HBM is **more cores** (tensor parallelism) or **more HBM per core** (Trn2).

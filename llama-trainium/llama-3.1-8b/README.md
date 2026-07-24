# Llama-3.1-8B on Trainium — tensor-parallel (✅ validated)

Llama-3.1-8B (~16 GB bf16) doesn't fit on one 16 GB Trn1 NeuronCore, so it's
sharded across **2 NeuronCores** with native `torch.distributed`
(`backend="neuron"`) + Hugging Face `tp_plan="auto"`.

## Status: ✅ validated

Per-position top-1 agreement (neuron **TP=2** bf16 vs cpu fp32) = **100.0% (37/37 positions)**.

| Step | Result |
|---|---|
| Shard 8B across 2 cores (`tp_plan="auto"`) | loads ~2.5 s (~8 GB/core) |
| Cross-core all-reduce (with host CC) | ✅ completes |
| Agreement vs CPU fp32 | ✅ **100.0%** |

## Run (2 steps)

```bash
# meta-llama/Llama-3.1-8B is gated: accept its license and `hf auth login` first.
python3 cpu_ref.py meta-llama/Llama-3.1-8B ref31.pt

NEURON_RT_NUM_CORES=2 TORCH_NEURONX_ENABLE_HOST_CC=1 TORCH_NEURONX_ENABLE_ASYNC_NRT=1 \
  torchrun --nnodes 1 --nproc_per_node=2 --rdzv_backend c10d --rdzv_endpoint localhost:29500 \
  tp_validate.py meta-llama/Llama-3.1-8B llama-3.1-8b ref31.pt
```

Expected tail:
```
[llama-3.1-8b] per-position top-1 agreement (neuron TP2 bf16 vs cpu fp32): 100.0%  (37 positions)
PORT_OK
```

## The one thing that makes TP work: host collective communication

**`TORCH_NEURONX_ENABLE_HOST_CC=1` is required.** Without it, the intra-node
all-reduce tries the OFI/EFA device path (which can't initialize in-container →
`aws-ofi-nccl initialization failed / is EFA enabled?`) and **hangs forever** on
the collective barrier. With host CC the collective completes instantly. The
OFI/EFA warning itself is benign — it also appears in working runs.

## Notes

- **Restart the container between TP runs** — the beta teardown leaves the Neuron
  runtime in a state that breaks the next `init_process_group`.
- On Trn1, **TP=2 (both cores on one chip) works**; cross-chip TP (TP≥4) currently
  fails at the device-barrier init — fine here since the 8B only needs TP=2.

## Files
- `cpu_ref.py` — compute + save the CPU fp32 reference (run once, single process).
- `tp_validate.py` — TP forward + per-position agreement check.

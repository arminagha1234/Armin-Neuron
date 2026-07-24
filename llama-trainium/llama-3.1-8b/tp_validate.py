"""Tensor-parallel validation on Trainium (native PyTorch, no XLA).

Shards an HF model across N NeuronCores with tp_plan="auto" and checks the
per-position top-1 agreement of the Neuron bf16 output vs a pre-computed CPU
fp32 reference (see cpu_ref.py).

Two steps:
  # 1. Precompute the CPU reference (single, non-distributed process):
  python3 cpu_ref.py <model_path> ref.pt
  # 2. Run the TP validation (host collectives are REQUIRED intra-node):
  NEURON_RT_NUM_CORES=2 TORCH_NEURONX_ENABLE_HOST_CC=1 TORCH_NEURONX_ENABLE_ASYNC_NRT=1 \
    torchrun --nnodes 1 --nproc_per_node=2 --rdzv_backend c10d --rdzv_endpoint localhost:29500 \
    tp_validate.py <model_path> <name> ref.pt

Key points:
  * TORCH_NEURONX_ENABLE_HOST_CC=1 is ESSENTIAL. Without it the all-reduce tries
    the intra-node OFI/EFA device path (which can't init in-container) and hangs
    forever on the collective barrier. The 'aws-ofi-nccl init failed' warning is
    benign (it appears in working runs too).
  * On Trn1, TP=2 (both cores on one chip) works. Cross-chip TP (TP>=4) currently
    fails at the device-barrier init on the beta backend -> use Trn2 for models
    that need TP>=4 (e.g. a 13B).
  * Restart the container between TP runs (teardown leaves stale runtime state).
"""
import sys, time, os, torch, torch_neuronx
import torch.distributed as dist
from transformers import AutoModelForCausalLM

mp, name, ref = sys.argv[1], sys.argv[2], sys.argv[3]
dist.init_process_group(backend="neuron")
rank = dist.get_rank()
world = dist.get_world_size()
device = torch.device("neuron")

d = torch.load(ref)
ids, cpu_pred = d["ids"], d["cpu_pred"]

model = AutoModelForCausalLM.from_pretrained(mp, dtype=torch.bfloat16, tp_plan="auto", attn_implementation="eager")
model.eval()
t0 = time.time()
with torch.no_grad():
    out = model(ids.to(device), use_cache=False).logits
if hasattr(out, "full_tensor"):   # gather the sharded (DTensor) logits
    out = out.full_tensor()
nrn_pred = out.float().argmax(-1)[0].cpu()

if rank == 0:
    n = cpu_pred.shape[0]
    match = (cpu_pred == nrn_pred).sum().item() / n * 100.0
    print(f"[{name}] TP{world} forward {time.time()-t0:.1f}s")
    print(f"[{name}] per-position top-1 agreement (neuron TP{world} bf16 vs cpu fp32): {match:.1f}%  ({n} positions)")
    print("PORT_OK" if match >= 95.0 else "PORT_MISMATCH")

# The beta backend can SIGSEGV in destroy_process_group teardown AFTER the work
# is done; force a clean exit so it doesn't mask a successful result.
sys.stdout.flush()
os._exit(0)

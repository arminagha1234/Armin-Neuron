"""Tensor-parallel validation on Trainium (native PyTorch, no XLA).

Same script as the llama-3.1-8b example — shards an HF model across N NeuronCores
with tp_plan="auto" and checks per-position top-1 agreement vs a pre-computed CPU
fp32 reference (cpu_ref.py). A 13B needs TP>=4.

  python3 cpu_ref.py <model_path> ref.pt
  NEURON_RT_NUM_CORES=4 TORCH_NEURONX_ENABLE_HOST_CC=1 TORCH_NEURONX_ENABLE_ASYNC_NRT=1 \
    torchrun --nnodes 1 --nproc_per_node=4 --rdzv_backend c10d --rdzv_endpoint localhost:29500 \
    tp_validate.py <model_path> <name> ref.pt

NOTE (Trn1): a 13B needs TP>=4, which spans >1 chip. Cross-chip TP currently fails
at the device-barrier init on the Trn1 beta backend (TP=2 fits on one chip but a
13B OOMs at TP=2). Run this on a Trn2, where TP fits and cross-chip collectives
work. See README.md.
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
if hasattr(out, "full_tensor"):
    out = out.full_tensor()
nrn_pred = out.float().argmax(-1)[0].cpu()

if rank == 0:
    n = cpu_pred.shape[0]
    match = (cpu_pred == nrn_pred).sum().item() / n * 100.0
    print(f"[{name}] TP{world} forward {time.time()-t0:.1f}s")
    print(f"[{name}] per-position top-1 agreement (neuron TP{world} bf16 vs cpu fp32): {match:.1f}%  ({n} positions)")
    print("PORT_OK" if match >= 95.0 else "PORT_MISMATCH")

sys.stdout.flush()
os._exit(0)

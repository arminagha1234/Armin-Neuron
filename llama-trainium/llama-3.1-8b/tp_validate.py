"""Tensor-parallel Llama-3.1-8B on Trainium (native PyTorch) — 2-core TP.

An 8B model (~16 GB bf16) does NOT fit on a single 16 GB Trn1 NeuronCore, so we
shard it across cores with native torch.distributed (backend="neuron") and HF's
tp_plan="auto". Launch one process per core:

    NEURON_RT_NUM_CORES=2 torchrun --nproc_per_node=2 tp_validate.py meta-llama/Llama-3.1-8B

Status (Trn1, current beta): the model SHARDS and LOADS across 2 cores fine, but
the cross-core collective (all-reduce) in the forward is not completing on this
beta build (barrier timeout / OFI-EFA init failure). See README.md. On a Trn2
(larger HBM/core) the single-core llama-7b/validate.py pattern works without TP.
"""
import sys, time, torch, torch_neuronx
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = sys.argv[1] if len(sys.argv) > 1 else "meta-llama/Llama-3.1-8B"
name = sys.argv[2] if len(sys.argv) > 2 else model_path

dist.init_process_group(backend="neuron")
rank = dist.get_rank()
world = dist.get_world_size()
device = torch.device("neuron")

def rprint(*a):
    if rank == 0:
        print(*a, flush=True)

prompt = ("Artificial intelligence is transforming the world. In this paper we "
          "describe how large language models can be trained efficiently on custom "
          "accelerators such as AWS Trainium. The key idea is to")
tok = AutoTokenizer.from_pretrained(model_path)
ids = tok(prompt, return_tensors="pt").input_ids
rprint(f"[{name}] world_size={world}  prompt_tokens={ids.shape[1]}")

rprint("loading TP-sharded model on neuron (tp_plan=auto)...")
t0 = time.time()
model = AutoModelForCausalLM.from_pretrained(
    model_path, dtype=torch.bfloat16, tp_plan="auto", attn_implementation="eager")
model.eval()
rprint(f"[{name}] model loaded (sharded across {world} cores) in {time.time()-t0:.1f}s")

t0 = time.time()
with torch.no_grad():
    out = model(ids.to(device), use_cache=False).logits
if hasattr(out, "full_tensor"):   # gather if the LM head output is a DTensor
    out = out.full_tensor()
nrn_pred = out.float().argmax(-1)[0].cpu()
rprint(f"[{name}] neuron TP{world} forward {time.time()-t0:.1f}s")

if rank == 0:
    print("loading CPU fp32 reference...", flush=True)
    cpu_m = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.float32, attn_implementation="eager")
    with torch.no_grad():
        cpu_pred = cpu_m(ids, use_cache=False).logits.float().argmax(-1)[0].cpu()
    n = cpu_pred.shape[0]
    match = (cpu_pred == nrn_pred).sum().item() / n * 100.0
    print(f"[{name}] per-position top-1 agreement (neuron TP{world} bf16 vs cpu fp32): {match:.1f}%  ({n} positions)")
    print("PORT_OK" if match >= 95.0 else "PORT_MISMATCH")

dist.barrier()
dist.destroy_process_group()

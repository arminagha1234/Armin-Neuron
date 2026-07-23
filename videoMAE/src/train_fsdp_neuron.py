"""Task 14: multi-core FSDP2 training of VideoMAEv2 with the native PyTorch beta (no XLA).

Uses the beta's "neuron" torch.distributed backend + PyTorch FSDP2 (torch.distributed.fsdp.fully_shard),
matching the pattern in the beta's own examples/compile/distributed.py and tests/.../fsdp_v2/.

Launch (trn2.3xlarge = 2 logical NeuronCores under default LNC2):
    torchrun --standalone --nproc_per_node=2 train_fsdp_neuron.py
"""
import os
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch_neuronx  # registers 'neuron' device + dist backend
from torch.distributed.fsdp import fully_shard
from huggingface_hub import snapshot_download

from modeling_videomaev2_native import build_videomaev2_base, load_pretrained_weights

NUM_CLASSES = 400
BATCH = 2
STEPS = 6


def main():
    dist.init_process_group("neuron")
    rank = dist.get_rank()
    world = dist.get_world_size()
    torch.neuron.set_device(rank)
    device = torch.device(f"neuron:{rank}")

    st = os.path.join(
        snapshot_download("OpenGVLab/VideoMAEv2-Base", allow_patterns=["model.safetensors"]),
        "model.safetensors",
    )

    torch.manual_seed(0)
    model = build_videomaev2_base(num_classes=NUM_CLASSES)
    load_pretrained_weights(model, st)          # pretrained backbone; head is fresh
    model = model.to(device)

    # make all ranks identical (the fresh head is random per process otherwise)
    for p in model.parameters():
        dist.broadcast(p, src=0)

    # FSDP2: shard each transformer block, then the root module
    for blk in model.blocks:
        fully_shard(blk)
    fully_shard(model)

    # report weight sharding (each rank should hold ~1/world of the params)
    try:
        local = sum(
            (p.to_local().numel() if hasattr(p, "to_local") else p.numel())
            for p in model.parameters()
        )
        full = sum(
            (p.to_local().numel() if hasattr(p, "to_local") else p.numel())
            for p in model.parameters()
        )
        if rank == 0:
            print(f"world_size={world}  per-rank sharded params ~= {local/1e6:.2f} M", flush=True)
    except Exception as e:
        if rank == 0:
            print("shard-size probe skipped:", e, flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # per-rank data => data-parallel across ranks, weights FSDP-sharded
    torch.manual_seed(100 + rank)
    x = torch.randn(BATCH, 3, 16, 224, 224, device=device)
    labels = torch.randint(0, NUM_CLASSES, (BATCH,), device=device)

    if rank == 0:
        print("step |   loss   | sec  (step 0 includes fwd+bwd NEFF compile)", flush=True)
    for step in range(STEPS):
        t0 = time.time()
        opt.zero_grad()
        loss = F.cross_entropy(model(x), labels)
        loss.backward()
        opt.step()
        l = float(loss.detach().cpu())
        if rank == 0:
            print(f"{step:>4d} | {l:8.4f} | {time.time()-t0:5.1f}", flush=True)

    if rank == 0:
        print("FSDP_TRAIN_OK", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

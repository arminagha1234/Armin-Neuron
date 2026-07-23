"""Task 5: multi-core FSDP2 *pretraining* of VideoMAE v2 (native PyTorch beta, no XLA).

Full self-supervised objective (encoder-on-visible + decoder + tube masking + normalized-pixel
MSE reconstruction), all params, sharded across NeuronCores with PyTorch FSDP2 (fully_shard).
Data-parallel varied structured video per rank; weights FSDP-sharded.

Launch (trn2.3xlarge = 2 logical NeuronCores under default LNC2):
    torchrun --standalone --nproc_per_node=2 train_fsdp_pretrain.py
"""
import time

import numpy as np
import torch
import torch.distributed as dist
import torch_neuronx  # registers 'neuron' device + dist backend
from torch.distributed.fsdp import fully_shard

from modeling_pretrain_native import build_pretrain_videomae_base, tube_mask_indices
from pretrain_neuron import make_structured_clips, make_target, Tp, Hp, Wp

BATCH = 2
STEPS = 20
MASK_RATIO = 0.9


def main():
    dist.init_process_group("neuron")
    rank = dist.get_rank()
    world = dist.get_world_size()
    torch.neuron.set_device(rank)
    device = torch.device(f"neuron:{rank}")

    torch.manual_seed(0)
    model = build_pretrain_videomae_base().train().to(device)

    # sync initial weights across ranks
    for p in model.parameters():
        dist.broadcast(p, src=0)

    # FSDP2: shard each encoder + decoder block, then the root
    for blk in model.encoder.blocks:
        fully_shard(blk)
    for blk in model.decoder.blocks:
        fully_shard(blk)
    fully_shard(model)

    local = sum((p.to_local().numel() if hasattr(p, "to_local") else p.numel())
                for p in model.parameters())
    if rank == 0:
        print(f"world_size={world}  per-rank sharded params ~= {local/1e6:.2f} M", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=1.5e-4, weight_decay=0.05)

    rng = np.random.RandomState(100 + rank)   # per-rank varied data (data-parallel)

    if rank == 0:
        print("step |  recon_loss | sec  (step 0 includes fwd+bwd NEFF compile)", flush=True)
    losses = []
    for step in range(STEPS):
        t0 = time.time()
        images = make_structured_clips(BATCH, rng).to(device)
        ids_keep, ids_mask = tube_mask_indices(BATCH, Tp, Hp, Wp, MASK_RATIO, rng)
        ids_keep, ids_mask = ids_keep.to(device), ids_mask.to(device)
        with torch.no_grad():
            target = make_target(images)
            Cpx = target.shape[-1]
            labels = torch.gather(target, 1, ids_mask.unsqueeze(-1).expand(-1, -1, Cpx))
        opt.zero_grad()
        outputs = model(images, ids_keep, ids_mask)
        loss = ((outputs - labels) ** 2).mean()
        loss.backward()
        opt.step()
        l = float(loss.detach().cpu())
        losses.append(l)
        if rank == 0:
            print(f"{step:>4d} | {l:11.6f} | {time.time()-t0:5.1f}", flush=True)

    if rank == 0:
        f5 = sum(losses[:5]) / 5
        l5 = sum(losses[-5:]) / 5
        print(f"\nmean(first5)={f5:.6f}  mean(last5)={l5:.6f}  decreased={l5 < f5}", flush=True)
        print("FSDP_PRETRAIN_OK" if l5 < f5 else "FSDP_PRETRAIN_NO_PROGRESS", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

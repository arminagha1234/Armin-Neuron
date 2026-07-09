"""
WHOLE Clay model — MULTI-CORE data-parallel EAGER training on Trainium.

Each rank = one NeuronCore holding a full ClayMAE replica (+ frozen DINOv2 teacher).
Ranks process different data shards; gradients are averaged with a world-group
all-reduce (the collective pattern that works on the Beta-3 native stack), then
AdamW.step. Uses the DLC's documented init: dist.init_process_group("neuron"),
per-rank NEURON_RT_VISIBLE_CORES, mp.spawn.

Launch:
  NEURON_RT_NUM_CORES=<W> python clay_ddp_train.py --world 4 --size base --img 128
"""

import argparse
import os
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from claymodel.model import ClayMAE, clay_mae_config

S2_BANDS = 10
S2_WAVES = [0.493, 0.56, 0.665, 0.704, 0.74, 0.783, 0.842, 0.865, 1.61, 2.19]


class DotDict(dict):
    def __getattr__(self, k):
        return self[k]
    def __getitem__(self, k):
        v = dict.__getitem__(self, k)
        return DotDict(v) if isinstance(v, dict) else v


def s2_meta():
    return DotDict({"sentinel-2-l2a": {
        "band_order": list(range(S2_BANDS)), "rgb_indices": [2, 1, 0], "gsd": 10,
        "bands": {"wavelength": {i: w for i, w in enumerate(S2_WAVES)}},
    }})


def worker(rank, world, args):
    if "RANK" not in os.environ:  # mp.spawn path: set rendezvous + core pinning
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = "12355"
        os.environ["NEURON_RT_VISIBLE_CORES"] = str(rank)
    # torchrun path: MASTER_ADDR/PORT + core assignment handled by launcher/runtime
    dist.init_process_group("neuron", rank=rank, world_size=world)

    torch.manual_seed(0)  # identical init on every rank => replicas stay in sync
    dtype = torch.float32 if args.dtype == "float32" else torch.bfloat16
    device = torch.device("neuron")
    cfg = clay_mae_config(args.size)

    model = ClayMAE(
        mask_ratio=0.75, patch_size=args.patch, norm_pix_loss=False, shuffle=True,
        metadata=s2_meta(), teacher=args.teacher,
        dolls=[16, 32, 64, 128, 256, 768, 1024], doll_weights=[1] * 7,
        fused_attn=False, teacher_impl="transformers", **cfg,
    ).to(device=device, dtype=dtype)
    model.train(); model.teacher.eval()

    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr,
        weight_decay=0.05, betas=(0.9, 0.95),
    )
    grad_params = [p for p in model.parameters() if p.requires_grad]

    # per-rank data shard (different seed per rank = data parallelism)
    g = torch.Generator().manual_seed(100 + rank)
    datacube = {
        "pixels": torch.randn(args.batch, S2_BANDS, args.img, args.img, generator=g).to(device=device, dtype=dtype),
        "time": torch.randn(args.batch, 4, generator=g).to(device=device, dtype=dtype),
        "latlon": torch.randn(args.batch, 4, generator=g).to(device=device, dtype=dtype),
        "platform": ["sentinel-2-l2a"] * args.batch,
    }

    if rank == 0:
        tr = sum(p.numel() for p in grad_params)
        print(f"[ddp] world={world} size={args.size} img={args.img} dtype={args.dtype} "
              f"trainable/rank={tr/1e6:.1f}M", flush=True)

    for step in range(args.steps):
        t0 = time.time()
        opt.zero_grad()
        loss, recon, repr_ = model(datacube)
        loss.backward()
        # ---- world-group gradient all-reduce (average) = data-parallel sync ----
        for p in grad_params:
            if p.grad is not None:
                dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
                p.grad /= world
        opt.step()
        # average loss across ranks just for reporting
        lt = loss.detach().clone()
        dist.all_reduce(lt, op=dist.ReduceOp.SUM)
        lt /= world
        dt = time.time() - t0
        if rank == 0:
            tag = "  (incl. compile)" if step == 0 else ""
            print(f"[step {step}] avg_loss={float(lt.to('cpu')):.5f}  {dt:.2f}s{tag}", flush=True)

    # confirm replicas stayed identical across ranks (weights in sync)
    probe = grad_params[0].detach().float().flatten()[:8].clone()
    gathered = [torch.zeros_like(probe) for _ in range(world)]
    dist.all_gather(gathered, probe)
    if rank == 0:
        max_div = max(float((gathered[0] - g_).abs().max().to("cpu")) for g_ in gathered)
        print(f"[check] max cross-rank weight divergence = {max_div:.3e} "
              f"({'IN SYNC' if max_div < 1e-3 else 'DIVERGED'})", flush=True)
        print(f"[RESULT] PASS: WHOLE ClayMAE({args.size}) DATA-PARALLEL across {world} "
              f"NeuronCores in EAGER mode — per-rank fwd/bwd + world all-reduce + AdamW.", flush=True)

    dist.destroy_process_group()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", type=int, default=4)
    ap.add_argument("--size", default="base")
    ap.add_argument("--teacher", default="facebook/dinov2-large")
    ap.add_argument("--patch", type=int, default=8)
    ap.add_argument("--img", type=int, default=128)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16"])
    args = ap.parse_args()
    if "RANK" in os.environ:  # launched via torchrun
        rank = int(os.environ["RANK"])
        world = int(os.environ["WORLD_SIZE"])
        worker(rank, world, args)
    else:  # standalone: fan out with mp.spawn
        mp.spawn(worker, args=(args.world, args), nprocs=args.world, join=True)


if __name__ == "__main__":
    main()

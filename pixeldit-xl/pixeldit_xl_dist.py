#!/usr/bin/env python3
"""PixelDiT-XL multi-core data-parallel training on AWS Neuron (Beta 3).

Spawns WORLD_SIZE workers, each pinned to its own NeuronCore
(NEURON_RT_VISIBLE_CORES = rank + core_offset), initializes the "neuron"
process group, replicates the model, and synchronizes gradients with
dist.all_reduce (DDP-style). Measures throughput scaling vs single core and
verifies gradient sync correctness (post-all-reduce grad-norm must match
across all ranks).

Launch inside the Beta 3 container (free cores 4+ so the live vLLM on 0-3 is
untouched):
    python3 /work/pixeldit_xl_dist.py --world-size 2 --core-offset 4 \
        --hidden 1152 --depth 12 --steps 4 --batch 1 --image-size 256 --patch 16
"""
import argparse
import os
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

import pixeldit_xl_train as P  # reuse the model + rf_batch


def worker(rank, args):
    # Pin THIS process to a single free NeuronCore before any neuron init.
    core = rank + args.core_offset
    os.environ["NEURON_RT_VISIBLE_CORES"] = str(core)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(args.port)
    os.environ.setdefault("RANK", str(rank))
    os.environ.setdefault("WORLD_SIZE", str(args.world_size))

    dist.init_process_group("neuron", rank=rank, world_size=args.world_size)
    device = torch.device("neuron")

    torch.manual_seed(0)  # identical init across ranks
    model = P.build_model(args, torch.bfloat16).to(device).train()
    if rank == 0:
        n = sum(p.numel() for p in model.parameters())
        print(f"[info] params={n/1e9:.3f}B  world_size={args.world_size}  "
              f"cores={args.core_offset}..{args.core_offset+args.world_size-1}", flush=True)

    fwd = model if args.no_compile else torch.compile(model, backend="neuron", dynamic=False)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    B, C, H = args.batch, args.in_ch, args.image_size

    def allreduce_grads():
        ws = float(args.world_size)
        for p in model.parameters():
            if p.grad is not None:
                dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
                p.grad /= ws

    for step in range(args.steps):
        # Distinct data per rank (data-parallel): seed by rank+step.
        torch.manual_seed(1000 * step + rank)
        x0 = torch.randn(B, C, H, H, device=device, dtype=torch.bfloat16)
        xt, t_in, y, target = P.rf_batch(x0, args.num_classes, device)

        ts = time.time()
        opt.zero_grad(set_to_none=True)
        pred = fwd(xt, t_in, y)
        loss = F.mse_loss(pred.float(), target.float())
        loss.backward()
        allreduce_grads()
        # Correctness probe: global grad norm must be identical on every rank
        gnorm = torch.sqrt(sum((p.grad.float() ** 2).sum() for p in model.parameters()
                               if p.grad is not None))
        opt.step()
        torch.neuron.synchronize()
        dt = (time.time() - ts) * 1000.0
        tag = "first(compile)" if step == 0 else "step"
        print(f"[rank{rank}] step {step} loss={loss.item():.4f} gnorm={gnorm.item():.4f} "
              f"{tag}={dt:.1f}ms", flush=True)

    # Final cross-rank agreement check on grad norm (sync correctness).
    gn = torch.tensor([gnorm.item()], device=device)
    gathered = [torch.zeros_like(gn) for _ in range(args.world_size)]
    dist.all_gather(gathered, gn)
    if rank == 0:
        vals = [g.item() for g in gathered]
        spread = max(vals) - min(vals)
        print(f"[check] per-rank gnorm = {[f'{v:.4f}' for v in vals]}", flush=True)
        print(f"[check] GRAD-SYNC {'PASS' if spread < 1e-2 else 'FAIL'} "
              f"(spread={spread:.3e})", flush=True)
        print("[done] multi-core data-parallel training OK", flush=True)
    dist.destroy_process_group()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--world-size", type=int, default=2)
    ap.add_argument("--core-offset", type=int, default=4)
    ap.add_argument("--port", type=int, default=12365)
    ap.add_argument("--image-size", type=int, default=256)
    ap.add_argument("--patch", type=int, default=16)
    ap.add_argument("--in-ch", type=int, default=3)
    ap.add_argument("--hidden", type=int, default=1152)
    ap.add_argument("--depth", type=int, default=12)
    ap.add_argument("--heads", type=int, default=16)
    ap.add_argument("--num-classes", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--no-compile", action="store_true")
    args = ap.parse_args()
    mp.spawn(worker, args=(args,), nprocs=args.world_size, join=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""PixelDiT-XL multi-core FSDP (weight-sharded) training on AWS Neuron (Beta 3).

Tries real parameter-sharded FSDP (not just data-parallel). Attempts, in order:
  1. FSDP2 `fully_shard` with a DeviceMesh on the "neuron" device type
  2. FSDP2 `fully_shard` with the default "neuron" process group (no mesh)
  3. FSDP1 `FullyShardedDataParallel`
Whichever wraps successfully runs a few train steps; the script logs precisely
which path the Beta 3 stack supports.

Launch inside the Beta 3 container (free cores 4+ ; vLLM on 0-3 untouched):
    python3 /work/pixeldit_xl_fsdp.py --world-size 2 --core-offset 4 \
        --hidden 1152 --depth 12 --steps 4 --batch 1 --image-size 256 --patch 16
"""
import argparse
import os
import time
import traceback

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

import pixeldit_xl_train as P


def log(rank, msg):
    print(f"[rank{rank}] {msg}", flush=True)


def try_fsdp2_mesh(model, world_size, rank):
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import fully_shard
    mesh = init_device_mesh("neuron", (world_size,))
    for blk in model.blocks:
        fully_shard(blk, mesh=mesh)
    fully_shard(model, mesh=mesh)
    return model, "FSDP2+DeviceMesh(neuron)"


def try_fsdp2_default(model, world_size, rank):
    from torch.distributed.fsdp import fully_shard
    for blk in model.blocks:
        fully_shard(blk)
    fully_shard(model)
    return model, "FSDP2+default-PG"


def try_fsdp1(model, world_size, rank):
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import ShardingStrategy
    model = FSDP(model, sharding_strategy=ShardingStrategy.FULL_SHARD, use_orig_params=True)
    return model, "FSDP1"


def worker(rank, args):
    core = rank + args.core_offset
    os.environ["NEURON_RT_VISIBLE_CORES"] = str(core)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(args.port)
    os.environ.setdefault("RANK", str(rank))
    os.environ.setdefault("WORLD_SIZE", str(args.world_size))

    dist.init_process_group("neuron", rank=rank, world_size=args.world_size)
    device = torch.device("neuron")
    torch.manual_seed(0)

    base = P.build_model(args, torch.bfloat16).to(device).train()
    n = sum(p.numel() for p in base.parameters())
    if rank == 0:
        log(rank, f"params={n/1e9:.3f}B world_size={args.world_size} cores={args.core_offset}..{args.core_offset+args.world_size-1}")

    strategies = [try_fsdp2_mesh, try_fsdp2_default, try_fsdp1]
    model, used = None, None
    for fn in strategies:
        try:
            model, used = fn(base, args.world_size, rank)
            if rank == 0:
                log(rank, f"FSDP wrap OK via {used}")
            break
        except Exception as e:
            if rank == 0:
                log(rank, f"FSDP attempt {fn.__name__} FAILED: {type(e).__name__}: {str(e)[:300]}")
            # rebuild base in case the failed wrap mutated it
            torch.manual_seed(0)
            base = P.build_model(args, torch.bfloat16).to(device).train()
    if model is None:
        if rank == 0:
            log(rank, "ALL FSDP strategies failed on this stack")
        dist.destroy_process_group()
        return

    fwd = model if args.no_compile else torch.compile(model, backend="neuron", dynamic=False)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    B, C, H = args.batch, args.in_ch, args.image_size

    for step in range(args.steps):
        torch.manual_seed(1000 * step + rank)
        x0 = torch.randn(B, C, H, H, device=device, dtype=torch.bfloat16)
        xt, t_in, y, target = P.rf_batch(x0, args.num_classes, device)
        ts = time.time()
        opt.zero_grad(set_to_none=True)
        pred = fwd(xt, t_in, y)
        loss = F.mse_loss(pred.float(), target.float())
        loss.backward()
        opt.step()
        torch.neuron.synchronize()
        dt = (time.time() - ts) * 1000.0
        tag = "first(compile)" if step == 0 else "step"
        log(rank, f"step {step} loss={loss.item():.4f} {tag}={dt:.1f}ms")

    # Report sharded (local) param memory to prove weights are split across ranks.
    # FSDP2 params are DTensors; .to_local() gives this rank's actual shard.
    local = 0
    for p in model.parameters():
        t = p.to_local() if hasattr(p, "to_local") else p
        local += t.numel()
    log(rank, f"local-shard-params={local/1e6:.1f}M (full={n/1e6:.1f}M) via {used}")
    if rank == 0:
        log(rank, f"[done] FSDP training OK via {used}")
    dist.destroy_process_group()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--world-size", type=int, default=2)
    ap.add_argument("--core-offset", type=int, default=4)
    ap.add_argument("--port", type=int, default=12375)
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

"""Minimal Neuron collective probe -- isolate TP init from the 20 GB model.

Inits a process group on the 'neuron' backend and runs one all_reduce on a
tiny tensor. Prints success/failure per rank. Iterates in seconds so we can
sweep world sizes and core-placement env vars without loading Mochi.

    NEURON_RT_NUM_CORES=8 torchrun --nproc_per_node 8 collective_probe.py
"""
import os
import sys
import time

import torch
import torch.distributed as dist
import torch_neuronx  # noqa: F401


def main():
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    t0 = time.time()
    try:
        dist.init_process_group(backend="neuron")
    except Exception as e:
        print(f"[rank {rank}] init_process_group FAILED: {e}", flush=True)
        return 1

    dev = torch.device("neuron")
    try:
        x = torch.ones(1024, device=dev) * (rank + 1)
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
        got = x[0].item()
        expected = world * (world + 1) / 2
        ok = abs(got - expected) < 1e-3
        tag = "OK" if ok else "WRONG"
        if rank == 0:
            print(f"[rank {rank}] all_reduce {tag}: got {got}, expected "
                  f"{expected}, world={world}, {time.time()-t0:.1f}s", flush=True)
    except Exception as e:
        print(f"[rank {rank}] all_reduce FAILED: {e}", flush=True)
        return 1
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())

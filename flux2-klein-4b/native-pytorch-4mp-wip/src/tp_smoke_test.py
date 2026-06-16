#!/usr/bin/env python3
"""TP smoke test — validate torchrun + Neuron process group on Beta 3.

The smallest possible multi-rank test before attempting the full TP=4
FLUX lift. Per beta3-only.md, Beta 3 uses standard c10d rendezvous.
Per neuron-tp-on-beta2.md, the PG backend is still 'neuron'.

This script:
  1. init_process_group
  2. allocates a tiny tensor per rank
  3. all_reduce
  4. verifies the sum is correct across ranks
  5. exits cleanly

Run:
    NEURON_RT_VIRTUAL_CORE_SIZE=1 \\
    torchrun --nproc_per_node=2 --rdzv_backend c10d \\
        --rdzv_endpoint localhost:29500 tp_smoke_test.py

If this hangs > 2 min, the PG setup is wrong — kill and debug
interactively. Do NOT scale to 4 ranks or the full model until this
passes.
"""
import os
import sys
import time

import torch
# CRITICAL: importing torch_neuronx.distributed registers the `neuron`
# ProcessGroup backend. Without this import, init_process_group(backend=
# 'neuron') and the collective ops fail with ENC no_mesh errors.
import torch_neuronx              # noqa: F401
import torch_neuronx.distributed  # noqa: F401
import torch.distributed as dist


def main():
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    print(f"[rank {rank}] starting, world_size={world_size}", flush=True)

    # Beta 3: try c10d rendezvous + neuron PG backend.
    t0 = time.time()
    try:
        dist.init_process_group(backend="neuron")
    except Exception as e:
        print(f"[rank {rank}] backend='neuron' failed: {e}", flush=True)
        print(f"[rank {rank}] retrying backend='xla'", flush=True)
        try:
            dist.init_process_group(backend="xla")
        except Exception as e2:
            print(f"[rank {rank}] backend='xla' also failed: {e2}", flush=True)
            sys.exit(2)
    print(f"[rank {rank}] init_process_group OK in {time.time()-t0:.1f}s", flush=True)

    device = torch.device("neuron")
    # Each rank contributes (rank+1). Sum across N ranks should be
    # N*(N+1)/2.
    x = torch.ones(4, dtype=torch.float32, device=device) * (rank + 1)
    dist.all_reduce(x)
    torch.neuron.synchronize() if hasattr(torch, "neuron") else None
    result = x.cpu()
    expected = sum(range(1, world_size + 1))
    ok = torch.allclose(result, torch.full((4,), float(expected)))
    print(f"[rank {rank}] all_reduce result={result.tolist()} "
          f"expected={expected} OK={ok}", flush=True)

    dist.destroy_process_group()
    print(f"[rank {rank}] done, PG destroyed", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

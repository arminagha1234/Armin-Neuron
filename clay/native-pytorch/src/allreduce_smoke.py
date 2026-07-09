"""Minimal multi-core collective smoke test on the neuron backend."""
import os, torch, torch.distributed as dist

rank = int(os.environ["RANK"]); world = int(os.environ["WORLD_SIZE"])
dist.init_process_group("neuron", rank=rank, world_size=world)
print(f"[rank {rank}] init OK", flush=True)
x = torch.ones(8, device="neuron") * (rank + 1)
dist.all_reduce(x, op=dist.ReduceOp.SUM)
val = float(x.float().sum().to("cpu")) / 8
expected = sum(range(1, world + 1))
print(f"[rank {rank}] all_reduce result={val} expected={expected} "
      f"{'OK' if abs(val-expected)<1e-3 else 'MISMATCH'}", flush=True)
dist.destroy_process_group()

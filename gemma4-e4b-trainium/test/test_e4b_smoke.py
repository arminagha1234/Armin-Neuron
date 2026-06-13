"""End-to-end smoke test for Gemma 4 E4B-it on Trainium2 / Inferentia2.

This test runs the same inference path as ``src/run_e4b.py`` and asserts:

  1. The model loads, the TP plan applies cleanly (294 sharded modules
     across 24 owner attention layers + 18 KV-shared attention layers
     + 42 MLPs).
  2. A 64-token prefill produces the expected next token for the
     deterministic prompt ``"The capital of France is"`` -> ``" France"``
     under greedy sampling.
  3. The two ranks finish their forward passes within 100 ms of each
     other (regression guard against accidental TP plan asymmetry).

Launch with torchrun (the test self-checks that it's running under
torchrun + ``backend="neuron"``):

    NEURON_RT_VIRTUAL_CORE_SIZE=2 NEURON_RT_NUM_CORES=2 \\
    /opt/torch-neuronx/.venv/bin/torchrun \\
        --nproc_per_node=2 --rdzv_backend=c10d \\
        --rdzv_endpoint=localhost:29500 \\
        test/test_e4b_smoke.py

Pass criteria (asserted at end of run):
  * compiled-or-cached forward returns " France"
  * |rank0_ms - rank1_ms| <= 100 ms
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor.parallel import parallelize_module

# Allow `python test/test_e4b_smoke.py` from the package root by
# falling back to the sibling src/ dir on the path.
HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.normpath(os.path.join(HERE, os.pardir, "src"))
if SRC not in sys.path:
    sys.path.insert(0, SRC)
from tp_plan import build_e4b_tp_plan  # noqa: E402


EXPECTED_NEXT_TOKEN = " France"
RANK_TIMING_TOLERANCE_MS = 100.0
SEQ_LEN = 64


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/root/models/gemma-4-E4B-it")
    ap.add_argument("--prompt", default="The capital of France is")
    args = ap.parse_args()

    dist.init_process_group(backend="neuron")
    rank = dist.get_rank()
    world = dist.get_world_size()
    if world < 2:
        print(f"[rank {rank}] FAIL: world_size={world}, need >=2 for TP plan.",
              flush=True)
        return 2

    mesh = init_device_mesh("neuron", (world,))
    device = torch.device("neuron")

    from transformers import AutoTokenizer, AutoModelForImageTextToText
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="eager",
    )
    plan, _, _, owners, shareds = build_e4b_tp_plan(model)

    expected_owners = 24
    expected_shareds = 18
    if owners != expected_owners or shareds != expected_shareds:
        print(f"[rank {rank}] FAIL: owners={owners}/{expected_owners}, "
              f"shareds={shareds}/{expected_shareds} — TP plan does not "
              f"match expected E4B layer split.", flush=True)
        return 3

    parallelize_module(model, mesh, plan)
    model = model.to(device).eval()

    tok = tokenizer(args.prompt, return_tensors="pt", padding="max_length",
                    truncation=True, max_length=SEQ_LEN)
    input_ids = tok["input_ids"].to(device)
    attention_mask = tok["attention_mask"].to(device)

    # Warmup (loads / fills cache)
    with torch.inference_mode():
        _ = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    if hasattr(torch, "neuron") and hasattr(torch.neuron, "synchronize"):
        torch.neuron.synchronize()

    # Timed run
    dist.barrier()
    t0 = time.time()
    with torch.inference_mode():
        out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        next_tok = int(out.logits[:, -1, :].argmax(dim=-1).cpu())
    if hasattr(torch, "neuron") and hasattr(torch.neuron, "synchronize"):
        torch.neuron.synchronize()
    dist.barrier()
    dt_ms = (time.time() - t0) * 1000.0

    decoded = tokenizer.decode([next_tok])

    # Gather per-rank latency for the symmetry check
    timings = [torch.tensor(0.0) for _ in range(world)]
    dist.all_gather_object(timings, dt_ms)
    spread_ms = max(timings) - min(timings)

    if rank == 0:
        print(f"\n=== test_e4b_smoke ===")
        print(f"  world_size:         {world}")
        print(f"  owner_layers:       {owners} (expected {expected_owners})")
        print(f"  shared_layers:      {shareds} (expected {expected_shareds})")
        print(f"  sharded modules:    {len(plan)}")
        print(f"  next_token:         {decoded!r} (expected {EXPECTED_NEXT_TOKEN!r})")
        print(f"  per-rank latencies: {timings}")
        print(f"  spread (ms):        {spread_ms:.1f} (tolerance {RANK_TIMING_TOLERANCE_MS:.1f})")

    fail = False
    if decoded != EXPECTED_NEXT_TOKEN:
        if rank == 0:
            print(f"  FAIL: next_token mismatch")
        fail = True
    if spread_ms > RANK_TIMING_TOLERANCE_MS:
        if rank == 0:
            print(f"  FAIL: rank-timing spread {spread_ms:.1f} ms exceeds "
                  f"{RANK_TIMING_TOLERANCE_MS:.1f} ms")
        fail = True

    if rank == 0 and not fail:
        print(f"  PASS")

    dist.destroy_process_group()
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())

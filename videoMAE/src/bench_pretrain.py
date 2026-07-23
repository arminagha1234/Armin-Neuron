"""Steady-state throughput benchmark for native VideoMAE v2 pretraining on Trainium.

Measures the training step (fwd + bwd + optimizer) throughput, skipping the step-0
compile via warmup. Sweeps batch size and dtype. Reports median step time, videos/sec,
and peak device memory. Because the beta runs async, we force a sync each step via
loss.item().

Usage:
  python bench_pretrain.py --device neuron --batches 1,2,4,8 --dtypes fp32,bf16
  python bench_pretrain.py --device cpu   --batches 1,2 --dtypes fp32 --iters 5
"""
import argparse
import statistics
import time

import numpy as np
import torch

from modeling_pretrain_native import build_pretrain_videomae_base, tube_mask_indices
from pretrain_neuron import make_structured_clips, make_target, Tp, Hp, Wp

DT = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}


def bench_one(model, device, batch, dtype, warmup, iters, mask_ratio=0.9):
    rng = np.random.RandomState(0)
    images = make_structured_clips(batch, rng).to(device).to(dtype)
    ids_keep, ids_mask = tube_mask_indices(batch, Tp, Hp, Wp, mask_ratio, rng)
    ids_keep, ids_mask = ids_keep.to(device), ids_mask.to(device)
    with torch.no_grad():
        target = make_target(make_structured_clips(batch, rng)).to(device).to(dtype)
        Cpx = target.shape[-1]
        labels = torch.gather(target, 1, ids_mask.unsqueeze(-1).expand(-1, -1, Cpx))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)

    def step():
        opt.zero_grad()
        out = model(images, ids_keep, ids_mask)
        loss = ((out - labels) ** 2).mean()
        loss.backward()
        opt.step()
        return float(loss.detach().cpu())  # forces sync

    try:
        import torch_neuronx
        torch_neuronx.reset_peak_memory_stats(device)
    except Exception:
        pass

    for _ in range(warmup):
        step()
    ts = []
    for _ in range(iters):
        t0 = time.time()
        step()
        ts.append(time.time() - t0)
    med = statistics.median(ts)

    peak_gb = float("nan")
    try:
        import torch_neuronx
        peak_gb = torch_neuronx.max_memory_allocated(device) / 1e9
    except Exception:
        pass
    return med, batch / med, peak_gb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="neuron", choices=["cpu", "neuron"])
    ap.add_argument("--batches", default="1,2,4,8")
    ap.add_argument("--dtypes", default="fp32,bf16")
    ap.add_argument("--warmup", type=int, default=6)
    ap.add_argument("--iters", type=int, default=15)
    ap.add_argument("--compile", action="store_true",
                    help="wrap model in torch.compile(backend='neuron')")
    args = ap.parse_args()

    if args.device == "neuron":
        import torch_neuronx  # noqa: F401

    dev = torch.device(args.device)
    batches = [int(b) for b in args.batches.split(",")]
    dtypes = [d.strip() for d in args.dtypes.split(",")]

    mode = "torch.compile(neuron)" if args.compile else "eager"
    print(f"# VideoMAE v2 pretraining throughput  (device={args.device}, mode={mode}, warmup={args.warmup}, iters={args.iters})")
    print(f"{'dtype':>6} {'batch':>6} {'step_ms':>9} {'videos/s':>9} {'peak_GB':>8}")
    for d in dtypes:
        torch.manual_seed(0)
        model = build_pretrain_videomae_base().train().to(dev).to(DT[d])
        if args.compile:
            backend = "neuron" if args.device == "neuron" else "inductor"
            # dynamic=False: force a static graph per shape (beta rejects dynamic shapes)
            model = torch.compile(model, backend=backend, dynamic=False)
        for b in batches:
            try:
                med, vps, peak = bench_one(model, dev, b, DT[d], args.warmup, args.iters)
                print(f"{d:>6} {b:>6} {med*1e3:>9.1f} {vps:>9.2f} {peak:>8.2f}", flush=True)
            except Exception as e:
                print(f"{d:>6} {b:>6}   FAILED: {repr(e)[:220]}", flush=True)
    print("BENCH_DONE", flush=True)


if __name__ == "__main__":
    main()

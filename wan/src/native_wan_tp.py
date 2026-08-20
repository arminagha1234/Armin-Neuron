"""
SP/TP native-PyTorch Wan 2.2 TI2V-5B DiT on Trn2 (torchrun + DTensor).

Adapts the LTX-2 DiT TP plan (knowledge-pack/parallelism/ltx2_tp_plan.py) to the
diffusers Wan model. Shards attention (to_q/k/v Colwise, to_out.0 Rowwise) + FFN
(net.0.proj Colwise, net.2 Rowwise), patches attn.heads -> heads/world, and
installs the EXACT across-heads adaptive QK-RMSNorm (all-reduce the sum-of-squares
so the norm denominator is the full 3072, not the local shard). Wan's RoPE is
position-only / head-shared, so NO per-rank RoPE slice is needed.

Also applies the official Wan-port P0 compile flags via NEURON_CC_FLAGS (set in
the launcher). Dumps the DiT output on rank 0 so TP=N can be parity-checked
(cosine) against TP=1.

  torchrun --nnodes 1 --nproc_per_node <N> native_wan_tp.py [--compile]
"""
from __future__ import annotations
import argparse, os, time
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor.parallel import (
    ColwiseParallel, RowwiseParallel, parallelize_module)

MODEL = "/home/ubuntu/wan22"
OUTDIR = "/home/ubuntu/kernel_research"


def _r0():
    return int(os.environ.get("RANK", "0")) == 0


def _log(m):
    if _r0():
        print(f"[wantp] {m}", flush=True)


class _AdaptiveQKNorm(nn.Module):
    """Across-heads RMSNorm that is correct under Colwise head-sharding: local
    sum-of-squares, all-reduce for the global RMS over the FULL inner dim, apply
    this rank's slice of the (replicated) weight. (LTX-2 fix #4, verbatim.)"""
    def __init__(self, weight, eps, world, rank):
        super().__init__()
        self.full_weight = weight
        self.eps = eps or 1e-6
        self.world = world
        self.rank = rank

    def forward(self, x):
        in_dim = x.shape[-1]
        full = self.full_weight.shape[0]
        sharded = in_dim < full
        local_sq = (x.float() ** 2).sum(dim=-1, keepdim=True)
        if sharded and self.world > 1 and dist.is_initialized():
            dist.all_reduce(local_sq, op=dist.ReduceOp.SUM)
            denom = full
            w = self.full_weight.narrow(0, self.rank * in_dim, in_dim)
        else:
            denom = in_dim
            w = self.full_weight if in_dim == full else self.full_weight.narrow(0, 0, in_dim)
        rms = (local_sq / denom + self.eps).rsqrt()
        return (x.float() * rms).to(x.dtype) * w


def _sync(o):
    o = o[0] if isinstance(o, (tuple, list)) else o
    o = getattr(o, "sample", o)
    if hasattr(o, "to_local"):
        o = o.to_local()
    return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--batch", type=int, default=1)  # CFG-batching: B=2 folds cond+uncond into one forward
    ap.add_argument("--profile", action="store_true")  # host-vs-device probe + Perfetto trace
    a = ap.parse_args()

    dist.init_process_group(backend="neuron")
    world = dist.get_world_size()
    rank = dist.get_rank()
    dev = torch.device("neuron")

    from diffusers.models.transformers.transformer_wan import WanTransformer3DModel
    t0 = time.time()
    dit = WanTransformer3DModel.from_pretrained(
        MODEL, subfolder="transformer", torch_dtype=torch.bfloat16)
    cfg = dit.config
    mesh = init_device_mesh("neuron", (world,))

    if world > 1:
        n = 0
        for blk in dit.blocks:
            plan = {}
            for an in ("attn1", "attn2"):
                if hasattr(blk, an):
                    plan[f"{an}.to_q"] = ColwiseParallel()
                    plan[f"{an}.to_k"] = ColwiseParallel()
                    plan[f"{an}.to_v"] = ColwiseParallel()
                    plan[f"{an}.to_out.0"] = RowwiseParallel()
            plan["ffn.net.0.proj"] = ColwiseParallel()
            plan["ffn.net.2"] = RowwiseParallel()
            parallelize_module(blk, mesh, plan)
            n += 1
        # patch heads -> heads/world (unflatten uses attn.heads on the sharded dim)
        for blk in dit.blocks:
            for an in ("attn1", "attn2"):
                at = getattr(blk, an, None)
                if at is not None and hasattr(at, "heads"):
                    at.heads = at.heads // world
        # exact across-heads adaptive QK-norm
        for blk in dit.blocks:
            for an in ("attn1", "attn2"):
                at = getattr(blk, an, None)
                if at is None:
                    continue
                for nm in ("norm_q", "norm_k"):
                    norm = getattr(at, nm, None)
                    if norm is not None and getattr(norm, "weight", None) is not None:
                        setattr(at, nm, _AdaptiveQKNorm(
                            norm.weight, getattr(norm, "eps", 1e-6), world, rank))
        _log(f"sharded {n} blocks, heads->{24//world}, adaptive QK-norm installed")

    dit = dit.to(dev).eval()
    _log(f"world={world} load+shard {time.time()-t0:.1f}s")

    B, C, T, Hl, Wl = a.batch, cfg.in_channels, 13, 30, 52
    torch.manual_seed(0)   # identical inputs across ranks/world for parity
    hidden = torch.randn(B, C, T, Hl, Wl, dtype=torch.bfloat16, device=dev)
    timestep = torch.full((B,), 1000.0, dtype=torch.bfloat16, device=dev)
    enc = torch.randn(B, 512, cfg.text_dim, dtype=torch.bfloat16, device=dev)

    def fwd():
        with torch.no_grad():
            return dit(hidden_states=hidden, timestep=timestep,
                       encoder_hidden_states=enc, return_dict=False)

    run = fwd
    tag = "eager"
    if a.compile:
        cm = torch.compile(dit, backend="neuron", dynamic=False)
        def crun():
            with torch.no_grad():
                return cm(hidden_states=hidden, timestep=timestep,
                          encoder_hidden_states=enc, return_dict=False)
        run = crun
        tag = "compiled"

    t0 = time.time(); out = run(); o = _sync(out); float(o.flatten()[:1].cpu())
    _log(f"first({tag}) {time.time()-t0:.1f}s")
    for _ in range(2):
        float(_sync(run()).flatten()[:1].cpu())
    t0 = time.time()
    for _ in range(a.iters):
        float(_sync(run()).flatten()[:1].cpu())
    ms = (time.time() - t0) / a.iters * 1000
    _log(f"TP={world} {tag}: {ms:.0f} ms/forward  (baseline TP1 compiled 943)")

    if a.profile:
        # (1) host-vs-device proxy: `ms` above is SYNCED per-iter (host+device serialized).
        # Here we dispatch all iters and sync ONCE => pipelined/fwd. All ranks run it
        # (collectives need every rank); if pipelined << synced, host dispatch dominates.
        t0 = time.time()
        outs = [run() for _ in range(a.iters)]
        float(_sync(outs[-1]).flatten()[:1].cpu())
        pipe_ms = (time.time() - t0) / a.iters * 1000
        # (2) Perfetto trace + device-time (rank 0); other ranks run a matching loop.
        if _r0():
            dev_note = ""
            try:
                from torch.profiler import profile, ProfilerActivity
                acts = [ProfilerActivity.CPU]
                if hasattr(ProfilerActivity, "PrivateUse1"):
                    acts.append(ProfilerActivity.PrivateUse1)
                with profile(activities=acts) as prof:
                    for _ in range(a.iters):
                        float(_sync(run()).flatten()[:1].cpu())
                ka = prof.key_averages()
                dev = sum(getattr(e, "self_device_time_total", 0) for e in ka) / 1e3 / a.iters
                prof.export_chrome_trace(f"{OUTDIR}/dit_tp{world}_trace.json")
                dev_note = f"profiler device/fwd={dev:.0f}ms" if dev > 0 else "profiler device-time N/A"
            except Exception as e:  # noqa: BLE001
                dev_note = f"profiler skipped ({type(e).__name__})"
            host = max(0.0, ms - pipe_ms)
            verdict = "HOST-BOUND" if pipe_ms < 0.75 * ms else "COMPUTE-BOUND"
            _log(f"PROFILE synced/fwd={ms:.0f}ms  pipelined/fwd={pipe_ms:.0f}ms  "
                 f"host_overhead~{host:.0f}ms ({100*host/ms:.0f}%)  => {verdict};  {dev_note}")
        else:
            for _ in range(a.iters):
                float(_sync(run()).flatten()[:1].cpu())

    # parity dump: ALL ranks must run the forward (it contains collectives —
    # QK-norm all-reduce + Rowwise all-gather — that hang if a rank is absent);
    # only rank 0 saves the signature for the TP=N vs TP=1 cosine.
    o = _sync(run()).float().cpu().flatten()
    if _r0():
        torch.save(o, f"{OUTDIR}/wan_out_tp{world}_{tag}.pt")
        _log(f"saved output sig -> wan_out_tp{world}_{tag}.pt  norm={o.norm().item():.3f}")
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
    os._exit(0)


if __name__ == "__main__":
    main()

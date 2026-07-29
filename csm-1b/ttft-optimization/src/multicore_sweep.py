"""Multi-core throughput sweep (measured TP-adjacent parallelism, no TP port needed).

CSM decode is single-core, but a SERVICE runs many independent frame/tile streams. This
measures aggregate throughput as N independent single-core workers run in parallel, each
pinned to its own NeuronCore (NEURON_RT_VISIBLE_CORES=k). Answers: how many concurrent CSM
streams can one trn2 box serve, and does per-worker latency degrade (NUMA / memory-BW
contention) as workers scale?

This is (b) in the TP ask — real, runnable now (unlike a TP-sharded backbone, which is a
port; see TP_ANALYSIS_AND_PLAN.md). Each worker times the compiled backbone decode step.

Driver: this script is the WORKER (one core). A shell driver launches N of them with
different NEURON_RT_VISIBLE_CORES and aggregates. Run one worker:
    NEURON_RT_VISIBLE_CORES=3 python3 multicore_sweep.py --tag w3 --iters 30
"""
import os
import sys
import time
import argparse
import statistics
import torch
import torch_neuronx  # noqa
from transformers import CsmForConditionalGeneration

M = os.environ.get("CSM_MODEL", "/host/csm_1b")
DEV = torch.device("neuron")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="w")
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--n", type=int, default=1024, help="prefill length to time")
    args = ap.parse_args()

    t_load = time.perf_counter()
    model = CsmForConditionalGeneration.from_pretrained(M, dtype=torch.bfloat16).eval()
    bb = model.backbone_model.to(DEV)
    for m in bb.modules():
        for k, v in list(vars(m).items()):
            if torch.is_tensor(v) and not getattr(v, "is_meta", False) and v.device.type != "neuron":
                setattr(m, k, v.to(DEV))
    H = bb.config.hidden_size
    N = args.n
    emb = torch.randn(1, N, H, dtype=torch.bfloat16, device=DEV)
    pos = torch.arange(N, device=DEV).unsqueeze(0)
    cbb = torch.compile(bb, backend="neuron", dynamic=False)

    def run():
        with torch.no_grad():
            return cbb(inputs_embeds=emb, position_ids=pos, use_cache=False).last_hidden_state.cpu()

    run()  # compile
    load_s = time.perf_counter() - t_load
    ts = []
    t0 = time.perf_counter()
    for _ in range(args.iters):
        t = time.perf_counter(); run(); ts.append((time.perf_counter() - t) * 1000)
    wall = time.perf_counter() - t0
    med = statistics.median(ts)
    thru = args.iters / wall  # prefills/sec for this worker
    core = os.environ.get("NEURON_RT_VISIBLE_CORES", "?")
    print(f"[mc] tag={args.tag} core={core} N={N}: median={med:.1f}ms thru={thru:.2f}/s "
          f"(load+compile {load_s:.0f}s)", flush=True)


if __name__ == "__main__":
    sys.exit(main())

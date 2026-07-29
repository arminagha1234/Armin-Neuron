"""RIGOROUS single-config prefill verification — ONE length per fresh process (no dynamo
cache/guard contamination), dynamo.explain graph-break check, median+std over many iters,
compile-time separated from run-time. Answers: is the 4k 'compiled slower than eager'
real, or a graph break / recompile artifact?

Run per length: python3 prefill_verify.py --n 4096 --dtype bf16 --iters 20
"""
import os
import sys
import time
import argparse
import statistics
import torch
import torch_neuronx  # noqa
import torch._dynamo as dynamo
from transformers import CsmForConditionalGeneration

M = os.environ.get("CSM_MODEL", "/host/csm_1b")
DEV = torch.device("neuron")


def timed(fn, iters, warm=3):
    for _ in range(warm):
        fn()
    ts = []
    for _ in range(iters):
        t = time.perf_counter(); fn(); ts.append((time.perf_counter() - t) * 1000)
    return min(ts), statistics.median(ts), statistics.pstdev(ts), max(ts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--dtype", default="bf16")
    ap.add_argument("--iters", type=int, default=20)
    args = ap.parse_args()
    N = args.n
    dt = torch.bfloat16 if args.dtype == "bf16" else torch.float32

    model = CsmForConditionalGeneration.from_pretrained(M, dtype=dt).eval()
    bb = model.backbone_model.to(DEV)
    for m in bb.modules():
        for k, v in list(vars(m).items()):
            if torch.is_tensor(v) and not getattr(v, "is_meta", False) and v.device.type != "neuron":
                setattr(m, k, v.to(DEV))
    H = bb.config.hidden_size
    emb = torch.randn(1, N, H, dtype=dt, device=DEV)
    pos = torch.arange(N, device=DEV).unsqueeze(0)

    def eager():
        with torch.no_grad():
            return bb(inputs_embeds=emb, position_ids=pos, use_cache=False).last_hidden_state.cpu()

    # --- graph-break inspection (the key check for the 4k reversal) ---
    def fwd():
        return bb(inputs_embeds=emb, position_ids=pos, use_cache=False).last_hidden_state
    try:
        with torch.no_grad():
            expl = dynamo.explain(fwd)()
        print(f"[v] N={N}: dynamo.explain graph_count={expl.graph_count} "
              f"graph_break_count={expl.graph_break_count} op_count={expl.op_count}", flush=True)
        for i, b in enumerate(expl.break_reasons[:5]):
            print(f"[v]   break[{i}]: {getattr(b,'reason',b)}", flush=True)
    except Exception as e:
        print(f"[v] N={N}: dynamo.explain failed: {type(e).__name__}: {str(e)[:120]}", flush=True)

    # --- eager timing ---
    e_best, e_med, e_std, e_max = timed(eager, args.iters)

    # --- compiled timing (fresh dynamo state) ---
    dynamo.reset()
    if hasattr(torch_neuronx, "reset_dynamo_metrics"):
        torch_neuronx.reset_dynamo_metrics()
    cbb = torch.compile(bb, backend="neuron", dynamic=False, fullgraph=False)

    def comp():
        with torch.no_grad():
            return cbb(inputs_embeds=emb, position_ids=pos, use_cache=False).last_hidden_state.cpu()

    t = time.perf_counter(); comp(); compile_s = time.perf_counter() - t
    # how many NEFFs actually compiled?
    try:
        m = torch_neuronx.get_dynamo_metrics()
        print(f"[v] N={N}: NEFFs compiled={len(m)} nodes={[g.graph_node_count for g in m]}", flush=True)
    except Exception:
        pass
    c_best, c_med, c_std, c_max = timed(comp, args.iters)

    # correctness: compiled vs eager cosine
    with torch.no_grad():
        a = comp().float().flatten(); b = eager().float().flatten()
    cos = torch.nn.functional.cosine_similarity(a, b, dim=0).item()

    print(f"[v] === N={N} ({args.dtype}) ===", flush=True)
    print(f"[v] EAGER    best={e_best:.1f} median={e_med:.1f} std={e_std:.1f} max={e_max:.1f} ms", flush=True)
    print(f"[v] COMPILED best={c_best:.1f} median={c_med:.1f} std={c_std:.1f} max={c_max:.1f} ms "
          f"(compile {compile_s:.0f}s)", flush=True)
    print(f"[v] speedup(median) eager/comp = {e_med/max(c_med,1e-6):.2f}x | cos(comp,eager)={cos:.5f}", flush=True)


if __name__ == "__main__":
    sys.exit(main())

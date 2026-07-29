"""FAIR on-device test of the CSM-1B depth decoder — the config the NeuronCore is built for.

Prior device runs were UNFAIR: per-step eager dispatch + host sync ran 12-80x above the
NeuronCore HBM floor. This harness runs the WHOLE 31-step depth loop as ONE compiled graph
(backend="neuron"), weights resident on device (read once, reused), argmax on-device, and
exactly ONE host boundary at the very end (.cpu() on the final [B,K] codebook tensor). No
.item() / no host sync mid-loop.

Ladder:
  --mode eager    : the per-step eager baseline (unfair reference; ~1106 ms reported prior)
  --mode compile  : torch.compile(backend="neuron") on the full unrolled K-step loop  (THE fair test)

Correctness oracle: a CPU fp32 serial depth decode (same math). We require argmax match.

Reports torch._dynamo metrics (graph count / breaks) so a failure to capture the loop as a
single graph is surfaced explicitly with the reason, not hidden.
"""
import os
import sys
import time
import argparse
import torch
import torch_neuronx  # noqa: F401  registers the neuron device + dynamo backend
from transformers import AutoProcessor, CsmForConditionalGeneration
from transformers.cache_utils import StaticCache

MODEL = os.environ.get("CSM_MODEL", "/host/csm_1b")
DEV = torch.device("neuron")


def capture_depth_inputs(model, proc, text):
    """Run the full model once (CPU) and capture the (backbone_last_hidden_state,
    first_codebook_id) the parent feeds the depth decoder + the stock ref codebooks."""
    cap = {}
    dd = model.depth_decoder
    orig = dd.generate

    def spy(*a, **k):
        cap["input_ids"] = k["input_ids"].detach().clone()
        cap["backbone_last_hidden_state"] = k["backbone_last_hidden_state"].detach().clone()
        out = orig(*a, **k)
        seq = out if isinstance(out, torch.Tensor) else out.sequences
        cap["ref_codebooks"] = seq.detach().clone()
        return out

    dd.generate = spy
    inputs = proc(text, add_special_tokens=True, return_tensors="pt")
    with torch.no_grad():
        model.generate(**inputs, output_audio=False, do_sample=False, max_new_tokens=1)
    dd.generate = orig
    return cap


def make_depth_loop(model, num_codebooks):
    """Return a callable depth_loop(backbone_hidden, first_cb, K) that runs the full serial
    depth decode as ONE python-unrolled loop of the HF depth forward, argmax on-device,
    static-index (OOB-free), StaticCache resident. K is a python int (compile-time constant
    so the loop fully unrolls into a single graph)."""
    dd = model.depth_decoder
    mdl = dd.model
    head = dd.codebooks_head

    @torch.no_grad()
    def depth_loop(backbone_hidden, first_cb, K):
        dev = backbone_hidden.device
        B = backbone_hidden.shape[0]
        vocab = mdl.vocab_size
        cache = StaticCache(config=mdl.config, max_batch_size=B,
                            max_cache_len=num_codebooks, device=dev,
                            dtype=backbone_hidden.dtype)
        # prefill positions [0,1]: pos0 = backbone hidden, pos1 = codebook 0
        ids = torch.cat([torch.zeros((B, 1), dtype=torch.long, device=dev),
                         first_cb.view(B, 1)], dim=1)
        emb = mdl.embed_tokens(ids).clone()
        emb[:, 0] = backbone_hidden
        cp = torch.arange(0, 2, device=dev)
        out = mdl(inputs_embeds=emb, past_key_values=cache, cache_position=cp,
                  use_cache=True)
        cb = torch.argmax(out.last_hidden_state[:, -1, :] @ head.weight[0], dim=-1)
        outs = [first_cb.view(B), cb]
        for k in range(2, K):
            emb = mdl.embed_tokens(cb.view(B, 1) + (k - 1) * vocab)
            cp = torch.tensor([k], device=dev)
            out = mdl(inputs_embeds=emb, past_key_values=cache, cache_position=cp,
                      use_cache=True)
            cb = torch.argmax(out.last_hidden_state[:, -1, :] @ head.weight[k - 1], dim=-1)
            outs.append(cb)
        return torch.stack(outs, dim=1)  # [B, K] on device

    return depth_loop


def cpu_fp32_reference(model, bb_hidden_cpu, first_cb_cpu, num_codebooks):
    """Serial depth decode in fp32 on CPU — the correctness oracle."""
    dd = model.depth_decoder
    mdl = dd.model.float()
    head = dd.codebooks_head.float()
    B = bb_hidden_cpu.shape[0]
    vocab = mdl.vocab_size
    with torch.no_grad():
        cache = StaticCache(config=mdl.config, max_batch_size=B,
                            max_cache_len=num_codebooks, device="cpu",
                            dtype=torch.float32)
        ids = torch.cat([torch.zeros((B, 1), dtype=torch.long), first_cb_cpu.view(B, 1)], dim=1)
        emb = mdl.embed_tokens(ids).clone()
        emb[:, 0] = bb_hidden_cpu.float()
        out = mdl(inputs_embeds=emb, past_key_values=cache,
                  cache_position=torch.arange(0, 2), use_cache=True)
        cb = torch.argmax(out.last_hidden_state[:, -1, :] @ head.weight[0], dim=-1)
        outs = [first_cb_cpu.view(B), cb]
        for k in range(2, num_codebooks):
            emb = mdl.embed_tokens(cb.view(B, 1) + (k - 1) * vocab)
            out = mdl(inputs_embeds=emb, past_key_values=cache,
                      cache_position=torch.tensor([k]), use_cache=True)
            cb = torch.argmax(out.last_hidden_state[:, -1, :] @ head.weight[k - 1], dim=-1)
            outs.append(cb)
    return torch.stack(outs, dim=1)  # [B, num_codebooks]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    ap.add_argument("--mode", choices=["eager", "compile"], default="compile")
    ap.add_argument("--k", type=int, default=32, help="stop point (full = num_codebooks)")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--capture-scalar", action="store_true",
                    help="set dynamo capture_scalar_outputs=True to trace through .item()")
    args = ap.parse_args()
    if args.capture_scalar:
        import torch._dynamo as _d
        _d.config.capture_scalar_outputs = True
        print("[fair] dynamo.config.capture_scalar_outputs = True", flush=True)
    dt = torch.bfloat16 if args.dtype == "bf16" else torch.float32

    print(f"[fair] loading {args.model} (dtype={args.dtype}, mode={args.mode})...", flush=True)
    proc = AutoProcessor.from_pretrained(args.model)
    model = CsmForConditionalGeneration.from_pretrained(args.model, dtype=dt).eval()
    model.codec_model = model.codec_model.float()
    num_cb = model.config.num_codebooks
    K = min(args.k, num_cb)
    gc = model.depth_decoder.generation_config
    gc.do_sample = False; gc.temperature = None; gc.top_k = None; gc.top_p = None

    print("[fair] capturing depth inputs + CPU fp32 reference...", flush=True)
    cap = capture_depth_inputs(model, proc, "[0]Hello from Trainium, latency test.")
    first_cb_cpu = cap["input_ids"][:, 1].clone()
    bb_hidden_cpu = cap["backbone_last_hidden_state"].clone()

    # correctness oracle (fp32 CPU serial) — do this BEFORE moving depth to device
    ref = cpu_fp32_reference(model, bb_hidden_cpu, first_cb_cpu, num_cb).cpu()
    # restore depth model dtype (cpu_fp32_reference floated it in place)
    model.depth_decoder.model = model.depth_decoder.model.to(dt)
    model.depth_decoder.codebooks_head = model.depth_decoder.codebooks_head.to(dt)

    # move depth decoder resident on device (read once, reused across steps)
    model.depth_decoder.to(DEV)
    for m in model.depth_decoder.modules():
        for k, v in list(vars(m).items()):
            if torch.is_tensor(v) and not getattr(v, "is_meta", False) \
                    and v.device.type != "neuron":
                setattr(m, k, v.to(DEV))
    bb_d = bb_hidden_cpu.to(dt).to(DEV)
    fcb_d = first_cb_cpu.to(DEV)

    depth_loop = make_depth_loop(model, num_cb)

    import torch._dynamo as dynamo
    if args.mode == "compile":
        torch_neuronx.reset_dynamo_metrics()
        dynamo.reset()
        # capture explain (graph breaks) once, on the real inputs
        try:
            expl = dynamo.explain(lambda: depth_loop(bb_d, fcb_d, K))()
            print(f"[fair] dynamo.explain: graph_count={expl.graph_count} "
                  f"graph_break_count={expl.graph_break_count} op_count={expl.op_count}",
                  flush=True)
            if expl.graph_break_count:
                for i, brk in enumerate(expl.break_reasons):
                    print(f"[fair]   break[{i}]: {getattr(brk, 'reason', brk)}", flush=True)
        except Exception as e:
            print(f"[fair] dynamo.explain failed: {type(e).__name__}: {e}", flush=True)
        run = torch.compile(depth_loop, backend="neuron", dynamic=False)
    else:
        run = depth_loop

    print(f"[fair] warm/compile pass (K={K})...", flush=True)
    t = time.perf_counter()
    part = run(bb_d, fcb_d, K).cpu()  # one host boundary at the end
    warm_ms = (time.perf_counter() - t) * 1000
    print(f"[fair] first (compile+run) pass: {warm_ms:.1f} ms", flush=True)

    if args.mode == "compile":
        try:
            m = torch_neuronx.get_dynamo_metrics()
            print(f"[fair] neuron dynamo metrics: {m}", flush=True)
        except Exception as e:
            print(f"[fair] get_dynamo_metrics: {e}", flush=True)

    # timed warm iterations — single host boundary each
    best = 1e9; times = []
    for _ in range(args.iters):
        t = time.perf_counter()
        run(bb_d, fcb_d, K).cpu()
        ms = (time.perf_counter() - t) * 1000
        times.append(ms); best = min(best, ms)
    times.sort()
    med = times[len(times) // 2]
    print(f"[fair] warm depth ms/frame (K={K}): best={best:.2f} median={med:.2f} "
          f"max={times[-1]:.2f}", flush=True)

    # correctness vs fp32 CPU oracle
    n = min(K, ref.shape[1])
    match = int((part[:, :n] == ref[:, :n]).all())
    n_agree = int((part[:, :n] == ref[:, :n]).sum())
    total = part[:, :n].numel()
    print(f"[fair] argmax match vs CPU fp32 oracle (first {n}): "
          f"{'EXACT' if match else 'DIFF'} ({n_agree}/{total} codebooks agree)", flush=True)
    print(f"[fair] device codes[:, :{n}] = {part[:, :n].tolist()}", flush=True)
    print(f"[fair] oracle codes[:, :{n}] = {ref[:, :n].tolist()}", flush=True)

    CPU_MS = 137.0; FLOOR_MS = 14.0
    print(f"\n[fair] === RESULT ({args.mode}, {args.dtype}, K={K}) ===", flush=True)
    print(f"[fair] depth {best:.2f} ms/frame | vs 137ms CPU: {CPU_MS/best:.2f}x | "
          f"vs ~14ms floor: {best/FLOOR_MS:.2f}x-above | "
          f"argmax {'EXACT' if match else 'DIFF'}", flush=True)


if __name__ == "__main__":
    sys.exit(main())

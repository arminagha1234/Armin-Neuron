"""FAIR on-device test v2 — hand-rolled single-graph depth loop.

fair_depth_neuronx.py proved the STOCK HF depth forward cannot fuse: it calls
`.item()` on the KV-cache seq length (position_ids = arange(past_seen, ...)) and builds
create_causal_mask per step, forcing a data-dependent graph break EVERY step (3 breaks,
4 subgraphs, per-step host sync). That is the exact "unfair" per-step-sync wall, just with
compiled kernels — 138 ms, 0.99x CPU.

This harness hand-rolls the depth transformer step (reusing the real weights) with:
  * python-int positions  -> the K-step loop UNROLLS into a single fixed-shape graph
  * a plain-list KV cache (torch.cat), NO Cache object, NO .item(), NO create_causal_mask
  * argmax on-device, exactly ONE host boundary (.cpu()) at the very end
so torch.compile(backend="neuron") can capture the WHOLE 31-step loop as ONE graph with
weights RESIDENT (read once, reused). This is THE fair config.

Config (dumped from /host/csm_1b depth decoder):
  layers=4 hidden=1024 backbone_hidden=2048 mlp_inter=8192 nh=8 nkv=2 hd=128
  vocab=2051 num_codebooks=32 rope=llama3 rms_eps=1e-5 act=silu, no biases.

Correctness oracle: CPU fp32 serial decode via the same hand-rolled math (argmax match).
"""
import os
import sys
import time
import argparse
import torch
import torch_neuronx  # noqa: F401
from transformers import AutoProcessor, CsmForConditionalGeneration

MODEL = os.environ.get("CSM_MODEL", "/host/csm_1b")
DEV = torch.device("neuron")


def capture_depth_inputs(model, proc, text):
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


def _q8(w, dev, compute_dtype):
    """Per-output-channel symmetric int8 quant of a linear weight [out, in].
    Returns (w_int8 [out,in] int8, scale [out,1] compute_dtype). Dequant = w_int8*scale."""
    wf = w.detach().float()
    amax = wf.abs().amax(dim=1, keepdim=True).clamp_(min=1e-8)  # [out,1]
    scale = amax / 127.0
    q = torch.clamp(torch.round(wf / scale), -127, 127).to(torch.int8)
    return q.to(dev).contiguous(), scale.to(dev, compute_dtype).contiguous()


class DepthWeights:
    """Flat handle on the depth decoder's real weights (all on one device/dtype)."""

    def __init__(self, dd, dev, dtype, quant=None):
        self.quant = quant  # None or "int8" (weight-only, MLP+attn linears)
        m = dd.model
        self.nL = m.config.num_hidden_layers
        self.nh = m.config.num_attention_heads
        self.nkv = m.config.num_key_value_heads
        self.hd = m.config.head_dim
        self.groups = self.nh // self.nkv
        self.scaling = self.hd ** -0.5
        self.eps = m.config.rms_norm_eps
        self.vocab = m.config.vocab_size

        def g(t):
            return t.detach().to(dev, dtype).contiguous()

        # quantize the big linears (q/k/v/o/gate/up/down) to int8 weight-only if requested;
        # embed/proj/norm/head stay in compute dtype (head is the argmax matmul; embed is a
        # gather not a GEMV). This halves the ~230MB bandwidth-dominant term (88% is MLP).
        def lin(w):
            return _q8(w, dev, dtype) if quant == "int8" else g(w)

        self.embed = g(m.embed_tokens.weight)          # [num_cb*vocab, 2048]
        self.proj = g(m.inputs_embeds_projector.weight)  # [1024, 2048]
        self.norm_w = g(m.norm.weight)                   # [1024]
        self.head = g(dd.codebooks_head.weight)          # [31, 1024, 2051]
        self.L = []
        for i in range(self.nL):
            lyr = m.layers[i]
            self.L.append(dict(
                ln1=g(lyr.input_layernorm.weight),
                ln2=g(lyr.post_attention_layernorm.weight),
                q=lin(lyr.self_attn.q_proj.weight),
                k=lin(lyr.self_attn.k_proj.weight),
                v=lin(lyr.self_attn.v_proj.weight),
                o=lin(lyr.self_attn.o_proj.weight),
                gate=lin(lyr.mlp.gate_proj.weight),
                up=lin(lyr.mlp.up_proj.weight),
                down=lin(lyr.mlp.down_proj.weight),
            ))
        # rope cos/sin table [max_len, hd] on device
        max_len = m.config.num_codebooks + 1
        with torch.no_grad():
            pos = torch.arange(max_len, device="cpu")[None].float()
            dummy = torch.zeros(1, 1, self.hd)
            cos, sin = m.rotary_emb.to("cpu")(dummy, position_ids=pos)  # [1, max_len, hd]
        self.cos = cos[0].to(dev, dtype)  # [max_len, hd]
        self.sin = sin[0].to(dev, dtype)


def rmsnorm(x, w, eps):
    xf = x.float()
    var = xf.pow(2).mean(-1, keepdim=True)
    xf = xf * torch.rsqrt(var + eps)
    return w * xf.to(x.dtype)


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def make_handroll_loop(W: DepthWeights, num_codebooks):
    """Return depth_loop(backbone_hidden [B,2048], first_cb [B], K:int) -> codes [B,K].
    Single-token decode per position; KV cache = python list per layer (unrolled)."""
    is_q8 = W.quant == "int8"

    def mm(x, w):
        """x @ w.T where w is either a plain [out,in] tensor or an int8 (q,scale) pair.
        int8 path: dequant weight in-graph (q.to(dtype)*scale) then matmul — halves the
        HBM bytes read for the weight (the bandwidth-dominant term)."""
        if is_q8:
            q, scale = w                       # q:[out,in] int8, scale:[out,1]
            wd = q.to(x.dtype) * scale         # [out,in] dequantized in compute dtype
            return x @ wd.T
        return x @ w.T

    @torch.no_grad()
    def step(h_embed, kv, cos_row, sin_row):
        # h_embed: [B, 1, 1024]  (already projected into hidden dim)
        B = h_embed.shape[0]
        cos = cos_row.view(1, 1, 1, -1)
        sin = sin_row.view(1, 1, 1, -1)
        x = h_embed
        for i in range(W.nL):
            L = W.L[i]
            residual = x
            hn = rmsnorm(x, L["ln1"], W.eps)                       # [B,1,1024]
            q = mm(hn, L["q"]).view(B, 1, W.nh, W.hd).transpose(1, 2)   # [B,nh,1,hd]
            k = mm(hn, L["k"]).view(B, 1, W.nkv, W.hd).transpose(1, 2)  # [B,nkv,1,hd]
            v = mm(hn, L["v"]).view(B, 1, W.nkv, W.hd).transpose(1, 2)
            # rope on q,k (single position)
            q = (q * cos) + (rotate_half(q) * sin)
            k = (k * cos) + (rotate_half(k) * sin)
            # append to cache
            kc = torch.cat([kv[i][0], k], dim=2) if kv[i][0] is not None else k
            vc = torch.cat([kv[i][1], v], dim=2) if kv[i][1] is not None else v
            kv[i] = (kc, vc)
            # GQA repeat kv heads
            kr = kc[:, :, None, :, :].expand(B, W.nkv, W.groups, kc.shape[2], W.hd)\
                   .reshape(B, W.nh, kc.shape[2], W.hd)
            vr = vc[:, :, None, :, :].expand(B, W.nkv, W.groups, vc.shape[2], W.hd)\
                   .reshape(B, W.nh, vc.shape[2], W.hd)
            attn = (q.float() @ kr.float().transpose(2, 3)) * W.scaling  # fp32 scores (bf16-fix: head+qk)
            attn = torch.softmax(attn.float(), dim=-1).to(x.dtype)
            ao = (attn @ vr).transpose(1, 2).reshape(B, 1, W.nh * W.hd)  # [B,1,1024]
            ao = mm(ao, L["o"])
            x = residual + ao
            residual = x
            hn = rmsnorm(x, L["ln2"], W.eps)
            mlp = mm(torch.nn.functional.silu(mm(hn, L["gate"])) * mm(hn, L["up"]), L["down"])
            x = residual + mlp
        x = rmsnorm(x, W.norm_w, W.eps)
        return x  # [B,1,1024]

    @torch.no_grad()
    def depth_loop(backbone_hidden, first_cb, K):
        dev = backbone_hidden.device
        B = backbone_hidden.shape[0]
        kv = [(None, None) for _ in range(W.nL)]
        # position 0: backbone hidden -> projected, warm the cache (no head)
        h0 = (backbone_hidden.view(B, 1, -1) @ W.proj.T)            # [B,1,1024]
        step(h0, kv, W.cos[0], W.sin[0])
        # position 1: first codebook token (offset 0)
        e1 = torch.nn.functional.embedding(first_cb.view(B, 1), W.embed)  # [B,1,2048]
        h1 = e1 @ W.proj.T
        out = step(h1, kv, W.cos[1], W.sin[1])
        cb = torch.argmax(out[:, -1, :].float() @ W.head[0].float(), dim=-1)  # fp32 head
        outs = [first_cb.view(B), cb]
        for k in range(2, K):
            idx = cb.view(B, 1) + (k - 1) * W.vocab
            e = torch.nn.functional.embedding(idx, W.embed)         # [B,1,2048]
            hk = e @ W.proj.T
            out = step(hk, kv, W.cos[k], W.sin[k])
            cb = torch.argmax(out[:, -1, :].float() @ W.head[k - 1].float(), dim=-1)  # fp32 head
            outs.append(cb)
        return torch.stack(outs, dim=1)  # [B,K]

    return depth_loop


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    ap.add_argument("--mode", choices=["eager", "compile"], default="compile")
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--quant", choices=["none", "int8"], default="none",
                    help="int8 = weight-only int8 on q/k/v/o/gate/up/down (halves 230->115MB)")
    ap.add_argument("--model", default=MODEL)
    args = ap.parse_args()
    quant = None if args.quant == "none" else args.quant
    dt = torch.bfloat16 if args.dtype == "bf16" else torch.float32

    print(f"[fair2] loading {args.model} (dtype={args.dtype}, mode={args.mode})...", flush=True)
    proc = AutoProcessor.from_pretrained(args.model)
    model = CsmForConditionalGeneration.from_pretrained(args.model, dtype=dt).eval()
    model.codec_model = model.codec_model.float()
    num_cb = model.config.num_codebooks
    K = min(args.k, num_cb)
    gc = model.depth_decoder.generation_config
    gc.do_sample = False; gc.temperature = None; gc.top_k = None; gc.top_p = None

    print("[fair2] capturing depth inputs + stock reference...", flush=True)
    cap = capture_depth_inputs(model, proc, "[0]Hello from Trainium, latency test.")
    first_cb_cpu = cap["input_ids"][:, 1].clone()
    bb_hidden_cpu = cap["backbone_last_hidden_state"].clone()
    stock_ref = cap["ref_codebooks"].cpu()  # stock HF depth decode (the shipped serial path)

    # CPU fp32 oracle via the SAME hand-rolled math (proves the kernel math is correct)
    Wf = DepthWeights(model.depth_decoder, torch.device("cpu"), torch.float32)
    loop_cpu = make_handroll_loop(Wf, num_cb)
    ref = loop_cpu(bb_hidden_cpu.float(), first_cb_cpu, num_cb).cpu()
    # cross-check hand-rolled fp32 vs stock HF. Stock sequences carry a leading position-0
    # token, so stock[:, 1:] aligns with our [first_cb, cb1, ...] convention.
    sref = stock_ref[:, 1:]
    n0 = min(ref.shape[1], sref.shape[1])
    hr_vs_stock = int((ref[:, :n0] == sref[:, :n0]).all())
    print(f"[fair2] hand-rolled fp32 vs stock HF (aligned): "
          f"{'EXACT' if hr_vs_stock else 'DIFF'} "
          f"({int((ref[:, :n0]==sref[:, :n0]).sum())}/{ref[:, :n0].numel()})", flush=True)

    # device weights + loop (optionally int8 weight-only)
    W = DepthWeights(model.depth_decoder, DEV, dt, quant=quant)
    depth_loop = make_handroll_loop(W, num_cb)
    bb_d = bb_hidden_cpu.to(dt).to(DEV)
    fcb_d = first_cb_cpu.to(DEV)

    import torch._dynamo as dynamo
    if args.mode == "compile":
        torch_neuronx.reset_dynamo_metrics()
        dynamo.reset()
        try:
            expl = dynamo.explain(lambda: depth_loop(bb_d, fcb_d, K))()
            print(f"[fair2] dynamo.explain: graph_count={expl.graph_count} "
                  f"graph_break_count={expl.graph_break_count} op_count={expl.op_count}",
                  flush=True)
            for i, brk in enumerate(expl.break_reasons):
                print(f"[fair2]   break[{i}]: {getattr(brk, 'reason', brk)}", flush=True)
        except Exception as e:
            print(f"[fair2] dynamo.explain failed: {type(e).__name__}: {e}", flush=True)
        run = torch.compile(depth_loop, backend="neuron", dynamic=False)
    else:
        run = depth_loop

    print(f"[fair2] warm/compile pass (K={K})...", flush=True)
    t = time.perf_counter()
    part = run(bb_d, fcb_d, K).cpu()
    warm_ms = (time.perf_counter() - t) * 1000
    print(f"[fair2] first (compile+run) pass: {warm_ms:.1f} ms", flush=True)

    if args.mode == "compile":
        try:
            m = torch_neuronx.get_dynamo_metrics()
            print(f"[fair2] neuron dynamo graphs compiled: {len(m)}", flush=True)
            for g in m:
                print(f"[fair2]   graph {g.graph_name}: nodes={g.graph_node_count} "
                      f"compile_us={g.torch_neuronx_compile_us}", flush=True)
        except Exception as e:
            print(f"[fair2] get_dynamo_metrics: {e}", flush=True)

    best = 1e9; times = []
    for _ in range(args.iters):
        t = time.perf_counter()
        run(bb_d, fcb_d, K).cpu()
        ms = (time.perf_counter() - t) * 1000
        times.append(ms); best = min(best, ms)
    times.sort(); med = times[len(times) // 2]
    print(f"[fair2] warm depth ms/frame (K={K}): best={best:.2f} median={med:.2f} "
          f"max={times[-1]:.2f}", flush=True)

    n = min(K, ref.shape[1])
    match = int((part[:, :n] == ref[:, :n]).all())
    n_agree = int((part[:, :n] == ref[:, :n]).sum())
    print(f"[fair2] argmax match vs CPU fp32 oracle (first {n}): "
          f"{'EXACT' if match else 'DIFF'} ({n_agree}/{part[:, :n].numel()})", flush=True)
    print(f"[fair2] device codes = {part[:, :n].tolist()}", flush=True)
    print(f"[fair2] oracle codes = {ref[:, :n].tolist()}", flush=True)

    CPU_MS = 137.0; FLOOR_MS = 14.0
    print(f"\n[fair2] === RESULT ({args.mode}, {args.dtype}, quant={args.quant}, K={K}) ===",
          flush=True)
    print(f"[fair2] depth {best:.2f} ms/frame | vs 137ms CPU: {CPU_MS/best:.2f}x | "
          f"vs ~14ms floor: {best/FLOOR_MS:.2f}x-above | "
          f"argmax vs fp32 {'EXACT' if match else 'DIFF'} ({n_agree}/{part[:, :n].numel()})",
          flush=True)


if __name__ == "__main__":
    sys.exit(main())

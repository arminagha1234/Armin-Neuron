"""fair_depth_exact.py — selective-fp32 variant of the hand-rolled depth loop.

Goal: find the MINIMAL set of ops that must run in fp32 on-device to recover the
codebook-argmax match vs the fp32 oracle (device bf16 = 7/32, CPU bf16 = 31/32 — so the
device adds precision loss beyond plain bf16 rounding). Each precision knob is an
independent flag so we can flip one at a time and measure match% + ms.

Knobs (env EXACT_KNOBS = comma list, e.g. "head,rope"):
  head : head/argmax matmul (out @ head[k]) in fp32   (2051-way argmax, tiny cost)
  rope : cos/sin table + rope apply in fp32
  qk   : Q@K^T scores accumulated in fp32 (softmax already .float())
  av   : attn@V in fp32
  lin  : all attn/mlp linears (q/k/v/o/gate/up/down + proj) in fp32
  (empty = pure bf16 baseline; "head,rope,qk,av,lin" ~= full fp32)

Reuses DepthWeights / rmsnorm / rotate_half / capture_depth_inputs from fair_depth_handroll.
Does NOT modify fair_depth_handroll.py.
"""
import os
import sys
import time
import torch
import torch_neuronx  # noqa: F401
from transformers import AutoProcessor, CsmForConditionalGeneration

sys.path.insert(0, "/host")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fair_depth_handroll import DepthWeights, rotate_half, capture_depth_inputs  # noqa: E402

MODEL = os.environ.get("CSM_MODEL", "/host/csm_1b")
DEV = torch.device("neuron")


def rmsnorm(x, w, eps):
    xf = x.float()
    var = xf.pow(2).mean(-1, keepdim=True)
    xf = xf * torch.rsqrt(var + eps)
    return w * xf.to(x.dtype)


def make_exact_loop(W, num_cb, knobs, head_f32, cos_f32, sin_f32):
    """head_f32:[31,1024,2051], cos_f32/sin_f32:[max_len,hd] are fp32 device tensors
    (true-fp32 weights, not bf16-upcast) used when the matching knob is on."""
    K_head = "head" in knobs
    K_rope = "rope" in knobs
    K_qk = "qk" in knobs
    K_av = "av" in knobs
    K_lin = "lin" in knobs
    cdt = W.cos.dtype  # compute dtype (bf16 on device)

    def mm(x, w):
        if K_lin:
            return (x.float() @ w.float().T).to(cdt)
        return x @ w.T

    @torch.no_grad()
    def step(h_embed, kv, kpos):
        B = h_embed.shape[0]
        if K_rope:
            cos = cos_f32[kpos].view(1, 1, 1, -1)
            sin = sin_f32[kpos].view(1, 1, 1, -1)
        else:
            cos = W.cos[kpos].view(1, 1, 1, -1)
            sin = W.sin[kpos].view(1, 1, 1, -1)
        x = h_embed
        for i in range(W.nL):
            L = W.L[i]
            residual = x
            hn = rmsnorm(x, L["ln1"], W.eps)
            q = mm(hn, L["q"]).view(B, 1, W.nh, W.hd).transpose(1, 2)
            k = mm(hn, L["k"]).view(B, 1, W.nkv, W.hd).transpose(1, 2)
            v = mm(hn, L["v"]).view(B, 1, W.nkv, W.hd).transpose(1, 2)
            if K_rope:
                qf, kf = q.float(), k.float()
                q = ((qf * cos) + (rotate_half(qf) * sin)).to(cdt)
                k = ((kf * cos) + (rotate_half(kf) * sin)).to(cdt)
            else:
                q = (q * cos) + (rotate_half(q) * sin)
                k = (k * cos) + (rotate_half(k) * sin)
            kc = torch.cat([kv[i][0], k], dim=2) if kv[i][0] is not None else k
            vc = torch.cat([kv[i][1], v], dim=2) if kv[i][1] is not None else v
            kv[i] = (kc, vc)
            kr = kc[:, :, None, :, :].expand(B, W.nkv, W.groups, kc.shape[2], W.hd)\
                   .reshape(B, W.nh, kc.shape[2], W.hd)
            vr = vc[:, :, None, :, :].expand(B, W.nkv, W.groups, vc.shape[2], W.hd)\
                   .reshape(B, W.nh, vc.shape[2], W.hd)
            if K_qk:
                attn = (q.float() @ kr.float().transpose(2, 3)) * W.scaling  # fp32 scores
            else:
                attn = (q @ kr.transpose(2, 3)) * W.scaling
            attn = torch.softmax(attn.float(), dim=-1)  # softmax always fp32
            if K_av:
                ao = (attn @ vr.float()).to(cdt).transpose(1, 2).reshape(B, 1, W.nh * W.hd)
            else:
                ao = (attn.to(cdt) @ vr).transpose(1, 2).reshape(B, 1, W.nh * W.hd)
            ao = mm(ao, L["o"])
            x = residual + ao
            residual = x
            hn = rmsnorm(x, L["ln2"], W.eps)
            mlp = mm(torch.nn.functional.silu(mm(hn, L["gate"])) * mm(hn, L["up"]), L["down"])
            x = residual + mlp
        x = rmsnorm(x, W.norm_w, W.eps)
        return x

    def head_mm(vec, k_idx):
        """vec:[B,1024] -> logits over 2051. fp32 head if knob on."""
        if K_head:
            return vec.float() @ head_f32[k_idx]
        return vec @ W.head[k_idx]

    @torch.no_grad()
    def depth_loop(backbone_hidden, first_cb, K):
        B = backbone_hidden.shape[0]
        kv = [(None, None) for _ in range(W.nL)]
        h0 = (backbone_hidden.view(B, 1, -1) @ W.proj.T)
        step(h0, kv, 0)
        e1 = torch.nn.functional.embedding(first_cb.view(B, 1), W.embed)
        h1 = e1 @ W.proj.T
        out = step(h1, kv, 1)
        cb = torch.argmax(head_mm(out[:, -1, :], 0), dim=-1)
        outs = [first_cb.view(B), cb]
        for k in range(2, K):
            idx = cb.view(B, 1) + (k - 1) * W.vocab
            e = torch.nn.functional.embedding(idx, W.embed)
            hk = e @ W.proj.T
            out = step(hk, kv, k)
            cb = torch.argmax(head_mm(out[:, -1, :], k - 1), dim=-1)
            outs.append(cb)
        return torch.stack(outs, dim=1)

    return depth_loop


def main():
    K = int(os.environ.get("EXACT_K", "32"))
    iters = int(os.environ.get("EXACT_ITERS", "10"))
    # configs to sweep in ONE device process (each compiles its own graph):
    cfg_env = os.environ.get("EXACT_CONFIGS", "|head")  # '|' separated; '' = pure bf16
    configs = cfg_env.split("|")
    torch.set_num_threads(24)

    print(f"[exact] loading {MODEL} bf16 ...", flush=True)
    proc = AutoProcessor.from_pretrained(MODEL)
    model = CsmForConditionalGeneration.from_pretrained(MODEL, dtype=torch.bfloat16).eval()
    model.codec_model = model.codec_model.float()
    num_cb = model.config.num_codebooks
    K = min(K, num_cb)
    gc = model.depth_decoder.generation_config
    gc.do_sample = False; gc.temperature = None; gc.top_k = None; gc.top_p = None

    cap = capture_depth_inputs(model, proc, "[0]Comprehensive profile test on Trainium.")
    first_cb = cap["input_ids"][:, 1].clone()
    bb = cap["backbone_last_hidden_state"].clone()

    # fp32 oracle (CPU) via hand-rolled fp32 (all knobs on == full fp32 math)
    from fair_depth_handroll import make_handroll_loop
    Wf = DepthWeights(model.depth_decoder, torch.device("cpu"), torch.float32)
    ref = make_handroll_loop(Wf, num_cb)(bb.float(), first_cb, num_cb).cpu()
    n = min(K, ref.shape[1])

    # device bf16 weights (shared across configs)
    Wd = DepthWeights(model.depth_decoder, DEV, torch.bfloat16)
    # true-fp32 head + cos/sin on device (for the head/rope knobs)
    head_f32 = model.depth_decoder.codebooks_head.weight.detach().float().to(DEV).contiguous()
    m = model.depth_decoder.model
    max_len = num_cb + 1
    with torch.no_grad():
        pos = torch.arange(max_len)[None].float()
        dummy = torch.zeros(1, 1, Wd.hd)
        cosf, sinf = m.rotary_emb.to("cpu")(dummy, position_ids=pos)
    cos_f32 = cosf[0].to(DEV, torch.float32)
    sin_f32 = sinf[0].to(DEV, torch.float32)
    bb_d = bb.to(torch.bfloat16).to(DEV)
    fcb_d = first_cb.to(DEV)

    results = []
    for cfg in configs:
        knobs = set(s for s in cfg.split(",") if s)
        label = "bf16-baseline" if not knobs else "+".join(sorted(knobs))
        print(f"\n[exact] === config: {label} (knobs={sorted(knobs)}) ===", flush=True)
        loop = make_exact_loop(Wd, num_cb, knobs, head_f32, cos_f32, sin_f32)
        run = torch.compile(loop, backend="neuron", dynamic=False)
        t = time.perf_counter()
        codes = run(bb_d, fcb_d, K).cpu()
        compile_ms = (time.perf_counter() - t) * 1000
        best = 1e9
        for _ in range(iters):
            t = time.perf_counter()
            run(bb_d, fcb_d, K).cpu()
            best = min(best, (time.perf_counter() - t) * 1000)
        n_agree = int((codes[:, :n] == ref[:, :n]).sum())
        total = codes[:, :n].numel()
        print(f"[exact] {label}: match {n_agree}/{total} | depth {best:.2f} ms "
              f"| compile {compile_ms/1000:.0f}s", flush=True)
        print(f"[exact] dev codes = {codes[0, :n].tolist()}", flush=True)
        results.append((label, n_agree, total, best))

    print("\n[exact] ====== SUMMARY ======", flush=True)
    print(f"[exact] oracle fp32 codes = {ref[0, :n].tolist()}", flush=True)
    for label, a, tot, ms in results:
        print(f"[exact]   {label:28s} {a:3d}/{tot}  {ms:6.2f} ms", flush=True)


if __name__ == "__main__":
    sys.exit(main())

"""CSM-1B HAND-ROLLED per-frame decode loop — compiled backbone + compiled depth + CPU codec.

Replaces `model.generate` with a manual loop that wires in BOTH proven on-device wins:
  (1) the compiled device DEPTH loop (fair_depth_handroll.py — 8-18 ms, was 153 ms CPU), and
  (2) a COMPILED BACKBONE decode step (the profiled hidden 128 ms eager cost).

Per CSM's own _sample structure (generation_csm.py):
  frame i>=1: embed frame(i-1)'s 32 codes -> backbone decode step (KV cache) -> hidden
              -> codebook0 = argmax(lm_head(hidden)) -> depth decoder -> codebooks 1..31
              -> 32 codes -> codec.  Those 32 codes are the backbone input for frame i+1.

WHY the backbone step compiles as ONE fixed-shape resident-weight graph (the compile_backbone.py
fix): the stock HF backbone.forward + StaticCache does `position_ids = arange(cache.get_seq_length())`
and `index_copy_` into the cache — a data-dependent .item() (graph break) + a scatter that dynamo
replays as a stale artifact.  Here we hand-roll the step with:
  * python-int / runtime-tensor positions (NO .item(), NO arange(get_seq_length()))
  * a FIXED-MAXLEN KV buffer written FUNCTIONALLY via a one-hot mask (NO index_copy_) so the
    shape is identical every frame -> compiles ONCE, weights resident, reused across all frames
  * a runtime additive causal mask (0 for valid positions, -inf beyond `cur`)
  * argmax on-device; exactly one host boundary per frame (the 32 codes -> CPU for the codec)

The prompt prefill (text -> KV) is done ONCE by the real backbone (its keys are post-rope, which is
exactly what our hand-rolled attention expects); we extract the prompt K/V into the fixed buffer and
hand-roll only the autoregressive decode steps — which is where the 128 ms/frame lives.

Backbone config (dumped from /host/csm_1b):
  layers=16 hidden=2048 nh=32 nkv=8 (GQA groups=4) hd=64 mlp=8192 rope_theta=5e5 rms_eps=1e-5
  act=silu, no biases, vocab=2051, num_codebooks=32, lm_head separate (not tied).
"""
import os
import sys
import time
import argparse
import torch
import torch_neuronx  # noqa: F401
from transformers import AutoProcessor, CsmForConditionalGeneration

sys.path.insert(0, os.path.dirname(__file__))  # bundle: fair_depth_handroll.py is a sibling in src/
sys.path.insert(0, "/host")  # box layout: primitives live flat in /host
from fair_depth_handroll import DepthWeights, make_handroll_loop  # noqa: E402

MODEL = os.environ.get("CSM_MODEL", "/host/csm_1b")
DEV = torch.device("neuron")
NEG = -1e9  # additive-mask "-inf" that stays finite in bf16


def rmsnorm(x, w, eps):
    xf = x.float()
    var = xf.pow(2).mean(-1, keepdim=True)
    xf = xf * torch.rsqrt(var + eps)
    return w * xf.to(x.dtype)


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


class BackboneWeights:
    """Flat handle on the backbone's real weights (all on one device/dtype), analogous to
    DepthWeights.  Includes the audio-frame embedding table + offsets and the separate lm_head."""

    def __init__(self, model, dev, dtype):
        bb = model.backbone_model
        cfg = model.config
        self.nL = cfg.num_hidden_layers
        self.nh = cfg.num_attention_heads
        self.nkv = cfg.num_key_value_heads
        self.hd = cfg.head_dim
        self.groups = self.nh // self.nkv
        self.scaling = self.hd ** -0.5
        self.eps = cfg.rms_norm_eps
        self.vocab = cfg.vocab_size
        self.hidden = cfg.hidden_size

        def g(t):
            return t.detach().to(dev, dtype).contiguous()

        emb = bb.embed_tokens
        self.embed_audio = g(emb.embed_audio_tokens.weight)          # [num_cb*codebook_size, 2048]
        self.audio_offsets = emb.audio_tokens_offsets.detach().to(dev).long().contiguous()  # [num_cb]
        self.norm_w = g(bb.norm.weight)
        self.lm_head = g(model.lm_head.weight)                       # [vocab, 2048]
        self.L = []
        for i in range(self.nL):
            lyr = bb.layers[i]
            self.L.append(dict(
                ln1=g(lyr.input_layernorm.weight),
                ln2=g(lyr.post_attention_layernorm.weight),
                q=g(lyr.self_attn.q_proj.weight),
                k=g(lyr.self_attn.k_proj.weight),
                v=g(lyr.self_attn.v_proj.weight),
                o=g(lyr.self_attn.o_proj.weight),
                gate=g(lyr.mlp.gate_proj.weight),
                up=g(lyr.mlp.up_proj.weight),
                down=g(lyr.mlp.down_proj.weight),
            ))
        # rope cos/sin table [max_pos, hd] on device (same rotary_emb the real backbone uses)
        max_pos = cfg.max_position_embeddings
        with torch.no_grad():
            pos = torch.arange(max_pos, device="cpu")[None].float()
            dummy = torch.zeros(1, 1, self.hd)
            cos, sin = bb.rotary_emb.to("cpu")(dummy, position_ids=pos)   # [1, max_pos, hd]
        self.cos = cos[0].to(dev, dtype)
        self.sin = sin[0].to(dev, dtype)
        self.dtype = dtype
        self.dev = dev


def frame_embed(W: BackboneWeights, codes):
    """Embed a [B, num_cb] audio frame into [B, 1, 2048] like CsmBackboneModelEmbeddings:
    embed_audio_tokens(codes + audio_tokens_offsets).sum(codebook)."""
    idx = codes + W.audio_offsets            # [B, num_cb]
    e = torch.nn.functional.embedding(idx, W.embed_audio)   # [B, num_cb, 2048]
    return e.sum(dim=1, keepdim=True)        # [B, 1, 2048]


def make_backbone_step(W: BackboneWeights):
    """Return step(codes[B,num_cb], kc[list], vc[list], onehot[1,1,MAX,1], addmask[1,1,1,MAX],
                   cos_cur[hd], sin_cur[hd]) -> (cb0[B], hidden[B,2048], kc', vc').

    ONE fixed-shape graph.  KV buffers are [B, nkv, MAX, hd] (fixed MAX), written functionally
    at the current position via the one-hot mask (no index_copy_, no dynamic shape).  Attention
    runs over the full MAX buffer with a runtime additive causal mask, so the *same* graph serves
    every frame -> compiles once, weights resident."""

    @torch.no_grad()
    def step(codes, kc, vc, onehot, addmask, cos_cur, sin_cur):
        B = codes.shape[0]
        cos = cos_cur.view(1, 1, 1, -1)
        sin = sin_cur.view(1, 1, 1, -1)
        x = frame_embed(W, codes)                              # [B,1,2048]
        new_kc, new_vc = [], []
        for i in range(W.nL):
            L = W.L[i]
            residual = x
            hn = rmsnorm(x, L["ln1"], W.eps)
            q = (hn @ L["q"].T).view(B, 1, W.nh, W.hd).transpose(1, 2)    # [B,nh,1,hd]
            k = (hn @ L["k"].T).view(B, 1, W.nkv, W.hd).transpose(1, 2)   # [B,nkv,1,hd]
            v = (hn @ L["v"].T).view(B, 1, W.nkv, W.hd).transpose(1, 2)
            q = (q * cos) + (rotate_half(q) * sin)
            k = (k * cos) + (rotate_half(k) * sin)
            # functional write into the fixed KV buffer at the current position (one-hot):
            #   kc_new = kc*(1-onehot) + k*onehot   (k broadcasts across the MAX axis)
            kcw = kc[i] * (1.0 - onehot) + k * onehot          # [B,nkv,MAX,hd]
            vcw = vc[i] * (1.0 - onehot) + v * onehot
            new_kc.append(kcw)
            new_vc.append(vcw)
            # GQA repeat kv heads to nh
            kr = kcw[:, :, None, :, :].expand(B, W.nkv, W.groups, kcw.shape[2], W.hd)\
                    .reshape(B, W.nh, kcw.shape[2], W.hd)
            vr = vcw[:, :, None, :, :].expand(B, W.nkv, W.groups, vcw.shape[2], W.hd)\
                    .reshape(B, W.nh, vcw.shape[2], W.hd)
            attn = (q.float() @ kr.float().transpose(2, 3)) * W.scaling  # fp32 QK (match depth fix)
            attn = attn + addmask                               # mask future positions
            attn = torch.softmax(attn.float(), dim=-1).to(x.dtype)
            ao = (attn @ vr).transpose(1, 2).reshape(B, 1, W.nh * W.hd)
            ao = ao @ L["o"].T
            x = residual + ao
            residual = x
            hn = rmsnorm(x, L["ln2"], W.eps)
            mlp = (torch.nn.functional.silu(hn @ L["gate"].T) * (hn @ L["up"].T)) @ L["down"].T
            x = residual + mlp
        x = rmsnorm(x, W.norm_w, W.eps)                         # [B,1,2048]
        h = x[:, 0, :]                                          # [B,2048]
        cb0 = torch.argmax(h.float() @ W.lm_head.float().T, dim=-1)  # fp32 head (match depth fix)
        return cb0, h, new_kc, new_vc

    return step


# ----------------------------------------------------------------------------------------------
# Prefill: run the REAL backbone once on the prompt to (a) get frame-0 hidden + cb0 and (b) extract
# the prompt K/V (post-rope) into our fixed buffer.  This is a one-time cost off the per-frame path.
# ----------------------------------------------------------------------------------------------
def prefill_capture(model, proc, text, W, MAX):
    """Returns (h0[B,2048], cb0[B], kc0 list, vc0 list, P int) — P = prompt length."""
    cap = {}
    bb = model.backbone_model
    _rf = bb.forward

    def spy(*a, **k):
        out = _rf(*a, **k)
        # prefill call has seq_len > 1; record its cache + hidden
        pkv = k.get("past_key_values", None)
        ie = k.get("inputs_embeds", None)
        if ie is not None and ie.shape[1] > 1:
            cap["P"] = ie.shape[1]
            cap["hidden"] = out.last_hidden_state.detach().clone()
            cap["pkv"] = pkv
        return out

    bb.forward = spy
    inputs = proc(text, add_special_tokens=True, return_tensors="pt")
    with torch.no_grad():
        model.generate(**inputs, output_audio=False, do_sample=False, max_new_tokens=1,
                       cache_implementation="static")
    bb.forward = _rf

    P = cap["P"]
    hidden = cap["hidden"]                        # [B, P, 2048] (cpu/model dtype)
    h0 = hidden[:, -1, :].to(W.dtype).to(W.dev)   # frame-0 backbone_last_hidden_state
    cb0 = torch.argmax(hidden[:, -1, :].float() @ model.lm_head.weight.detach().float().T, dim=-1)
    cb0 = cb0.to(W.dev)
    # extract prompt K/V (post-rope) from the StaticCache -> fixed buffer [B,nkv,MAX,hd]
    pkv = cap["pkv"]
    B = hidden.shape[0]
    kc0, vc0 = [], []
    for i in range(W.nL):
        kk = pkv.layers[i].keys if hasattr(pkv, "layers") else pkv.key_cache[i]
        vv = pkv.layers[i].values if hasattr(pkv, "layers") else pkv.value_cache[i]
        kbuf = torch.zeros(B, W.nkv, MAX, W.hd, dtype=W.dtype, device=W.dev)
        vbuf = torch.zeros(B, W.nkv, MAX, W.hd, dtype=W.dtype, device=W.dev)
        kbuf[:, :, :P, :] = kk[:, :, :P, :].to(W.dtype).to(W.dev)
        vbuf[:, :, :P, :] = vv[:, :, :P, :].to(W.dtype).to(W.dev)
        kc0.append(kbuf)
        vc0.append(vbuf)
    return h0, cb0, kc0, vc0, P


def codec_decode(model, frame_codes, num_cb):
    """CPU Mimi decode of [B,num_cb] codes -> waveform (flattened)."""
    B, K = frame_codes.shape
    if K < num_cb:
        frame_codes = torch.cat(
            [frame_codes, torch.zeros((B, num_cb - K), dtype=frame_codes.dtype)], dim=1)
    x = frame_codes.transpose(0, 1).unsqueeze(0).contiguous().clamp_(0, 2047)
    with torch.no_grad():
        a = model.codec_model.decode(x)
    a = a[0] if isinstance(a, (list, tuple)) else getattr(a, "audio_values", a)
    return a.detach().float().cpu().flatten()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--depth-k", type=int, default=32, help="codebooks per frame (32=full)")
    ap.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    ap.add_argument("--mode", choices=["compile", "eager"], default="compile")
    ap.add_argument("--out", default="/host/manual_loop.wav")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--text", default="[0]Hello from Trainium, hand rolled decode loop.")
    args = ap.parse_args()
    dt = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    torch.set_num_threads(24)

    print(f"[manual] loading {MODEL} (dtype={args.dtype}, mode={args.mode})...", flush=True)
    proc = AutoProcessor.from_pretrained(MODEL)
    model = CsmForConditionalGeneration.from_pretrained(MODEL, dtype=dt).eval()
    model.codec_model = model.codec_model.float()
    num_cb = model.config.num_codebooks
    K = min(args.depth_k, num_cb)
    gc = model.depth_decoder.generation_config
    gc.do_sample = False; gc.temperature = None; gc.top_k = None; gc.top_p = None

    # ---- stock reference (for code validation) ----
    print("[manual] running stock model.generate reference...", flush=True)
    inputs = proc(args.text, add_special_tokens=True, return_tensors="pt")
    with torch.no_grad():
        ref_seq = model.generate(**inputs, output_audio=False, do_sample=False,
                                 max_new_tokens=args.frames, cache_implementation="static")
    ref_frames = ref_seq.detach().cpu()  # [B, frames, num_cb]
    print(f"[manual] stock ref frames shape: {tuple(ref_frames.shape)}", flush=True)

    # ---- weights on device ----
    Wbb = BackboneWeights(model, DEV, dt)
    Wd = DepthWeights(model.depth_decoder, DEV, dt, quant=None)
    depth_loop = make_handroll_loop(Wd, num_cb)
    bb_step = make_backbone_step(Wbb)

    MAX = 0  # set after prefill

    # ---- prefill (one-time) ----
    print("[manual] prefill capture (real backbone, one-time)...", flush=True)
    # MAX must cover prompt + frames + slack; discover P via a throwaway prefill first
    h0, cb0, kc0, vc0, P = prefill_capture(model, proc, args.text, Wbb, MAX=2048)
    MAX = P + args.frames + 2
    # rebuild buffers trimmed to MAX (compile on a tight fixed shape)
    kc0 = [b[:, :, :MAX, :].contiguous() for b in kc0]
    vc0 = [b[:, :, :MAX, :].contiguous() for b in vc0]
    B = cb0.shape[0]
    print(f"[manual] prompt len P={P}, MAX={MAX}, frames={args.frames}", flush=True)

    def make_masks(cur):
        onehot = torch.zeros(1, 1, MAX, 1, dtype=dt, device=DEV)
        onehot[0, 0, cur, 0] = 1.0
        add = torch.full((1, 1, 1, MAX), NEG, dtype=dt, device=DEV)
        add[0, 0, 0, :cur + 1] = 0.0
        return onehot, add

    # ---- dynamo.explain on the backbone step (task: report graph breaks honestly) ----
    import torch._dynamo as dynamo
    if args.mode == "compile":
        _cur = P
        _oh, _am = make_masks(_cur)
        _prev = frames0_seed = cb0.new_zeros(B, num_cb).long()
        try:
            dynamo.reset()
            expl = dynamo.explain(
                lambda: bb_step(_prev, kc0, vc0, _oh, _am, Wbb.cos[_cur], Wbb.sin[_cur]))()
            print(f"[manual] BACKBONE dynamo.explain: graph_count={expl.graph_count} "
                  f"graph_break_count={expl.graph_break_count} op_count={expl.op_count}", flush=True)
            for i, brk in enumerate(expl.break_reasons):
                print(f"[manual]   bb break[{i}]: {getattr(brk, 'reason', brk)}", flush=True)
        except Exception as e:
            print(f"[manual] BACKBONE dynamo.explain failed: {type(e).__name__}: {e}", flush=True)

    # ---- compile depth + backbone step ----
    if args.mode == "compile":
        torch_neuronx.reset_dynamo_metrics()
        dynamo.reset()
        depth_run = torch.compile(depth_loop, backend="neuron", dynamic=False)
        bb_run = torch.compile(bb_step, backend="neuron", dynamic=False)
    else:
        depth_run, bb_run = depth_loop, bb_step

    # =========================================================================================
    # HAND-ROLLED PER-FRAME DECODE LOOP
    #   frame 0: from prefill (h0, cb0) -> depth -> 32 codes
    #   frame i: backbone step(prev codes) -> (cb0, hidden) -> depth -> 32 codes
    # =========================================================================================
    def run_loop(measure=False):
        timings = {"backbone": [], "depth": [], "codec": []}
        wavs = []
        frames_out = []
        # frame 0 (from prefill)
        t = time.perf_counter()
        codes0 = depth_run(h0, cb0, K).cpu()  # [B,K]
        d_ms = (time.perf_counter() - t) * 1000
        if measure:
            timings["depth"].append(d_ms)
        full0 = codes0[:, :num_cb] if codes0.shape[1] >= num_cb else codes0
        frames_out.append(full0[:, :num_cb] if full0.shape[1] == num_cb else full0)
        t = time.perf_counter()
        wavs.append(codec_decode(model, codes0, num_cb))
        if measure:
            timings["codec"].append((time.perf_counter() - t) * 1000)
        prev_codes = full0.to(DEV)  # [B,num_cb] on device for next backbone step
        # ensure prev_codes is full num_cb width
        if prev_codes.shape[1] < num_cb:
            pad = torch.zeros(B, num_cb - prev_codes.shape[1], dtype=prev_codes.dtype, device=DEV)
            prev_codes = torch.cat([prev_codes, pad], dim=1)
        kc, vc = kc0, vc0
        for fi in range(1, args.frames):
            cur = P + (fi - 1)  # absolute position of this frame's backbone input token
            onehot, addmask = make_masks(cur)
            cos_cur = Wbb.cos[cur]
            sin_cur = Wbb.sin[cur]
            t = time.perf_counter()
            cb0_f, hidden_f, kc, vc = bb_run(prev_codes.long(), kc, vc, onehot, addmask,
                                             cos_cur, sin_cur)
            # single host boundary for the codebook0 int (needed to seed depth); hidden stays device
            cb0_cpu = cb0_f.cpu()
            b_ms = (time.perf_counter() - t) * 1000
            if measure:
                timings["backbone"].append(b_ms)
            t = time.perf_counter()
            codes_f = depth_run(hidden_f, cb0_f, K).cpu()  # [B,K]
            d_ms = (time.perf_counter() - t) * 1000
            if measure:
                timings["depth"].append(d_ms)
            full_f = codes_f[:, :num_cb] if codes_f.shape[1] >= num_cb else codes_f
            frames_out.append(full_f)
            t = time.perf_counter()
            wavs.append(codec_decode(model, codes_f, num_cb))
            if measure:
                timings["codec"].append((time.perf_counter() - t) * 1000)
            prev_codes = full_f.to(DEV)
            if prev_codes.shape[1] < num_cb:
                pad = torch.zeros(B, num_cb - prev_codes.shape[1], dtype=prev_codes.dtype, device=DEV)
                prev_codes = torch.cat([prev_codes, pad], dim=1)
        return frames_out, wavs, timings

    # ---- warm / compile ----
    print("[manual] warm+compile pass (backbone + depth graphs)...", flush=True)
    t = time.perf_counter()
    frames_out, wavs, _ = run_loop(measure=False)
    print(f"[manual] first (compile+run {args.frames} frames): {(time.perf_counter()-t)*1000:.0f} ms",
          flush=True)

    # ---- warm timing (best-of per component, isolated) ----
    # Backbone step timing: re-run the compiled step in isolation (fixed inputs) best-of-iters.
    cur = P
    onehot, addmask = make_masks(cur)
    prev = frames_out[0].to(DEV).long()
    if prev.shape[1] < num_cb:
        prev = torch.cat([prev, torch.zeros(B, num_cb - prev.shape[1], dtype=prev.dtype, device=DEV)], dim=1)

    def one_bb():
        cb, h, _, _ = bb_run(prev, kc0, vc0, onehot, addmask, Wbb.cos[cur], Wbb.sin[cur])
        return cb.cpu()

    def one_depth():
        return depth_run(h0, cb0, K).cpu()

    one_bb(); one_depth()
    bb_ts, d_ts, c_ts = [], [], []
    for _ in range(args.iters):
        t = time.perf_counter(); one_bb(); bb_ts.append((time.perf_counter()-t)*1000)
        t = time.perf_counter(); one_depth(); d_ts.append((time.perf_counter()-t)*1000)
    # warm codec
    codec_decode(model, frames_out[0], num_cb)
    for _ in range(args.iters):
        t = time.perf_counter(); codec_decode(model, frames_out[0], num_cb); c_ts.append((time.perf_counter()-t)*1000)
    bb_ts.sort(); d_ts.sort(); c_ts.sort()
    bb_best, bb_med = bb_ts[0], bb_ts[len(bb_ts)//2]
    d_best, d_med = d_ts[0], d_ts[len(d_ts)//2]
    c_best, c_med = c_ts[0], c_ts[len(c_ts)//2]

    # ---- full-loop measured pass ----
    frames_out, wavs, timings = run_loop(measure=True)

    # ---- validate codes vs stock ----
    n_frames = min(len(frames_out), ref_frames.shape[1])
    manual = torch.stack([f[0] for f in frames_out[:n_frames]], dim=0)  # [n_frames, num_cb]
    refm = ref_frames[0, :n_frames, :]  # [n_frames, num_cb]
    kcmp = min(K, num_cb)
    agree = int((manual[:, :kcmp] == refm[:, :kcmp]).sum())
    total = manual[:, :kcmp].numel()
    exact = bool((manual[:, :kcmp] == refm[:, :kcmp]).all())
    # frame-0 codebook0 agreement (should always match — same prefill)
    f0cb0_match = bool(manual[0, 0] == refm[0, 0])

    # audio cosine (manual vs a stock-codec render of the ref frames)
    try:
        stock_wav = codec_decode(model, refm.to(torch.long), num_cb)
        manual_wav = torch.cat(wavs[:n_frames]) if wavs else torch.tensor([])
        L = min(stock_wav.numel(), manual_wav.numel())
        if L > 0:
            cos = torch.nn.functional.cosine_similarity(
                stock_wav[:L].unsqueeze(0), manual_wav[:L].unsqueeze(0)).item()
        else:
            cos = float("nan")
    except Exception as e:
        cos = float("nan")
        print(f"[manual] audio cosine skipped: {e}", flush=True)

    # ---- TTFT (frame 0 = depth+codec only; no backbone step for frame 0) ----
    #   frame 0 critical path = depth(K) + codec  (prefill hidden already available)
    #   steady per-frame = backbone + depth + codec
    ttft0 = d_best + c_best
    per_frame = bb_best + d_best + c_best

    print("\n[manual] ================= RESULT =================", flush=True)
    print(f"[manual] mode={args.mode} dtype={args.dtype} depth-K={K} frames={args.frames} P={P} MAX={MAX}",
          flush=True)
    print(f"[manual] backbone step : best={bb_best:.2f} median={bb_med:.2f} ms  "
          f"(loop mean={sum(timings['backbone'])/max(len(timings['backbone']),1):.2f})", flush=True)
    print(f"[manual] depth  (K={K}) : best={d_best:.2f} median={d_med:.2f} ms", flush=True)
    print(f"[manual] codec 1 frame : best={c_best:.2f} median={c_med:.2f} ms", flush=True)
    print(f"[manual] --- TTFT (frame0 = depth+codec)   = {ttft0:.1f} ms", flush=True)
    print(f"[manual] --- steady per-frame (bb+depth+codec) = {per_frame:.1f} ms", flush=True)
    print(f"[manual] codes vs stock: {'EXACT' if exact else 'DIFF'} ({agree}/{total}) "
          f"frame0.cb0 match={f0cb0_match}", flush=True)
    print(f"[manual] audio cosine (manual vs stock-codec): {cos:.4f}", flush=True)
    print(f"[manual] manual frame0 codes[:8] = {manual[0,:8].tolist()}", flush=True)
    print(f"[manual] stock  frame0 codes[:8] = {refm[0,:8].tolist()}", flush=True)
    if n_frames > 1:
        print(f"[manual] manual frame1 codes[:8] = {manual[1,:8].tolist()}", flush=True)
        print(f"[manual] stock  frame1 codes[:8] = {refm[1,:8].tolist()}", flush=True)

    try:
        import soundfile as sf
        wav = torch.cat(wavs)
        sf.write(args.out, wav.numpy(), 24000)
        print(f"[manual] wrote {args.out} ({wav.numel()} samples, {wav.numel()/24000:.2f}s)", flush=True)
    except Exception as e:
        print(f"[manual] (wav save skipped: {e})", flush=True)


if __name__ == "__main__":
    sys.exit(main())

"""B1' — fuse the CSM depth decoder's 31-step loop into ONE device graph.

B1 showed per-step offload is the wrong shape (31 host<->device round-trips/frame).
The fix: run the whole depth loop as a single device graph — one round-trip/frame.

Why HF's `depth_decoder.generate` can't be that single graph: its loop has host-side
control flow (stopping criteria, `unfinished_sequences`, sampling) that forces a sync
every step. So we hand-write the decode loop:
  - keep all state as device tensors (StaticCache on device),
  - argmax on-device, feed the next step, NO `.item()`/host sync mid-loop,
  - static python-int codebook index (kills the dynamic `self.weight[cache_position-1]`
    and `embed offset` gathers that broke shape-stability),
  - exactly ONE `mark_step` + one transfer at the end.

Validates the fused output against the model's own `depth_decoder.generate` (greedy)
for bit-exactness, and times fused-device vs the CPU baseline (~156ms/frame).

Usage:
    python b1p_fused_depth.py                 # correctness + timing
    python b1p_fused_depth.py --frames 20     # more timing samples
"""
import os, sys, argparse, time, torch
import torch_xla.core.xla_model as xm
from transformers import AutoProcessor, CsmForConditionalGeneration
from transformers.cache_utils import StaticCache

MODEL = os.environ.get("CSM_MODEL", os.path.expanduser("~/csm/csm_1b"))


def _to(obj, dev):
    if torch.is_tensor(obj):
        return obj.to(dev)
    if obj.__class__.__name__.endswith("Cache"):
        return obj
    if isinstance(obj, (list, tuple)):
        return type(obj)(_to(x, dev) for x in obj)
    if isinstance(obj, dict):
        return {k: _to(v, dev) for k, v in obj.items()}
    try:
        from transformers.utils import ModelOutput
        if isinstance(obj, ModelOutput):
            for k in list(obj.keys()):
                obj[k] = _to(obj[k], dev)
            return obj
    except Exception:
        pass
    return obj


def _offload_forward(module, dev):
    module.to(dev)
    for m in module.modules():
        for k, v in list(vars(m).items()):
            if torch.is_tensor(v) and v.device.type != "xla":
                setattr(m, k, v.to(dev))
    real = module.forward

    def wrapped(*a, **k):
        out = real(*_to(a, dev), **_to(k, dev)); xm.mark_step(); return _to(out, "cpu")
    module.forward = wrapped


def capture_depth_inputs(model, proc, text):
    """Run the full model once (CPU) and capture the (backbone_last_hidden_state,
    first_codebook_id) the parent feeds to the depth decoder, plus the reference
    codebook ids the stock depth decoder produces."""
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


@torch.no_grad()
def fused_depth_decode(model, backbone_hidden, first_cb, num_codebooks):
    """Single-graph greedy depth decode, OOB-free AND correct.

    Key: the device NRT_EXEC_OOB comes from the stock dynamic gathers whose offset is a
    runtime *device tensor* (`embed_tokens(ids + cache_position*vocab)`,
    `head.weight[cache_position-1]`) — the compiler can't bound the indirect copy. We
    avoid both by using compile-time python-int offsets:
      - precompute `inputs_embeds` with a static offset and pass it in, so the stock
        forward SKIPS its embed gather (and we still get correct mask/rotary/layers/cache),
      - apply the head as a static weight-slice matmul.
    Everything stays on device; caller does the single mark_step."""
    dd = model.depth_decoder
    mdl = dd.model
    head = dd.codebooks_head
    dev = backbone_hidden.device
    B = backbone_hidden.shape[0]
    vocab = mdl.vocab_size

    cache = StaticCache(config=mdl.config, max_batch_size=B, max_cache_len=num_codebooks,
                        device=dev, dtype=backbone_hidden.dtype)

    # ---- prefill positions [0,1]: ids=[0, cb0], static offset 0; pos0 = backbone hidden
    ids = torch.cat([torch.zeros((B, 1), dtype=torch.long, device=dev),
                     first_cb.view(B, 1).to(dev)], dim=1)            # [B,2]
    emb = mdl.embed_tokens(ids)                                      # offset 0 (static)
    emb = emb.clone()
    emb[:, 0] = backbone_hidden
    cp = torch.arange(0, 2, device=dev)
    out = mdl(inputs_embeds=emb, past_key_values=cache, cache_position=cp, use_cache=True)
    cb = torch.argmax(out.last_hidden_state[:, -1, :] @ head.weight[0], dim=-1)   # weight[0]
    outs = [first_cb.view(B).to(dev), cb]

    # ---- decode positions 2..num_codebooks-1 (static python-int offset (k-1)*vocab)
    for k in range(2, num_codebooks):
        emb = mdl.embed_tokens(cb.view(B, 1) + (k - 1) * vocab)     # static offset
        cp = torch.tensor([k], device=dev)
        out = mdl(inputs_embeds=emb, past_key_values=cache, cache_position=cp, use_cache=True)
        cb = torch.argmax(out.last_hidden_state[:, -1, :] @ head.weight[k - 1], dim=-1)
        outs.append(cb)

    return torch.stack(outs, dim=1)                                 # [B, num_codebooks]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=10)
    ap.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16",
                    help="fp32 makes argmax stable across CPU/device to validate the "
                         "math (rules out a structural bug vs a bf16 argmax-flip cascade)")
    ap.add_argument("--model", default=MODEL)
    args = ap.parse_args()
    dt = torch.bfloat16 if args.dtype == "bf16" else torch.float32

    dev = xm.xla_device()
    print(f"[b1'] loading {args.model} (dtype={args.dtype}) ...")
    proc = AutoProcessor.from_pretrained(args.model)
    model = CsmForConditionalGeneration.from_pretrained(args.model, dtype=dt).eval()
    model.codec_model = model.codec_model.float()
    num_cb = model.config.num_codebooks
    # CSM's depth decoder SAMPLES by default (stochastic) -> force greedy so we can
    # validate the fused argmax decode against a deterministic reference.
    gc = model.depth_decoder.generation_config
    gc.do_sample = False
    gc.temperature = None
    gc.top_k = None
    gc.top_p = None

    # 1) capture a real (backbone_hidden, first_cb) + CPU reference codebooks
    print("[b1'] capturing depth-decoder inputs + CPU reference (stock generate)...")
    cap = capture_depth_inputs(model, proc, "[0]Hello from Trainium, latency test.")
    first_cb = cap["input_ids"][:, 1]                 # [B]
    bb_hidden = cap["backbone_last_hidden_state"]     # [B, backbone_hidden]
    ref = cap["ref_codebooks"][:, 1:]                 # drop placeholder -> [B, num_cb-? ]
    print(f"[b1']   first_cb={first_cb.tolist()}  bb_hidden={tuple(bb_hidden.shape)}  "
          f"ref_codebooks={tuple(ref.shape)}")

    # 2) CPU timing of stock depth decode (baseline ~156ms)
    def cpu_depth():
        with torch.no_grad():
            out = model.depth_decoder.generate(
                input_ids=cap["input_ids"], backbone_last_hidden_state=bb_hidden.clone())
        return out if isinstance(out, torch.Tensor) else out.sequences
    for _ in range(2):
        t = time.perf_counter(); cpu_depth(); cpu_ms = (time.perf_counter() - t) * 1000
    print(f"[b1'] CPU stock depth decode: {cpu_ms:.1f}ms/frame")

    # 3) move depth decoder to device (resident), run fused single-graph decode
    print("[b1'] moving depth_decoder to device, running fused single-graph decode...")
    model.depth_decoder.to(dev)
    for m in model.depth_decoder.modules():
        for k, v in list(vars(m).items()):
            if torch.is_tensor(v) and v.device.type != "xla":
                setattr(m, k, v.to(dev))
    bb_d = bb_hidden.to(dev); fcb_d = first_cb.to(dev)

    def fused():
        out = fused_depth_decode(model, bb_d, fcb_d, num_cb)
        xm.mark_step()
        return out.cpu()

    print("[b1'] warm (compile fused graph)...")
    fused_out = fused()
    # correctness vs CPU reference (greedy => should match)
    ref_full = torch.cat([first_cb.view(-1, 1), ref], dim=1) if ref.shape[1] == num_cb - 1 else ref
    n = min(fused_out.shape[1], ref_full.shape[1])
    match = int((fused_out[:, :n] == ref_full[:, :n]).sum())
    print(f"[b1'] correctness: {match}/{n} codebooks match CPU reference "
          f"({'EXACT' if match == n else 'MISMATCH'})")
    if match != n:
        print(f"[b1']   fused={fused_out[0,:n].tolist()}")
        print(f"[b1']   ref  ={ref_full[0,:n].tolist()}")

    best = 1e9
    for _ in range(args.frames):
        t = time.perf_counter(); fused(); best = min(best, (time.perf_counter() - t) * 1000)
    print(f"[b1'] FUSED device depth decode: {best:.1f}ms/frame  (best of {args.frames})")
    print(f"\n[b1'] depth decode: CPU {cpu_ms:.1f}ms -> fused-device {best:.1f}ms "
          f"({cpu_ms/max(best,1e-6):.2f}x)")


if __name__ == "__main__":
    sys.exit(main())

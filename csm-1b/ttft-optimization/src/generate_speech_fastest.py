"""CSM-1B fastest TTFT generator — depth loop COMPILED ON-DEVICE (the fair-test winner).

Supersedes generate_speech_fast.py's CPU depth path. Wires in the measured on-device win:
the whole depth loop as ONE torch.compile(backend="neuron") graph with weights resident
(DEPTH_ON_DEVICE_FAIR.md: bf16 full-31 = 17.9ms = 7.6x faster than 137ms CPU; K=16 = 8ms).

TTFT stack (this box): backbone(~38 Neuron) + depth(~8-18 on-device) + codec(~23 CPU) +
host(~60) => ~129ms at K=16 / ~139ms full — vs ~198ms with CPU depth.

Levers on: compiled resident-weight depth loop (device), partial-depth K first frame,
CPU Mimi codec (won't compile on Neuron), backbone captured on CPU for the hidden state.
Reuses the hand-rolled loop from fair_depth_handroll.py (sibling in src/).
"""
import os
import sys
import time
import argparse
import torch
import torch_neuronx  # noqa: F401
from transformers import AutoProcessor, CsmForConditionalGeneration

sys.path.insert(0, os.path.dirname(__file__))  # bundle: fair_depth_handroll.py is a sibling in src/
sys.path.insert(0, "/host")  # box layout: fair_depth_handroll.py lives flat in /host
from fair_depth_handroll import (  # noqa: E402
    DepthWeights, make_handroll_loop, capture_depth_inputs,
)

MODEL = os.environ.get("CSM_MODEL", "/host/csm_1b")
DEV = torch.device("neuron")


def codec_decode(model, frame_codes, num_cb):
    """CPU Mimi decode of [B,K] codes (pad to num_cb) -> waveform."""
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
    ap.add_argument("--first-frame-k", type=int, default=16)
    ap.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    ap.add_argument("--out", default="/host/fastest_k16.wav")
    ap.add_argument("--iters", type=int, default=10)
    args = ap.parse_args()
    dt = torch.bfloat16 if args.dtype == "bf16" else torch.float32

    print(f"[fastest] loading {MODEL} (dtype={args.dtype})...", flush=True)
    proc = AutoProcessor.from_pretrained(MODEL)
    model = CsmForConditionalGeneration.from_pretrained(MODEL, dtype=dt).eval()
    model.codec_model = model.codec_model.float()
    num_cb = model.config.num_codebooks
    K = min(args.first_frame_k, num_cb)
    gc = model.depth_decoder.generation_config
    gc.do_sample = False
    gc.temperature = None
    gc.top_k = None
    gc.top_p = None

    print("[fastest] capturing backbone hidden state...", flush=True)
    cap = capture_depth_inputs(model, proc, "[0]Hello from Trainium, fastest TTFT test.")
    first_cb = cap["input_ids"][:, 1].clone()
    bb_hidden = cap["backbone_last_hidden_state"].clone()

    # Build the on-device compiled depth loop (resident weights).
    print("[fastest] building on-device depth weights + compiling loop...", flush=True)
    W = DepthWeights(model.depth_decoder, DEV, dt, quant=None)
    depth_loop = make_handroll_loop(W, num_cb)
    compiled = torch.compile(depth_loop, backend="neuron", dynamic=False)
    bb_d = bb_hidden.to(dt).to(DEV)
    fcb_d = first_cb.to(DEV)

    # Warm (compile) both K (partial first frame) and full-31 (refine) graphs.
    t = time.perf_counter()
    part = compiled(bb_d, fcb_d, K).cpu()
    print(f"[fastest] compile+run partial K={K}: {(time.perf_counter()-t)*1000:.0f} ms", flush=True)
    t = time.perf_counter()
    full = compiled(bb_d, fcb_d, num_cb).cpu()
    print(f"[fastest] compile+run full K={num_cb}: {(time.perf_counter()-t)*1000:.0f} ms", flush=True)

    # Warm depth timing (device).
    bestK = 1e9
    bestF = 1e9
    for _ in range(args.iters):
        t = time.perf_counter(); compiled(bb_d, fcb_d, K).cpu(); bestK = min(bestK, (time.perf_counter()-t)*1000)
        t = time.perf_counter(); compiled(bb_d, fcb_d, num_cb).cpu(); bestF = min(bestF, (time.perf_counter()-t)*1000)
    print(f"[fastest] warm depth ms/frame: K={K} -> {bestK:.2f} ms | full-{num_cb} -> {bestF:.2f} ms",
          flush=True)

    # Warm the CPU codec (first decode pays a one-time setup; a real server warms it).
    _ = codec_decode(model, full[:, :num_cb], num_cb)
    _ = codec_decode(model, full[:, :num_cb], num_cb)
    cbest = 1e9
    for _ in range(args.iters):
        cc = compiled(bb_d, fcb_d, K).cpu()
        t = time.perf_counter(); codec_decode(model, cc, num_cb); cbest = min(cbest, (time.perf_counter()-t)*1000)
    print(f"[fastest] warm codec ms/frame: {cbest:.2f} ms", flush=True)

    # First-audio: partial-K depth (device) + codec (CPU), warmed.
    t0 = time.perf_counter()
    codes_first = compiled(bb_d, fcb_d, K).cpu()
    depth_ms = (time.perf_counter() - t0) * 1000
    t1 = time.perf_counter()
    wav_first = codec_decode(model, codes_first, num_cb)
    codec_ms = (time.perf_counter() - t1) * 1000

    # prefix-exactness vs full
    exact = bool((codes_first[:, :K] == full[:, :K]).all())
    print(f"[fastest] FIRST frame K={K}: depth {depth_ms:.1f} + codec {codec_ms:.1f} = "
          f"{depth_ms+codec_ms:.1f} ms critical path (prefix-exact vs full: {exact})", flush=True)

    BACKBONE, HOST = 38.0, 60.0
    ttft = BACKBONE + bestK + cbest + HOST
    ttft_full = BACKBONE + bestF + cbest + HOST
    print(f"\n[fastest] === TTFT (depth ON-DEVICE compiled, warm codec) ===", flush=True)
    print(f"[fastest] K={K}: backbone {BACKBONE} + depth {bestK:.1f} + codec {cbest:.1f} "
          f"+ host {HOST} = ~{ttft:.0f} ms", flush=True)
    print(f"[fastest] full-{num_cb}: ~{ttft_full:.0f} ms  (vs ~198ms CPU-depth / ~446ms baseline)",
          flush=True)

    try:
        import soundfile as sf
        sf.write(args.out, wav_first.numpy(), 24000)
        print(f"[fastest] wrote {args.out} ({wav_first.numel()} samples)", flush=True)
    except Exception as e:
        print(f"[fastest] (wav save skipped: {e})", flush=True)


if __name__ == "__main__":
    sys.exit(main())

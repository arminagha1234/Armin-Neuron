"""CSM-1B STREAMING speech on Trainium — emit frame 0 ASAP, measure true TTFA.

CSM's generate calls streamer.put(codes) per audio frame. We hook a streamer that
decodes each frame's 32 codes to audio immediately (Mimi @ 12.5fps -> 1920 samples =
80ms/frame) and records time-to-first-audio (TTFA) — the metric that matters for an
interactive voice agent, instead of waiting for the whole clip.

bf16 backbone+depth, fp32 codec (validated). Run twice for a warm (cached) TTFA.
"""
import os, sys, time, argparse, torch
import torch_xla.core.xla_model as xm
from transformers import AutoProcessor, CsmForConditionalGeneration

MODEL = os.environ.get("CSM_MODEL", "/scratch/csm/csm_1b")
TEXT = "[0]Hello from Trainium, this is a streaming latency test."


def _to(obj, dev):
    if torch.is_tensor(obj): return obj.to(dev)
    try:
        from transformers.utils import ModelOutput
        if isinstance(obj, ModelOutput):
            for k in list(obj.keys()): obj[k] = _to(obj[k], dev)
            return obj
    except Exception: pass
    if obj.__class__.__name__.endswith("Cache"): return obj
    if isinstance(obj, (list, tuple)): return type(obj)(_to(x, dev) for x in obj)
    if isinstance(obj, dict): return {k: _to(v, dev) for k, v in obj.items()}
    return obj


def _offload(module, dev, method="forward"):
    module.to(dev)
    for m in module.modules():
        for k, v in list(vars(m).items()):
            if torch.is_tensor(v) and v.device.type != "xla":
                setattr(m, k, v.to(dev))
    real = getattr(module, method)
    def wrapped(*a, **k):
        out = real(*_to(a, dev), **_to(k, dev)); xm.mark_step(); return _to(out, "cpu")
    setattr(module, method, wrapped)


class AudioFrameStreamer:
    """Receives per-frame codes from generate; decodes each to audio immediately."""
    def __init__(self, model, eos_id):
        self.model = model
        self.eos_id = eos_id
        self.t0 = None
        self.ttfa = None
        self.frames = []
        self.n = 0

    def put(self, value):
        # value: codes for one frame, shape [B, num_codebooks] (or [num_codebooks])
        codes = value
        if codes.dim() == 1:
            codes = codes[None, :]
        if (codes[0, :-1] == self.eos_id).all():   # eos frame
            return
        # codes may contain ids >= codebook_size (eos/specials) -> clamp to avoid
        # OOB in the RVQ embedding lookup on-device.
        c = codes.to(torch.long).clamp_(0, 2047).reshape(1, codes.shape[-1], 1)
        with torch.no_grad():
            a = self.model.codec_model.decode(c)
        a = a[0] if isinstance(a, (list, tuple)) else getattr(a, "audio_values", a)
        a = a.detach().float().cpu().flatten()
        self.frames.append(a)
        self.n += 1
        if self.ttfa is None:
            self.ttfa = (time.perf_counter() - self.t0) * 1000.0

    def end(self):
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=12)
    ap.add_argument("--out", default="/scratch/csm/stream_out.wav")
    args = ap.parse_args()

    dev = xm.xla_device()
    proc = AutoProcessor.from_pretrained(MODEL)
    model = CsmForConditionalGeneration.from_pretrained(MODEL, dtype=torch.bfloat16).eval()
    model.codec_model = model.codec_model.float()
    _offload(model.backbone_model, dev)
    _offload(model.codec_model, dev, method="decode")
    eos_id = model.config.codebook_eos_token_id

    inputs = proc(TEXT, add_special_tokens=True, return_tensors="pt")
    gkw = dict(output_audio=False, do_sample=False, max_new_tokens=args.frames,
               cache_implementation="static")   # fixed-shape KV cache -> no per-frame recompiles

    def run(label):
        st = AudioFrameStreamer(model, eos_id)
        st.t0 = time.perf_counter()
        with torch.no_grad():
            model.generate(**inputs, **gkw, streamer=st)
        total = (time.perf_counter() - st.t0) * 1000.0
        print(f"[{label}] TTFA={st.ttfa:.1f}ms  frames={st.n}  total={total:.1f}ms  "
              f"per-frame={(total/max(st.n,1)):.1f}ms")
        return st

    print("[stream] WARM pass (compiles)...")
    run("warm")
    print("[stream] MEASURE pass (cached)...")
    st = run("measure")

    if st.frames:
        wav = torch.cat(st.frames)
        try:
            import soundfile as sf
            sf.write(args.out, wav.numpy(), 24000)
            print(f"[stream] wrote {args.out} ({wav.numel()} samples, {wav.numel()/24000:.1f}s)")
        except ImportError:
            pass
    print(f"\n  STREAMING TTFA vs targets: 100ms / 500ms")


if __name__ == "__main__":
    sys.exit(main())

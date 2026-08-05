"""Resident FireRedTTS server on Trainium (native PyTorch, all modules on the NeuronCore).

Loads the model ONCE, offloads GPT + flow + vocoder to a NeuronCore, torch.compiles the
GPT decode (`--gpt-compile`), and WARMS UP the graphs at startup so every request runs on
the hot/compiled path. Serves synthesis over a tiny stdlib HTTP endpoint and reports, per
request, the GPT TTFT (prefill->first token) and the end-to-end response time.

    python serve_fireredtts.py --model /root/firered/pretrained_models --port 8000
    # one-shot synthesis (whole utterance, then respond):
    curl 'http://127.0.0.1:8000/tts?text=Hello%20from%20Trainium&out=/root/firered/srv.wav'
    curl 'http://127.0.0.1:8000/health'
    # STREAMING: audio arrives per sentence as it's ready (progressive delivery).
    #   raw 24kHz mono s16le PCM -> pipe straight to a player:
    curl -sN 'http://127.0.0.1:8000/tts_stream?text=First%20sentence.%20Second%20sentence.' \
        | ffplay -f s16le -ar 24000 -ch_layout mono -nodisp -autoexit -
    #   or NDJSON events with per-chunk timing (shows TTFA over the wire):
    curl -sN 'http://127.0.0.1:8000/tts_stream?text=First.%20Second.&format=ndjson'

SECURITY: this binds to 127.0.0.1 and has NO authentication — it is a local dev/benchmark
server only. Do not expose it to a network or bind to 0.0.0.0 without adding auth.

APC (automatic prefix caching) is not used; the GPT KV cache is per-request/in-process.
"""
import argparse
import base64
import json
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse, parse_qs, unquote

import torch

from firered_patch import (
    apply_device_patches,
    bypass_text_normalizer,
    patch_flow_conformer_contiguous,
    patch_gpt_kv_cache_bucketed,
)

NEURON = "neuron"
_STATE = {}


def build(args):
    def _tn_ok():
        try:
            import tn  # noqa
            return True
        except Exception:
            return False

    if args.no_tn or not _tn_ok():
        bypass_text_normalizer()
    apply_device_patches()
    from fireredtts.fireredtts import FireRedTTS
    import fireredtts as _f
    from run_fireredtts_neuron import _offload, _offload_vocoder_bucketed, find_config

    repo_root = os.path.dirname(os.path.abspath(next(iter(_f.__path__))))
    os.chdir(repo_root)
    cfg = find_config(args.model)
    print(f"[serve] loading FireRedTTS (cfg={cfg})...", flush=True)
    tts = FireRedTTS(config_path=cfg, pretrained_path=args.model, device="cpu")

    sel = {"vocoder", "gpt", "flow"} if args.offload == "all" else set(args.offload.split(","))
    if "vocoder" in sel:
        _offload_vocoder_bucketed(tts.token2wav.generator, NEURON, bucket=args.bucket,
                                  compile_fwd=args.vocoder_compile)
    if "flow" in sel:
        patch_flow_conformer_contiguous()
        _offload(tts.token2wav.flow, NEURON, "inference", compile_fwd=args.flow_compile)
    if "gpt" in sel:
        patch_gpt_kv_cache_bucketed(tts, NEURON, bucket=args.gpt_bucket,
                                    num_return_sequences=args.gpt_seqs,
                                    prefill_bucket=args.gpt_prefill_bucket,
                                    compile_fwd=args.gpt_compile)

    # Per-request timers for the flow + vocoder stages (they run once per request).
    t2w = {"flow_s": 0.0, "voc_s": 0.0}
    _STATE["t2w"] = t2w
    _flow = tts.token2wav.flow.inference
    _voc = tts.token2wav.generator.forward

    def timed_flow(*a, **k):
        t = time.time(); o = _flow(*a, **k); t2w["flow_s"] += time.time() - t; return o

    def timed_voc(*a, **k):
        t = time.time(); o = _voc(*a, **k); t2w["voc_s"] += time.time() - t; return o

    tts.token2wav.flow.inference = timed_flow
    tts.token2wav.generator.forward = timed_voc

    _STATE["tts"] = tts
    _STATE["args"] = args
    return tts


def synth(text, out_path):
    tts = _STATE["tts"]
    args = _STATE["args"]
    st = getattr(tts, "_gpt_stats", None)
    if st:
        st.update({"ttft": None, "prefill_s": None, "decode_steps": 0, "decode_s": 0.0})
    t2w = _STATE.get("t2w")
    if t2w:
        t2w.update({"flow_s": 0.0, "voc_s": 0.0})
    t0 = time.time()
    with torch.no_grad():
        wav = tts.synthesize(prompt_wav=args.prompt_wav, text=text, lang=args.lang)
    total = time.time() - t0
    wav = wav.detach().cpu()
    if out_path:
        import torchaudio
        torchaudio.save(out_path, wav, 24000)
    resp = {
        "text": text,
        "samples": int(wav.shape[-1]),
        "audio_s": round(wav.shape[-1] / 24000, 3),
        "total_ms": round(total * 1000, 1),
        "out": out_path or None,
    }
    if st and st.get("ttft") is not None:
        steps = max(st["decode_steps"], 1)
        resp["gpt_ttft_ms"] = round(st["ttft"] * 1000, 1)
        resp["gpt_prefill_ms"] = round(st["prefill_s"] * 1000, 1)
        resp["gpt_decode_ms_per_step"] = round(1000 * st["decode_s"] / steps, 1)
        resp["gpt_decode_total_ms"] = round(1000 * st["decode_s"], 1)
        resp["gpt_decode_steps"] = steps
    if t2w:
        resp["flow_ms"] = round(1000 * t2w["flow_s"], 1)
        resp["vocoder_ms"] = round(1000 * t2w["voc_s"], 1)
    return resp


def _wav_to_pcm16(wav):
    """[1, T] or [T] float waveform in [-1, 1] -> little-endian s16 PCM bytes (24kHz mono)."""
    w = wav.squeeze(0) if wav.dim() > 1 else wav
    return (w.clamp(-1.0, 1.0) * 32767.0).to(torch.int16).cpu().numpy().tobytes()


def stream_synth(text):
    """Generator over sentence chunks: yields (idx, n_chunks, chunk_text, wav_cpu, synth_s,
    ready_s, gpt_s). Uses the stock per-chunk ``synthesize_base`` (same offloaded modules as
    ``/tts``), so the concatenation is identical to non-streaming — only delivery is
    incremental. ``ready_s`` is the time since the request started (i.e. TTFA for chunk 0)."""
    tts = _STATE["tts"]
    args = _STATE["args"]
    from fireredtts.modules.text_normalizer.utils import text_split

    chunks = text_split(text=text)
    t0 = time.time()
    for i, sub in enumerate(chunks):
        st = getattr(tts, "_gpt_stats", None)
        if st:
            st.update({"ttft": None, "prefill_s": None, "decode_steps": 0, "decode_s": 0.0})
        tprev = time.time()
        with torch.no_grad():
            wav = tts.synthesize_base(prompt_wav=args.prompt_wav, text=sub, lang=args.lang)
        wav = wav.detach().cpu()
        gpt_s = 0.0
        if st:
            gpt_s = (st.get("prefill_s") or 0.0) + (st.get("decode_s") or 0.0)
        yield i, len(chunks), sub, wav, time.time() - tprev, time.time() - t0, gpt_s


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # quiet default logging
        pass

    def _send(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path)
        if u.path == "/health":
            return self._send(200, {"status": "ok", "warm": _STATE.get("warm", False)})
        if u.path == "/tts":
            q = parse_qs(u.query)
            text = unquote(q.get("text", ["Hello from Trainium."])[0])
            out = q.get("out", [None])[0]
            t = time.time()
            try:
                resp = synth(text, out)
            except Exception as e:
                return self._send(500, {"error": repr(e)})
            print(f"[serve] req total={resp.get('total_ms')}ms ttft={resp.get('gpt_ttft_ms')}ms "
                  f"gpt_decode={resp.get('gpt_decode_total_ms')}ms flow={resp.get('flow_ms')}ms "
                  f"voc={resp.get('vocoder_ms')}ms text={text!r}", flush=True)
            return self._send(200, resp)
        if u.path == "/tts_stream":
            q = parse_qs(u.query)
            text = unquote(q.get("text", ["Hello from Trainium. This is a streaming test."])[0])
            out = q.get("out", [None])[0]
            fmt = q.get("format", ["pcm"])[0]
            try:
                (self._stream_ndjson if fmt == "ndjson" else self._stream_pcm)(text, out)
            except (BrokenPipeError, ConnectionResetError):
                print("[serve] stream client disconnected", flush=True)
            except Exception as e:
                try:  # headers likely already sent; best-effort
                    self._send(500, {"error": repr(e)})
                except Exception:
                    pass
            return
        return self._send(404, {"error": "use /tts?text=... , /tts_stream?text=... , or /health"})

    def _stream_pcm(self, text, out):
        """Stream raw 24kHz mono s16le PCM, flushing each sentence chunk as it's ready
        (progressive delivery; body is delimited by connection close)."""
        self.send_response(200)
        self.send_header("Content-Type", "audio/L16; rate=24000; channels=1")
        self.send_header("X-Audio-Format", "s16le-mono-24000")
        self.send_header("Connection", "close")
        self.end_headers()
        pieces = []
        for i, ntot, sub, wav, synth_s, ready_s, gpt_s in stream_synth(text):
            self.wfile.write(_wav_to_pcm16(wav))
            self.wfile.flush()
            pieces.append(wav)
            print(f"[serve] stream pcm chunk {i + 1}/{ntot} ready@{ready_s:.2f}s "
                  f"synth={synth_s:.2f}s (gpt {gpt_s:.2f}s) audio={wav.shape[-1] / 24000:.2f}s "
                  f"| {sub[:40]!r}", flush=True)
        if out and pieces:
            import torchaudio
            torchaudio.save(out, torch.cat(pieces, dim=-1), 24000)

    def _stream_ndjson(self, text, out):
        """Stream one NDJSON event per chunk (metadata + base64 s16le PCM), then a final
        summary line. Exposes per-chunk timing (ready_ms/ttfa_ms) so the streaming win is
        visible over the wire."""
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.send_header("Connection", "close")
        self.end_headers()
        pieces = []
        for i, ntot, sub, wav, synth_s, ready_s, gpt_s in stream_synth(text):
            evt = {
                "chunk": i + 1, "n_chunks": ntot, "text": sub,
                "ready_ms": round(ready_s * 1000, 1), "synth_ms": round(synth_s * 1000, 1),
                "gpt_ms": round(gpt_s * 1000, 1), "audio_s": round(wav.shape[-1] / 24000, 3),
                "samples": int(wav.shape[-1]), "sample_rate": 24000,
                "pcm_s16le_b64": base64.b64encode(_wav_to_pcm16(wav)).decode(),
            }
            if i == 0:
                evt["ttfa_ms"] = round(ready_s * 1000, 1)
            self.wfile.write((json.dumps(evt) + "\n").encode())
            self.wfile.flush()
            pieces.append(wav)
            print(f"[serve] stream ndjson chunk {i + 1}/{ntot} ready@{ready_s:.2f}s "
                  f"synth={synth_s:.2f}s | {sub[:40]!r}", flush=True)
        if pieces:
            total = torch.cat(pieces, dim=-1)
            self.wfile.write((json.dumps({
                "done": True, "n_chunks": len(pieces), "total_samples": int(total.shape[-1]),
                "total_audio_s": round(total.shape[-1] / 24000, 3),
            }) + "\n").encode())
            self.wfile.flush()
            if out:
                import torchaudio
                torchaudio.save(out, total, 24000)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("FIRERED_MODEL", "/root/firered/pretrained_models"))
    ap.add_argument("--prompt-wav", default="examples/prompt_1.wav")
    ap.add_argument("--lang", default="en")
    # Default: only the (compiled) GPT on the NeuronCore. The flow decoder and BigVGAN
    # vocoder are FASTER on CPU here — eager-Neuron dispatch makes their conv stacks slow
    # (~1.4s/~10s) and neither torch.compiles cleanly (flow: GroupNorm+shape crashes;
    # vocoder: NCC_ITIN902). GPT+compile on Neuron gives the ~40ms TTFT; flow+vocoder on CPU
    # keep total latency low and CONSTANT (no per-request recompiles). Use --offload all to
    # experiment with everything on the NeuronCore.
    ap.add_argument("--offload", default="gpt")
    ap.add_argument("--gpt-bucket", type=int, default=256)
    ap.add_argument("--gpt-prefill-bucket", type=int, default=64)
    ap.add_argument("--gpt-seqs", type=int, default=1)
    ap.add_argument("--gpt-compile", action="store_true", default=True)
    ap.add_argument("--no-gpt-compile", dest="gpt_compile", action="store_false")
    ap.add_argument("--vocoder-compile", action="store_true", default=True)
    ap.add_argument("--no-vocoder-compile", dest="vocoder_compile", action="store_false")
    ap.add_argument("--flow-compile", action="store_true", default=False)
    ap.add_argument("--bucket", type=int, default=512)
    ap.add_argument("--no-tn", action="store_true")
    ap.add_argument("--warmups", type=int, default=2)
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    # resolve prompt to absolute before chdir happens in build()
    args.prompt_wav = os.path.abspath(args.prompt_wav) if os.path.exists(args.prompt_wav) else args.prompt_wav
    build(args)
    # prompt path may be repo-relative after chdir
    if not os.path.isabs(args.prompt_wav):
        args.prompt_wav = os.path.abspath(args.prompt_wav)

    # Warm up on a FULL-LENGTH sentence (not a short phrase): compiles the GPT graphs AND
    # primes the CPU flow+vocoder oneDNN conv primitives for a representative mel length.
    # Without this the FIRST /tts_stream chunk pays a large one-time primitive-selection
    # cost (~20-50s) that dominates time-to-first-audio.
    warm_text = ("This is a warmup pass with a full length sentence so the vocoder "
                 "convolution primitives are primed for typical chunk sizes.")
    print(f"[serve] warming up ({args.warmups} passes, compiling graphs + priming primitives)...", flush=True)
    for i in range(args.warmups):
        r = synth(warm_text, None)
        print(f"[serve]   warmup {i+1}: {r['total_ms']}ms ttft={r.get('gpt_ttft_ms')}ms", flush=True)
    _STATE["warm"] = True

    srv = HTTPServer(("127.0.0.1", args.port), Handler)
    print(f"[serve] READY on http://127.0.0.1:{args.port}  "
          f"(/tts?text=...  /tts_stream?text=...[&format=ndjson]  /health)", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()

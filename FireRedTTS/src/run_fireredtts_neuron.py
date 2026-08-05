"""FireRedTTS v1 — Trainium run with the NATIVE PyTorch backend (TorchNeuron).

Native PyTorch on Neuron = eager execution on ``torch.device("neuron")`` (and
optionally ``torch.compile(backend="neuron")``), with NO torch_xla / mark_step.

Strategy: FireRedTTS's speech comes out of an HF ``.generate()`` loop whose per-step
bookkeeping uses dynamic int64 control flow that does not belong on the device. So we
keep the generate loop, sampling, tokenizer and speaker encoder on CPU, and offload the
heavy fixed-shape compute to the NeuronCore by moving those submodules to the neuron
device and marshalling their inputs/outputs CPU<->device:

  - ``token2wav.generator``  (BigVGAN vocoder, a pure conv stack)  -> Neuron   [default]
  - ``gpt.gpt``              (30-layer GPT-2 transformer forward)  -> Neuron   [--offload gpt/all]
  - ``token2wav.flow``       (flow-matching decoder)               -> Neuron   [--offload flow/all]

Usage:
    python run_fireredtts_neuron.py \
        --model ./pretrained_models \
        --prompt-wav ./FireRedTTS/examples/prompt_1.wav \
        --text "Hello from Trainium." --lang en \
        --offload vocoder --out neuron.wav

The first offloaded call compiles for the chip (minutes); later calls reuse the cache.
Run inside the native-PyTorch Neuron DLC container (torch.device("neuron") support).
"""
import argparse
import os
import sys
import time

import torch

# torch_neuronx registers the "neuron" PrivateUse1 device for native (eager) execution.
import torch_neuronx  # noqa: F401

from firered_patch import (
    apply_device_patches,
    bypass_text_normalizer,
    patch_flow_conformer_contiguous,
    patch_gpt_fixed_shape,
    patch_gpt_kv_cache_bucketed,
)

NEURON = "neuron"


def _to(obj, dev):
    """Recursively move tensors in obj to dev. Leaves *Cache objects untouched."""
    if torch.is_tensor(obj):
        return obj.to(dev)
    try:
        from transformers.utils import ModelOutput

        if isinstance(obj, ModelOutput):
            for k in list(obj.keys()):
                obj[k] = _to(obj[k], dev)
            return obj
    except Exception:
        pass
    if obj.__class__.__name__.endswith("Cache"):
        return obj
    if isinstance(obj, (list, tuple)):
        return type(obj)(_to(x, dev) for x in obj)
    if isinstance(obj, dict):
        return {k: _to(v, dev) for k, v in obj.items()}
    return obj


def _offload(module, dev, method="forward", compile_fwd=False):
    """Run ``module.method`` on the NeuronCore; marshal I/O CPU<->device. With compile_fwd,
    torch.compile(backend="neuron") the method to fuse its ops into one graph."""
    module.to(dev)
    for m in module.modules():  # move stray registered tensors that .to() misses
        for k, v in list(vars(m).items()):
            if torch.is_tensor(v) and v.device.type != dev:
                setattr(m, k, v.to(dev))
    real = getattr(module, method)
    if compile_fwd:
        real = torch.compile(real, backend="neuron", dynamic=False)

    def wrapped(*args, **kwargs):
        out = real(*_to(args, dev), **_to(kwargs, dev))
        return _to(out, "cpu")

    setattr(module, method, wrapped)


def _offload_vocoder_bucketed(module, dev, bucket=64, hop=240, compile_fwd=False):
    """Offload the BigVGAN generator to Neuron, padding the mel time-dim up to a multiple
    of ``bucket`` frames so only a few FIXED shapes are ever compiled (NEFF is reused
    across runs instead of recompiling per stochastic length), then trimming the waveform
    back to the true length. ``hop`` = product of upsample_rates (5*3*2*2*2*2 = 240) =
    output samples per mel frame. Also sidesteps the odd-length NCC_ITEN406 conv failure.
    """
    import torch.nn.functional as F

    module.to(dev)
    for m in module.modules():
        for k, v in list(vars(m).items()):
            if torch.is_tensor(v) and v.device.type != dev:
                setattr(m, k, v.to(dev))
    real = module.forward
    if compile_fwd:
        # BigVGAN is a big conv stack over long upsampled sequences; in eager mode each op
        # is dispatched individually (~10 s/call). Fuse into one Neuron graph.
        real = torch.compile(real, backend="neuron", dynamic=False)

    def wrapped(mel, *args, **kwargs):
        t = mel.shape[-1]
        padded = ((t + bucket - 1) // bucket) * bucket  # round up to bucket multiple
        if padded != t:
            mel = F.pad(mel, (0, padded - t))
        out = real(_to(mel, dev), *_to(args, dev), **_to(kwargs, dev))
        out = _to(out, "cpu")
        return out[..., : t * hop]  # trim padding-induced tail

    module.forward = wrapped


def find_config(model_dir: str) -> str:
    """HF repo's config.json is empty (0 bytes); real config is the repo's
    configs/config_24k.json. Prefer a non-empty model/config.json, else the repo copy."""
    cand = os.path.join(model_dir, "config.json")
    if os.path.exists(cand) and os.path.getsize(cand) > 0:
        return cand
    repo_cfg = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "FireRedTTS", "configs", "config_24k.json"
    )
    if os.path.exists(repo_cfg):
        return repo_cfg
    sys.exit(f"Could not find a valid config in {model_dir} or the repo configs/.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("FIRERED_MODEL", "./pretrained_models"))
    ap.add_argument("--prompt-wav", required=True, help="Reference voice .wav to clone.")
    ap.add_argument("--text", default="Hello from Trainium.")
    ap.add_argument("--lang", default="auto", choices=["zh", "en", "auto"])
    ap.add_argument(
        "--offload",
        default="vocoder",
        help="Comma-separated heavy module(s) to run on the NeuronCore: any of "
        "'vocoder', 'flow', 'gpt', or 'all'. Start with 'vocoder', then 'vocoder,flow'.",
    )
    ap.add_argument("--out", default="neuron.wav")
    ap.add_argument(
        "--warmup",
        action="store_true",
        help="Do one throwaway synthesis first to compile/cache the graphs, then measure "
        "the real run warm (reports resident-server TTFT / per-step, not cold compile).",
    )
    ap.add_argument(
        "--bucket",
        type=int,
        default=64,
        help="Pad the vocoder mel time-dim up to a multiple of this many frames so the "
        "NeuronCore compiles a few fixed shapes and reuses them across runs. 0 disables.",
    )
    ap.add_argument(
        "--gpt-mode",
        default="kvcache",
        choices=["kvcache", "recompute"],
        help="GPT decode strategy on device. 'kvcache' (default): on-device fixed-length "
        "KV cache, each step processes 1 token (fast). 'recompute': re-run the full "
        "sequence each step (use_cache=False), simpler but O(n^2).",
    )
    ap.add_argument(
        "--gpt-bucket",
        type=int,
        default=256,
        help="For --offload gpt: pad the transformer sequence / KV cache to a multiple of "
        "this many tokens (fixed-shape). Fewer, larger buckets = fewer compiles.",
    )
    ap.add_argument(
        "--gpt-seqs",
        type=int,
        default=7,
        help="For --offload gpt: num_return_sequences in the AR decode (upstream uses 7; "
        "lower = faster on device, slightly lower quality).",
    )
    ap.add_argument(
        "--gpt-prefill-bucket",
        type=int,
        default=64,
        help="For --gpt-mode kvcache: pad the (short) prompt to this small multiple so the "
        "prefill (TTFT critical path) stays cheap instead of padding to the decode bucket.",
    )
    ap.add_argument(
        "--gpt-compile",
        action="store_true",
        help="For --gpt-mode kvcache: torch.compile(backend='neuron') the transformer "
        "forward to fuse the per-step op-dispatches into one graph (much faster decode).",
    )
    ap.add_argument(
        "--vocoder-compile",
        action="store_true",
        help="torch.compile(backend='neuron') the BigVGAN vocoder (fuses its conv stack; "
        "eager is ~10s/call). Note: can hit compiler limits at large --bucket.",
    )
    ap.add_argument(
        "--flow-compile",
        action="store_true",
        help="torch.compile(backend='neuron') the flow-matching decoder inference.",
    )
    ap.add_argument(
        "--no-tn",
        action="store_true",
        help="Bypass WeTextProcessing/pynini text normalizer (use if it won't install).",
    )
    args = ap.parse_args()

    def _tn_available():
        try:
            import tn  # noqa: F401
            return True
        except Exception:
            return False

    if args.no_tn or not _tn_available():
        print("[neuron] using lite text normalizer (WeTextProcessing/pynini not in use)")
        bypass_text_normalizer()
    args.prompt_wav = os.path.abspath(args.prompt_wav)
    args.out = os.path.abspath(args.out)
    args.model = os.path.abspath(args.model)

    apply_device_patches()
    from fireredtts.fireredtts import FireRedTTS

    # Upstream config uses repo-relative asset paths (flow codebook.npy); run from repo root.
    import fireredtts as _f

    repo_root = os.path.dirname(os.path.abspath(next(iter(_f.__path__))))
    os.chdir(repo_root)

    cfg = find_config(args.model)
    print(f"[neuron] loading FireRedTTS on CPU (config={cfg}, cwd={repo_root})...")
    tts = FireRedTTS(config_path=cfg, pretrained_path=args.model, device="cpu")

    sel = {t.strip() for t in args.offload.split(",") if t.strip()}
    if "all" in sel:
        sel = {"vocoder", "gpt", "flow"}
    if "vocoder" in sel:
        if args.bucket > 0:
            print(f"[neuron] offloading BigVGAN vocoder -> NeuronCore (native, mel bucket={args.bucket}, compile={args.vocoder_compile})")
            _offload_vocoder_bucketed(tts.token2wav.generator, NEURON, bucket=args.bucket,
                                      compile_fwd=args.vocoder_compile)
        else:
            print("[neuron] offloading BigVGAN vocoder -> NeuronCore (native, no bucketing)")
            _offload(tts.token2wav.generator, NEURON, "forward")
    if "gpt" in sel:
        print(f"[neuron] offloading GPT-2 transformer -> NeuronCore (native, mode={args.gpt_mode}, "
              f"gpt_bucket={args.gpt_bucket}, seqs={args.gpt_seqs})")
        if args.gpt_mode == "kvcache":
            patch_gpt_kv_cache_bucketed(tts, NEURON, bucket=args.gpt_bucket,
                                        num_return_sequences=args.gpt_seqs,
                                        prefill_bucket=args.gpt_prefill_bucket,
                                        compile_fwd=args.gpt_compile)
        else:
            patch_gpt_fixed_shape(tts, NEURON, bucket=args.gpt_bucket, num_return_sequences=args.gpt_seqs)
    if "flow" in sel:
        print(f"[neuron] offloading flow-matching decoder -> NeuronCore (native, compile={args.flow_compile})")
        patch_flow_conformer_contiguous()  # fix conformer rel-pos strided add
        _offload(tts.token2wav.flow, NEURON, "inference", compile_fwd=args.flow_compile)

    if args.warmup:
        print("[neuron] warmup pass (compiling graphs)...")
        with torch.no_grad():
            _ = tts.synthesize(prompt_wav=args.prompt_wav, text="Warming up the chip.", lang=args.lang)
        st = getattr(tts, "_gpt_stats", None)
        if st:  # reset so the timed run reflects warm steady-state
            st.update({"ttft": None, "prefill_s": None, "decode_steps": 0, "decode_s": 0.0})

    print(f"[neuron] synthesizing: {args.text!r}{' (first call compiles)' if not args.warmup else ' (warm)'}...")
    t0 = time.time()
    with torch.no_grad():
        wav = tts.synthesize(prompt_wav=args.prompt_wav, text=args.text, lang=args.lang)
    dt = time.time() - t0

    wav = wav.detach().cpu()
    dur = wav.shape[-1] / 24000
    print(f"[neuron] {dt:.1f}s -> {wav.shape[-1]} samples (~{dur:.1f}s @ 24kHz)")

    st = getattr(tts, "_gpt_stats", None)
    if st and st.get("ttft") is not None:
        steps = max(st["decode_steps"], 1)
        print(f"[neuron] GPT TTFT (prefill->1st token): {st['ttft']*1000:.0f} ms "
              f"(prefill {st['prefill_s']*1000:.0f} ms) | decode {st['decode_steps']} steps, "
              f"{1000*st['decode_s']/steps:.1f} ms/step avg")

    import torchaudio

    torchaudio.save(args.out, wav, 24000)
    print(f"[neuron] wrote {args.out}")


if __name__ == "__main__":
    main()

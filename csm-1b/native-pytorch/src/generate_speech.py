"""CSM-1B text-to-speech on Trainium — one-command tool.

For a researcher who just wants speech out of CSM on a Trainium box. Handles the
Neuron details automatically: loads CSM, offloads the heavy compute (Llama backbone
+ Mimi codec) to a NeuronCore, keeps the generate loop + tiny depth decoder on CPU,
and writes a .wav.

Usage:
    python generate_speech.py --text "[0]Hello from Trainium." --out hello.wav
    python generate_speech.py --text "[0]How are you today?" --max-new-tokens 256

Notes:
  - "[0]" / "[1]" prefix selects the speaker id.
  - First run compiles for the chip (a few minutes); later runs are faster.
  - The 1B model runs on a single NeuronCore -> a trn2.3xlarge is enough.
  - Requires the native-PyTorch Neuron beta (torch_xla 2.9). The public beta's
    older torch_xla has int64-cast quirks that break CSM's RoPE/mask casts.
"""
import os, sys, time, argparse, torch
import torch_xla.core.xla_model as xm
from transformers import AutoProcessor, CsmForConditionalGeneration

MODEL = os.environ.get("CSM_MODEL", os.path.expanduser("~/csm/csm_1b"))


def _to(obj, dev):
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


def _offload(module, dev, method="forward"):
    """Run `module.method` on the NeuronCore; inputs/outputs marshalled CPU<->device."""
    module.to(dev)
    for m in module.modules():                      # move stray RVQ tensors .to() misses
        for k, v in list(vars(m).items()):
            if torch.is_tensor(v) and v.device.type != "xla":
                setattr(m, k, v.to(dev))
    real = getattr(module, method)

    def wrapped(*args, **kwargs):
        out = real(*_to(args, dev), **_to(kwargs, dev))
        xm.mark_step()
        return _to(out, "cpu")
    setattr(module, method, wrapped)


def load_model(model_path: str = MODEL):
    dev = xm.xla_device()
    proc = AutoProcessor.from_pretrained(model_path)
    model = CsmForConditionalGeneration.from_pretrained(model_path, dtype=torch.float32).eval()
    _offload(model.backbone_model, dev)             # 16-layer transformer -> Neuron
    _offload(model.codec_model, dev, method="decode")  # Mimi codec -> Neuron
    return proc, model


def generate(proc, model, text: str, max_new_tokens: int = 256) -> torch.Tensor:
    inputs = proc(text, add_special_tokens=True, return_tensors="pt")
    with torch.no_grad():
        audio = model.generate(**inputs, output_audio=True, do_sample=False,
                               max_new_tokens=max_new_tokens)
    a = audio[0] if isinstance(audio, (list, tuple)) else audio
    return a.detach().float().cpu().flatten()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", default="[0]Hello from Trainium.")
    ap.add_argument("--out", default="speech.wav")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    args = ap.parse_args()

    print("[csm] loading + offloading to NeuronCore (first call compiles)...")
    proc, model = load_model()
    print(f"[csm] generating: {args.text!r}")
    t0 = time.time()
    wav = generate(proc, model, args.text, args.max_new_tokens)
    print(f"[csm] {time.time()-t0:.1f}s -> {wav.numel()} samples (~{wav.numel()/24000:.1f}s @ 24kHz)")

    try:
        import soundfile as sf
        sf.write(args.out, wav.numpy(), 24000)
        print(f"[csm] wrote {args.out}")
    except ImportError:
        torch.save(wav, args.out + ".pt")
        print(f"[csm] soundfile not installed; saved tensor -> {args.out}.pt (pip install soundfile for wav)")


if __name__ == "__main__":
    sys.exit(main())

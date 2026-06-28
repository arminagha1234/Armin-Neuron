"""CSM-1B end-to-end on Trainium via heavy-module OFFLOAD.

HF `generate` won't lower to Neuron (its loop/cache bookkeeping emits int64 dynamic
ops). So we keep the model + generate loop on CPU and offload only the heavy compute
modules to a NeuronCore:
  - backbone_model (16-layer transformer)  -> Neuron
  - codec_model (Mimi decode)              -> Neuron
  - depth_decoder (4 layers), embeddings, lm_head, sampling, loop -> CPU

Each offloaded forward moves its inputs CPU->xla, runs on Neuron, then returns CPU
tensors, so the CPU-side generate machinery is unaffected. `use_cache=False` avoids
KV-cache device juggling (backbone re-prefills each frame — fine for short clips).
"""
import sys, time, argparse, torch
import torch_xla.core.xla_model as xm
from transformers import AutoProcessor, CsmForConditionalGeneration

MODEL = "/scratch/csm/csm_1b"
TEXT = "[0]Hello from Trainium."


def move_stray(mod, dev):
    n = 0
    for m in mod.modules():
        for k, v in list(vars(m).items()):
            if torch.is_tensor(v) and v.device.type != "xla":
                setattr(m, k, v.to(dev)); n += 1
    return n


def _to(obj, dev):
    if torch.is_tensor(obj):
        return obj.to(dev)
    # Preserve transformers ModelOutput containers (positional [0] indexing is used)
    try:
        from transformers.utils import ModelOutput
        if isinstance(obj, ModelOutput):
            for k in list(obj.keys()):
                obj[k] = _to(obj[k], dev)
            return obj
    except Exception:
        pass
    # Leave Cache objects alone (their tensors already live on the compute device)
    if obj.__class__.__name__.endswith("Cache"):
        return obj
    if isinstance(obj, (list, tuple)):
        return type(obj)(_to(x, dev) for x in obj)
    if isinstance(obj, dict):
        return {k: _to(v, dev) for k, v in obj.items()}
    return obj


def offload(module, dev, method="forward"):
    """Move `module` to `dev` and wrap `method` to accept/return CPU tensors."""
    move_stray(module.to(dev), dev)
    real = getattr(module, method)

    def wrapped(*args, **kwargs):
        args = _to(args, dev); kwargs = _to(kwargs, dev)
        out = real(*args, **kwargs)
        xm.mark_step()
        return _to(out, "cpu")
    setattr(module, method, wrapped)
    return module


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-new-tokens", type=int, default=16)
    ap.add_argument("--out", default="/scratch/csm/neuron_out.wav")
    ap.add_argument("--offload-depth", action="store_true",
                    help="also run the depth decoder on Neuron (all compute on-device)")
    args = ap.parse_args()

    dev = xm.xla_device()
    proc = AutoProcessor.from_pretrained(MODEL)
    inputs = proc(TEXT, add_special_tokens=True, return_tensors="pt")
    gen_kw = dict(output_audio=True, do_sample=False, max_new_tokens=args.max_new_tokens)

    # CPU reference
    m_cpu = CsmForConditionalGeneration.from_pretrained(MODEL, dtype=torch.float32).eval()
    with torch.no_grad():
        a_cpu = m_cpu.generate(**inputs, **gen_kw)
    a_cpu = (a_cpu[0] if isinstance(a_cpu, (list, tuple)) else a_cpu).detach().float().cpu().flatten()
    print(f"[cpu] audio samples={a_cpu.numel()} std={a_cpu.std():.4e}")

    # Offloaded model (backbone + codec on Neuron, rest CPU)
    m = CsmForConditionalGeneration.from_pretrained(MODEL, dtype=torch.float32).eval()
    offload(m.backbone_model, dev)
    offload(m.codec_model, dev, method="decode")
    msg = "backbone_model (forward) + codec_model (decode)"
    if args.offload_depth:
        offload(m.depth_decoder.model, dev)
        msg += " + depth_decoder.model (forward)"
    print(f"[neuron] offloaded to NeuronCore: {msg}")

    t0 = time.time()
    with torch.no_grad():
        a_n = m.generate(**inputs, **gen_kw)
    a_n = (a_n[0] if isinstance(a_n, (list, tuple)) else a_n).detach().float().cpu().flatten()
    print(f"[neuron] generate {time.time()-t0:.1f}s audio samples={a_n.numel()} std={a_n.std():.4e}")

    n = min(a_cpu.numel(), a_n.numel())
    if n > 0:
        cos = torch.nn.functional.cosine_similarity(a_cpu[:n], a_n[:n], dim=0).item()
        print(f"[compare] cosine cpu-vs-neuron = {cos:.6f}")
    try:
        import soundfile as sf
        sf.write(args.out, a_n.numpy(), 24000)
        print(f"[neuron] wrote {args.out}")
    except Exception as e:
        print("write err", e)


if __name__ == "__main__":
    sys.exit(main())

"""Fine-grained CPU fallback: force specific aten ops in the trunk to run on
CPU (via PrivateUse1 impl override) while everything else stays on Neuron.

OPS env = comma-separated aten op names to fall back, e.g. "cat" or "cat,chunk".
Goal: keep the heavy einsum/linear/layernorm on Neuron, offload only the
slice/concat op that triggers the tensor_set_slice assertion.
"""
import os, time, torch
import torch_neuronx

OPS = [s for s in os.environ.get("OPS", "cat").split(",") if s]
MODELS = "/home/ubuntu/workspace/native_venv/lib/python3.12/site-packages/downloads/models_v2/"

def _to_cpu(x):
    if isinstance(x, torch.Tensor):
        return x.cpu()
    if isinstance(x, (list, tuple)):
        return type(x)(_to_cpu(v) for v in x)
    if isinstance(x, dict):
        return {k: _to_cpu(v) for k, v in x.items()}
    return x

def _find_dev(x):
    if isinstance(x, torch.Tensor) and x.device.type == "neuron":
        return x.device
    if isinstance(x, (list, tuple)):
        for v in x:
            d = _find_dev(v)
            if d: return d
    return None

def _to_dev(x, dev):
    if isinstance(x, torch.Tensor):
        return x.to(dev)
    if isinstance(x, (list, tuple)):
        return type(x)(_to_dev(v, dev) for v in x)
    return x

_INVOKED = {}
def make_fallback(op, name):
    def fb(*args, **kwargs):
        if name not in _INVOKED:
            _INVOKED[name] = 0
            print(f"[FALLBACK INVOKED] aten::{name} (first call)", flush=True)
        _INVOKED[name] += 1
        dev = _find_dev(args) or _find_dev(list(kwargs.values()))
        r = op(*[_to_cpu(a) for a in args], **{k: _to_cpu(v) for k, v in kwargs.items()})
        return _to_dev(r, dev) if dev is not None else r
    return fb

lib = torch.library.Library("aten", "IMPL")
for name in OPS:
    try:
        op = getattr(torch.ops.aten, name).default
    except Exception as e:
        print(f"[fallback] SKIP aten::{name} (no .default: {e})", flush=True)
        continue
    lib.impl(name, make_fallback(op, name), "PrivateUse1")
    print(f"[fallback] registered CPU fallback for aten::{name}", flush=True)

d = torch.load("/home/ubuntu/cap_trunk.pt.inputs.pt", weights_only=False)
crop, kw = d["crop_size"], d["kw"]
device = torch.device("neuron")
m = torch.jit.load(MODELS + "trunk.pt", map_location="cpu").to(device)
kw_dev = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in kw.items()}
fwd = getattr(m, f"forward_{crop}")

print(f"[fallback OPS={OPS}] running trunk forward_{crop} on neuron ...", flush=True)
t0 = time.time()
with torch.no_grad():
    out = fwd(**kw_dev)
outs = [o.detach().to("cpu").float() for o in (out if isinstance(out, (list, tuple)) else [out])]
print(f"TRUNK NEURON OK in {time.time()-t0:.1f}s; outputs={[tuple(o.shape) for o in outs]}", flush=True)

# parity vs CPU reference
try:
    ref = torch.load("/home/ubuntu/cap_trunk.pt.ref.pt", weights_only=False)
    refs = [r.detach().float() for r in (ref if isinstance(ref, (list, tuple)) else [ref])]
    for i, (a, b) in enumerate(zip(outs, refs)):
        print(f"  out[{i}] max_abs_diff={(a-b).abs().max().item():.4e} mean={(a-b).abs().mean().item():.4e}")
except Exception as e:
    print("  ref compare:", e)
print("TRUNK_FALLBACK_DONE", flush=True)

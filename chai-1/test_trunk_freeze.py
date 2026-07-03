"""Try to run the trunk on Neuron after graph transforms that change how the
eager backend partitions it into NEFFs.

Modes (env MODE):
  freeze   - torch.jit.freeze (inline submodules, constant-fold)
  ofi      - freeze + optimize_for_inference
  compile  - torch.compile(backend="neuron") wrapping the scripted call
"""
import os, time, torch
import torch_neuronx

MODE = os.environ.get("MODE", "freeze")
MODELS = "/home/ubuntu/workspace/native_venv/lib/python3.12/site-packages/downloads/models_v2/"
d = torch.load("/home/ubuntu/cap_trunk.pt.inputs.pt", weights_only=False)
crop, kw = d["crop_size"], d["kw"]
device = torch.device("neuron")

m = torch.jit.load(MODELS + "trunk.pt", map_location="cpu").eval()
method = f"forward_{crop}"

# Move to device FIRST, then freeze, so constant-folded buffers land on neuron.
DEVFIRST = os.environ.get("DEVFIRST", "1") == "1"
if DEVFIRST:
    m = m.to(device)

if MODE in ("freeze", "ofi"):
    try:
        m = torch.jit.freeze(m, preserved_attrs=[method])
        print(f"[{MODE}] froze module (preserved {method}); devfirst={DEVFIRST}", flush=True)
    except Exception as e:
        print("freeze failed:", repr(e)[:200], flush=True)
    if MODE == "ofi":
        try:
            m = torch.jit.optimize_for_inference(m, other_methods=[method])
            print("[ofi] optimize_for_inference applied", flush=True)
        except Exception as e:
            print("ofi failed:", repr(e)[:200], flush=True)

if not DEVFIRST:
    m = m.to(device)
CONTIG = os.environ.get("CONTIG", "0") == "1"
def prep(v):
    if not torch.is_tensor(v):
        return v
    if CONTIG:
        v = v.contiguous()
    return v.to(device)
kw_dev = {k: prep(v) for k, v in kw.items()}
if CONTIG:
    print("[cfg] forced .contiguous() on all inputs", flush=True)
fwd = getattr(m, method)

if MODE == "compile":
    def call(**kwargs):
        return fwd(**kwargs)
    fwd_run = torch.compile(call, backend="neuron")
    runner = lambda: fwd_run(**kw_dev)
else:
    runner = lambda: fwd(**kw_dev)

print(f"[{MODE}] running trunk on neuron ...", flush=True)
t0 = time.time()
with torch.no_grad():
    out = runner()
outs = [o.detach().to("cpu").float() for o in (out if isinstance(out, (list, tuple)) else [out])]
print(f"[{MODE}] TRUNK NEURON OK in {time.time()-t0:.1f}s; outputs={[tuple(o.shape) for o in outs]}", flush=True)
print("TRUNK_FREEZE_DONE", flush=True)

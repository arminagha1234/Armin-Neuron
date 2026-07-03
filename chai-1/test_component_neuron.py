"""Run one chai TorchScript component eagerly on the Neuron device and compare
to the captured CPU reference.

Usage: python test_component_neuron.py <comp_key>
  e.g. python test_component_neuron.py trunk.pt
"""
import os, sys, time, torch
import torch_neuronx  # registers "neuron" backend

if os.environ.get("DET") == "1":
    # disable adaptive eager op-grouping -> run ops unfused/deterministically
    torch.use_deterministic_algorithms(True)
    print("[cfg] deterministic algorithms ON (adaptive eager grouping disabled)", flush=True)

COMP = sys.argv[1] if len(sys.argv) > 1 else "trunk.pt"
MODELS = "/home/ubuntu/workspace/native_venv/lib/python3.12/site-packages/downloads/models_v2/"
CAP = f"/home/ubuntu/cap_{COMP}.inputs.pt"
REF = f"/home/ubuntu/cap_{COMP}.ref.pt"

d = torch.load(CAP, weights_only=False)
crop, kw = d["crop_size"], d["kw"]
print(f"[{COMP}] crop={crop} keys={list(kw.keys())}", flush=True)

device = torch.device("neuron")
m = torch.jit.load(MODELS + COMP, map_location="cpu").to(device)
kw_dev = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in kw.items()}
fwd = getattr(m, f"forward_{crop}")

print(f"[{COMP}] running forward_{crop} on neuron ...", flush=True)
t0 = time.time()
with torch.no_grad():
    out = fwd(**kw_dev)

def flatten(o):
    if isinstance(o, dict):
        return list(o.values())
    if isinstance(o, (list, tuple)):
        return list(o)
    return [o]

outs = [o.detach().to("cpu").float() for o in flatten(out) if torch.is_tensor(o)]
print(f"[{COMP}] NEURON OK in {time.time()-t0:.1f}s; outputs={[tuple(o.shape) for o in outs]}", flush=True)

try:
    ref = torch.load(REF, weights_only=False)
    refs = [r.detach().float() for r in flatten(ref) if torch.is_tensor(r)]
    for i, (a, b) in enumerate(zip(outs, refs)):
        if a.shape != b.shape:
            print(f"  out[{i}] SHAPE MISMATCH neuron={tuple(a.shape)} cpu={tuple(b.shape)}")
            continue
        print(f"  out[{i}] shape={tuple(a.shape)} max_abs_diff={ (a-b).abs().max().item():.4e} "
              f"mean_abs_diff={ (a-b).abs().mean().item():.4e}")
except Exception as e:
    print("  ref compare failed:", e)
print(f"COMPONENT_DONE {COMP}", flush=True)

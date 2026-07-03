"""
Minimal reproducer: chai-1 trunk fails on Native PyTorch (Beta 3) with a
Neuron runtime assertion, while the same module runs fine on CPU.

    tensor.c:185: tensor_set_slice:
      Assertion `(tensor_source->_size) >= (offset + size)' failed.

Environment (Beta 3):
  - DLC: 421672808698.dkr.ecr.us-east-1.amazonaws.com/concourse-release-0461d3b
  - driver aws-neuronx-dkms 2.28.0.0, runtime-lib 2.32.19.0
  - native_venv: torch 2.11.0 + torch_neuronx (native "neuron" PrivateUse1)
  - neuronx-cc 2.0.253257.0a0, nki 0.4.0b4
  - Hardware: trn2.48xlarge

Inputs: trunk_inputs.pt  (captured from a real chai-1 CPU inference; the trunk
is called as module.forward_256(**kwargs); token bucket = 256).

Run:
  # on CPU (works):
  DEVICE=cpu    python reproduce.py
  # on Neuron (fails with the assertion):
  DEVICE=neuron python reproduce.py

The trunk ships only as a pre-traced TorchScript artifact (chai has no eager
model source), so this loads the .pt and moves it to the target device.
"""
import os, time, torch
import torch_neuronx  # registers the "neuron" backend

DEVICE = os.environ.get("DEVICE", "neuron")
# Point this at the downloaded chai trunk artifact:
TRUNK = os.environ.get(
    "TRUNK_PT",
    "/home/ubuntu/workspace/native_venv/lib/python3.12/site-packages/downloads/models_v2/trunk.pt",
)

d = torch.load("trunk_inputs.pt", weights_only=False)
crop, kw = d["crop_size"], d["kw"]
print("token bucket (crop_size):", crop)
for k, v in kw.items():
    if torch.is_tensor(v):
        print(f"  {k:34s} {tuple(v.shape)} {v.dtype}")

device = torch.device(DEVICE)
m = torch.jit.load(TRUNK, map_location="cpu").to(device)
kw_dev = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in kw.items()}
fwd = getattr(m, f"forward_{crop}")

print(f"\nrunning trunk.forward_{crop} on device={DEVICE} ...", flush=True)
t0 = time.time()
with torch.no_grad():
    out = fwd(**kw_dev)
outs = [o.detach().to("cpu").float() for o in (out if isinstance(out, (list, tuple)) else [out])]
print(f"OK in {time.time()-t0:.1f}s; outputs={[tuple(o.shape) for o in outs]}")
print("DONE")

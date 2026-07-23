"""Task 12: native-PyTorch VideoMAEv2 forward on torch.device('neuron'), validated vs CPU.
No XLA. Exercises the real Conv3d tubelet embed + 12 ViT blocks with pretrained weights.
"""
import os
import time

import torch
import torch_neuronx  # registers 'neuron' device
from huggingface_hub import snapshot_download

from modeling_videomaev2_native import build_videomaev2_base, load_pretrained_weights

REPO = "OpenGVLab/VideoMAEv2-Base"
local = snapshot_download(REPO, allow_patterns=["model.safetensors", "config.json"])
st = os.path.join(local, "model.safetensors")

torch.manual_seed(0)
model = build_videomaev2_base().eval()
missing, unexpected = load_pretrained_weights(model, st)
n_params = sum(p.numel() for p in model.parameters())
print(f"params: {n_params/1e6:.2f} M   missing_keys: {missing}   unexpected_keys: {unexpected}")

# Fixed input (B, C=3, T=16, H=224, W=224)
x = torch.randn(1, 3, 16, 224, 224)

# ---- CPU reference ----
with torch.no_grad():
    cpu_out = model(x)
print(f"CPU  out: shape={tuple(cpu_out.shape)} mean={float(cpu_out.mean()):.5f} std={float(cpu_out.std()):.5f}")

# ---- Neuron (native eager) ----
dev = torch.device("neuron")
model_n = model.to(dev)
x_n = x.to(dev)
t0 = time.time()
with torch.no_grad():
    neu_out = model_n(x_n)
neu_cpu = neu_out.cpu()
print(f"NEURON out: shape={tuple(neu_cpu.shape)} (first-run compile+exec {time.time()-t0:.1f}s)")

diff = (cpu_out - neu_cpu).abs()
print(f"max_abs_diff={float(diff.max()):.3e}  mean_abs_diff={float(diff.mean()):.3e}")
ok = torch.allclose(cpu_out, neu_cpu, rtol=1e-2, atol=1e-2)
print(f"allclose(rtol=1e-2, atol=1e-2): {ok}")
print("FORWARD_OK" if ok else "FORWARD_MISMATCH")

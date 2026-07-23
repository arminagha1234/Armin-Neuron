"""Task 13: native-PyTorch single-core training step for VideoMAEv2 on torch.device('neuron').
No XLA. Fine-tuning setup: pretrained ViT-B backbone + fresh linear head, CrossEntropy + AdamW.
Overfits one fixed batch to prove grads flow through Conv3d/attention and the optimizer updates.
"""
import os
import time

import torch
import torch.nn.functional as F
import torch_neuronx  # registers 'neuron' device
from huggingface_hub import snapshot_download

from modeling_videomaev2_native import build_videomaev2_base, load_pretrained_weights

REPO = "OpenGVLab/VideoMAEv2-Base"
NUM_CLASSES = 400   # Kinetics-400-style head
BATCH = 2
STEPS = 10

local = snapshot_download(REPO, allow_patterns=["model.safetensors"])
st = os.path.join(local, "model.safetensors")

torch.manual_seed(0)
model = build_videomaev2_base(num_classes=NUM_CLASSES).train()
missing, unexpected = load_pretrained_weights(model, st)
print(f"loaded pretrained backbone. missing (expect head.*): {missing}  unexpected: {unexpected}")

dev = torch.device("neuron")
model = model.to(dev)
opt = torch.optim.AdamW(model.parameters(), lr=1e-4)

# one fixed batch (overfit it -> loss should fall)
x = torch.randn(BATCH, 3, 16, 224, 224).to(dev)
labels = torch.randint(0, NUM_CLASSES, (BATCH,)).to(dev)

print("step |   loss   | sec  (step 0 includes fwd+bwd NEFF compile)")
losses = []
for step in range(STEPS):
    t0 = time.time()
    opt.zero_grad()
    logits = model(x)
    loss = F.cross_entropy(logits, labels)
    loss.backward()
    opt.step()
    l = float(loss.detach().cpu())
    losses.append(l)
    print(f"{step:>4d} | {l:8.4f} | {time.time()-t0:5.1f}")

print(f"\nfirst_loss={losses[0]:.4f}  last_loss={losses[-1]:.4f}  decreased={losses[-1] < losses[0]}")
print("TRAIN_OK" if losses[-1] < losses[0] else "TRAIN_NO_PROGRESS")

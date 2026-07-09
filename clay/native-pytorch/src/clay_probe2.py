"""Localize backward crash: encoder advanced-index gather vs decoder scatter-assign."""
import torch, traceback
dev = torch.device("neuron")
torch.manual_seed(0)
B, L, D, keep = 2, 64, 384, 16

def stage(name, fn):
    try:
        fn(); print(f"[OK]   {name}", flush=True)
    except Exception as e:
        print(f"[FAIL] {name}: {type(e).__name__}: {str(e)[:150]}", flush=True)

# 1: advanced-index GATHER backward:  y = x[batch_idx, idx, :]
def g1():
    x = torch.randn(B, L, D, device=dev, requires_grad=True)
    idx = torch.stack([torch.randperm(L, device=dev)[:keep] for _ in range(B)])
    bi = torch.arange(B, device=dev).unsqueeze(1)
    y = x[bi, idx, :]
    y.sum().backward()
    _ = float(x.grad.float().sum().to("cpu"))
stage("1: advanced-index gather backward (encoder mask_out style)", g1)

# 2: advanced-index SCATTER-ASSIGN backward:  buf[batch_idx, idx, :] = src
def g2():
    src = torch.randn(B, keep, D, device=dev, requires_grad=True)
    idx = torch.stack([torch.randperm(L, device=dev)[:keep] for _ in range(B)])
    bi = torch.arange(B, device=dev).unsqueeze(1)
    buf = torch.zeros(B, L, D, device=dev)
    buf[bi, idx, :] = src
    buf.sum().backward()
    _ = float(src.grad.float().sum().to("cpu"))
stage("2: advanced-index scatter-assign backward (decoder style)", g2)

# 3: torch.gather backward (Neuron-friendly alternative)
def g3():
    x = torch.randn(B, L, D, device=dev, requires_grad=True)
    idx = torch.stack([torch.randperm(L, device=dev)[:keep] for _ in range(B)])
    y = torch.gather(x, 1, idx.unsqueeze(-1).expand(-1, -1, D))
    y.sum().backward()
    _ = float(x.grad.float().sum().to("cpu"))
stage("3: torch.gather backward (alternative)", g3)

# 4: torch.scatter backward (Neuron-friendly alternative)
def g4():
    src = torch.randn(B, keep, D, device=dev, requires_grad=True)
    idx = torch.stack([torch.randperm(L, device=dev)[:keep] for _ in range(B)])
    buf = torch.zeros(B, L, D, device=dev)
    buf = buf.scatter(1, idx.unsqueeze(-1).expand(-1, -1, D), src)
    buf.sum().backward()
    _ = float(src.grad.float().sum().to("cpu"))
stage("4: torch.scatter backward (alternative)", g4)

print("[DONE]", flush=True)

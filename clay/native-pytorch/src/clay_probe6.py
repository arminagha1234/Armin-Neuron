"""Find the crashing op inside Transformer backward. Safe candidates first, crash last."""
import torch
from torch import nn
from claymodel.backbone import Transformer, FeedForward
dev = torch.device("neuron")
torch.manual_seed(0)
B,N,dim = 2,17,384

def stage(name, fn):
    try:
        fn(); print(f"[OK]   {name}", flush=True)
    except Exception as e:
        print(f"[FAIL] {name}: {type(e).__name__}: {str(e)[:150]}", flush=True)

# 1: Transformer backward, fused_attn=False (candidate FIX)
def a1():
    tr = Transformer(dim=dim, depth=2, heads=6, dim_head=64, mlp_dim=dim*2, fused_attn=False).to(dev)
    x = torch.randn(B,N,dim,device=dev,requires_grad=True)
    tr(x).sum().backward(); _=float(x.grad.float().sum().to("cpu"))
stage("1: Transformer backward fused_attn=FALSE", a1)

# 2: FeedForward backward (LayerNorm+Linear+GELU+Linear)
def a2():
    ff = FeedForward(dim, dim*2).to(dev)
    x = torch.randn(B,N,dim,device=dev,requires_grad=True)
    ff(x).sum().backward(); _=float(x.grad.float().sum().to("cpu"))
stage("2: FeedForward backward", a2)

# 3: LayerNorm backward
def a3():
    ln = nn.LayerNorm(dim).to(dev)
    x = torch.randn(B,N,dim,device=dev,requires_grad=True)
    ln(x).sum().backward(); _=float(x.grad.float().sum().to("cpu"))
stage("3: LayerNorm backward", a3)

# 4: chunk(3) backward  (to_qkv split)
def a4():
    x = torch.randn(B,N,dim*3,device=dev,requires_grad=True)
    q,k,v = x.chunk(3, dim=-1)
    (q.sum()+k.sum()+v.sum()).backward(); _=float(x.grad.float().sum().to("cpu"))
stage("4: chunk(3,dim=-1) backward", a4)

# 5: Transformer backward, fused_attn=TRUE (expected crash)
def a5():
    tr = Transformer(dim=dim, depth=2, heads=6, dim_head=64, mlp_dim=dim*2, fused_attn=True).to(dev)
    x = torch.randn(B,N,dim,device=dev,requires_grad=True)
    tr(x).sum().backward(); _=float(x.grad.float().sum().to("cpu"))
stage("5: Transformer backward fused_attn=TRUE", a5)

print("[DONE]", flush=True)

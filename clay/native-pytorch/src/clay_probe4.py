"""Isolate encoder backward crash: Transformer(SDPA) vs patch-embed+mask path."""
import torch, math
import torch.nn.functional as F
from claymodel.backbone import Transformer
from claymodel.model import Encoder
dev = torch.device("neuron")
torch.manual_seed(0)
B, C, grid, patch, dim = 2, 6, 8, 16, 384
H = W = grid*patch
waves = torch.linspace(0.4,2.2,C,device=dev); gsd=torch.tensor(10.0,device=dev)
time_=torch.randn(B,4,device=dev); latlon=torch.randn(B,4,device=dev)
pixels=torch.randn(B,C,H,W,device=dev)

def stage(name, fn):
    try:
        fn(); print(f"[OK]   {name}", flush=True)
    except Exception as e:
        print(f"[FAIL] {name}: {type(e).__name__}: {str(e)[:150]}", flush=True)

# 1: Transformer (SDPA) backward alone
def t1():
    tr = Transformer(dim=dim, depth=2, heads=6, dim_head=64, mlp_dim=dim*2, fused_attn=True).to(dev)
    x = torch.randn(B, 17, dim, device=dev, requires_grad=True)
    tr(x).sum().backward()
    _ = float(x.grad.float().sum().to("cpu"))
stage("1: Transformer/SDPA backward", t1)

# 2: encoder patch-embed + add_encodings + mask_out, loss on unmasked (NO transformer)
def t2():
    enc = Encoder(mask_ratio=0.75, patch_size=patch, shuffle=True, dim=dim,
                  depth=2, heads=6, dim_head=64, mlp_ratio=2).to(dev)
    p,_ = enc.to_patch_embed(pixels.requires_grad_(True), waves)
    p = enc.add_encodings(p, time_, latlon, gsd)
    unmasked,_,_,_ = enc.mask_out(p)
    unmasked.sum().backward()
    g = sum((q.grad**2).sum() for q in enc.parameters() if q.grad is not None)
    _ = float(g.float().to("cpu"))
stage("2: patch-embed+mask_out backward (no transformer)", t2)

# 3: manual SDPA backward (raw F.scaled_dot_product_attention)
def t3():
    q = torch.randn(B,6,17,64,device=dev,requires_grad=True)
    k = torch.randn(B,6,17,64,device=dev,requires_grad=True)
    v = torch.randn(B,6,17,64,device=dev,requires_grad=True)
    o = F.scaled_dot_product_attention(q,k,v,dropout_p=0.0)
    o.sum().backward()
    _ = float(q.grad.float().sum().to("cpu"))
stage("3: raw F.scaled_dot_product_attention backward", t3)

print("[DONE]", flush=True)

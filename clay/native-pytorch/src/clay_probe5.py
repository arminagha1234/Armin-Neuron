"""Confirm SDPA backward is the crash & that manual attention backward works."""
import torch
import torch.nn.functional as F
dev = torch.device("neuron")
torch.manual_seed(0)
B,Hd,N,Dh = 2,6,17,64

def stage(name, fn):
    try:
        fn(); print(f"[OK]   {name}", flush=True)
    except Exception as e:
        print(f"[FAIL] {name}: {type(e).__name__}: {str(e)[:150]}", flush=True)

# forward-only SDPA (should be fine)
def s_fwd():
    q=torch.randn(B,Hd,N,Dh,device=dev); k=torch.randn(B,Hd,N,Dh,device=dev); v=torch.randn(B,Hd,N,Dh,device=dev)
    o=F.scaled_dot_product_attention(q,k,v,dropout_p=0.0)
    _=float(o.float().sum().to("cpu"))
stage("SDPA forward-only", s_fwd)

# SDPA backward
def s_bwd():
    q=torch.randn(B,Hd,N,Dh,device=dev,requires_grad=True)
    k=torch.randn(B,Hd,N,Dh,device=dev,requires_grad=True)
    v=torch.randn(B,Hd,N,Dh,device=dev,requires_grad=True)
    F.scaled_dot_product_attention(q,k,v,dropout_p=0.0).sum().backward()
    _=float(q.grad.float().sum().to("cpu"))
stage("SDPA backward", s_bwd)

# manual attention backward (fused_attn=False path)
def m_bwd():
    q=torch.randn(B,Hd,N,Dh,device=dev,requires_grad=True)
    k=torch.randn(B,Hd,N,Dh,device=dev,requires_grad=True)
    v=torch.randn(B,Hd,N,Dh,device=dev,requires_grad=True)
    attn=(torch.matmul(q,k.transpose(-1,-2))*(Dh**-0.5)).softmax(dim=-1)
    torch.matmul(attn,v).sum().backward()
    _=float(q.grad.float().sum().to("cpu"))
stage("manual matmul+softmax attention backward", m_bwd)

print("[DONE]", flush=True)

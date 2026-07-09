"""Localize which sub-module's BACKWARD crashes: encoder vs decoder vs dynamic-embed."""
import torch, traceback
import torch.nn.functional as F
from einops import rearrange
from claymodel.model import Encoder, Decoder
from claymodel.factory import DynamicEmbedding
dev = torch.device("neuron")
torch.manual_seed(0)
B, C, grid, patch, dim = 2, 6, 8, 16, 384
H = W = grid*patch
waves = torch.linspace(0.4, 2.2, C, device=dev)
gsd = torch.tensor(10.0, device=dev)
time_ = torch.randn(B,4,device=dev); latlon = torch.randn(B,4,device=dev)
pixels = torch.randn(B,C,H,W,device=dev)

def stage(name, fn):
    try:
        fn(); print(f"[OK]   {name}", flush=True)
    except Exception as e:
        print(f"[FAIL] {name}: {type(e).__name__}: {str(e)[:150]}", flush=True)

# 1: DynamicEmbedding ENCODER-side (conv2d dynamic weight) backward
def d1():
    de = DynamicEmbedding(wave_dim=128, num_latent_tokens=128, patch_size=patch,
                          embed_dim=dim, is_decoder=False).to(dev)
    x = pixels.clone().requires_grad_(True)
    out,_ = de(x, waves)
    out.sum().backward()
    _ = float(x.grad.float().sum().to("cpu"))
stage("1: DynamicEmbedding(conv2d) backward", d1)

# 2: DynamicEmbedding DECODER-side (F.linear dynamic weight) backward
def d2():
    de = DynamicEmbedding(wave_dim=128, num_latent_tokens=128, patch_size=patch,
                          embed_dim=dim, is_decoder=True).to(dev)
    L = grid*grid + 1
    x = torch.randn(B, L, dim, device=dev, requires_grad=True)
    out,_ = de(x, waves)
    out.sum().backward()
    _ = float(x.grad.float().sum().to("cpu"))
stage("2: DynamicEmbedding(linear/decoder) backward", d2)

# 3: full Encoder backward
def d3():
    enc = Encoder(mask_ratio=0.75, patch_size=patch, shuffle=True, dim=dim,
                  depth=2, heads=6, dim_head=64, mlp_ratio=2).to(dev)
    out = enc({"pixels":pixels,"time":time_,"latlon":latlon,"gsd":gsd,"waves":waves})
    out[0].sum().backward()
    g = sum((p.grad**2).sum() for p in enc.parameters() if p.grad is not None)
    _ = float(g.float().to("cpu"))
stage("3: full Encoder backward", d3)

# 4: full Decoder backward (encoder output detached as leaf)
def d4():
    enc = Encoder(mask_ratio=0.75, patch_size=patch, shuffle=True, dim=dim,
                  depth=2, heads=6, dim_head=64, mlp_ratio=2).to(dev)
    dec = Decoder(mask_ratio=0.75, patch_size=patch, encoder_dim=dim, dim=192,
                  depth=2, heads=4, dim_head=64, mlp_ratio=2).to(dev)
    with torch.no_grad():
        e = enc({"pixels":pixels,"time":time_,"latlon":latlon,"gsd":gsd,"waves":waves})
    enc_out = e[0].clone().requires_grad_(True)
    px,_ = dec(enc_out, e[1], e[2], e[3], time_, latlon, gsd, waves)
    px.sum().backward()
    g = sum((p.grad**2).sum() for p in dec.parameters() if p.grad is not None)
    _ = float(g.float().to("cpu"))
stage("4: full Decoder backward", d4)

print("[DONE]", flush=True)

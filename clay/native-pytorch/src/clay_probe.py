"""Stage-by-stage probe to localize which op the Neuron eager runtime rejects."""
import argparse, traceback
import torch
from claymodel.model import Encoder, Decoder

ap = argparse.ArgumentParser()
ap.add_argument("--device", default="neuron")
args = ap.parse_args()
dev = torch.device(args.device)
torch.manual_seed(0)

B, C, grid, patch, dim = 2, 6, 8, 16, 384
H = W = grid * patch
waves = torch.linspace(0.4, 2.2, C, device=dev)
gsd = torch.tensor(10.0, device=dev)
time_ = torch.randn(B, 4, device=dev)
latlon = torch.randn(B, 4, device=dev)
pixels = torch.randn(B, C, H, W, device=dev)

def stage(name, fn):
    try:
        out = fn()
        # force materialization
        if isinstance(out, torch.Tensor):
            _ = float(out.float().sum().to("cpu"))
        print(f"[OK]   {name}")
        return out
    except Exception as e:
        print(f"[FAIL] {name}: {type(e).__name__}: {str(e)[:200]}")
        traceback.print_exc()
        raise SystemExit(1)

enc = Encoder(mask_ratio=0.75, patch_size=patch, shuffle=True, dim=dim,
              depth=2, heads=6, dim_head=64, mlp_ratio=2).to(dev)

# A: dynamic patch embed (conv2d with runtime-generated weight)
patches, _ = stage("A: DynamicEmbedding conv2d patch-embed",
                    lambda: enc.to_patch_embed(pixels, waves))
patches = enc.add_encodings(patches, time_, latlon, gsd)
stage("B: add_encodings (hstack+repeat+cat)", lambda: patches)

# C: mask_out — advanced indexing patches[batch_idx, unmasked_idx, :]
def do_mask():
    return enc.mask_out(patches)[0]
stage("C: mask_out advanced-index gather", do_mask)

# D: full encoder forward
def do_enc():
    return enc({"pixels": pixels, "time": time_, "latlon": latlon,
                "gsd": gsd, "waves": waves})[0]
enc_out = stage("D: full Encoder.forward", do_enc)
full = enc({"pixels": pixels, "time": time_, "latlon": latlon, "gsd": gsd, "waves": waves})

dec = Decoder(mask_ratio=0.75, patch_size=patch, encoder_dim=dim, dim=192,
              depth=2, heads=4, dim_head=64, mlp_ratio=2).to(dev)

# E: decoder scatter-assignment (decoder_patches[batch_idx, idx, :] = ...)
def do_dec():
    return dec(full[0], full[1], full[2], full[3], time_, latlon, gsd, waves)[0]
stage("E: Decoder.forward (scatter-assign reconstruct)", do_dec)

import random
from einops import rearrange, reduce
import torch.nn.functional as F

# F: channel-drop augmentation (python loop + 4D advanced-index assign)
def channel_drop(px, ll):
    _px = px.clone()
    bs, ch, _, _ = _px.size()
    for i in range(bs):
        if torch.any(ll[i] != 0):
            r = random.random()
            if r < 0.10:
                _px[i, :, :, :] = 0
            elif r < 0.30:
                idx = torch.randperm(ch)[: ch // 2]
                _px[i, idx, :, :] = 0
    return _px
random.seed(0)
stage("F: channel_drop (python loop + 4D index-assign)", lambda: channel_drop(pixels, latlon))

# G: per_pixel_loss
def per_pixel_loss(cube, px, mm, ps=patch):
    pp = rearrange(cube, "B C (h p1) (w p2) -> B (h w) (C p1 p2)", p1=ps, p2=ps)
    loss = F.l1_loss(pp, px, reduction="none")
    loss = reduce(loss, "B L D -> B L", reduction="mean")
    return (loss * mm).sum() / mm.sum()
dec_pixels = dec(full[0], full[1], full[2], full[3], time_, latlon, gsd, waves)[0]
loss = stage("G: per_pixel_loss", lambda: per_pixel_loss(pixels, dec_pixels, full[3]))

# H: backward through the whole graph
def do_bwd():
    enc2 = enc({"pixels": pixels, "time": time_, "latlon": latlon, "gsd": gsd, "waves": waves})
    px2 = dec(enc2[0], enc2[1], enc2[2], enc2[3], time_, latlon, gsd, waves)[0]
    l = per_pixel_loss(pixels, px2, enc2[3])
    l.backward()
    g = sum((p.grad.detach()**2).sum() for p in list(enc.parameters())+list(dec.parameters()) if p.grad is not None)
    return g
stage("H: backward (fwd+bwd full graph)", do_bwd)

print("\n[PROBE COMPLETE] all stages ran on", args.device)

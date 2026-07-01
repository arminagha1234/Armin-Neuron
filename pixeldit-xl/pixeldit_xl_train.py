#!/usr/bin/env python3
"""PixelDiT-XL (~1B) training on AWS Neuron via native PyTorch (Beta 3 stack).

Pixel-space Diffusion Transformer (DiT-XL class, scaled to ~1B params) trained
with a rectified-flow velocity objective. Uses torch.device("neuron") +
torch.compile(backend="neuron") per the Native PyTorch User Guide (Beta 3).

Modes:
  (default)  single-core training smoke / benchmark on random or real data
  --check    CPU-vs-Neuron forward parity test (correctness)
  --data-dir DIR   train on an ImageFolder dataset instead of synthetic noise
  --save-dir DIR   checkpoint every --save-every steps; --resume CKPT to resume

Run inside the Beta 3 container, e.g.:
    NEURON_RT_VISIBLE_CORES=6 python3 /work/pixeldit_xl_train.py \
        --steps 3 --batch 1 --image-size 256 --patch 16

Beta 3 constraints honored: dynamic=False, bf16 compute, plain SDPA (no attn
bias), GELU(tanh), LayerNorm(elementwise_affine=False) adaLN-Zero blocks.
"""
import argparse
import math
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden, freq_dim=256):
        super().__init__()
        self.freq_dim = freq_dim
        self.mlp = nn.Sequential(
            nn.Linear(freq_dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden)
        )

    def forward(self, t):
        half = self.freq_dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / half
        )
        args = t[:, None].float() * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return self.mlp(emb.to(self.mlp[0].weight.dtype))


class LabelEmbedder(nn.Module):
    def __init__(self, num_classes, hidden):
        super().__init__()
        self.emb = nn.Embedding(num_classes + 1, hidden)  # +1 = CFG null token

    def forward(self, y):
        return self.emb(y)


class PatchEmbed(nn.Module):
    def __init__(self, img, patch, in_ch, hidden):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, hidden, kernel_size=patch, stride=patch)
        self.num_patches = (img // patch) ** 2

    def forward(self, x):
        x = self.proj(x)  # B, hidden, H/p, W/p
        return x.flatten(2).transpose(1, 2)  # B, N, hidden


class Attention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.heads = heads
        self.hd = dim // heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.heads, self.hd).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        x = F.scaled_dot_product_attention(q, k, v)  # full attn, no bias/mask
        x = x.transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


class Block(nn.Module):
    def __init__(self, dim, heads, mlp_ratio=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(dim, heads)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(approximate="tanh"),
            nn.Linear(dim * mlp_ratio, dim),
        )
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    def __init__(self, dim, patch, out_ch):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(dim, patch * patch * out_ch)
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim))

    def forward(self, x, c):
        shift, scale = self.adaLN(c).chunk(2, dim=1)
        x = modulate(self.norm(x), shift, scale)
        return self.linear(x)


class PixelDiTXL(nn.Module):
    def __init__(self, img=256, patch=16, in_ch=3, hidden=1408, depth=28, heads=16, num_classes=1000):
        super().__init__()
        self.in_ch, self.patch, self.img = in_ch, patch, img
        self.x_embed = PatchEmbed(img, patch, in_ch, hidden)
        n = self.x_embed.num_patches
        self.pos = nn.Parameter(torch.zeros(1, n, hidden))
        self.t_embed = TimestepEmbedder(hidden)
        self.y_embed = LabelEmbedder(num_classes, hidden)
        self.blocks = nn.ModuleList([Block(hidden, heads) for _ in range(depth)])
        self.final = FinalLayer(hidden, patch, in_ch)
        nn.init.normal_(self.pos, std=0.02)

    def unpatchify(self, x):
        B = x.shape[0]
        p, c = self.patch, self.in_ch
        h = w = self.img // p
        x = x.reshape(B, h, w, p, p, c).permute(0, 5, 1, 3, 2, 4).reshape(B, c, h * p, w * p)
        return x

    def forward(self, x, t, y):
        x = self.x_embed(x) + self.pos
        c = self.t_embed(t) + self.y_embed(y)
        for blk in self.blocks:
            x = blk(x, c)
        x = self.final(x, c)
        return self.unpatchify(x)


def build_model(args, dtype):
    return PixelDiTXL(
        img=args.image_size, patch=args.patch, in_ch=args.in_ch, hidden=args.hidden,
        depth=args.depth, heads=args.heads, num_classes=args.num_classes,
    ).to(dtype)


def rf_batch(x0, num_classes, device):
    """Build a rectified-flow training batch from clean images x0."""
    B = x0.shape[0]
    noise = torch.randn_like(x0)
    t = torch.rand(B, device=device)
    y = torch.randint(0, num_classes, (B,), device=device)
    tb = t.view(B, 1, 1, 1).to(x0.dtype)
    xt = (1 - tb) * x0 + tb * noise
    target = noise - x0
    return xt, (t * 1000.0), y, target


def make_data_iter(args, device):
    """Yield (x0) clean-image batches: real ImageFolder if --data-dir else synthetic."""
    B, C, H = args.batch, args.in_ch, args.image_size
    if args.data_dir:
        import torchvision  # noqa
        from torchvision import transforms
        from torchvision.datasets import ImageFolder
        tf = transforms.Compose([
            transforms.Resize(H), transforms.CenterCrop(H),
            transforms.ToTensor(), transforms.Normalize([0.5] * C, [0.5] * C),
        ])
        ds = ImageFolder(args.data_dir, transform=tf)
        loader = torch.utils.data.DataLoader(ds, batch_size=B, shuffle=True, drop_last=True, num_workers=2)
        print(f"[data] ImageFolder {args.data_dir}: {len(ds)} images, {len(ds.classes)} classes")
        while True:
            for imgs, _ in loader:
                yield imgs.to(device, dtype=torch.bfloat16)
    else:
        while True:
            yield torch.randn(B, C, H, H, device=device, dtype=torch.bfloat16)


def run_check(args):
    """CPU-vs-Neuron forward parity test in fp32 (correctness)."""
    torch.manual_seed(0)
    print(f"[check] building model hidden={args.hidden} depth={args.depth} img={args.image_size} patch={args.patch}")
    cpu_model = build_model(args, torch.float32).eval()
    B, C, H = args.batch, args.in_ch, args.image_size
    x = torch.randn(B, C, H, H)
    t = (torch.rand(B) * 1000.0)
    y = torch.randint(0, args.num_classes, (B,))

    with torch.inference_mode():
        out_cpu = cpu_model(x, t, y).float()

    device = torch.device("neuron")
    nrn_model = cpu_model.to(device)  # identical weights
    xn, tn, yn = x.to(device), t.to(device), y.to(device)
    with torch.inference_mode():
        out_nrn = nrn_model(xn, tn, yn).float().cpu()
    torch.neuron.synchronize()

    diff = (out_cpu - out_nrn).abs()
    rel = diff / (out_cpu.abs() + 1e-5)
    max_abs = diff.max().item()
    mean_abs = diff.mean().item()
    max_rel = rel.max().item()
    print(f"[check] output shape {tuple(out_cpu.shape)}")
    print(f"[check] max_abs={max_abs:.3e}  mean_abs={mean_abs:.3e}  max_rel={max_rel:.3e}")
    ok = max_abs < 2e-2 and mean_abs < 1e-3
    print(f"[check] PARITY {'PASS' if ok else 'FAIL'} (fp32 CPU vs Neuron)")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image-size", type=int, default=256)
    ap.add_argument("--patch", type=int, default=16)
    ap.add_argument("--in-ch", type=int, default=3)
    ap.add_argument("--hidden", type=int, default=1408)
    ap.add_argument("--depth", type=int, default=28)
    ap.add_argument("--heads", type=int, default=16)
    ap.add_argument("--num-classes", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--no-compile", action="store_true")
    ap.add_argument("--check", action="store_true", help="CPU-vs-Neuron parity test then exit")
    ap.add_argument("--data-dir", default=None, help="ImageFolder root for real data")
    ap.add_argument("--save-dir", default=None, help="dir to write checkpoints")
    ap.add_argument("--save-every", type=int, default=0)
    ap.add_argument("--resume", default=None, help="checkpoint path to resume from")
    args = ap.parse_args()

    if args.check:
        raise SystemExit(run_check(args))

    torch.manual_seed(0)
    print(f"[stage] building PixelDiT-XL hidden={args.hidden} depth={args.depth} "
          f"heads={args.heads} img={args.image_size} patch={args.patch}")
    model = build_model(args, torch.bfloat16)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[info] parameter count = {n_params/1e9:.3f} B ({n_params:,})")
    print(f"[info] tokens/seq = {model.x_embed.num_patches}")

    device = torch.device("neuron")
    t0 = time.time()
    model = model.to(device).train()
    print(f"[stage] model on neuron in {time.time()-t0:.1f}s")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    start_step = 0
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict({k: v.to(device) for k, v in ckpt["model"].items()})
        opt.load_state_dict(ckpt["opt"])
        start_step = ckpt.get("step", 0) + 1
        print(f"[ckpt] resumed from {args.resume} at step {start_step}")

    fwd = model
    if not args.no_compile:
        print("[stage] torch.compile(backend='neuron', dynamic=False)")
        fwd = torch.compile(model, backend="neuron", dynamic=False)

    data = make_data_iter(args, device)
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)

    for step in range(start_step, start_step + args.steps):
        x0 = next(data)
        xt, t_in, y, target = rf_batch(x0, args.num_classes, device)
        ts = time.time()
        opt.zero_grad(set_to_none=True)
        pred = fwd(xt, t_in, y)
        loss = F.mse_loss(pred.float(), target.float())
        loss.backward()
        opt.step()
        torch.neuron.synchronize()
        dt = (time.time() - ts) * 1000.0
        tag = "first (compile+run)" if step == start_step else "step"
        print(f"[train] step {step} loss={loss.item():.4f}  {tag} = {dt:.1f} ms")

        if args.save_dir and args.save_every and (step + 1) % args.save_every == 0:
            path = os.path.join(args.save_dir, f"pixeldit_step{step}.pt")
            sd = {k: v.detach().to("cpu") for k, v in model.state_dict().items()}
            torch.save({"model": sd, "opt": opt.state_dict(), "step": step,
                        "args": vars(args)}, path)
            print(f"[ckpt] saved {path}")

    print("[done] PixelDiT-XL training works on Neuron (Beta 3 native PyTorch).")


if __name__ == "__main__":
    main()

"""
REAL VideoMAE v2 self-supervised pretraining (not a smoke test) — native PyTorch.

Runs the authentic objective end-to-end: tube-masked encoder -> decoder -> per-patch
normalized-pixel reconstruction MSE on the MASKED patches only, training ALL params
(encoder + decoder + mask token + encoder_to_decoder).

Data: freshly-generated *structured* synthetic video every step (drifting low-frequency
sinusoidal gratings). Structure is reconstructable, so a falling recon loss over steps
reflects genuine learning — not memorization of one fixed batch.

Usage:
  python pretrain_neuron.py --device cpu    --steps 3     # sanity
  python pretrain_neuron.py --device neuron --steps 30    # Trainium
"""
import argparse
import math
import time

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange

from modeling_pretrain_native import build_pretrain_videomae_base, tube_mask_indices

PATCH, TUBELET, T, H, W, C = 16, 2, 16, 224, 224, 3
Tp, Hp, Wp = T // TUBELET, H // PATCH, W // PATCH   # 8, 14, 14


def make_structured_clips(B, rng):
    """(B,C,T,H,W) drifting sinusoidal gratings in [0,1] — varied but reconstructable."""
    tt = torch.arange(T).view(1, 1, T, 1, 1).float()
    yy = torch.linspace(0, 2 * math.pi, H).view(1, 1, 1, H, 1)
    xx = torch.linspace(0, 2 * math.pi, W).view(1, 1, 1, 1, W)
    g = torch.from_numpy(rng.random((B, C, 1, 1, 1)).astype("float32"))
    fx = g * 2 + 1
    fy = torch.from_numpy(rng.random((B, C, 1, 1, 1)).astype("float32")) * 2 + 1
    ph = torch.from_numpy(rng.random((B, C, 1, 1, 1)).astype("float32")) * 2 * math.pi
    sp = (torch.from_numpy(rng.random((B, C, 1, 1, 1)).astype("float32")) - 0.5) * 0.6
    img = torch.sin(fx * xx + fy * yy + ph + sp * tt)
    return ((img + 1) / 2).float()


def make_target(images):
    """Patchify + per-patch normalization -> (B, N, 3*tubelet*patch^2). Matches engine."""
    sq = rearrange(images, "b c (t p0) (h p1) (w p2) -> b (t h w) (p0 p1 p2) c",
                   p0=TUBELET, p1=PATCH, p2=PATCH)
    norm = (sq - sq.mean(dim=-2, keepdim=True)) / (
        sq.var(dim=-2, unbiased=True, keepdim=True).sqrt() + 1e-6)
    return rearrange(norm, "b n p c -> b n (p c)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu", choices=["cpu", "neuron"])
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1.5e-4)
    ap.add_argument("--mask-ratio", type=float, default=0.9)
    args = ap.parse_args()

    if args.device == "neuron":
        import torch_neuronx  # noqa: F401  registers the 'neuron' device

    rng = np.random.RandomState(0)
    torch.manual_seed(0)

    model = build_pretrain_videomae_base().train()
    n_enc = sum(p.numel() for p in model.encoder.parameters())
    n_dec = sum(p.numel() for p in model.decoder.parameters())
    n_tot = sum(p.numel() for p in model.parameters())
    print(f"model params: total {n_tot/1e6:.2f}M (encoder {n_enc/1e6:.2f}M + decoder {n_dec/1e6:.2f}M + heads)")

    dev = torch.device(args.device)
    model = model.to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)

    print(f"device={args.device}  batch={args.batch}  mask_ratio={args.mask_ratio}")
    print("step |  recon_loss | sec   (step 0 includes fwd+bwd NEFF compile)")
    losses = []
    for step in range(args.steps):
        t0 = time.time()
        images = make_structured_clips(args.batch, rng)                       # (B,C,T,H,W)
        ids_keep, ids_mask = tube_mask_indices(args.batch, Tp, Hp, Wp, args.mask_ratio, rng)
        images = images.to(dev)
        ids_keep = ids_keep.to(dev)
        ids_mask = ids_mask.to(dev)

        with torch.no_grad():
            target = make_target(images)                                      # (B,N,1536)
            Cpx = target.shape[-1]
            labels = torch.gather(target, 1, ids_mask.unsqueeze(-1).expand(-1, -1, Cpx))

        opt.zero_grad()
        outputs = model(images, ids_keep, ids_mask)                           # (B,N_mask,1536)
        loss = ((outputs - labels) ** 2).mean()
        loss.backward()
        opt.step()

        l = float(loss.detach().cpu())
        losses.append(l)
        print(f"{step:>4d} | {l:11.6f} | {time.time()-t0:5.1f}", flush=True)

    first5 = sum(losses[:5]) / min(5, len(losses))
    last5 = sum(losses[-5:]) / min(5, len(losses))
    print(f"\nmean(first5)={first5:.6f}  mean(last5)={last5:.6f}  decreased={last5 < first5}")
    print("PRETRAIN_OK" if last5 < first5 else "PRETRAIN_NO_PROGRESS")


if __name__ == "__main__":
    main()

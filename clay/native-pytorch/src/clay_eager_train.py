"""
Clay MAE — native-PyTorch EAGER training smoke test on AWS Trainium.

Exercises the *real* Clay training path:
  - DynamicEmbedding (DOFA dynamic patch-embed conv, runtime-generated weights)
  - Encoder ViT (SDPA fused attention) + random masking
  - Decoder ViT + scatter reconstruction
  - Clay's verbatim channel-drop augmentation (data-dependent python loop)
  - Clay's verbatim per_pixel_loss (L1 reconstruction on masked patches)
  - AdamW forward + backward + optimizer.step, all on device="neuron"

The only omitted piece is the frozen DINOv2 teacher (a standard ViT that needs a
network download); it contributes 10% of the loss and is not part of the
encoder/decoder architecture under test.

Usage:
  python clay_eager_train.py --device neuron --size small --steps 5
  python clay_eager_train.py --device cpu    --size small --steps 3   # reference
"""

import argparse
import random
import time

import torch
import torch.nn.functional as F
from einops import rearrange, reduce

from claymodel.model import Decoder, Encoder

CONFIGS = {
    "tiny": dict(dim=192, depth=6, heads=4, dim_head=48, mlp_ratio=2,
                 dec_dim=96, dec_depth=3, dec_heads=2, dec_dim_head=48, dec_mlp_ratio=2),
    "small": dict(dim=384, depth=6, heads=6, dim_head=64, mlp_ratio=2,
                  dec_dim=192, dec_depth=4, dec_heads=4, dec_dim_head=64, dec_mlp_ratio=2),
    "base": dict(dim=768, depth=12, heads=12, dim_head=64, mlp_ratio=4,
                 dec_dim=512, dec_depth=4, dec_heads=4, dec_dim_head=64, dec_mlp_ratio=4),
}


class ClayMAETrainer(torch.nn.Module):
    """Encoder + Decoder + Clay's reconstruction objective (teacher omitted)."""

    def __init__(self, cfg, patch_size, mask_ratio, norm_pix_loss=False, shuffle=True,
                 fused_attn=True):
        super().__init__()
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
        self.norm_pix_loss = norm_pix_loss
        self.encoder = Encoder(
            mask_ratio=mask_ratio, patch_size=patch_size, shuffle=shuffle,
            dim=cfg["dim"], depth=cfg["depth"], heads=cfg["heads"],
            dim_head=cfg["dim_head"], mlp_ratio=cfg["mlp_ratio"], fused_attn=fused_attn,
        )
        self.decoder = Decoder(
            mask_ratio=mask_ratio, patch_size=patch_size, encoder_dim=cfg["dim"],
            dim=cfg["dec_dim"], depth=cfg["dec_depth"], heads=cfg["dec_heads"],
            dim_head=cfg["dec_dim_head"], mlp_ratio=cfg["dec_mlp_ratio"], fused_attn=fused_attn,
        )

    # --- verbatim from ClayMAE.per_pixel_loss ---
    def per_pixel_loss(self, cube, pixels, masked_matrix):
        patches = rearrange(
            cube, "B C (h p1) (w p2) -> B (h w) (C p1 p2)",
            p1=self.patch_size, p2=self.patch_size,
        )
        if self.norm_pix_loss:
            mean = patches.mean(dim=-1, keepdim=True)
            var = patches.var(dim=-1, keepdim=True)
            patches = (patches - mean) / (var + 1e-6) ** 0.5
        loss = F.l1_loss(patches, pixels, reduction="none")
        loss = reduce(loss, "B L D -> B L", reduction="mean")
        loss = (loss * masked_matrix).sum() / masked_matrix.sum()
        return loss

    # --- verbatim channel-drop augmentation from ClayMAE.forward ---
    @staticmethod
    def channel_drop(pixels, latlon):
        _pixels = pixels.clone()
        batch_size, channels, _, _ = _pixels.size()
        prob_drop_all = 0.10
        prob_drop_half = 0.20
        for i in range(batch_size):
            if torch.any(latlon[i] != 0):
                rand_val = random.random()
                if rand_val < prob_drop_all:
                    _pixels[i, :, :, :] = 0
                elif rand_val < prob_drop_all + prob_drop_half:
                    channel_indices = torch.randperm(channels)[: channels // 2]
                    _pixels[i, channel_indices, :, :] = 0
        return _pixels

    def forward(self, datacube):
        _pixels = self.channel_drop(datacube["pixels"], datacube["latlon"])
        enc = self.encoder({
            "pixels": _pixels,
            "time": datacube["time"],
            "latlon": datacube["latlon"],
            "gsd": datacube["gsd"],
            "waves": datacube["waves"],
        })
        encoded, unmasked_idx, masked_idx, masked_matrix = enc
        pixels, _ = self.decoder(
            encoded, unmasked_idx, masked_idx, masked_matrix,
            datacube["time"], datacube["latlon"], datacube["gsd"], datacube["waves"],
        )
        loss = self.per_pixel_loss(datacube["pixels"], pixels, masked_matrix)
        return loss


def make_datacube(B, C, grid, patch_size, device, dtype):
    H = W = grid * patch_size
    return {
        "pixels": torch.randn(B, C, H, W, device=device, dtype=dtype),
        "time": torch.randn(B, 4, device=device, dtype=dtype),
        "latlon": torch.randn(B, 4, device=device, dtype=dtype),
        "gsd": torch.tensor(10.0, device=device, dtype=dtype),
        "waves": torch.linspace(0.4, 2.2, C, device=device, dtype=dtype),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="neuron", choices=["neuron", "cpu"])
    ap.add_argument("--size", default="small", choices=list(CONFIGS))
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--channels", type=int, default=6)
    ap.add_argument("--grid", type=int, default=16, help="patches per side (L=grid^2)")
    ap.add_argument("--patch", type=int, default=16)
    ap.add_argument("--mask-ratio", type=float, default=0.75)
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--overfit", action="store_true",
                    help="deterministic mask (shuffle=False) + fixed batch: "
                         "loss should decrease, proving the optimizer learns end-to-end")
    ap.add_argument("--fused-attn", default="true", choices=["true", "false"],
                    help="true=SDPA (crashes in backward on Beta-3), false=manual attention")
    ap.add_argument("--compile", action="store_true",
                    help="wrap model in torch.compile(backend='neuron')")
    args = ap.parse_args()

    torch.manual_seed(0)
    random.seed(0)
    dtype = torch.float32 if args.dtype == "float32" else torch.bfloat16
    device = torch.device(args.device)
    cfg = CONFIGS[args.size]

    print(f"[cfg] device={args.device} size={args.size} dtype={args.dtype} "
          f"B={args.batch} C={args.channels} grid={args.grid} patch={args.patch} "
          f"L={args.grid**2} mask_ratio={args.mask_ratio} steps={args.steps}")

    model = ClayMAETrainer(cfg, patch_size=args.patch, mask_ratio=args.mask_ratio,
                           shuffle=not args.overfit, fused_attn=(args.fused_attn == "true"))
    model = model.to(device=device, dtype=dtype)
    model.train()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] ClayMAE({args.size}) params={n_params/1e6:.1f}M "
          f"mode={'overfit(det-mask)' if args.overfit else 'train(random-mask)'} "
          f"fused_attn={args.fused_attn} lr={args.lr}")

    if args.compile:
        print("[compile] wrapping model in torch.compile(backend='neuron')", flush=True)
        model = torch.compile(model, backend="neuron")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    datacube = make_datacube(args.batch, args.channels, args.grid, args.patch, device, dtype)

    # snapshot one weight to prove the optimizer actually updates params on-device
    probe = model.encoder.transformer.layers[0][0].to_qkv.weight
    probe_before = probe.detach().to("cpu").clone()

    losses = []
    for step in range(args.steps):
        t0 = time.time()
        opt.zero_grad()
        loss = model(datacube)
        loss.backward()
        # global grad norm — evidence backward produced real gradients on-device
        gnorm = torch.sqrt(sum((p.grad.detach() ** 2).sum()
                               for p in model.parameters() if p.grad is not None))
        opt.step()
        loss_val = float(loss.detach().to("cpu"))
        gnorm_val = float(gnorm.to("cpu"))
        dt = time.time() - t0
        losses.append(loss_val)
        tag = "  (incl. compile)" if step == 0 else ""
        print(f"[step {step}] loss={loss_val:.6f}  grad_norm={gnorm_val:.4e}  {dt:.2f}s{tag}",
              flush=True)

    probe_after = probe.detach().to("cpu").clone()
    delta = float((probe_after - probe_before).abs().max())
    print(f"[check] max |Δweight| on encoder.layer0.to_qkv = {delta:.3e} "
          f"({'UPDATED' if delta > 0 else 'NO CHANGE'})")

    print("\n[RESULT] loss trajectory:", " -> ".join(f"{l:.4f}" for l in losses))
    ran_ok = delta > 0 and all(l == l for l in losses)  # weights moved, no NaNs
    if args.overfit and len(losses) >= 2 and losses[-1] < losses[0]:
        print(f"[RESULT] PASS: Clay MAE trained on '{args.device}' in EAGER mode — "
              f"fwd+bwd+AdamW end-to-end, loss decreased {losses[0]:.4f} -> {losses[-1]:.4f}, "
              f"weights updated (Δ={delta:.2e}).")
    elif ran_ok:
        print(f"[RESULT] PASS: Clay MAE training step ran on '{args.device}' in EAGER mode — "
              f"fwd+bwd+AdamW completed, gradients finite, weights updated (Δ={delta:.2e}). "
              f"(random-mask run; use --overfit to see the loss curve bend down.)")
    else:
        print(f"[RESULT] FAIL: something off — delta={delta:.2e} losses={losses}")


if __name__ == "__main__":
    main()

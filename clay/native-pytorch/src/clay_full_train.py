"""
WHOLE Clay model — native-PyTorch EAGER training on Trainium.

Runs the real ClayMAE end-to-end:
  Encoder + Decoder + DynamicEmbedding + masking + channel-drop
  + reconstruction loss (90%)
  + frozen DINOv2 teacher (timm) + representation loss (10%)
  through forward -> backward -> AdamW on device="neuron".

Uses Clay's real config: size/patch/teacher from configs/config.yaml,
sentinel-2-l2a metadata from configs/metadata.yaml.
"""

import argparse
import time
import torch

from claymodel.model import ClayMAE, clay_mae_config


class DotDict(dict):
    """Attribute access for nested dicts (mimics the Box object Clay uses)."""
    def __getattr__(self, k):
        return self[k]
    def __getitem__(self, k):
        v = dict.__getitem__(self, k)
        return DotDict(v) if isinstance(v, dict) else v


# sentinel-2-l2a block from Clay configs/metadata.yaml
S2_META = DotDict({
    "sentinel-2-l2a": {
        "band_order": ["blue", "green", "red", "rededge1", "rededge2", "rededge3",
                       "nir", "nir08", "swir16", "swir22"],
        "rgb_indices": [2, 1, 0],
        "gsd": 10,
        "bands": {
            "wavelength": {
                "blue": 0.493, "green": 0.56, "red": 0.665, "rededge1": 0.704,
                "rededge2": 0.74, "rededge3": 0.783, "nir": 0.842, "nir08": 0.865,
                "swir16": 1.61, "swir22": 2.19,
            }
        },
    }
})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="neuron", choices=["neuron", "cpu"])
    ap.add_argument("--size", default="base", choices=["tiny", "small", "base", "large"])
    ap.add_argument("--teacher-impl", default="transformers", choices=["transformers", "timm"])
    ap.add_argument("--teacher", default="facebook/dinov2-large")
    ap.add_argument("--patch", type=int, default=8)
    ap.add_argument("--img", type=int, default=256)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--mask-ratio", type=float, default=0.75)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    ap.add_argument("--fused-attn", default="false", choices=["true", "false"])
    ap.add_argument("--compile", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(0)
    dtype = torch.float32 if args.dtype == "float32" else torch.bfloat16
    device = torch.device(args.device)
    cfg = clay_mae_config(args.size)

    print(f"[cfg] device={args.device} size={args.size} teacher={args.teacher} "
          f"patch={args.patch} img={args.img} L={(args.img//args.patch)**2} "
          f"B={args.batch} dtype={args.dtype} fused_attn={args.fused_attn} "
          f"mask_ratio={args.mask_ratio} lr={args.lr}", flush=True)

    print("[build] instantiating ClayMAE (this downloads the DINOv2 teacher)...", flush=True)
    model = ClayMAE(
        mask_ratio=args.mask_ratio, patch_size=args.patch, norm_pix_loss=False,
        shuffle=True, metadata=S2_META, teacher=args.teacher,
        dolls=[16, 32, 64, 128, 256, 768, 1024], doll_weights=[1] * 7,
        fused_attn=(args.fused_attn == "true"), teacher_impl=args.teacher_impl, **cfg,
    )
    model = model.to(device=device, dtype=dtype)
    model.train()
    model.teacher.eval()  # keep teacher frozen/eval
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"[model] trainable={trainable/1e6:.1f}M  frozen(teacher)={frozen/1e6:.1f}M", flush=True)

    if args.compile:
        print("[compile] wrapping model in torch.compile(backend='neuron')", flush=True)
        model = torch.compile(model, backend="neuron")

    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr,
        weight_decay=0.05, betas=(0.9, 0.95),
    )

    C = len(S2_META["sentinel-2-l2a"]["band_order"])  # 10
    datacube = {
        "pixels": torch.randn(args.batch, C, args.img, args.img, device=device, dtype=dtype),
        "time": torch.randn(args.batch, 4, device=device, dtype=dtype),
        "latlon": torch.randn(args.batch, 4, device=device, dtype=dtype),
        "platform": ["sentinel-2-l2a"] * args.batch,
    }

    probe = model.encoder.transformer.layers[0][0].to_qkv.weight
    probe_before = probe.detach().to("cpu").float().clone()

    losses = []
    for step in range(args.steps):
        t0 = time.time()
        opt.zero_grad()
        loss, recon, repr_ = model(datacube)
        loss.backward()
        opt.step()
        lv = float(loss.detach().to("cpu"))
        rv = float(recon.detach().to("cpu"))
        pv = float(repr_.detach().to("cpu"))
        dt = time.time() - t0
        losses.append(lv)
        tag = "  (incl. compile+teacher dl)" if step == 0 else ""
        print(f"[step {step}] loss={lv:.5f} (recon={rv:.5f} repr={pv:.5f})  {dt:.2f}s{tag}",
              flush=True)

    delta = float((probe.detach().to("cpu").float() - probe_before).abs().max())
    print(f"[check] max |Δweight| encoder.layer0.to_qkv = {delta:.3e} "
          f"({'UPDATED' if delta > 0 else 'NO CHANGE'})", flush=True)
    print("[RESULT] loss:", " -> ".join(f"{l:.4f}" for l in losses), flush=True)
    if delta > 0 and all(l == l for l in losses):
        print(f"[RESULT] PASS: WHOLE ClayMAE({args.size}) + DINOv2 teacher trained on "
              f"'{args.device}' in EAGER mode — recon+repr loss, fwd+bwd+AdamW, "
              f"weights updated.", flush=True)
    else:
        print(f"[RESULT] FAIL: delta={delta} losses={losses}", flush=True)


if __name__ == "__main__":
    main()

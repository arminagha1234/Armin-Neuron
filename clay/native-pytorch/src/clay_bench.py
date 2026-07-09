"""
Clay throughput sweep on Trainium (single NeuronCore).

Primary metrics (exact, measured): step time, samples/sec, patch-tokens/sec.
Secondary (rough, caveated): MFU vs trn1 per-core BF16 peak (95 TFLOPS).

Sweeps batch sizes; builds the model once and re-benchmarks per batch.
OOM at a given batch is caught and reported, sweep continues.
"""

import argparse
import time
import torch

from claymodel.model import ClayMAE, clay_mae_config

S2_WAVES = [0.493, 0.56, 0.665, 0.704, 0.74, 0.783, 0.842, 0.865, 1.61, 2.19]
S2_BANDS = 10


class DotDict(dict):
    def __getattr__(self, k): return self[k]
    def __getitem__(self, k):
        v = dict.__getitem__(self, k)
        return DotDict(v) if isinstance(v, dict) else v


def s2_meta():
    return DotDict({"sentinel-2-l2a": {
        "band_order": list(range(S2_BANDS)), "rgb_indices": [2, 1, 0], "gsd": 10,
        "bands": {"wavelength": {i: w for i, w in enumerate(S2_WAVES)}}}})


def count(params):
    return sum(p.numel() for p in params)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="base")
    ap.add_argument("--teacher", default="facebook/dinov2-large")
    ap.add_argument("--patch", type=int, default=8)
    ap.add_argument("--img", type=int, default=128)
    ap.add_argument("--mask-ratio", type=float, default=0.75)
    ap.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16"])
    ap.add_argument("--batches", default="1,2,4,8,16,32")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--peak-tflops", type=float, default=95.0, help="trn1 per-core BF16 peak")
    args = ap.parse_args()

    dtype = torch.float32 if args.dtype == "float32" else torch.bfloat16
    device = torch.device("neuron")
    cfg = clay_mae_config(args.size)
    L = (args.img // args.patch) ** 2
    tok_enc = int((1 - args.mask_ratio) * L) + 1
    tok_dec = L + 1
    tok_teacher = (518 // 14) ** 2 + 1

    model = ClayMAE(
        mask_ratio=args.mask_ratio, patch_size=args.patch, norm_pix_loss=False, shuffle=True,
        metadata=s2_meta(), teacher=args.teacher,
        dolls=[16, 32, 64, 128, 256, 768, 1024], doll_weights=[1] * 7,
        fused_attn=False, teacher_impl="transformers", **cfg,
    ).to(device=device, dtype=dtype)
    model.train(); model.teacher.eval()
    if args.compile:
        model = torch.compile(model, backend="neuron")

    P_enc = count(model.encoder.transformer.parameters()) if not args.compile else count([p for n,p in model.named_parameters() if "encoder.transformer" in n])
    P_dec = count(model.decoder.transformer.parameters()) if not args.compile else count([p for n,p in model.named_parameters() if "decoder.transformer" in n])
    P_teacher = count(model.teacher.parameters()) if not args.compile else count([p for n,p in model.named_parameters() if "teacher" in n])

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=5e-6)

    print(f"[bench] size={args.size} img={args.img} patch={args.patch} L={L} dtype={args.dtype} "
          f"compile={args.compile} attn=manual peak={args.peak_tflops}TF/core", flush=True)
    print(f"[bench] tok_enc={tok_enc} tok_dec={tok_dec} tok_teacher={tok_teacher} "
          f"P_enc_tf={P_enc/1e6:.0f}M P_dec_tf={P_dec/1e6:.0f}M P_teacher={P_teacher/1e6:.0f}M", flush=True)
    hdr = f"{'batch':>5} {'step_s':>8} {'samp/s':>8} {'tok/s':>10} {'MFU%':>6}"
    print(hdr, flush=True); print("-" * len(hdr), flush=True)

    for b in [int(x) for x in args.batches.split(",")]:
        try:
            dc = {
                "pixels": torch.randn(b, S2_BANDS, args.img, args.img, device=device, dtype=dtype),
                "time": torch.randn(b, 4, device=device, dtype=dtype),
                "latlon": torch.randn(b, 4, device=device, dtype=dtype),
                "platform": ["sentinel-2-l2a"] * b,
            }
            for _ in range(args.warmup):
                opt.zero_grad(); loss, _, _ = model(dc); loss.backward(); opt.step()
                _ = float(loss.detach().to("cpu"))
            t0 = time.time()
            for _ in range(args.iters):
                opt.zero_grad(); loss, _, _ = model(dc); loss.backward(); opt.step()
            _ = float(loss.detach().to("cpu"))  # sync
            dt = (time.time() - t0) / args.iters
            samp = b / dt
            toks = samp * L
            # rough training FLOPs/step: 6*N*tok for trainable (fwd+bwd), 2*N*tok teacher fwd
            flops = (6 * (P_enc * tok_enc + P_dec * tok_dec) + 2 * P_teacher * tok_teacher) * b
            mfu = 100.0 * (flops / dt) / (args.peak_tflops * 1e12)
            print(f"{b:>5} {dt:>8.3f} {samp:>8.2f} {toks:>10.0f} {mfu:>6.1f}", flush=True)
        except Exception as e:
            msg = str(e)[:80].replace("\n", " ")
            print(f"{b:>5}   OOM/ERR: {msg}", flush=True)
            break


if __name__ == "__main__":
    main()

"""
REAL end-to-end Wan 2.2 TI2V-5B on Trn2 — a TRUE glued T5 -> DiT -> VAE run that
produces an actual .mp4, times the real wall-clock e2e (not a sum-of-stages
estimate), and runs the "no visible difference" correctness gate.

Architecture (TP=1, single process, most robust):
  * T5 (UMT5) text-encode on CPU  -> avoids the T5(LNC1)/DiT(LNC2) flag conflict
    and never touches HBM. Cheap (~0.4-1.3s). We delete it before the neuron work.
  * DiT denoise on neuron, torch.compile(backend="neuron", dynamic=False).
  * VAE decode on neuron in bf16 (what we ship).
  * Everything glued through diffusers WanPipeline.__call__ so the measured
    wall-clock is a genuine end-to-end number.

We wrap pipe.vae.decode to (a) time it and (b) capture the exact un-normalized
latent the pipeline feeds the VAE plus the neuron bf16 decode output. That
captured latent drives the PART-2 correctness gate: decode the SAME latent on
CPU in fp32 (the reference) and compare per-frame PSNR / SSIM / max-abs-diff.

Run (venv activated):
  NEURON_CC_FLAGS="--model-type=transformer -O2 --auto-cast=none" \
      python -u e2e_video.py
Optional golden compare:
  ... python -u e2e_video.py --golden /path/to/h100_golden.mp4
"""
from __future__ import annotations
import argparse
import gc
import math
import os
import time

import numpy as np
import torch

# --- surgical fix: UniPCMultistepScheduler's corrector (order>=2, kicks in at
# step 2) builds R as float32 and b as float64, then calls torch.linalg.solve,
# which errors "A and B must have the same dtype". This is a host-side scheduler
# numerics detail, unrelated to Trainium. Promote to a common dtype only when
# they differ so the UniPC scheduler runs unchanged otherwise. ---
_ORIG_SOLVE = torch.linalg.solve


def _solve_dtype_safe(A, B, *a, **k):
    if torch.is_tensor(A) and torch.is_tensor(B) and A.dtype != B.dtype:
        c = torch.promote_types(A.dtype, B.dtype)
        A, B = A.to(c), B.to(c)
    return _ORIG_SOLVE(A, B, *a, **k)


torch.linalg.solve = _solve_dtype_safe

MODEL = "/home/ubuntu/wan22"
DEV = torch.device("neuron")
OUT_MP4 = "/home/ubuntu/kernel_research/wan_out.mp4"
PROMPT = "a cat playing piano, cinematic, high detail"
NEG = ""
H, W, NF, STEPS, GUID = 480, 832, 49, 50, 5.0

# per-stage timers / captures
T: dict[str, float] = {}
CAP: dict[str, object] = {}


def _sync(o):
    o = getattr(o, "sample", o)
    o = getattr(o, "last_hidden_state", o)
    if isinstance(o, (tuple, list)):
        o = o[0]
    if torch.is_tensor(o):
        float(o.detach().float().flatten()[:1].cpu())
    return o


# ----------------------------------------------------------------------------
# metrics (numpy-only, no scipy/skimage hard-dep)
# ----------------------------------------------------------------------------
def _gaussian_kernel1d(sigma=1.5, radius=5):
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    k = np.exp(-(x ** 2) / (2 * sigma ** 2))
    return k / k.sum()


def _blur(img, k):
    # separable reflect-padded 1D gaussian along H then W. img: (H,W) float64
    r = len(k) // 2
    p = np.pad(img, ((r, r), (0, 0)), mode="reflect")
    out = np.zeros_like(img)
    for i, w in enumerate(k):
        out += w * p[i:i + img.shape[0], :]
    p = np.pad(out, ((0, 0), (r, r)), mode="reflect")
    out2 = np.zeros_like(img)
    for i, w in enumerate(k):
        out2 += w * p[:, i:i + img.shape[1]]
    return out2


def ssim_frame(a, b, data_range=255.0):
    """Gaussian-windowed SSIM on a single grayscale frame (Wang et al. 2004)."""
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    k = _gaussian_kernel1d()
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    mu_a, mu_b = _blur(a, k), _blur(b, k)
    mu_a2, mu_b2, mu_ab = mu_a * mu_a, mu_b * mu_b, mu_a * mu_b
    sa = _blur(a * a, k) - mu_a2
    sb = _blur(b * b, k) - mu_b2
    sab = _blur(a * b, k) - mu_ab
    num = (2 * mu_ab + c1) * (2 * sab + c2)
    den = (mu_a2 + mu_b2 + c1) * (sa + sb + c2)
    return float(np.mean(num / den))


try:
    from skimage.metrics import structural_similarity as _sk_ssim

    def ssim_gray(a, b):
        return float(_sk_ssim(a, b, data_range=255.0))
    SSIM_IMPL = "skimage"
except Exception:  # noqa: BLE001
    def ssim_gray(a, b):
        return ssim_frame(a, b)
    SSIM_IMPL = "numpy-manual"


def psnr(a, b, data_range=255.0):
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    mse = np.mean((a - b) ** 2)
    if mse <= 1e-12:
        return 99.0
    return float(20 * math.log10(data_range / math.sqrt(mse)))


def to_uint8_frames(video_bcfhw):
    """(B,C,F,H,W) in ~[-1,1] -> uint8 (F,H,W,C)."""
    v = video_bcfhw
    if torch.is_tensor(v):
        v = v.detach().float().cpu().numpy()
    v = v[0]                              # (C,F,H,W)
    v = np.transpose(v, (1, 2, 3, 0))     # (F,H,W,C)
    v = (v / 2.0 + 0.5).clip(0, 1)
    return (v * 255.0).round().astype(np.uint8)


def gray(frames_uint8):
    # (F,H,W,C) -> (F,H,W) luma
    return frames_uint8.astype(np.float64) @ np.array([0.299, 0.587, 0.114])


# ----------------------------------------------------------------------------
# PART 1 — glued pipeline -> real e2e + mp4
# ----------------------------------------------------------------------------
def build_pipeline():
    from diffusers import WanPipeline
    print("loading WanPipeline (bf16)...", flush=True)
    t0 = time.time()
    pipe = WanPipeline.from_pretrained(MODEL, torch_dtype=torch.bfloat16)
    print(f"  loaded in {time.time()-t0:.1f}s", flush=True)

    # --- T5 encode on CPU (robust: no LNC conflict, no HBM) ---
    # fp32 on CPU: bf16 matmuls are emulated (slow) on x86; compute fp32 then
    # cast the final embeds to bf16 for the DiT. Faster AND representative.
    print("encoding prompt with T5 on CPU (fp32 compute)...", flush=True)
    pipe.text_encoder.to(device="cpu", dtype=torch.float32)
    t0 = time.time()
    with torch.no_grad():
        pe, npe = pipe.encode_prompt(
            prompt=PROMPT, negative_prompt=NEG,
            do_classifier_free_guidance=True, num_videos_per_prompt=1,
            max_sequence_length=512, device=torch.device("cpu"),
            dtype=torch.bfloat16)
    T["t5"] = time.time() - t0
    print(f"  T5 encode (CPU): {T['t5']:.2f}s  embeds={tuple(pe.shape)}", flush=True)

    # free T5 entirely, then place DiT+VAE on neuron
    pipe.text_encoder = None
    gc.collect()
    pipe.transformer.to(DEV)
    pipe.vae.to(DEV)

    # force execution device to neuron (text_encoder is gone)
    type(pipe)._execution_device = property(lambda self: DEV)

    pe = pe.to(DEV)
    npe = npe.to(DEV)

    # compile DiT (our win) + instrument the denoise loop
    pipe.transformer.forward = torch.compile(
        pipe.transformer.forward, backend="neuron", dynamic=False)
    _tf = pipe.transformer.forward

    def _to_dev(x):
        return x.to(DEV) if torch.is_tensor(x) else x

    def tf(*a, **k):
        # scheduler is forced to CPU (below), so `timestep` arrives on CPU;
        # move all tensor inputs to neuron for the DiT forward.
        a = tuple(_to_dev(x) for x in a)
        k = {kk: _to_dev(vv) for kk, vv in k.items()}
        t = time.time(); r = _tf(*a, **k); _sync(r)
        T["dit"] = T.get("dit", 0.0) + time.time() - t
        T["dit_n"] = T.get("dit_n", 0) + 1
        return r
    pipe.transformer.forward = tf

    # Force the scheduler entirely onto CPU: set_timesteps(device="cpu") so all
    # internal sigmas/timesteps live on host, keeping the corrector's linalg on
    # CPU. (The DiT wrapper above re-hosts `timestep` to neuron for the forward.)
    _set_ts = pipe.scheduler.set_timesteps

    def set_ts_cpu(num_inference_steps=None, device=None, **kw):
        return _set_ts(num_inference_steps=num_inference_steps,
                       device=torch.device("cpu"), **kw)
    pipe.scheduler.set_timesteps = set_ts_cpu

    # Run the scheduler step on CPU. UniPC's corrector does linalg.solve / einsum
    # / reductions; if the latents live on neuron those host-math ops dispatch to
    # the device (unsupported / tiny-op compile failures like aten::any). Keep the
    # cheap scheduler math on CPU and only move latents to neuron for the DiT.
    _sched_step = pipe.scheduler.step

    def sched_step_cpu(model_output, timestep, sample, *a, **k):
        mo = model_output.detach().to("cpu")
        smp = sample.detach().to("cpu")
        ts = timestep.to("cpu") if torch.is_tensor(timestep) else timestep
        out = _sched_step(mo, ts, smp, *a, **k)
        if isinstance(out, tuple):
            return (out[0].to(DEV),) + tuple(out[1:])
        if hasattr(out, "prev_sample"):
            out.prev_sample = out.prev_sample.to(DEV)
        return out
    pipe.scheduler.step = sched_step_cpu

    # instrument + capture VAE decode
    _vd = pipe.vae.decode

    def vd(latents, *a, **k):
        CAP["latent"] = latents.detach().float().cpu().clone()
        t = time.time(); r = _vd(latents, *a, **k)
        o = _sync(r)
        T["vae"] = T.get("vae", 0.0) + time.time() - t
        CAP["neuron_decode"] = o.detach().float().cpu().clone()
        return r
    pipe.vae.decode = vd

    return pipe, pe, npe


def generate(pipe, pe, npe, tag):
    kw = dict(prompt=None, negative_prompt=None,
              prompt_embeds=pe, negative_prompt_embeds=npe,
              height=H, width=W, num_frames=NF, num_inference_steps=STEPS,
              guidance_scale=GUID, output_type="np")
    # re-attach t5 time for the timed summary (encode happened once, up front)
    t0 = time.time()
    result = pipe(**kw)
    e2e = time.time() - t0
    return result, e2e


def part1(pipe, pe, npe):
    t5 = T.get("t5", 0.0)               # measured once, up front, before pipe()
    print("\n=== WARMUP generation (includes DiT+VAE compile) ===", flush=True)
    t0 = time.time()
    _res, _e = generate(pipe, pe, npe, "warmup")
    print(f"  warmup e2e (w/ compile): {time.time()-t0:.1f}s  "
          f"stages={ {k: round(v,2) for k,v in T.items()} }", flush=True)

    print("\n=== TIMED generation (steady state) ===", flush=True)
    T.clear()
    res, e2e_pipe = generate(pipe, pe, npe, "timed")

    frames = res.frames[0]                 # (F,H,W,C) float [0,1]
    t_export0 = time.time()
    from diffusers.utils import export_to_video
    export_to_video(frames, OUT_MP4, fps=16)
    T["export"] = time.time() - t_export0

    dit = T.get("dit", 0.0)
    vae = T.get("vae", 0.0)
    exp = T.get("export", 0.0)
    total = t5 + e2e_pipe + exp          # t5 was measured before the pipe() call
    print("\n===================== REAL MEASURED E2E =====================", flush=True)
    print(f"  T5 text-encode (CPU):   {t5:.2f}s", flush=True)
    print(f"  DiT denoise (neuron):   {dit:.1f}s  "
          f"({T.get('dit_n',0)} forwards, "
          f"{dit/max(1,T.get('dit_n',1))*1000:.0f} ms/fwd)", flush=True)
    print(f"  VAE decode (neuron bf16): {vae:.1f}s", flush=True)
    print(f"  export mp4:             {exp:.2f}s", flush=True)
    other = e2e_pipe - dit - vae
    print(f"  sched/host overhead:    {other:.1f}s", flush=True)
    print(f"  --------------------------------------------------", flush=True)
    print(f"  GLUED pipe() wall-clock: {e2e_pipe:.1f}s", flush=True)
    print(f"  FULL e2e (T5+pipe+export): {total:.1f}s", flush=True)
    print(f"  (vs sum-of-stages estimate ~109s TP1 / ~45s TP4)", flush=True)
    print("=============================================================", flush=True)
    return frames


# ----------------------------------------------------------------------------
# PART 2 — correctness gate
# ----------------------------------------------------------------------------
def cpu_reference_decode():
    """Decode the captured latent on CPU in fp32 (the reference)."""
    from diffusers import AutoencoderKLWan
    print("\n[gate] loading CPU fp32 VAE for reference decode...", flush=True)
    m = AutoencoderKLWan.from_pretrained(MODEL, subfolder="vae",
                                         torch_dtype=torch.float32).to("cpu").eval()
    lat = CAP["latent"].to(torch.float32)
    t0 = time.time()
    with torch.no_grad():
        out = m.decode(lat, return_dict=False)[0]
    print(f"[gate] CPU fp32 decode: {time.time()-t0:.1f}s", flush=True)
    del m; gc.collect()
    return out.detach().float().cpu()


def part2(neuron_frames_np, golden_path=None):
    print("\n========== PART 2: CORRECTNESS GATE ==========", flush=True)
    verdict_ok = True

    # ----- decode-level Neuron(bf16) vs CPU(fp32) parity -----
    neuron_dec = CAP.get("neuron_decode")
    if neuron_dec is None:
        print("[gate] ERROR: no neuron decode captured", flush=True)
        return
    cpu_dec = cpu_reference_decode()

    nf = to_uint8_frames(neuron_dec)   # (F,H,W,C) uint8
    cf = to_uint8_frames(cpu_dec)
    gn, gc_ = gray(nf), gray(cf)
    F = nf.shape[0]
    psnrs = [psnr(nf[i], cf[i]) for i in range(F)]
    ssims = [ssim_gray(gn[i], gc_[i]) for i in range(F)]
    maxabs = [int(np.abs(nf[i].astype(int) - cf[i].astype(int)).max()) for i in range(F)]
    p_min, p_mean = min(psnrs), float(np.mean(psnrs))
    s_min, s_mean = min(ssims), float(np.mean(ssims))
    ma_max = max(maxabs)
    print(f"[decode parity] Neuron-bf16 vs CPU-fp32  (SSIM impl: {SSIM_IMPL})", flush=True)
    print(f"    PSNR  min={p_min:.1f} dB  mean={p_mean:.1f} dB   (gate >40)", flush=True)
    print(f"    SSIM  min={s_min:.4f}  mean={s_mean:.4f}       (gate >0.98)", flush=True)
    print(f"    max-abs pixel diff (0-255): {ma_max}", flush=True)
    parity_ok = (p_min > 40.0) and (s_min > 0.98)
    verdict_ok &= parity_ok
    print(f"    -> decode parity: {'PASS' if parity_ok else 'REVIEW'}", flush=True)

    # ----- frame sanity (porting checklist) -----
    print("[frame sanity]", flush=True)
    fint = (neuron_frames_np * 255.0)
    any_nan = bool(np.isnan(neuron_frames_np).any()) or bool(
        torch.isnan(neuron_dec).any())
    stds = [float(fint[i].std()) for i in range(neuron_frames_np.shape[0])]
    means = [float(fint[i].mean()) for i in range(neuron_frames_np.shape[0])]
    n_frames = neuron_frames_np.shape[0]
    std_min = min(stds)
    flat = [i for i, s in enumerate(stds) if s <= 60.0]
    print(f"    frames present: {n_frames} (expect {NF})", flush=True)
    print(f"    NaN anywhere: {any_nan}", flush=True)
    print(f"    per-frame std (0-255): min={std_min:.1f} "
          f"mean={float(np.mean(stds)):.1f} max={max(stds):.1f}  (gate >60)", flush=True)
    print(f"    per-frame mean (0-255): min={min(means):.1f} "
          f"max={max(means):.1f}", flush=True)
    sanity_ok = (n_frames == NF) and (not any_nan) and (std_min > 60.0)
    if flat:
        print(f"    NOTE: {len(flat)} frame(s) with std<=60 (idx {flat[:6]}...)",
              flush=True)
    verdict_ok &= sanity_ok
    print(f"    -> frame sanity: {'PASS' if sanity_ok else 'REVIEW'}", flush=True)

    # ----- optional golden compare -----
    if golden_path and os.path.exists(golden_path):
        golden_compare(neuron_frames_np, golden_path)

    print("\n" + "=" * 46, flush=True)
    print(f"CORRECTNESS: {'PASS' if verdict_ok else 'REVIEW'}", flush=True)
    print("=" * 46, flush=True)


def golden_compare(neuron_frames_np, golden_path):
    print(f"\n[golden] comparing vs {golden_path}", flush=True)
    try:
        import imageio.v3 as iio
        g = iio.imread(golden_path, index=None)  # (F,H,W,C) uint8
    except Exception as e:  # noqa: BLE001
        print(f"[golden] could not read golden: {e}", flush=True)
        return
    n = (neuron_frames_np * 255.0).round().astype(np.uint8)
    F = min(len(g), len(n))
    lpips_fn = None
    try:
        import lpips  # noqa: F401
        import torch as _t
        lpips_fn = lpips.LPIPS(net="alex")
        print("[golden] LPIPS available", flush=True)
    except Exception:  # noqa: BLE001
        print("[golden] LPIPS unavailable (skipped)", flush=True)
    ps, ss, lp = [], [], []
    for i in range(F):
        a, b = n[i], g[i]
        if a.shape != b.shape:
            print(f"[golden] shape mismatch {a.shape} vs {b.shape}; skipping", flush=True)
            return
        ps.append(psnr(a, b))
        ss.append(ssim_gray(gray(a[None])[0], gray(b[None])[0]))
        if lpips_fn is not None:
            import torch as _t
            ta = _t.tensor(a).permute(2, 0, 1)[None].float() / 127.5 - 1
            tb = _t.tensor(b).permute(2, 0, 1)[None].float() / 127.5 - 1
            lp.append(float(lpips_fn(ta, tb).item()))
    print(f"[golden] PSNR mean={np.mean(ps):.1f} dB  SSIM mean={np.mean(ss):.4f}"
          + (f"  LPIPS mean={np.mean(lp):.4f}" if lp else ""), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden", default=None)
    args = ap.parse_args()

    pipe, pe, npe = build_pipeline()
    frames = part1(pipe, pe, npe)
    print(f"\nvideo written: {OUT_MP4}", flush=True)
    part2(frames, golden_path=args.golden)


if __name__ == "__main__":
    main()
    os._exit(0)

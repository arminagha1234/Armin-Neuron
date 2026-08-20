"""
TeaCache (timestep-embedding-aware residual caching) for Wan 2.2 TI2V-5B on Trn2.

Implements + MEASURES TeaCache (arXiv 2411.14324) on the WanTransformer3DModel
denoise loop to cut the number of DiT backbone forwards. TI2V-5B runs
num_inference_steps steps x 2 (CFG cond+uncond) backbone forwards; TeaCache skips
forwards whose backbone output residual barely changes.

Mechanism (applied to WanTransformer3DModel.forward):
  * cheap proxy = the AdaLN-modulated input at block 0:
        modulated = norm1(hidden_in) * (1 + scale_msa) + shift_msa
    where (scale_msa, shift_msa) come from block0.scale_shift_table + timestep_proj.
  * per denoise step, per CFG pass (separate cond/uncond caches), accumulate the
    rescaled relative-L1 distance of `modulated` vs the previous step. While the
    accumulated distance stays < rel_l1_thresh, SKIP the 30-block backbone and
    reuse the cached residual (prev  output-minus-input) added to the current
    input. When it crosses the threshold (or first/last step), run the real
    backbone, refresh the cached residual, and reset the accumulator.
  * rescaling polynomial: no Wan2.2-TI2V-5B coefficients are published, so we use
    IDENTITY (direct relative-L1). Thresholds are therefore calibrated for the raw
    L1 signal; we report that explicitly.

Design for Trn2 / torch.compile:
  * Only the 30-block stack is torch.compile'd (backend="neuron", the heavy ~95%).
  * The TeaCache controller (proxy, skip decision, residual add) stays in EAGER
    python OUTSIDE the compiled region, so data-dependent control flow never
    enters the graph (no recompiles / graph breaks).

Runs the REAL WanPipeline denoise (output_type="latent") for baseline (no cache)
and thresh in {0.1, 0.15, 0.2}, sharing one loaded pipeline + one compiled block
stack. Reports executed-vs-skipped forwards, skip%, wall time, projected DiT/e2e,
and final-latent cosine vs baseline. With --decode it also VAE-decodes and reports
per-frame PSNR.

  source /home/ubuntu/workspace/native_venv/bin/activate
  python -u teacache_wan.py [--decode] [--steps 50]
"""
from __future__ import annotations
import argparse, math, os, time
from typing import Any
import numpy as np
import torch

MODEL = "/home/ubuntu/wan22"
OUTDIR = "/home/ubuntu/kernel_research"
DEV = torch.device("neuron")
SEED = 0  # every pipe() call MUST start from the SAME initial latent, else baseline
          # vs TeaCache cosine just compares two different random videos


def fresh_gen():
    """A fresh CPU generator at the fixed seed -> identical initial noise every run.
    (Without this, each pipe() call draws new random latents and baseline-vs-TeaCache
    cosine is meaningless — it just compares two different videos.)"""
    return torch.Generator(device="cpu").manual_seed(SEED)

# Best-known REAL DiT baselines (from prior measurement on this box):
#   TP1  compiled : 943 ms/forward -> 100 fwd = 94.3 s
#   TP4/LNC2 CFG-batched: 50 x 614 ms = 30.7 s  (the config used for e2e projection)
DIT_BASELINE_S = 30.7          # TP4/LNC2 CFG-batched full-denoise DiT time
T5_S = 0.4                     # measured T5 text-encode
VAE_S = 14.1                   # measured bf16 VAE decode


class TeaCacheController:
    """Holds TeaCache state; provides a drop-in replacement for
    WanTransformer3DModel.forward.

    Correct TeaCache caches the FEATURE-space block-stack residual
    (hidden_after_blocks - hidden_after_patch_embed, in the inner_dim=3072 space) --
    NOT the latent-space (noise_pred - latent) residual, which is not temporally
    smooth. To keep that correct residual AND avoid the pathologically slow
    torch_neuronx EAGER device ops, everything runs inside torch.compile regions:

      * proxy_fn(hidden, ts, enc) -> block-0 AdaLN-modulated input   [every step]
      * run_fn (hidden, ts, enc)  -> (output, feature_residual)      [on RUN]
      * skip_fn(hidden, ts, enc, residual) -> output                 [on SKIP:
            same forward but the 30-block stack is replaced by  post_patch+residual]

    The only work outside compiled code is python scalar control flow + one .item()
    on the proxy relative-L1 metric.
    """

    def __init__(self, transformer, num_steps: int, num_cfg: int = 2, poly=None):
        self.t = transformer
        self.num_steps = num_steps
        self.num_cfg = num_cfg          # cond + uncond
        self.poly = poly                # None => identity rescale (direct L1)
        self.thresh = None              # set per-run; None => TeaCache OFF (baseline)
        self.proxy_fn = torch.compile(self._proxy, backend="neuron", dynamic=False)
        self.run_fn = torch.compile(self._run, backend="neuron", dynamic=False)
        self.skip_fn = torch.compile(self._skip, backend="neuron", dynamic=False)
        self.reset()
        self.reset_calib()

    def reset(self):
        self.state = {"cond": self._blank(), "uncond": self._blank()}
        self.call_count = 0
        self.executed = 0
        self.skipped = 0
        self.exec_time = 0.0
        self.skip_time = 0.0

    @staticmethod
    def _blank():
        return {"prev_mod": None, "accum": 0.0, "residual": None, "calls": 0, "prev_out": None}

    def reset_calib(self):
        self.calib = {"cond": [], "uncond": []}   # (x=input-proxy relL1, y=output relL1)

    # ---- shared front/back halves of the diffusers WanTransformer3DModel.forward ----
    def _front(self, hidden_states, timestep, encoder_hidden_states):
        """patch_embed + condition_embedder; returns everything the block stack and
        the tail need, plus the block-0 modulated proxy."""
        t = self.t
        bsz, _, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = t.config.patch_size
        shapes = (bsz, num_frames // p_t, height // p_h, width // p_w, p_t, p_h, p_w)
        rotary_emb = t.rope(hidden_states)
        hs = t.patch_embedding(hidden_states)
        hs = hs.flatten(2).transpose(1, 2)                       # post-patch feature hidden
        if timestep.ndim == 2:
            ts_seq_len = timestep.shape[1]
            timestep = timestep.flatten()
        else:
            ts_seq_len = None
        temb, timestep_proj, enc, enc_img = t.condition_embedder(
            timestep, encoder_hidden_states, None, timestep_seq_len=ts_seq_len)
        if ts_seq_len is not None:
            timestep_proj = timestep_proj.unflatten(2, (6, -1))
        else:
            timestep_proj = timestep_proj.unflatten(1, (6, -1))
        blk0 = t.blocks[0]
        if timestep_proj.ndim == 4:
            parts = (blk0.scale_shift_table.unsqueeze(0) + timestep_proj.float()).chunk(6, dim=2)
            shift_msa, scale_msa = parts[0].squeeze(2), parts[1].squeeze(2)
        else:
            parts = (blk0.scale_shift_table + timestep_proj.float()).chunk(6, dim=1)
            shift_msa, scale_msa = parts[0], parts[1]
        modulated = blk0.norm1(hs.float()) * (1 + scale_msa) + shift_msa
        return hs, enc, temb, timestep_proj, rotary_emb, modulated, shapes

    def _tail(self, hidden, temb, shapes):
        t = self.t
        bsz, pnf, ph, pw, p_t, p_h, p_w = shapes
        if temb.ndim == 3:
            shift, scale = (t.scale_shift_table.unsqueeze(0).to(temb.device) + temb.unsqueeze(2)).chunk(2, dim=2)
            shift, scale = shift.squeeze(2), scale.squeeze(2)
        else:
            shift, scale = (t.scale_shift_table.to(temb.device) + temb.unsqueeze(1)).chunk(2, dim=1)
        shift = shift.to(hidden.device)
        scale = scale.to(hidden.device)
        hidden = (t.norm_out(hidden.float()) * (1 + scale) + shift).type_as(hidden)
        hidden = t.proj_out(hidden)
        hidden = hidden.reshape(bsz, pnf, ph, pw, p_t, p_h, p_w, -1)
        hidden = hidden.permute(0, 7, 1, 4, 2, 5, 3, 6)
        return hidden.flatten(6, 7).flatten(4, 5).flatten(2, 3)

    def _proxy(self, hidden_states, timestep, encoder_hidden_states):
        _, _, _, _, _, modulated, _ = self._front(hidden_states, timestep, encoder_hidden_states)
        return modulated

    def _run(self, hidden_states, timestep, encoder_hidden_states):
        hs, enc, temb, timestep_proj, rotary_emb, _, shapes = self._front(
            hidden_states, timestep, encoder_hidden_states)
        post_patch = hs
        for block in self.t.blocks:
            hs = block(hs, enc, timestep_proj, rotary_emb)
        residual = hs - post_patch                              # FEATURE-space residual
        out = self._tail(hs, temb, shapes)
        return out, residual

    def _skip(self, hidden_states, timestep, encoder_hidden_states, residual):
        hs, enc, temb, timestep_proj, rotary_emb, _, shapes = self._front(
            hidden_states, timestep, encoder_hidden_states)
        hs = hs + residual                                      # reuse cached block-stack delta
        return self._tail(hs, temb, shapes)

    def forward(self, hidden_states, timestep, encoder_hidden_states,
                encoder_hidden_states_image=None, return_dict=True, attention_kwargs=None):
        key = "cond" if (self.call_count % self.num_cfg) == 0 else "uncond"
        self.call_count += 1
        st = self.state[key]

        modulated = self.proxy_fn(hidden_states, timestep, encoder_hidden_states)
        step_idx = st["calls"]

        should_calc = True
        rel = -1.0
        if st["prev_mod"] is not None:
            rel = ((modulated - st["prev_mod"]).abs().mean()
                   / st["prev_mod"].abs().mean().clamp_min(1e-8)).item()

        force_skip = getattr(self, "force_skip_steps", None)
        if force_skip is not None:
            # deterministic skip pattern (diagnostic): skip listed step indices,
            # but never the first or last step, and only if a residual exists.
            is_edge = step_idx == 0 or step_idx == (self.num_steps - 1)
            should_calc = not (step_idx in force_skip and not is_edge and st["residual"] is not None)
        elif self.thresh is not None:
            is_first = step_idx == 0
            is_last = step_idx == (self.num_steps - 1)
            if is_first or is_last or st["prev_mod"] is None or st["residual"] is None:
                should_calc = True
                st["accum"] = 0.0
            else:
                rescaled = max(0.0, float(self.poly(rel))) if self.poly is not None else rel
                st["accum"] += rescaled
                if st["accum"] < self.thresh:
                    should_calc = False
                else:
                    should_calc = True
                    st["accum"] = 0.0
        if getattr(self, "log_rel", False):
            print(f"[relL1] {key} step={step_idx:02d} rel={rel:.5f} "
                  f"accum={st['accum']:.5f} {'RUN' if should_calc else 'skip'}", flush=True)
        # .clone() is MANDATORY: compiled-graph output buffers are reused/overwritten
        # by later compiled calls on neuron, so a plain .detach() reference gets
        # corrupted by an intervening forward (e.g. the uncond pass) before we reuse it.
        st["prev_mod"] = modulated.detach().clone()
        st["calls"] += 1

        tb = time.time()
        if should_calc:
            out, residual = self.run_fn(hidden_states, timestep, encoder_hidden_states)
            # CALIBRATION: record (input-proxy relL1, output relL1) pairs to fit the
            # rescaling polynomial (output-change as a function of the cheap proxy).
            if getattr(self, "calibrate", False) and rel >= 0 and st["prev_out"] is not None:
                y = ((out - st["prev_out"]).abs().mean()
                     / st["prev_out"].abs().mean().clamp_min(1e-8)).item()
                self.calib[key].append((rel, y))
            if getattr(self, "calibrate", False):
                st["prev_out"] = out.detach().clone()
            # CONTROL: inject a random perturbation of a target relative-L2 magnitude at
            # chosen steps (no caching) to measure the sampler's intrinsic sensitivity.
            psteps = getattr(self, "perturb_steps", None)
            if psteps is not None and step_idx in psteps:
                rel = getattr(self, "perturb_rel", 0.025)
                noise = torch.randn_like(out)
                out = out + rel * (out.norm() / (noise.norm() + 1e-8)) * noise
            if getattr(self, "verify", False) and st["residual"] is not None and step_idx not in (0,):
                skip_out = self.skip_fn(hidden_states, timestep, encoder_hidden_states, st["residual"])
                a = out.detach().float().flatten().double().cpu()
                b = skip_out.detach().float().flatten().double().cpu()
                c = float((a @ b) / (a.norm() * b.norm() + 1e-12))
                print(f"[verify] {key} step={step_idx:02d} per-fwd cos(run,skip)={c:.6f}", flush=True)
            st["residual"] = residual.detach().clone()
            _sync(out)
            self.executed += 1
            self.exec_time += time.time() - tb
        else:
            out = self.skip_fn(hidden_states, timestep, encoder_hidden_states, st["residual"])
            _sync(out)
            self.skipped += 1
            self.skip_time += time.time() - tb

        if not return_dict:
            return (out,)
        from diffusers.models.modeling_outputs import Transformer2DModelOutput
        return Transformer2DModelOutput(sample=out)


def _sync(o):
    if isinstance(o, (tuple, list)):
        o = o[0]
    o = getattr(o, "sample", o)
    if torch.is_tensor(o):
        float(o.detach().float().flatten()[:1].cpu())
    return o


def build_pipe(prompt, negative_prompt):
    """Load pipe, pre-encode the prompt with T5 on HOST (the eager 'neuron' device
    is a single ~24GB logical core under LNC2 — T5 11GB + transformer 10GB + NEFF
    overflow it), then move ONLY transformer+VAE to neuron. Returns pipe + the
    pre-computed prompt embeddings (already on the neuron device)."""
    from diffusers import WanPipeline
    print("loading WanPipeline (bf16)...", flush=True)
    t0 = time.time()
    pipe = WanPipeline.from_pretrained(MODEL, torch_dtype=torch.bfloat16)
    print(f"  loaded in {time.time()-t0:.1f}s", flush=True)

    print("  pre-encoding prompt with T5 on HOST (CPU)...", flush=True)
    te = time.time()
    with torch.no_grad():
        prompt_embeds, negative_prompt_embeds = pipe.encode_prompt(
            prompt=prompt, negative_prompt=negative_prompt,
            do_classifier_free_guidance=True, device=torch.device("cpu"),
            dtype=torch.bfloat16)
    print(f"    T5 encode {time.time()-te:.1f}s  prompt_embeds={tuple(prompt_embeds.shape)}", flush=True)

    # free T5 host memory; move backbone + VAE to device
    pipe.text_encoder = None
    pipe.transformer.to(DEV)
    pipe.vae.to(DEV)
    prompt_embeds = prompt_embeds.to(DEV)
    negative_prompt_embeds = negative_prompt_embeds.to(DEV)
    print("  placement: transformer+VAE on neuron, T5 encoded on host (freed)", flush=True)
    try:
        print(f"  pipe._execution_device={pipe._execution_device}", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"  (exec_device probe: {type(e).__name__})", flush=True)
    return pipe, prompt_embeds, negative_prompt_embeds


def run_once(pipe, ctrl, thresh, kw, tag):
    ctrl.reset()
    ctrl.thresh = thresh
    kw = dict(kw, generator=fresh_gen())
    t0 = time.time()
    out = pipe(**kw)
    wall = time.time() - t0
    lat = out.frames if hasattr(out, "frames") else out[0]
    lat = lat.detach().float().cpu()
    executed, skipped = ctrl.executed, ctrl.skipped
    total = executed + skipped
    skip_frac = skipped / max(1, total)
    print(f"\n[{tag}] thresh={thresh}  wall={wall:.1f}s  forwards: executed={executed} "
          f"skipped={skipped} (total={total})  skip%={100*skip_frac:.1f}", flush=True)
    print(f"    exec_time={ctrl.exec_time:.1f}s  skip_time={ctrl.skip_time:.2f}s  "
          f"(per-exec {ctrl.exec_time/max(1,executed)*1000:.0f}ms)", flush=True)
    return dict(tag=tag, thresh=thresh, wall=wall, executed=executed, skipped=skipped,
                total=total, skip_frac=skip_frac, latent=lat,
                exec_time=ctrl.exec_time, skip_time=ctrl.skip_time)


def cosine(a, b):
    a = a.flatten().double(); b = b.flatten().double()
    return float((a @ b) / (a.norm() * b.norm() + 1e-12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--threshes", type=str, default="0.1,0.15,0.2")
    ap.add_argument("--decode", action="store_true")
    ap.add_argument("--diag", action="store_true", help="diagnostic: isolate skip-math via forced skip patterns")
    ap.add_argument("--verify", action="store_true", help="measure per-forward cos(run,skip) reconstruction fidelity")
    ap.add_argument("--perturb", action="store_true", help="control: inject random per-step perturbation to gauge sampler chaos")
    ap.add_argument("--calibrate", action="store_true", help="fit the rescaling polynomial then sweep with it")
    ap.add_argument("--prompt", type=str, default="a cat playing the piano, cinematic")
    ap.add_argument("--guidance", type=float, default=5.0, help="CFG scale (1.0 disables CFG -> 1 fwd/step)")
    a = ap.parse_args()
    threshes = [float(x) for x in a.threshes.split(",")]
    num_cfg = 2 if a.guidance > 1.0 else 1

    pipe, prompt_embeds, negative_prompt_embeds = build_pipe(a.prompt, "")
    ctrl = TeaCacheController(pipe.transformer, num_steps=a.steps, num_cfg=num_cfg, poly=None)
    # install the TeaCache forward on the transformer instance
    pipe.transformer.forward = ctrl.forward

    kw = dict(prompt=None, prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_prompt_embeds,
              height=480, width=832, num_frames=49,
              num_inference_steps=a.steps, guidance_scale=a.guidance, output_type="latent")
    print(f"  guidance_scale={a.guidance}  num_cfg={num_cfg} (fwds/step)", flush=True)

    print("\n=== WARMUP (compiles the block stack once; TeaCache OFF) ===", flush=True)
    t0 = time.time()
    ctrl.reset(); ctrl.thresh = None
    try:
        _ = pipe(**dict(kw, generator=fresh_gen()))
    except Exception as e:  # noqa: BLE001
        import traceback; traceback.print_exc()
        print(f"WARMUP FAILED: {type(e).__name__}: {str(e)[:300]}", flush=True)
        return
    print(f"  warmup done in {time.time()-t0:.1f}s  (executed={ctrl.executed} forwards)", flush=True)

    if a.verify:
        print("\n=== VERIFY per-forward reconstruction fidelity cos(run,skip) ===", flush=True)
        ctrl.reset(); ctrl.thresh = None; ctrl.force_skip_steps = None; ctrl.verify = True
        _ = pipe(**dict(kw, generator=fresh_gen()))
        ctrl.verify = False
        return

    if a.calibrate:
        print("\n=== CALIBRATION: recording (input-proxy relL1, output relL1) ===", flush=True)
        ctrl.reset(); ctrl.reset_calib(); ctrl.thresh = None; ctrl.calibrate = True
        _ = pipe(**dict(kw, generator=fresh_gen()))
        ctrl.calibrate = False
        pairs = ctrl.calib["cond"] + ctrl.calib["uncond"]
        xs = np.array([p[0] for p in pairs]); ys = np.array([p[1] for p in pairs])
        coeffs = np.polyfit(xs, ys, 4)
        ctrl.poly = np.poly1d(coeffs)
        print(f"  fitted degree-4 polynomial on {len(pairs)} pairs:", flush=True)
        print(f"  coeffs (highest power first) = {list(coeffs)}", flush=True)
        print(f"  x(input relL1) range [{xs.min():.4f},{xs.max():.4f}]  "
              f"y(output relL1) range [{ys.min():.4f},{ys.max():.4f}]", flush=True)
        np.save(f"{OUTDIR}/teacache_wan_poly_coeffs.npy", coeffs)

    # baseline: TeaCache OFF (thresh=None)
    base = run_once(pipe, ctrl, None, kw, "baseline")
    torch.save(base["latent"], f"{OUTDIR}/teacache_latent_baseline.pt")

    if a.perturb:
        n = a.steps
        print("\n=== CONTROL: sampler sensitivity to a random per-step perturbation ===", flush=True)
        combos = [(0.025, {n // 2}, "1step@mid"),
                  (0.0003, {n // 2}, "1step@mid"),
                  (0.025, set(range(2, n - 1, 5)), "every5th")]
        for rel, steps_set, name in combos:
            ctrl.reset(); ctrl.thresh = None; ctrl.force_skip_steps = None
            ctrl.perturb_steps = steps_set; ctrl.perturb_rel = rel
            torch.manual_seed(1234)
            out = pipe(**dict(kw, generator=fresh_gen()))
            lat = (out.frames if hasattr(out, "frames") else out[0]).detach().float().cpu()
            cos = cosine(lat, base["latent"])
            print(f"  [perturb rel={rel} {name} ({len(steps_set)} steps)] cos_final={cos:.5f}", flush=True)
        ctrl.perturb_steps = None
        return

    if a.diag:
        n = a.steps
        mid = n // 2
        patterns = {
            "skip1@mid": {mid},
            "skip@25/50/75%": {n // 4, n // 2, 3 * n // 4},
            "skip_alt(even)": set(range(2, n - 1, 2)),   # ~50% interspersed
        }
        print("\n=== DIAGNOSTIC: forced-skip patterns (isolate skip math) ===", flush=True)
        # first, a log_rel pass at a tiny thresh to see the rel-L1 scale
        ctrl.reset(); ctrl.thresh = None; ctrl.force_skip_steps = None; ctrl.log_rel = True
        print("--- per-step rel-L1 (baseline trajectory, first 12 shown) ---", flush=True)
        ctrl._logcap = 0
        _ = pipe(**dict(kw, generator=fresh_gen()))
        ctrl.log_rel = False
        for name, patt in patterns.items():
            ctrl.reset(); ctrl.thresh = 0.0; ctrl.force_skip_steps = patt
            t0 = time.time(); out = pipe(**dict(kw, generator=fresh_gen())); wall = time.time() - t0
            lat = (out.frames if hasattr(out, "frames") else out[0]).detach().float().cpu()
            cos = cosine(lat, base["latent"])
            print(f"  [{name}] executed={ctrl.executed} skipped={ctrl.skipped} "
                  f"cos={cos:.5f} wall={wall:.1f}s", flush=True)
        ctrl.force_skip_steps = None
        return

    results = [base]
    for th in threshes:
        results.append(run_once(pipe, ctrl, th, kw, f"tc{th}"))

    base = results[0]
    print("\n" + "=" * 78, flush=True)
    print("TEACACHE RESULTS  (Wan 2.2 TI2V-5B, 480x832, 49f, %d steps, CFG x2)" % a.steps, flush=True)
    print("rescale polynomial: IDENTITY (no published Wan2.2-5B coeffs) => raw rel-L1", flush=True)
    print("=" * 78, flush=True)
    print(f"{'run':>10} {'thr':>5} {'exec':>5} {'skip':>5} {'skip%':>6} "
          f"{'proj_DiT':>9} {'proj_e2e':>9} {'cosVSbase':>10} {'wall':>7}", flush=True)
    for r in results:
        exec_frac = r["executed"] / max(1, r["total"])
        proj_dit = DIT_BASELINE_S * exec_frac
        proj_e2e = T5_S + proj_dit + VAE_S
        cos = 1.0 if r is base else cosine(r["latent"], base["latent"])
        thr = "-" if r["thresh"] is None else f"{r['thresh']}"
        print(f"{r['tag']:>10} {thr:>5} {r['executed']:>5} {r['skipped']:>5} "
              f"{100*r['skip_frac']:>5.1f}% {proj_dit:>8.1f}s {proj_e2e:>8.1f}s "
              f"{cos:>10.5f} {r['wall']:>6.1f}s", flush=True)
    print(f"\nH100 golden e2e = 33.0s.  Projections use DiT baseline {DIT_BASELINE_S}s "
          f"(TP4/LNC2 CFG-batched) + T5 {T5_S}s + VAE {VAE_S}s.", flush=True)

    # save latents for offline inspection / decode
    for r in results:
        torch.save(r["latent"], f"{OUTDIR}/teacache_latent_{r['tag']}.pt")

    if a.decode:
        print("\n=== VAE decode + per-frame PSNR (bf16) ===", flush=True)
        decode_and_psnr(pipe, results, base)


def decode_and_psnr(pipe, results, base):
    vae = pipe.vae
    zdim = vae.config.z_dim
    lat_mean = torch.tensor(vae.config.latents_mean).view(1, zdim, 1, 1, 1).to(DEV, torch.bfloat16)
    lat_std = 1.0 / torch.tensor(vae.config.latents_std).view(1, zdim, 1, 1, 1).to(DEV, torch.bfloat16)

    def decode(lat):
        x = lat.to(DEV, torch.bfloat16)
        x = x / lat_std + lat_mean
        with torch.no_grad():
            v = vae.decode(x, return_dict=False)[0]
        return v.detach().float().cpu()

    base_vid = None
    for r in results:
        t0 = time.time()
        vid = decode(r["latent"])
        dt = time.time() - t0
        if r is base:
            base_vid = vid
            print(f"  [{r['tag']}] decoded in {dt:.1f}s  shape={tuple(vid.shape)}", flush=True)
            continue
        # PSNR on a few frames (vid shape B,C,T,H,W)
        T = base_vid.shape[2]
        idxs = sorted(set([0, T // 2, T - 1]))
        psnrs = []
        for fi in idxs:
            a = base_vid[:, :, fi]; b = vid[:, :, fi]
            mse = torch.mean((a - b) ** 2).item()
            rng = (a.max() - a.min()).item() or 1.0
            psnr = 10 * math.log10((rng ** 2) / (mse + 1e-12))
            psnrs.append(psnr)
        overall_mse = torch.mean((base_vid - vid) ** 2).item()
        print(f"  [{r['tag']}] decoded in {dt:.1f}s  frame PSNR@{idxs}="
              f"{[round(p,1) for p in psnrs]}dB  meanMSE={overall_mse:.4e}", flush=True)


if __name__ == "__main__":
    main()
    os._exit(0)

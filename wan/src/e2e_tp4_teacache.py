"""
CAPSTONE harness — REAL glued end-to-end Wan 2.2 TI2V-5B on Trn2 at TP=4 with
calibrated TeaCache wired in.  Produces an actual .mp4 and a real measured
wall-clock e2e (T5 -> DiT(TP=4) -> VAE -> mp4), plus the correctness gate.

This file GLUES three validated pieces (read their headers for the why):
  1. native_wan_tp.py   — the TP=4 DTensor plan (Colwise q/k/v + ffn.net.0.proj,
     Rowwise to_out.0 + ffn.net.2, heads->heads/world, the across-heads
     _AdaptiveQKNorm all-reduce). Validated 331 ms/fwd, cos 0.9991 vs TP=1.
  2. e2e_video.py       — the glued WanPipeline harness + 3 host-side fixes:
       (i)  UniPC torch.linalg.solve fp32/fp64 dtype-promote wrapper,
       (ii) scheduler forced onto CPU (set_timesteps(device="cpu") + CPU
            scheduler.step, re-host `timestep` to neuron inside the DiT forward),
       (iii)T5 encoded on CPU/host; VAE decode bf16 on neuron; mp4 via imageio;
            the decode Neuron-vs-CPU PSNR/SSIM correctness gate.
  3. teacache_wan.py    — the TeaCacheController that monkeypatches
     WanTransformer3DModel.forward: block-0 AdaLN-modulated proxy, FEATURE-space
     block-stack residual cache (after_blocks - post_patch), separate cond/uncond
     caches, first/last forced, fitted-polynomial skip decision.

MULTI-RANK CORRECTNESS (the hard part):
  * All 4 ranks run the denoise loop in lockstep. The DiT forward contains
    collectives (QK-norm all-reduce + Rowwise all-reduce), so every rank MUST make
    the SAME run/skip decision or they deadlock.
  * The TeaCache proxy (block-0 AdaLN-modulated input) is computed from data that
    is REPLICATED across ranks (patch_embedding + condition_embedder outputs are
    NOT column-sharded), so each rank independently computes the same relative-L1
    distance -> the same decision. As a hard guarantee against any float edge case
    we ALSO broadcast rank-0's boolean decision every step (a tiny collective, far
    cheaper than a DiT forward) so ranks can never diverge / hang. (--no-bcast to
    rely purely on determinism.)
  * Scheduler runs on CPU deterministically; identical seed => identical latents on
    every rank, so no latent broadcast is needed.
  * Prompt embeds: T5 encoded once on host; the identical embeds tensor is placed
    on every rank's device (each rank encodes identically from the same seed/model).
  * VAE decode: the Rowwise plan all-reduces the DiT output so the final latent is
    REPLICATED on every rank; the pipeline decodes it (no collectives in the VAE),
    and only rank 0 saves the mp4 / runs the gate.
  * Teardown: dist.barrier() + destroy_process_group; only rank 0 does I/O.

LAUNCH (on the box, venv active):
  source /home/ubuntu/workspace/native_venv/bin/activate
  NEURON_CC_FLAGS="--model-type=transformer -O2 --auto-cast=none" \
    torchrun --nnodes 1 --nproc_per_node 4 \
      /home/ubuntu/kernel_research/e2e_tp4_teacache.py \
      --threshes 0.05,0.10 --poly-coeffs /home/ubuntu/kernel_research/teacache_wan_poly_coeffs.npy

  (The VAE recompiles under the transformer flags on first decode (~5 min) then
   caches; if you prefer the unet flags for the VAE stage, run once to warm it or
   set NEURON_CC_FLAGS per-stage.)
"""
from __future__ import annotations
import argparse, gc, math, os, time
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor.parallel import (
    ColwiseParallel, RowwiseParallel, parallelize_module)

# --------------------------------------------------------------------------- #
# host-side fix #1 (from e2e_video.py): UniPC corrector builds R as float32 and
# b as float64 then calls torch.linalg.solve -> "A and B must have the same
# dtype". Promote only when they differ so UniPC otherwise runs unchanged.
# --------------------------------------------------------------------------- #
_ORIG_SOLVE = torch.linalg.solve


def _solve_dtype_safe(A, B, *a, **k):
    if torch.is_tensor(A) and torch.is_tensor(B) and A.dtype != B.dtype:
        c = torch.promote_types(A.dtype, B.dtype)
        A, B = A.to(c), B.to(c)
    return _ORIG_SOLVE(A, B, *a, **k)


torch.linalg.solve = _solve_dtype_safe

MODEL = "/home/ubuntu/wan22"
OUTDIR = "/home/ubuntu/kernel_research"
EMB_PATH = f"{OUTDIR}/_prompt_embeds.pt"   # rank-0 T5 embeds -> file broadcast
DEV = torch.device("neuron")
PROMPT = "a cat playing piano, cinematic, high detail"
NEG = ""
H, W, NF, STEPS, GUID = 480, 832, 49, 50, 5.0
SEED = 0  # identical initial latent every pipe() call -> meaningful cosine

# per-stage timers / captures (rank 0 uses them for reporting)
T: dict[str, float] = {}
CAP: dict[str, object] = {}


def _r0():
    return int(os.environ.get("RANK", "0")) == 0


def _log(m):
    if _r0():
        print(f"[cap] {m}", flush=True)


def _log_all(m):
    print(f"[cap r{os.environ.get('RANK','0')}] {m}", flush=True)


def _host_free_gb():
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable"):
                    return int(line.split()[1]) / 1024.0 / 1024.0
    except Exception:  # noqa: BLE001
        pass
    return -1.0


def _memlog(tag):
    """Lightweight host-RAM guard (per-core HBM is not cheaply queryable in-proc;
    the real device-HBM protection is the placement discipline below)."""
    _log_all(f"[mem] {tag}: host MemAvailable {_host_free_gb():.1f} GB")


def fresh_gen():
    return torch.Generator(device="cpu").manual_seed(SEED)


def _sync(o):
    o = getattr(o, "sample", o)
    o = getattr(o, "last_hidden_state", o)
    if isinstance(o, (tuple, list)):
        o = o[0]
    if torch.is_tensor(o):
        float(o.detach().float().flatten()[:1].cpu())
    return o


# =========================================================================== #
# TP=4 sharding  (verbatim from native_wan_tp.py)
# =========================================================================== #
class _AdaptiveQKNorm(nn.Module):
    """Across-heads RMSNorm correct under Colwise head-sharding: local sum-of-
    squares, all-reduce for the global RMS over the FULL inner dim, apply this
    rank's slice of the (replicated) weight. (LTX-2 fix #4, verbatim.)"""

    def __init__(self, weight, eps, world, rank):
        super().__init__()
        self.full_weight = weight
        self.eps = eps or 1e-6
        self.world = world
        self.rank = rank

    def forward(self, x):
        in_dim = x.shape[-1]
        full = self.full_weight.shape[0]
        sharded = in_dim < full
        local_sq = (x.float() ** 2).sum(dim=-1, keepdim=True)
        if sharded and self.world > 1 and dist.is_initialized():
            dist.all_reduce(local_sq, op=dist.ReduceOp.SUM)
            denom = full
            w = self.full_weight.narrow(0, self.rank * in_dim, in_dim)
        else:
            denom = in_dim
            w = self.full_weight if in_dim == full else self.full_weight.narrow(0, 0, in_dim)
        rms = (local_sq / denom + self.eps).rsqrt()
        return (x.float() * rms).to(x.dtype) * w


def shard_transformer(dit, mesh, world, rank):
    """Apply the validated Wan TP plan in-place to a WanTransformer3DModel."""
    if world <= 1:
        return dit
    n = 0
    for blk in dit.blocks:
        plan = {}
        for an in ("attn1", "attn2"):
            if hasattr(blk, an):
                plan[f"{an}.to_q"] = ColwiseParallel()
                plan[f"{an}.to_k"] = ColwiseParallel()
                plan[f"{an}.to_v"] = ColwiseParallel()
                plan[f"{an}.to_out.0"] = RowwiseParallel()
        plan["ffn.net.0.proj"] = ColwiseParallel()
        plan["ffn.net.2"] = RowwiseParallel()
        parallelize_module(blk, mesh, plan)
        n += 1
    # patch heads -> heads/world (unflatten uses attn.heads on the sharded dim)
    for blk in dit.blocks:
        for an in ("attn1", "attn2"):
            at = getattr(blk, an, None)
            if at is not None and hasattr(at, "heads"):
                at.heads = at.heads // world
    # exact across-heads adaptive QK-norm
    for blk in dit.blocks:
        for an in ("attn1", "attn2"):
            at = getattr(blk, an, None)
            if at is None:
                continue
            for nm in ("norm_q", "norm_k"):
                norm = getattr(at, nm, None)
                if norm is not None and getattr(norm, "weight", None) is not None:
                    setattr(at, nm, _AdaptiveQKNorm(
                        norm.weight, getattr(norm, "eps", 1e-6), world, rank))
    _log(f"sharded {n} blocks, heads->{24 // world}, adaptive QK-norm installed")
    return dit


# =========================================================================== #
# TeaCache controller (from teacache_wan.py) + TP-awareness
# =========================================================================== #
class TeaCacheController:
    """Drop-in WanTransformer3DModel.forward that caches the FEATURE-space
    block-stack residual. TP-aware: re-hosts scheduler-CPU inputs to neuron, and
    broadcasts rank-0's run/skip decision so all ranks stay in lockstep through
    the collectives inside the compiled block stack."""

    def __init__(self, transformer, num_steps, num_cfg=2, poly=None,
                 bcast=True):
        self.t = transformer
        self.num_steps = num_steps
        self.num_cfg = num_cfg
        self.poly = poly           # None => identity rescale (direct rel-L1)
        self.thresh = None         # set per-run; None => TeaCache OFF (baseline)
        self.bcast = bcast
        self.world = dist.get_world_size() if dist.is_initialized() else 1
        self.proxy_fn = torch.compile(self._proxy, backend="neuron", dynamic=False)
        self.run_fn = torch.compile(self._run, backend="neuron", dynamic=False)
        self.skip_fn = torch.compile(self._skip, backend="neuron", dynamic=False)
        self.reset()

    def reset(self):
        self.state = {"cond": self._blank(), "uncond": self._blank()}
        self.call_count = 0
        self.executed = 0
        self.skipped = 0
        self.exec_time = 0.0
        self.skip_time = 0.0

    @staticmethod
    def _blank():
        return {"prev_mod": None, "accum": 0.0, "residual": None, "calls": 0}

    @staticmethod
    def _to_dev(x):
        return x.to(DEV) if torch.is_tensor(x) else x

    # ---- shared front/back halves of WanTransformer3DModel.forward ----
    def _front(self, hidden_states, timestep, encoder_hidden_states):
        t = self.t
        bsz, _, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = t.config.patch_size
        shapes = (bsz, num_frames // p_t, height // p_h, width // p_w, p_t, p_h, p_w)
        rotary_emb = t.rope(hidden_states)
        hs = t.patch_embedding(hidden_states)
        hs = hs.flatten(2).transpose(1, 2)                    # post-patch (replicated)
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
            hs = block(hs, enc, timestep_proj, rotary_emb)      # collectives inside
        residual = hs - post_patch                              # FEATURE-space, replicated
        out = self._tail(hs, temb, shapes)
        return out, residual

    def _skip(self, hidden_states, timestep, encoder_hidden_states, residual):
        hs, enc, temb, timestep_proj, rotary_emb, _, shapes = self._front(
            hidden_states, timestep, encoder_hidden_states)
        hs = hs + residual
        return self._tail(hs, temb, shapes)

    def _agree(self, should_calc):
        """Broadcast rank-0's boolean decision to all ranks (lockstep guard)."""
        if not (self.bcast and self.world > 1 and dist.is_initialized()):
            return should_calc
        dec = torch.tensor([1 if should_calc else 0], dtype=torch.int32, device=DEV)
        try:
            dist.broadcast(dec, src=0)
        except Exception:                                       # noqa: BLE001
            # fall back to all_reduce(SUM): rank0 contributes, others 0, then
            # rank0's value is broadcast implicitly (world copies of it).
            dec = torch.tensor(
                [(1 if should_calc else 0) if dist.get_rank() == 0 else 0],
                dtype=torch.int32, device=DEV)
            dist.all_reduce(dec, op=dist.ReduceOp.SUM)
        return bool(int(dec.item()))

    def forward(self, hidden_states, timestep, encoder_hidden_states,
                encoder_hidden_states_image=None, return_dict=True,
                attention_kwargs=None):
        # scheduler runs on CPU -> re-host inputs to neuron for the DiT forward.
        hidden_states = self._to_dev(hidden_states)
        timestep = self._to_dev(timestep)
        encoder_hidden_states = self._to_dev(encoder_hidden_states)

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

        if self.thresh is not None:
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

        # lockstep guard across ranks (proxy is replicated so this is a no-op in
        # the common case; it prevents any float-edge divergence -> deadlock).
        should_calc = self._agree(should_calc)

        st["prev_mod"] = modulated.detach().clone()   # .clone() mandatory (buffer reuse)
        st["calls"] += 1

        tb = time.time()
        if should_calc:
            out, residual = self.run_fn(hidden_states, timestep, encoder_hidden_states)
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


# =========================================================================== #
# metrics + frame utils  (from e2e_video.py)
# =========================================================================== #
def _gaussian_kernel1d(sigma=1.5, radius=5):
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    k = np.exp(-(x ** 2) / (2 * sigma ** 2))
    return k / k.sum()


def _blur(img, k):
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
    a = a.astype(np.float64); b = b.astype(np.float64)
    k = _gaussian_kernel1d()
    c1 = (0.01 * data_range) ** 2; c2 = (0.03 * data_range) ** 2
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
    a = a.astype(np.float64); b = b.astype(np.float64)
    mse = np.mean((a - b) ** 2)
    if mse <= 1e-12:
        return 99.0
    return float(20 * math.log10(data_range / math.sqrt(mse)))


def to_uint8_frames(video_bcfhw):
    v = video_bcfhw
    if torch.is_tensor(v):
        v = v.detach().float().cpu().numpy()
    v = v[0]
    v = np.transpose(v, (1, 2, 3, 0))
    v = (v / 2.0 + 0.5).clip(0, 1)
    return (v * 255.0).round().astype(np.uint8)


def gray(frames_uint8):
    return frames_uint8.astype(np.float64) @ np.array([0.299, 0.587, 0.114])


def cosine(a, b):
    a = a.flatten().double(); b = b.flatten().double()
    return float((a @ b) / (a.norm() * b.norm() + 1e-12))


# =========================================================================== #
# pipeline build  (T5 host encode + TP shard + scheduler-CPU + VAE capture)
# =========================================================================== #
def build(world, rank, mesh, poly):
    from diffusers import WanPipeline
    _memlog("pre-load")

    # --- HARDENING #2: staggered from_pretrained (one rank loads at a time) so the
    # 4-process host-RAM peak never stacks 4x(transformer+T5). Ranks 1-3 free T5
    # immediately after load (they never encode); only rank 0 keeps T5. ---
    pipe = None
    t0 = time.time()
    for i in range(world):
        if rank == i:
            _log_all(f"loading WanPipeline (bf16, staggered slot {i})...")
            pipe = WanPipeline.from_pretrained(
                MODEL, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
            if rank != 0:
                pipe.text_encoder = None            # HARDENING #1: no T5 on ranks 1-3
                gc.collect()
        if dist.is_initialized() and world > 1:
            dist.barrier()
    _log(f"  pipe loaded (staggered) in {time.time()-t0:.1f}s")
    _memlog("post-load")

    # --- HARDENING #1: T5 encode on CPU (fp32) on RANK 0 ONLY; broadcast the
    # prompt-embeds to ranks 1-3 via a shared-filesystem file (robust: no large-
    # tensor neuron collective, no shape negotiation). T5 never touches device. ---
    if rank == 0:
        _log("rank0: encoding prompt with T5 on CPU (fp32)...")
        pipe.text_encoder.to(device="cpu", dtype=torch.float32)
        t0 = time.time()
        with torch.no_grad():
            pe, npe = pipe.encode_prompt(
                prompt=PROMPT, negative_prompt=NEG, do_classifier_free_guidance=True,
                num_videos_per_prompt=1, max_sequence_length=512,
                device=torch.device("cpu"), dtype=torch.bfloat16)
        T["t5"] = time.time() - t0
        _log(f"  T5 encode (CPU): {T['t5']:.2f}s  embeds={tuple(pe.shape)}")
        pipe.text_encoder = None
        gc.collect()
        torch.save({"pe": pe.cpu(), "npe": npe.cpu()}, EMB_PATH)
    if dist.is_initialized() and world > 1:
        dist.barrier()
    if rank != 0:
        d = torch.load(EMB_PATH, map_location="cpu")
        pe, npe = d["pe"], d["npe"]
    if dist.is_initialized() and world > 1:
        dist.barrier()   # all ranks have read embeds before any later overwrite
    _memlog("post-T5-free")

    # --- TP shard the DiT; place ONLY the sharded DiT on device. ---
    # HARDENING #3: VAE stays on CPU on ranks 1-3 and is loaded to device on rank 0
    # only (for the final decode). During denoise the device holds only sharded DiT.
    shard_transformer(pipe.transformer, mesh, world, rank)
    pipe.transformer.to(DEV)
    if rank == 0:
        pipe.vae.to(DEV)
    type(pipe)._execution_device = property(lambda self: DEV)
    pe = pe.to(DEV); npe = npe.to(DEV)
    _memlog("post-shard-to-device")

    # --- fix #2: force the scheduler entirely onto CPU ---
    _set_ts = pipe.scheduler.set_timesteps

    def set_ts_cpu(num_inference_steps=None, device=None, **kw):
        return _set_ts(num_inference_steps=num_inference_steps,
                       device=torch.device("cpu"), **kw)
    pipe.scheduler.set_timesteps = set_ts_cpu

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

    # --- VAE decode capture (drives the correctness gate); time it (rank 0) ---
    # HARDENING #3: real bf16 decode on rank 0's device only. Ranks 1-3 (no VAE on
    # device) return a correctly-shaped zeros result so pipeline post-processing
    # doesn't crash; and since there are NO collectives between end-of-denoise and
    # the run_generation barrier, ranks may diverge safely here (see the try/except
    # tolerance in run_generation). This keeps ranks 1-3's device free of the VAE.
    _vd = pipe.vae.decode

    def vd(latents, *a, **k):
        if _r0():
            CAP["latent"] = latents.detach().float().cpu().clone()
            t = time.time(); r = _vd(latents, *a, **k)
            o = _sync(r)
            T["vae"] = T.get("vae", 0.0) + time.time() - t
            CAP["neuron_decode"] = o.detach().float().cpu().clone()
            return r
        # ranks 1-3: skip real decode, return zeros of the decoded-video shape.
        b = latents.shape[0]
        z = torch.zeros(b, 3, NF, H, W, dtype=torch.float32)
        if not k.get("return_dict", True):
            return (z,)
        class _D:  # minimal DecoderOutput stand-in
            pass
        r = _D(); r.sample = z
        return r
    pipe.vae.decode = vd

    # --- install the TeaCache controller as the transformer forward ---
    num_cfg = 2 if GUID > 1.0 else 1
    ctrl = TeaCacheController(pipe.transformer, num_steps=STEPS, num_cfg=num_cfg,
                              poly=poly, bcast=True)
    pipe.transformer.forward = ctrl.forward
    return pipe, pe, npe, ctrl


def gen_kwargs(pe, npe):
    return dict(prompt=None, negative_prompt=None,
                prompt_embeds=pe, negative_prompt_embeds=npe,
                height=H, width=W, num_frames=NF, num_inference_steps=STEPS,
                guidance_scale=GUID, output_type="np")


# =========================================================================== #
# robust mp4 export (rank 0 only): try imageio-ffmpeg, then diffusers/opencv;
# NEVER raise (a write-backend gap must not lose the measured e2e). Always also
# dump raw frames to .npy so the video data is recoverable offline.
# =========================================================================== #
def _export_video(frames, mp4_path):
    te = time.time()
    arr = np.asarray(frames)                    # (F,H,W,C) float [0,1]
    u8 = (arr.clip(0, 1) * 255.0).round().astype(np.uint8)
    npy_path = mp4_path.replace(".mp4", "_frames.npy")
    try:
        np.save(npy_path, u8)
        _log(f"  frames dumped -> {npy_path} {u8.shape}")
    except Exception as e:  # noqa: BLE001
        _log(f"  (frames npy dump failed: {type(e).__name__})")
    # backend 1: imageio + ffmpeg
    try:
        import imageio
        imageio.mimwrite(mp4_path, list(u8), fps=16, codec="libx264",
                         quality=8, macro_block_size=None)
        T["export"] = time.time() - te
        _log(f"  mp4 via imageio -> {mp4_path}")
        return mp4_path
    except Exception as e:  # noqa: BLE001
        _log(f"  imageio export failed ({type(e).__name__}: {str(e)[:80]}); trying diffusers/opencv")
    # backend 2: diffusers export_to_video (opencv)
    try:
        from diffusers.utils import export_to_video
        export_to_video(list(frames), mp4_path, fps=16)
        T["export"] = time.time() - te
        _log(f"  mp4 via diffusers/opencv -> {mp4_path}")
        return mp4_path
    except Exception as e:  # noqa: BLE001
        T["export"] = time.time() - te
        _log(f"  mp4 export FAILED on all backends ({type(e).__name__}); "
             f"frames preserved at {npy_path}")
        return None


# =========================================================================== #
# a single glued generation  (all ranks run; rank 0 reports/exports)
# =========================================================================== #
def run_generation(pipe, ctrl, pe, npe, thresh, tag, export_mp4):
    ctrl.reset()
    ctrl.thresh = thresh
    T.pop("vae", None)
    kw = dict(gen_kwargs(pe, npe), generator=fresh_gen())
    if dist.is_initialized():
        dist.barrier()
    t0 = time.time()
    # HARDENING #3: only rank 0 runs the real VAE decode; ranks 1-3 may hit a
    # post-denoise (VAE/post-proc) error because VAE isn't on their device. That is
    # SAFE — there are no collectives after the last DiT forward, so a rank can
    # finish or fail post-denoise and still reach the barrier below without a
    # deadlock. Tolerate it on non-rank-0; re-raise on rank 0 (we need its output).
    try:
        res = pipe(**kw)
    except Exception as e:  # noqa: BLE001
        if _r0():
            raise
        _log_all(f"tolerated post-denoise error: {type(e).__name__}: {str(e)[:120]}")
        res = None
    e2e_pipe = time.time() - t0
    if dist.is_initialized():
        dist.barrier()

    if not _r0():
        # non-rank-0: nothing to report/export; return a minimal record.
        return dict(tag=tag, thresh=thresh, e2e_pipe=e2e_pipe,
                    executed=ctrl.executed, skipped=ctrl.skipped,
                    skip_frac=ctrl.skipped / max(1, ctrl.executed + ctrl.skipped),
                    dit_exec=ctrl.exec_time, dit_skip=ctrl.skip_time,
                    vae=0.0, mp4=None, frames=None, latent=None, neuron_decode=None)

    frames = res.frames[0]                      # (F,H,W,C) float [0,1]
    latent = CAP.get("latent")                  # captured un-normalized VAE input
    dit_exec, dit_skip = ctrl.exec_time, ctrl.skip_time
    vae = T.get("vae", 0.0)
    executed, skipped = ctrl.executed, ctrl.skipped
    total = executed + skipped
    skip_frac = skipped / max(1, total)

    mp4_path = None
    if _r0():
        if export_mp4:
            mp4_path = _export_video(frames, f"{OUTDIR}/wan_tp4_{tag}.mp4")
        t5 = T.get("t5", 0.0)
        other = e2e_pipe - dit_exec - dit_skip - vae
        print("\n===================== REAL MEASURED E2E "
              f"[{tag}] =====================", flush=True)
        print(f"  thresh={thresh}  forwards: executed={executed} skipped={skipped} "
              f"(total={total})  skip%={100*skip_frac:.1f}", flush=True)
        print(f"  T5 text-encode (CPU):    {t5:.2f}s", flush=True)
        print(f"  DiT denoise (neuron TP=4): exec={dit_exec:.1f}s "
              f"skip={dit_skip:.1f}s  ({executed} exec @ "
              f"{dit_exec/max(1,executed)*1000:.0f} ms/fwd)", flush=True)
        print(f"  VAE decode (neuron bf16): {vae:.1f}s", flush=True)
        if export_mp4:
            print(f"  export mp4:              {T.get('export',0):.2f}s -> {mp4_path}", flush=True)
        print(f"  sched/host overhead:     {other:.1f}s", flush=True)
        print(f"  GLUED pipe() wall-clock: {e2e_pipe:.1f}s", flush=True)
        full = t5 + e2e_pipe + T.get("export", 0.0)
        print(f"  FULL e2e (T5+pipe+export): {full:.1f}s   "
              f"(H100 golden = 33.0s)", flush=True)
        print("=" * 66, flush=True)

    return dict(tag=tag, thresh=thresh, e2e_pipe=e2e_pipe, executed=executed,
                skipped=skipped, skip_frac=skip_frac, dit_exec=dit_exec,
                dit_skip=dit_skip, vae=vae, mp4=mp4_path, frames=frames,
                latent=latent.clone() if latent is not None else None,
                neuron_decode=(CAP.get("neuron_decode").clone()
                               if CAP.get("neuron_decode") is not None else None))


# =========================================================================== #
# correctness gate  (rank 0 only; VAE has no collectives)
# =========================================================================== #
def cpu_reference_decode(latent):
    from diffusers import AutoencoderKLWan
    print("\n[gate] loading CPU fp32 VAE for reference decode...", flush=True)
    m = AutoencoderKLWan.from_pretrained(MODEL, subfolder="vae",
                                         torch_dtype=torch.float32).to("cpu").eval()
    lat = latent.to(torch.float32)
    t0 = time.time()
    with torch.no_grad():
        out = m.decode(lat, return_dict=False)[0]
    print(f"[gate] CPU fp32 decode: {time.time()-t0:.1f}s", flush=True)
    del m; gc.collect()
    return out.detach().float().cpu()


def gate(run, baseline_latent=None, skip_parity=False):
    print(f"\n========== CORRECTNESS GATE [{run['tag']}] ==========", flush=True)
    ok = True
    neuron_dec = run["neuron_decode"]
    if neuron_dec is None or run["latent"] is None:
        print("[gate] ERROR: no decode/latent captured", flush=True)
        return
    # The Neuron-bf16-vs-CPU-fp32 parity decode uses a single-threaded CPU fp32
    # VAE (~36 min for 49 frames under OMP_NUM_THREADS=1). --no-parity skips it and
    # relies on the cheap frame-sanity + latent-cosine checks (used for TeaCache
    # runs, whose fidelity is captured by latent-cos vs the parity-verified base).
    if not skip_parity:
        cpu_dec = cpu_reference_decode(run["latent"])
        nf = to_uint8_frames(neuron_dec); cf = to_uint8_frames(cpu_dec)
        gn, gc_ = gray(nf), gray(cf)
        F = nf.shape[0]
        psnrs = [psnr(nf[i], cf[i]) for i in range(F)]
        ssims = [ssim_gray(gn[i], gc_[i]) for i in range(F)]
        p_min, p_mean = min(psnrs), float(np.mean(psnrs))
        s_min, s_mean = min(ssims), float(np.mean(ssims))
        print(f"[decode parity] Neuron-bf16 vs CPU-fp32  (SSIM impl: {SSIM_IMPL})", flush=True)
        print(f"    PSNR  min={p_min:.1f} mean={p_mean:.1f} dB   (gate >40)", flush=True)
        print(f"    SSIM  min={s_min:.4f} mean={s_mean:.4f}       (gate >0.98)", flush=True)
        parity_ok = (p_min > 40.0) and (s_min > 0.98)
        ok &= parity_ok
        print(f"    -> decode parity: {'PASS' if parity_ok else 'REVIEW'}", flush=True)
    else:
        print("[decode parity] SKIPPED (--no-parity)", flush=True)

    frames_np = run["frames"]
    fint = frames_np * 255.0
    any_nan = bool(np.isnan(frames_np).any()) or bool(torch.isnan(neuron_dec).any())
    stds = [float(fint[i].std()) for i in range(frames_np.shape[0])]
    n_frames = frames_np.shape[0]; std_min = min(stds)
    print(f"[frame sanity] frames={n_frames}(expect {NF}) NaN={any_nan} "
          f"std min={std_min:.1f} mean={float(np.mean(stds)):.1f} (gate >60)", flush=True)
    sanity_ok = (n_frames == NF) and (not any_nan) and (std_min > 60.0)
    ok &= sanity_ok
    print(f"    -> frame sanity: {'PASS' if sanity_ok else 'REVIEW'}", flush=True)

    if baseline_latent is not None:
        c = cosine(run["latent"], baseline_latent)
        print(f"[latent cos vs TP4 no-cache] {c:.5f}", flush=True)
    print(f"CORRECTNESS [{run['tag']}]: {'PASS' if ok else 'REVIEW'}", flush=True)


# =========================================================================== #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshes", type=str, default="0.05,0.10")
    ap.add_argument("--poly-coeffs", type=str,
                    default=f"{OUTDIR}/teacache_wan_poly_coeffs.npy")
    ap.add_argument("--no-bcast", action="store_true",
                    help="rely purely on proxy determinism (no per-step decision broadcast)")
    ap.add_argument("--no-parity", action="store_true",
                    help="skip the slow single-threaded CPU-fp32 reference decode in the "
                         "gate (keep frame-sanity + latent-cos); use for TeaCache sweeps")
    a = ap.parse_args()
    threshes = [float(x) for x in a.threshes.split(",") if x.strip()]

    dist.init_process_group(backend="neuron")
    world = dist.get_world_size(); rank = dist.get_rank()
    mesh = init_device_mesh("neuron", (world,))

    # load the calibrated rescaling polynomial (fitted on .73); identity if absent.
    poly = None
    if os.path.exists(a.poly_coeffs):
        coeffs = np.load(a.poly_coeffs)
        poly = np.poly1d(coeffs)
        _log(f"loaded TeaCache poly coeffs from {a.poly_coeffs}: {list(coeffs)}")
    else:
        _log(f"NO poly coeffs at {a.poly_coeffs} -> IDENTITY rescale (raw rel-L1)")

    pipe, pe, npe, ctrl = build(world, rank, mesh, poly)
    if a.no_bcast:
        ctrl.bcast = False

    # WARMUP: compiles proxy/run/skip block stacks + VAE once (TeaCache OFF).
    _log("=== WARMUP (compiles DiT TP=4 block stack + VAE; TeaCache OFF) ===")
    t0 = time.time()
    ctrl.reset(); ctrl.thresh = None
    try:
        _ = pipe(**dict(gen_kwargs(pe, npe), generator=fresh_gen()))
    except Exception as e:  # noqa: BLE001
        if _r0():
            raise
        _log_all(f"warmup tolerated post-denoise error: {type(e).__name__}: {str(e)[:120]}")
    if dist.is_initialized():
        dist.barrier()
    _memlog("post-warmup")
    _log(f"warmup done in {time.time()-t0:.1f}s (executed={ctrl.executed})")

    results = []
    # (a) TP=4 no-cache  -> mp4 + gate + baseline latent for cosine
    results.append(run_generation(pipe, ctrl, pe, npe, None, "nocache", export_mp4=True))
    base_latent = results[0]["latent"]

    # (b),(c) TeaCache thresholds
    for th in threshes:
        tag = f"tc{th}".replace("0.", "0p")
        results.append(run_generation(pipe, ctrl, pe, npe, th, tag, export_mp4=True))

    # correctness gates (rank 0)
    if _r0():
        gate(results[0], skip_parity=a.no_parity)
        for r in results[1:]:
            gate(r, baseline_latent=base_latent, skip_parity=a.no_parity)

        print("\n" + "=" * 78, flush=True)
        print("CAPSTONE SUMMARY  (Wan 2.2 TI2V-5B, 480x832, 49f, %d steps, TP=4, CFGx2)"
              % STEPS, flush=True)
        print(f"{'run':>10} {'thr':>6} {'exec':>5} {'skip':>5} {'skip%':>6} "
              f"{'pipe_s':>7} {'full_e2e':>9} {'mp4':>28}", flush=True)
        t5 = T.get("t5", 0.0)
        for r in results:
            full = t5 + r["e2e_pipe"]
            thr = "-" if r["thresh"] is None else f"{r['thresh']}"
            mp4 = os.path.basename(r["mp4"]) if r["mp4"] else "-"
            print(f"{r['tag']:>10} {thr:>6} {r['executed']:>5} {r['skipped']:>5} "
                  f"{100*r['skip_frac']:>5.1f}% {r['e2e_pipe']:>6.1f}s "
                  f"{full:>8.1f}s {mp4:>28}", flush=True)
        print("H100 golden e2e = 33.0s.", flush=True)
        print("=" * 78, flush=True)

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
    os._exit(0)


if __name__ == "__main__":
    main()

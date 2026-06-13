# LTX-2 18.88B on Trainium2 — Native PyTorch (Beta 3) — **WORKING**

✅ **End-to-end inference working as of 2026-06-12.** First-ever LTX-2
inference outside Lightricks' GPU stack, using native PyTorch +
`torch_neuronx` on the Beta 3 stack — no NxDI, no NxDT, no vLLM.

Generated outputs (commit-pinned, see this folder):
- `ltx2_run.png` — frame 0 of the canonical run (512×384, 25-frame clip)
- `ltx2_run.mp4` — full 25-frame clip at 24 fps

## TL;DR benchmark (trn2.48xlarge TP=4, Beta 3)

| Metric | Trainium2 native PyTorch | H100 (1× of p5.48xl) | gap |
|---|---:|---:|---:|
| Setup | 31.1 s | 49 s | 0.63× (Trn faster) |
| TTFI (cold + NEFF compile) | **169.3 s** | 52.3 s | 3.24× |
| Warm mean (8 steps, 384×512, 25f) | **165.4 s** (n=5, σ=0.74, p95=166.2) | 2.84 s | **58.2×** |
| Per-step transformer only | 6.33 s | 326 ms | 19.4× |

**The 58× warm gap is dominated by CPU flat tax**, not Trainium per-
step. Trainium spends only 31% of warm time in the transformer
(50 s of 164 s). The other 113 s is CPU host work: Gemma-3 12B text
encoder + LTX-2 connectors + video VAE encode/decode + audio VAE.
Closing that gap (move encoder + VAE onto Neuron) is multi-day work
and the largest remaining lever.

Full report: [`BENCHMARK_TRN2_48XL.md`](BENCHMARK_TRN2_48XL.md)

## Goal

Port `Lightricks/LTX-2` (18.88B audio-video DiT) to Trainium2 using
**native PyTorch + torch_neuronx on the Beta 3 stack**, TP=4,
mirroring the approach that shipped for Qwen-Image-Edit.

## What works (verified end-to-end)

- ✅ Beta 3 DLC + driver install, `torch.device("neuron")` ops
- ✅ Meta-init build of the 18.88B transformer (no full-weight CPU stage)
- ✅ TP=4 `parallelize_module` covering all six attention paths per
  block (`attn1, attn2, audio_attn1, audio_attn2, audio_to_video_attn,
  video_to_audio_attn`) plus video FFN + audio FFN
- ✅ Sharded weight loader (~5s) — `ltx2_meta_loader.py`
- ✅ `attn.heads` patched to heads/N for all 6 attention types
- ✅ All 4 RoPE modules monkey-patched at class level: coords built on
  CPU then moved to Neuron; cos/sin sliced per-rank by head range
- ✅ Adaptive QK norm with cross-rank all-reduce installed AFTER weight
  load (resolves `rms_norm_across_heads` under sharded inner_dim)
- ✅ CPU↔Neuron transfer wrapper at transformer boundary
  (`_NeuronTransformerWrapper`)
- ✅ VAE + audio_VAE explicitly pinned to CPU with tensor-arg-coercing
  decode wrappers
- ✅ Pipeline runs end-to-end: text encoder (CPU Gemma-3 12B) →
  connectors (CPU) → 8-step denoising loop on TP=4 Neuron transformer
  → VAE decode (CPU) → audio_VAE decode (CPU) → MP4 export

## The eight engineering fixes (the recipe)

These are documented in `BENCHMARK_TRN2_48XL.md` and inline in
`ltx2_tp_plan.py` / `ltx2_run.py`:

1. **Beta 3 device API** — `torch.device("neuron")`, not
   `privateuseone:N`
2. **Meta-init build** — no full bf16 weight stage on a single core
   (would OOM the 24 GB-per-core budget at ~38 GB)
3. **TP=4 plan with all six attention paths** — including
   `video_to_audio_attn` (the breakthrough fix; missing this leaves
   V2A.to_q full-size and breaks RoPE shape matching)
4. **Adaptive QK norm with cross-rank all-reduce** — installed AFTER
   weight load so the loader can materialize `norm_q.weight` /
   `norm_k.weight` first; replaces stock RMSNorm with a TP-aware
   version that all-reduces sum-of-squares
5. **RoPE class-level monkey-patch** — coords built on CPU then moved
   to Neuron (eliminates meta leaks during shape inference); cos/sin
   sliced per-rank to match the rank's head range
6. **`attn.heads` → heads/N** — block forward does `unflatten(2,
   (attn.heads, -1))`, so each rank's `attn.heads` must reflect its
   local head count
7. **CPU↔Neuron transfer wrapper at transformer boundary** — single
   chokepoint moves all tensor inputs to Neuron before the forward,
   avoiding chasing each individual arg through diffusers' internals
8. **VAE + audio_VAE pinned to CPU** with tensor-arg-coercing decode
   wrappers (the pipeline auto-moves both to the execution device,
   we override that)

## Files

| File | Purpose |
|---|---|
| `ltx2_run.py` | TP=4 runner: meta-init → parallelize → load → install adaptive QK norm → patch RoPE → swap into pipeline → CPU/Neuron patches → generate |
| `ltx2_tp_plan.py` | TP plan (1344 entries) + `apply_tp_fixes` (heads patching) + `install_adaptive_qk_norm` (TP RMSNorm) + `patch_rope_rank_slice` (4-RoPE rank slice + CPU-then-neuron coords) |
| `ltx2_meta_loader.py` | Sharded weight loader: regex-based shard rules + module-walk resolver; replicates `norm_[qk]` |
| `bench_ltx2.py` | Benchmark harness (TTFI + warm + sweep) |
| `ltx2_beta3.py` | Single-core smoke (verified Beta 3 stack — also OOMs as expected without TP) |
| `ltx2_beta3_fsdp.py` | Early FSDP attempt (superseded by `ltx2_run.py`) |
| `ltx2_naive_trn2.py` | Naive single-core attempt (OOMs — documents need for TP) |
| `setup_beta3.sh` | Beta 3 host setup (driver from DLC artifacts) |
| `BENCHMARK_TRN2_48XL.md` | Full benchmark report + recipe |
| `bench_summary.json` | Measurement data (TTFI, warm samples, per-step) |
| `LTX2_BETA3_STATUS.md` | Pre-resolution status doc (kept as historical record) |
| `ltx2_run.png` | Generated frame 0 (512×384 8-bit RGB) |
| `ltx2_run.mp4` | Generated 25-frame video at 24 fps |

## Repro

```bash
# Beta 3 DLC pulled, driver installed via runtime_artifacts/*.deb
# In the beta3 container:
HF_HOME=/path/to/hf/cache HF_HUB_OFFLINE=1 \
NEURON_RT_VIRTUAL_CORE_SIZE=2 NEURON_RT_NUM_CORES=4 \
torchrun --nproc_per_node=4 --rdzv_backend c10d --rdzv_endpoint localhost:29500 \
    ltx2_run.py --num-steps 8 --num-frames 25 --no-compile

# Bench:
torchrun --nproc_per_node=4 --rdzv_backend c10d --rdzv_endpoint localhost:29500 \
    bench_ltx2.py --n-canonical-warm 5 --canonical-steps 8
```

## Open work

1. **Reduce 113 s CPU flat tax** — biggest single lever. Move
   Gemma-3 text encoder + LTX-2 video/audio VAEs onto Neuron via
   NKI/torch.compile. Could shrink warm latency from 164 s to ~50-70 s.
2. **Steady-state memory hygiene** — explicit `gc.collect()` at
   iteration boundary (already in `bench_ltx2.py` v2) gives 5 clean
   warm samples with σ=0.74 s. Earlier v1 OOM-killed (-9) after 3.
3. **Higher-resolution validation** — only ran 384×512/25f. H100
   reference goes to 768×1024 in 5.5 s; Trainium would scale to
   ~9× per-step at that resolution.
4. **NKI fused attention for LTX-2 attention shapes** — same ~12%
   wins available as on Qwen-Image-Edit.
5. **`torch.compile(backend="neuron")` on the transformer** — current
   run uses `--no-compile`. Compile path would dynamically capture and
   should win another 10-15% per step at the cost of TTFI.

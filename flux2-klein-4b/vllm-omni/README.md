# FLUX.2-klein-4B on Trainium2 — vLLM-Omni serving path

End-to-end FLUX.2-klein 4B image generation on Trainium2 served through
the vllm-omni Omni runtime. This path is what you want when FLUX.2-klein
needs to live alongside other modalities (LLM + image + video + audio)
inside a single vllm-omni service with shared scheduling, KV manager,
and request orchestration.

> If your goal is **lowest-latency standalone FLUX.2-klein**, use the
> [native PyTorch path](../native-pytorch/) instead — it's ~15× faster
> per step on this model because it skips the omni engine layer.

## Status

Working end-to-end as of 2026-06-13. NEFF compiles, denoising loop
runs on Neuron, latents unpack on CPU, VAE decode on CPU, real PNGs
produced.

| Phase | 256×256 / 4 steps | 512×512 / 8 steps |
|---|---:|---:|
| Cold TTFI (compile + first gen) | 548.8 s | 1483 s |
| Bench mean (n=2 after warm-up) | **73.75 s** (σ 0.20 s) | **290.84 s** (σ 0.23 s) |
| Per-step transformer | **18.44 s** | **36.35 s** |
| NEFF size | 12 MB | 165 MB |

Per-step ~doubles for a 4× token bump (256² → 512²), as expected for an
attention-dominated 4B DiT on a single core.

## Architecture

`NeuronFlux2KleinPipeline` (subclass of vllm-omni's `Flux2KleinPipeline`)
auto-registers via `PIPELINE_REGISTRY`. The class:

1. Skips the parent `__init__` (which would put text-encoder + VAE on
   Neuron). Inits `nn.Module` directly; loads components on CPU.
2. Overrides `to()` to move only the transformer to Neuron.
3. Overrides `encode_prompt`, `_encode_vae_image`,
   `prepare_image_latents` to keep CPU-only components on CPU.
4. Wraps the DiT in a `_NeuronTransformerWrapper` that coerces all
   tensor kwargs to the inner device + `.contiguous()`.
5. Overrides `forward()` with a dummy_run short-circuit so vllm-omni's
   warmup doesn't compile a useless dummy graph.

## The eight-step Neuron-quirk fix sequence

The vllm-omni FLUX.2-klein pipeline runs through Dynamo + FX capture +
neuronx-cc + the omni runtime. Each layer has its own assumptions that
broke for FLUX.2-klein. The fixes, in the order the failures surfaced:

1. **`torch.polar` (complex64) elimination** — Dynamo inlines bytecode,
   so monkey-patching `Flux2PosEmbed.forward` does not survive the FX
   trace. Fix: edit upstream `flux2_klein_transformer.py` directly to
   compute cos/sin via real arithmetic in fp32, then move to the ids
   device. See `src/upstream_patch.py`.
2. **`float64` elimination (NCC_ESPP004)** — Neuron Beta has no f64.
   Pinned all positional/RoPE arithmetic to fp32.
3. **Non-contiguous slicing in FX graph** — `chunk` / `split` /
   slice-via-indexing produce views with non-standard strides that
   compile but fail at execute. Inserted `.contiguous()` after every
   such op in the upstream transformer source (~10 patch sites).
4. **Wrapper boundary contiguity** — CPU round-trip in
   `_NeuronTransformerWrapper.forward` for any non-contiguous input.
5. **Scheduler `step()` bf16→fp32 upcast** — patched
   `scheduling_flow_match_euler_discrete.py:486` to skip the upcast,
   plus split a combined `.to(device, dtype)` call on lines 213-214.
6. **VAE decode device mismatch** — patched
   `pipeline_flux2_klein.py:992` to move latents to CPU before VAE
   decode (VAE is pinned to CPU per our v1 design).
7. **NEFF compile** — clean compile (~8 min for 256², ~14 min for 512²,
   all passes "Failed: 0").
8. **`_unpack_latents_with_ids` scatter on non-contiguous expand** —
   the scatter into `out` from `flat_ids.unsqueeze(1).expand(-1, ch)`
   tripped Neuron's contiguity check. Patched the call site at line
   ~981 to move latents+ids to CPU before unpack and keep them there
   through VAE decode. See `src/patch_unpack.py`.

## Files

| File | Role |
|---|---|
| `src/neuron_flux2_klein_pipeline.py` | The Neuron-aware Flux2KleinPipeline subclass (~600 lines) |
| `src/run_flux2_klein_omni.py` | Runner script (mirrors LTX-2 omni runner pattern) |
| `src/flux2_klein_stage.yaml` | Stage config for the Omni engine (TP=1 default; 4B fits on 1 core) |
| `src/upstream_patch.py` | Patches `Flux2PosEmbed.forward` (real-arithmetic RoPE) |
| `src/patch_unpack.py` | Patches `_unpack_latents_with_ids` call site to run on CPU |
| `src/merge_lora.py` | Offline LoRA fuse tool (use to merge a LoRA into base before serving) |
| `results/flux2_klein_256x256.png` | First end-to-end output, 256² / 4 steps |
| `results/flux2_klein_512x512.png` | Scaled output, 512² / 8 steps |

## Reproduction

Inside a vllm-omni container with FLUX.2-klein-4B weights cached:

```bash
# One-time: apply the upstream source patches
python3 src/upstream_patch.py
python3 src/patch_unpack.py

# Run
PYTHONDONTWRITEBYTECODE=1 \
VLLM_NEURON_COMPILATION_TIMEOUT=3600 \
HF_TOKEN=<your_token> \
python3 src/run_flux2_klein_omni.py \
  --tensor-parallel-size 1 \
  --bench-runs 2 \
  --height 512 --width 512 --num-steps 8 \
  --output flux2_klein.png
```

After every failed run, restart the container to clear zombies:
`sudo docker restart vllm_omni`.

### LoRA serving

To serve with a LoRA fused into the base, merge offline first:

```bash
python3 src/merge_lora.py \
    --base-model black-forest-labs/FLUX.2-klein-4B \
    --lora <provider>/flux-2-klein-4B-zoom-lora \
    --lora-scale 1.1 \
    --out-dir /work/flux2_klein_merged
```

Then run with `--model-path /work/flux2_klein_merged`.

## When to choose this path vs native PyTorch

| Need | Use |
|---|---|
| Lowest-latency standalone FLUX.2-klein | [native-pytorch](../native-pytorch/) |
| FLUX.2-klein inside a multi-modal vllm-omni service | This (vllm-omni) |
| LoRA hot-swap at request time (vllm-omni multi-LoRA) | This (vllm-omni) |
| Trainium2 cost story for batch image gen | Either — both run on the same instance class |

## Known issues

1. **Compile cost is high** (~8-14 min depending on shape). NEFF cache
   persists across runs; container restart preserves it.
2. **Scheduler / unpack / VAE patches are out-of-tree** — they edit
   upstream `vllm_omni` source files in-container. Track upstream and
   re-apply when vllm-omni updates.
3. **No TP** for 4B — single-core fits comfortably. Larger FLUX models
   would need TP=2 with the same pattern as the Gemma4-E4B contrib.

## License

Apache-2.0 (contrib code). Model weights:
[FLUX.2 Community License](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B).

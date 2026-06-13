# FLUX.2-klein 4B on vllm-omni / Neuron — WIP status

**Date:** 2026-06-13
**Customer driver:** external image-to-image zoom LoRA on top of
FLUX.2-klein-4B base.
**Stack:** vllm-omni 0.19.0rc1 + vllm-omni-neuron plugin + diffusers 0.38.0,
torch 2.7.0, torch-neuronx, in container
`421672808698.dkr.ecr.us-east-1.amazonaws.com/concourse-release-1cb0647:pr-655cd3ee9b8d69818e52f37fbe1bb2a445bfbd60`.

## TL;DR

`NeuronFlux2KleinPipeline` written and registered. Engine boots,
weights load, dummy_run is short-circuited as designed, real warm-up
call enters `forward()`, FX graph is captured.

**Current blocker:** segfault inside
`PjRtComputationClient::ExecuteComputation` of the compiled graph,
because **`torch.polar` (complex64) survives Dynamo trace** despite
class-level + instance-level + module-level monkey-patches.

The torch.polar comes from diffusers' `get_1d_rotary_pos_embed(...,
use_real=False)` which `Flux2PosEmbed.forward` calls. Neuron Beta 3
release notes explicitly call out: complex64/128 not supported, use
real-valued RoPE replacements. We tried four ways to get rid of
`torch.polar` from the FX graph; none stuck (see "Patches that didn't
hold" below).

## What works

1. ✅ `neuron_flux2_klein_pipeline.py` imports cleanly
2. ✅ Auto-registers via `PIPELINE_REGISTRY` — overrides upstream
   `Flux2KleinPipeline`
3. ✅ Pipeline `__init__` builds correctly: encoder + tokenizer + VAE
   on CPU, `Flux2Transformer2DModel` on Neuron under
   `_NeuronTransformerWrapper`
4. ✅ Engine init in 23 s including weight load
5. ✅ `to(neuron_device)` only moves the transformer (encoder + VAE
   stay on CPU as designed)
6. ✅ `dummy_run` short-circuit triggers — no useless warmup compile
7. ✅ Real warmup call enters `forward()` — neither the dtype trap on
   `t.expand(...).to(latents.dtype)` nor any Python-level error stops it
8. ✅ Captured FX graph: `device_rewriter.rewrite_count: 9` —
   `_NeuronTransformerWrapper` is doing its job moving inputs

## Patches that DID hold (visible in FX graph diff)

These all show up correctly in `fxgraph_v6.txt` after the patch:

- ✅ Timesteps `torch.arange(device=neuron:0)` → `device=cpu`
  (the `time_proj` patch under `time_guidance_embed`)
- ✅ RoPE-axis `torch.arange(dtype=float64)` → `dtype=float32, device=cpu`
  (the `Flux2PosEmbed` swap with CPU+fp32 freq compute)

## Patches that DIDN'T hold (Dynamo unwrap)

Despite class-level + instance-level + module-level patches in
`_patch_pos_embed_to_cpu`:

- ❌ **`torch.polar` (complex64) — 16 calls in graph at 4-axis × 2 streams × 2 (img/txt)**

  Tried:
  1. Instance-level `pe.forward = _wrapped` (Dynamo unwrapped)
  2. Class-level `Flux2PosEmbed.forward = _patched_forward` (Dynamo unwrapped)
  3. Submodule swap `transformer.pos_embed = _NeuronFluxPosEmbed(...)` AND
     `transformer.rope_prepare.pos_embed = _NeuronFluxPosEmbed(...)`
     (Dynamo still inlined the original Flux2PosEmbed.forward)
  4. Module-level `diffusers.models.embeddings.get_1d_rotary_pos_embed`
     replacement (didn't trigger — Dynamo already had the original
     bytecode in its symbolic-eval table)

  All three "patched" / "swapped" log lines fire correctly in the worker
  process before the trace runs. The cache hash is byte-identical
  before vs after the patches — meaning Dynamo's symbolic eval is
  reproducing the original `Flux2PosEmbed.forward` regardless of any
  attribute or class assignment.

## Files

| File | Purpose |
|---|---|
| `neuron_flux2_klein_pipeline.py` | Neuron Flux2-klein subclass with all 12+ patches |
| `run_flux2_klein_omni.py` | Runner script (mirrors `run_ltx2_omni.py`) |
| `flux2_klein_stage.yaml` | Stage config (TP=1 default) |
| `merge_lora.py` | Offline LoRA-merge tool for the image-to-image zoom LoRA |
| `fxgraph_v6.txt` | Captured FX graph showing the surviving `torch.polar` calls |
| `FLUX2_KLEIN_OMNI_STATUS.md` | This file |

## Deepest issue (the root cause to fix next session)

vllm-omni's compile path uses Dynamo with a compile cache keyed on
**module bytecode**, not on instance attributes or class method
assignments. When we monkey-patch `Flux2PosEmbed.forward` (or swap
the submodule), Dynamo's symbolic evaluator still resolves to the
original `forward` body via the module's `__code__` attribute and
inlines it in the graph.

Three approaches that should work but each is more invasive:

1. **Patch upstream vllm-omni's `flux2_klein_transformer.py` directly**
   to take `is_npu = ids.device.type in ("npu", "xla", "neuron")` so
   it picks the float32 path. Quick & dirty; needs a PR upstream.

2. **Subclass `Flux2Transformer2DModel`** in our plugin and override
   the `forward()` method to call our pos_embed BEFORE the compile
   boundary. The transformer wrapper would precompute cos/sin on CPU
   and pass them as kwargs to a stripped inner forward.

3. **Compile the transformer in PIECES** — e.g., compile each
   `Flux2TransformerBlock` separately, with the rope-prep happening
   in eager Python between block calls. This breaks the fullgraph
   compile but eliminates the polar-in-graph problem because
   rope_prep never enters any compiled subgraph.

Option 1 is the smallest delta. Option 2 follows the pattern that
worked for LTX-2 native PyTorch (precompute, pass as input). Option 3
is the most robust but slower.

## Next session steps

1. **Try Option 1**: Edit
   `/opt/conda/lib/python3.12/site-packages/vllm_omni/diffusion/models/flux2_klein/flux2_klein_transformer.py`
   line 628 to add `or is_xla` to the `is_npu` check, then submit
   that as an upstream PR.
2. **If that fails** (e.g. arange-on-device trips first), fall back to
   Option 2: subclass `Flux2Transformer2DModel` with a forward that
   takes precomputed `(txt_cos, txt_sin, img_cos, img_sin)` as input
   kwargs and skips the `rope_prepare` call entirely.
3. **Validate**: tiny shape (256×256, 4 steps) → real PNG output.
4. **Then merge LoRA + production shape** (1024×1024, 28 steps,
   image-to-image zoom-LoRA).

## How to repro the segfault

```bash
# Inside vllm_omni container, with FLUX.2-klein-4B weights cached:
cd /workspace
HF_TOKEN=<token> python3 run_flux2_klein_omni.py \
    --tensor-parallel-size 1 --bench-runs 0 \
    --height 256 --width 256 --num-steps 4 \
    --output /work/flux2_klein_tiny.png
```

Reliably segfaults at `PjRtComputationClient::ExecuteComputation`
during the warm-up call.

After each segfault you must restart the vllm_omni container; the
unreaped subprocess zombies prevent Neuron Runtime from initializing
on the next attempt.

## Key takeaway

vllm-omni does support diffusion serving on Neuron via the plugin
pattern, but the compile path is Dynamo-based and doesn't honor
runtime monkey-patches the way native PyTorch + diffusers does
(which is the path Qwen-Image-Edit and LTX-2 native shipped on).

For models with operator gaps that need patching (complex64,
device-binding ops, etc.), the cleanest fix is to **edit the upstream
transformer source** in the vllm-omni package (or maintain a fork)
rather than monkey-patch.

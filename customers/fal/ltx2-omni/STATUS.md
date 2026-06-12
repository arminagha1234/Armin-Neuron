# LTX-2 19B on vLLM-Omni Trainium — current status

**Date:** 2026-06-12
**Container:** `vllm-omni-neuron Beta 1` — `concourse-release-1cb0647:pr-655cd...`
**Box:** `i-02a51e30b3a33408d` (us-east-2, trn2.48xlarge), `3.150.135.217`

## Where we are

| Stage | Status |
|---|---|
| Pull Beta 1 Omni image | ✅ done (14.7 GB compressed → ~44 GB on disk) |
| Move docker storage to NVMe | ✅ done (`/opt/dlami/nvme/docker`) |
| Container `vllm_omni` running | ✅ up, all 16 Neuron devices visible |
| HelloWorldPipeline smoke | ✅ passes (`NEURON_SKIP_EFA_AFFINITY=1` required) |
| Wan2.2 dev-mode T2V | ✅ end-to-end, real .mp4 generated |
| Download `Lightricks/LTX-2` weights | ✅ done (~291 GB at HF cache, includes all variants) |
| Write `NeuronLTX2Pipeline` subclass | ✅ done — see `neuron_ltx2_pipeline.py` |
| Auto-register via `PIPELINE_REGISTRY` | ✅ overrides base `LTX2Pipeline` correctly |
| Encoders + VAE load from `from_pretrained()` | ✅ works |
| Transformer weights load via framework iterable | ✅ fixed (strip + re-prefix `transformer.`) |
| Stage `_dummy_run()` | ❌ **fails with dtype-cast error** |

## Current blocker

```
RuntimeError: Expected self.dtype() == dst.dtype() to be true, but got false.
```

Fires during `_dummy_run()` (the engine's warmup forward pass that
runs even with `skip_warmup: true` in the stage YAML). Symptom is the
same pattern Wan2.2 hits — Neuron's lazy backend rejects combined
`.to(device, dtype)` calls on Neuron tensors, the base LTX2 pipeline's
`forward()` / `encode_prompt()` / `prepare_latents()` / VAE decode
contains at least one such call site.

## What's needed to fix

1. Run with verbose Python trace to find the exact call site:
   ```
   python -c "import sys; sys.settrace(lambda f, e, a: print(f, e))" run_ltx2_omni.py ...
   ```
   Or just bisect by overriding pipeline methods one at a time and
   adding prints.

2. Override the failing method following the Neuron Wan pattern. Likely
   candidates (mirror the ones overridden in `neuron_wan_pipeline.py`):
   - `_decode_latents()` — split the `.to(dtype, device)` into two
     separate moves
   - `prepare_latents()` — force CPU generator, force bfloat16 in two
     steps
   - `encode_prompt()` — encode on CPU, move embeddings to Neuron
     after the dtype is final

3. Each iteration is a 2-3 min cycle (no full transformer compile yet
   because dummy run dies before compile is invoked). Once dummy runs
   clean, the first real call kicks off the transformer compile
   (~30-60 min for LTX-2 19B at TP=4).

## Files in this directory

- `neuron_ltx2_pipeline.py` — the Neuron subclass of `LTX2Pipeline`.
  Drop into `vllm_omni_neuron/diffusion/models/` to register.
- `run_ltx2_omni.py` — runner mirroring `examples/wan22/run.py`.
  Has a `--bench-runs` flag for warm timing once the pipeline works.
- `ltx2_stage.yaml` — Omni stage config. Currently set to TP=4 on
  devices 0-3.

## Key env vars (mandatory)

```bash
NEURON_SKIP_EFA_AFFINITY=1                       # without this, workers die at init
NEURON_USE_VANILLA_TORCH_XLA=1
TORCH_NEURONX_DISABLE_FALLBACK_EXECUTION=1
VLLM_SLEEP_WHEN_IDLE=1
NEURON_LOGICAL_NC_CONFIG=2
VLLM_NEURON_COMPILATION_TIMEOUT=3600
NEURON_SCRATCHPAD_PAGE_SIZE=2048
NEURON_CC_FLAGS="--model-type=transformer --optlevel 1"
```

## Quick reproduce

```bash
ssh ubuntu@3.150.135.217
sudo docker exec -e NEURON_SKIP_EFA_AFFINITY=1 \
  -e TOKENIZERS_PARALLELISM=false \
  vllm_omni bash -c \
  'cd /work/ltx2 && python run_ltx2_omni.py --dev --tensor-parallel-size 4'
```

## Next session plan

1. Add per-method print logging to the NeuronLTX2Pipeline overrides
   (encode_prompt, prepare_latents, decode_latents)
2. Run with `--dev` and capture which override fires before the
   dtype error
3. Fix that call site, repeat until dummy_run passes
4. First real call → transformer compile (~30-60 min)
5. First video out
6. Bench: warm time, vs the other agent's PyTorch comparison

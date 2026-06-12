# LTX-2 19B on vLLM-Omni Trainium — current status

**Date:** 2026-06-12
**Container:** `vllm-omni-neuron Beta 1` — `concourse-release-1cb0647:pr-655cd...`
**Box:** `i-02a51e30b3a33408d` (us-east-2, trn2.48xlarge), `3.150.135.217`

## TL;DR

LTX-2 19B is **blocked by a vendor-side bug in vLLM-Omni Beta 1**, not
by anything we wrote. The base `vllm_omni.diffusion.models.ltx2.LTX2Pipeline`
in this image is incompatible with the `diffusers==0.38.0` it ships with:

```
LTX2TextConnectors.forward() got an unexpected keyword argument 'additive_mask'
```

The vllm-omni base calls `self.connectors(prompt_embeds, additive_attention_mask, additive_mask=True)`,
but diffusers 0.38 has `LTX2TextConnectors.forward(hidden_states, attention_mask, padding_side, scale_factor)`.
No `additive_mask` parameter exists. This is a vendor internal API
mismatch that needs to be fixed in either vllm-omni-base or diffusers.

## Where we got to

| Stage | Status |
|---|---|
| Pull Beta 1 Omni image | ✅ |
| Container `vllm_omni` running | ✅ |
| HelloWorldPipeline + Wan22 dev mode | ✅ end-to-end |
| Download `Lightricks/LTX-2` weights (~291 GB) | ✅ |
| Write `NeuronLTX2Pipeline` subclass | ✅ |
| Auto-register via `PIPELINE_REGISTRY` | ✅ |
| Encoders + VAE load via `from_pretrained()` | ✅ |
| Transformer weights load via framework iterable | ✅ |
| `_get_gemma_prompt_embeds` override (CPU encoder) | ✅ |
| Mask dtype fix (cast on CPU before move) | ✅ |
| Stage initializes, all 4 TP workers ready | ✅ |
| First forward call to base `LTX2Pipeline.forward()` | ❌ vendor bug above |

## The exact failure

```
File ".../vllm_omni/diffusion/models/ltx2/pipeline_ltx2.py", line 947, in forward
    connector_prompt_embeds, connector_audio_prompt_embeds, connector_attention_mask = self.connectors(
        prompt_embeds, additive_attention_mask, additive_mask=True
    )
TypeError: LTX2TextConnectors.forward() got an unexpected keyword argument 'additive_mask'
```

**Diffusers 0.38 LTX2TextConnectors signature** (the one installed in the image):
```python
forward(self, text_encoder_hidden_states, attention_mask, padding_side='left', scale_factor=8)
```

**vllm-omni-base expects:** an `additive_mask=True` kwarg that doesn't exist.

## Three paths forward

### Path 1 — file the vendor bug (best outcome, slowest)
Open issue with AWS Neuron team for vLLM-Omni:
> "Beta 1 vllm_omni.diffusion.models.ltx2.LTX2Pipeline calls
> LTX2TextConnectors with additive_mask=True kwarg that doesn't exist
> in diffusers 0.38 (the version shipped in the same image)."

Wait for fix in Beta 2/3.

### Path 2 — override forward() to use diffusers' own LTX2Pipeline.__call__
Skip vllm-omni's broken pipeline_ltx2.forward() entirely. Instantiate
diffusers.LTX2Pipeline directly, plug in our Neuron-resident transformer
as a swap-in for `pipe.transformer`, and call `pipe(...)` as the diffusers
pipeline. This works around the vendor bug. ~half-day port.

Drawback: we no longer use vllm-omni's serving infrastructure (request
batching, async scheduling). But we DO produce real LTX-2 video on
Neuron, which is the customer-facing deliverable.

### Path 3 — wait for Beta 2 Omni
Check if a Beta 2/3 vllm-omni image has fixed this. The Beta 3 native
DLC (`concourse-release-0461d3b:latest`) doesn't have `vllm_omni`
installed at all, only `torch_neuronx`. So there's no Beta 3 Omni
image yet.

## Recommendation

Path 2 is fastest to a real LTX-2 video on Trainium today. Path 1 is
the right long-term fix.

## Files in this directory

- `neuron_ltx2_pipeline.py` — our subclass (overrides `__init__`, `to`,
  `_get_gemma_prompt_embeds`, `compile`, `load_weights`)
- `run_ltx2_omni.py` — runner mirroring `examples/wan22/run.py` with a
  `--bench-runs N` flag for warm timing
- `ltx2_stage.yaml` — Omni stage config (TP=4)
- `STATUS.md` — this file

## Validated works on this hardware

- Wan2.2-T2V-A14B dev mode generated a real .mp4 at
  `customers/fal/results/wan22_dev_first.mp4` — proves the vLLM-Omni
  serving path is fully functional for video diffusion on this Neuron
  setup.
- The bug above is **specifically in the LTX-2 pipeline glue inside
  vLLM-Omni Beta 1**, not in the runtime/scheduler/worker plumbing.

## Key env vars (mandatory)

```bash
NEURON_SKIP_EFA_AFFINITY=1                 # without this, workers die at init
NEURON_USE_VANILLA_TORCH_XLA=1
TORCH_NEURONX_DISABLE_FALLBACK_EXECUTION=1
VLLM_SLEEP_WHEN_IDLE=1
NEURON_LOGICAL_NC_CONFIG=2
VLLM_NEURON_COMPILATION_TIMEOUT=3600
NEURON_SCRATCHPAD_PAGE_SIZE=2048
NEURON_CC_FLAGS="--model-type=transformer --optlevel 1"
TOKENIZERS_PARALLELISM=false
```

## Reproduce

```bash
ssh ubuntu@3.150.135.217
sudo docker exec -e NEURON_SKIP_EFA_AFFINITY=1 \
  -e TOKENIZERS_PARALLELISM=false \
  vllm_omni bash -c \
  'cd /work/ltx2 && python run_ltx2_omni.py --dev --tensor-parallel-size 4'
```

The run will get to `_dummy_run()` and fail with the connectors error
above.

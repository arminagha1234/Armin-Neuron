# LTX-2 18.88B on Trainium2 — vLLM-Omni path

**Status: WIP / BLOCKED**

The vLLM-Omni `NeuronLTX2Pipeline` (registered as `model_arch="LTX2Pipeline"`)
is integrated but not producing correct output. See `native-pytorch/` for the
validated path.

## What works

- NeuronLTX2Pipeline loads and initializes (text encoder + connectors +
  transformer on Neuron TP=4, VAE on CPU)
- The BMM-SDPA replacement is injected at the omni transformer source level
- Full denoising loop runs and produces output frames

## What doesn't work

The omni runtime's shared-memory diffusion-stage dispatcher silently kills the
LTX-2 denoising process when any other diffusion model (e.g. FLUX.2-klein)
shares the same container. The omni runtime allows only ONE diffusion engine
per container. On the test box (`3.150.135.217`), an automated FLUX timing
benchmark loop runs continuously in the same `vllm_omni` container, preventing
any LTX-2 validation run from completing.

Additionally, even when the run does execute (before being killed), the
stock `F.scaled_dot_product_attention` on Neuron's compiled bf16 lazy backend
produces numerically wrong output (damped activations + outliers — documented
in `neuron/examples/LTX/BLUR_INVESTIGATION.md`). The BMM-SDPA fix was applied
but never successfully validated end-to-end because of the container
contention.

## Blocked on

1. A dedicated container / stage-slot for LTX-2 (no sharing with FLUX bench)
2. The BMM-SDPA fix needs end-to-end validation (noise_pred std should match
   CPU reference: 1.05 not 0.84, no ±11 outliers)
3. The vLLM-Omni framework may need a `forward_neuron` method in its SDPA
   backend dispatch (currently only `forward_cuda/xpu/npu/hip` exist;
   Neuron falls through to the abstract class)

## For now: use `native-pytorch/`

The native PyTorch path with `torchrun` + `torch_neuronx` + the ten-fix
recipe produces correct video output at TP=4. See
[../native-pytorch/README.md](../native-pytorch/README.md).

## Files

| File | Role |
|---|---|
| `src/neuron_ltx2_pipeline.py` | (placeholder) NeuronLTX2Pipeline wrapper |
| `src/run_ltx2_omni.py` | (placeholder) omni entrypoint |
| `src/ltx2_stage.yaml` | Stage config (TP=4, devices 0-3) |

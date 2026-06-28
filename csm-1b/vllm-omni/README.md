# CSM-1B — vLLM-Omni Neuron pipeline (CsmPipeline)

This is the **vLLM-Omni** path for serving Sesame CSM-1B text-to-speech on Trainium:
a `CsmPipeline` that plugs into the `vllm_omni_neuron` plugin alongside
`Wan22Pipeline` / `HelloWorldPipeline`, for the OpenAI-compatible
`/v1/audio/speech` serving endpoint.

## Status
- ✅ **Implemented + registered.** `src/csm_pipeline.py` implements the omni pipeline
  interface (`PIPELINE_REGISTRY`, `__init__/to/compile/load_weights/forward`) and
  registers in the plugin (verified):
  ```
  Registered diffusion model CsmForConditionalGeneration -> ...csm_pipeline.CsmPipeline
  Registered diffusion model CsmPipeline           -> ...csm_pipeline.CsmPipeline
  ```
  `forward(request) -> DiffusionOutput(output=<24kHz waveform>)`.
- ⚠️ **In-container execution pending a runtime fix.** The pipeline constructs in the
  omni beta container, but `forward` hits the container's **older torch_xla** rejecting
  CSM's int64 casts (RoPE `position_ids.float()`, attention-mask `.to(bool)`):
  `RuntimeError: Expected self.dtype() == dst.dtype()`. The **identical compute runs
  end-to-end and produces audio on the native-PyTorch Neuron beta (torch_xla 2.9)** —
  see `../native-pytorch/` for the validated, audio-producing path. So this is a
  runtime/version gap, not a pipeline-logic problem.

## How it works
CSM's `generate` can't be lowered to Neuron (int64 dynamic loop), so the pipeline keeps
the generate loop on host and **offloads the heavy modules to the NeuronCore** (Llama
backbone + Mimi codec); the tiny depth decoder stays on host. This mirrors
`Wan22Pipeline`'s host-orchestration + on-device-forward structure.

## To finish the omni serving path
1. Match the container's torch_xla to the native-PyTorch beta (torch_xla 2.9), OR
   patch the two int64 casts (mask `.to(bool)` and RoPE `position_ids.float()`) to be
   torch_xla-safe on the container's version.
2. Add fixed-shape bucketing for the backbone/depth forwards (the omni engine's
   compiled-graph path) so per-request latency is stable.
3. Serve: `vllm serve <csm-1b> --omni` and hit `/v1/audio/speech`.

## Files
- `src/csm_pipeline.py` — the `CsmPipeline` (drop into
  `vllm_omni_neuron/diffusion/models/`).

The validated, working audio path today is **[../native-pytorch/](../native-pytorch/)**
(offload-based generate + the one-command `generate_speech.py`). The offload logic
there is exactly what this pipeline wraps.

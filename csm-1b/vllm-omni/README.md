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
- ⚠️ **In-container execution blocked by a torch_xla 2.10 regression.** The pipeline
  constructs in the omni beta container, but `forward` fails because the container
  ships **torch_xla 2.10.0** (newer than the validated 2.9.0), which has multiple
  strictness regressions CSM trips: int64→float `.float()` (`_to_copy`), xla
  `torch.autocast` (missing `get_amp_supported_dtype`), `torch.cat` contiguity, and
  even `.contiguous()` on a slice raising `Expected self.is_contiguous()`. Per-line
  patching is futile once `.contiguous()` itself is broken. The **identical compute
  runs end-to-end and produces audio on torch_xla 2.9** (the native-PyTorch beta) —
  see `../native-pytorch/`. So this is a runtime-version gap, not pipeline logic.

## How it works
CSM's `generate` can't be lowered to Neuron (int64 dynamic loop), so the pipeline keeps
the generate loop on host and **offloads the heavy modules to the NeuronCore** (Llama
backbone + Mimi codec); the tiny depth decoder stays on host. This mirrors
`Wan22Pipeline`'s host-orchestration + on-device-forward structure.

## To finish the omni serving path
1. Run on a **torch_xla 2.9 runtime** (an omni beta image built on 2.9, or pin
   torch/torch_xla to 2.9 in the container after a vllm-omni 0.19 compat check). The
   container's torch_xla 2.10 has the regressions above; 2.9 is the validated runtime.
2. Add fixed-shape bucketing for the backbone/depth forwards (the omni engine's
   compiled-graph path) so per-request latency is stable.
3. Serve: `vllm serve <csm-1b> --omni` and hit `/v1/audio/speech`.

## Files
- `src/csm_pipeline.py` — the `CsmPipeline` (drop into
  `vllm_omni_neuron/diffusion/models/`).

The validated, working audio path today is **[../native-pytorch/](../native-pytorch/)**
(offload-based generate + the one-command `generate_speech.py`). The offload logic
there is exactly what this pipeline wraps.

# Cosmos-Predict2-2B — vLLM-Omni serving (WIP)

**Status:** WIP — see [`../native-pytorch/`](../native-pytorch/) for the
working standalone path.

## Plan

Port the Cosmos-Predict2-2B pipeline (`Cosmos2TextToImagePipeline` and
`Cosmos2VideoToWorldPipeline`) to vLLM-Omni for production serving. The
diffusers pipeline classes are auto-registerable; the porting fixes
identified in the native-pytorch path (`DummySafetyChecker`, torchvision
transforms shim, DiT-on-Neuron forward wrapper) are reusable.

## Target benchmark (matches GPU production preview)

| Workload | Target |
|---|---|
| Text2Image @ 1024² × 20 steps | TBD |
| Video2World @ 480×832 × 25 frames | TBD |

## What needs to land

- Register `Cosmos2TextToImagePipeline` + `Cosmos2VideoToWorldPipeline`
  with `PIPELINE_REGISTRY`.
- Verify the framework-level diffusion preprocessing path doesn't
  re-introduce the bf16-in-FX-graph quality regression seen on other
  diffusion ports (the native PyTorch path bypasses this).
- Port the dummy safety checker via the same shim; production
  deployments should swap in `cosmos_guardrail`.

## License

Apache-2.0.

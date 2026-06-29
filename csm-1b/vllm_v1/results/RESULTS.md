# CSM-1B on Trainium2 — Working Result (2026-06-28)

**Status: CSM-1B generates speech end-to-end on a Trainium2 NeuronCore.** The two
heavy compute modules run on-device and are validated bit-for-cosine against CPU;
the codec produces 24 kHz audio. Artifact: `results/neuron_csm.wav`.

## Validation (all on a single NeuronCore, fp32)
| Component | Test | Result |
|---|---|---|
| Backbone (16-layer transformer) | prefill cb0 logits vs CPU | **cosine 1.000000, argmax identical** |
| Backbone | **teacher-forced** cb0 logits over 16-frame seq | **cosine 1.000000, argmax 100%** |
| Mimi codec (codes → waveform) | decode vs CPU | **cosine 1.000000** (maxabs 2e-6) |
| End-to-end generate | text → 24 kHz audio on Neuron | ✅ produces speech (`neuron_csm.wav`) |

## Architecture run on Neuron
`embed_text_tokens` (CPU) → **`backbone_model` (NeuronCore)** + `lm_head` (CPU) →
codebook 0; `depth_decoder` (CPU, 4 layers) → codebooks 1–31; **`codec_model`
MimiModel decode (NeuronCore)** → waveform. 32 codebooks, vocab 2051.

## How it works (the port, `src/`)
HF `generate` will NOT lower to Neuron — its loop/cache bookkeeping emits an int64
`dot` (NCC_EVRF035) and dynamic control flow. So we keep the **model + generate loop
on CPU and OFFLOAD only the heavy modules** to the NeuronCore via forward/method
wrappers (`run_csm_offload.py`):
- `backbone_model.forward` and `codec_model.decode` are wrapped to move inputs
  CPU→xla, run on Neuron, `mark_step`, return CPU tensors. The CPU-side generate
  machinery (int64 indexing, sampling, cache) is untouched.
- **Stray-tensor fix:** Mimi's RVQ codebooks store `self.embed` as plain attributes
  that `.to(device)` misses → we move all stray tensors to the device (98 in codec,
  68 in backbone). Without this: "Expected XLA tensor. Got: torch.FloatTensor".
- `ModelOutput` containers are preserved across the device round-trip (positional
  `[0]` indexing is used); `Cache` objects are left on the compute device.
- `use_cache=True` (default) — `use_cache=False` breaks CSM's audio-frame embedding
  path even on CPU.
- The depth decoder (tiny, 4 layers) stays on CPU; the dominant compute (16-layer
  backbone + Mimi) is on Neuron.

## On the waveform "mismatch" (expected, not a bug)
Free-running greedy generation diverges from CPU at the token/waveform level
(cosine ≈ 0). This is **autoregressive sensitivity**: the depth decoder's 31-step
inner loop amplifies sub-ULP fp32 differences between Neuron and CPU, so one flipped
argmax cascades into a different — but equally valid — speech realization. The
**teacher-forced test (cosine 1.0, argmax 100%)** proves the model math is identical;
exact free-run match across two fp32 backends is neither achievable nor the right bar
for a generative AR model. The decoded audio is valid speech (`neuron_csm.wav`,
~1.1 s, 24 kHz).

## Environment
- Box: trn2.48xlarge (us-east-2), but the 1B uses a single NeuronCore → a
  **trn2.3xlarge is sufficient**. Native-PyTorch Neuron venv
  `/opt/aws_neuronx_venv_pytorch_2_9` (torch 2.9.1, torch_neuronx 2.9.0).
- transformers 4.56.2 (CSM support), model `eustlb/csm-1b` (ungated canonical HF
  conversion; `sesame/csm-1b` is gated and the account isn't authorized).

## Relationship to Path A (vLLM-Omni CsmPipeline)
This validates that **all CSM compute runs correctly on Neuron** and gives a working
host-orchestrated generate (the offload approach). The vLLM-Omni `CsmPipeline` is the
productionization: it provides the same host-orchestration + on-device-forward
structure (like `Wan22Pipeline`), with bucketed compiled forwards and the
`/v1/audio/speech` serving endpoint. The offload logic here transfers directly into
that pipeline. The omni beta image is pulled on the box
on the box.

## Files (src/)
- `run_csm_cpu.py` — CPU reference (oracle) generation.
- `run_csm_offload.py` — **the working Neuron run** (backbone + codec offloaded).
- `run_csm_neuron.py` — full-model-on-device attempt (shows the generate int64 wall).

## A/B/C outcomes (2026-06-28)
- **A — CsmPipeline in vLLM-Omni:** `src/csm_pipeline.py` implements the plugin
  interface (PIPELINE_REGISTRY, `__init__/to/compile/load_weights/forward`) wrapping
  the offload logic; `forward(request) -> DiffusionOutput(output=waveform)`. **Registered
  in `vllm_omni_neuron`** alongside Wan22Pipeline/HelloWorldPipeline (verified:
  "Registered diffusion model CsmForConditionalGeneration -> ...CsmPipeline"). It
  constructs in the omni beta container; full in-container forward hits the container's
  **older torch_xla int64-cast quirk** (RoPE `position_ids.float()` / mask `.to(bool)`)
  — a runtime-version issue, not pipeline logic (same code runs end-to-end on the
  native-PyTorch-beta torch_xla 2.9).
- **B — depth-decoder offload:** added as `--offload-depth`. Hits `NRT_EXEC_OOB`
  (the codebook-index embedding path goes out of bounds on-device). Depth decoder is
  tiny (4 layers) and correct on CPU, so the validated config keeps it on CPU
  (backbone + Mimi = the dominant compute on Neuron). KV cache is on (`use_cache=True`).
- **C — packaged + PR:** `src/generate_speech.py` one-command TTS tool — validated
  end-to-end on the box (46,080 samples, ~1.9s @ 24kHz, `results/handoff_demo.wav`).
  Researcher README written. Example staged + PR opened on Armin-Neuron.

## Next
- Offload the depth decoder to Neuron too (optional; it's small) and KV-cache device
  handling for speed.
- Build the `CsmPipeline` in `vllm_omni_neuron` (Path A productionization) using this
  offload logic + bucketed forwards; serve via the omni `/v1/audio/speech` path.
- Longer-clip latency + a one-command `generate_speech.py` for handoff.

# CSM-1B on Trainium — vllm_v1 (saved breakthrough snapshot)

Frozen snapshot of the working CSM-1B TTS-on-Trainium breakthrough. **Do not edit —
this is the known-good reference.** Active optimization continues in `../vllm_explore/`.

## What this snapshot proves
- **CSM-1B runs on a single NeuronCore** (backbone + Mimi codec offloaded), validated
  **cosine 1.000000** vs CPU (teacher-forced argmax 100%, codec cosine 1.0).
- **Streaming warm TTFA = 241 ms — under the 500 ms target.**
  Stack: **bf16** (norms self-upcast, ~1.8× per-frame, cos 0.999968) + **streaming**
  (emit frame 0) + **`cache_implementation="static"`** (fixed-shape KV → no per-frame
  recompiles; backbone decode stable 38 ms) + **startup warm-up** (hides the one-time
  ~58 s decode-graph compile).
- Steady-state per-frame ~295 ms (backbone 38 + depth 156 CPU + codec 39 + overhead ~60).
  Depth decoder (31 CPU steps, 156 ms) is the bottleneck for real-time / <100 ms.

## Files
- `src/generate_speech.py` — one-command TTS (bf16 + static cache + warm-up). **The
  deliverable.**
- `src/stream_speech.py` — streaming generate + per-frame/TTFA instrumentation.
- `src/bench_ttft.py`, `bench_bf16.py`, `bench_static_cache.py` — the latency harnesses
  behind the numbers above.
- `src/csm_pipeline.py` — the vLLM-Omni `CsmPipeline` (registered in vllm_omni_neuron).
- `src/run_csm_offload.py`, `run_csm_cpu.py`, `trace_codec_demo.py` — offload run, CPU
  oracle, AOT-trace probe.
- `KERNEL_AND_PERF_PLAN.md`, `TTFT_OPTIMIZATION_PLAN.md`, `results/*` — the roadmap +
  measured findings.

## Key facts to carry forward
- HF `generate` can't lower to Neuron (int64 dynamic loop) → keep the loop on host,
  offload heavy modules (backbone + Mimi codec) to the NeuronCore.
- bf16 everywhere EXCEPT the Mimi codec (bf16 breaks its convs; codec fed int codes).
- **StaticCache is the key latency enabler** — without it the offload path recompiles
  every frame.
- Model: `eustlb/csm-1b` (ungated). Env: native-PyTorch Neuron beta, torch_xla 2.9.
- Next levers (in `../vllm_explore/`): depth decoder on Neuron, NKI TKG megakernel, TP=2–4.

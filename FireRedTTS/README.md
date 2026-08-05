# Running FireRedTTS (v1) on AWS Trainium — native PyTorch

Guide to run **FireRedTTS v1** — an open-source, LLM-style zero-shot voice-cloning
text-to-speech system ([FireRedTeam/FireRedTTS](https://huggingface.co/FireRedTeam/FireRedTTS),
Apache-2.0) — on **AWS Trainium2** using the **native PyTorch backend (TorchNeuron:
`torch.device("neuron")` eager execution, no torch_xla)**.

FireRedTTS turns text + a short reference voice clip into 24 kHz speech in that voice.

> **Status (2026-08-04): the full pipeline runs on the NeuronCore.**
> - ✅ **Recommended config: compiled GPT on the NeuronCore, flow + vocoder on CPU** —
>   served warm this gives **~42 ms TTFT** and **~4 s constant** total response. Putting the
>   flow/vocoder on the NeuronCore too is *slower* (eager-dispatch-bound, and neither
>   compiles cleanly) — see [Where each module actually runs fastest](#where-each-module-actually-runs-fastest).
>   All modules CAN run on the NeuronCore (`--offload all`) and produce valid 24 kHz speech;
>   it's just not the fastest.
> - ✅ **GPT verified numerically faithful:** greedy decode matches the CPU reference
>   **64/64 tokens (100%)** (eager and compiled), last-position hidden bit-identical (cosine 1.0).
> - ✅ **BigVGAN vocoder** on Neuron — `--bucket` pads the mel to fixed lengths and avoids the
>   odd-length `NCC_ITEN406` conv failure — but eager exec is ~10 s and it won't compile, so
>   **CPU (~1.1 s) is faster**.
> - ✅ **Flow decoder** on Neuron — fixed a neuronx-cc crash on the conformer's rel-pos
>   attention (strided `add`) via `.contiguous()`; but it recompiles per length / some
>   lengths still crash, so **CPU (~1.4 s) is the stable choice**.
> - ✅ **GPT AR decoder** — an **on-device fixed-length KV cache** (`--gpt-mode kvcache`,
>   default): each step processes only the 1 new token, the per-layer cache stays resident
>   on the NeuronCore, and the past is left-padded to `--gpt-bucket`. Add **`--gpt-compile`**
>   to `torch.compile(backend="neuron")` the decode step (fuses the ~300 eager op-dispatches
>   into one graph). Verified numerically faithful — greedy matches CPU **64/64 tokens**
>   both eager and compiled. (A `recompute` mode, `use_cache=False`, is kept as a fallback.)
>
> **Decode latency (`--offload gpt`, warm):**
>
> | | eager | **`--gpt-compile` + `--warmup`** |
> |---|---|---|
> | per decode step | ~80 ms | **~14.5 ms** (≈5.5x) |
> | TTFT (prefill→1st token) | ~394 ms | **~44 ms** (≈9x) |
>
> The win comes from fusing the per-step op-dispatches: native-*eager* Neuron dispatches each
> aten op to the device individually (~300/step for the 30-layer forward ≈ 80 ms of launch
> overhead), while `torch.compile` traces the forward into a single NEFF (one launch). Cost:
> a one-time graph compile (~30 s/shape) — amortized by `--warmup` for a resident server.
>
> Only the tokenizer, sampling loop, and ECAPA-TDNN speaker encoder remain on CPU. The
> largest remaining fresh-run cost is that the flow decoder + vocoder **recompile per output
> length** (sampling varies the length each run) — see [next steps](#status--next-steps).
> APC (automatic prefix caching) is intentionally **not** used; the KV cache is per-request,
> in-process only.

---

## Architecture (what runs where)

```
text ──▶ tokenizer (tiktoken/whisper BPE)                              [CPU]
      ──▶ GPT-2 AR decoder  (30 layers, 1024 dim, generates audio codes) [NeuronCore] ✅
      ──▶ Token2Wav:
             flow-matching (conformer + CFM decoder, 10 steps) → mel     [NeuronCore] ✅
             BigVGAN vocoder (conv stack)          mel → 24 kHz waveform  [NeuronCore] ✅
reference .wav ──▶ ECAPA-TDNN speaker encoder → speaker embedding        [CPU]
sampling + generate loop bookkeeping                                     [CPU]
```

Checkpoints (HF `FireRedTeam/FireRedTTS`, ~7.3 GB): `fireredtts_gpt.pt`,
`fireredtts_token2wav.pt`, `fireredtts_speaker.bin` (+ an empty `config.json` — the real
config is the repo's `configs/config_24k.json`).

---

## Prerequisites

1. **Instance:** a `trn2.3xlarge` (1 Neuron device / 4 cores / 96 GB HBM is plenty).
2. **Native-PyTorch stack:** the standard Neuron DLAMI venv (`torch-neuronx`) is
   **torch_xla-based** and rejects `torch.device("neuron")`. Native TorchNeuron ships in
   the DLC image below. This is required for the "native PyTorch" path.

```
421672808698.dkr.ecr.us-east-1.amazonaws.com/concourse-release-0461d3b:latest
```

## Step 1 — Connect and pull the native-PyTorch image

```bash
ssh -i <your-key>.pem ubuntu@<instance-public-dns>

# ECR login (cross-account; works with a Neuron instance role or Isengard Admin creds)
aws ecr get-login-password --region us-east-1 \
  | sudo docker login --username AWS --password-stdin 421672808698.dkr.ecr.us-east-1.amazonaws.com

IMG=421672808698.dkr.ecr.us-east-1.amazonaws.com/concourse-release-0461d3b:latest
sudo docker pull "$IMG"
```

## Step 2 — Start the container

```bash
mkdir -p /home/ubuntu/firered
IMG=421672808698.dkr.ecr.us-east-1.amazonaws.com/concourse-release-0461d3b:latest
sudo docker run -d --privileged --name firered \
  -v /home/ubuntu/firered:/root/firered \
  --env NEURON_SKIP_EFA_AFFINITY=1 \
  --shm-size=8g "$IMG" sleep infinity
```
`NEURON_SKIP_EFA_AFFINITY=1` is required on a 3xl (no EFA). Verify the native device:
```bash
sudo docker exec firered python -c \
  'import torch, torch_neuronx; d=torch.device("neuron"); print((torch.ones(8,device=d)*2).sum().item())'  # -> 16.0
```

## Step 3 — Copy these scripts + set up the model

Copy `src/*` into `/home/ubuntu/firered/` (they appear at `/root/firered` in the container), then:
```bash
sudo docker exec firered bash -lc 'cd /root/firered && WORK=/root/firered bash setup_env.sh'
```
`setup_env.sh` clones the v1 code (the **`main`** branch), installs the pinned deps, and
verifies the neuron device. See the script for why each pin matters.

## Step 4 — Download the checkpoints

```bash
sudo docker exec firered bash -lc \
  'cd /root/firered && python download_model.py --out /root/firered/pretrained_models'
```

## Step 5 — Synthesize

```bash
sudo docker exec firered bash -lc 'cd /root/firered && \
  PYTHONPATH=/root/firered/FireRedTTS FIRERED_MODEL=/root/firered/pretrained_models \
  python run_fireredtts_neuron.py \
    --prompt-wav FireRedTTS/examples/prompt_1.wav \
    --text "Hello from Trainium." --lang en --no-tn \
    --offload all --out /root/firered/neuron.wav'
```
`--offload all` runs the GPT decoder, flow decoder, and vocoder on the NeuronCore. The
first call compiles each module's shapes for the chip (a few minutes); NEFFs are cached so
later calls skip most of it. Output is a 24 kHz `.wav`. `PYTHONPATH` is required because
the upstream `fireredtts` package has no `__init__.py` (namespace package). Tuning knobs:
`--offload` (subset like `vocoder,flow`), `--gpt-mode` (`kvcache` default / `recompute`),
`--gpt-bucket` (KV-cache / seq padding granularity), `--gpt-prefill-bucket` (prompt prefill
padding, for TTFT), `--gpt-compile` (fuse decode into one Neuron graph — ~5.5x faster
steps), `--warmup` (compile graphs on a throwaway run first, then measure/serve warm),
`--gpt-seqs` (candidate count; 7 = upstream quality, 1 = fastest), `--bucket` (vocoder mel).

**CPU reference (correctness oracle, runs anywhere):**
```bash
sudo docker exec firered bash -lc 'cd /root/firered && \
  PYTHONPATH=/root/firered/FireRedTTS FIRERED_MODEL=/root/firered/pretrained_models \
  python run_fireredtts_cpu.py --prompt-wav FireRedTTS/examples/prompt_1.wav \
    --text "Hello from Trainium." --lang en --no-tn --out /root/firered/cpu.wav'
```

---

## Offload strategy

FireRedTTS emits speech from an HF `.generate()` loop whose per-step bookkeeping uses
dynamic control flow that doesn't belong on the device. So `run_fireredtts_neuron.py`
keeps the generate loop, sampling, tokenizer and speaker encoder on CPU, and **moves
selected heavy submodules to `torch.device("neuron")`** (native eager), marshalling their
inputs/outputs CPU↔device. Pick modules with `--offload` (comma-separated):

| `--offload` | Runs on NeuronCore | Result |
|---|---|---|
| `vocoder` (default) | BigVGAN | ✅ `--bucket` pads the mel for NEFF reuse + to avoid `NCC_ITEN406` |
| `flow` | flow-matching decoder | ✅ needs the conformer rel-pos `.contiguous()` fix (auto-applied) |
| `gpt` | GPT-2 transformer forward | ✅ fixed-shape: `use_cache=False` + left-pad to `--gpt-bucket` |
| `all` | all three | ✅ full pipeline on the NeuronCore |

Each fix is applied automatically in `firered_patch.py` when the corresponding module is
selected — see [Offload strategy](#offload-strategy).

**How each module is made to fit fixed-shape Neuron execution:**

- **BigVGAN vocoder** (`_offload_vocoder_bucketed`) — pad the input mel time-dim up to a
  multiple of `--bucket` frames, run, then trim the output waveform (`hop = 240` samples
  per mel frame). Only a few shapes compile, and the odd-length `NCC_ITEN406` strided-conv
  failure is avoided.
- **Flow decoder** (`patch_flow_conformer_contiguous`) — the conformer's relative-position
  attention builds `matrix_bd` from `rel_shift`, a heavily strided *view*; adding it to the
  contiguous `matrix_ac` crashed neuronx-cc (internal error on the strided `add`). Forcing
  `rel_shift(...).contiguous()` materializes it first and it lowers cleanly.
- **GPT-2 AR decoder** — HF `.generate()` grows the KV cache one token per step, so the
  transformer forward changes shape every step (a recompile each step). Two fixed-shape
  fixes (`--gpt-mode`), both keeping `gpt.wte` (the shared `mel_embedding`) on CPU since the
  transformer only ever sees `inputs_embeds`:
  - `kvcache` (`patch_gpt_kv_cache_bucketed`, default): keep `use_cache=True` so each step
    processes only the 1 new token; the per-layer `(k,v)` cache **stays resident on the
    NeuronCore** (never marshalled to CPU) and is **left-padded to `--gpt-bucket`** before
    each call, then the returned present is un-padded to its true length for HF's
    bookkeeping. The pad mask is built with an `arange` comparison — an early version used
    an in-place `mask[:, :padn] = 0`, whose dynamic slice size triggered a **recompile every
    step** past the first bucket (3 s/step); the `arange` form fixed that. The prompt prefill
    uses a smaller `--gpt-prefill-bucket` to keep TTFT cheap.
  - `recompute` (`patch_gpt_fixed_shape`): `use_cache=False` + **left-pad the whole sequence
    to `--gpt-bucket`**. Real tokens right-aligned so `logits[:, -1]` is still the true last
    token. Simpler, but O(n²).
  Both are numerically safe (`gpt.wpe` is a no-op, positions baked into `emb`) — verified
  64/64 (kvcache) and 48/48 (recompute) greedy-token match vs CPU.

## Measured (trn2.3xlarge, single NeuronCore, native torch-neuronx)

| Run | Output samples | Wall time | Result |
|---|---:|---:|---|
| CPU reference | 60,480 (2.52 s) | 18.6 s | ✅ all on CPU (oracle) |
| Neuron `vocoder --bucket 64` (cold) | 73,920 (3.08 s) | 142 s | ✅ compiles the 320-frame bucket |
| Neuron `vocoder --bucket 64` (warm) | 72,000 (3.00 s) | **20.3 s** | ✅ NEFF reused |
| Neuron `flow` (cold) | 72,960 (3.04 s) | 100.7 s | ✅ conformer fix |
| Neuron `gpt --gpt-seqs 1` | 77,760 (3.24 s) | 28.0 s | ✅ fixed-shape AR |
| Neuron `all --gpt-seqs 1` (warm) | 72,960 (3.04 s) | **21.0 s** | ✅ full pipeline, cached |
| Neuron `all --gpt-seqs 7` | 69,120 (2.88 s) | 114 s | ✅ full quality (batch-7 compile) |

**GPT decode latency (`--offload gpt`, warm):**

| Metric | `kvcache` eager | `kvcache` **`--gpt-compile --warmup`** | `recompute` |
|---|---|---|---|
| TTFT (prefill → 1st token) | ~394 ms | **~44 ms** | ~0.2 s |
| per decode step | ~80 ms, flat | **~14.5 ms, flat** | ~75 ms @128, O(n²) |
| one-time compile | ~3 s/bucket | ~30 s/shape (amortized by `--warmup`) | ~3 s/bucket |

The compiled decode is ~5.5x faster per step and ~9x lower TTFT; the tradeoff is a larger
one-time graph compile, which a resident server pays once via `--warmup`.

**Correctness:** GPT greedy decode matches CPU **64/64 (kvcache) / 48/48 (recompute) tokens
(100%)**; padded-vs-unpadded last-position hidden is bit-identical (cosine 1.0). All outputs
are valid speech (peak 1.0, RMS ~0.18, not silence). With sampling on (`do_sample=True`),
Neuron and CPU waveforms differ run-to-run — expected.

### Where each module actually runs fastest

Putting *everything* on the NeuronCore is **not** fastest here. Native-*eager* Neuron
dispatches each op individually, which is slow for the big conv stacks — and neither the
flow nor the vocoder `torch.compile`s cleanly. Measured, per module:

| Module | eager Neuron | CPU | `torch.compile` on Neuron | Best |
|---|---|---|---|---|
| GPT decode step | ~80 ms | (loop on CPU) | **~14.5 ms** | **Neuron + compile** |
| Flow decoder | ~1.4 s + recompiles / some lengths crash | **~1.4 s, stable** | crashes (GroupNorm + shape) | **CPU** |
| BigVGAN vocoder | ~10–13 s (dispatch-bound) | **~1.1 s** | fails (`NCC_ITIN902`) | **CPU** |

### Everything on the NeuronCore, end-to-end (measured, greedy/fixed length)

You asked for the full pipeline on the NeuronCore. Measured with GPT (kvcache+compile) +
flow (compiled) + vocoder (eager, bucketed) all on device, greedy/deterministic length so
the warm run reuses graphs (6.4 s of audio):

| stage | cold (compiles) | **warm** |
|---|---:|---:|
| TTFT | 29.9 s | **46 ms** |
| GPT decode (159 steps) | 17.5 s | 2.35 s (~14.8 ms/step) |
| flow (compiled) | 314 s | **0.98 s** |
| vocoder (eager) | 184 s | **19.5 s** |
| **total** | **572 s** | **24.3 s** |

Takeaways: (1) TTFT is still ~46 ms warm. (2) The **compiled flow is actually fast warm
(0.98 s — beats CPU's ~1.4 s)**, but takes ~314 s to compile and recompiles per length, so
it's only viable with fixed-length decode. (3) The **eager vocoder (19.5 s) dominates** and
can't be compiled (`NCC_ITIN902`). So all-on-Neuron warm is **~24 s vs ~4 s for the hybrid**
(GPT-on-Neuron + flow/vocoder-on-CPU) — the hybrid is ~6x faster, entirely because of the
eager vocoder. That's why the recommended config keeps flow + vocoder on CPU.

The fast, **constant-latency** config is: **compiled GPT on the NeuronCore, flow +
vocoder on CPU**. Flow bucketing was tried and **abandoned** — the flow's GroupNorm
normalizes over the whole (padded) time axis, so padding corrupts the real region
(measured mu cosine 0.9965, not exact); making it correct needs masked GroupNorm throughout
(a large rework). The vocoder can't be compiled (`NCC_ITIN902`) and is 10x slower eager
than on CPU, so it stays on CPU.

## Serving (recommended) — `serve_fireredtts.py`

A resident server that loads once, compiles + **warms up** the GPT graph, and serves so
every request runs on the hot path. Recommended config = **GPT compiled on Neuron, flow +
vocoder on CPU** (`--offload gpt --gpt-compile`, the default).

```bash
sudo docker exec -d firered bash -lc 'cd /root/firered && \
  PYTHONPATH=/root/firered/FireRedTTS FIRERED_MODEL=/root/firered/pretrained_models \
  python serve_fireredtts.py --offload gpt --gpt-compile --warmups 2'
# then (localhost only, no auth — dev/benchmark server):
curl 'http://127.0.0.1:8000/tts?text=Hello%20from%20Trainium&out=/root/firered/out.wav'
curl 'http://127.0.0.1:8000/health'
# STREAMING endpoint — audio arrives per sentence as it's ready (see Streaming below):
curl -sN 'http://127.0.0.1:8000/tts_stream?text=First%20sentence.%20Second%20sentence.' \
  | ffplay -f s16le -ar 24000 -ch_layout mono -nodisp -autoexit -   # raw s16le PCM
curl -sN 'http://127.0.0.1:8000/tts_stream?text=First.%20Second.&format=ndjson'  # per-chunk timing
```

Measured warm, per request (trn2.3xlarge), text "Hello from Trainium today":

| | value |
|---|---|
| **TTFT (GPT prefill → 1st token)** | **~42 ms** (consistent) |
| GPT decode (~70 steps) | ~1.0 s |
| flow (CPU) | ~1.45 s |
| vocoder (CPU) | ~1.1 s |
| **total response** | **~4.0 s, constant** (no per-request recompiles) |

That's down from ~13–21 s when flow+vocoder were forced onto the NeuronCore. Security: the
server binds `127.0.0.1` with no auth — local dev/benchmark only; do not expose it.

## Streaming (sentence-level) — `stream_fireredtts.py`

FireRedTTS v1 is **not** a natively streaming model — its flow-matching decoder (a
full-context conformer + GroupNorm over the whole time axis + a CFM ODE over the whole mel)
and the BigVGAN vocoder are non-causal and run on the whole utterance. But v1's own
`synthesize()` already splits text into sentence-sized chunks (`text_split`, merged to
>~30 chars) and synthesizes each chunk **independently**, then concatenates. So we can
deliver each chunk's audio the moment it's ready — the concatenated output is identical to
non-streaming `synthesize()`; only the *delivery* is incremental. GPT stays compiled on the
NeuronCore; flow+vocoder on CPU (the recommended hybrid).

```bash
sudo docker exec firered bash -lc 'cd /root/firered && \
  PYTHONPATH=/root/firered/FireRedTTS FIRERED_MODEL=/root/firered/pretrained_models \
  python stream_fireredtts.py --prompt-wav FireRedTTS/examples/prompt_1.wav \
    --offload gpt --gpt-compile --warmup --no-tn --out stream.wav'
```

**Measured (trn2.3xlarge, warm, 5-sentence passage, ~21.9 s of audio):**

| chunk | synth | (GPT-Neuron / token2wav-CPU) | audio | ready@ |
|---|---:|---|---:|---:|
| 1 | 11.8 s | 5.1 s / 6.7 s | 5.6 s | 11.8 s |
| 2 | 10.8 s | 4.6 s / 6.2 s | 5.0 s | 22.6 s |
| 3 | 8.5 s | 3.7 s / 4.9 s | 4.0 s | 31.1 s |
| 4 | 9.4 s | 4.1 s / 5.3 s | 4.5 s | 40.5 s |
| 5 | 6.9 s | 3.1 s / 3.8 s | 2.9 s | 47.4 s |

- **TTFA (time-to-first-audio): 11.8 s vs 47.4 s non-streaming — ~4.0x sooner.** The win
  scales with passage length (TTFA ≈ one chunk; non-streaming ≈ all chunks).
- **Warm the vocoder at a representative length.** The first token2wav call at a *new* mel
  length pays a large one-time CPU oneDNN conv primitive-selection cost (measured 20–50 s)
  that otherwise dominates TTFA. `--warmup` runs a full-length throwaway sentence to prime
  those primitives (and compile the GPT graphs), cutting chunk-1 from ~50 s to ~12 s.
- **This is progressive delivery, not real-time streaming.** Per-chunk synthesis is
  ~7–11 s for ~3–5 s of audio (**rtf ~2.2**), dominated by CPU flow+vocoder (~4–7 s) and the
  batch-7 GPT decode (~3–5 s). Since rtf > 1, gapless playback of a long passage eventually
  underruns — you just get the *first* audio ~4x sooner. For a single short sentence there
  is only one chunk, so streaming == non-streaming.

**Over HTTP:** `serve_fireredtts.py` exposes the same thing at `/tts_stream?text=...` — raw
24 kHz mono `s16le` PCM by default (pipe straight to a player), or `&format=ndjson` for one
JSON event per chunk (metadata + base64 PCM + per-chunk `ready_ms`/`ttfa_ms`). The server
runs `--gpt-seqs 1` (single candidate), so GPT is only ~1.5 s/chunk and CPU token2wav
dominates. Measured client-side over the wire (2-sentence request): first audio chunk
received at **7.3 s** (3.8 s of playable audio) vs **13.7 s** to receive the whole clip. The
`--warmup`/`--warmups` passes use a full-length sentence so the first request doesn't pay
the token2wav primitive-selection penalty.

**Why not sub-sentence (true low-latency) streaming on v1?** The flow can't be safely
chunked below a sentence: its conformer encoder attends over the full token sequence, the
length regulator's GroupNorm normalizes over the whole time axis (chunking corrupts the real
region — measured mu cosine 0.9965), and the CFM decoder solves an ODE over the whole mel.
The v1 conformer *does* carry WeNet chunk-attention machinery (inherited from CosyVoice),
but the v1 checkpoint isn't trained for dynamic-chunk inference. **True low-latency streaming
is what FireRedTTS-1S is for** — a separate model (semantic tokenizer + semantic LM +
acoustic LM + BigCodec causal codec + DiT flow) with its own checkpoints and the
`fireredtts-1s` branch. Porting 1S is a whole separate effort.

## Gotchas (all handled by the scripts)

| Symptom | Cause / fix |
|---|---|
| Cloned code doesn't match v1 (has `models/`, streaming) | Default branch is `fireredtts-1s`. Use `git clone --branch main`. |
| `ModuleNotFoundError: fireredtts` after `pip install -e .` | Top-level pkg has no `__init__.py`; use `PYTHONPATH=.../FireRedTTS`. |
| `No module named 'tn'` / pynini won't build | WeTextProcessing needs pynini (no cp312 wheel). Use `--no-tn` (lite normalizer; spell out numbers). |
| `cannot import name 'cached_download'` | `diffusers==0.27.2` needs `huggingface_hub==0.25.2`. |
| `TorchCodec is required for load_with_torchcodec` | torchaudio 2.11 I/O → torchcodec. Scripts route `load`/`save` through `soundfile`. |
| `deserialize object on a CUDA device ... is False` | `speaker.py` `torch.load` has no `map_location`; scripts default it to CPU. |
| `No such file: fireredtts/modules/flow/codebook.npy` | Config uses a repo-relative path; scripts `chdir` to the repo root. |
| `Expected ... device type at start of device string: neuron` | You're on the torch_xla DLAMI venv, not the native DLC image. |
| Vocoder `torch.compile` → `NCC_ITIN902 TensorInitialization` | BigVGAN doesn't compile on Neuron; run the vocoder on CPU (~1.1 s vs ~10 s eager-Neuron). |
| Full pipeline slower than expected on Neuron | flow+vocoder are faster on CPU; use `--offload gpt --gpt-compile`, not `all`. |

## Status & next steps

The recommended config (compiled GPT on Neuron, flow+vocoder on CPU, served warm) gives
~42 ms TTFT and ~4 s constant total. Remaining wins:
- **Compile the flow cleanly** — `torch.compile(backend="neuron")` on the flow *runs* and
  produces valid audio, but `make_pad_mask` does `max_len = lengths.max().item()`, a
  data-dependent `.item()` that forces **dynamo graph breaks** (so it's many subgraphs, not
  one NEFF, and no clear win over CPU's ~1.4 s). Rewriting `make_pad_mask` to take an
  explicit `max_len` (no `.item()`) + masked GroupNorm (plain input bucketing corrupts the
  real region — measured mu cosine 0.9965 because GroupNorm normalizes over the padded time
  axis) would allow a clean fixed-shape compiled flow. Sizeable rework.
- **A compilable vocoder path** — BigVGAN `torch.compile` fails with `NCC_ITIN902`
  (TensorInitialization) at every bucket tried (128/256/512) — an architecture-level
  compiler limitation, not a size one. **NKI kernel — DONE for the SnakeBeta activation:**
  `src/snakebeta_nki.py` is a device-validated NKI kernel for BigVGAN's signature SnakeBeta
  (`x + (1/β)·sin²(αx)`, per-channel). On device (`torch_neuronx.wrap_nki`) it matches the
  PyTorch reference to **max_diff 1.1e-5 (allclose) over the real ±91 input range**, and is
  **1.8× faster than eager-Neuron and 39× faster than CPU** for the op (1.29 ms vs 2.32 /
  49.9 ms at `[512, 32768]`). Three device facts cracked getting there: (1) per-channel
  scales must use `tensor_scalar` with a `[P,1]` operand — a per-partition `activation(scale=)`
  is wrong on device; (2) **`nl.sin` only stays accurate for |x| ≲ 4.18 and diverges past it**
  (that was the real cause of earlier "garbage", not a wrap_nki bug), so the kernel
  **range-reduces** `α·x` into [-π, π] via `x − 2π·round(x/2π)` — and since there is no
  floor/mod ISA op, a float→int32→float `tensor_copy` (which rounds half-to-even) supplies
  `round`; (3) `wrap_nki` multi-arg calls work fine. **Still open:** SnakeBeta is one op —
  the full vocoder also needs the transposed-conv upsampling, dilated-conv resblocks, and
  anti-aliasing filter convs as NKI kernels, and stitching one activation kernel between
  CPU convs adds 109 CPU↔device round-trips. So for the end-to-end pipeline the **CPU
  vocoder (1.1 s) is still the pragmatic choice**; the kernel is a validated building block
  for a future full-vocoder port.
- **Streaming** — sentence-level streaming is implemented + measured
  (`stream_fireredtts.py`): ~4x lower time-to-first-audio on a multi-sentence passage
  (11.8 s vs 47.4 s), see [Streaming](#streaming-sentence-level--stream_fireredttspy). It's
  progressive delivery (rtf ~2.2), not real-time. True sub-sentence low-latency streaming
  needs the FireRedTTS-1S model (a separate causal semantic→acoustic decoder — separate
  port).
- **APC stays off** by design — the KV cache here is per-request/in-process, not a
  cross-request prefix cache.

## Files (`src/`)

- `setup_env.sh` — clone + install everything inside the container (pinned).
- `download_model.py` — fetch the 4 checkpoints from HF.
- `firered_patch.py` — device-agnostic + Neuron-enabling patches: honor `self.device`,
  CPU `torch.load`, soundfile audio I/O, WeTextProcessing bypass, `patch_flow_conformer_contiguous`
  (flow fix), `patch_gpt_kv_cache_bucketed` (on-device KV cache + TTFT stats), and
  `patch_gpt_fixed_shape` (recompute fallback).
- `run_fireredtts_cpu.py` — CPU reference (correctness oracle).
- `run_fireredtts_neuron.py` — native-PyTorch Neuron run; `--offload`, `--gpt-compile`,
  `--vocoder-compile`, `--warmup`, `--gpt-mode`, bucketing flags.
- `serve_fireredtts.py` — resident warm server (recommended config: compiled GPT on
  Neuron, flow+vocoder on CPU); ~42 ms TTFT, ~4 s constant total. `/tts` (one-shot),
  `/tts_stream` (per-sentence PCM or NDJSON), `/health`. Localhost, no auth.
- `stream_fireredtts.py` — sentence-level streaming: yields each sentence's audio as it's
  ready (~4x lower time-to-first-audio on multi-sentence text). Reuses the stock per-chunk
  `synthesize_base`, so output matches non-streaming exactly. See [Streaming](#streaming-sentence-level--stream_fireredttspy).
- `snakebeta_nki.py` — NKI kernel for BigVGAN's SnakeBeta activation. Device-validated
  (allclose vs PyTorch, max_diff 1.1e-5) with sin range-reduction; 1.8× vs eager-Neuron,
  39× vs CPU. Invoke via `torch_neuronx.wrap_nki`. See [next steps](#status--next-steps).

## Credits & license

Model: [FireRedTeam/FireRedTTS](https://huggingface.co/FireRedTeam/FireRedTTS) (Apache-2.0),
code [github.com/FireRedTeam/FireRedTTS](https://github.com/FireRedTeam/FireRedTTS) (`main`
branch = v1). Builds on Tortoise-TTS / XTTS-v2 (AR), CosyVoice/Matcha-TTS (flow-matching),
and BigVGAN-v2 (vocoder). Content rephrased from public sources for compliance.

# Model Landscape (Aug 2026)

Researched snapshot of the top open-weight models per modality, used to seed
the leaderboard tracks. **Point-in-time — the discovery job (see `plan.md`
phase 3) replaces this with live weekly data.** Kept here as the initial
hand-curated baseline and as a record of what we knew when we designed the
tracks.

Sources are linked per row. Where sources disagree on parameter counts, both
figures are noted — verify against the actual model config before sizing
instances.

## Important distinction: seed models vs. top models

These are **different lists** and conflating them causes bad decisions.

| | Purpose | Selection criteria |
|---|---|---|
| **Seed models** (phase 1) | Prove the optimizer loop works | Small enough to iterate fast, permissive license, architecturally diverse, fits trn2.3xlarge |
| **Top models** (phase 3) | Publish competitive recipes | Whatever the community and customers actually care about, regardless of size |

Our phase-1 seeds (Gemma 4 31B, Muse Glimmer 30B, Qwen3.8-27B) are good
*seeds*. They are **not** the top open models by capability — the leaders are
much larger MoEs. That is fine and intentional: you do not debug a new
optimizer loop on a 2.8-trillion-parameter model.

---

## Track A: Text-to-text (LLM)

### Current capability leaders

| Model | Params | License | Notes |
|-------|--------|---------|-------|
| [**Kimi K3**](https://github.com/MoonshotAI/Kimi-K3) | 2.8T total / 104B active (16 of 896 experts) | **Custom "Kimi K3 License"** | Largest open-weight model shipped. Kimi Delta Attention (KDA) + Attention Residuals. Native vision, 1M context. Scores 57 on Artificial Analysis Intelligence Index. |
| [**GLM-5.2**](https://codersera.com/blog/glm-5-2-complete-guide-2026/amp/) | 744B total / ~40B active | **MIT** | 1M context. 62.1% SWE-bench Pro — highest open-weight coding score. ~3x throughput of peers. |
| [**DeepSeek V4 Pro**](https://deepinfra.com/blog/kimi-k3-vs-deepseek-v4-pro-vs-glm-5-2) | not confirmed (V4-Flash is 284B) | **MIT** | 93.5% LiveCodeBench — #1 globally, open or closed. CSA+HCA backbone. |
| **Inkling** (Thinking Machines) | not confirmed | open weights | Strongest US-built open-weight model, 41 on AA Index |
| [**Gemma 4 31B**](https://blog.google/innovation-and-ai/technology/developers-tools/gemma-4/) | 31B dense | **Apache 2.0** | 85.2% MMLU-Pro. #3 open model on Arena text. 256K context. Multimodal. |
| [**Muse Glimmer 30B**](https://research.meta.ai/blog/introducing-muse-glimmer-open-agentic-model) | 30B dense | **Apache 2.0** | Meta's first open-weight since Llama 4. Agentic, perception encoder. No MAU cap. |
| **Qwen3.8-27B** | 27B | **Apache 2.0** | Gated DeltaNet hybrid attention |
| **Qwen3.6-27B** | 27B dense | **Apache 2.0** | 1M context |

### Licensing spread — matters for the leaderboard filter

The 2026 releases span a wider license range than previous years:

- **MIT**: GLM-5.2, DeepSeek V4 — cleanest terms
- **Apache 2.0**: Gemma 4, Muse Glimmer, Qwen3.x — clean, standard attribution
- **Custom with conditions**: Kimi K3 — revenue-triggered separate agreement
  for Model-as-a-Service operators, UI attribution mandate above 100M MAU
- **Territory-restricted**: MiniMax H3 — reportedly excludes US, EU, UK,
  Korea. If accurate, unusable for us entirely.

The discovery job's license filter must handle all four classes, not just
"is it on HuggingFace."

---

## Track B: Text-to-image

| Model | Params | License | Strength |
|-------|--------|---------|----------|
| [**FLUX.2 / FLUX.1 [dev]**](https://www.thundercompute.com/blog/best-open-source-image-generation-models) | 12B (FLUX.1 dev) | non-commercial for [dev] — check per variant | Prompt adherence, photorealism |
| [**Qwen-Image**](https://localaimaster.com/blog/best-local-image-models-compared) | 20B MMDiT | Apache 2.0 | Readable text *inside* images |
| [**Z-Image / Z-Image-Turbo**](https://www.mindstudio.ai/blog/what-is-z-image-turbo-qwen) | small (exact TBC) | open | Fast, small, bilingual text. #25 overall on Arena T2I, **top among open models** |
| **SDXL 1.0** | 3.5B | OpenRAIL | Deepest LoRA/style ecosystem |
| **Hunyuan Image 3** | TBC | custom | Competitive quality |

Note: FLUX variant licenses differ substantially ([dev] vs [schnell] vs
[pro]). Check per checkpoint, not per family.

---

## Track C: Text-to-video

| Model | Params | License | Strength |
|-------|--------|---------|----------|
| [**LTX-2.5**](https://ltx.io/blog/open-source-video-generation-models-guide) | 22B | check | Video **and audio in a single forward pass**, separate decoder per stream — audio arrives pre-aligned. Current production-grade standout. |
| [**Wan 2.2**](https://localaimaster.com/blog/local-ai-video-generation) | TBC | Apache 2.0 (2.1 was) | Best pure silent-video quality on a 24GB card |
| [**HunyuanVideo 1.5**](https://ltx.io/blog/best-open-source-video-generation-models) | TBC | custom | Full-attention transformer, joint video+text |
| [**Wan 2.1 1.3B**](https://www.turingpost.com/p/6-opensource-video-generation-models) | 1.3B | Apache 2.0 | Low-VRAM entry point — 8.19 GB for 5s @ 480p. Good smoke-test model. |
| CogVideoX / Open-Sora / AnimateDiff | various | various | Older generation, still referenced |
| MiniMax H3 (Hailuo 3.0) | 33B? / 195.9 GiB bf16? | **territory-restricted** | Omni-modal, native audio. **Parked — see license note above.** |

Wan 2.1 1.3B is worth calling out as the *video track's* equivalent of a
seed model: small enough to iterate the loop quickly before touching a 22B
DiT.

---

## Track D: Speech (ASR / STT)

| Model | Params | License | Strength |
|-------|--------|---------|----------|
| [**Whisper large-v3**](https://www.assemblyai.com/blog/top-open-source-stt-options-for-voice-applications) | 1.55B | MIT | Most versatile all-rounder, 99+ languages |
| [**NVIDIA Canary-Qwen 2.5B**](https://northflank.com/blog/best-open-source-speech-to-text-stt-model-in-2026-benchmarks) | 2.5B | check (NVIDIA terms) | Best English accuracy on leaderboards |
| **NVIDIA Parakeet TDT** | TBC | check | Fastest batch throughput; ultra-low-latency streaming |
| **IBM Granite Speech 3.3 8B** | 8B | Apache 2.0 | Enterprise English ASR + translation |
| **Whisper large-v3 Turbo / Distil-Whisper** | smaller | MIT | Much faster throughput |
| **Moonshine** | tiny | MIT | Edge / mobile |

## Track E: Text-to-speech (TTS)

| Model | Params | License | Strength |
|-------|--------|---------|----------|
| [**Kokoro-82M**](https://localaimaster.com/blog/best-local-tts-models) | 82M | Apache 2.0 | Lightweight winner — ~2-3 GB VRAM, runs on CPU |

TTS is the thinnest track. Worth including for completeness but low priority
versus A-D.

---

## Recommended track priority

| Priority | Track | Why |
|----------|-------|-----|
| 1 | Text-to-text | Biggest Trainium commercial driver, most models, richest optimization space |
| 2 | Text-to-image | Mature open ecosystem, fast iteration (seconds per image), clear metrics |
| 3 | Text-to-video | High compute demand = strong Trainium value story, but slow iteration |
| 4 | Speech (ASR) | Well-defined metric (WER), small models, quick wins |
| 5 | TTS | Thin field, smallest models |

Start with Track A only. Add Track B once the loop is proven, because image
generation forces us to build non-token benchmark shapes and non-token
equivalence — which validates that the framework generalizes beyond LLMs.

## Phase-1 seed set (unchanged by this research)

| # | Model | Track | Why this one |
|---|-------|-------|--------------|
| 1 | Gemma 4 31B | A | Standard dense transformer, Apache 2.0, 256K context exercises `stress` |
| 2 | Muse Glimmer 30B | A | Apache 2.0, dense + perception encoder, agentic tuning |
| 3 | Qwen3.8-27B | A | Gated DeltaNet — forces the hybrid-attention adapter |

All three fit trn2.3xlarge at bf16. All three are Apache 2.0. Deliberately
*not* the capability leaders — those are 744B-2.8T MoEs and belong in phase
3, not in a loop we are still debugging.

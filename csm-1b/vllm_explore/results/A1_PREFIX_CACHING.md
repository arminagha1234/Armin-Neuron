# A1 — Prefix caching probe for CSM-1B (RESULT: negative, pivot to Tier B)

**Question:** gemma4 got TTFT 0.83→0.42s by caching repeated prefill context. CSM is
conversational, so does caching the dialogue history collapse per-turn TTFA — or is
CSM's TTFA dominated by the per-frame floor regardless of context?

**Method:** `src/multiturn_ttft.py`. Grow a single-speaker conversation context from
1→6 turns and time the full `generate(max_new_tokens=1)` call (prefill + ONE frame =
backbone-step + 31-step depth loop; `output_audio=False`, StaticCache, bf16 backbone,
best-of-3 warm). The slope of first-frame latency vs context tokens isolates the
prefill-growth cost that prefix caching would eliminate.

## Result

| ctx_turns | ctx_tokens | frame1_ms |
|---|---|---|
| 1 |  25 | 216.6 |
| 2 |  45 | 250.4 |
| 3 |  65 | 251.5 |
| 4 |  85 | 253.0 |
| 5 | 105 | 251.9 |
| 6 | 125 | 250.7 |

## Verdict: prefix caching does NOT help CSM TTFA

- After the one-time **25→45 token jump (+34ms)** — a StaticCache prefill-shape/bucket
  padding artifact, not real prefill growth — first-frame latency is **dead flat at
  ~250–253ms** while context grows 45→125 tokens (≈3×).
- True marginal prefill cost from 45 tokens on is **~0 ms/token** (the headline
  "+0.341 ms/token" the script prints is an artifact of including the single 25→45
  bucket step in the slope; the 45→125 segment is flat).
- Extrapolated: even a **1000-token** dialogue history would add only ~tens of ms of
  prefill, versus the **156ms depth loop paid on every single frame**.

CSM's TTFA floor is **per-frame compute (backbone decode step + 31-step depth decoder),
not prefill**. This is structurally different from gemma4, whose bottleneck was the
prefill/decode collective floor that prefix caching shortcuts. So gemma4's #1 lever does
not transfer to CSM.

## Decision
- **Drop Tier A (prefix caching)** as a TTFA lever for CSM. (Caching still has value for
  throughput/compute saved on long histories, but it does not reduce interactive TTFA.)
- **Skip A2 (prefill buckets)** for TTFA too — prefill is already a negligible fraction
  of first-frame latency. One small static prefill bucket is enough.
- **Pivot to Tier B — the depth decoder (156ms/frame).** This is the elephant. Next:
  - **B1.** Depth decoder on Neuron + StaticCache (fix the `NRT_EXEC_OOB` codebook-index
    embedding path) — get the 31 serial CPU steps onto the device.
  - **B2.** NKI TKG megakernel on backbone + depth steps to collapse per-op dispatch.
  - **B3.** Parallel / speculative codebook decoding to amortize the 31 serial steps.
- **A3 (TP=2–4 backbone)** remains worthwhile but secondary — it shaves the 38ms backbone
  step, not the 156ms depth loop.

## Repro
```bash
CSM_MODEL=<csm_1b path> python src/multiturn_ttft.py --turns 6 --words-per-turn 20
```
Reference baseline: `../vllm_v1` (warm streaming TTFA 241ms, steady ~295ms/frame).

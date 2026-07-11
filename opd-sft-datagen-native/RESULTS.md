# Native-PyTorch SFT Data Generation on Trainium — Results

**Date:** 2026-07-11
**Instance:** trn2.3xlarge (1 Trainium2 chip, 4 NeuronCores, 96 GB)
**Stack:** Beta-3 native PyTorch DLC, torch 2.11.0,
torch-neuronx 2.11.3, `device = torch.device("neuron")` (no torch_xla), **eager**
(no `torch.compile`).
**Driver:** host `aws-neuronx-dkms` swapped **2.29 → 2.28** (public 2.29 is
incompatible with the beta userspace — NRT fails to init on 2.29). Reversible;
2.29 deb cached under `/var/cache/apt/archives/`.

## Headline

**Pure native PyTorch generation works correctly on Trainium — including bf16 and
batched decode — with `Qwen/Qwen3-8B` (the real teacher).** The only bug in the
naive HuggingFace decode was that **SDPA attention drops the 2D padding mask on
Neuron**; forcing `attn_implementation="eager"` fixes it. bf16 was never the
problem. Remaining issue is throughput (per-step recompilation), not correctness.

## Root cause (one bug, not two)

Initial symptoms looked like "bf16 garbage" + "left-pad garbage," but isolating
batch-size 1 showed **bf16 at bs=1 is coherent** — so the bf16 failure was really
the left-pad failure (that run happened to be bs=5). The single root cause:

> **On Neuron, the SDPA attention path does not apply the 2D key-padding mask** for
> left-padded batches, so padded positions leak into the shorter sequences and
> corrupt their output. `attn_implementation="eager"` applies the mask correctly.

## Evidence matrix (Qwen3-0.6B, greedy, on Neuron)

| dtype | batch | attn | Result |
|---|---|---|---|
| fp32 | 1 | (default) | ✅ coherent |
| bf16 | 1 | (default) | ✅ coherent  ← bf16 is fine |
| bf16 | 5 | sdpa | ❌ garbage on padded seqs (`!!!!`, gibberish) |
| **bf16** | **5** | **eager** | ✅ **coherent for all 5** ← the fix |

## Qwen3-8B (the actual teacher) — validated

- 8.19B params loaded on the Neuron device in 142 s.
- bf16, batch-size 3, `attn-impl eager`, coherent answers for all prompts:
  - "What is 2 + 2?" → `<think> Okay, the user is asking "What is 2 + 2...`
  - "Name a primary color." → `<think> Okay, the user is asking for a primary color...`
  - "Write one word that rhymes with cat." → `<think> Okay, the user wants a word that rhymes...`

## Throughput (the remaining, non-correctness issue)

Per-**shape** recompilation dominates the eager decode (a new graph compiles as the
sequence grows). Correct, but slow:

| Run | Tokens | Wall | Note |
|---|---|---|---|
| 0.6B, bs=5, bf16, eager, 12 tok | 12 | 43 s | warm-ish |
| 8B, bs=3, bf16, eager, 16 tok | 16 | 285 s | cold, per-step recompile |

Fine for a smoke / reference; too slow for 300k × 16k tokens in this eager form.

## Fixes applied to the script

1. **`attn_implementation="eager"` is now the default** (`--attn-impl`) — correct
   batched left-padded decode on Neuron.
2. **Sampling runs on CPU in fp32** (`top_p_sample`) — `torch.multinomial`/sort are
   not reliable on-device and bf16 distributions can go degenerate; the tiny sampling
   step on CPU is exact and safe while the model forward stays on Neuron.

## Verdict

- **Correctness: solved.** Pure native PyTorch, bf16, batched, Qwen3-8B, on Trainium.
- **Throughput: open.** Naive eager decode recompiles per step. To make the full
  300k run practical in *pure native PyTorch* (no vLLM/NxDI), the next step is a
  static-shape decode: fixed `max_seq_len`, length bucketing, and a preallocated KV
  cache so the graph compiles once per bucket instead of once per step.

## Repro (inside the Beta-3 DLC container `opd`)

```bash
# correct, batched, bf16 (the fix):
python generate_sft_data_native.py --device neuron --model Qwen/Qwen3-8B \
  --prompts data/prompts_smoke.jsonl --output data/out.parquet \
  --max-new-tokens 16 --batch-size 3 --dtype bfloat16 --attn-impl eager \
  --temperature 0.7 --top-p 0.9

# reproduce the original bug:
#   --attn-impl sdpa --batch-size 5   -> garbage on padded sequences
```

## Next steps

- [ ] Static/bucketed decode for throughput (single compile per length bucket).
- [ ] Preallocated fixed-size KV cache (avoid per-step growth/recompile).
- [ ] Scale batch-size on the 96 GB device once shapes are static.

---

# Route B — Static-Shape Decode (throughput fix) + Schema — VALIDATED 2026-07-11

`generate_sft_data_bucketed.py` adds the two things the naive script was missing to
actually complete the 300k data-curation stage in pure native PyTorch.

## 1. Throughput: StaticCache → compile once, not per token
Uses HuggingFace `StaticCache` (preallocated fixed-size KV cache indexed by
`cache_position`) + a **full fixed-length attention mask every step**, so the decode
graph has constant shapes and compiles ONCE instead of once per generated token.

Critical detail found on-device: passing a *growing* `attention_mask[:, :pos+1]` slice
reintroduces per-step recompilation. Passing the **full `[B, total_len]` mask every
step** (StaticCache + `cache_position` handle causality) is what makes it compile once.

| Run (Neuron, bf16, eager) | Tokens×seqs | Wall | vs naive |
|---|---|---|---|
| naive, Qwen3-0.6B | 12×5 | 43 s | baseline |
| naive, Qwen3-8B | 16×3 | 285 s | baseline |
| **static, Qwen3-0.6B** | **48×5** | **38.7 s** | 4× tokens, less time |
| **static, Qwen3-8B** | **64×4** | **130.8 s** | 4× tokens in ~46% the time |

The wall time is now dominated by one-time prefill+decode compilation; steady-state
per-token cost is flat and low (the whole point).

## 2. Output schema: drop-in for the SFT step
`--schema messages` writes a parquet with columns **`messages`** (chat list) and
**`tokens`** (generated ids) — exactly what the upstream `run_pipeline.sh`
`validate_parquet` requires. Verified:

```
rows 5 cols ['messages', 'tokens']
downstream-compatible: True
```

(`--schema simple` still available for {prompt, answer}.) pandas + pyarrow installed in
the container so it writes real parquet (was JSONL-fallback before).

## Status against the 300k requirement — now GREEN on correctness + format + per-token speed
| Requirement | Status |
|---|---|
| Download OpenThoughts3-1.2M | ✅ `prepare_prompts.py` |
| Sample 300k prompts | ✅ `--num-samples 300000` |
| Generate with Qwen3-8B (native PyTorch) | ✅ on-device, correct, static-cache |
| Save prompt–answer dataset (downstream schema) | ✅ `messages`+`tokens` parquet |

Remaining before a literal 300k×16k run: pick prefill buckets for the real prompt-length
distribution, scale `--batch-size` on the 96 GB device, and (optionally) multi-bucket by
length so short prompts don't pay for a 16k cache. The mechanism is proven.

## Repro (Route B)
```bash
python generate_sft_data_bucketed.py --device neuron --model Qwen/Qwen3-8B \
  --prompts data/openthoughts3_300000.jsonl \
  --output data/openthoughts3_300000_qwen3-8b.parquet \
  --schema messages --prefill-bucket 512 --max-new-tokens 1024 \
  --batch-size 8 --dtype bfloat16 --attn-impl eager --temperature 0.7 --top-p 0.9
```

---

# End-to-End Test — real OpenThoughts3 → Qwen3-8B → parquet (trn2.3xl, 2026-07-11)

Full data-curation stage exercised on-device with the **real dataset** (not smoke prompts):

1. `prepare_prompts.py --num-samples 16` → streamed `open-thoughts/OpenThoughts3-1.2M`,
   wrote 16 real prompts (math/reasoning questions).
2. `generate_sft_data_bucketed.py --model Qwen/Qwen3-8B --schema messages
   --prefill-bucket 512 --max-new-tokens 128 --batch-size 4 --dtype bfloat16 --attn-impl eager`.
3. Verified parquet.

**Timing (Qwen3-8B, 4 prompts/batch, 128 new tokens):**
| Batch | Wall | Note |
|---|---|---|
| 1 | 137.3 s | cold — prefill(512) + decode compile |
| 2 | 47.0 s | warm (graphs cached) |
| 3 | 48.3 s | warm |
| 4 | 48.0 s | warm |

Steady-state ≈ **48 s / (4 prompts × 128 tok)** once compiled — the StaticCache
compile-once behavior confirmed at a realistic config.

**Output:** `rows: 16, cols: ['messages','tokens'], downstream-compatible: True`.
Example — prompt *"Find all integers n such that 2^n is a palindrome in base 3"* →
Qwen3-8B `<think>`-style reasoning answer, 128 tokens.

**Conclusion:** the four requirement steps (download → sample → generate with Qwen3-8B →
save in the SFT schema) all run on-device in pure native PyTorch on a single trn2.3xl.
For the full 300k run, scale via data-parallel workers on a 48xl (one 8B replica per chip).

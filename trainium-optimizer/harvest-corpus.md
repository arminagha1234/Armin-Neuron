# Harvest Corpus

Survey of the `aws-neuron` GitHub org, and the design of a **Stage 0.5:
Harvest** that mines working code before any optimization is attempted.

## Why this stage exists — a concrete case

The `internal-prior-optimization-run` run spent Round 3 hand-building
**Context Parallel**: splitting prior KV cache across ranks and merging with an
online softmax reduction.

`nki-library` already ships this:

> **Attention KV Parallel Segmented CTE Kernel** — implements KV-parallel
> segmented prefill attention **with online softmax merging for context
> parallelism**.

Same for the all_gather elimination work. The library ships:

> **FGCC Kernel** — implements fused all-gather and matrix multiplication
> (Fine-Grained Gather Collective Compute)
>
> **Fine-Grained AllGather Kernel** — fine-grained ring-based all-gather

A harvest step would have surfaced these on minute one. Whether the library
versions would have won on that specific model is unknown — but they should
have been the *first* thing tried, not the thing reinvented in hour eleven.

**That is the argument for making harvest a formal stage rather than an
assumption.** Your instinct here is right.

## The corpus (aws-neuron org, surveyed 2026-08-18)

### Tier 1 — kernel sources, harvest these first

| Repo | Stars | What it gives us |
|------|-------|-----------------|
| [`nki-library`](https://github.com/aws-neuron/nki-library) | 69 | **The production kernel library.** Bundled into `neuronx-cc` as the `nkilib` namespace — much of it is importable with zero install. See inventory below. |
| [`nki-samples`](https://github.com/aws-neuron/nki-samples) | 71 | Sample kernels, broader and more pedagogical than nki-library |
| [`nki-moe`](https://github.com/aws-neuron/nki-moe) | 48 | **MLSys competition for the best MoE NKI kernels.** A corpus of *competitively optimized* MoE kernels. Direct gold for any MoE model. |
| [`nki-llama`](https://github.com/aws-neuron/nki-llama) | 21 | NKI kernels developed specifically for Llama 3.2 1B inference — a worked end-to-end example |
| [`nkipy`](https://github.com/aws-neuron/nkipy) | 30 | Rapid prototyping on Trainium. Useful for Stage 4 iteration speed. |
| [`torch2nki`](https://github.com/aws-neuron/torch2nki) | 3 | Scaling NKI kernels within PyTorch — relevant to the native-PyTorch path |
| `nki-synthesizer` | 0 | Kernel synthesis. Dormant since Apr 2025, but worth a look given our Stage 4. |

### Tier 2 — serving stacks (the optimization target)

| Repo | Stars | Notes |
|------|-------|-------|
| [`upstreaming-to-vllm`](https://github.com/aws-neuron/upstreaming-to-vllm) | 25 | Public vLLM-Neuron |
| `vllm-neuron` | 22 | Private dev repo, updated 2026-08-15 — very active |
| [`neuronx-distributed-inference`](https://github.com/aws-neuron/neuronx-distributed-inference) | 40 | NxDI — the autoport target |
| [`torch-neuronx`](https://github.com/aws-neuron/torch-neuronx) | 29 | The native-PyTorch path |
| [`neuronx-distributed`](https://github.com/aws-neuron/neuronx-distributed) | 68 | Distributed primitives |
| [`torchtitan-neuron`](https://github.com/aws-neuron/torchtitan-neuron) | 6 | PyTorch-native training platform |
| `transformers-neuronx` | 112 | **Deprecated — do not use.** NAD house rules prohibit it. |

### Tier 3 — examples and reference material

| Repo | Stars | Notes |
|------|-------|-------|
| [`aws-neuron-samples`](https://github.com/aws-neuron/aws-neuron-samples) | 162 | Largest example corpus — inference and training |
| [`neuron-workshops`](https://github.com/aws-neuron/neuron-workshops) | 48 | Build On Trainium notebooks |
| [`aws-neuron-sdk`](https://github.com/aws-neuron/aws-neuron-sdk) | 626 | SDK + canonical docs |
| [`neuron-agentic-development`](https://github.com/aws-neuron/neuron-agentic-development) | 50 | The NAD agents/skills we already have locally |
| `neuronx-distributed-training` | 13 | Training-side patterns |
| `deep-learning-containers` | 23 | The DLC images, including the native-PyTorch one |

## `nki-library` inventory

Worth enumerating in full, because this *is* our Stage-2 candidate set. Every
one of these is a pre-built kernel we should try before writing anything.

### Production kernels

| Kernel | Relevance |
|--------|-----------|
| Attention CTE | Core prefill attention, multiple variants |
| **Attention KV Parallel Segmented CTE** | **Context parallelism with online softmax merging** |
| Attention TKG | Decode-optimized attention |
| MLP | With optional norm fusion |
| MoE CTE / MoE TKG | Prefill and decode MoE |
| Output Projection CTE / TKG | |
| QKV | With optional norm fusion |
| RMSNorm-Quant | RMSNorm + fp8 quantization fused |
| RMSNorm MX Prefill | RMSNorm + MX quant + optional router top-K, token-major |
| RoPE | With optional LNC sharding |
| Router Top-K | MoE routing |
| Cumsum | |

### Experimental kernels (higher risk, high value)

Selected highlights:

| Kernel | Why it matters |
|--------|---------------|
| **FGCC** | Fused all-gather + matmul. Addresses the exact collective bottleneck auto_research attacked by hand. |
| **Fine-Grained AllGather** | Ring-based all-gather |
| **Ring Attention Forward / Backward** | Sequence-parallel attention, online softmax reduction |
| Attention Block TKG | Megakernel: RMSNorm + QKV + RoPE + output projection fused |
| Transformer TKG | Full transformer forward megakernel for decode |
| MXFP8 family (Matmul, MLP, MoE Bwd, Quantize) | Low-precision path |
| Blockwise MM Backward | Dropless MoE |
| SSD / Selective Scan / Linear Scan | **Mamba-style state-space kernels** — directly relevant to Gated DeltaNet (Qwen3.5/3.8) and Kimi K3's KDA |
| **NeuroTile** | Tile-iterator library abstracting HBM/SBUF/PSUM tiling, sharding, access patterns. A Stage-4 authoring aid. |
| GpSIMD Top-K | Uses `nisa.topk` |
| Depthwise Conv1D / Conv3D family | Vision paths |
| Gather / Scatter-Add | Indirect DMA patterns |

Two notes with real consequences:

1. **Bundled vs. package.** The compiler ships a validated `nkilib` internally.
   The standalone `pip install nki-library` gives newer kernels but is *not*
   guaranteed compatible with the current compiler — pin to the branch matching
   your compiler version. `NKILIB_FORCE_BUNDLED_LIBRARY=true` reverts.
   **Our toolchain stamp must record which of the two was used**, or results
   are not reproducible.
2. **The SSD / Selective Scan / Linear Scan kernels matter for our seed set.**
   Qwen3.8-27B uses Gated DeltaNet — a linear-attention recurrence. These
   Mamba-style scan kernels are the closest existing primitives. That
   materially de-risks seed model #2.

## Stage 0.5: Harvest

Inserted between Stage 0 (baseline) and Stage 1 (config). It writes nothing
and compiles nothing — it builds the **candidate inventory** that Stages 2-4
draw from.

```
Stage 0    BASELINE      autoport output, verified + measured
              |
              v
Stage 0.5  HARVEST       mine working code; build candidate inventory
              |          (no compiles, no edits — pure discovery)
              v
Stage 1    CONFIG        ...
```

### What it does

1. **Classify the model.** Architecture family, param count, attention type,
   MoE or dense, target shapes.
2. **Profile the baseline** to get the op inventory — which ops exist, which
   are hot. (Cheap: one profile run on the `fast` tier.)
3. **Match ops against the corpus**, in priority order:

   | Priority | Source | Rationale |
   |----------|--------|-----------|
   | 1 | Knowledge bank `verified/` | Already proven *on Neuron, by us*, with shape constraints recorded |
   | 2 | Bundled `nkilib` | Compiler-validated, zero install risk |
   | 3 | `nki-library` package (pinned branch) | Newer kernels, compatibility caveat |
   | 4 | `nki-moe` (MoE models only) | Competition-grade MoE kernels |
   | 5 | `nki-samples`, `nki-llama` | Worked examples, may need adaptation |
   | 6 | `aws-neuron-samples`, `neuron-workshops` | Config patterns and full pipelines more than kernels |
   | 7 | External refs (vLLM, SGLang, TRT-LLM, FlashAttention) | Stage 3's corpus — patterns, not drop-ins |

4. **Emit a harvest manifest** — the inventory, ranked, with shape-constraint
   checks already applied.

### The manifest

```yaml
harvest:
  model: google/gemma-4-31B
  family: dense_causal_lm
  profiled_ops:
    - op: attention_prefill
      cost_share: 0.47          # 47% of step time
      candidates:
        - source: bank_verified
          id: gqa-paged-attention-headdim128
          shape_match: exact
          prior_evidence: "2 models, 1.4x"
          priority: 1
        - source: nkilib_bundled
          id: nkilib.attention_cte
          shape_match: exact
          priority: 2
        - source: nkilib_bundled
          id: nkilib.attention_kv_parallel_segmented_cte
          shape_match: needs_cp_enabled
          note: "context parallelism w/ online softmax merge"
          priority: 2
        - source: external
          id: vllm::paged_attention
          shape_match: pattern_only
          priority: 7
          defer_to_stage: 3
    - op: rmsnorm
      cost_share: 0.08
      candidates:
        - source: nkilib_bundled
          id: nkilib.rmsnorm_quant
          shape_match: exact
          note: "fuses quant — only if fp8 path is enabled"
          priority: 2
  unmatched_ops:
    - op: gated_delta_recurrence
      cost_share: 0.31
      note: >
        No exact match. Closest primitives: nkilib.selective_scan,
        nkilib.linear_scan, nkilib.ssd (Mamba-family). Flag for Stage 3
        adaptation or Stage 4 invention.
```

`unmatched_ops` is the important output — it is the **pre-computed Stage 3/4
work queue**, derived from evidence rather than from the agent wandering into
kernel work and guessing where to start.

### Why harvest before config

Counterintuitive, since config is cheaper. Two reasons:

1. **Harvest is free.** No compiles, no edits. Pure reads plus one profile run
   we need anyway.
2. **It shapes the config search.** Discovering that
   `attention_kv_parallel_segmented_cte` exists means `cp_degree` becomes a
   *meaningful* config axis. Without that knowledge, Stage 1 would sweep
   `cp_degree` against a kernel that cannot exploit it and conclude context
   parallelism does not help.

### Guardrails for this stage

- **Read-only.** Harvest must not modify anything. Its output is a manifest.
- **Time-boxed to 30 minutes.** Corpus search is bounded; it is a lookup, not
  a research project.
- **Record corpus versions** — commit SHAs for each repo consulted, plus
  whether bundled or packaged `nkilib`. Harvest results go stale exactly as
  bank lessons do.
- **Every harvested candidate that gets promoted emits a lesson** with
  `origin: harvested` and the source repo + commit. This is a fourth
  provenance value alongside `borrowed` / `invented` / `hybrid`.

## Provenance taxonomy, revised

Stage 0.5 adds a category that is meaningfully distinct from the others:

| `origin` | Meaning | Effort | Risk |
|----------|---------|--------|------|
| `harvested` | Used an existing AWS-maintained kernel as-is | Minimal | Low — maintained upstream |
| `borrowed` | Ported a pattern from an external reference (vLLM, etc.) | Medium | Medium — new code, proven algorithm |
| `hybrid` | Borrowed algorithm, substantially restructured for Neuron | High | Medium-high |
| `invented` | Novel, designed from profile + roofline | Highest | Highest |

Reporting all four separately keeps the "did it invent anything" metric
honest. A run that is 90% `harvested` is a *good outcome* — it means the
ecosystem already had the answer — but it is not invention, and conflating the
two would be self-flattering.

Expect the distribution to shift over time: early runs heavily `harvested`,
later runs pushing into `borrowed` and `invented` as the easy inventory is
exhausted.

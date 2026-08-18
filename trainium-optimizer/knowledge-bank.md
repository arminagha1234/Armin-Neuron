# Knowledge Bank

The memory layer. Without it, every model pays full search cost. With it,
marginal cost of model N+1 drops sharply after ~20 well-covered ones.

Prior art we're taking from: TVM's AutoTVM cost-model store, cuDNN's
autotune cache, AutoML meta-learning warmstart. Nothing new conceptually —
the interesting engineering is what shape the lessons take and how they're
indexed.

## What a "lesson" is

A structured entry, not a note. Every lesson has all of these fields
because free-text "TP=8 works" becomes technical debt fast.

```yaml
# lesson-id: dense-llm-32b-tp-baseline
type: config_prior
applicability:
  architecture_family: dense_causal_lm
  parents_ok: [llama, qwen, mistral, phi]
  param_count_range: [20e9, 40e9]
  seq_len_range: [512, 8192]
  batch_range: [1, 32]
  neuron_sdk_versions: ["2.26.*", "2.27.*", "2.28.*"]
intervention:
  kind: config_delta
  spec:
    tp_degree: 8
    weights_dtype: bf16
    activations_dtype: bf16
    attention_kernel: gqa_paged
    batching: continuous
evidence:
  measurements:
    - model: qwen2.5-32b
      instance: trn2.48xlarge
      sdk: 2.28.0
      throughput_tps: 4200
      before_this_change_tps: 2100
      timestamp: 2026-08-14T00:00:00Z
    - model: llama-3.1-70b     # different size but same family — noted
      instance: trn2.48xlarge
      sdk: 2.28.0
      throughput_tps: 1800
      before_this_change_tps: 1100
      timestamp: 2026-09-01T00:00:00Z
confidence:
  n_models_validated: 2
  architecture_diversity: 1     # only dense LLMs; not diverse yet
  human_verified: true
provenance:
  authored_by: aghaebra
  from_optimizer_run: null      # null when hand-authored; else run id
  source_references:
    - https://awsdocs-neuron.readthedocs-hosted.com/en/latest/...
expiration:
  reverify_on_sdk: any_minor_bump
  last_reverified: 2026-09-01
```

### Types of lessons

Six types, each with a slightly different shape but the same fields above.

| Type | What it says | Example |
|------|--------------|---------|
| `config_prior` | Start with these settings for this class of model | dense LLM ≥32B → TP=8, BF16, batch 32 |
| `op_rewrite` | Apply this graph transformation when this pattern is seen | RMSNorm→attention fusion when N > 512 |
| `nki_kernel` | Use this custom NKI kernel for this op signature | GQA paged attention kernel, head_dim=128 |
| `anti_pattern` | Never do X for architecture Y — save the compile | TP=16 on Trn2 → weight spill, 3× slower |
| `reference_translation` | vLLM/SGLang/TRT-LLM's X maps to Neuron's Y | vLLM `paged_attention` → NKI pattern in `neuron/…` |
| `equivalence_tolerance` | Family-specific numerical tolerance advice | Diffusion needs looser rtol than dense LLM |

### Layer tagging and migration risk

Every lesson records **which layer of the stack it lives at**, because that
determines whether it survives a backend migration.

This matters concretely: vLLM-Neuron today is built on XLA. When the native
PyTorch (TorchNeuron) path matures and the serving stack migrates onto it,
some lessons carry over unchanged and some die. Tagging up front means we can
query for exactly what needs re-verification instead of re-testing everything.

```yaml
layer: kernel | collective | framework | config | graph
backend_validated:
  - vllm-neuron-xla
  # - native-pytorch      # appended once re-verified post-migration
migration_risk: low | medium | high
```

| `layer` | What it covers | Survives backend migration? | `migration_risk` |
|---------|---------------|----------------------------|------------------|
| `kernel` | NKI kernels | **Yes** — NKI sits below the framework boundary | low |
| `collective` | TP/CP/EP communication patterns | Mostly — primitives are stable | low-medium |
| `config` | TP degree, dtype, batching | Concepts yes, exact knob names no | medium |
| `framework` | vLLM / NxDI internals patches | **Often not** | high |
| `graph` | XLA passes, fusion config | **Likely not** — XLA-specific | high |

### Why this shapes Stage 3/4 priorities

Given two candidate optimizations with similar expected value, **prefer the
one at a lower layer.** A NKI kernel worth +15% is more valuable than a
framework-internals patch worth +15%, because one survives the migration and
one does not.

This should be an explicit tiebreaker in the proposer, not just human instinct:

```python
def rank_candidates(candidates):
    return sorted(candidates, key=lambda c: (
        -c.expected_gain,
        LAYER_DURABILITY[c.layer],   # kernel=0, collective=1, config=2,
                                      # framework=3, graph=4 — lower is better
    ))
```

Supporting evidence from the reference data: the two biggest rounds were
"model code + **NKI flash attention**" (+405%) and "**Context Parallel +
Local-Q**" (+193%). The NKI and collective portions of those transfer. The
vLLM-internals portions do not.

### Post-migration re-verification query

When the backend migrates, the work is scoped by a single query:

```
bank.query(migration_risk="high") -> re-verify these
bank.query(migration_risk="low")  -> assume valid, spot-check only
```

Without layer tags this is "re-test the entire bank," which at a few hundred
lessons is weeks of compute.

### `nki_kernel` provenance: harvested vs. borrowed vs. invented

Every kernel lesson records where it came from. This is what makes the
borrow-vs-invent metrics in
[`optimization-stages.md`](./optimization-stages.md#measuring-whether-it-actually-invents-anything)
computable rather than guesswork.

```yaml
type: nki_kernel
layer: kernel                  # see layer tagging above
backend_validated: [vllm-neuron-xla]
migration_risk: low
provenance:
  origin: harvested | borrowed | invented | hybrid
  # harvested — mandatory:
  harvested_from:
    repo: https://github.com/aws-neuron/nki-library
    commit: 7f3a1b2
    symbol: nkilib.attention_kv_parallel_segmented_cte
    bundled_or_package: bundled     # which nkilib was actually loaded
  # borrowed / hybrid — mandatory:
  source_references:
    - repo: https://github.com/vllm-project/vllm
      commit: a1b2c3d4
      license: Apache-2.0
      what_was_taken: "block-wise KV indexing scheme"
  # invented — mandatory:
  invention_rationale: >
    No reference handles Gated DeltaNet recurrent state on a systolic
    array. Designed from the roofline bound: DMA-bound at 34% of peak
    bandwidth, so restructured to batch state reads across 4 timesteps.
  beat_borrowed_by: 0.12    # required when origin=invented AND a borrowed
                            # alternative existed. Must exceed 0.05 margin.
```

`origin: hybrid` covers the common real case — a borrowed algorithm with
substantial Neuron-specific restructuring. Distinguishing all three is what
keeps the invention metrics honest instead of self-congratulatory.

Kernels that *lost* to their borrowed alternative still get recorded, in
`provisional/`, with `beat_borrowed_by` below the margin. Failed invention
attempts are information for the next model.

### Why anti-patterns get their own type

They save as much compute as positive lessons — often more. If the bank
knows "TP=16 on Trn2 always spills for this family," the proposer prunes
that branch **without compiling**. At 5-20 min per compile, that is
minutes-to-hours saved per pruned candidate.

Anti-patterns get their own top-level folder per family, and the proposer
reads them on **every** iteration, before generating candidates:

```
                     +---------------------------+
                     |  proposer.next_k(...)     |
                     +-------------+-------------+
                                   |
                 1. read config_priors (what to try)
                                   |
                 2. generate candidate configs
                                   |
                                   v
                     +---------------------------+
                     |  read anti-patterns/      |  <-- pruning filter
                     |  for this family + SDK    |
                     +-------------+-------------+
                                   |
                 3. DROP any candidate matching a
                    known-bad pattern (no compile,
                    no measurement, zero cost)
                                   |
                                   v
                        surviving candidates -> compile
```

An anti-pattern entry carries a **matcher** — a predicate over the config —
rather than a fixed config, so it prunes a whole region of the space:

```yaml
# lesson-id: tp16-weight-spill-small-models
type: anti_pattern
applicability:
  architecture_family: dense_causal_lm
  param_count_range: [0, 30e9]
  neuron_sdk_versions: ["2.26.*", "2.27.*", "2.28.*"]
matcher:                          # prune any candidate where this holds
  tp_degree: {gte: 16}
reason: >
  At TP>=16 with under 30B params, per-core weight shards get small enough
  that collective overhead dominates and the compiler spills. Measured 3.1x
  slower than TP=8 on the same model.
evidence:
  measurements:
    - model: qwen2.5-14b
      instance: trn2.48xlarge
      sdk: 2.28.0
      throughput_tps: 680
      comparison_config_tps: 2110    # TP=8
      timestamp: 2026-08-14T00:00:00Z
confidence:
  n_models_validated: 3
  architecture_diversity: 1
  human_verified: true
savings:
  compiles_avoided_estimate: 2       # per optimizer run
```

The `savings` field lets us report how much the anti-pattern folder is
actually earning us — see "Metrics for the bank itself" below.

## Storage + retrieval

### On disk

```
knowledge-bank/
  lessons/
    dense-causal-lm/
      config-priors/
        llama-tp-baseline.yaml
        qwen-quantization.yaml
      op-rewrites/
        rmsnorm-attention-fusion.yaml
      nki-kernels/
        gqa-paged-attention.yaml
      anti-patterns/
        tp-16-spill.yaml
    moe/
      ...
  index/
    by-family.json      # denormalized index, machine-generated
    by-sdk.json
    stale-list.json     # generated by verify-all
  render/
    site/               # static HTML built from lessons/
```

Lessons are the source of truth. Indexes are regenerated. Rendered site is
regenerated. This keeps `git blame` on individual lessons useful — every
change has provenance.

### Query at optimizer start

```python
def start_optimization(model):
    family = classify(model)                        # dense_lm / moe / diffusion / ...
    priors = bank.query(
        architecture_family=family,
        param_count=model.total_params,
        seq_len=model.max_seq_len,
        batch=target_batch,
        sdk_version=current_sdk,
    )
    # priors is a ranked list:
    #   1. config_priors ordered by (confidence × recency × n_similar_models)
    #   2. relevant anti_patterns for pruning
    #   3. nki_kernels applicable to the graph's ops
    return priors
```

### Confidence + ranking

Confidence = f(n_models_validated, architecture_diversity, recency,
human_verified). Rough weights (tune later):

- 1 measurement on 1 model, machine-generated: 0.2
- 3 measurements on 1 architecture: 0.5
- 5+ measurements across ≥2 architecture families: 0.8
- Human-authored + human-verified: floor of 0.6

The proposer prefers high-confidence entries but doesn't ignore low-confidence
ones — they just get proposed later in the search.

## Failure modes to design around

### Stale lessons

Neuron compiler moves fast. A lesson from SDK 2.20 might be actively wrong
by 2.28. Mitigations:

- Every lesson stamps the SDK versions it's been validated on
- `verify-all --sdk NEW_VERSION` runs a spot-check of the top-N most-consulted
  lessons on each SDK release; auto-marks stale ones
- Optimizer's proposer down-weights lessons whose last-reverified is > 2
  SDK versions old

### Seed-model bias

If phase 1 uses 3 dense LLMs, the bank will over-index on dense-LLM tricks
and mislead on MoE/diffusion. Mitigations:

- Seed set spans families deliberately (see `plan.md` phase 1 seed choice)
- Confidence formula rewards architecture diversity, not just count
- Adapters (per-family, in leaderboard/) can override bank priors when a
  family is under-represented

### Bank rot / low-quality lessons

Autogenerated lessons from an unverified optimizer run are cheap to produce
and can drown out human-curated ones. Mitigations:

- Two tiers: `provisional/` (auto-generated) and `verified/` (human-signed-off)
- Proposer only pulls from `verified/` in v0. `provisional/` is a queue for
  humans to review.
- Verification is: read the lesson, decide if it generalizes, promote or drop

### License / attribution when borrowing

Reference translations (vLLM patterns, SGLang patterns) must cite the
source. In-lesson field `source_references` is mandatory for
`reference_translation` type. We're borrowing semantics, not copying code
— but the reference deserves the citation.

## Seed content (day-one bootstrap)

Don't wait for the optimizer to rediscover things the Neuron team already
knows. Suggested first 30-50 lessons, hand-authored:

### Config priors (~10)

- Dense LLM 7B → TP=1 or 2, bf16 baseline
- Dense LLM 13B → TP=4, bf16
- Dense LLM 32B → TP=8, bf16 (this is the canonical one)
- Dense LLM 70B → TP=16 on trn2.48xlarge, bf16
- MoE Nx7B (Mixtral-shaped) → expert-parallel + TP hybrid
- Diffusion (SDXL-shaped) → …
- Encoder-only (BERT-family) → …
- Continuous batching almost always wins for LLM inference above batch 4
- Static batching wins for prefill-heavy workloads
- KV cache in bf16 unless we've validated fp8 for that family

### Anti-patterns (~5-8)

- TP=16 on Trn2 causes weight spill for models under 30B — never propose
- FP8 activations on RMSNorm-heavy models can accumulate error past
  tolerance — skip unless we have per-op rtol tuning
- Placement groups off for our compute nodes (they over-constrain)
- Don't set MinCount != MaxCount for MLCB queues (PC rejects it)
- Whichever attention kernel breaks GQA on a given SDK version — track it

### NKI kernels (~10)

Existing production-tested NKI kernels for: RoPE, RMSNorm, sampling
(top-k / top-p), softmax variants, GQA attention. Each becomes a
`nki_kernel` lesson with its op signature + shape constraints + evidence.

### Reference translations (~5-10)

- vLLM's `paged_attention` → our NKI pattern for paged K/V
- SGLang's radix cache → adaptation notes for Neuron memory hierarchy
- TensorRT-LLM's fused RMSNorm+RoPE → NKI equivalent
- HuggingFace's dynamic batching → continuous batching in vLLM-Neuron

## Metrics for the bank itself

Track over time so we know if the flywheel is working:

- `bank_hit_rate`: fraction of optimizer proposals that come from bank priors
- `avg_iterations_to_convergence`: should drop as bank grows
- `n_positive_lessons` / `n_negative_lessons` / `n_kernels`
- `stale_ratio`: lessons whose last-reverified is > 2 SDK versions old
- `human_verified_ratio`: how much of the bank is trusted

If `bank_hit_rate` stays low as bank grows, we're capturing lessons that
aren't actually reused — refine the schema or coverage.

# Knowledge Bank (store)

Live store of lessons. Schema and design rationale live in
[`../knowledge-bank.md`](../knowledge-bank.md) — this folder is the data.

## Layout

```
knowledge-bank/
  verified/                 <-- proposer reads ONLY this in v0
    dense-causal-lm/
      config-priors/
      op-rewrites/
      nki-kernels/
      anti-patterns/        <-- read every iteration, prunes before compile
      reference-translations/
    moe-causal-lm/
      ...
    diffusion/
    speech/
    encoder-only/
  provisional/              <-- optimizer writes here; humans triage weekly
    <same family structure>
  index/                    <-- generated, do not hand-edit
  templates/                <-- copy these when authoring a new lesson
```

## Two tiers

| Tier | Written by | Read by proposer | Promotion |
|------|-----------|------------------|-----------|
| `verified/` | Humans (authored or promoted) | Yes | — |
| `provisional/` | Optimizer, automatically | No (v0) | Weekly human triage |

Rationale: auto-generated lessons are cheap and unreliable. Letting them
straight into the proposer's input would let one bad measurement poison
future runs. See `../open-questions.md` Q10.

## Authoring a lesson

1. Copy the matching template from `templates/`
2. Fill every field — partial lessons are worse than none, because the
   applicability predicate is what makes retrieval work
3. Put it under `verified/<family>/<type>/`
4. Run `make index` to regenerate `index/`

## The anti-patterns folder specifically

`anti-patterns/` is the highest-leverage folder here. Every entry prunes
candidates *before* a compile happens, so each one directly buys back
5-20 minutes per pruned candidate per run.

Add an anti-pattern any time:
- A config measurably underperforms a known-good alternative
- A config fails to compile or times out
- A config OOMs above the HBM ceiling
- A config passes performance but fails equivalence

That last one matters — "fast but wrong" is exactly the trap the bank should
remember, and it is easy to lose if only wins get recorded.

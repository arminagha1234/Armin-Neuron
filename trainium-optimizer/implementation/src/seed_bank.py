"""
Seed the knowledge bank with initial verified lessons.

Do not make the optimizer rediscover things the Neuron ecosystem already
knows. These lessons come from the auto_research run's real measured findings
(the three all_gather-elimination rewrites that carried it to a large (multiple-x)) plus
well-established config priors and anti-patterns.

All are backend-relevant and tagged by layer. The three op_rewrites are
COLLECTIVE-layer, so they survive an XLA -> native-PyTorch migration (they are
about the math, not the framework).

    python seed_bank.py --bank-root ../../knowledge-bank

See ../../knowledge-bank.md for the schema and ../../references-analysis.md
for where these came from.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from bank import Applicability, Confidence, KnowledgeBank, Lesson, LessonType, Symptom, Tier
from ledger import Layer, Origin


def seed_lessons() -> list[Lesson]:
    sdk = ["2.26.*", "2.27.*", "2.28.*"]
    dense = "dense_causal_lm"
    moe = "moe_causal_lm"

    return [
        # --- collective-layer op_rewrites (the auto_research wins) ----------
        Lesson(
            lesson_id="local-q-eliminate-hidden-allgather",
            type=LessonType.OP_REWRITE,
            applicability=Applicability(dense, (7e9, 200e9), neuron_sdk_versions=sdk),
            layer=Layer.COLLECTIVE, migration_risk="low-medium",
            origin=Origin.BORROWED,
            intervention={"spec": {"attention_sharding": "local_q"}},
            reason=(
                "Standard TP all-gathers full hidden states then each rank "
                "computes QKV on a shard. Local-Q inverts it: each rank "
                "computes full QKV on its local tokens (seq/TP), then "
                "all-gathers only the small K/V. Saves the hidden all_gather "
                "and cuts QKV compute by TP x."
            ),
            symptoms_addressed=[Symptom(
                bottleneck="collective_bound",
                signature="all_gather of hidden states dominates step time",
                observed_via="CC engine busy, PE idle in profile",
            )],
            source="internal-prior-optimization-run",
            confidence=Confidence(n_models_validated=4, architecture_diversity=2,
                                  human_verified=True),
            last_reverified_sdk="2.28.0",
            evidence=[{"note": "part of the +193% Round 3 on Tongyi-30B-A3B"}],
        ),
        Lesson(
            lesson_id="context-parallel-kv-split",
            type=LessonType.OP_REWRITE,
            applicability=Applicability(dense, (7e9, 200e9),
                                        seq_len_range=(8192, 10_000_000),
                                        neuron_sdk_versions=sdk),
            layer=Layer.COLLECTIVE, migration_risk="low-medium",
            origin=Origin.BORROWED,
            intervention={"spec": {"context_parallel": True}},
            reason=(
                "Prior KV cache split across ranks; each rank attends to 1/TP "
                "of prior context, merged via online softmax reduction (exact). "
                "Cuts prior-attention compute by TP x. Biggest wins at long "
                "context where attention dominates."
            ),
            symptoms_addressed=[Symptom(
                bottleneck="compute_bound",
                signature="prior-context attention scan dominates at long seq",
                observed_via="PE busy in attention, cost grows with context len",
            )],
            source="internal-prior-optimization-run + nki-library kv_parallel_segmented",
            confidence=Confidence(n_models_validated=4, architecture_diversity=2,
                                  human_verified=True),
            last_reverified_sdk="2.28.0",
        ),
        Lesson(
            lesson_id="local-moe-eliminate-input-allgather",
            type=LessonType.OP_REWRITE,
            applicability=Applicability(moe, (20e9, 300e9), neuron_sdk_versions=sdk),
            layer=Layer.COLLECTIVE, migration_risk="low-medium",
            origin=Origin.BORROWED,
            intervention={"spec": {"moe_sharding": "local_moe"}},
            reason=(
                "Each rank keeps full MoE/MLP weights, processes only local "
                "tokens, all-reduces output. Eliminates the MoE/MLP input "
                "all_gather entirely. MoE models gained most from this (11-17x)."
            ),
            symptoms_addressed=[Symptom(
                bottleneck="collective_bound",
                signature="MoE input all_gather dominates",
                observed_via="CC engine busy at expert dispatch",
            )],
            source="internal-prior-optimization-run",
            confidence=Confidence(n_models_validated=3, architecture_diversity=1,
                                  human_verified=True),
            last_reverified_sdk="2.28.0",
        ),

        # --- config priors --------------------------------------------------
        Lesson(
            lesson_id="dense-30b-tp8-bf16",
            type=LessonType.CONFIG_PRIOR,
            applicability=Applicability(dense, (20e9, 40e9), neuron_sdk_versions=sdk),
            layer=Layer.CONFIG, migration_risk="medium",
            intervention={"spec": {"tp_degree": 8, "weights_dtype": "bf16",
                                   "kv_cache_dtype": "bf16"}},
            reason="~30B dense fits well at TP=8 bf16 on trn2; good starting point.",
            confidence=Confidence(n_models_validated=3, architecture_diversity=1,
                                  human_verified=True),
            last_reverified_sdk="2.28.0",
        ),
        Lesson(
            lesson_id="dense-7b-tp2-bf16",
            type=LessonType.CONFIG_PRIOR,
            applicability=Applicability(dense, (3e9, 10e9), neuron_sdk_versions=sdk),
            layer=Layer.CONFIG, migration_risk="medium",
            intervention={"spec": {"tp_degree": 2, "weights_dtype": "bf16"}},
            reason="Small dense models: TP=2 avoids collective overhead of higher TP.",
            confidence=Confidence(n_models_validated=2, human_verified=True),
            last_reverified_sdk="2.28.0",
        ),

        # --- anti-patterns (prune before compile) ---------------------------
        Lesson(
            lesson_id="tp16-spill-under-30b",
            type=LessonType.ANTI_PATTERN,
            applicability=Applicability(dense, (0, 30e9), neuron_sdk_versions=sdk),
            layer=Layer.CONFIG, migration_risk="medium",
            matcher={"tp_degree": {"gte": 16}},
            reason=("At TP>=16 with under 30B params, per-core weight shards get "
                    "small enough that collective overhead dominates and the "
                    "compiler spills. ~3x slower than TP=8."),
            confidence=Confidence(n_models_validated=3, human_verified=True),
            last_reverified_sdk="2.28.0",
        ),
        Lesson(
            lesson_id="fp8-activations-rmsnorm-heavy",
            type=LessonType.ANTI_PATTERN,
            applicability=Applicability(dense, (0, 200e9), neuron_sdk_versions=sdk),
            layer=Layer.CONFIG, migration_risk="medium",
            matcher={"activations_dtype": "fp8"},
            reason=("FP8 activations on RMSNorm-heavy models can accumulate error "
                    "past equivalence tolerance. Only enable with per-op rtol "
                    "tuning validated."),
            confidence=Confidence(n_models_validated=2, human_verified=True),
            last_reverified_sdk="2.28.0",
        ),
    ]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bank-root", type=Path, default=Path("../../knowledge-bank"))
    a = ap.parse_args()

    bank = KnowledgeBank(a.bank_root.resolve())
    lessons = seed_lessons()
    for lesson in lessons:
        # Seed lessons are the hand-authored, human-curated bootstrap — they
        # go straight to verified so the proposer trusts them from run 1.
        lesson.tier = Tier.VERIFIED
        lesson.confidence.human_verified = True
        p = bank.save(lesson)
        print(f"  seeded {lesson.tier.value}/{lesson.type.value}: {lesson.lesson_id}")
    stats = bank.stats(current_sdk="2.28.0")
    print(f"\nbank now: {stats['verified']} verified, {stats['provisional']} provisional")
    print(f"by type: {stats['by_type']}")


if __name__ == "__main__":
    main()

"""
End-to-end orchestrator test on the mock backend.

Exercises the full Stage-1 loop with zero hardware: seed beam from bank priors
-> expand -> prune anti-patterns -> compile/equivalence/measure with guardrails
-> keep/discard -> ledger. Then asserts the ledger reflects a real search:
the incumbent improved, anti-patterns were pruned, and every attempt is logged.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from backends.mock import MockBackend
from bank import (
    Applicability,
    Confidence,
    KnowledgeBank,
    Lesson,
    LessonType,
    Tier,
)
from guardrails import Guardrails
from ledger import Layer, Ledger, Stage, Status
from orchestrator import ModelSpec, Orchestrator, always_equivalent


def _seed_bank(root: Path) -> KnowledgeBank:
    bank = KnowledgeBank(root)
    # A config prior that should seed the beam.
    bank.save(Lesson(
        lesson_id="dense-31b-continuous-flash",
        type=LessonType.CONFIG_PRIOR,
        applicability=Applicability(
            architecture_family="dense_causal_lm",
            param_count_range=(20e9, 40e9),
            neuron_sdk_versions=["2.28.*"],
        ),
        layer=Layer.CONFIG, migration_risk="medium", tier=Tier.VERIFIED,
        intervention={"spec": {"batching": "continuous", "attention_kernel": "flash"}},
        confidence=Confidence(n_models_validated=3, human_verified=True),
        last_reverified_sdk="2.28.0",
    ))
    # An anti-pattern that should prune TP>=16 (which the mock models as slow).
    bank.save(Lesson(
        lesson_id="tp16-spill",
        type=LessonType.ANTI_PATTERN,
        applicability=Applicability(
            architecture_family="dense_causal_lm",
            param_count_range=(0, 40e9),
            neuron_sdk_versions=["2.28.*"],
        ),
        layer=Layer.CONFIG, migration_risk="medium", tier=Tier.VERIFIED,
        matcher={"tp_degree": {"gte": 16}},
        reason="weight spill; slower than TP=8",
        confidence=Confidence(n_models_validated=3, human_verified=True),
        last_reverified_sdk="2.28.0",
    ))
    return bank


@pytest.fixture
def orchestrator(tmp_path: Path) -> Orchestrator:
    return Orchestrator(
        backend=MockBackend(seed=42),
        bank=_seed_bank(tmp_path / "bank"),
        guards=Guardrails(),
        ledger=Ledger(tmp_path / "run"),
        equivalence=always_equivalent,
        sdk_version="2.28.0",
    )


SPEC = ModelSpec(
    model_id="google/gemma-4-31B",
    family="dense_causal_lm",
    param_count=31e9,
    parent="gemma",
    probe_shape="chat 1k/512",
    probe_batch=1,
)


def test_stage1_improves_over_baseline(orchestrator: Orchestrator):
    orchestrator.ledger.init()
    best = orchestrator.run_stage1_config(SPEC)

    # The incumbent should be meaningfully better than the mock's baseline
    # (tp=1, static, default attention ~= 600 tok/s).
    assert best.metric > 600
    # And it should have moved toward the known-good knobs.
    assert best.config["tp_degree"] in (4, 8)   # not 1, not the pruned 16/32
    assert best.config["tp_degree"] < 16


def test_ledger_records_every_attempt(orchestrator: Orchestrator):
    orchestrator.ledger.init()
    orchestrator.run_stage1_config(SPEC)
    rows = orchestrator.ledger.read()
    # Many attempts, keeps and discards both present.
    assert len(rows) > 5
    assert any(r.status is Status.KEEP for r in rows)
    assert any(r.status is Status.DISCARD for r in rows)


def test_antipatterns_were_pruned(orchestrator: Orchestrator):
    orchestrator.ledger.init()
    orchestrator.run_stage1_config(SPEC)
    rows = orchestrator.ledger.read()
    pruned = [r for r in rows if r.description.startswith("pruned:")]
    assert pruned, "expected TP>=16 candidates to be pruned by the anti-pattern"
    assert all(r.compile_s == 0.0 for r in pruned), "pruning must cost no compile"


def test_no_tp16_was_ever_compiled(orchestrator: Orchestrator):
    """The anti-pattern must prevent TP>=16 from reaching a compile at all."""
    orchestrator.ledger.init()
    orchestrator.run_stage1_config(SPEC)
    rows = orchestrator.ledger.read()
    # Any row that actually compiled (compile_s > 0) must not be a tp16/32 move.
    compiled = [r for r in rows if r.compile_s > 0]
    assert not any("tp_degree=16" in r.description or "tp_degree=32" in r.description
                   for r in compiled)


def test_incumbent_monotonic_in_ledger(orchestrator: Orchestrator):
    """Kept rows should be non-decreasing in metric — a KEEP only happens on
    improvement."""
    orchestrator.ledger.init()
    orchestrator.run_stage1_config(SPEC)
    kept = [r.metric for r in orchestrator.ledger.kept()]
    assert kept == sorted(kept), f"kept metrics not monotonic: {kept}"


def test_equivalence_failure_blocks_promotion(tmp_path: Path):
    """A candidate that fails equivalence must never be kept, even if fast."""
    from orchestrator import EquivalenceResult

    def always_fail(_neff, _spec):
        return EquivalenceResult(passed=False, correctness_pct=42.0,
                                 notes="synthetic failure")

    orch = Orchestrator(
        backend=MockBackend(seed=1),
        bank=_seed_bank(tmp_path / "bank"),
        guards=Guardrails(),
        ledger=Ledger(tmp_path / "run"),
        equivalence=always_fail,
        sdk_version="2.28.0",
    )
    orch.ledger.init()
    orch.run_stage1_config(SPEC)
    rows = orch.ledger.read()
    # The baseline is the reference and is kept. Every CONFIG *change* fails
    # equivalence, so none is kept, and the incumbent stays at baseline.
    config_changes = [r for r in rows if r.stage is Stage.CONFIG]
    assert config_changes, "expected some config candidates to be attempted"
    assert all(r.status is Status.DISCARD for r in config_changes)
    assert any("equivalence fail" in r.description for r in rows)
    assert orch.incumbent.provenance == "baseline"

"""
End-to-end demo: run the real orchestrator search against the mock backend and
chart the trajectory it actually produces.

Unlike make_sample_run.py (which writes a hand-authored ledger to show the
chart format), this runs the genuine Stage-1 beam search — seed from bank
priors, expand, prune anti-patterns, compile/measure under guardrails,
keep/discard — and charts whatever comes out. Proof the pieces work together.

    python examples/run_mock_search.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from backends.mock import MockBackend           # noqa: E402
from bank import (                              # noqa: E402
    Applicability, Confidence, KnowledgeBank, Lesson, LessonType, Tier,
)
from guardrails import Guardrails               # noqa: E402
from ledger import Layer, Ledger                # noqa: E402
from orchestrator import ModelSpec, Orchestrator  # noqa: E402
from trajectory_chart import build_chart        # noqa: E402


def seed_bank(root: Path) -> KnowledgeBank:
    bank = KnowledgeBank(root)
    bank.save(Lesson(
        lesson_id="dense-31b-continuous-flash",
        type=LessonType.CONFIG_PRIOR,
        applicability=Applicability(
            architecture_family="dense_causal_lm",
            param_count_range=(20e9, 40e9), neuron_sdk_versions=["2.28.*"],
        ),
        layer=Layer.CONFIG, migration_risk="medium", tier=Tier.VERIFIED,
        intervention={"spec": {"batching": "continuous", "attention_kernel": "flash"}},
        confidence=Confidence(n_models_validated=3, human_verified=True),
        last_reverified_sdk="2.28.0",
    ))
    bank.save(Lesson(
        lesson_id="tp16-spill",
        type=LessonType.ANTI_PATTERN,
        applicability=Applicability(
            architecture_family="dense_causal_lm",
            param_count_range=(0, 40e9), neuron_sdk_versions=["2.28.*"],
        ),
        layer=Layer.CONFIG, migration_risk="medium", tier=Tier.VERIFIED,
        matcher={"tp_degree": {"gte": 16}},
        reason="weight spill under 30B; slower than TP=8",
        confidence=Confidence(n_models_validated=3, human_verified=True),
        last_reverified_sdk="2.28.0",
    ))
    return bank


def main() -> None:
    out_dir = Path(__file__).parent / "mock-search"
    led = Ledger(out_dir)
    if led.path.exists():
        led.path.unlink()
    led.init()

    orch = Orchestrator(
        backend=MockBackend(seed=7),
        bank=seed_bank(out_dir / "bank"),
        guards=Guardrails(),
        ledger=led,
        sdk_version="2.28.0",
    )
    spec = ModelSpec(
        model_id="google/gemma-4-31B", family="dense_causal_lm",
        param_count=31e9, parent="gemma", probe_shape="chat 1k/512", probe_batch=1,
    )

    best = orch.run_stage1_config(spec)

    print(f"model    : {spec.model_id}")
    print(f"baseline : {led.baseline().metric:.0f} tok/s")
    print(f"best     : {best.metric:.0f} tok/s  ({led.speedup():.2f}x)")
    print(f"config   : {best.config}")
    print(f"attempts : {len(led.read())} rows "
          f"({len(led.kept())} kept, {len(led.read()) - len(led.kept())} discarded)")
    pruned = [r for r in led.read() if r.description.startswith('pruned:')]
    print(f"pruned   : {len(pruned)} candidates (zero compile cost)")
    print(f"compile  : {led.compile_time_total_s():.0f}s total")

    chart = build_chart(
        run_dir=out_dir, out_path=out_dir / "trajectory.png",
        model="Gemma 4 31B (mock backend)", hardware="mock", tp=None,
        shape="chat 1k/512", sdk="2.28.0",
    )
    print(f"chart    : {chart} ({chart.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()

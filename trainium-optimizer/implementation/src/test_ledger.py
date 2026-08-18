"""
Tests for the ledger and the trajectory chart.

The synthetic trajectory mirrors the real numbers from
`internal-prior-optimization-run`'s Round 2 (845 -> 4,269 tok/s),
including its failure pattern, so the chart is exercised against a shape we
know occurs in practice rather than a tidy monotonic curve.

Run:
    cd implementation/src && python -m pytest test_ledger.py -v
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ledger import (
    LAYER_DURABILITY,
    MIGRATION_RISK,
    Layer,
    Ledger,
    Origin,
    Row,
    Stage,
    Status,
)


def _row(metric, stage=Stage.CONFIG, origin=Origin.NONE, layer=Layer.CONFIG,
         status=Status.KEEP, desc="x", mfu=-1.0, compile_s=0.0, source=""):
    return Row(
        commit=f"c{int(metric)}", stage=stage, origin=origin, layer=layer,
        source=source, metric=metric, mfu=mfu, correctness=100.0,
        compile_s=compile_s, status=status, description=desc,
    )


@pytest.fixture
def realistic_run(tmp_path: Path) -> Ledger:
    """A trajectory shaped like the reference implementation's Round 2."""
    led = Ledger(tmp_path / "run")
    led.init()

    rows = [
        # baseline
        _row(845.0, Stage.BASELINE, Origin.NONE, Layer.NONE,
             desc="baseline BF16 experts FP8 KV TP4", mfu=0.33, compile_s=391),

        # harvest is read-only; it emits a manifest, not a measurement

        # config: several tries, a couple of wins, some losses
        _row(899.5, Stage.CONFIG, desc="batch1024 + block128", mfu=0.35, compile_s=331),
        _row(860.0, Stage.CONFIG, status=Status.DISCARD,
             desc="1024-token segment (slower)", compile_s=290),
        _row(858.0, Stage.CONFIG, status=Status.DISCARD,
             desc="16-token KV blocks (slower)", compile_s=310),
        _row(1031.5, Stage.CONFIG, desc="GQA broadcast", mfu=0.41, compile_s=305),

        # known kernels — harvested from nkilib
        _row(1124.5, Stage.KNOWN_KERNEL, Origin.HARVESTED, Layer.KERNEL,
             desc="nkilib rmsnorm_quant fused", mfu=0.44, compile_s=402,
             source="nki-library@7f3a1b2"),
        _row(1100.0, Stage.KNOWN_KERNEL, Origin.HARVESTED, Layer.KERNEL,
             status=Status.DISCARD, desc="nkilib attention_tkg (wrong regime)",
             compile_s=380, source="nki-library@7f3a1b2"),

        # borrow — the big win, as in the reference
        _row(1555.8, Stage.BORROW, Origin.BORROWED, Layer.KERNEL,
             desc="BF16 QK/PV softmax", mfu=0.62, compile_s=455, source="vllm@a1b2c3d"),
        _row(4071.1, Stage.BORROW, Origin.BORROWED, Layer.KERNEL,
             desc="NKI flash attention tiling + online softmax", mfu=1.62,
             compile_s=612, source="flashattn@bsd3"),

        # invent — attempted, mostly failed, one marginal win
        _row(4102.0, Stage.INVENT, Origin.INVENTED, Layer.KERNEL,
             status=Status.DISCARD,
             desc="6-way SBUF split (+0.8%, under 5% margin)", compile_s=740),
        _row(4090.0, Stage.INVENT, Origin.INVENTED, Layer.KERNEL,
             status=Status.DISCARD, desc="DMA prefetch restructure", compile_s=700),
        _row(4055.0, Stage.INVENT, Origin.INVENTED, Layer.KERNEL,
             status=Status.DISCARD, desc="alt tiling order", compile_s=690),

        # graph rewrite
        _row(4269.0, Stage.GRAPH_REWRITE, Origin.NONE, Layer.GRAPH,
             desc="ModelLen + O3", mfu=1.68, compile_s=520),
    ]
    for r in rows:
        led.append(r)
    return led


# -- ledger ------------------------------------------------------------------

def test_roundtrip(realistic_run: Ledger):
    rows = realistic_run.read()
    assert len(rows) == 13
    assert rows[0].stage is Stage.BASELINE
    assert rows[0].metric == 845.0


def test_baseline_and_incumbent(realistic_run: Ledger):
    assert realistic_run.baseline().metric == 845.0
    assert realistic_run.incumbent().metric == 4269.0


def test_speedup(realistic_run: Ledger):
    assert realistic_run.speedup() == pytest.approx(4269.0 / 845.0, rel=1e-6)
    assert realistic_run.speedup() == pytest.approx(5.05, abs=0.01)


def test_kept_excludes_discards(realistic_run: Ledger):
    kept = realistic_run.kept()
    assert len(kept) == 7
    assert all(r.kept for r in kept)


def test_provenance_counts(realistic_run: Ledger):
    c = realistic_run.provenance_counts()
    assert c["harvested"] == 1
    assert c["borrowed"] == 2
    assert c["invented"] == 0      # all invention attempts were discarded
    assert c["hybrid"] == 0


def test_invention_stats_reflect_failure(realistic_run: Ledger):
    """Three attempts, zero promotions — the expected early-run shape."""
    s = realistic_run.invention_stats()
    assert s["invention_attempts"] == 3
    assert s["invention_promoted"] == 0
    assert s["invention_win_rate"] == 0.0
    assert s["invention_rate"] == 0.0


def test_stage_summary_attributes_gain(realistic_run: Ledger):
    s = realistic_run.stage_summary()
    # borrow should be credited with the dominant gain, matching the
    # reference implementation's finding that kernel work carried the run
    assert s["borrow"]["gain_pct"] > s["config"]["gain_pct"]
    assert s["invent"]["attempts"] == 3
    assert s["invent"]["promoted"] == 0


def test_compile_time_tracked(realistic_run: Ledger):
    # compile dominates loop cost; make sure we can see it
    assert realistic_run.compile_time_total_s() > 5000


def test_description_sanitized(tmp_path: Path):
    """Tabs/newlines would corrupt the TSV — they must be normalized."""
    led = Ledger(tmp_path / "r")
    led.init()
    led.append(_row(100.0, desc="has\ttab and\nnewline"))
    assert led.read()[0].description == "has tab and newline"


def test_init_is_idempotent(tmp_path: Path):
    led = Ledger(tmp_path / "r")
    led.init()
    led.append(_row(100.0))
    led.init()                       # must not clobber
    assert len(led.read()) == 1


def test_empty_ledger_is_safe(tmp_path: Path):
    led = Ledger(tmp_path / "nope")
    assert led.read() == []
    assert led.incumbent() is None
    assert led.speedup() is None


# -- layer / migration semantics ---------------------------------------------

def test_kernel_is_most_durable():
    assert MIGRATION_RISK[Layer.KERNEL] == "low"
    assert MIGRATION_RISK[Layer.FRAMEWORK] == "high"
    assert MIGRATION_RISK[Layer.GRAPH] == "high"


def test_layer_durability_ordering():
    """The proposer tiebreaker: prefer kernel over framework at equal gain."""
    assert LAYER_DURABILITY[Layer.KERNEL] < LAYER_DURABILITY[Layer.COLLECTIVE]
    assert LAYER_DURABILITY[Layer.COLLECTIVE] < LAYER_DURABILITY[Layer.CONFIG]
    assert LAYER_DURABILITY[Layer.CONFIG] < LAYER_DURABILITY[Layer.FRAMEWORK]
    assert LAYER_DURABILITY[Layer.FRAMEWORK] < LAYER_DURABILITY[Layer.GRAPH]


def test_migration_risk_query(realistic_run: Ledger):
    """Post-migration scope should be a single filter, not a full re-test."""
    rows = realistic_run.read()
    high = [r for r in rows if r.migration_risk == "high"]
    low = [r for r in rows if r.migration_risk == "low"]
    assert len(high) == 1                 # the graph rewrite
    assert len(low) >= 6                  # the kernel work survives
    assert all(r.layer is Layer.KERNEL for r in low)


# -- chart -------------------------------------------------------------------

def test_chart_renders(realistic_run: Ledger, tmp_path: Path):
    from trajectory_chart import build_chart

    out = build_chart(
        run_dir=realistic_run.run_dir,
        out_path=tmp_path / "timeline.png",
        model="Tongyi-30B-A3B (synthetic)",
        hardware="trn2.48xlarge", tp=4, shape="chat 1k/512", sdk="2.28.0",
        roofline=48000.0,
    )
    assert out.exists()
    assert out.stat().st_size > 20_000      # a real chart, not a blank canvas


def test_chart_without_roofline_or_mfu(tmp_path: Path):
    """Optional inputs must degrade gracefully, not crash."""
    from trajectory_chart import build_chart

    led = Ledger(tmp_path / "run2")
    led.init()
    led.append(_row(100.0, Stage.BASELINE, layer=Layer.NONE, desc="base"))
    led.append(_row(150.0, Stage.CONFIG, desc="tp8"))

    out = build_chart(
        run_dir=led.run_dir, out_path=tmp_path / "t2.png", model="Minimal",
    )
    assert out.exists()

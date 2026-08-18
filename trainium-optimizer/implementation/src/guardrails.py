"""
Guardrails — hard constraints on the optimizer.

These are gates, not suggestions. A candidate that violates one is rejected,
not warned about. Everything here is track-independent; per-track benchmark
shapes live in the adapter.

See ../../guardrails.md for the full design and the per-track shape tables.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from backends.base import Measurements


class Outcome(StrEnum):
    """The four-valued result taxonomy. Keeping these distinct matters: a model
    that cannot do 64k context is not broken, and the leaderboard must not
    imply it is."""

    OK = "ok"
    NOT_APPLICABLE = "not_applicable"   # model structurally cannot do this shape
    OOM = "oom"                         # could in principle, no config fit under ceiling
    FAILED = "failed"                   # compile error, equivalence fail, crash


@dataclass(frozen=True)
class Guardrails:
    hbm_ceiling: float = 0.85           # fraction of available HBM, at peak
    compile_timeout_s: float = 1800.0   # 30 min, family-overridable
    min_warmup_iters: int = 3
    min_measured_iters: int = 10
    max_p99_p50_ratio: float = 3.0      # reject noisier measurements, rerun
    no_improvement_streak: int = 5
    marginal_improvement_pct: float = 2.0   # below this counts as noise
    max_iterations: int = 100
    invention_margin_pct: float = 5.0   # Stage-4 promotion bar over borrowed

    def hbm_ok(self, m: Measurements) -> bool:
        # Peak HBM must be measured at full KV occupancy, not step 0 — a config
        # that is fine at token 1 and OOMs at token 65,536 is the exact failure
        # this prevents. Enforcing that is the backend's job; here we gate the
        # reported peak.
        return m.hbm_utilization <= self.hbm_ceiling

    def compile_ok(self, compile_seconds: float) -> bool:
        return compile_seconds <= self.compile_timeout_s

    def measurement_trustworthy(self, m: Measurements) -> bool:
        if m.warmup_iters < self.min_warmup_iters:
            return False
        if m.measured_iters < self.min_measured_iters:
            return False
        if m.metric_p50 > 0 and m.metric_p99 > 0:
            ratio = m.metric_p50 / m.metric_p99 if m.metric_p99 < m.metric_p50 else 1.0
            # p99 is the slower (smaller) number for throughput; guard the spread
            if m.metric_p50 / max(m.metric_p99, 1e-9) > self.max_p99_p50_ratio:
                return False
        return True

    def is_improvement(self, candidate_metric: float, incumbent_metric: float,
                       is_invention: bool = False) -> bool:
        """Promotion test. Stage 4 must clear a higher bar than the noise floor.

        The invention margin is not only about noise (we treat <2% as noise);
        it is a risk premium — borrowed code has been exercised by thousands of
        users, our invented kernel by us, today. On a near-tie, prefer the
        better-tested artifact.
        """
        if incumbent_metric <= 0:
            return candidate_metric > 0
        margin = self.invention_margin_pct if is_invention else self.marginal_improvement_pct
        return candidate_metric >= incumbent_metric * (1 + margin / 100.0)


@dataclass
class StoppingState:
    """Tracks termination conditions across a stage. Compute is uncapped;
    these are search-quality criteria, not budget."""

    guards: Guardrails
    iterations: int = 0
    no_improvement: int = 0
    marginal_streak: int = 0
    _recent_gains: list[float] = field(default_factory=list)

    def record(self, improved: bool, gain_pct: float = 0.0) -> None:
        self.iterations += 1
        if improved:
            self.no_improvement = 0
        else:
            self.no_improvement += 1
        if 0 < gain_pct < self.guards.marginal_improvement_pct:
            self.marginal_streak += 1
        elif improved:
            self.marginal_streak = 0

    def should_stop(self) -> tuple[bool, str]:
        if self.iterations >= self.guards.max_iterations:
            return True, "max_iterations"
        if self.no_improvement >= self.guards.no_improvement_streak:
            return True, "no_improvement_streak"
        if self.marginal_streak >= 3:
            return True, "marginal_only"
        return False, ""

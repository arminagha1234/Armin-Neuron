"""
Config proposer — beam search over configs, seeded by bank priors.

Revised from the original "hand-priored greedy" after reading Autocomp, which
proved beam search on Trainium/NKI specifically. Beam is barely more complex
than greedy (keep top-k rather than top-1) and is far more robust to the local
optima that greedy falls into when config axes interact — which they do (a
fusion that helps at TP=8/bf16 can hurt at TP=16/fp8).

We keep the hand-priored part: bank config_priors seed the initial beam and
bias which axes get perturbed first.

This proposer covers Stage 1 (config). Stages 2-5 (kernels) are driven by the
harvest manifest and profile-guided candidate generation, not this module.

See ../../references-analysis.md and ../../optimization-stages.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from bank import KnowledgeBank, Lesson, LessonType, Tier
from ledger import Layer, LAYER_DURABILITY


@dataclass
class Candidate:
    config: dict[str, Any]
    metric: float = 0.0
    parent: dict[str, Any] | None = None
    provenance: str = ""       # what changed vs. parent, for the ledger description
    layer: Layer = Layer.CONFIG

    def key(self) -> tuple:
        return tuple(sorted(self.config.items()))


class BeamProposer:
    """Beam search over the backend's config axes.

    Parameters mirror Autocomp's defaults, which are tuned for this task:
      beam_size=4, num_plan_candidates=8.
    """

    def __init__(
        self,
        axes: dict[str, list[Any]],
        bank: KnowledgeBank | None = None,
        beam_size: int = 4,
        plans_per_parent: int = 8,
    ) -> None:
        self.axes = axes
        self.bank = bank
        self.beam_size = beam_size
        self.plans_per_parent = plans_per_parent
        self._seen: set[tuple] = set()

    # -- seeding -------------------------------------------------------------

    def seed(
        self,
        baseline: dict[str, Any],
        family: str,
        param_count: float,
        seq_len: int,
        batch: int,
        sdk_version: str,
    ) -> list[Candidate]:
        """Initial beam: baseline plus any bank config_priors that apply.

        A good prior can land within a few percent of the Stage-1 optimum on
        the first candidate, which is the whole point of the knowledge bank.
        """
        beam = [Candidate(config=dict(baseline), provenance="baseline")]
        self._seen.add(beam[0].key())

        if self.bank is not None:
            priors = self.bank.query_interventions(
                family=family, param_count=param_count, seq_len=seq_len,
                batch=batch, sdk_version=sdk_version,
                types=(LessonType.CONFIG_PRIOR,),
            )
            for lesson in priors[: self.beam_size]:
                cfg = {**baseline, **lesson.intervention.get("spec", {})}
                cand = Candidate(
                    config=cfg, parent=baseline,
                    provenance=f"prior:{lesson.lesson_id}",
                )
                if cand.key() not in self._seen:
                    self._seen.add(cand.key())
                    beam.append(cand)
        return beam

    # -- expansion -----------------------------------------------------------

    def expand(self, beam: list[Candidate]) -> list[Candidate]:
        """Generate the next round of candidates by perturbing one axis at a
        time from each beam member.

        Single-axis perturbation is deliberate: it keeps credit assignment
        clean. When a candidate improves, we know which axis did it — the
        feedback-richness-vs-credit-assignment tension the ADAS survey names.
        """
        out: list[Candidate] = []
        for parent in beam:
            generated = 0
            for axis, values in self.axes.items():
                for v in values:
                    if generated >= self.plans_per_parent:
                        break
                    if parent.config.get(axis) == v:
                        continue
                    cfg = {**parent.config, axis: v}
                    cand = Candidate(
                        config=cfg, parent=parent.config,
                        provenance=f"{axis}={v}",
                    )
                    if cand.key() in self._seen:
                        continue
                    self._seen.add(cand.key())
                    out.append(cand)
                    generated += 1
        return out

    # -- selection -----------------------------------------------------------

    def select(self, scored: list[Candidate]) -> list[Candidate]:
        """Keep the top beam_size, breaking ties toward more durable layers.

        The layer tiebreaker is the migration-risk preference made concrete: a
        kernel-level win beats a framework-level win of equal magnitude,
        because one survives the XLA -> native-PyTorch migration.
        """
        return sorted(
            scored,
            key=lambda c: (-c.metric, LAYER_DURABILITY.get(c.layer, 9)),
        )[: self.beam_size]

    def exhausted(self, beam: list[Candidate]) -> bool:
        """True when every single-axis move from the beam has been seen."""
        for parent in beam:
            for axis, values in self.axes.items():
                for v in values:
                    if parent.config.get(axis) == v:
                        continue
                    if ({**parent.config, axis: v}.items().__iter__() and
                            tuple(sorted({**parent.config, axis: v}.items())) not in self._seen):
                        return False
        return True

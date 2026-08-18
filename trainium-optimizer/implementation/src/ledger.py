"""
Experiment ledger — the append-only record of every optimization attempt.

Every candidate the optimizer tries gets a row, kept or discarded. This is the
source of truth that the trajectory chart and the report are generated from,
and the substrate for the borrow-vs-invent metrics.

Schema follows `internal-prior-optimization-run`'s results.tsv, with
four additions: `stage`, `origin`, `layer`, and `source` — which are what make
the provenance and migration-risk queries possible.

Design notes:
  - TSV, not CSV. Descriptions contain commas constantly and quoting them is a
    recurring source of parse bugs. Tabs essentially never appear in prose.
  - Append-only. Rows are never edited or deleted. A wrong row gets a
    superseding row, so the history stays honest.
  - `commit` is load-bearing: it links each row to the diff that produced it,
    which is what lets the chart hyperlink every point to real code.

See ../../trajectory-reporting.md for the full artifact design.
"""

from __future__ import annotations

import csv
import subprocess
from dataclasses import asdict, dataclass, field, fields
from enum import StrEnum
from pathlib import Path


LEDGER_FILENAME = "results.tsv"


class Stage(StrEnum):
    """The optimization pipeline stages. See ../../optimization-stages.md."""

    BASELINE = "baseline"
    HARVEST = "harvest"
    CONFIG = "config"
    KNOWN_KERNEL = "known_kernel"
    BORROW = "borrow"
    INVENT = "invent"
    GRAPH_REWRITE = "graph_rewrite"


class Origin(StrEnum):
    """Where a change came from. Keeps the invention metric honest.

    A run that is mostly HARVESTED is a good outcome — the ecosystem already
    had the answer — but it is not invention, and merging the two would
    flatter us.
    """

    NONE = ""              # config-only changes have no kernel provenance
    HARVESTED = "harvested"    # existing AWS-maintained kernel, used as-is
    BORROWED = "borrowed"      # pattern ported from an external reference
    HYBRID = "hybrid"          # borrowed algorithm, restructured for Neuron
    INVENTED = "invented"      # novel, designed from profile + roofline


class Layer(StrEnum):
    """Which layer of the stack the change lives at.

    Determines whether it survives an XLA -> native-PyTorch migration.
    See ../../knowledge-bank.md#layer-tagging-and-migration-risk.
    """

    NONE = ""
    KERNEL = "kernel"          # NKI — below the framework boundary, durable
    COLLECTIVE = "collective"  # TP/CP/EP patterns — mostly durable
    CONFIG = "config"          # concepts durable, knob names not
    FRAMEWORK = "framework"    # vLLM/NxDI internals — often not durable
    GRAPH = "graph"            # XLA passes — likely not durable


class Status(StrEnum):
    KEEP = "keep"
    DISCARD = "discard"


# Migration risk is a pure function of layer, so derive it rather than storing
# a field that can drift out of sync.
MIGRATION_RISK: dict[Layer, str] = {
    Layer.NONE: "none",
    Layer.KERNEL: "low",
    Layer.COLLECTIVE: "low-medium",
    Layer.CONFIG: "medium",
    Layer.FRAMEWORK: "high",
    Layer.GRAPH: "high",
}

# Lower is more durable. Used as a proposer tiebreaker: prefer a kernel-level
# win over a framework-level win of equal magnitude, because one survives.
LAYER_DURABILITY: dict[Layer, int] = {
    Layer.KERNEL: 0,
    Layer.COLLECTIVE: 1,
    Layer.CONFIG: 2,
    Layer.FRAMEWORK: 3,
    Layer.GRAPH: 4,
    Layer.NONE: 5,
}


@dataclass
class Row:
    """One experiment. Field order here defines column order in the TSV."""

    commit: str
    stage: Stage
    origin: Origin
    layer: Layer
    source: str            # e.g. "nki-library@7f3a1b2", "vllm@a1b2c3d"; "" if n/a
    metric: float          # the primary metric — tok/s for track A
    mfu: float             # normalizes across model sizes; -1 if not computed
    correctness: float     # percent; the hard gate
    compile_s: float
    status: Status
    description: str

    def __post_init__(self) -> None:
        # Tabs and newlines would corrupt the TSV. Normalize rather than
        # raising, because a run should never die on a cosmetic issue.
        self.description = " ".join(str(self.description).split())

    @property
    def migration_risk(self) -> str:
        return MIGRATION_RISK[self.layer]

    @property
    def kept(self) -> bool:
        return self.status is Status.KEEP


HEADER: list[str] = [f.name for f in fields(Row)]


class Ledger:
    """Append-only TSV of experiments.

    Usage:
        led = Ledger(Path("optimization_runs/gemma-4-31b"))
        led.init(baseline_row)
        led.append(row)
        rows = led.read()
    """

    def __init__(self, run_dir: Path | str) -> None:
        self.run_dir = Path(run_dir)
        self.path = self.run_dir / LEDGER_FILENAME

    # -- write ---------------------------------------------------------------

    def init(self, baseline: Row | None = None) -> None:
        """Create the ledger with a header. Idempotent — will not clobber."""
        if self.path.exists():
            return
        self.run_dir.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", newline="") as fh:
            csv.writer(fh, delimiter="\t").writerow(HEADER)
        if baseline is not None:
            self.append(baseline)

    def append(self, row: Row) -> None:
        """Append one experiment. Every attempt is recorded, keep or discard."""
        if not self.path.exists():
            self.init()
        with self.path.open("a", newline="") as fh:
            w = csv.writer(fh, delimiter="\t")
            d = asdict(row)
            w.writerow([d[name] for name in HEADER])

    # -- read ----------------------------------------------------------------

    def read(self) -> list[Row]:
        if not self.path.exists():
            return []
        out: list[Row] = []
        with self.path.open(newline="") as fh:
            for rec in csv.DictReader(fh, delimiter="\t"):
                out.append(
                    Row(
                        commit=rec["commit"],
                        stage=Stage(rec["stage"]),
                        origin=Origin(rec["origin"]),
                        layer=Layer(rec["layer"]),
                        source=rec["source"],
                        metric=float(rec["metric"]),
                        mfu=float(rec["mfu"]),
                        correctness=float(rec["correctness"]),
                        compile_s=float(rec["compile_s"]),
                        status=Status(rec["status"]),
                        description=rec["description"],
                    )
                )
        return out

    # -- derived views -------------------------------------------------------

    def kept(self) -> list[Row]:
        """The winning path — what the trajectory line is drawn through."""
        return [r for r in self.read() if r.kept]

    def incumbent(self) -> Row | None:
        """Current best. Note this is max-by-metric, not last-kept: a KEEP is
        only recorded on improvement, but reading max is robust to a ledger
        that was hand-edited or merged."""
        keeps = self.kept()
        return max(keeps, key=lambda r: r.metric) if keeps else None

    def baseline(self) -> Row | None:
        for r in self.read():
            if r.stage is Stage.BASELINE:
                return r
        return None

    def speedup(self) -> float | None:
        base, best = self.baseline(), self.incumbent()
        if base is None or best is None or base.metric <= 0:
            return None
        return best.metric / base.metric

    def provenance_counts(self) -> dict[str, int]:
        """Promoted changes by origin. The invention metric's numerator."""
        counts = {o.value: 0 for o in Origin if o is not Origin.NONE}
        for r in self.kept():
            if r.origin is not Origin.NONE:
                counts[r.origin.value] += 1
        return counts

    def invention_stats(self) -> dict[str, float]:
        """Is Stage 4 productive, or flailing?

        Answers the open question of whether the system ever creates rather
        than copies. See ../../optimization-stages.md.
        """
        rows = self.read()
        attempted = [r for r in rows if r.stage is Stage.INVENT]
        promoted = [r for r in attempted if r.kept]
        promoted_any = self.kept()
        promoted_kernels = [r for r in promoted_any if r.origin is not Origin.NONE]

        return {
            "invention_attempts": len(attempted),
            "invention_promoted": len(promoted),
            "invention_win_rate": (
                len(promoted) / len(attempted) if attempted else 0.0
            ),
            "invention_rate": (
                len(promoted) / len(promoted_kernels) if promoted_kernels else 0.0
            ),
        }

    def compile_time_total_s(self) -> float:
        """Compile dominates the loop's cost. Worth reporting explicitly."""
        return sum(r.compile_s for r in self.read())

    def stage_summary(self) -> dict[str, dict[str, float]]:
        """Per-stage: attempts, promotions, and the gain the stage delivered.

        `gain` is measured as best-in-stage over best-before-stage, so it
        credits the stage rather than the individual candidate.
        """
        rows = self.read()
        summary: dict[str, dict[str, float]] = {}
        running_best = 0.0

        for stage in Stage:
            in_stage = [r for r in rows if r.stage is stage]
            if not in_stage:
                continue
            promoted = [r for r in in_stage if r.kept]
            best_here = max((r.metric for r in promoted), default=running_best)
            gain = (
                (best_here / running_best - 1.0)
                if running_best > 0 and best_here > running_best
                else 0.0
            )
            summary[stage.value] = {
                "attempts": len(in_stage),
                "promoted": len(promoted),
                "best_metric": best_here,
                "gain_pct": gain * 100.0,
                "compile_s": sum(r.compile_s for r in in_stage),
            }
            running_best = max(running_best, best_here)

        return summary


# -- git helpers -------------------------------------------------------------
# Git is the state machine: the branch head IS the incumbent, and DISCARD is
# a hard reset. See ../../optimization-stages.md#git-as-the-state-machine.


def current_commit(repo: Path | str = ".") -> str:
    """Short SHA of HEAD. Returns "unknown" outside a repo rather than raising,
    so a missing git context degrades the ledger instead of killing the run."""
    try:
        out = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True,
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def discard(repo: Path | str = ".") -> None:
    """Revert the last commit — the DISCARD half of keep/discard.

    Deliberately a hard reset: the branch head must always equal the
    incumbent, so a rejected candidate leaves no trace in the tree. The
    ledger keeps the record.
    """
    subprocess.run(
        ["git", "-C", str(repo), "reset", "--hard", "HEAD~1"],
        check=True, capture_output=True,
    )

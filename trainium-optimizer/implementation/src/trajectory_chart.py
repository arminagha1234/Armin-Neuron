"""
Trajectory chart — "how did it improve", generated from the ledger.

Modeled on `internal-prior-optimization-run`'s generate_chart.py,
with one significant change: that version hardcodes its data arrays
("simplified from ~100 experiments"). This reads the ledger directly, so the
chart can never drift from the record and it scales to 100 models.

Design choices carried over from the reference because they work:
  - dark GitHub palette, readable in a README and in dark-mode docs
  - points colored by stage, with vertical stage separators
  - gain annotations on meaningful jumps only, not every point
  - a star on the single largest gain
  - a red callout box counting the failures per stage
  - final result badge
  - subtitle carrying the conditions (without them the number is meaningless)

Added here:
  - roofline ceiling line, so the chart answers "how much headroom is left"
  - MFU on a secondary axis, showing whether gains were real efficiency
  - discarded candidates as faded X markers, so search width is visible
  - marker shape by provenance (harvested / borrowed / invented / config)

Usage:
    python -m trajectory_chart --run-dir optimization_runs/gemma-4-31b \\
        --model "Gemma 4 31B" --hardware trn2.48xlarge --tp 8 \\
        --shape "chat 1k/512" --sdk 2.28.0 --roofline 48000
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

from ledger import Ledger, Origin, Row, Stage  # noqa: E402


# -- theme -------------------------------------------------------------------

THEME = {
    "figure.facecolor": "#0d1117",
    "axes.facecolor": "#161b22",
    "axes.edgecolor": "#30363d",
    "text.color": "#c9d1d9",
    "axes.labelcolor": "#8b949e",
    "xtick.color": "#8b949e",
    "ytick.color": "#8b949e",
    "grid.color": "#21262d",
    "grid.alpha": 0.8,
    "font.family": "DejaVu Sans",
    "font.size": 10,
}

STAGE_COLOR: dict[Stage, str] = {
    Stage.BASELINE: "#8b949e",
    Stage.HARVEST: "#a371f7",
    Stage.CONFIG: "#58a6ff",
    Stage.KNOWN_KERNEL: "#39c5cf",
    Stage.BORROW: "#3fb950",
    Stage.INVENT: "#f0883e",
    Stage.GRAPH_REWRITE: "#f85149",
}

# Provenance is encoded as marker shape so it reads at a glance alongside the
# stage color.
ORIGIN_MARKER: dict[Origin, str] = {
    Origin.NONE: "s",        # square — config, no kernel provenance
    Origin.HARVESTED: "o",   # circle — off the shelf
    Origin.BORROWED: "^",    # triangle — ported from a reference
    Origin.HYBRID: "P",      # plus — borrowed then restructured
    Origin.INVENTED: "D",    # diamond — novel
}

# Below this, annotating every gain makes the chart unreadable.
ANNOTATE_THRESHOLD_PCT = 5.0

# Execution-mode encoding (eager vs torch.compile). Marker-edge ring colors.
COMPILED_EDGE = "#3fb950"   # bright green ring = torch.compile(backend="neuron")
EAGER_EDGE = "#6e7681"      # muted grey ring = eager

# First-forward time (seconds) above which we treat a point as compiled when
# the description carries no explicit compile_mode signal. Eager's lazy build
# is seconds; a NEFF compile is minutes.
COMPILE_S_HEURISTIC = 60.0


def _fmt_thousands(v: float, _pos: int | None = None) -> str:
    if v >= 1000:
        return f"{v / 1000:.1f}K"
    return f"{int(v)}"


def _stage_bands(kept: list[Row]) -> list[tuple[float, Stage]]:
    """X positions where the stage changes, for the separator lines."""
    bands: list[tuple[float, Stage]] = []
    prev: Stage | None = None
    for i, r in enumerate(kept):
        if r.stage is not prev:
            bands.append((i - 0.5, r.stage))
            prev = r.stage
    return bands


def build_chart(
    run_dir: Path,
    out_path: Path,
    model: str,
    hardware: str = "",
    tp: int | None = None,
    shape: str = "",
    sdk: str = "",
    roofline: float | None = None,
    metric_label: str = "tok/s",
) -> Path:
    led = Ledger(run_dir)
    rows = led.read()
    if not rows:
        raise SystemExit(f"ledger is empty: {led.path}")

    kept = [r for r in rows if r.kept]
    discarded = [r for r in rows if not r.kept]
    if not kept:
        raise SystemExit("no kept rows — nothing to plot")

    plt.rcParams.update(THEME)
    has_mfu = any(r.mfu >= 0 for r in kept)
    fig, ax = plt.subplots(figsize=(14, 7))
    fig.subplots_adjust(top=0.86, bottom=0.16, left=0.08, right=0.92)

    # -- title + conditions --------------------------------------------------
    fig.suptitle(
        f"{model} — Optimization Trajectory",
        fontsize=16, fontweight="bold", color="#f0f6fc", y=0.97,
    )
    cond = " │ ".join(
        p for p in (
            hardware,
            f"TP={tp}" if tp else "",
            shape,
            f"SDK {sdk}" if sdk else "",
            f"{min(r.correctness for r in kept):.0f}% correctness",
        ) if p
    )
    fig.text(0.5, 0.915, cond, ha="center", fontsize=9.5, color="#8b949e")

    xs = list(range(len(kept)))
    ys = [r.metric for r in kept]

    # -- the winning path ----------------------------------------------------
    ax.plot(xs, ys, color="#3fb950", linewidth=2, alpha=0.55, zorder=2)
    ax.fill_between(xs, ys, alpha=0.07, color="#3fb950", zorder=1)

    # Per-point execution mode (eager vs torch.compile). Only meaningful for
    # backends that expose a compile_mode axis (native PyTorch); a no-op for
    # mock/XLA runs that carry no such signal, so those charts are unchanged.
    modes = _classify_modes(kept) if _mode_signal_present(rows) else [None] * len(kept)

    for x, r, mode in zip(xs, kept, modes):
        # Compiled points get a bright ring; eager a muted ring. Stage color
        # stays the fill so all three encodings (stage/provenance/mode) coexist.
        edge = ("white" if mode is None
                else COMPILED_EDGE if mode == "compiled" else EAGER_EDGE)
        ax.scatter(
            x, r.metric,
            c=STAGE_COLOR[r.stage],
            marker=ORIGIN_MARKER[r.origin],
            s=95, zorder=4, edgecolors=edge,
            linewidths=0.6 if mode is None else 1.8,
        )
        # Explicit per-point text tag, in the mode's color, just below the point.
        if mode is not None:
            ax.annotate(
                "compiled" if mode == "compiled" else "eager",
                xy=(x, r.metric), xytext=(0, -13), textcoords="offset points",
                ha="center", va="top", fontsize=6.5,
                color=COMPILED_EDGE if mode == "compiled" else EAGER_EDGE,
                fontweight="bold" if mode == "compiled" else "normal",
            )

    # Prominent callout on the first eager -> compiled transition (the money jump).
    for i in range(1, len(modes)):
        if modes[i] == "compiled" and modes[i - 1] == "eager":
            prev, cur = kept[i - 1].metric, kept[i].metric
            pct = (cur / prev - 1.0) * 100.0 if prev > 0 else 0.0
            ax.annotate(
                f"torch.compile\n(backend=neuron)\n{'+' if pct >= 0 else ''}{pct:.0f}% vs eager",
                xy=(i, cur), xytext=(max(0, i - 1.6), cur + max(ys) * 0.09),
                fontsize=8.5, color=COMPILED_EDGE, fontweight="bold", ha="center",
                arrowprops=dict(arrowstyle="->", color=COMPILED_EDGE, lw=1.4),
            )
            break

    # -- discarded candidates, faded -----------------------------------------
    # Placed at the x of the kept row they were competing against, so the
    # width of the search is visible without implying they advanced the path.
    for r in discarded:
        x = _nearest_x_for_stage(kept, r.stage)
        if x is None:
            continue
        ax.scatter(
            x + 0.22, r.metric,
            marker="x", c="#f85149", s=42, alpha=0.32, zorder=3, linewidths=1.2,
        )

    # -- roofline ceiling ----------------------------------------------------
    if roofline:
        ax.axhline(roofline, color="#f0f6fc", linestyle=":", linewidth=1.3, alpha=0.55)
        attainment = ys[-1] / roofline * 100
        ax.text(
            len(kept) - 0.5, roofline,
            f"  roofline bound — {attainment:.0f}% attained",
            fontsize=8.5, color="#f0f6fc", alpha=0.7, va="bottom", ha="right",
        )

    # -- stage separators + labels ------------------------------------------
    bands = _stage_bands(kept)
    ymax = max(ys) * (1.30 if roofline is None else 1.18)
    for xpos, stage in bands:
        if xpos > 0:
            ax.axvline(xpos, color="#30363d", linestyle="--", linewidth=1, alpha=0.7)
        ax.text(
            xpos + 0.35, ymax * 0.965, stage.value,
            fontsize=8.5, color=STAGE_COLOR[stage], alpha=0.9, ha="left",
        )

    # -- gain annotations, meaningful jumps only ----------------------------
    gains: list[tuple[int, float, float]] = []
    for i in range(1, len(kept)):
        prev, cur = kept[i - 1].metric, kept[i].metric
        if prev > 0:
            pct = (cur / prev - 1.0) * 100.0
            if pct >= ANNOTATE_THRESHOLD_PCT:
                gains.append((i, cur, pct))

    star_i = max(gains, key=lambda g: g[2])[0] if gains else None
    for i, y, pct in gains:
        is_star = i == star_i
        ax.annotate(
            f"+{pct:.0f}%",
            xy=(i, y), xytext=(i, y + max(ys) * 0.055),
            fontsize=11 if is_star else 8.5,
            color="#f0883e" if is_star else "#8b949e",
            fontweight="bold" if is_star else "normal",
            ha="center",
        )

    if star_i is not None:
        r = kept[star_i]
        ax.annotate(
            f"★ largest single gain\n{r.stage.value}"
            + (f" / {r.origin.value}" if r.origin is not Origin.NONE else ""),
            xy=(star_i, kept[star_i].metric),
            xytext=(max(0, star_i - 2.0), kept[star_i].metric * 0.74),
            fontsize=9, color="#f0883e", fontweight="bold", ha="center",
            arrowprops=dict(arrowstyle="->", color="#f0883e", lw=1.5),
        )

    # -- failure callout -----------------------------------------------------
    # Failures belong on the chart. The reference implementation's
    # "~40 MoE kernel experiments: all <1%" is arguably more useful to the
    # next engineer than knowing which one worked.
    if discarded:
        by_stage: dict[str, int] = {}
        for r in discarded:
            by_stage[r.stage.value] = by_stage.get(r.stage.value, 0) + 1
        worst = max(by_stage.items(), key=lambda kv: kv[1])
        ax.text(
            len(kept) * 0.5, min(ys) * 0.9,
            f"{len(discarded)} discarded overall\n"
            f"({worst[1]} in {worst[0]})",
            fontsize=8.5, color="#f85149", ha="center", style="italic",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="#f8514912",
                      edgecolor="#f8514940"),
        )

    # -- final badge ---------------------------------------------------------
    # Anchored below-right of the last point and nudged down, so it never
    # collides with that point's gain annotation (which sits above it).
    speedup = led.speedup()
    badge = f"{ys[-1]:,.0f} {metric_label}"
    if speedup:
        badge += f"   ({speedup:.1f}× baseline)"
    ax.annotate(
        badge,
        xy=(len(kept) - 1, ys[-1]),
        xytext=(len(kept) - 1.4, ys[-1] - max(ys) * 0.11),
        fontsize=11, fontweight="bold", color="#f0f6fc",
        ha="right", va="top",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#a371f7",
                  edgecolor="none", alpha=0.92),
        arrowprops=dict(arrowstyle="-", color="#a371f7", lw=1.0, alpha=0.6),
    )

    # -- MFU secondary axis --------------------------------------------------
    if has_mfu:
        ax2 = ax.twinx()
        ax2.plot(
            xs, [r.mfu for r in kept],
            color="#f0883e", linewidth=1.2, linestyle="--", alpha=0.6, zorder=2,
        )
        ax2.set_ylabel("MFU %", color="#f0883e", fontsize=9)
        ax2.tick_params(axis="y", colors="#f0883e", labelsize=8)
        ax2.set_ylim(0, max(r.mfu for r in kept) * 1.9)
        ax2.grid(False)

    # -- axes ----------------------------------------------------------------
    ax.set_xticks(xs)
    ax.set_xticklabels(
        [_short_label(r.description) for r in kept], fontsize=7.5, rotation=38,
        ha="right",
    )
    ax.set_ylabel(metric_label)
    ax.set_ylim(min(ys) * 0.8, ymax)
    ax.yaxis.set_major_formatter(FuncFormatter(_fmt_thousands))
    ax.grid(True, axis="y", alpha=0.45)

    # -- legend --------------------------------------------------------------
    handles = [
        Line2D([], [], marker=m, color="none", markerfacecolor="#8b949e",
               markeredgecolor="white", markersize=8, label=o.value or "config")
        for o, m in ORIGIN_MARKER.items()
    ]
    handles.append(
        Line2D([], [], marker="x", color="#f85149", linestyle="none",
               markersize=7, label="discarded", alpha=0.5)
    )
    # Only advertise the eager/compiled ring encoding when it's actually in use.
    if _mode_signal_present(rows):
        handles.append(
            Line2D([], [], marker="o", color="none", markerfacecolor="#8b949e",
                   markeredgecolor=COMPILED_EDGE, markeredgewidth=1.8,
                   markersize=8, label="compiled (torch.compile)")
        )
        handles.append(
            Line2D([], [], marker="o", color="none", markerfacecolor="#8b949e",
                   markeredgecolor=EAGER_EDGE, markeredgewidth=1.8,
                   markersize=8, label="eager")
        )
    ax.legend(
        handles=handles, loc="lower right", fontsize=8, frameon=True,
        facecolor="#161b22", edgecolor="#30363d", ncol=3,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _nearest_x_for_stage(kept: list[Row], stage: Stage) -> int | None:
    """X position of the last kept row in this stage, for placing failures."""
    idxs = [i for i, r in enumerate(kept) if r.stage is stage]
    return idxs[-1] if idxs else None


def _mode_signal_present(rows: list[Row]) -> bool:
    """True if this run has any compile-mode signal at all. Keeps the eager/
    compiled labeling native-PyTorch-specific — mock/XLA runs (no compile_mode
    axis) get no labels and render exactly as before."""
    for r in rows:
        d = r.description.lower()
        if "compile_mode=" in d or "compile-default" in d or "eager" in d:
            return True
    return False


def _row_mode(desc: str) -> str | None:
    """Explicit mode from a row's provenance description, or None if it doesn't
    change the mode (e.g. a tp_degree=8 delta inherits the current mode)."""
    d = desc.lower()
    if "compile_mode=compile-default" in d or "compile-default" in d:
        return "compiled"
    if "compile_mode=eager" in d:
        return "eager"
    return None


def _classify_modes(kept: list[Row]) -> list[str]:
    """Label every kept point eager/compiled by carrying the mode forward along
    the winning path. Baseline is eager (the native backend's naive start);
    the mode flips only when a compile_mode candidate is promoted, and later
    points (tp/dtype changes) inherit whatever mode was in effect."""
    modes: list[str] = []
    current = "eager"                      # native baseline is eager
    for r in kept:
        explicit = _row_mode(r.description)
        if explicit is not None:
            current = explicit
        elif r.description.strip().lower() == "baseline":
            current = "eager"
        elif r.compile_s >= COMPILE_S_HEURISTIC and _row_mode(r.description) is None:
            # weak fallback: a minutes-long first forward implies a NEFF compile
            current = "compiled"
        modes.append(current)
    return modes


def _short_label(desc: str, width: int = 22) -> str:
    """Two-line compact axis labels, in the style of the reference chart."""
    if len(desc) <= width:
        return desc
    words, lines, cur = desc.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > width:
            lines.append(cur)
            cur = w
            if len(lines) == 2:
                break
        else:
            cur = f"{cur} {w}".strip()
    if cur and len(lines) < 2:
        lines.append(cur)
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", required=True, type=Path)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--model", required=True)
    ap.add_argument("--hardware", default="")
    ap.add_argument("--tp", type=int, default=None)
    ap.add_argument("--shape", default="")
    ap.add_argument("--sdk", default="")
    ap.add_argument("--roofline", type=float, default=None)
    ap.add_argument("--metric-label", default="tok/s")
    a = ap.parse_args()

    out = a.out or (a.run_dir / "optimization_timeline.png")
    p = build_chart(
        run_dir=a.run_dir, out_path=out, model=a.model, hardware=a.hardware,
        tp=a.tp, shape=a.shape, sdk=a.sdk, roofline=a.roofline,
        metric_label=a.metric_label,
    )
    print(f"wrote {p}")


if __name__ == "__main__":
    main()

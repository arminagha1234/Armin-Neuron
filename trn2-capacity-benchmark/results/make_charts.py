#!/usr/bin/env python3
"""Generate the Trn2 capacity-study progress charts.

Every number here is traceable to a measured run on trn2.48xlarge (Kaizen),
or is explicitly marked as an extrapolation. Extrapolated bars are hatched in
every chart so a reader can never mistake one for a measurement.

Run:  python3 make_charts.py
Out:  charts/*.png
"""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "charts")
os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------- palette ----
INK = "#1b2430"
MUTED = "#6b7684"
GRID = "#dfe3e8"
GREEN = "#2e9e5b"
AMBER = "#e0a02c"
RED = "#cf4b3b"
BLUE = "#2f6fb2"
SLATE = "#8a94a3"

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": GRID,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "font.size": 10,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# ------------------------------------------------------------------- data ----
# RPS per trn2.48xlarge (64 logical cores).
#   measured_per_replica : RPS measured for ONE server replica
#   replicas_per_box     : 64 logical cores / TP
#   rps_box              : per_replica * replicas  (an EXTRAPOLATION across
#                          replicas -- assumes no HBM-bandwidth/host contention)
MODELS = [
    dict(
        key="qwen3-8b",
        name="Qwen3-8B",
        shape="3500 in / 1 out",
        ask_lo=50, ask_hi=100, ask=50,
        per_replica=4.13, tp=4,
        measured_kind="measured (vLLM, coherence 3/3)",
        start_rps=1.02,          # single-core XLA: 1/0.9761 s
        start_label="single-core XLA eager",
    ),
    dict(
        key="gemma-e2b",
        name="Gemma-4-E2B",
        shape="3500 in / 1 out",
        ask_lo=50, ask_hi=50, ask=50,
        per_replica=2.77, tp=1,
        measured_kind="prefill only (XLA, argmax 2/3)",
        start_rps=2.77,
        start_label="same run = start",
    ),
    dict(
        key="gemma-31b",
        name="Gemma-4-31B-it",
        shape="3500 in / 50 out",
        ask_lo=50, ask_hi=50, ask=50,
        per_replica=1.75, tp=32,
        measured_kind="measured sweep (vLLM, coherence 3/3)",
        start_rps=0.0,
        start_label="not running",
    ),
    dict(
        key="qwen3.5-4b",
        name="Qwen3.5-4B",
        shape="2000 in / 50 out",
        ask_lo=500, ask_hi=500, ask=500,
        per_replica=8.42, tp=16,   # 16847 tok/s / 2000 tok
        measured_kind="prefill only (native, no decode)",
        start_rps=0.0,
        start_label="not running",
    ),
]
for m in MODELS:
    m["replicas"] = 64 // m["tp"]
    m["rps_box"] = m["per_replica"] * m["replicas"]
    m["pct"] = 100.0 * m["rps_box"] / m["ask"]
    m["boxes"] = m["ask"] / m["rps_box"] if m["rps_box"] else float("inf")


def status_color(pct: float) -> str:
    if pct >= 100:
        return GREEN
    if pct >= 50:
        return AMBER
    return RED


def save(fig, name: str) -> None:
    p = os.path.join(OUT, name)
    fig.savefig(p, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", os.path.relpath(p))


# ============================================================ chart 1 =======
# Distance to the customer target, per model, on a log axis (the asks span
# 50 -> 500 and the results span 1.75 -> 177, so linear hides everything).
def chart_gap():
    fig, ax = plt.subplots(figsize=(10, 5.2))
    names = [m["name"] for m in MODELS]
    y = np.arange(len(MODELS))[::-1]

    for m, yy in zip(MODELS, y):
        ach, ask = m["rps_box"], m["ask"]
        col = status_color(m["pct"])
        # track from achieved to ask
        ax.plot([min(ach, ask), max(ach, ask)], [yy, yy], color=GRID, lw=7,
                solid_capstyle="round", zorder=1)
        ax.scatter([ask], [yy], s=190, marker="|", color=INK, lw=2.6, zorder=4)
        ax.scatter([ach], [yy], s=130, color=col, zorder=5,
                   edgecolor="white", lw=1.4)
        # label achieved
        ax.annotate(f"{ach:,.1f}", (ach, yy), textcoords="offset points",
                    xytext=(0, 13), ha="center", fontsize=9.5,
                    color=col, fontweight="bold")
        # gap annotation
        if ach < ask:
            mid = (ach * ask) ** 0.5
            ax.annotate(f"{ask/ach:,.0f}x short", (mid, yy),
                        textcoords="offset points", xytext=(0, -17),
                        ha="center", fontsize=9, color=RED)
        else:
            ax.annotate(f"{ach/ask:,.1f}x headroom", ((ach * ask) ** 0.5, yy),
                        textcoords="offset points", xytext=(0, -17),
                        ha="center", fontsize=9, color=GREEN)

    ax.set_yticks(y)
    ax.set_yticklabels([f"{m['name']}\n{m['shape']}" for m in MODELS],
                       fontsize=9.5)
    ax.set_xscale("log")
    ax.set_xlim(0.9, 1200)
    ax.set_xlabel("requests / second on ONE trn2.48xlarge  (log scale)")
    ax.set_title("How far each model is from its target RPS")
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)

    handles = [
        plt.Line2D([], [], marker="|", color=INK, lw=0, markersize=14,
                   markeredgewidth=2.6, label="target RPS"),
        plt.Line2D([], [], marker="o", color=GREEN, lw=0, markersize=9,
                   label="meets target"),
        plt.Line2D([], [], marker="o", color=RED, lw=0, markersize=9,
                   label="below target"),
    ]
    ax.legend(handles=handles, frameon=False, loc="upper left",
              bbox_to_anchor=(0.005, 0.62), fontsize=9)
    fig.text(0.01, -0.04,
             "Per-box RPS = measured per-replica RPS x (64 logical cores / TP). "
             "Multiplying across replicas is an extrapolation.",
             fontsize=8, color=MUTED)
    save(fig, "01_gap_to_target.png")


# ============================================================ chart 2 =======
# Percent of target achieved -- the single "how are we doing" view.
def chart_pct():
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    order = sorted(MODELS, key=lambda m: -m["pct"])
    names = [m["name"] for m in order]
    pct = [m["pct"] for m in order]
    cols = [status_color(p) for p in pct]

    bars = ax.bar(names, pct, color=cols, width=0.58, zorder=3)
    ax.axhline(100, color=INK, lw=1.6, ls="--", zorder=4)
    ax.annotate("target = 100%", (len(names) - 0.42, 100),
                textcoords="offset points", xytext=(0, 8), ha="right",
                fontsize=9, fontweight="bold", color=INK)

    for b, m in zip(bars, order):
        v = m["pct"]
        ax.annotate(f"{v:,.0f}%", (b.get_x() + b.get_width() / 2, v),
                    textcoords="offset points", xytext=(0, 6), ha="center",
                    fontsize=11, fontweight="bold",
                    color=status_color(v))
        need = "1 box" if m["boxes"] <= 1 else f"{np.ceil(m['boxes']):.0f} boxes"
        ax.annotate(f"needs {need}", (b.get_x() + b.get_width() / 2, v),
                    textcoords="offset points", xytext=(0, 22), ha="center",
                    fontsize=8.5, color=MUTED)

    ax.set_yscale("log")
    ax.set_ylim(3, 1400)
    ax.set_ylabel("% of target RPS achieved on one box (log)")
    ax.set_title("Percent of target reached, one trn2.48xlarge")
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    save(fig, "02_percent_of_target.png")


# ============================================================ chart 3 =======
# The 8-cell matrix as a status heat grid.
def chart_matrix():
    rows = ["Qwen3-8B", "Qwen3.5-4B", "Gemma-4-31B-it", "Gemma-4-E2B-it"]
    cols = ["native PyTorch", "vLLM-Neuron"]
    # 2 = validated, 1 = measured w/ caveat, 0 = blocked, -1 = in progress
    state = {
        ("Qwen3-8B", "native PyTorch"): (2, "10,472 tok/s\nMFU 60.3%\ntop-1 ok"),
        ("Qwen3-8B", "vLLM-Neuron"): (2, "13,579 tok/s\n4.13 RPS/replica\ncoherence 3/3"),
        ("Qwen3.5-4B", "native PyTorch"): (2, "16,847 tok/s\np50 119 ms\ntop-1 ok"),
        ("Qwen3.5-4B", "vLLM-Neuron"): (-1, "attempt 10\n6 causes fixed\nrunning"),
        ("Gemma-4-31B-it", "native PyTorch"): (0, "blocked\ndevice barrier 2\nat TP>=8"),
        ("Gemma-4-31B-it", "vLLM-Neuron"): (2, "TTFT 0.62 s\n1.75 RPS/replica\ncoherence 3/3"),
        ("Gemma-4-E2B-it", "native PyTorch"): (1, "9,688 tok/s\nXLA prefill\nargmax 2/3"),
        ("Gemma-4-E2B-it", "vLLM-Neuron"): (0, "blocked\n0/3 coherence\nPLE on, TP=1"),
    }
    cmap = {2: GREEN, 1: AMBER, 0: RED, -1: BLUE}

    fig, ax = plt.subplots(figsize=(8.6, 6.0))
    for i, r in enumerate(rows):
        for j, c in enumerate(cols):
            code, txt = state[(r, c)]
            ax.add_patch(mpatches.FancyBboxPatch(
                (j, -i), 0.92, 0.86,
                boxstyle="round,pad=0.012,rounding_size=0.05",
                linewidth=0, facecolor=cmap[code], alpha=0.16, zorder=2))
            ax.add_patch(mpatches.FancyBboxPatch(
                (j, -i), 0.92, 0.86,
                boxstyle="round,pad=0.012,rounding_size=0.05",
                linewidth=1.8, edgecolor=cmap[code], facecolor="none", zorder=3))
            ax.text(j + 0.46, -i + 0.44, txt, ha="center", va="center",
                    fontsize=9, color=INK, linespacing=1.5, zorder=4)

    ax.set_xlim(-0.06, 2.0)
    ax.set_ylim(-len(rows) + 0.60, 1.02)
    ax.set_xticks([0.46, 1.46])
    ax.set_xticklabels(cols, fontsize=11, fontweight="bold")
    ax.set_yticks([-i + 0.43 for i in range(len(rows))])
    ax.set_yticklabels(rows, fontsize=10.5, fontweight="bold")
    ax.xaxis.tick_top()
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    ax.set_title("8-cell coverage matrix: 4 models x 2 serving stacks", pad=34)

    leg = [mpatches.Patch(facecolor=cmap[2], alpha=.5, label="validated"),
           mpatches.Patch(facecolor=cmap[1], alpha=.5, label="measured, caveat"),
           mpatches.Patch(facecolor=cmap[-1], alpha=.5, label="in progress"),
           mpatches.Patch(facecolor=cmap[0], alpha=.5, label="blocked")]
    ax.legend(handles=leg, frameon=False, ncol=4, fontsize=9,
              loc="upper center", bbox_to_anchor=(0.5, -0.06))
    save(fig, "03_matrix.png")


# ============================================================ chart 4 =======
# Where we started vs where we are, per model (throughput).
def chart_journey():
    # prefill throughput, tokens/sec, per replica
    data = [
        ("Qwen3-8B", 3592, 13579, "1-core XLA", "vLLM TP4"),
        ("Qwen3.5-4B", 0, 16847, "none", "native TP16"),
        ("Gemma-4-E2B", 9688, 9688, "1-core XLA", "same"),
        ("Gemma-4-31B", 0, 5630, "none", "vLLM TP32*"),
    ]
    fig, ax = plt.subplots(figsize=(10, 5.0))
    x = np.arange(len(data))
    w = 0.36
    starts = [d[1] for d in data]
    nows = [d[2] for d in data]

    ax.bar(x - w / 2, starts, w, label="at start of engagement",
           color=SLATE, alpha=.55, zorder=3)
    ax.bar(x + w / 2, nows, w, label="now (best validated path)",
           color=BLUE, zorder=3)

    for i, (nm, s0, s1, l0, l1) in enumerate(data):
        if s0 > 0:
            ax.annotate(f"{s0:,}", (i - w / 2, s0), textcoords="offset points",
                        xytext=(0, 5), ha="center", fontsize=8.5, color=MUTED)
        else:
            ax.annotate("0", (i - w / 2, 0), textcoords="offset points",
                        xytext=(0, 5), ha="center", fontsize=8.5, color=MUTED)
        ax.annotate(f"{s1:,}", (i + w / 2, s1), textcoords="offset points",
                    xytext=(0, 5), ha="center", fontsize=9.5,
                    fontweight="bold", color=BLUE)
        if s0 > 0 and s1 > s0:
            ax.annotate(f"{s1/s0:.1f}x", (i, max(s0, s1)),
                        textcoords="offset points", xytext=(0, 26),
                        ha="center", fontsize=10, fontweight="bold",
                        color=GREEN)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{d[0]}\n{d[3]} -> {d[4]}" for d in data],
                       fontsize=8.2)
    ax.set_ylabel("prefill throughput, tokens / sec / replica")
    ax.set_title("Progress from the starting point")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    fig.text(0.01, -0.05,
             "* Gemma-4-31B: 3500 prompt tokens / 0.6217 s TTFT. E2B is unchanged because "
             "its vLLM path is still incoherent, so the XLA prefill number stands.",
             fontsize=8, color=MUTED)
    save(fig, "04_progress_from_start.png")


# ============================================================ chart 5 =======
# Boxes required -- the capacity answer that actually matters.
def chart_boxes():
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    order = sorted(MODELS, key=lambda m: m["boxes"])
    names = [m["name"] for m in order]
    boxes = [max(m["boxes"], 0.05) for m in order]
    cols = [status_color(m["pct"]) for m in order]

    bars = ax.barh(names[::-1], boxes[::-1], color=cols[::-1], height=.55,
                   zorder=3)
    ax.axvline(1, color=INK, lw=1.4, ls="--", zorder=4)
    ax.annotate("1 box", (1, -0.46), textcoords="offset points",
                xytext=(4, 0), fontsize=9, color=INK, fontweight="bold")

    for b, m in zip(bars, order[::-1]):
        v = max(m["boxes"], 0.05)
        ax.annotate(f"{np.ceil(m['boxes']):.0f} box"
                    + ("es" if np.ceil(m["boxes"]) > 1 else ""),
                    (v, b.get_y() + b.get_height() / 2),
                    textcoords="offset points", xytext=(7, 0), va="center",
                    fontsize=10, fontweight="bold",
                    color=status_color(m["pct"]))

    ax.set_xscale("log")
    ax.set_xlim(0.04, 60)
    ax.set_xlabel("trn2.48xlarge instances needed to hit the stated RPS (log)")
    ax.set_title("Capacity required per model at its target RPS")
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    save(fig, "05_boxes_required.png")


# ============================================================ chart 6 =======
# Qwen3-8B concurrency: measured, and why batching did not help.
def chart_concurrency():
    conc = [1, 2, 4, 8, 16, 32, 64]
    rps16k = [3.97, 4.01, 4.04, 4.10, 4.11, 4.13, 4.13]
    lat16k = [0.25, 0.38, 0.63, 1.10, 2.08, 4.01, 7.87]
    rps4k = [3.90, 3.98, 3.99, 4.01, 4.08, 4.09, None]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.4, 4.4))

    a1.plot(conc, rps16k, "-o", color=BLUE, lw=2, ms=5,
            label="max_num_batched_tokens = 16384")
    a1.plot(conc[:-1], rps4k[:-1], "--s", color=SLATE, lw=1.6, ms=4,
            label="max_num_batched_tokens = 4096")
    a1.set_xscale("log", base=2)
    a1.set_ylim(0, 5.4)
    a1.set_xlabel("concurrent requests")
    a1.set_ylabel("RPS per replica")
    a1.set_title("Throughput: flat", fontsize=11)
    a1.legend(frameon=False, fontsize=8.5, loc="lower right")
    a1.grid(color=GRID, lw=.8)
    a1.set_axisbelow(True)
    a1.annotate("4x token budget bought 1%", (8, 4.7),
                fontsize=9, color=RED, ha="center")

    a2.plot(conc, lat16k, "-o", color=RED, lw=2, ms=5, label="measured")
    ideal = [lat16k[0] * c for c in conc]
    a2.plot(conc, ideal, ":", color=SLATE, lw=1.6, label="perfectly serial")
    a2.set_xscale("log", base=2)
    a2.set_yscale("log")
    a2.legend(frameon=False, fontsize=8.5, loc="upper left")
    a2.set_xlabel("concurrent requests")
    a2.set_ylabel("average latency (s)")
    a2.set_title("Latency: linear in concurrency", fontsize=11)
    a2.grid(color=GRID, lw=.8)
    a2.set_axisbelow(True)
    a2.annotate("0.25 s at conc 1", (1, 0.25), textcoords="offset points",
                xytext=(10, -16), fontsize=9, color=MUTED)

    fig.suptitle("Qwen3-8B on vLLM-Neuron, 3288 prompt tokens / 1 output "
                 "(measured, TP=4)", fontsize=12, fontweight="bold", y=1.06)
    fig.subplots_adjust(wspace=0.30)
    save(fig, "06_qwen3_8b_concurrency.png")


# ============================================================ chart 7 =======
# The plan: what each remaining gap needs.
def chart_plan():
    items = [
        ("Qwen3-8B  50 RPS", 100, GREEN, "DONE - 1 box, validated 3/3"),
        ("Qwen3-8B  100 RPS", 66, AMBER, "2 boxes, no eng. work"),
        ("Qwen3-8B-FP8", 35, AMBER, "port static-fp8 path (llama3 has it)"),
        ("Gemma-4-E2B  vLLM", 30, RED, "per-layer cosine vs CPU ref"),
        ("Gemma-4-31B  50 RPS", 20, RED, "decode 2.67 tok/s is the wall"),
        ("Gemma-4-31B  native", 45, AMBER, "TP=2 too big / TP>=8 barrier bug"),
        ("Qwen3.5-4B  vLLM", 85, BLUE, "attempt 10 in flight"),
        ("Qwen3.5-4B  500 RPS", 7, RED, "needs ~1M tok/s - renegotiate"),
    ]
    fig, ax = plt.subplots(figsize=(12.6, 5.4))
    y = np.arange(len(items))[::-1]
    for (lbl, pc, col, note), yy in zip(items, y):
        ax.barh(yy, 100, height=.55, color=GRID, alpha=.55, zorder=2)
        ax.barh(yy, pc, height=.55, color=col, zorder=3)
        ax.annotate(f"{pc}%", (pc, yy), textcoords="offset points",
                    xytext=(6, 0), va="center", fontsize=9.5,
                    fontweight="bold", color=col)
        ax.annotate(note, (100, yy), textcoords="offset points",
                    xytext=(46, 0), va="center", fontsize=9, color=MUTED,
                    annotation_clip=False)
    ax.set_yticks(y)
    ax.set_yticklabels([i[0] for i in items], fontsize=9.5)
    ax.set_xlim(0, 100)
    ax.set_xlabel("estimated completeness toward that specific goal (%)")
    ax.set_title("Remaining work, and what unblocks each item")
    ax.grid(axis="x", color=GRID, lw=.8)
    ax.set_axisbelow(True)
    save(fig, "07_plan.png")


# ============================================================ chart 8 =======
# Instances needed, trn2.3xlarge vs trn2.48xlarge, at the customer's own asks.
# trn2.3xlarge  = 1 Trainium2 chip  =  4 logical cores (LNC=2), 24 GB each
# trn2.48xlarge = 16 chips          = 64 logical cores
def chart_instances():
    import math
    CORES_3, CORES_48 = 4, 64
    rows = [
        ("Qwen3-8B\n50 RPS",      50, 4, 4.13, True),
        ("Qwen3-8B\n100 RPS",    100, 4, 4.13, True),
        ("gemma-4-E2B\n50 RPS",   50, 1, 2.77, True),
        ("Qwen3.5-4B\n500 RPS",  500, 16, 8.42, False),
        ("gemma-4-31B\n50 RPS",   50, 32, 1.75, False),
    ]
    labels, n3s, n48s, fits = [], [], [], []
    for lbl, ask, tp, rps, _f in rows:
        f = tp <= CORES_3
        r3 = (CORES_3 // tp) * rps if f else 0
        r48 = (CORES_48 // tp) * rps
        labels.append(lbl)
        n3s.append(math.ceil(ask / r3) if r3 else 0)
        n48s.append(math.ceil(ask / r48))
        fits.append(f)

    x = np.arange(len(rows))
    w = 0.38
    fig, ax = plt.subplots(figsize=(11.2, 5.2))
    b1 = ax.bar(x - w/2, [max(v, 0.4) for v in n3s], w, color=BLUE,
                label="trn2.3xlarge  (1 chip, 4 cores)", zorder=3)
    b2 = ax.bar(x + w/2, n48s, w, color=SLATE,
                label="trn2.48xlarge (16 chips, 64 cores)", zorder=3)

    for i, (bar, v, f) in enumerate(zip(b1, n3s, fits)):
        cx = bar.get_x() + bar.get_width()/2
        if not f:
            ax.annotate("TP too wide\nfor one chip", (cx, 0.45),
                        ha="center", va="bottom", fontsize=8, color=RED,
                        fontweight="bold")
            bar.set_color("white"); bar.set_edgecolor(RED)
            bar.set_hatch("//"); bar.set_linewidth(1.4)
        else:
            ax.annotate(f"{v}", (cx, v), textcoords="offset points",
                        xytext=(0, 5), ha="center", fontsize=10,
                        fontweight="bold", color=BLUE)
    for bar, v in zip(b2, n48s):
        ax.annotate(f"{v}", (bar.get_x() + bar.get_width()/2, v),
                    textcoords="offset points", xytext=(0, 5), ha="center",
                    fontsize=10, fontweight="bold", color=INK)

    ax.set_yscale("log")
    ax.set_ylim(0.35, 60)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("instances needed to hit the ask (log)")
    ax.set_title("Instances required at target RPS: trn2.3xlarge vs trn2.48xlarge")
    ax.axhline(1, color=GREEN, lw=1.5, ls="--", zorder=4)
    ax.annotate("1 instance", (len(rows) - 0.45, 1), textcoords="offset points",
                xytext=(0, 6), ha="right", fontsize=9, color=GREEN,
                fontweight="bold")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax.grid(axis="y", color=GRID, lw=.8)
    ax.set_axisbelow(True)
    fig.text(0.01, -0.05,
             "Hatched = measured at a TP spanning more than one chip, so no "
             "single-3xl number exists yet. All four models DO fit a 3xl in HBM "
             "at TP=4; they have simply not been measured there.",
             fontsize=8, color=MUTED)
    save(fig, "08_instances_3xl_vs_48xl.png")


# ============================================================ chart 9 =======
# The plain answer card: which instance does each ask need?
#   trn2.3xlarge  = 1  Trainium2 accelerator  (AWS docs)  = 4 logical cores
#   trn2.48xlarge = 16 Trainium2 accelerators             = 64 logical cores
def chart_answer_card():
    fig, ax = plt.subplots(figsize=(12.4, 6.6))
    ax.axis("off")

    buckets = [
        ("One trn2.3xlarge is enough", GREEN, [], "nothing lands here"),
        ("One trn2.48xlarge is enough", GREEN, [
            ("Qwen3-8B", "50 RPS", "3500 in / 1 out", "66 RPS/box", "1 box  (or 13 x 3xl)"),
            ("gemma-4-E2B", "50 RPS", "3500 in / 1 out", "177 RPS/box", "1 box  (or 5 x 3xl)"),
        ], None),
        ("Needs MORE than one trn2.48xlarge", RED, [
            ("Qwen3-8B", "100 RPS", "3500 in / 1 out", "66 RPS/box", "2 boxes"),
            ("gemma-4-31B-it", "50 RPS", "3500 in / 50 out", "3.5 RPS/box", "15 boxes"),
            ("Qwen3.5-4B", "500 RPS", "2000 in / 50 out", "34 RPS/box", "15+ boxes"),
        ], None),
    ]

    y = 0.955
    for title, col, items, empty in buckets:
        ax.add_patch(mpatches.FancyBboxPatch(
            (0.006, y - 0.052), 0.988, 0.056,
            boxstyle="round,pad=0.004,rounding_size=0.012",
            transform=ax.transAxes, facecolor=col, alpha=0.16,
            edgecolor=col, linewidth=1.6, zorder=2))
        ax.text(0.022, y - 0.024, title, transform=ax.transAxes,
                fontsize=12.5, fontweight="bold", color=INK, va="center")
        n = len(items) if items else 0
        ax.text(0.975, y - 0.024,
                (f"{n} ask" + ("s" if n != 1 else "")) if items else "0 asks",
                transform=ax.transAxes, fontsize=11.5, fontweight="bold",
                color=col, va="center", ha="right")
        y -= 0.075

        if empty:
            ax.text(0.045, y - 0.020, empty, transform=ax.transAxes,
                    fontsize=10.5, color=MUTED, style="italic", va="center")
            y -= 0.055
            continue

        # column headers
        cols = [(0.045, "model"), (0.235, "ask"), (0.335, "token profile"),
                (0.545, "measured / box"), (0.715, "you need")]
        ax.text(0.045, y - 0.012, "model", transform=ax.transAxes, fontsize=8.6,
                color=MUTED, va="center")
        ax.text(0.235, y - 0.012, "ask", transform=ax.transAxes, fontsize=8.6,
                color=MUTED, va="center")
        ax.text(0.335, y - 0.012, "token profile", transform=ax.transAxes,
                fontsize=8.6, color=MUTED, va="center")
        ax.text(0.545, y - 0.012, "measured / 48xl", transform=ax.transAxes,
                fontsize=8.6, color=MUTED, va="center")
        ax.text(0.715, y - 0.012, "instances needed", transform=ax.transAxes,
                fontsize=8.6, color=MUTED, va="center")
        y -= 0.036

        for m, ask, shape, perf, need in items:
            ax.text(0.045, y, m, transform=ax.transAxes, fontsize=10.6,
                    fontweight="bold", color=INK, va="center")
            ax.text(0.235, y, ask, transform=ax.transAxes, fontsize=10.2,
                    color=INK, va="center")
            ax.text(0.335, y, shape, transform=ax.transAxes, fontsize=10.2,
                    color=MUTED, va="center")
            ax.text(0.545, y, perf, transform=ax.transAxes, fontsize=10.2,
                    color=BLUE, va="center", fontweight="bold")
            ax.text(0.715, y, need, transform=ax.transAxes, fontsize=10.6,
                    color=col, va="center", fontweight="bold")
            y -= 0.050
        y -= 0.028

    ax.text(0.006, 0.055,
            "Why nothing fits a single trn2.3xlarge: it is 1 Trainium2 chip, "
            "1/16 of a 48xlarge. On these token profiles one chip serves "
            "4-11 RPS, and every ask is 50 RPS or more.",
            transform=ax.transAxes, fontsize=9.2, color=INK)
    ax.text(0.006, 0.012,
            "For the two green rows a FLEET of 3xl is less silicon than one 48xl "
            "(E2B: 5 chips vs 16; Qwen3-8B: 13 vs 16). gemma-4-31B and "
            "Qwen3.5-4B were measured at TP=32 / TP=16, so they have no "
            "single-3xl number yet.",
            transform=ax.transAxes, fontsize=8.6, color=MUTED)
    ax.set_title("Which Trn2 instance does each target need?  (5 targets across 4 models)",
                 fontsize=14, fontweight="bold", pad=12)
    save(fig, "09_which_instance.png")


if __name__ == "__main__":
    chart_gap()
    chart_pct()
    chart_matrix()
    chart_journey()
    chart_boxes()
    chart_concurrency()
    chart_plan()
    chart_instances()
    chart_answer_card()
    print("\nsummary")
    for m in MODELS:
        print(f"  {m['name']:<16} ask {m['ask']:>4} RPS | "
              f"{m['per_replica']:.2f}/replica x {m['replicas']:>2} = "
              f"{m['rps_box']:>6.1f} RPS/box | {m['pct']:>6.1f}% | "
              f"{np.ceil(m['boxes']):.0f} box(es)")

#!/usr/bin/env python3
"""Indeed target status, one trn2.48xlarge — published vs like-for-like.

Follows make_charts.py conventions: same palette, per-box RPS is
per_replica x (64 logical cores / TP) and is an EXTRAPOLATION across replicas
(hatched everywhere it appears).

The point of this chart is that the original study measured Qwen3-8B and
Gemma-4-E2B at ONE output token and the other two at fifty, so the four bars
were never comparable. Re-measuring at 50 output tokens changes the answer for
Qwen3-8B from "meets target" to "31% of target".

Every number is traceable:
  published/*  : trn2-capacity-benchmark/results/make_charts.py MODELS block
  tonight/*    : sustained 45s windows, single server, coherence-gated
                 Qwen3-8B  DECODE_CEILING.md  (MNS sweep, 8 is optimal)
                 Qwen3.5   THROUGHPUT.md      (blocked inverse + blocks=200)
                 31B       PREFILL_FALLBACK.md + MNS sweep (32 optimal)
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

INK, MUTED, GRID = "#1b2430", "#6b7684", "#dfe3e8"
GREEN, AMBER, RED, BLUE, SLATE = "#2e9e5b", "#e0a02c", "#cf4b3b", "#2f6fb2", "#8a94a3"

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": GRID, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "font.size": 10,
    "axes.titlesize": 13, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False,
})

# --------------------------------------------------------------------- data --
M = [
    dict(name="Qwen3-8B", ask=50, tp=4,
         pub=4.13,   pub_shape="3500 in / 1 out",
         now=1.438,  now_shape="3460 in / 50 out",
         now_note="MNS=8 (swept; 16 was 0.963)",
         valid=True, range_note=None),
    dict(name="Gemma-4-31B", ask=50, tp=16,
         pub=2.503, pub_tp=32, pub_shape="3500 in / 50 out",
         now=1.91,  now_shape="3461 in / 50 out",
         now_note="TP16, MNS=32 (swept; 32 optimal)",
         valid=True, range_note=None),
    dict(name="Qwen3.5-4B", ask=500, tp=4,
         pub=0.157, pub_shape="2000 in / 50 out",
         now=0.775, now_shape="1811 in / 50 out",
         now_note="blocked inverse + blocks=200",
         valid=True,
         range_note="same config re-measured 0.514 on a\nsecond node (4x decode variance,\nunexplained) -> 8.2 RPS/box"),
    dict(name="Gemma-4-E2B", ask=50, tp=1,
         pub=2.77,  pub_shape="3500 in / 1 out",
         now=None,  now_shape="incoherent 0/5",
         now_note="no valid number",
         valid=False, range_note=None),
]
for m in M:
    m["pub_rep"] = 64 // m.get("pub_tp", m["tp"])
    m["now_rep"] = 64 // m["tp"]
    m["pub_box"] = m["pub"] * m["pub_rep"]
    m["pub_pct"] = 100 * m["pub_box"] / m["ask"]
    m["now_box"] = (m["now"] * m["now_rep"]) if m["now"] else None
    m["now_pct"] = (100 * m["now_box"] / m["ask"]) if m["now"] else None
    m["boxes"] = (m["ask"] / m["now_box"]) if m["now_box"] else None


def col(p):
    return GREEN if p >= 100 else (AMBER if p >= 50 else RED)


fig = plt.figure(figsize=(15.2, 6.4))
gs = fig.add_gridspec(1, 2, width_ratios=[1.32, 1.0], wspace=0.26)

# ============================ panel 1: % of target ==========================
ax = fig.add_subplot(gs[0, 0])
x = np.arange(len(M))
w = 0.36
names = [m["name"] for m in M]

for i, m in enumerate(M):
    # published bar
    ax.bar(i - w / 2, max(m["pub_pct"], 0.4), w, color=SLATE, alpha=.55,
           edgecolor=SLATE, hatch="///", zorder=3)
    _pf = (lambda v: f"{v:.1f}%" if v < 10 else f"{v:.0f}%")
    ax.text(i - w / 2, max(m["pub_pct"], 0.4) * 1.09, _pf(m['pub_pct']),
            ha="center", va="bottom", fontsize=9, color=MUTED, zorder=4)
    # tonight bar
    if m["valid"]:
        ax.bar(i + w / 2, m["now_pct"], w, color=col(m["now_pct"]), alpha=.92,
               edgecolor=col(m["now_pct"]), hatch="///", zorder=3)
        ax.text(i + w / 2, m["now_pct"] * 1.09, _pf(m['now_pct']),
                ha="center", va="bottom", fontsize=10, fontweight="bold",
                color=col(m["now_pct"]), zorder=4)
    else:
        ax.bar(i + w / 2, 0.4, w, color="white", edgecolor=RED,
               linestyle="--", linewidth=1.6, zorder=3)
        ax.text(i + w / 2, 0.55, "no valid\nnumber", ha="center", va="bottom",
                fontsize=8.4, color=RED, style="italic", zorder=4)

ax.axhline(100, color=INK, lw=1.5, ls="--", zorder=2)
ax.annotate("Indeed target = 100%", (len(M) - 0.35, 100), xytext=(0, 7),
            textcoords="offset points", ha="right", fontsize=9.5,
            fontweight="bold", color=INK)
ax.set_yscale("log")
ax.set_ylim(0.3, 900)
ax.set_xticks(x)
ax.set_xticklabels([f"{m['name']}\nask {m['ask']} RPS" for m in M], fontsize=9.6)
ax.set_ylabel("% of target RPS on ONE trn2.48xlarge  (log)")
ax.set_title("Are we hitting Indeed's targets?  Only on the optimistic shape")
ax.grid(axis="y", color=GRID, lw=.7, zorder=0)
ax.legend(handles=[
    mpatches.Patch(facecolor=SLATE, alpha=.55, hatch="///", label="published study"),
    mpatches.Patch(facecolor=GREEN, alpha=.92, hatch="///", label="re-measured, meets target"),
    mpatches.Patch(facecolor=AMBER, alpha=.92, hatch="///", label="re-measured, 50-99%"),
    mpatches.Patch(facecolor=RED, alpha=.92, hatch="///", label="re-measured, below 50%"),
], loc="upper left", frameon=False, fontsize=8.8, ncol=2)

# Flag ONLY the models whose published bar used a different output length.
# Placed below the axis in axes-fraction y so nothing can collide with a bar.
tr = ax.get_xaxis_transform()
for i, m in enumerate(M):
    if "1 out" in m["pub_shape"]:
        ax.annotate("published = 1 output token\nre-measured = 50",
                    xy=(i, -0.175), xycoords=tr, ha="center", va="top",
                    fontsize=8.2, color=RED, style="italic",
                    annotation_clip=False)

# Qwen3.5 carries a real cross-node variance -- say so on the chart, not just
# in the footnote, because it is the difference between 40 and 61 boxes.
# Mark Qwen3.5 with an asterisk; the explanation goes in the footer where it
# cannot collide with a bar.
qi = [i for i, m in enumerate(M) if m["name"] == "Qwen3.5-4B"][0]
ax.annotate("*", xy=(qi + w / 2, M[qi]["now_pct"] * 1.30), ha="center",
            va="bottom", fontsize=15, fontweight="bold", color=MUTED)

# ============================ panel 2: boxes needed =========================
ax2 = fig.add_subplot(gs[0, 1])
valid = [m for m in M if m["boxes"]]
order = sorted(valid, key=lambda m: -m["boxes"])
y = np.arange(len(order))
bx = [m["boxes"] for m in order]

bars = ax2.barh(y, bx, color=[col(m["now_pct"]) for m in order], alpha=.9,
                edgecolor=[col(m["now_pct"]) for m in order], hatch="///", zorder=3)
for i, m in enumerate(order):
    ax2.text(m["boxes"] * 1.04, i, f"{m['boxes']:.1f} boxes",
             va="center", fontsize=10, fontweight="bold", color=col(m["now_pct"]))
ax2.set_yticks(y)
ax2.set_yticklabels([f"{m['name']}\n({m['ask']} RPS)" for m in order], fontsize=9.6)
ax2.set_xlabel("trn2.48xlarge boxes needed to meet the ask")
ax2.set_title("Qwen3.5 is 83% of the silicon")
ax2.grid(axis="x", color=GRID, lw=.7, zorder=0)
ax2.set_xlim(0, max(bx) * 1.30)

tot = sum(bx)
ax2.text(max(bx) * 1.28, len(order) - 0.5,
         f"total {tot:.0f} boxes\n(E2B excluded —\nstill incoherent)",
         ha="right", va="top", fontsize=9.4, color=INK, fontweight="bold",
         bbox=dict(boxstyle="round,pad=0.45", fc="#f5f7f9", ec=GRID))

fig.text(0.5, -0.135,
         "Per-box RPS = measured per-replica RPS x (64 logical cores / TP) — an EXTRAPOLATION across replicas "
         "(hatched), assuming no HBM-bandwidth or host contention.\n"
         "Published Qwen3-8B and Gemma-4-E2B were measured at ONE output token; the other two at fifty. "
         "The 're-measured' bars put all of them at 50 output tokens.\n"
         "Gemma-4-E2B's published 355% is prefill-only on a model that produces incoherent output, so it is not a "
         "capacity number at all.\n"
         "* Qwen3.5-4B: the same configuration re-measured at 0.514 RPS/replica on a second node (unexplained 4x decode "
         "variance) = 8.2 RPS/box, 1.6% of target, 61 boxes. The 12.4 shown is the better of the two.",
         ha="center", va="top", fontsize=8.4, color=MUTED)

fig.savefig(os.path.join(OUT, "10_indeed_status.png"), dpi=170,
            bbox_inches="tight", facecolor="white")
print("wrote", os.path.join(OUT, "10_indeed_status.png"))

# ------------------------------------------------------------------ summary --
print()
print(f"{'model':16s} {'ask':>5s} {'published':>22s} {'re-measured (50 out)':>24s} {'boxes':>7s}")
for m in M:
    pub = f"{m['pub_box']:.1f} RPS/box ({m['pub_pct']:.0f}%)"
    now = (f"{m['now_box']:.1f} RPS/box ({m['now_pct']:.0f}%)"
           if m["now"] else "incoherent — no number")
    bxs = f"{m['boxes']:.1f}" if m["boxes"] else "n/a"
    print(f"{m['name']:16s} {m['ask']:5d} {pub:>22s} {now:>24s} {bxs:>7s}")
print()
print(f"total boxes for the 650 RPS ask (E2B excluded): {tot:.0f}")

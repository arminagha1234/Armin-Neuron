#!/usr/bin/env python3
"""Render the no-APC (cold) TTFT charts for Gemma4-31B on Trainium2.

Regenerates the assets/*.png from the measured numbers below. Pure matplotlib.
    python3 make_perf_chart.py

All numbers: trn2.48xlarge, TP=32, bf16 KV, APC OFF, unique random prompt per
request (true cold prefill), 40 output tokens, public GA vLLM-Neuron 0.21.
- <=16k: single-shot prefill.
- 32k/64k: segmented prefill with the VALIDATED SWA windowed-prior fix
  (patches/patch_swa_window_prior_v2.py) -> the "best" long-context numbers.
  Full-span (pre-fix) baseline shown for contrast.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

TRN2_C = "#ff6a00"   # Trainium orange (best/optimized)
BASE_C = "#9aa0a6"   # gray (full-span baseline)
SLA_C = "#2ca02c"    # SLA green

SIZES = ["4k", "8k", "16k", "32k", "64k"]
CONC_FULL = [1, 2, 4, 8, 16, 32]
CONC_LONG = [1, 2, 4]

# TTFT seconds (mean). <=16k single-shot; 32k/64k = SWA-windowed (best).
TTFT = {
    "4k":  [0.224, 0.331, 0.605, 1.878, 4.983, 11.459],
    "8k":  [0.390, 0.585, 0.969, 2.606, 6.406, 14.176],
    "16k": [0.754, 1.127, 1.944, 4.215, 9.429, 19.901],
    "32k": [2.046, 3.051, 7.132],     # windowed (best)
    "64k": [4.064, 6.071, 11.991],    # windowed (best)
}
# Full-span (pre-fix) baseline for 32k/64k, conc 1/2/4.
BASE = {
    "32k": [3.056, 4.589, 8.335],
    "64k": [6.089, 9.151, 15.938],
}
BEST_C1 = {s: TTFT[s][0] for s in SIZES}          # conc=1 best per size
BASE_C1 = {"32k": BASE["32k"][0], "64k": BASE["64k"][0]}

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
os.makedirs(OUT, exist_ok=True)


def _lbl(ax, bars, fmt="{:.2f}s"):
    for b in bars:
        h = b.get_height()
        ax.annotate(fmt.format(h), (b.get_x() + b.get_width() / 2, h),
                    textcoords="offset points", xytext=(0, 3), ha="center",
                    fontsize=8.5, fontweight="bold")


# ---- Chart 1: headline — cold TTFT by input size (conc=1), best config ----
def chart_by_size():
    fig, ax = plt.subplots(figsize=(9, 5.2))
    x = np.arange(len(SIZES))
    vals = [BEST_C1[s] for s in SIZES]
    bars = ax.bar(x, vals, color=TRN2_C, width=0.62, zorder=3, label="best (optimized)")
    _lbl(ax, bars)
    # show full-span baseline for 32k/64k as hollow bars behind
    for i, s in enumerate(SIZES):
        if s in BASE_C1:
            ax.bar(x[i], BASE_C1[s], color="none", edgecolor=BASE_C, width=0.62,
                   linewidth=1.6, linestyle="--", zorder=2)
            ax.annotate(f"full-span {BASE_C1[s]:.2f}s", (x[i], BASE_C1[s]),
                        textcoords="offset points", xytext=(0, 3), ha="center",
                        fontsize=7.5, color=BASE_C)
            ax.annotate("-33%", (x[i], BEST_C1[s]), textcoords="offset points",
                        xytext=(0, 18), ha="center", fontsize=9, color=TRN2_C, fontweight="bold")
    ax.axhline(0.5, color=SLA_C, ls="--", lw=1.4, zorder=1)
    ax.annotate("500 ms SLA", (len(SIZES) - 0.5, 0.5), color=SLA_C, fontsize=9,
                va="bottom", ha="right")
    ax.set_xticks(x); ax.set_xticklabels(SIZES)
    ax.set_ylabel("TTFT (s)"); ax.set_xlabel("input size")
    ax.set_title("Gemma4-31B cold TTFT by input size — Trainium2 TP=32, bf16, no-APC (conc=1)\n"
                 "<=16k single-shot · 32k/64k SWA-windowed (dashed = full-span baseline)",
                 fontsize=11.5, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3, zorder=0)
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    p = os.path.join(OUT, "ttft_cold_by_size.png")
    fig.savefig(p, dpi=140); print("wrote", p)


# ---- Chart 2: the windowing win — baseline vs windowed at 32k/64k ----
def chart_win():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for ax, s in zip(axes, ["32k", "64k"]):
        x = np.arange(len(CONC_LONG)); w = 0.38
        b1 = ax.bar(x - w / 2, BASE[s], w, color=BASE_C, label="full-span (baseline)", zorder=3)
        b2 = ax.bar(x + w / 2, TTFT[s][:3], w, color=TRN2_C, label="SWA-windowed (best)", zorder=3)
        _lbl(ax, b1); _lbl(ax, b2)
        for i in range(len(CONC_LONG)):
            d = 100 * (BASE[s][i] - TTFT[s][i]) / BASE[s][i]
            ax.annotate(f"-{d:.0f}%", (x[i] + w / 2, TTFT[s][i]), textcoords="offset points",
                        xytext=(0, 16), ha="center", fontsize=8.5, color=TRN2_C, fontweight="bold")
        ax.set_xticks(x); ax.set_xticklabels([f"c={c}" for c in CONC_LONG])
        ax.set_title(f"{s} input", fontsize=12, fontweight="bold")
        ax.set_ylabel("TTFT (s)"); ax.set_xlabel("concurrency")
        ax.grid(True, axis="y", alpha=0.3, zorder=0)
        ax.legend(fontsize=9)
    fig.suptitle("SWA windowed-prior fix — long-context cold TTFT (token-parity validated)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    p = os.path.join(OUT, "ttft_windowing_win.png")
    fig.savefig(p, dpi=140); print("wrote", p)


# ---- Chart 3: concurrency sweep (best config), all sizes ----
def chart_sweep():
    fig, ax = plt.subplots(figsize=(9, 5.6))
    colors = ["#1f77b4", "#2ca02c", "#9467bd", "#ff6a00", "#d62728"]
    for s, c in zip(SIZES, colors):
        conc = CONC_FULL if len(TTFT[s]) == 6 else CONC_LONG
        ax.plot(conc, [v * 1000 for v in TTFT[s]], "-o", color=c, lw=2.2, ms=6, label=s, zorder=3)
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_xticks(CONC_FULL); ax.set_xticklabels(CONC_FULL)
    ax.axhline(500, color=SLA_C, ls="--", lw=1.3)
    ax.annotate("500 ms SLA", (32, 500), color=SLA_C, fontsize=9, va="bottom", ha="right")
    ax.set_xlabel("concurrency (requests)"); ax.set_ylabel("TTFT (ms, log)")
    ax.set_title("Gemma4-31B cold TTFT vs concurrency — Trainium2 TP=32, bf16, no-APC\n"
                 "<=16k single-shot · 32k/64k SWA-windowed (best) · lower is better",
                 fontsize=11.5, fontweight="bold")
    ax.grid(True, which="major", alpha=0.3)
    ax.legend(title="input size", fontsize=9)
    fig.tight_layout()
    p = os.path.join(OUT, "ttft_cold_conc_sweep.png")
    fig.savefig(p, dpi=140); print("wrote", p)


if __name__ == "__main__":
    chart_by_size()
    chart_win()
    chart_sweep()

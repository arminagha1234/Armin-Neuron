#!/usr/bin/env python3
"""TP-scaling charts for Gemma4-31B cold TTFT: Trn2 TP32/TP16/TP8 vs H100.

Regenerates assets/ttft_tp_scaling_conc1_lnc2.png and assets/ttft_tp_vs_h100_lnc2.png.
    python3 make_perf_chart_tp.py

Trn2 = trn2.48xlarge, bf16 KV, APC OFF, unique random prompt per request (cold),
40 output tokens, public GA vLLM-Neuron 0.21 (SDK 2.31 / cc 2.26).
Topology: LNC=2 (logical-neuroncore-config=2) -> 4 logical cores/chip, 96GB/chip
= 24GB/core. TP maps 1 rank -> 1 core, so TP32=8 chips, TP16=4 chips, TP8=2 chips.
- <=8k single-shot; 16k TP32 single-shot / TP16 segmented; 32k/64k SWA-windowed.
- TP8 (2 chips): 4k/8k only — long-context 66560-config HBM-OOMs (27.45GB > 24GB/core).

GPU baseline (comparable-scale H100 config):
- "H100 2×80GB (TP2)": 2× H100 80GB, TP=2 — a comparable-scale GPU config. Trn2 TP32
  BEATS it at every size 4k/8k/16k/32k/64k
  (0.224/0.390/0.754/2.046/4.064 vs 0.240/0.461/1.008/2.377/5.778).
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CONC = [1, 2, 4, 8, 16, 32]
CL = [1, 2, 4]
SIZES = ["4k", "8k", "16k", "32k", "64k"]

# TTFT seconds (mean). None = not available.
TP32 = {"4k": [0.224,0.331,0.605,1.878,4.983,11.459], "8k": [0.390,0.585,0.969,2.606,6.406,14.176],
        "16k": [0.754,1.127,1.944,4.215,9.429,19.901], "32k": [2.046,3.051,7.132], "64k": [4.064,6.071,11.991]}
TP16 = {"4k": [0.307,0.449,0.744,2.36,5.852,13.323], "8k": [0.557,0.833,1.384,3.433,8.013,17.339],
        "16k": [1.515,2.271,5.89,14.1,31.065,65.229], "32k": [3.01,4.514,9.519], "64k": [6.056,9.09,17.01]}
TP8  = {"4k": [0.572,0.855,2.56,6.201,14.155,29.574], "8k": [1.126,1.684,3.853,8.704,18.612,38.636],
        "16k": None, "32k": None, "64k": None}   # long-ctx OOM
# 2x H100 80GB, TP=2 (comparable-scale GPU config). 4k/8k/16k out=40; 32k/64k longer output
# (TTFT is first-token latency, so output length doesn't affect it).
H100_TP2 = {"4k": [0.240,0.359,0.637,1.139,2.075,3.964], "8k": [0.461,0.694,1.197,2.157,4.100,8.021],
            "16k": [1.008,1.510,2.569,4.651,8.812,16.894], "32k": [2.377,3.565,5.927,10.714,20.192,42.746],
            "64k": [5.778,8.699,14.594,26.003,48.894,94.487]}

# order = bar order left->right; H100 baseline last
SERIES = [("TP32", TP32), ("TP16", TP16), ("TP8", TP8), ("H100_TP2", H100_TP2)]
COLORS = {"TP32": "#ff6a00", "TP16": "#d1691f", "TP8": "#8a4b1f", "H100_TP2": "#1f77b4"}
LABELS = {"TP32": "Trn2 TP32 (8 chips)", "TP16": "Trn2 TP16 (4 chips)", "TP8": "Trn2 TP8 (2 chips)",
          "H100_TP2": "H100 ×2 80GB (TP2)"}
OFF = {"TP32": -1.5, "TP16": -0.5, "TP8": 0.5, "H100_TP2": 1.5}
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
os.makedirs(OUT, exist_ok=True)


def c1(d, s):
    v = d.get(s)
    return v[0] if v else None


# ---- Chart 1: conc=1 TP-scaling bars (per size), incl. H100 ×2 baseline ----
def chart_conc1():
    fig, ax = plt.subplots(figsize=(13, 6.5))
    x = np.arange(len(SIZES)); w = 0.2
    for name, d in SERIES:
        vals = [c1(d, s) for s in SIZES]
        xs = [x[i] + OFF[name] * w for i in range(len(SIZES)) if vals[i] is not None]
        ys = [v for v in vals if v is not None]
        bars = ax.bar(xs, ys, w, label=LABELS[name], color=COLORS[name], zorder=3)
        for b, y in zip(bars, ys):
            ax.annotate(f"{y:.2f}", (b.get_x() + b.get_width() / 2, y), textcoords="offset points",
                        xytext=(0, 3), ha="center", fontsize=8, fontweight="bold", rotation=90)
    for s in ["16k", "32k", "64k"]:
        i = SIZES.index(s)
        ax.annotate("TP8\nOOM", (x[i] + OFF["TP8"] * w, 0.28), ha="center", va="bottom",
                    fontsize=7.5, color=COLORS["TP8"], fontweight="bold")
    # 500 ms SLA reference — dotted horizontal line
    ax.axhline(0.5, color="#2ca02c", ls=":", lw=2.2, zorder=4)
    ax.annotate("500 ms SLA", (len(SIZES) - 0.5, 0.5), color="#2ca02c", fontsize=10,
                fontweight="bold", va="bottom", ha="right")
    ax.set_yscale("log"); ax.set_xticks(x); ax.set_xticklabels(SIZES, fontsize=11)
    ax.set_ylabel("TTFT (s, log scale)", fontsize=11); ax.set_xlabel("input size", fontsize=11)
    ax.set_title("Gemma4-31B cold TTFT — TP scaling vs H100 (conc=1)\n"
                 "trn2.48xl LNC=2 (4 cores/chip): TP32=8 chips · TP16=4 · TP8=2 · bf16, no-APC · "
                 "Trn2 TP32 beats H100 ×2 (TP2) at every size 4k–64k",
                 fontsize=10.5, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3, zorder=0)
    ax.legend(ncol=4, fontsize=9.5, loc="upper left")
    fig.tight_layout()
    p = os.path.join(OUT, "ttft_tp_scaling_conc1_lnc2.png"); fig.savefig(p, dpi=140); print("wrote", p)


# ---- Chart 2: concurrency-sweep panels (4k/8k/16k/32k/64k) ----
def chart_sweep():
    panels = ["4k", "8k", "16k", "32k", "64k"]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle("Gemma4-31B cold TTFT vs concurrency — Trn2 TP32/TP16/TP8 vs H100 (bf16, no-APC, LNC=2)\n"
                 "TP32=8 chips · TP16=4 chips · TP8=2 chips · H100 ×2 80GB (TP2) baseline",
                 fontsize=12, fontweight="bold")
    axf = axes.flat
    for ax, s in zip(axf, panels):
        for name, d in SERIES:
            v = d.get(s)
            if not v:
                continue
            conc = CONC if len(v) == 6 else CL
            ax.plot(conc, [x * 1000 for x in v], "-o", color=COLORS[name], lw=2, ms=5,
                    label=LABELS[name], zorder=3)
        # 500 ms SLA reference — dotted horizontal line
        ax.axhline(500, color="#2ca02c", ls=":", lw=1.8, zorder=2)
        ax.annotate("500 ms SLA", (CONC[-1], 500), color="#2ca02c", fontsize=8,
                    fontweight="bold", va="bottom", ha="right")
        ax.set_xscale("log", base=2); ax.set_yscale("log")
        ax.set_xticks(CONC); ax.set_xticklabels(CONC)
        ax.set_title(f"{s} input", fontsize=12, fontweight="bold")
        ax.set_xlabel("concurrency"); ax.set_ylabel("TTFT (ms, log)")
        ax.grid(True, which="major", alpha=0.3)
        ax.legend(fontsize=8, loc="upper left")
    axf[5].axis("off")  # 6th panel unused
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    p = os.path.join(OUT, "ttft_tp_vs_h100_lnc2.png"); fig.savefig(p, dpi=140); print("wrote", p)


if __name__ == "__main__":
    chart_conc1()
    chart_sweep()

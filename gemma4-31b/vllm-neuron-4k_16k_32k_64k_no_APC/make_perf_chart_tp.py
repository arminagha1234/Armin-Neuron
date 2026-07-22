#!/usr/bin/env python3
"""TP-scaling charts for Gemma4-31B cold TTFT: TP32 vs TP16 vs TP8 vs H100.

Regenerates assets/ttft_tp_scaling_conc1.png and assets/ttft_tp_vs_h100.png.
    python3 make_perf_chart_tp.py

Trn2 = trn2.48xlarge, bf16 KV, APC OFF, unique random prompt per request (cold),
40 output tokens, public GA vLLM-Neuron 0.21 (SDK 2.31 / cc 2.26).
- <=8k: single-shot prefill (all TP degrees).
- 16k: TP32 single-shot; TP16 via segmented (single-shot 16k needs headroom).
- 32k/64k: segmented + SWA windowed-prior fix.
Topology: trn2.48xl with LNC=2 (logical-neuroncore-config=2) -> 4 logical cores/chip, 96GB/chip
= 24GB/core. TP maps 1 rank -> 1 core, so TP32=8 chips, TP16=4 chips, TP8=2 chips.
- TP8 (2 chips): 4k/8k only — the 66560-capacity (long-context) config HBM-OOMs at 8 cores
  (NCC_EOOM002: peak 27.45GB > 24GB per-core Trn2 limit). TP8 = short-ctx/density.
H100 baseline: vendor-typical vLLM serving (no 8k point measured).
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CONC = [1, 2, 4, 8, 16, 32]
CL = [1, 2, 4]
SIZES = ["4k", "8k", "16k", "32k", "64k"]

# TTFT seconds. None = not available (TP8 long-ctx OOM; no 8k H100).
TP32 = {"4k": [0.224,0.331,0.605,1.878,4.983,11.459], "8k": [0.390,0.585,0.969,2.606,6.406,14.176],
        "16k": [0.754,1.127,1.944,4.215,9.429,19.901], "32k": [2.046,3.051,7.132], "64k": [4.064,6.071,11.991]}
TP16 = {"4k": [0.307,0.449,0.744,2.36,5.852,13.323], "8k": [0.557,0.833,1.384,3.433,8.013,17.339],
        "16k": [1.515,2.271,5.89,14.1,31.065,65.229], "32k": [3.01,4.514,9.519], "64k": [6.056,9.09,17.01]}
TP8  = {"4k": [0.572,0.855,2.56,6.201,14.155,29.574], "8k": [1.126,1.684,3.853,8.704,18.612,38.636],
        "16k": None, "32k": None, "64k": None}   # long-ctx OOM
H100 = {"4k": [0.121,0.164,0.301,0.468,0.806,1.494], "8k": None,
        "16k": [0.449,0.627,1.009,1.727,3.207,6.156], "32k": [0.992,1.372,2.201,3.827,7.094,13.597],
        "64k": [2.249,3.192,5.139,9.005,16.773,32.258]}

COLORS = {"TP32": "#ff6a00", "TP16": "#d1691f", "TP8": "#8a4b1f", "H100": "#1f77b4"}
# LNC=2 on trn2.48xlarge: 4 logical cores/chip, so TP32=8 chips, TP16=4 chips, TP8=2 chips.
LABELS = {"TP32": "TP32 (8 chips)", "TP16": "TP16 (4 chips)", "TP8": "TP8 (2 chips)", "H100": "H100 (GPU)"}
SERIES = [("TP32", TP32), ("TP16", TP16), ("TP8", TP8), ("H100", H100)]
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
os.makedirs(OUT, exist_ok=True)


def c1(d, s):
    v = d.get(s)
    return v[0] if v else None


# ---- Chart 1: conc=1 TP-scaling bars (per size) ----
def chart_conc1():
    fig, ax = plt.subplots(figsize=(11, 5.6))
    x = np.arange(len(SIZES)); w = 0.2
    offs = {"TP32": -1.5, "TP16": -0.5, "TP8": 0.5, "H100": 1.5}
    for name, d in SERIES:
        vals = [c1(d, s) for s in SIZES]
        xs = [x[i] + offs[name] * w for i in range(len(SIZES)) if vals[i] is not None]
        ys = [v for v in vals if v is not None]
        bars = ax.bar(xs, ys, w, label=LABELS[name], color=COLORS[name], zorder=3)
        for b, y in zip(bars, ys):
            ax.annotate(f"{y:.2f}", (b.get_x() + b.get_width() / 2, y), textcoords="offset points",
                        xytext=(0, 2), ha="center", fontsize=7, rotation=90)
    # OOM markers for TP8 long context
    for s in ["16k", "32k", "64k"]:
        i = SIZES.index(s)
        ax.annotate("TP8\nOOM", (x[i] + 0.5 * w, 0.3), ha="center", va="bottom",
                    fontsize=6.5, color=COLORS["TP8"], fontweight="bold")
    ax.axhline(0.5, color="#2ca02c", ls="--", lw=1.2)
    ax.annotate("500 ms SLA", (len(SIZES) - 0.6, 0.5), color="#2ca02c", fontsize=8, va="bottom", ha="right")
    ax.set_yscale("log")
    ax.set_xticks(x); ax.set_xticklabels(SIZES)
    ax.set_ylabel("TTFT (s, log)"); ax.set_xlabel("input size")
    ax.set_title("Gemma4-31B cold TTFT — TP scaling vs H100 (conc=1)\n"
                 "trn2.48xl LNC=2 (4 cores/chip): TP32=8 chips · TP16=4 · TP8=2 · bf16, no-APC · TP8 long-ctx OOMs (24GB/core)",
                 fontsize=10.5, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3, zorder=0)
    ax.legend(ncol=4, fontsize=9, loc="upper left")
    fig.tight_layout()
    p = os.path.join(OUT, "ttft_tp_scaling_conc1_lnc2.png"); fig.savefig(p, dpi=140); print("wrote", p)


# ---- Chart 2: conc-sweep panels (4k/16k/32k/64k) vs H100 ----
def chart_sweep():
    panels = ["4k", "16k", "32k", "64k"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle("Gemma4-31B cold TTFT vs concurrency — Trn2 TP32/TP16/TP8 vs H100 (bf16, no-APC, LNC=2)\n"
                 "TP32=8 chips · TP16=4 chips · TP8=2 chips · TP8 omitted at 16k/32k/64k (HBM OOM, 24GB/core)",
                 fontsize=12, fontweight="bold")
    for ax, s in zip(axes.flat, panels):
        for name, d in SERIES:
            v = d.get(s)
            if not v:
                continue
            conc = CONC if len(v) == 6 else CL
            ax.plot(conc, [x * 1000 for x in v], "-o", color=COLORS[name], lw=2, ms=5, label=LABELS[name], zorder=3)
        ax.set_xscale("log", base=2); ax.set_yscale("log")
        ax.set_xticks(CONC); ax.set_xticklabels(CONC)
        ax.set_title(f"{s} input", fontsize=12, fontweight="bold")
        ax.set_xlabel("concurrency"); ax.set_ylabel("TTFT (ms, log)")
        ax.grid(True, which="major", alpha=0.3)
        ax.legend(fontsize=8.5, loc="upper left")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    p = os.path.join(OUT, "ttft_tp_vs_h100_lnc2.png"); fig.savefig(p, dpi=140); print("wrote", p)


if __name__ == "__main__":
    chart_conc1()
    chart_sweep()

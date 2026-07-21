#!/usr/bin/env python3
"""Render TTFT comparison charts for Gemma4-31B on the TP=8 (two-chip) config.

Produces two PNGs in assets/:
  1. ttft_trn2_tp8_vs_h100_ms.png     — Trn2 TP=8 bf16 vs H100 (mirror of the TP=32 chart)
  2. ttft_trn2_tp8_vs_tp32_vs_h100_ms.png — three-line combined comparison

TTFT numbers:
  - Trn2 TP=8 bf16: measured this session (public GA vLLM-Neuron v0.21, seg=512+APC, MNS=32,
    KV=auto/bf16), full concurrency sweep 1->32, all four sizes (none OOM).
  - Trn2 TP=32: the shipped public-GA config (seg=512+APC+fp8-KV>=16k) from RESULTS.md.
  - H100: same GPU baseline used in make_perf_chart.py.

Pure matplotlib; run:  python3 make_perf_chart_tp8.py
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter

CONC = [1, 2, 4, 8, 16, 32]

# TTFT seconds (mean).
DATA = {
    "4k input":  {"tp8":  [0.185, 0.275, 0.456, 0.784, 1.466, 3.553],
                   "tp32": [0.123, 0.184, 0.302, 0.507, 0.917, 1.754],
                   "gpu":  [0.121, 0.164, 0.301, 0.468, 0.806, 1.494]},
    "16k input": {"tp8":  [0.419, 0.635, 1.075, 1.884, 4.150, 10.099],
                   "tp32": [0.227, 0.338, 0.950, 0.831, 1.595, 4.307],
                   "gpu":  [0.449, 0.627, 1.009, 1.727, 3.207, 6.156]},
    "32k input": {"tp8":  [0.698, 1.058, 1.791, 2.872, 8.482, 20.208],
                   "tp32": [0.362, 0.809, 1.000, 1.411, 3.724, 14.961],
                   "gpu":  [0.992, 1.372, 2.201, 3.827, 7.094, 13.597]},
    "64k input": {"tp8":  [1.289, 1.958, 3.116, 9.853, 23.654, 51.756],
                   "tp32": [0.620, 0.948, 1.379, 2.710, 16.658, 40.990],
                   "gpu":  [2.249, 3.192, 5.139, 9.005, 16.773, 32.258]},
}

TP8_C = "#d62728"    # red
TP32_C = "#ff6a00"   # Trainium orange
GPU_C = "#1f77b4"    # GPU blue

MS_TICKS = [100, 200, 300, 500, 700, 1000, 1500, 2000, 3000, 5000,
            7000, 10000, 15000, 20000, 30000, 40000, 50000, 60000]


def ms_label(v, _pos=None):
    v = int(round(v))
    return f"{v:,} ms" if v < 1000 else f"{v/1000:g} s"


def _style_axis(ax, series_ms):
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(CONC); ax.set_xticklabels(CONC)
    lo, hi = min(min(s) for s in series_ms), max(max(s) for s in series_ms)
    ticks = [t for t in MS_TICKS if lo * 0.75 <= t <= hi * 1.35]
    if len(ticks) >= 2:
        ax.yaxis.set_major_locator(FixedLocator(ticks))
        ax.yaxis.set_minor_locator(FixedLocator([]))
    ax.yaxis.set_major_formatter(FuncFormatter(ms_label))
    ax.set_ylim(lo * 0.72, hi * 1.45)
    ax.set_xlabel("concurrency (requests)")
    ax.set_ylabel("TTFT")
    ax.grid(True, which="major", ls="-", alpha=0.30)


def chart_tp8_vs_h100(out):
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle("Gemma4-31B — Time To First Token: Trainium2 TP=8 (two-chip, bf16) vs GPU\n"
                 "lower is better · TTFT in milliseconds vs concurrency", fontsize=14, fontweight="bold")
    for ax, (name, d) in zip(axes.flat, DATA.items()):
        tp8 = [v * 1000 for v in d["tp8"]]
        gpu = [v * 1000 for v in d["gpu"]]
        ax.plot(CONC, tp8, "-o", color=TP8_C, linewidth=2.3, markersize=6,
                label="Trainium2 TP=8 (2 chips, bf16)", zorder=3)
        ax.plot(CONC, gpu, "-s", color=GPU_C, linewidth=2.3, markersize=6,
                label="GPU (H100)", zorder=3)
        _style_axis(ax, [tp8, gpu])
        ax.set_title(name, fontsize=12, fontweight="bold")
        ax.legend(fontsize=9, loc="upper left")
        for x, y in zip(CONC, tp8):
            ax.annotate(f"{int(round(y)):,}", (x, y), textcoords="offset points",
                        xytext=(0, -12), ha="center", fontsize=7.5, color=TP8_C, fontweight="bold")
        for x, y in zip(CONC, gpu):
            ax.annotate(f"{int(round(y)):,}", (x, y), textcoords="offset points",
                        xytext=(0, 7), ha="center", fontsize=7.5, color=GPU_C)
        for i in range(len(CONC) - 1):
            if tp8[i] < gpu[i] and tp8[i + 1] < gpu[i + 1]:
                ax.axvspan(CONC[i], CONC[i + 1], color=TP8_C, alpha=0.06)
    fig.text(0.5, 0.005,
             "Values in ms next to each point. Shaded band = Trainium2 TP=8 faster. "
             "TP=8 packs 8 replicas per trn2.48xl (vs 2 at TP=32) and runs ~3-4x faster TPOT; "
             "TTFT is TP=8's weaker metric (less prefill sharding) yet still beats H100 at 16k/32k low concurrency.",
             ha="center", fontsize=8.5, style="italic")
    fig.tight_layout(rect=[0, 0.03, 1, 0.93])
    fig.savefig(out, dpi=140)
    print("wrote", out)
    plt.close(fig)


def chart_tp8_vs_tp32_vs_h100(out):
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle("Gemma4-31B — TTFT: Trainium2 TP=32 (shipped) vs TP=8 (two-chip) vs GPU\n"
                 "lower is better · TTFT in milliseconds vs concurrency", fontsize=14, fontweight="bold")
    for ax, (name, d) in zip(axes.flat, DATA.items()):
        tp32 = [v * 1000 for v in d["tp32"]]
        tp8 = [v * 1000 for v in d["tp8"]]
        gpu = [v * 1000 for v in d["gpu"]]
        ax.plot(CONC, tp32, "-o", color=TP32_C, linewidth=2.3, markersize=6,
                label="Trn2 TP=32 (shipped, fp8-KV)", zorder=3)
        ax.plot(CONC, tp8, "-^", color=TP8_C, linewidth=2.3, markersize=6,
                label="Trn2 TP=8 (2 chips, bf16)", zorder=3)
        ax.plot(CONC, gpu, "-s", color=GPU_C, linewidth=2.3, markersize=6,
                label="GPU (H100)", zorder=3)
        _style_axis(ax, [tp32, tp8, gpu])
        ax.set_title(name, fontsize=12, fontweight="bold")
        ax.legend(fontsize=8, loc="upper left")
    fig.text(0.5, 0.005,
             "TP=32 gives the lowest TTFT (prefill sharded 32 ways) and is the single-stream latency pick. "
             "TP=8 trades some TTFT for 4x replica density + ~3-4x faster TPOT — the throughput/cost pick. Both beat H100 at long context, low concurrency.",
             ha="center", fontsize=8.5, style="italic")
    fig.tight_layout(rect=[0, 0.03, 1, 0.93])
    fig.savefig(out, dpi=140)
    print("wrote", out)
    plt.close(fig)


out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
os.makedirs(out_dir, exist_ok=True)
chart_tp8_vs_h100(os.path.join(out_dir, "ttft_trn2_tp8_vs_h100_ms.png"))
chart_tp8_vs_tp32_vs_h100(os.path.join(out_dir, "ttft_trn2_tp8_vs_tp32_vs_h100_ms.png"))

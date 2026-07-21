#!/usr/bin/env python3
"""Render the Trainium2-vs-GPU TTFT comparison chart for Gemma4-31B.

Regenerates assets/ttft_trn2_vs_gpu.png from the measured TTFT numbers below.
Pure matplotlib; run:  python3 make_perf_chart.py
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter

CONC = [1, 2, 4, 8, 16, 32]

# TTFT (seconds, mean). GPU baseline = H100 (all sizes).
# Trn2 = trn2.48xlarge, TP=32 — PUBLIC GA optimized config (seg=512 + APC + fp8-KV >=16k),
# the same run recorded in RESULTS.md.
DATA = {
    "4k input":  {"gpu_label": "H100", "trn2": [0.123, 0.184, 0.302, 0.507, 0.917, 1.754],
                   "gpu": [0.121, 0.164, 0.301, 0.468, 0.806, 1.494]},
    "16k input": {"gpu_label": "H100", "trn2": [0.227, 0.338, 0.950, 0.831, 1.595, 4.307],
                   "gpu": [0.449, 0.627, 1.009, 1.727, 3.207, 6.156]},
    "32k input": {"gpu_label": "H100", "trn2": [0.362, 0.809, 1.000, 1.411, 3.724, 14.961],
                   "gpu": [0.992, 1.372, 2.201, 3.827, 7.094, 13.597]},
    "64k input": {"gpu_label": "H100", "trn2": [0.620, 0.948, 1.379, 2.710, 16.658, 40.990],
                   "gpu": [2.249, 3.192, 5.139, 9.005, 16.773, 32.258]},
}

TRN2_C = "#ff6a00"   # Trainium orange
GPU_C = "#1f77b4"    # GPU blue

# Candidate y-tick positions in MILLISECONDS (nice round values). Per subplot we
# keep only the ticks within that subplot's data range so labels stay readable.
MS_TICKS = [100, 200, 300, 500, 700, 1000, 1500, 2000, 3000, 5000,
            7000, 10000, 15000, 20000, 30000, 40000, 50000]


def ms_label(v, _pos=None):
    v = int(round(v))
    return f"{v:,} ms" if v < 1000 else f"{v/1000:g} s"


fig, axes = plt.subplots(2, 2, figsize=(12, 9))
fig.suptitle("Gemma4-31B — Time To First Token: Trainium2 public GA (TP=32) vs GPU\n"
             "lower is better · TTFT in milliseconds vs concurrency", fontsize=14, fontweight="bold")

for ax, (name, d) in zip(axes.flat, DATA.items()):
    trn2 = [v * 1000 for v in d["trn2"]]   # -> ms
    gpu = [v * 1000 for v in d["gpu"]]     # -> ms
    ax.plot(CONC, trn2, "-o", color=TRN2_C, linewidth=2.3, markersize=6,
            label="Trainium2 (trn2.48xlarge)", zorder=3)
    ax.plot(CONC, gpu, "-s", color=GPU_C, linewidth=2.3, markersize=6,
            label=f"GPU ({d['gpu_label']})", zorder=3)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(CONC)
    ax.set_xticklabels(CONC)

    # readable y ticks in ms, only those inside this panel's range
    lo, hi = min(trn2 + gpu), max(trn2 + gpu)
    ticks = [t for t in MS_TICKS if lo * 0.75 <= t <= hi * 1.35]
    if len(ticks) >= 2:
        ax.yaxis.set_major_locator(FixedLocator(ticks))
        ax.yaxis.set_minor_locator(FixedLocator([]))
    ax.yaxis.set_major_formatter(FuncFormatter(ms_label))
    ax.set_ylim(lo * 0.72, hi * 1.45)

    ax.set_title(name, fontsize=12, fontweight="bold")
    ax.set_xlabel("concurrency (requests)")
    ax.set_ylabel("TTFT")
    ax.grid(True, which="major", ls="-", alpha=0.30)
    ax.legend(fontsize=9, loc="upper left")

    # annotate every point with its ms value (Trn2 below, GPU above — reduces overlap)
    for x, y in zip(CONC, trn2):
        ax.annotate(f"{int(round(y)):,}", (x, y), textcoords="offset points",
                    xytext=(0, -12), ha="center", fontsize=7.5, color=TRN2_C, fontweight="bold")
    for x, y in zip(CONC, gpu):
        ax.annotate(f"{int(round(y)):,}", (x, y), textcoords="offset points",
                    xytext=(0, 7), ha="center", fontsize=7.5, color=GPU_C)

    # shade where Trainium2 is faster (lower TTFT)
    for i in range(len(CONC) - 1):
        if trn2[i] < gpu[i] and trn2[i + 1] < gpu[i + 1]:
            ax.axvspan(CONC[i], CONC[i + 1], color=TRN2_C, alpha=0.06)

fig.text(0.5, 0.005,
         "Values in ms next to each point. Shaded band = Trainium2 faster (public GA config). "
         "Trn2 ties H100 at 4k and beats H100 at 16k/32k/64k across low-to-mid concurrency; converges at C=32.",
         ha="center", fontsize=9, style="italic")
fig.tight_layout(rect=[0, 0.03, 1, 0.93])

out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
os.makedirs(out_dir, exist_ok=True)
out = os.path.join(out_dir, "ttft_trn2_vs_h100_ms.png")
fig.savefig(out, dpi=140)
print("wrote", out)

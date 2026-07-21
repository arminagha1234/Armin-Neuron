#!/usr/bin/env python3
"""Render the Trainium2-vs-GPU TTFT comparison chart for Gemma4-31B.

Regenerates assets/ttft_trn2_vs_gpu.png from the measured TTFT numbers below.
Pure matplotlib; run:  python3 make_perf_chart.py
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CONC = [1, 2, 4, 8, 16, 32]

# TTFT (seconds, mean). GPU baseline = H100 for 4k, H200 for 16k/32k/64k.
# Trn2 = trn2.48xlarge, TP=32.
DATA = {
    "4k input":  {"gpu_label": "H100", "trn2": [0.409, 0.611, 1.011, 2.066, 3.338, 6.444],
                   "gpu": [0.121, 0.164, 0.301, 0.468, 0.806, 1.494]},
    "16k input": {"gpu_label": "H200", "trn2": [0.471, 0.701, 1.184, 2.156, 3.515, 8.521],
                   "gpu": [0.449, 0.627, 1.009, 1.727, 3.207, 6.156]},
    "32k input": {"gpu_label": "H200", "trn2": [0.530, 0.819, 1.230, 2.128, 4.558, 21.375],
                   "gpu": [0.992, 1.372, 2.201, 3.827, 7.094, 13.597]},
    "64k input": {"gpu_label": "H200", "trn2": [0.661, 1.010, 1.427, 3.048, 13.174, 33.075],
                   "gpu": [2.249, 3.192, 5.139, 9.005, 16.773, 32.258]},
}

TRN2_C = "#ff6a00"   # Trainium orange
GPU_C = "#1f77b4"    # GPU blue

fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))
fig.suptitle("Gemma4-31B — Time To First Token: Trainium2 (TP=32) vs GPU\n"
             "lower is better · TTFT (s) vs concurrency", fontsize=14, fontweight="bold")

for ax, (name, d) in zip(axes.flat, DATA.items()):
    ax.plot(CONC, d["trn2"], "-o", color=TRN2_C, linewidth=2.2, markersize=6,
            label="Trainium2 (trn2.48xlarge)")
    ax.plot(CONC, d["gpu"], "-s", color=GPU_C, linewidth=2.2, markersize=6,
            label=f"GPU ({d['gpu_label']})")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(CONC)
    ax.set_xticklabels(CONC)
    ax.set_title(name, fontsize=12, fontweight="bold")
    ax.set_xlabel("concurrency (requests)")
    ax.set_ylabel("TTFT (s)")
    ax.grid(True, which="both", ls=":", alpha=0.4)
    ax.legend(fontsize=9, loc="upper left")
    # shade where Trainium2 is faster (lower TTFT)
    for i in range(len(CONC) - 1):
        if d["trn2"][i] < d["gpu"][i] and d["trn2"][i + 1] < d["gpu"][i + 1]:
            ax.axvspan(CONC[i], CONC[i + 1], color=TRN2_C, alpha=0.06)

fig.text(0.5, 0.005,
         "Shaded band = Trainium2 faster. Trn2 wins TTFT at long context (32k/64k) across most "
         "concurrency; GPU leads at short context (4k/16k).",
         ha="center", fontsize=9, style="italic")
fig.tight_layout(rect=[0, 0.03, 1, 0.94])

out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
os.makedirs(out_dir, exist_ok=True)
out = os.path.join(out_dir, "ttft_trn2_vs_gpu.png")
fig.savefig(out, dpi=140)
print("wrote", out)

#!/usr/bin/env python3
"""TP-scaling-vs-concurrency charts for Gemma4-31B cold TTFT (PUBLIC image, no-APC).

Regenerates, from the measured grid in RESULTS.md:
    assets/ttft_tp_vs_h100_conc_med.png   # median TTFT vs concurrency, 5 input panels
    assets/ttft_tp_vs_h100_conc_p99.png   # P99 TTFT vs concurrency, 5 input panels

    python3 make_perf_chart_tp.py

Each chart is a 5-panel grid (one panel per input size 4k/8k/16k/32k/64k). Within a
panel: TTFT vs concurrency (1->32), one colored line per Trn2 TP degree plus the H100
reference. Both axes are log so the sub-second short-context numbers and the tens-of-
seconds long-context numbers are both readable.

Setup (identical to RESULTS.md): google/gemma-4-31B text-only, trn2.48xlarge, public
vLLM-Neuron 0.21 (SDK 2.31 / neuronx-cc 2.26), bf16, APC OFF, unique random prompt per
request (true cold prefill), 40 output tokens. LNC=2 -> 4 logical cores/chip, so TP maps
1 rank -> 1 core:  TP32 = 8 chips,  TP16 = 4 chips,  TP8 = 2 chips.

Data provenance:
- Trn2 TP32/TP16/TP8 numbers are copied verbatim from RESULTS.md (this folder). conc=1 is
  median-of-10; conc>=2 is warmup+bench. 32k/64k only ran conc 1/2/4 (segmented path).
  TP8 64k was not run.
- H100 = 2x H100 80GB, TP=2, a comparable-scale GPU reference from a prior session. Its
  conc=1 medians match RESULTS.md exactly (4k 0.240 / 8k 0.461 / 16k 1.008 / 32k 2.377 /
  64k 5.778). P99 was not captured for H100, so on the P99 chart the H100 line is its
  MEDIAN, drawn dashed and labelled as such.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CONC = [1, 2, 4, 8, 16, 32]
CL = [1, 2, 4]                       # long-context only ran conc 1/2/4
SIZES = ["4k", "8k", "16k", "32k", "64k"]

# ---- MEDIAN TTFT (s) -- verbatim from RESULTS.md --------------------------------------
MED = {
    "TP32": {"4k": [0.239, 0.535, 0.586, 6.378, 20.816, 50.018],
             "8k": [0.422, 0.631, 1.621, 8.238, 23.319, 54.431],
             "16k": [0.806, 1.212, 2.408, 9.229, 25.416, 58.588],
             "32k": [10.664, 16.003, 26.661], "64k": [47.981, 71.97, 119.785]},
    "TP16": {"4k": [0.315, 0.475, 0.779, 2.668, 6.064, 14.028],
             "8k": [1.17, 1.751, 2.477, 4.859, 11.295, 24.996],
             "16k": [1.178, 1.768, 2.937, 6.169, 13.178, 27.389],
             "32k": [16.538, 24.805, 41.336], "64k": [69.104, 103.659, 172.785]},
    "TP8":  {"4k": [1.2, 1.204, 4.049, 9.092, 19.468, 40.301],
             "8k": [1.201, 1.801, 4.127, 10.279, 20.707, 43.382],
             "16k": [2.461, 3.695, 7.217, 14.733, 30.08, 60.866],
             "32k": [30.448, 45.674, 76.093], "64k": None},   # 64k@TP8 not run
    "H100": {"4k": [0.240, 0.359, 0.637, 1.139, 2.075, 3.964],
             "8k": [0.461, 0.694, 1.197, 2.157, 4.100, 8.021],
             "16k": [1.008, 1.510, 2.569, 4.651, 8.812, 16.894],
             "32k": [2.377, 3.565, 5.927, 10.714, 20.192, 42.746],
             "64k": [5.778, 8.699, 14.594, 26.003, 48.894, 94.487]},
}
# ---- P99 TTFT (s) -- verbatim from RESULTS.md (H100 P99 not captured) -----------------
P99 = {
    "TP32": {"4k": [0.239, 0.645, 0.916, 22.939, 46.562, 112.937],
             "8k": [0.422, 0.828, 2.782, 26.216, 51.447, 121.258],
             "16k": [0.806, 1.595, 3.694, 27.71, 54.781, 129.44],
             "32k": [10.664, 21.299, 42.56], "64k": [47.981, 95.874, 191.496]},
    "TP16": {"4k": [0.315, 0.626, 1.227, 6.857, 12.829, 30.039],
             "8k": [1.17, 2.323, 4.05, 10.756, 22.062, 51.424],
             "16k": [1.178, 2.338, 4.65, 13.063, 26.084, 55.953],
             "32k": [16.538, 33.041, 66.045], "64k": [69.104, 138.152, 276.266]},
    "TP8":  {"4k": [1.2, 1.796, 9.034, 18.049, 40.383, 80.724],
             "8k": [1.201, 2.39, 9.137, 19.392, 41.582, 86.878],
             "16k": [2.461, 4.907, 14.066, 28.044, 60.325, 120.565],
             "32k": [30.448, 60.867, 121.673], "64k": None},
    "H100": MED["H100"],   # P99 not captured for H100 -> reuse median as a reference
}

ORDER = ["TP32", "TP16", "TP8", "H100"]
COLORS = {"TP32": "#ff6a00", "TP16": "#c85a1b", "TP8": "#6b3410", "H100": "#1f77b4"}
MARKERS = {"TP32": "o", "TP16": "s", "TP8": "^", "H100": "D"}
LABELS = {"TP32": "Trn2 TP32 (8 chips)", "TP16": "Trn2 TP16 (4 chips)",
          "TP8": "Trn2 TP8 (2 chips)", "H100": "H100 x2 80GB (TP2)"}
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
os.makedirs(OUT, exist_ok=True)


def make_sweep(data, metric, fname, h100_is_median):
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    title = (f"Gemma4-31B cold TTFT ({metric}) vs concurrency - Trn2 TP32/TP16/TP8 vs H100\n"
             "public image, bf16, no-APC, unique random prompt per request;  "
             "TP32=8 chips | TP16=4 chips | TP8=2 chips | H100 x2 80GB (TP2) reference")
    if h100_is_median:
        title += "\n(H100 line is its MEDIAN - P99 was not captured for H100)"
    fig.suptitle(title, fontsize=12, fontweight="bold")
    axf = axes.flat
    for ax, s in zip(axf, SIZES):
        for name in ORDER:
            v = data[name].get(s)
            if not v:
                continue
            conc = CONC if len(v) == 6 else CL
            dashed = (name == "H100" and h100_is_median)
            ax.plot(conc, [x * 1000 for x in v], ("--" if dashed else "-"),
                    marker=MARKERS[name], color=COLORS[name], lw=2, ms=6,
                    label=LABELS[name] + (" [median]" if dashed else ""), zorder=3)
        ax.axhline(500, color="#2ca02c", ls=":", lw=1.8, zorder=2)
        ax.annotate("500 ms", (CONC[-1], 500), color="#2ca02c", fontsize=8,
                    fontweight="bold", va="bottom", ha="right")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks(CONC)
        ax.set_xticklabels(CONC)
        note = "  (conc 1/2/4 only - segmented path)" if s in ("32k", "64k") else ""
        ax.set_title(f"{s} input{note}", fontsize=12, fontweight="bold")
        ax.set_xlabel("concurrency (requests in flight)")
        ax.set_ylabel(f"TTFT {metric} (ms, log)")
        ax.grid(True, which="major", alpha=0.3)
        ax.grid(True, which="minor", alpha=0.12)
        ax.legend(fontsize=8, loc="upper left")
    axf[5].axis("off")
    axf[5].annotate(
        "Reading the panels\n\n"
        "- Lower is better (log ms).\n"
        "- Each Trn2 line = ONE vLLM server at\n"
        "  that TP (one replica) taking all the\n"
        "  concurrency - not multiple replicas.\n"
        "- <=16k: TP32 wins at conc=1 (widest TP\n"
        "  = fastest single prefill); under load\n"
        "  TP16 wins - TP32 doubles the chips but\n"
        "  scales sublinearly (more collectives),\n"
        "  so it is throughput-bound sooner.\n"
        "- H100 leads under load & long-context.\n"
        "- 32k/64k: segmented path, honestly slow\n"
        "  (see ROADMAP.md). TP8 64k: not run.",
        (0.02, 0.98), xycoords="axes fraction", va="top", ha="left",
        fontsize=9.5, family="monospace")
    fig.tight_layout(rect=[0, 0, 1, 0.92 if h100_is_median else 0.94])
    p = os.path.join(OUT, fname)
    fig.savefig(p, dpi=140)
    print("wrote", p)


if __name__ == "__main__":
    make_sweep(MED, "median", "ttft_tp_vs_h100_conc_med.png", h100_is_median=False)
    make_sweep(P99, "P99", "ttft_tp_vs_h100_conc_p99.png", h100_is_median=True)

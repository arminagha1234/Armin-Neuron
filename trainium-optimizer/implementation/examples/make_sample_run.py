"""
Generate a sample run + trajectory chart for documentation purposes.

Numbers mirror `internal-prior-optimization-run`'s Round 2
(845 -> 4,269 tok/s on Tongyi-30B-A3B) including its failure pattern, so the
chart demonstrates a realistic shape rather than a tidy monotonic curve.

    python examples/make_sample_run.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ledger import Layer, Ledger, Origin, Row, Stage, Status  # noqa: E402
from trajectory_chart import build_chart  # noqa: E402


def row(metric, stage, origin=Origin.NONE, layer=Layer.CONFIG,
        status=Status.KEEP, desc="", mfu=-1.0, compile_s=0.0, source=""):
    return Row(
        commit=f"c{int(metric)}", stage=stage, origin=origin, layer=layer,
        source=source, metric=metric, mfu=mfu, correctness=100.0,
        compile_s=compile_s, status=status, description=desc,
    )


TRAJECTORY = [
    row(845.0, Stage.BASELINE, layer=Layer.NONE,
        desc="baseline BF16 experts FP8 KV", mfu=0.33, compile_s=391),

    row(899.5, Stage.CONFIG, desc="batch1024 block128", mfu=0.35, compile_s=331),
    row(860.0, Stage.CONFIG, status=Status.DISCARD,
        desc="1024 segment slower", compile_s=290),
    row(858.0, Stage.CONFIG, status=Status.DISCARD,
        desc="16-tok KV blocks slower", compile_s=310),
    row(1031.5, Stage.CONFIG, desc="GQA broadcast", mfu=0.41, compile_s=305),

    row(1124.5, Stage.KNOWN_KERNEL, Origin.HARVESTED, Layer.KERNEL,
        desc="nkilib rmsnorm_quant", mfu=0.44, compile_s=402,
        source="nki-library@7f3a1b2"),
    row(1100.0, Stage.KNOWN_KERNEL, Origin.HARVESTED, Layer.KERNEL,
        status=Status.DISCARD, desc="nkilib attention_tkg wrong regime",
        compile_s=380, source="nki-library@7f3a1b2"),

    row(1555.8, Stage.BORROW, Origin.BORROWED, Layer.KERNEL,
        desc="BF16 QK PV softmax", mfu=0.62, compile_s=455, source="vllm@a1b2c3d"),
    row(4071.1, Stage.BORROW, Origin.BORROWED, Layer.KERNEL,
        desc="NKI flash attention tiling online softmax", mfu=1.62,
        compile_s=612, source="flashattn@bsd3"),

    row(4102.0, Stage.INVENT, Origin.INVENTED, Layer.KERNEL,
        status=Status.DISCARD, desc="6-way SBUF split under margin", compile_s=740),
    row(4090.0, Stage.INVENT, Origin.INVENTED, Layer.KERNEL,
        status=Status.DISCARD, desc="DMA prefetch restructure", compile_s=700),
    row(4055.0, Stage.INVENT, Origin.INVENTED, Layer.KERNEL,
        status=Status.DISCARD, desc="alt tiling order", compile_s=690),

    row(4269.0, Stage.GRAPH_REWRITE, layer=Layer.GRAPH,
        desc="ModelLen O3", mfu=1.68, compile_s=520),
]


def main() -> None:
    out_dir = Path(__file__).parent / "sample-run"
    led = Ledger(out_dir)
    if led.path.exists():
        led.path.unlink()
    led.init()
    for r in TRAJECTORY:
        led.append(r)

    chart = build_chart(
        run_dir=out_dir,
        out_path=out_dir / "optimization_timeline.png",
        model="Tongyi-30B-A3B (illustrative)",
        hardware="trn2.48xlarge", tp=4, shape="chat 1k/512", sdk="2.28.0",
        roofline=48000.0,
    )

    print(f"ledger  : {led.path}")
    print(f"chart   : {chart}  ({chart.stat().st_size:,} bytes)")
    print(f"speedup : {led.speedup():.2f}x")
    print(f"compile : {led.compile_time_total_s():,.0f}s total")
    print(f"origin  : {led.provenance_counts()}")
    inv = led.invention_stats()
    print(f"invent  : {inv['invention_promoted']}/{inv['invention_attempts']} "
          f"promoted (win rate {inv['invention_win_rate']:.0%})")
    print()
    print(f"{'STAGE':<15}{'TRIED':>6}{'KEPT':>6}{'GAIN':>9}{'COMPILE':>10}")
    print("-" * 46)
    for name, s in led.stage_summary().items():
        print(f"{name:<15}{s['attempts']:>6.0f}{s['promoted']:>6.0f}"
              f"{s['gain_pct']:>8.1f}%{s['compile_s']:>9.0f}s")


if __name__ == "__main__":
    main()

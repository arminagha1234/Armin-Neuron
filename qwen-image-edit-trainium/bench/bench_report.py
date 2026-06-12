"""Render bench_full.py JSON results into a customer-grade markdown report.

Usage:
    python serve/bench_report.py results/bench/<ts>/results.json \\
        > results/bench/<ts>/REPORT.md
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def fmt_ms(ms):
    if ms is None or ms == 0:
        return "—"
    if ms >= 1000:
        return f"{ms/1000:.2f}s"
    return f"{int(ms)}ms"


def fmt_s(ms):
    if ms is None:
        return "—"
    return f"{ms/1000:.2f}s"


def cost_per_image(latency_ms, price_usd_hr, workers):
    """Cost = (latency_ms / 3_600_000) * price_usd_hr / workers.
    Per-image cost — if the box runs `workers` parallel pipelines
    sharing the box price, each image costs price/workers/(images_per_hr)."""
    if not latency_ms:
        return None
    images_per_hr = (3_600_000 / latency_ms) * workers
    return price_usd_hr / images_per_hr


def section_canonical(d, price_usd_hr, workers):
    out = ["## Canonical workload (512×512, 28 steps, 1 image)\n"]
    cold = d.get("cold_ms")
    warm = d.get("warm", {})
    stages = d.get("warm_stages_ms", {})
    out.append(f"- Cold (1st request): {fmt_ms(cold)}")
    out.append(f"- Warm n: {warm.get('n', 0)}")
    out.append(f"- Warm mean: {fmt_ms(warm.get('mean'))}")
    out.append(f"- Warm p50: {fmt_ms(warm.get('p50'))}")
    out.append(f"- Warm p95: {fmt_ms(warm.get('p95'))}")
    out.append(f"- Warm p99: {fmt_ms(warm.get('p99'))}")
    out.append(f"- Warm stdev: {fmt_ms(warm.get('stdev'))}")
    if stages:
        total = stages.get("total_ms", 0)
        out.append("\n### Per-stage breakdown (warm mean)\n")
        out.append("| Stage | Time | % of total |")
        out.append("|---|---:|---:|")
        for k in ["encoder_ms", "vae_encode_ms", "denoise_ms",
                  "vae_decode_ms", "postprocess_ms"]:
            v = stages.get(k, 0)
            pct = (v / total * 100) if total else 0
            out.append(f"| {k.replace('_ms', '').replace('_', ' ')} | "
                       f"{fmt_ms(v)} | {pct:.1f}% |")
        out.append(f"| **total** | **{fmt_ms(total)}** | 100.0% |")

    if warm.get("p99"):
        cpi_1 = cost_per_image(warm["p99"], price_usd_hr, 1)
        cpi_n = cost_per_image(warm["p99"], price_usd_hr, workers)
        out.append("\n### Throughput & cost (p99 latency)\n")
        out.append(f"- Single worker: {3_600_000 / warm['p99']:.1f} img/hr → "
                   f"**${cpi_1:.4f} / image**")
        out.append(f"- Box ({workers} workers data-parallel): "
                   f"{3_600_000 / warm['p99'] * workers:.1f} img/hr → "
                   f"**${cpi_n:.4f} / image** (extrapolated)")
    return "\n".join(out) + "\n"


def section_ttfi(d):
    if not d:
        return ""
    if d.get("error"):
        return f"## TTFI\n\n_error: {d['error']}_\n"
    out = ["## TTFI (time-to-first-image)\n"]
    out.append(f"- Spawn → /health 200: {d.get('spawn_to_health_s')}s")
    out.append(f"- Spawn → first image returned: "
               f"**{d.get('spawn_to_first_image_s')}s**")
    out.append(f"- (Of which, denoising: "
               f"{fmt_ms(d.get('first_image_worker_ms'))})")
    out.append(f"- Cold compile penalty: ~"
               f"{(d.get('spawn_to_first_image_s', 0) - (d.get('first_image_worker_ms', 0) / 1000)):.1f}s")
    return "\n".join(out) + "\n"


def section_sweep(d, axis_key, axis_label, price_usd_hr, workers):
    if not d:
        return ""
    out = [f"## {axis_label} sweep\n"]
    out.append(f"| {axis_label} | Cold | Warm mean | Warm p99 | $/image (1 worker) | $/image ({workers}× DP) |")
    out.append("|---|---:|---:|---:|---:|---:|")
    for row in d:
        cold = row.get("cold_ms")
        warm = row.get("warm", {})
        mean = warm.get("mean")
        p99 = warm.get("p99") or mean or cold
        cpi_1 = cost_per_image(p99, price_usd_hr, 1) if p99 else None
        cpi_n = cost_per_image(p99, price_usd_hr, workers) if p99 else None
        v = row.get(axis_key)
        if axis_key == "height":
            v = f"{row.get('height')}×{row.get('width')}"
        out.append(f"| {v} | {fmt_ms(cold)} | {fmt_ms(mean)} | {fmt_ms(p99)} | "
                   f"{('$' + format(cpi_1, '.4f')) if cpi_1 else '—'} | "
                   f"{('$' + format(cpi_n, '.4f')) if cpi_n else '—'} |")
    return "\n".join(out) + "\n"


def section_memory(d, label):
    if not d:
        return f"## Per-core memory ({label})\n\n_unavailable_\n"
    if isinstance(d, dict):
        if d.get("error"):
            return f"## Per-core memory ({label})\n\n_unavailable: {d['error']}_\n"
        if d.get("raw_text"):
            return (f"## Per-core memory ({label})\n\n"
                    f"```\n{d['raw_text'][:2500]}\n```\n"
                    f"_note: {d.get('note', '')}_\n")
    out = [f"## Per-core memory ({label})\n"]
    out.append("| Device | Core | Used (MB) | Total (MB) | % |")
    out.append("|---|---:|---:|---:|---:|")
    for c in d:
        used = c.get("mem_used_mb") or 0
        total = c.get("mem_total_mb") or 0
        pct = (used / total * 100) if total else 0
        out.append(f"| {c.get('device')} | {c.get('core')} | {used} | {total} | {pct:.1f}% |")
    return "\n".join(out) + "\n"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("path", type=Path)
    args = p.parse_args()

    data = json.loads(args.path.read_text())
    meta = data.get("meta", {})
    workers = meta.get("workers_per_box", 4)
    price = meta.get("price_usd_hr", 21.50)

    lines = []
    lines.append(f"# fal Qwen-Image-Edit — Trainium2 benchmark report\n")
    lines.append(f"_Generated {meta.get('ended', meta.get('started', '?'))}_\n")
    lines.append(f"- Machine: **{meta.get('machine', 'trn2.48xlarge')}**")
    lines.append(f"- On-demand price: **${price}/hr**")
    lines.append(f"- Workers per box (data-parallel ceiling): **{workers}**")
    lines.append(f"- Tool: `{meta.get('tool', 'bench_full.py')}`")
    lines.append("")

    lines.append(section_ttfi(data.get("ttfi")))
    lines.append(section_canonical(data.get("canonical") or {}, price, workers))
    lines.append(section_sweep(data.get("resolution_sweep"), "height",
                                "Resolution", price, workers))
    lines.append(section_sweep(data.get("step_sweep"), "num_steps",
                                "Step count", price, workers))
    lines.append(section_sweep(data.get("multi_image_sweep"), "num_images",
                                "Input images", price, workers))
    lines.append(section_memory(data.get("memory_pre_run"), "before bench"))
    lines.append(section_memory(data.get("memory_post_run"), "after bench"))

    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Summarize benchmark results into one clean table + a CSV.
Reads <results>/{4k,16k,32k,64k}.json (produced by bench.py) and prints TTFT / TPOT / E2E /
throughput per input size and concurrency. Also writes <results>/summary.csv.
"""
import argparse
import csv
import json
import os

CONFIGS = ["4k", "16k", "32k", "64k"]


def fmt(x, nd=3):
    return "-" if x is None else f"{x:.{nd}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, help="results directory containing <size>.json files")
    args = ap.parse_args()

    csv_rows = [["input", "concurrency", "ttft_mean_s", "ttft_p99_s",
                 "tpot_mean_ms", "e2e_mean_s", "agg_tok_s", "tok_s_per_req"]]

    for cfg in CONFIGS:
        path = os.path.join(args.results, f"{cfg}.json")
        print(f"\n### {cfg} input, 40 output tokens")
        if not os.path.exists(path):
            print("   (not run)")
            continue
        rows = json.load(open(path))
        hdr = (f"  {'conc':>4} {'TTFT_s':>8} {'TTFT_p99':>9} {'TPOT_ms':>9} "
               f"{'E2E_s':>8} {'tok/s':>8} {'tok/s/req':>10}")
        print(hdr)
        print("  " + "-" * (len(hdr) - 2))
        for r in rows:
            c = r.get("concurrency")
            print(f"  {c:>4} {fmt(r.get('ttft_mean_s')):>8} {fmt(r.get('ttft_p99_s')):>9} "
                  f"{fmt(r.get('tpot_mean_ms'), 2):>9} {fmt(r.get('e2e_mean_s')):>8} "
                  f"{fmt(r.get('agg_output_tok_s'), 1):>8} {fmt(r.get('output_tok_s_per_req'), 2):>10}")
            csv_rows.append([cfg, c, r.get("ttft_mean_s"), r.get("ttft_p99_s"),
                             r.get("tpot_mean_ms"), r.get("e2e_mean_s"),
                             r.get("agg_output_tok_s"), r.get("output_tok_s_per_req")])

    out_csv = os.path.join(args.results, "summary.csv")
    with open(out_csv, "w", newline="") as f:
        csv.writer(f).writerows(csv_rows)
    print(f"\nCSV -> {out_csv}")


if __name__ == "__main__":
    main()

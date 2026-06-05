#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Aggregate results/*.json into a comparison table.

Emits:
  - a markdown table on stdout (paste straight into README / blog)
  - results/summary.csv

Run after run_matrix.sh:
  python3 analysis/analyze.py
"""

import csv
import glob
import json
import os

ORDER = ["baseline", "cpu_partial", "cpu_full", "mem_bandwidth", "cache_thrash",
         "io_stress", "irq_load", "combined", "thermal"]


def _key(label: str):
    base = label.replace("_tuned", "")
    idx = ORDER.index(base) if base in ORDER else len(ORDER)
    return (idx, label)


def load():
    rows = []
    for path in glob.glob("results/*.json"):
        with open(path) as f:
            s = json.load(f)
        r, c, j = s["response_us"], s["compute_us"], s["release_jitter_us"]
        rows.append({
            "profile": s["meta"]["label"],
            "resp_p50": r["p50"],
            "resp_p99": r["p99"],
            "resp_p99.99": r["p99.99"],
            "resp_max": r["max"],
            "compute_p99.99": c["p99.99"],
            "jitter_p99.99": j["p99.99"],
            "misses": s["deadline_miss_count"],
            "miss_rate_%": s["deadline_miss_rate"] * 100,
        })
    rows.sort(key=lambda x: _key(x["profile"]))
    return rows


def main():
    rows = load()
    if not rows:
        print("no results/*.json found -- run experiments/run_matrix.sh first")
        return

    cols = ["profile", "resp_p50", "resp_p99", "resp_p99.99", "resp_max",
            "compute_p99.99", "jitter_p99.99", "misses", "miss_rate_%"]
    hdr = ["profile", "resp p50", "resp p99", "resp p99.99", "resp max",
           "compute p99.99", "jitter p99.99", "misses", "miss %"]

    print("| " + " | ".join(hdr) + " |")
    print("|" + "|".join(["---"] * len(hdr)) + "|")
    for r in rows:
        cells = [r["profile"]] + [
            f"{r[c]:.3f}" if isinstance(r[c], float) else str(r[c])
            for c in cols[1:]
        ]
        print("| " + " | ".join(cells) + " |")
    print("\n(all latencies in microseconds)")

    os.makedirs("results", exist_ok=True)
    with open("results/summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print("wrote results/summary.csv")


if __name__ == "__main__":
    main()

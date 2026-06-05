#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Plots for the writeup. Reads results/*.json (+ *.csv for CDFs).

  results/tail_by_profile.png   p99.99 response across profiles
  results/jitter_vs_compute.png what the tail is made of (the key finding)
  results/cdf_tail.png          baseline vs worst-profile response CDF

  python3 analysis/plot.py
"""

import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402

ORDER = ["baseline", "cpu_partial", "cpu_full", "mem_bandwidth", "cache_thrash",
         "io_stress", "irq_load", "combined", "thermal"]


def load_summaries():
    out = {}
    for path in glob.glob("results/*.json"):
        if os.path.basename(path) == "summary.json":
            continue
        with open(path) as f:
            data = json.load(f)
        out[data["meta"]["label"]] = data
    labels = [p for p in ORDER if p in out] + \
             [l for l in out if l not in ORDER]
    return out, labels


def tail_by_profile(s, labels):
    vals = [s[l]["response_us"]["p99.99"] for l in labels]
    deadline = s[labels[0]]["meta"]["deadline_us"]
    plt.figure(figsize=(10, 4.5))
    bars = plt.bar(labels, vals)
    plt.axhline(deadline, ls="--", color="crimson",
                label=f"deadline {deadline:.0f} us")
    for b, v in zip(bars, vals):
        plt.text(b.get_x() + b.get_width() / 2, v, f"{v:.0f}",
                 ha="center", va="bottom", fontsize=8)
    plt.ylabel("response p99.99 (us)")
    plt.title("Tail latency under stress")
    plt.xticks(rotation=30, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/tail_by_profile.png", dpi=150)
    plt.close()


def jitter_vs_compute(s, labels):
    jit = [s[l]["release_jitter_us"]["p99.99"] for l in labels]
    comp = [s[l]["compute_us"]["p99.99"] for l in labels]
    x = np.arange(len(labels))
    plt.figure(figsize=(10, 4.5))
    plt.bar(x, comp, label="compute p99.99")
    plt.bar(x, jit, bottom=comp, label="release jitter p99.99")
    plt.xticks(x, labels, rotation=30, ha="right")
    plt.ylabel("us")
    plt.title("What the tail is made of: scheduling jitter vs GPU compute")
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/jitter_vs_compute.png", dpi=150)
    plt.close()


def cdf_tail(labels):
    # baseline vs the profile with the largest p99.99 (read from CSV raw).
    import csv

    def col(path):
        with open(path) as f:
            return np.array([float(r["response_us"])
                             for r in csv.DictReader(f)])

    if not os.path.exists("results/baseline.csv"):
        return
    base = col("results/baseline.csv")
    worst_label, worst = None, base
    for l in labels:
        p = f"results/{l}.csv"
        if l == "baseline" or not os.path.exists(p):
            continue
        d = col(p)
        if d.max() > worst.max() or worst_label is None:
            worst_label, worst = l, d
    if worst_label is None:
        return

    plt.figure(figsize=(8, 5))
    for arr, name in [(base, "baseline"), (worst, worst_label)]:
        xs = np.sort(arr)
        ys = np.arange(1, len(xs) + 1) / len(xs)
        plt.plot(xs, ys, label=name)
    plt.xlabel("response (us)")
    plt.ylabel("CDF")
    plt.ylim(0.99, 1.0001)   # zoom the tail
    plt.title("Response-time tail: baseline vs worst stress")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/cdf_tail.png", dpi=150)
    plt.close()


def main():
    s, labels = load_summaries()
    if not labels:
        print("no results/*.json -- run the matrix first")
        return
    tail_by_profile(s, labels)
    jitter_vs_compute(s, labels)
    cdf_tail(labels)
    print("wrote results/tail_by_profile.png, jitter_vs_compute.png, cdf_tail.png")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Part 3 high-load analysis: stress matrix + memory sweep, with breaking-point
detection. Reads both normal (p3_*) and high-load (p3hi_*) results so the
writeup can contrast them.

  python3 part3/analyze_part3_highload.py
"""
import csv
import glob
import json
import re

GPUT = re.compile(r"gpu@([\d.]+)C")
DEADLINE_MS = 10.0


def load(prefix, name):
    try:
        return json.load(open(f"results/{prefix}_{name}.json"))
    except FileNotFoundError:
        return None


def gpu_temp(path):
    try:
        t = [float(m.group(1)) for ln in open(path) if (m := GPUT.search(ln))]
        return max(t) if t else None
    except FileNotFoundError:
        return None


def near_misses(prefix, name):
    """count cycles within 1 ms of deadline, and over it."""
    try:
        r = [float(x["response_us"]) / 1000
             for x in csv.DictReader(open(f"results/{prefix}_{name}.csv"))]
    except FileNotFoundError:
        return None, None
    over = sum(1 for v in r if v > DEADLINE_MS)
    near = sum(1 for v in r if DEADLINE_MS - 1 <= v <= DEADLINE_MS)
    return near, over


def row(prefix, name):
    j = load(prefix, name)
    if not j:
        return None
    r = j["response_us"]
    near, over = near_misses(prefix, name)
    return (name, r["p50"] / 1000, r["p99.99"] / 1000, r["max"] / 1000,
            j["release_jitter_us"]["p99.99"], j["deadline_miss_count"],
            near, gpu_temp(f"results/{prefix}_{name}.tegra"))


def table(title, prefix, names):
    rows = [row(prefix, n) for n in names]
    rows = [x for x in rows if x]
    if not rows:
        return
    print(f"\n=== {title} ===")
    print("| profile | resp p50 | resp p99.99 | resp max | jitter p99.99 (us) "
          "| misses | within 1ms | gpu °C |")
    print("|---|---|---|---|---|---|---|---|")
    for name, p50, p9999, mx, jit, miss, near, temp in rows:
        flag = "  ⚠" if miss else ""
        print(f"| {name} | {p50:.3f} | {p9999:.3f} | {mx:.3f} | {jit:.1f} | "
              f"{miss}{flag} | {near} | {temp or '-'} |")


def main():
    table("High-load: memory bandwidth sweep (--stream)", "p3hi",
          ["mem_2", "mem_4", "mem_6", "mem_8"])
    table("High-load: vm pressure", "p3hi", ["vm_light", "vm_heavy"])
    table("High-load: other axes maxed", "p3hi",
          ["cache_max", "irq_storm", "combined_max"])
    print("\n(latencies in ms; 'within 1ms' = cycles in [9,10] ms; "
          f"deadline = {DEADLINE_MS:.0f} ms)")
    print("\nBreaking point: first profile with misses>0, or the max-load profile"
          " if none — that bounds what jetson_clocks survives.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Part 3 analysis: for each stressor, does the deadline break, and does it break
COMPUTE (GPU/memory) or JITTER (scheduler/IRQ)?

  python3 part3/analyze_part3.py
"""
import json
import re
import statistics as st

PROFILES = ["baseline", "cpu", "memory", "cache", "io", "irq", "combined"]
GR3D = re.compile(r"GR3D_FREQ\s+\d+%@\[?(\d+)\]?")
GPUT = re.compile(r"gpu@([\d.]+)C")


def load(prof):
    try:
        return json.load(open(f"results/p3_{prof}.json"))
    except FileNotFoundError:
        return None


def tegra_temp(prof):
    try:
        t = [float(m.group(1)) for ln in open(f"results/p3_{prof}.tegra")
             if (m := GPUT.search(ln))]
        return max(t) if t else None
    except FileNotFoundError:
        return None


def main():
    rows = []
    base = load("baseline")
    base_resp = base["response_us"]["p50"] / 1000 if base else None
    for prof in PROFILES:
        j = load(prof)
        if not j:
            continue
        r, c, jt = j["response_us"], j["compute_us"], j["release_jitter_us"]
        rows.append((prof, r["p50"] / 1000, r["p99.99"] / 1000, r["max"] / 1000,
                     c["p99.99"] / 1000, jt["p99.99"], j["deadline_miss_count"],
                     tegra_temp(prof)))

    print("\n=== Stress matrix (100 Hz, dynamic clocks) ===")
    print("| stressor | resp p50 | resp p99.99 | resp max | compute p99.99 | "
          "jitter p99.99 (us) | misses | gpu max °C |")
    print("|---|---|---|---|---|---|---|---|")
    for prof, p50, p9999, mx, cp, jit, miss, temp in rows:
        print(f"| {prof} | {p50:.3f} | {p9999:.3f} | {mx:.3f} | {cp:.3f} | "
              f"{jit:.1f} | {miss} | {temp or '-'} |")
    print("\n(latencies in ms unless noted)\n")

    # interpretation: which component each stressor moved most vs baseline
    if base:
        bc = base["compute_us"]["p99.99"]
        bj = base["release_jitter_us"]["p99.99"]
        print("=== vs baseline: what each stressor broke ===")
        for prof, p50, p9999, mx, cp, jit, miss, temp in rows:
            if prof == "baseline":
                continue
            dc = cp * 1000 - bc
            dj = jit - bj
            # With a clocked baseline, deltas should be >=0; treat small/negative
            # moves as no significant effect (run-to-run noise), not a "culprit".
            if dc < 100 and dj < 50:
                verdict = "no significant effect"
            elif dc > dj:
                verdict = "COMPUTE-bound"
            else:
                verdict = "JITTER-bound"
            print(f"  {prof:<9} compute {dc:+8.1f}us  jitter {dj:+7.1f}us  "
                  f"-> {verdict}" + ("  [DEADLINE MISSES]" if miss else ""))


if __name__ == "__main__":
    main()

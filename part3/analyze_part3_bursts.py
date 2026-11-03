#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Part 3 burst continuation: the miss-clustering statistic of
campaign/analyze_bursts.py, computed on the 16 stress-ng cells of
run_part3.sh (p3_*) and run_part3_highload.sh (p3hi_*), 100k cycles each.

Per cell, at quantile-anchored deadlines D = q-th percentile of response_us
(q99 -> 1% miss, q99.9 -> 0.1%): miss = response_us > D, bursts = maximal runs
of consecutive misses, P[cont] = P[miss_i+1 | miss_i], and P[cont] / P[miss]
(1 = independent misses).  Read-only; prints markdown to stdout.

  python3 part3/analyze_part3_bursts.py
"""
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"

# (file stem, stress-ng arguments as in run_part3.sh / run_part3_highload.sh)
CELLS = [
    ("p3_baseline", "none"),
    ("p3_cpu", "--cpu 6"),
    ("p3_memory", "--vm 4 --vm-bytes 25%"),
    ("p3_cache", "--cache 4"),
    ("p3_io", "--io 4"),
    ("p3_irq", "--timer 8 --timerfd 4 --clock 4"),
    ("p3_combined", "--cpu 4 --vm 2 --vm-bytes 25% --timer 4 --io 2"),
    ("p3hi_mem_2", "--stream 2"),
    ("p3hi_mem_4", "--stream 4"),
    ("p3hi_mem_6", "--stream 6"),
    ("p3hi_mem_8", "--stream 8"),
    ("p3hi_vm_light", "--vm 4 --vm-bytes 600m"),
    ("p3hi_vm_heavy", "--vm 6 --vm-bytes 800m"),
    ("p3hi_cache_max", "--cache 8 --cache-level 3"),
    ("p3hi_irq_storm", "--timer 16 --timerfd 8 --clock 8"),
    ("p3hi_combined_max", "--cpu 6 --stream 4 --vm 4 --vm-bytes 500m --cache 4 --io 4"),
]


def bursts_of(miss):
    """lengths of the maximal runs of True (campaign/analyze_bursts.py)"""
    out, i, n = [], 0, len(miss)
    while i < n:
        if miss[i]:
            j = i
            while j < n and miss[j]:
                j += 1
            out.append(j - i)
            i = j
        else:
            i += 1
    return out


def main():
    total = 0
    print("\n=== Burst continuation at quantile-anchored deadlines (response_us) ===")
    print("| cell | stress-ng | cycles | q | misses | bursts | mean len | max len "
          "| P[cont] | P[cont]/P[miss] |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for stem, args in CELLS:
        f = RES / f"{stem}.csv"
        hdr = f.open().readline().strip().split(",")
        resp = np.loadtxt(f, delimiter=",", skiprows=1)[:, hdr.index("response_us")]
        total += len(resp)
        for q in (99.0, 99.9):
            miss = resp > np.percentile(resp, q)
            bl = bursts_of(miss)
            p = miss.mean()
            cont = (np.count_nonzero(miss[1:] & miss[:-1])
                    / max(np.count_nonzero(miss[:-1]), 1))
            print(f"| {stem} | {args} | {len(resp)} | {q} | {np.count_nonzero(miss)} | "
                  f"{len(bl)} | {np.mean(bl) if bl else 0:.2f} | {max(bl, default=0)} | "
                  f"{cont:.2f} | {cont / p:.0f} |")
    print(f"\n{len(CELLS)} cells, {total} cycles in total.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
G2 campaign analysis.

Part A (RQ1): latency curve over the 4 lockable EMC points per workload —
p50/p99 table plus per-segment shift, the non-monotonicity evidence.

Part B (RQ2): tail structure under contention from the raw 100k-cycle CSVs —
percentiles, post-hoc deadline-miss curves (deadline as analysis parameter),
tail amplification (p99.99/p50), miss burstiness (P[miss | prev miss] vs
P[miss]) at a tight deadline, and the adversary's achieved GB/s per cell.
"""

import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent.parent / "results" / "campaign_g2"

WL_ORDER = ["mobilenet", "vit", "proxy", "cproxyv1", "cproxyv2", "slm"]
EMC_POINTS = [204, 665, 2133, 3199]

# post-hoc deadline grids (ms) per Part B workload
DEADLINES = {
    "mobilenet": [5.0, 5.5, 6.0, 7.0, 8.0, 10.0, 20.0],
    "proxy": [7.0, 8.0, 9.0, 10.0, 12.0, 20.0, 40.0],
}


def part_a():
    cells = defaultdict(dict)
    for f in sorted((OUT / "partA").glob("emc*.json")):
        m = re.match(r"emc(\d+)_(\w+)\.json", f.name)
        d = json.loads(f.read_text())
        cells[m.group(2)][int(m.group(1))] = d["compute_us"]

    print("== Part A: RQ1 EMC curve (compute p50 ms; shift vs next-lower point) ==")
    hdr = f"{'workload':<10}"
    for mhz in EMC_POINTS:
        hdr += f"{mhz:>8} {'Δ%':>7}"
    print(hdr + f"{'p99/p50@3199':>14}")
    for wl in WL_ORDER:
        row = cells[wl]
        line = f"{wl:<10}"
        prev = None
        for mhz in EMC_POINTS:
            p50 = row[mhz]["p50"] / 1000
            d = f"{(prev - p50) / p50 * 100:+.0f}%" if prev is not None else ""
            line += f"{p50:>8.2f} {d:>7}"
            prev = p50
        amp = row[3199]["p99"] / row[3199]["p50"]
        print(line + f"{amp:>14.3f}")
    print()


def load_cycles(csv_path: Path):
    a = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    return a[:, 1], a[:, 2], a[:, 3]   # jitter_us, compute_us, response_us


def part_b():
    print("== Part B: RQ2 tail under contention (response_us from raw 100k) ==")
    rows = []
    for f in sorted((OUT / "partB").glob("emc*_adv*_*.csv")):
        m = re.match(r"emc(\d+)_adv(\d+)_(\w+)\.csv", f.name)
        if not m:
            continue
        mhz, adv, wl = int(m.group(1)), int(m.group(2)), m.group(3)
        jitter, compute, response = load_cycles(f)
        bw = None
        bwf = f.with_suffix("").with_suffix("")  # strip .csv
        bwj = f.parent / f"{f.stem}.membw.json"
        if bwj.exists() and bwj.stat().st_size > 0:
            try:
                bw = json.loads(bwj.read_text())["avg_gbps"]
            except (json.JSONDecodeError, KeyError):
                pass
        rows.append((wl, mhz, adv, jitter, compute, response, bw))

    print(f"{'cell':<26} {'n':>7} {'p50':>8} {'p99':>8} {'p99.9':>8} "
          f"{'p99.99':>8} {'max':>9} {'p9999/p50':>10} {'advGB/s':>8}")
    for wl, mhz, adv, jitter, compute, response, bw in rows:
        q = np.percentile(response, [50, 99, 99.9, 99.99])
        print(f"emc{mhz}_adv{adv}_{wl:<10} {len(response):>7} "
              f"{q[0]/1000:>8.2f} {q[1]/1000:>8.2f} {q[2]/1000:>8.2f} "
              f"{q[3]/1000:>8.2f} {response.max()/1000:>9.2f} "
              f"{q[3]/q[0]:>10.3f} {bw if bw else '':>8}")
    print()

    print("== Part B: post-hoc deadline-miss rate (%) ==")
    for wl in ("mobilenet", "proxy"):
        dls = DEADLINES[wl]
        print(f"-- {wl}: deadlines " + "  ".join(f"{d}ms" for d in dls))
        for wl2, mhz, adv, jitter, compute, response, bw in rows:
            if wl2 != wl:
                continue
            cells = []
            for dl in dls:
                miss = float((response > dl * 1000).mean() * 100)
                cells.append(f"{miss:>8.3f}")
            print(f"  emc{mhz}_adv{adv}: " + " ".join(cells))
        print()

    print("== Part B: miss burstiness at tight deadline "
          "(P[miss|prev miss] / P[miss]; >1 = clustered) ==")
    for wl, mhz, adv, jitter, compute, response, bw in rows:
        # tight deadline = own p99.9 → 0.1% expected miss rate
        dl = np.percentile(response, 99.9)
        miss = response > dl
        p = miss.mean()
        both = np.count_nonzero(miss[1:] & miss[:-1])
        cond = both / max(np.count_nonzero(miss[:-1]), 1)
        ratio = cond / p if p > 0 else float("nan")
        print(f"  emc{mhz}_adv{adv}_{wl:<10} P[miss]={p*100:.3f}% "
              f"P[miss|prev]={cond*100:.3f}% ratio={ratio:,.1f}")
    print()

    print("== Part B: release jitter vs compute share of tail "
          "(p99.9 decomposition, us) ==")
    for wl, mhz, adv, jitter, compute, response, bw in rows:
        print(f"  emc{mhz}_adv{adv}_{wl:<10} "
              f"jitter p99.9={np.percentile(jitter, 99.9):>8.1f} "
              f"compute p99.9={np.percentile(compute, 99.9):>9.1f} "
              f"response p99.9={np.percentile(response, 99.9):>9.1f}")


def main():
    if not OUT.exists():
        sys.exit(f"no campaign results at {OUT}")
    part_a()
    part_b()


if __name__ == "__main__":
    main()

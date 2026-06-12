#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Adversary effectiveness verdict.

Reads results/pilot_contention/emc<MHz>_adv<N>_<workload>.json (harness
summaries) plus the matching .membw.json (adversary's own achieved GB/s) and
prints, per EMC point and workload, latency vs adversary level.

Pass criterion (pre-registered in run_adv_check.sh): proxy p50 slowdown >20%
at any adversary level. If even the bandwidth-bound victim doesn't feel a
bandwidth adversary, the adversary needs redesign before the RQ2 campaign.
"""

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "results" / "pilot_contention"
LABEL_RE = re.compile(r"emc(\d+)_adv(\d+)_(\w+)\.json$")
PASS_SLOWDOWN = 0.20


def main():
    cells = defaultdict(dict)  # (workload, emc_mhz) -> adv_n -> (summary, gbps)
    for f in sorted(OUT.glob("emc*_adv*_*.json")):
        if f.name.endswith(".membw.json"):
            continue
        m = LABEL_RE.search(f.name)
        if not m:
            continue
        mhz, adv, wl = int(m.group(1)), int(m.group(2)), m.group(3)
        gbps = None
        bw = OUT / f"emc{mhz}_adv{adv}_{wl}.membw.json"
        if bw.exists():
            try:
                gbps = json.loads(bw.read_text())["avg_gbps"]
            except (json.JSONDecodeError, KeyError):
                pass
        cells[(wl, mhz)][adv] = (json.loads(f.read_text()), gbps)

    if not cells:
        sys.exit(f"no results under {OUT} — run pilot_contention/run_adv_check.sh first")

    print(f"{'workload':<8} {'EMC':>5} {'adv':>4} {'p50 us':>10} {'p99 us':>10} "
          f"{'max us':>10} {'p50 slow':>9} {'p99 slow':>9} {'adv GB/s':>9}")
    verdict_pass = False
    missing_baseline = []
    for (wl, mhz) in sorted(cells):
        base = cells[(wl, mhz)].get(0)
        if base is None:
            missing_baseline.append((wl, mhz))
        base_p50 = base[0]["compute_us"]["p50"] if base else None
        base_p99 = base[0]["compute_us"]["p99"] if base else None
        for adv in sorted(cells[(wl, mhz)]):
            summary, gbps = cells[(wl, mhz)][adv]
            c = summary["compute_us"]
            if base is None:
                slow_s = slow99_s = "  n/a"
            else:
                slow = (c["p50"] - base_p50) / base_p50 if adv else 0.0
                slow99 = (c["p99"] - base_p99) / base_p99 if adv else 0.0
                slow_s = f"{slow * 100:>+8.1f}%"
                slow99_s = f"{slow99 * 100:>+8.1f}%"
                if wl == "proxy" and slow > PASS_SLOWDOWN:
                    verdict_pass = True
            print(f"{wl:<8} {mhz:>5} {adv:>4} {c['p50']:>10.1f} {c['p99']:>10.1f} "
                  f"{c['max']:>10.1f} {slow_s:>9} {slow99_s:>9} "
                  f"{gbps if gbps is not None else '':>9}")
        print()

    if missing_baseline:
        print(f"!! BASELINE (adv=0) MISSING for {missing_baseline} — "
              "those groups are excluded; verdict below is not complete")

    print("== verdict ==")
    if verdict_pass:
        print(f"ADVERSARY EFFECTIVE: proxy p50 slowdown exceeded "
              f"{PASS_SLOWDOWN:.0%}. Usable for the RQ2 contention matrix.")
    else:
        print("ADVERSARY INEFFECTIVE: bandwidth-bound victim unmoved — "
              "redesign (more threads, read mode, larger strides) before RQ2.")


if __name__ == "__main__":
    main()

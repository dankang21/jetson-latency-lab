#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
EMC pilot verdict: does the memory clock move inference latency enough to
carry RQ1, and is the effect workload-dependent (small on compute-heavy,
large on memory-bound)?

Reads results/pilot_emc/emc<MHz>_<workload>.json (written by run_emc_pilot.sh)
plus the .tegra traces to confirm the EMC lock actually held during each run.

Kill criteria (agreed before the campaign):
  - RQ1 is DEAD if p50 compute shift between max and min EMC is <5% for
    every workload.
  - RQ1 is ALIVE in its strongest form if the shift is small for cnn and
    large for proxy (workload-dependent error of the "constant b" model).

Uses compute_us (inference only), not response_us — release jitter is
Part 3's business, not this pilot's.
"""

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "results" / "pilot_emc"
KILL_THRESHOLD = 0.05  # 5% p50 shift, per the pre-registered kill criteria

LABEL_RE = re.compile(r"emc(\d+)_(\w+)\.json$")
# tegrastats prints "EMC_FREQ 7%@2133" when the rate is exposed, "EMC_FREQ 7%" when not
TEGRA_EMC_RE = re.compile(r"EMC_FREQ\s+\d+%(?:@(\d+))?")


def tegra_emc_mhz(path: Path):
    """Distinct EMC MHz values seen in a tegrastats trace (empty if not exposed)."""
    seen = set()
    try:
        for line in path.read_text().splitlines():
            m = TEGRA_EMC_RE.search(line)
            if m and m.group(1):
                seen.add(int(m.group(1)))
    except FileNotFoundError:
        pass
    return sorted(seen)


def main():
    cells = defaultdict(dict)  # workload -> emc_mhz -> summary
    for f in sorted(OUT.glob("emc*_*.json")):
        m = LABEL_RE.search(f.name)
        if not m:
            continue
        mhz, workload = int(m.group(1)), m.group(2)
        cells[workload][mhz] = json.loads(f.read_text())

    if not cells:
        sys.exit(f"no results under {OUT} — run pilot_emc/run_emc_pilot.sh first")

    print(f"{'workload':<8} {'EMC MHz':>8} {'p50 us':>10} {'p90':>10} "
          f"{'p99':>10} {'max':>10}  emc held (tegra)")
    verdicts = {}
    for workload in sorted(cells):
        by_mhz = cells[workload]
        for mhz in sorted(by_mhz):
            c = by_mhz[mhz]["compute_us"]
            held = tegra_emc_mhz(OUT / f"emc{mhz}_{workload}.tegra")
            held_s = ",".join(map(str, held)) if held else "n/a"
            print(f"{workload:<8} {mhz:>8} {c['p50']:>10.1f} {c['p90']:>10.1f} "
                  f"{c['p99']:>10.1f} {c['max']:>10.1f}  {held_s}")
        lo, hi = min(by_mhz), max(by_mhz)
        p50_lo = by_mhz[lo]["compute_us"]["p50"]
        p50_hi = by_mhz[hi]["compute_us"]["p50"]
        shift = (p50_lo - p50_hi) / p50_hi
        verdicts[workload] = shift
        print(f"{'':8} p50 shift EMC {hi}->{lo} MHz: {shift * 100:+.1f}%\n")

    print("== verdict ==")
    if all(abs(s) < KILL_THRESHOLD for s in verdicts.values()):
        print(f"RQ1 DEAD: every workload shifted <{KILL_THRESHOLD:.0%}. "
              "Drop the EMC contribution and restructure around RQ2/RQ3.")
    else:
        for w, s in sorted(verdicts.items(), key=lambda kv: -abs(kv[1])):
            print(f"  {w}: {s * 100:+.1f}%")
        spread = max(verdicts.values()) - min(verdicts.values())
        if spread >= KILL_THRESHOLD:
            print("RQ1 ALIVE, strongest form: the shift is workload-dependent — "
                  "exactly the b(f_emc) error structure the paper argues.")
        else:
            print("RQ1 alive but uniform across workloads — EMC matters, but "
                  "the workload-dependence claim needs more diverse models.")


if __name__ == "__main__":
    main()

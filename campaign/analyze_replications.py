#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Tracked analysis for the power-mode replications (the numbers behind the
25W replication sentences in RQ1 and Table powermode): EMC shifts under the
25W profile, the inversion check at GPU 918MHz, and the stock-governor EMC
traces under 15W / 25W / MAXN_SUPER.
"""

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TEGRA_RE = re.compile(r"EMC_FREQ (\d+)%@(\d+)")


def part_shifts():
    print("== 25W profile (CPU 1344 / GPU 918 MHz pinned): p50 by EMC ==")
    t = defaultdict(dict)
    for f in sorted((ROOT / "results/replication_25w").glob("r25_emc*_*.json")):
        m = re.match(r"r25_emc(\d+)_(\w+)\.json", f.name)
        t[m.group(2)][int(m.group(1))] = \
            json.loads(f.read_text())["compute_us"]["p50"] / 1000
    print(f"{'workload':<10}" + "".join(f"{m:>9}" for m in (204, 665, 2133, 3199))
          + f"{'2133 vs 3199':>14}")
    for wl in ("mobilenet", "proxy", "cproxyv2"):
        r = t[wl]
        shift = (r[2133] - r[3199]) / r[3199] * 100
        print(f"{wl:<10}"
              + "".join(f"{r.get(m, float('nan')):>9.2f}" for m in (204, 665, 2133, 3199))
              + f"{shift:>+13.1f}%")
    inv = (t["cproxyv2"][2133] - t["cproxyv2"][3199]) / t["cproxyv2"][3199]
    print(f"inversion at GPU 918MHz: {'YES' if inv < 0 else 'NO'} "
          f"({inv*100:+.1f}% — MAXN/GPU1020 showed -9.1%)")
    print()


def governor_traces():
    print("== stock-governor EMC traces (rates observed @10Hz sampling) ==")
    cases = [
        ("MAXN_SUPER", ROOT / "results/e1_governor", ["idle", "mobilenet", "proxy", "slm"]),
        ("25W", ROOT / "results/replication_25w",
         ["governor", "gov_idle", "gov_mobilenet", "gov_proxy", "gov_slm"]),
        ("15W", ROOT / "results/replication_15w", ["idle", "mobilenet", "proxy", "slm"]),
    ]
    for mode, base, names in cases:
        for name in names:
            f = base / f"{name}.tegra"
            if not f.exists():
                continue
            rates = [int(m.group(2)) for line in f.read_text().splitlines()
                     if (m := TEGRA_RE.search(line))]
            c = Counter(rates)
            trans = sum(1 for a, b in zip(rates, rates[1:]) if a != b)
            print(f"  {mode:<11} {name:<10} n={len(rates):>4} "
                  f"rates={dict(c.most_common())} transitions={trans}")
    print()


if __name__ == "__main__":
    part_shifts()
    governor_traces()

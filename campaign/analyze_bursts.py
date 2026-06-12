#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
RQ2 burst forensics on the G2 Part B raw cycles.

Per cell, at quantile-anchored tight deadlines (q99 -> 1% miss, q99.9 -> 0.1%):
  - burst structure: run lengths of consecutive misses, hazard of continuation,
    expected length under independence (~1.00-1.01) for contrast
  - burst timing: inter-burst-start intervals (median / CV; CV~1 = Poisson
    arrivals, CV<<1 = periodic, CV>>1 = clustered arrivals)
  - board-state join: tegrastats rows (timestamped on R36) within each burst
    window vs the cell-wide baseline — EMC busy%, GPU temp, RAM
  - EVT: GPD fit (PWM, Hosking & Wallis 1987 — no scipy on this box) over
    p99 exceedances: shape xi (>0 heavy, ~0 exponential, <0 bounded),
    EVT-extrapolated p99.99 vs empirical
  - the pre-registered RQ2 kill check: does mean+3sigma cover p99.9?

Anchoring: harness JSON timestamp is taken right after the measurement loop,
so cycle i's wall time = end - (iters - i)/hz (+-1s; tegra interval is 500ms,
so joins are ~2-sample resolution — burst-level, not cycle-level).
"""

import json
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent.parent / "results" / "campaign_g2" / "partB"

TEGRA_RE = re.compile(
    r"^(\d\d-\d\d-\d{4} \d\d:\d\d:\d\d)\s.*?RAM (\d+)/\d+MB.*?"
    r"EMC_FREQ (\d+)%(?:@(\d+))?.*?GR3D_FREQ (\d+)%.*?gpu@([\d.]+)C",
)


def parse_tegra(path: Path):
    rows = []
    for line in path.read_text().splitlines():
        m = TEGRA_RE.match(line)
        if not m:
            continue
        t = datetime.strptime(m.group(1), "%m-%d-%Y %H:%M:%S")
        rows.append((t.replace(tzinfo=timezone.utc).timestamp(),
                     int(m.group(2)), int(m.group(3)), int(m.group(5)),
                     float(m.group(6))))
    return np.array(rows) if rows else None


def bursts_of(miss: np.ndarray):
    """(start_idx, length) for each maximal run of True."""
    out = []
    i, n = 0, len(miss)
    while i < n:
        if miss[i]:
            j = i
            while j < n and miss[j]:
                j += 1
            out.append((i, j - i))
            i = j
        else:
            i += 1
    return out


def gpd_pwm(excess: np.ndarray):
    """GPD (xi, sigma) via probability-weighted moments."""
    x = np.sort(excess)
    n = len(x)
    if n < 20:
        return None, None
    b0 = x.mean()
    p = 1.0 - (np.arange(1, n + 1) - 0.35) / n
    b1 = (x * p).mean()
    denom = b0 - 2 * b1
    if abs(denom) < 1e-12:
        return None, None
    xi = 2.0 - b0 / denom
    sigma = 2.0 * b0 * b1 / denom
    return xi, sigma


def gpd_quantile(u, xi, sigma, zeta_u, p_target):
    """Quantile at exceedance prob p_target given threshold-exceed prob zeta_u."""
    r = p_target / zeta_u
    if abs(xi) < 1e-6:
        return u - sigma * math.log(r)
    return u + sigma / xi * (r ** (-xi) - 1.0)


def main():
    files = sorted(OUT.glob("emc*_adv*_*.csv"))
    files = [f for f in files if re.match(r"emc\d+_adv\d+_\w+\.csv$", f.name)]
    if not files:
        sys.exit(f"no Part B csvs under {OUT}")

    print("== burst structure at quantile-anchored deadlines ==")
    print(f"{'cell':<26} {'q':>5} {'misses':>7} {'bursts':>7} {'meanlen':>8} "
          f"{'maxlen':>7} {'len>=5':>7} {'P[cont]':>8} {'indep':>6}")
    for f in files:
        label = f.stem
        a = np.loadtxt(f, delimiter=",", skiprows=1)
        resp = a[:, 3]
        for q in (99.0, 99.9):
            dl = np.percentile(resp, q)
            miss = resp > dl
            bl = bursts_of(miss)
            lens = np.array([l for _, l in bl]) if bl else np.array([0])
            p = miss.mean()
            cont = (np.count_nonzero(miss[1:] & miss[:-1])
                    / max(np.count_nonzero(miss[:-1]), 1))
            print(f"{label:<26} {q:>5} {np.count_nonzero(miss):>7} "
                  f"{len(bl):>7} {lens.mean():>8.2f} {lens.max():>7} "
                  f"{np.count_nonzero(lens >= 5):>7} {cont:>8.3f} "
                  f"{1/(1-p):>6.3f}")
    print()

    print("== burst arrival pattern (q99.9; inter-burst-start intervals, s) ==")
    print(f"{'cell':<26} {'bursts':>7} {'median_s':>9} {'CV':>6}  interpretation")
    for f in files:
        label = f.stem
        meta = json.loads((f.parent / f"{f.stem}.json").read_text())["meta"]
        hz = meta["hz"]
        a = np.loadtxt(f, delimiter=",", skiprows=1)
        resp = a[:, 3]
        dl = np.percentile(resp, 99.9)
        bl = bursts_of(resp > dl)
        starts = np.array([i for i, _ in bl]) / hz
        if len(starts) < 3:
            print(f"{label:<26} {len(bl):>7}  too few bursts")
            continue
        iv = np.diff(starts)
        cv = iv.std() / iv.mean()
        interp = ("~periodic" if cv < 0.4 else
                  "~Poisson" if cv < 1.5 else "clustered arrivals")
        print(f"{label:<26} {len(bl):>7} {np.median(iv):>9.2f} {cv:>6.2f}  {interp}")
    print()

    print("== board state during bursts vs baseline (q99.9, bursts len>=2) ==")
    print(f"{'cell':<26} {'join':>5} {'EMC% burst/base':>16} "
          f"{'gputemp burst/base':>19} {'RAM MB burst/base':>18}")
    for f in files:
        label = f.stem
        meta = json.loads((f.parent / f"{f.stem}.json").read_text())["meta"]
        hz, iters = meta["hz"], meta["iters"]
        end = datetime.fromisoformat(meta["timestamp"]).timestamp()
        start = end - iters / hz
        tegra = parse_tegra(f.parent / f"{f.stem}.tegra")
        if tegra is None:
            print(f"{label:<26}  no tegra")
            continue
        a = np.loadtxt(f, delimiter=",", skiprows=1)
        resp = a[:, 3]
        dl = np.percentile(resp, 99.9)
        bl = [(i, l) for i, l in bursts_of(resp > dl) if l >= 2]
        # baseline: tegra rows inside the measurement window
        in_win = (tegra[:, 0] >= start) & (tegra[:, 0] <= end)
        base = tegra[in_win]
        sel = np.zeros(len(tegra), dtype=bool)
        for i, l in bl:
            t0, t1 = start + i / hz - 1.0, start + (i + l) / hz + 1.0
            sel |= (tegra[:, 0] >= t0) & (tegra[:, 0] <= t1)
        b = tegra[sel]
        if len(b) == 0 or len(base) == 0:
            print(f"{label:<26} {len(b):>5}  (no joined rows)")
            continue
        print(f"{label:<26} {len(b):>5} "
              f"{b[:,2].mean():>7.1f}/{base[:,2].mean():<8.1f} "
              f"{b[:,4].mean():>9.1f}/{base[:,4].mean():<9.1f} "
              f"{b[:,1].mean():>8.0f}/{base[:,1].mean():<9.0f}")
    print()

    print("== EVT (GPD over p99 exceedances) + Gaussian-margin kill check ==")
    print(f"{'cell':<26} {'xi':>7} {'p99.99 evt/emp us':>19} "
          f"{'mean+3s':>9} {'p99.9':>9} {'3s covers?':>10}")
    for f in files:
        label = f.stem
        a = np.loadtxt(f, delimiter=",", skiprows=1)
        resp = a[:, 3]
        u = np.percentile(resp, 99.0)
        exc = resp[resp > u] - u
        xi, sigma = gpd_pwm(exc)
        emp9999 = np.percentile(resp, 99.99)
        evt9999 = (gpd_quantile(u, xi, sigma, 0.01, 1e-4)
                   if xi is not None else float("nan"))
        m3 = resp.mean() + 3 * resp.std()
        p999 = np.percentile(resp, 99.9)
        print(f"{label:<26} {xi if xi is not None else float('nan'):>7.3f} "
              f"{evt9999:>9.1f}/{emp9999:<9.1f} {m3:>9.1f} {p999:>9.1f} "
              f"{'YES' if m3 >= p999 else 'NO':>10}")


if __name__ == "__main__":
    main()

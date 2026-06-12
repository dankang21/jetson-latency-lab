#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Trace-driven governor simulation (reviewer item: estimator/governor impact
end-to-end, no governor implementation needed).

Scenario: a deadline governor picks the LOWEST GPU frequency (energy proxy)
whose predicted latency + margin fits the deadline, targeting <=0.1% miss.
It was profiled at EMC 3199 MHz (the 25W/MAXN profile); the system is
deployed at EMC 2133 MHz (the 15W profile) -- exactly the power-mode pair
of Table powermode. Choices are scored against the MEASURED 300-cycle
distribution of the chosen (gpu, 2133) cell.

Policies (point model x margin model):
  A+G   GPU-only fit @3199, Gaussian margin (mu+3sigma of the @3199 cell)
  A+P   GPU-only fit @3199, GPD margin (p99.9 from @3199 cell exceedances)
  C+G   2-cell refit @2133 (knows deployment EMC), Gaussian margin (pooled)
  C+P   2-cell refit @2133, GPD margin (pooled 2-cell exceedances)

Metrics over a deadline grid per workload:
  viol%   fraction of deadlines where achieved miss > 1/300 (the granularity
          floor of a 300-cycle cell; the 100k Part B cells calibrate finer)
  miss%   mean achieved miss across violated deadlines
  f_avg   mean chosen GPU frequency in MHz (lower = cheaper, energy proxy)
  infeas% deadlines where the policy declares no feasible frequency
          (system then runs at max frequency; scored there)
"""

import json
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent.parent / "results" / "estimator_break"
WLS = ["mobilenet", "vit", "proxy", "cproxyv2"]
GPUS = [306, 408, 510, 612, 714, 816, 918, 1020]
TARGET = 1e-3


def gpd_pwm(excess):
    x = np.sort(excess)
    n = len(x)
    if n < 15:
        return None, None
    b0 = x.mean()
    pw = 1.0 - (np.arange(1, n + 1) - 0.35) / n
    b1 = (x * pw).mean()
    d = b0 - 2 * b1
    if abs(d) < 1e-12:
        return None, None
    return 2.0 - b0 / d, 2.0 * b0 * b1 / d


def gpd_q(u, xi, sigma, zeta, p):
    r = p / zeta
    if abs(xi) < 1e-6:
        return u - sigma * math.log(r)
    return u + sigma / xi * (r ** (-xi) - 1.0)


def load_cells():
    """wl -> (emc, gpu) -> per-cycle compute_us array"""
    d = defaultdict(dict)
    for f in OUT.glob("emc*_gpu*_*.csv"):
        m = re.match(r"emc(\d+)_gpu(\d+)_(\w+)\.csv", f.name)
        if not m:
            continue
        a = np.loadtxt(f, delimiter=",", skiprows=1)
        d[m.group(3)][(int(m.group(1)), int(m.group(2)))] = a[:, 2]  # compute_us
    return d


def fit_gpu_only(cells, emc):
    f = np.array([1e3 / g for g in GPUS])
    y = np.array([np.median(cells[(emc, g)]) for g in GPUS])
    X = np.column_stack([f, np.ones_like(f)])
    k, b = np.linalg.lstsq(X, y, rcond=None)[0]
    return k, b


def fit_two_cell(cells, emc):
    f1, f2 = 1e3 / 1020, 1e3 / 306
    y1 = np.median(cells[(emc, 1020)])
    y2 = np.median(cells[(emc, 306)])
    k = (y2 - y1) / (f2 - f1)
    return k, y1 - k * f1


def gpd_margin(samples, p_target):
    """quantile(p_target exceedance) - median, from a profiling sample."""
    u = np.percentile(samples, 90.0)
    exc = samples[samples > u] - u
    xi, sg = gpd_pwm(exc)
    if xi is None:
        return np.percentile(samples, 99.9) - np.median(samples)
    q = gpd_q(u, xi, sg, (samples > u).mean(), p_target)
    return q - np.median(samples)


DUMP = {}


def main():
    data = load_cells()
    print(f"{'workload':<10} {'policy':<6} {'viol%':>7} {'miss% (viol)':>13} "
          f"{'f_avg MHz':>10} {'infeas%':>8}")
    for wl in WLS:
        cells = data[wl]
        kA, bA = fit_gpu_only(cells, 3199)
        kC, bC = fit_two_cell(cells, 2133)

        # margins are RELATIVE (ratio to the cell median), then scaled by the
        # policy's own latency prediction: pooling raw samples across two
        # frequency levels would count the level gap as variance.
        def rel_gauss(samples):
            return (samples.mean() + 3 * samples.std()) / np.median(samples)

        def rel_gpd(samples):
            med = np.median(samples)
            return (med + gpd_margin(samples, TARGET)) / med

        normC = np.concatenate([
            cells[(2133, 1020)] / np.median(cells[(2133, 1020)]),
            cells[(2133, 306)] / np.median(cells[(2133, 306)]),
        ])
        ratio_CG, ratio_CP = rel_gauss(normC), rel_gpd(normC)
        margins = {}
        for g in GPUS:
            prof_A = cells[(3199, g)]
            margins[g] = {
                "A+G": (rel_gauss(prof_A) - 1),
                "A+P": (rel_gpd(prof_A) - 1),
                "C+G": (ratio_CG - 1),
                "C+P": (ratio_CP - 1),
            }

        pred = {
            "A+G": lambda g: kA * 1e3 / g + bA,
            "A+P": lambda g: kA * 1e3 / g + bA,
            "C+G": lambda g: kC * 1e3 / g + bC,
            "C+P": lambda g: kC * 1e3 / g + bC,
        }

        lat_lo = np.median(cells[(2133, 1020)])
        lat_hi = np.percentile(cells[(2133, 306)], 99.9)
        deadlines = np.linspace(0.95 * lat_lo, 1.3 * lat_hi, 60)

        for pol in ("A+G", "A+P", "C+G", "C+P"):
            viol, misses, fsel, infeas = 0, [], [], 0
            curve = {"D_ms": [], "miss_pct": [], "f_mhz": []}
            for D in deadlines:
                feasible = [g for g in GPUS
                            if pred[pol](g) * (1 + margins[g][pol]) <= D]
                if feasible:
                    g = min(feasible)
                else:
                    g = 1020
                    infeas += 1
                fsel.append(g)
                miss = (cells[(2133, g)] > D).mean()
                curve["D_ms"].append(round(D / 1000, 4))
                curve["miss_pct"].append(round(miss * 100, 4))
                curve["f_mhz"].append(g)
                if miss > 1.0 / len(cells[(2133, g)]):
                    viol += 1
                    misses.append(miss)
            n = len(deadlines)
            DUMP.setdefault(wl, {})[pol] = {
                "curve": curve,
                "viol_pct": round(viol / n * 100, 1),
                "miss_on_viol_pct": round(float(np.mean(misses)) * 100, 1) if misses else 0.0,
                "f_avg_mhz": round(float(np.mean(fsel))),
                "infeas_pct": round(infeas / n * 100, 1),
                "n_deadlines": n,
            }
            print(f"{wl:<10} {pol:<6} {viol/n*100:>6.0f}% "
                  f"{np.mean(misses)*100 if misses else 0:>12.2f} "
                  f"{np.mean(fsel):>10.0f} {infeas/n*100:>7.0f}%")
        print()


    (OUT / "govsim_curves.json").write_text(
        json.dumps(DUMP, indent=1))
    print(f"curves -> {OUT / 'govsim_curves.json'}")


if __name__ == "__main__":
    main()

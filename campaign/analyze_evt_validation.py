#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
EVT claims hardening (reviewer-grade validation of the RQ2 tail section).

The draft claimed "GPD extrapolates p99.99 within ~2%" from in-sample fits.
That is open to an overfitting objection: same data fitted and evaluated,
and an empirical p99.99 of 100k cycles is ~10 order statistics. This script
adds the out-of-sample and stability evidence:

1. Split-sample: GPD fitted on the FIRST 50k cycles (threshold = its p99)
   predicts the p99.9 / p99.99 of the LAST 50k. Temporal split, not random:
   harder, and matches deployment (profile first, run later).
2. Threshold sensitivity: shape xi and predicted p99.99 at u in
   {p98.5, p99, p99.5} on the full data.
3. Bootstrap CI: 500 resamples of the exceedances -> 90% CI on predicted
   p99.99; is the empirical value inside?
4. Margin -> QoS: set a deadline from the first 50k by (a) Gaussian
   mu+3sigma and (b) GPD-predicted p99.9; report the ACHIEVED miss rate on
   the last 50k against the 0.1% design target.
"""

import re
import sys
import math
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent.parent / "results" / "campaign_g2" / "partB"
RNG = np.random.default_rng(42)
B = 500


def gpd_pwm(excess):
    x = np.sort(excess)
    n = len(x)
    if n < 20:
        return None, None
    b0 = x.mean()
    pw = 1.0 - (np.arange(1, n + 1) - 0.35) / n
    b1 = (x * pw).mean()
    denom = b0 - 2 * b1
    if abs(denom) < 1e-12:
        return None, None
    return 2.0 - b0 / denom, 2.0 * b0 * b1 / denom


def gpd_q(u, xi, sigma, zeta, p):
    r = p / zeta
    if abs(xi) < 1e-6:
        return u - sigma * math.log(r)
    return u + sigma / xi * (r ** (-xi) - 1.0)


def main():
    files = sorted(f for f in OUT.glob("emc*_adv*_*.csv")
                   if re.match(r"emc\d+_adv\d+_\w+\.csv$", f.name))
    if not files:
        sys.exit("no Part B csvs")

    print("== 1. split-sample: fit first 50k -> predict last 50k ==")
    print(f"{'cell':<26} {'xi(1st)':>8} {'pred p99.9':>11} {'emp p99.9':>10} "
          f"{'err%':>6} {'pred p99.99':>12} {'emp p99.99':>11} {'err%':>6}")
    for f in files:
        resp = np.loadtxt(f, delimiter=",", skiprows=1)[:, 3]
        a, b = resp[:50000], resp[50000:]
        u = np.percentile(a, 99.0)
        xi, sg = gpd_pwm(a[a > u] - u)
        zeta = (a > u).mean()
        p999_pred = gpd_q(u, xi, sg, zeta, 1e-3)
        p9999_pred = gpd_q(u, xi, sg, zeta, 1e-4)
        p999_emp = np.percentile(b, 99.9)
        p9999_emp = np.percentile(b, 99.99)
        print(f"{f.stem:<26} {xi:>8.3f} {p999_pred:>11.1f} {p999_emp:>10.1f} "
              f"{(p999_pred-p999_emp)/p999_emp*100:>+5.1f}% "
              f"{p9999_pred:>12.1f} {p9999_emp:>11.1f} "
              f"{(p9999_pred-p9999_emp)/p9999_emp*100:>+5.1f}%")
    print()

    print("== 2. threshold sensitivity (full data) ==")
    print(f"{'cell':<26} " + "".join(f"{'xi@'+q:>10}{'p99.99':>9}"
                                     for q in ("98.5", "99", "99.5")))
    for f in files:
        resp = np.loadtxt(f, delimiter=",", skiprows=1)[:, 3]
        row = f"{f.stem:<26} "
        for q in (98.5, 99.0, 99.5):
            u = np.percentile(resp, q)
            xi, sg = gpd_pwm(resp[resp > u] - u)
            zeta = (resp > u).mean()
            row += f"{xi:>10.3f}{gpd_q(u, xi, sg, zeta, 1e-4):>9.1f}"
        print(row)
    print()

    print(f"== 3. bootstrap ({B} resamples of exceedances): 90% CI on "
          "predicted p99.99 vs empirical ==")
    print(f"{'cell':<26} {'pred':>9} {'CI5%':>9} {'CI95%':>9} "
          f"{'empirical':>10} {'inside?':>8}")
    for f in files:
        resp = np.loadtxt(f, delimiter=",", skiprows=1)[:, 3]
        u = np.percentile(resp, 99.0)
        exc = resp[resp > u] - u
        zeta = (resp > u).mean()
        preds = []
        for _ in range(B):
            s = RNG.choice(exc, size=len(exc), replace=True)
            xi, sg = gpd_pwm(s)
            if xi is not None:
                preds.append(gpd_q(u, xi, sg, zeta, 1e-4))
        preds = np.array(preds)
        xi, sg = gpd_pwm(exc)
        point = gpd_q(u, xi, sg, zeta, 1e-4)
        lo, hi = np.percentile(preds, [5, 95])
        emp = np.percentile(resp, 99.99)
        print(f"{f.stem:<26} {point:>9.1f} {lo:>9.1f} {hi:>9.1f} {emp:>10.1f} "
              f"{'YES' if lo <= emp <= hi else 'no':>8}")
    print()

    print("== 4. margin policy -> achieved QoS (deadline set from first 50k, "
          "applied to last 50k; design target 0.1% miss) ==")
    print(f"{'cell':<26} {'gauss DL us':>12} {'miss%':>8} "
          f"{'evt DL us':>10} {'miss%':>8}")
    for f in files:
        resp = np.loadtxt(f, delimiter=",", skiprows=1)[:, 3]
        a, b = resp[:50000], resp[50000:]
        dl_g = a.mean() + 3 * a.std()
        u = np.percentile(a, 99.0)
        xi, sg = gpd_pwm(a[a > u] - u)
        dl_e = gpd_q(u, xi, sg, (a > u).mean(), 1e-3)
        miss_g = (b > dl_g).mean() * 100
        miss_e = (b > dl_e).mean() * 100
        print(f"{f.stem:<26} {dl_g:>12.1f} {miss_g:>8.3f} "
              f"{dl_e:>10.1f} {miss_e:>8.3f}")


def declustered(resp, q=99.0, gap=2):
    """Runs-declustering: exceedance clusters separated by >= gap
    non-exceedances; fit GPD on cluster MAXIMA. Returns predicted p99.99
    using the cluster rate. Marginal-quantile heuristic under dependence."""
    u = np.percentile(resp, q)
    exceed_idx = np.flatnonzero(resp > u)
    if len(exceed_idx) < 20:
        return None
    clusters, cur = [], [exceed_idx[0]]
    for a, b in zip(exceed_idx, exceed_idx[1:]):
        if b - a <= gap:
            cur.append(b)
        else:
            clusters.append(cur)
            cur = [b]
    clusters.append(cur)
    maxima = np.array([resp[c].max() for c in clusters]) - u
    xi, sg = gpd_pwm(maxima)
    if xi is None:
        return None
    zeta_c = len(clusters) / len(resp)
    return gpd_q(u, xi, sg, zeta_c, 1e-4), len(clusters), len(exceed_idx)


def section5():
    files = sorted(f for f in OUT.glob("emc*_adv*_*.csv")
                   if re.match(r"emc\d+_adv\d+_\w+\.csv$", f.name))
    print()
    print("== 5. declustering sensitivity (runs method, gap=2): predicted "
          "p99.99 ==")
    print(f"{'cell':<26} {'raw fit':>9} {'declustered':>12} {'delta%':>7} "
          f"{'clusters/exceed':>16}")
    for f in files:
        resp = np.loadtxt(f, delimiter=",", skiprows=1)[:, 3]
        u = np.percentile(resp, 99.0)
        exc = resp[resp > u] - u
        xi, sg = gpd_pwm(exc)
        raw = gpd_q(u, xi, sg, (resp > u).mean(), 1e-4)
        d = declustered(resp)
        if d is None:
            print(f"{f.stem:<26} {raw:>9.1f}  (too few clusters)")
            continue
        dq, nc, ne = d
        print(f"{f.stem:<26} {raw:>9.1f} {dq:>12.1f} "
              f"{(dq-raw)/raw*100:>+6.1f}% {nc:>7}/{ne:<8}")



if __name__ == "__main__":
    main()
    section5()

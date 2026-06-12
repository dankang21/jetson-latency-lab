#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
E2/E3: how badly does a CPU x GPU-only estimator break across EMC points,
and how much does an EMC-aware term recover?

Data: results/estimator_break/emc{3199,2133,665}_gpu{306..1020}_{wl}.json
(8 GPU devfreq points x 3 EMC points x 4 workloads, 300 iters/cell, CPU
pinned).

Models (fit on p50 latency, least squares):
  A  CPU x GPU-only   : T = k/f_gpu + b            fit: EMC 3199 (8 pts)
  B  parametric EMC   : T = k/f_gpu + m/f_emc + b  fit: EMC 3199+665.6 (16)
  C  tabulated EMC    : T = k/f_gpu + b[f_emc]     fit: k,b from 3199; b
                        offset per EMC from ONE extra cell (gpu=1020) each
Test: all three evaluated on the held-out EMC 2133 sweep (and A also on
665.6 — the out-of-scope collapse). Errors are relative to measured p50.

Model C is what "profile the lockable points" means in practice: 4 extra
measurements per workload. Model B is the natural parametric guess; it
CANNOT represent the cproxyv2 inversion (m/f_emc is monotonic in f_emc),
which is the point of including it.
"""

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent.parent / "results" / "estimator_break"
WLS = ["mobilenet", "vit", "proxy", "cproxyv2"]
GPUS = [306, 408, 510, 612, 714, 816, 918, 1020]
EMCS = [3199, 2133, 665]


def load():
    d = defaultdict(dict)   # wl -> (emc, gpu) -> p50_us
    for f in OUT.glob("emc*_gpu*_*.json"):
        m = re.match(r"emc(\d+)_gpu(\d+)_(\w+)\.json", f.name)
        if not m:
            continue
        emc, gpu, wl = int(m.group(1)), int(m.group(2)), m.group(3)
        d[wl][(emc, gpu)] = json.loads(f.read_text())["compute_us"]["p50"]
    return d


def fit_A(cells):
    """T = k/f_gpu + b on EMC 3199."""
    f = np.array([1e3 / g for g in GPUS])           # 1/GHz
    y = np.array([cells[(3199, g)] for g in GPUS])
    X = np.column_stack([f, np.ones_like(f)])
    k, b = np.linalg.lstsq(X, y, rcond=None)[0]
    return k, b


def fit_B(cells):
    """T = k/f_gpu + m/f_emc + b on EMC 3199 + 665.6."""
    rows, y = [], []
    for emc in (3199, 665):
        femc = 665.6 if emc == 665 else float(emc)
        for g in GPUS:
            rows.append([1e3 / g, 1e3 / femc, 1.0])
            y.append(cells[(emc, g)])
    sol = np.linalg.lstsq(np.array(rows), np.array(y), rcond=None)[0]
    return sol  # k, m, b


def main():
    data = load()
    missing = [(wl, e, g) for wl in WLS for e in EMCS for g in GPUS
               if (e, g) not in data.get(wl, {})]
    if missing:
        sys.exit(f"missing cells: {missing[:5]} ... ({len(missing)})")

    print("== model A (CPU x GPU-only, fit @EMC3199): relative p50 error ==")
    print(f"{'workload':<10} {'in-scope@3199':>14} {'@2133 med/max':>16} "
          f"{'@665.6 med/max':>16}")
    errsA = {}
    for wl in WLS:
        cells = data[wl]
        k, b = fit_A(cells)
        def pred(g):
            return k * 1e3 / g + b
        e3199 = [abs(pred(g) - cells[(3199, g)]) / cells[(3199, g)]
                 for g in GPUS]
        e2133 = [(pred(g) - cells[(2133, g)]) / cells[(2133, g)] for g in GPUS]
        e665 = [(pred(g) - cells[(665, g)]) / cells[(665, g)] for g in GPUS]
        errsA[wl] = (e2133, e665)
        print(f"{wl:<10} {np.median(e3199)*100:>13.1f}% "
              f"{np.median(np.abs(e2133))*100:>7.1f}/{np.max(np.abs(e2133))*100:<7.1f} "
              f"{np.median(np.abs(e665))*100:>7.1f}/{np.max(np.abs(e665))*100:<7.1f}")
    print()

    print("== model comparison on held-out EMC 2133 (median/max |rel err| %) ==")
    print(f"{'workload':<10} {'A: GPU-only':>13} {'B: +m/f_emc':>13} "
          f"{'C1: b offset':>13} {'C2: 2-cell refit':>17}")
    for wl in WLS:
        cells = data[wl]
        kA, bA = fit_A(cells)
        kB, mB, bB = fit_B(cells)
        # C1: k from A; per-EMC offset from the single gpu=1020 cell at 2133
        offC = cells[(2133, 1020)] - (kA * 1e3 / 1020 + bA)
        # C2: refit BOTH k and b from just two cells at the target EMC
        # (gpu 1020 + 306) — the deployment-realistic per-point profile
        f1, f2 = 1e3 / 1020, 1e3 / 306
        y1, y2 = cells[(2133, 1020)], cells[(2133, 306)]
        kC2 = (y2 - y1) / (f2 - f1)
        bC2 = y1 - kC2 * f1
        eA, eB, eC, eC2 = [], [], [], []
        for g in GPUS:
            t = cells[(2133, g)]
            eA.append(abs(kA * 1e3 / g + bA - t) / t)
            eB.append(abs(kB * 1e3 / g + mB * 1e3 / 2133.0 + bB - t) / t)
            eC.append(abs(kA * 1e3 / g + bA + offC - t) / t)
            eC2.append(abs(kC2 * 1e3 / g + bC2 - t) / t)
        print(f"{wl:<10} "
              f"{np.median(eA)*100:>5.1f}/{np.max(eA)*100:<6.1f} "
              f"{np.median(eB)*100:>5.1f}/{np.max(eB)*100:<6.1f} "
              f"{np.median(eC)*100:>5.1f}/{np.max(eC)*100:<6.1f} "
              f"{np.median(eC2)*100:>5.1f}/{np.max(eC2)*100:<6.1f}")
    print()

    print("== governor consequence: sign of model-A error at 2133 ==")
    print("(positive = overestimates latency -> conservative;")
    print(" negative = underestimates -> picks a frequency that misses)")
    for wl in WLS:
        e2133, _ = errsA[wl]
        sign = "UNDER" if np.median(e2133) < 0 else "over"
        print(f"  {wl:<10} median signed err {np.median(e2133)*100:+.1f}% -> {sign}estimates")


if __name__ == "__main__":
    main()

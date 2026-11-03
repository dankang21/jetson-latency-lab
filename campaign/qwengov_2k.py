#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Table V (Qwen2.5-1.5B token-decode governor on the NX, EMC 2133 MHz): the 2,000-token column
"@2k [UB]", and the Qwen-1.5B 2,000-token entry of Table III (its D = 29 ms/tok cell).

For each deadline row of Table V (D = 27, 29, 31 ms/tok) the aware pick f* is the deployed rule of
campaign/analyze_qwen_deploy.py: T(f) = k/f + b least-squares over the EMC-2133 sweep cells
results/qwen_deploy/qsweep_emc2133_gpu<f>.json (compute_us p50, ms/token), r = 1.035, f* = the lowest
deployed clock (one with results/qwen_deploy/qdep_g<f>_r1.csv) with T(f) r <= D, else the top one;
the blind pick is the same rule on the EMC-3199 sweep.  The 2,000-token runs of the aware pick are
results/boundary_stats/q15_g<f*>_r<k>.csv (2,000 decoded tokens each: five runs, eight at 1173 MHz).
  per run   miss % = tokens with dur_ns > D (strict) / 2,000; also the miss of the first and the last 500
            tokens pooled over the runs (the decode-horizon erosion of the main text)
  measured  mean of the per-run miss % (every run has 2,000 tokens, so it equals the pooled rate)
  UB        run-level 95 % upper bound: 95th percentile of the mean per-run miss over B = 20,000
            resamples of whole runs (numpy default_rng(42), the analyze_joint.py hxt() convention);
            B = 10,000 with seed 0 is printed as a seed/B check
The table prints measured and UB to one decimal.  The blind arm, the 300-token cells and the
energy ordering of Table V are printed by campaign/analyze_qwen_deploy.py.

Read-only on results/ (python3 + numpy); prints to stdout, writes nothing.
Usage: qwengov_2k.py [--json]
"""
import argparse
import csv
import glob
import json
import os
import re

import numpy as np

LAB = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QD = os.path.join(LAB, "results", "qwen_deploy")
BS = os.path.join(LAB, "results", "boundary_stats")
R = 1.035
DEPLOYED = [714, 816, 918, 1020, 1122, 1173]     # analyze_qwen_deploy.py DEP
ROWS_D = (27, 29, 31)                            # Table V deadline rows, ms/token
TAU = 2.0
WIN = 500                                        # first / last tokens of a run (horizon windows)


def sweep_fit(emc):
    """(k, b) of T(f) = k/f + b, ms/token, least squares over the sweep cells at one EMC point."""
    pts = {}
    for f in glob.glob(os.path.join(QD, f"qsweep_emc{emc}_gpu*.json")):
        g = int(re.search(r"gpu(\d+)", f).group(1))
        v = json.load(open(f))["compute_us"]["p50"] / 1000.0
        if v:
            pts[g] = v
    fs = np.array(sorted(pts), float)
    ts = np.array([pts[k] for k in sorted(pts)], float)
    A = np.vstack([1 / fs, np.ones_like(fs)]).T
    (k, b), *_ = np.linalg.lstsq(A, ts, rcond=None)
    return float(k), float(b), pts


def picks(D):
    """(blind f*, aware f*) of the deployed rule at D ms/token."""
    G = [f for f in DEPLOYED if os.path.exists(os.path.join(QD, f"qdep_g{f}_r1.csv"))]
    out = []
    for emc in (3199, 2133):
        k, b, _ = sweep_fit(emc)
        out.append(next((f for f in G if (k / f + b) * R <= D), G[-1]))
    return tuple(out)


def boot_ub(per, B, seed):
    rng = np.random.default_rng(seed)
    return float(np.quantile([rng.choice(per, len(per), replace=True).mean() for _ in range(B)], 0.95))


def cell_2k(D, f):
    """Per-run and summary miss of the 2,000-token runs at clock f, deadline D ms/token."""
    files = sorted(glob.glob(os.path.join(BS, f"q15_g{f}_r*.csv")),
                   key=lambda p: int(re.search(r"_r(\d+)\.csv$", p).group(1)))
    assert files, f"no 2,000-token runs at {f} MHz under results/boundary_stats"
    per, n, head, tail = [], [], [], []
    for p in files:
        v = np.array([float(r["dur_ns"]) for r in csv.DictReader(open(p)) if r.get("dur_ns")]) / 1e6
        per.append(100.0 * float(np.mean(v > D)))
        n.append(int(v.size))
        head.append(v[:WIN])
        tail.append(v[-WIN:])
    per = np.array(per)
    return dict(D_ms=D, f=f, n=len(per), tokens=n, files=[os.path.relpath(p, LAB) for p in files],
                per_run_pct=[float(x) for x in per], mean_pct=float(per.mean()),
                ub_pct=boot_ub(per, 20000, 42), ub_check_pct=boot_ub(per, 10000, 0),
                runs_over_tau=int(np.sum(per > TAU)), min_pct=float(per.min()), max_pct=float(per.max()),
                head500_pct=100.0 * float(np.mean(np.concatenate(head) > D)),
                tail500_pct=100.0 * float(np.mean(np.concatenate(tail) > D)))


def table():
    rows = []
    for D in ROWS_D:
        fb, fa = picks(D)
        c = cell_2k(D, fa)
        c["blind_f"] = fb
        rows.append(c)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", action="store_true", help="print the rows as JSON")
    a = ap.parse_args()
    rows = table()
    if a.json:
        print(json.dumps(rows, indent=1))
        return
    kb, bb, _ = sweep_fit(3199)
    ka, ba, _ = sweep_fit(2133)
    print("Table V, 2,000-token column: aware pick, five 2,000-token repeats (eight at the D = 27 cell)")
    print(f"sweep fits (ms/token): blind @3199 k={kb:.1f} b={bb:.3f}; aware @2133 k={ka:.1f} b={ba:.3f}; r = {R}")
    for c in rows:
        print(f"\nD = {c['D_ms']} ms/tok: blind f* {c['blind_f']}, aware f* {c['f']}; {c['n']} runs "
              f"({', '.join(c['files'][:1])} ..), tokens per run {sorted(set(c['tokens']))}")
        print("    per-run miss % = " + " ".join(f"{x:.2f}" for x in c["per_run_pct"]))
        print(f"    measured (mean of runs) {c['mean_pct']:.3f} %; run-level 95 % UB {c['ub_pct']:.3f} % "
              f"(B = 20000, rng 42; check B = 10000, seed 0: {c['ub_check_pct']:.3f} %); "
              f"runs > tau {c['runs_over_tau']}/{c['n']}; range {c['min_pct']:.2f}-{c['max_pct']:.2f} %")
        print(f"    first {WIN} tokens {c['head500_pct']:.3f} %, last {WIN} tokens {c['tail500_pct']:.3f} % (pooled over the runs)")
    print("\n| D (ms/tok) | blind f* | aware f* | runs | @2k miss % [UB] |")
    print("|---:|---:|---:|---:|---|")
    for c in rows:
        print(f"| {c['D_ms']} | {c['blind_f']} | {c['f']} | {c['n']} | {c['mean_pct']:.1f} [{c['ub_pct']:.1f}] |")
    t3 = next(c for c in rows if c["D_ms"] == 29)
    print(f"\nTable III, Qwen-1.5B (NX) row (D = 29 ms/tok, aware {t3['f']} MHz): 2,000-token miss "
          f"{t3['mean_pct']:.1f} [{t3['ub_pct']:.1f}] % (exact {t3['mean_pct']:.3f} [{t3['ub_pct']:.3f}])")


if __name__ == "__main__":
    main()

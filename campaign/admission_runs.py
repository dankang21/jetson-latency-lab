#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Per-run miss rates behind the admission-outcome table (Table VI, tab:admission) and the
co-tenancy ablation ladder (Supp., tab:jointgov): every row recomputed from the per-run CSVs.

Read-only over results/ (python3 + numpy).  Run from anywhere:
    python3 campaign/admission_runs.py [--json]

Miss definitions (as campaign/analyze_joint.py):
  vision : response_us > D_us               (results/joint_gov/dep_co_g<f>_r<i>.csv, 1,000 cycles per run)
  decode : dur_ns / 1e3 > D_us per token    (results/boundary_stats/q15_g1173_r<i>.csv, 2,000 tokens;
                                             results/horizon_4k/q15h4k_g<f>_r<i>.csv, 4,000 tokens;
                                             results/joint_hxt/q15co_g1173_r<i>.csv, 2,000 tokens under the co-runner)
A run violates if its own miss % exceeds tau = 2 %.  'pooled' = misses / cycles over the row's runs
(all runs of a row have the same length, so it equals the mean of the per-run rates).
Run-level bound (analyze_joint.py hxt()): np.random.default_rng(42), B = 20000 resamples of whole runs,
the 95th percentile of the resampled mean per-run miss; B = 10000 with seed 0 is printed as a seed/B check.
With three runs that percentile is exactly (2 max + mid) / 3 (closed form printed), which cannot reach the
worst run, so a three-run row reports its largest run ('max', not a confidence bound) and a row with
n >= 8 its bootstrap bound.
Usage: admission_runs.py [--json]
"""
import argparse
import csv
import glob
import json
import os
import re

import numpy as np

LAB = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(LAB, "results")
TAU = 2.0

ROWS = [
    # (table, row label, file pattern under results/, column, scale, D_us, token window)
    ("Table VI", "vision co, D=9 ms, declined @1122", "joint_gov/dep_co_g1122_r*.csv", "response_us", 1.0, 9000, None),
    ("Table VI", "vision co, D=12 ms, 918 MHz", "joint_gov/dep_co_g918_r*.csv", "response_us", 1.0, 12000, None),
    ("Table VI", "decode, D=28, H=2k, 1173 MHz", "boundary_stats/q15_g1173_r*.csv", "dur_ns", 1e3, 28000, None),
    ("Table VI", "decode, D=28, H=4k, declined @1173", "horizon_4k/q15h4k_g1173_r*.csv", "dur_ns", 1e3, 28000, None),
    ("Table VI", "decode, D=28, H=4k @1173, last 500 tokens (dagger)", "horizon_4k/q15h4k_g1173_r*.csv", "dur_ns", 1e3, 28000, (3500, 4000)),
    ("Table VI", "decode co, D=28, declined @1173", "joint_hxt/q15co_g1173_r*.csv", "dur_ns", 1e3, 28000, None),
    ("Table VI", "decode co, D=43, declined @1173", "joint_hxt/q15co_g1173_r*.csv", "dur_ns", 1e3, 43000, None),
    ("Table VI", "decode co, D=44, 1173 MHz", "joint_hxt/q15co_g1173_r*.csv", "dur_ns", 1e3, 44000, None),
    ("ladder", "blind/EMC-aware (solo), D=9, 510", "joint_gov/dep_co_g510_r*.csv", "response_us", 1.0, 9000, None),
    ("ladder", "+tenancy flat M, D=9, 1020", "joint_gov/dep_co_g1020_r*.csv", "response_us", 1.0, 9000, None),
    ("ladder", "+tenancy M f_probe/f, D=9, 1122", "joint_gov/dep_co_g1122_r*.csv", "response_us", 1.0, 9000, None),
    ("ladder", "blind/EMC-aware (solo), D=12, 510", "joint_gov/dep_co_g510_r*.csv", "response_us", 1.0, 12000, None),
    ("ladder", "+tenancy flat M, D=12, 510", "joint_gov/dep_co_g510_r*.csv", "response_us", 1.0, 12000, None),
    ("ladder", "+tenancy M f_probe/f, D=12, 816", "joint_gov/dep_co_g816_r*.csv", "response_us", 1.0, 12000, None),
    ("ladder", "+tenancy max guard, D=12, 918", "joint_gov/dep_co_g918_r*.csv", "response_us", 1.0, 12000, None),
    ("ladder", "tenancy-blind (decode, co), 28/tok, 1173", "joint_hxt/q15co_g1173_r*.csv", "dur_ns", 1e3, 28000, None),
    ("horizon", "horizon-blind decode, D=28, H=4k @1122", "horizon_4k/q15h4k_g1122_r*.csv", "dur_ns", 1e3, 28000, None),
    ("horizon", "horizon-blind decode, D=28, H=4k @1122, last 500 tokens", "horizon_4k/q15h4k_g1122_r*.csv", "dur_ns", 1e3, 28000, (3500, 4000)),
]


def runs(pat, col, scale):
    fs = sorted(glob.glob(pat), key=lambda f: int(re.search(r"_r(\d+)\.csv$", f).group(1)))
    out = []
    for f in fs:
        v = np.array([float(x[col]) for x in csv.DictReader(open(f)) if x.get(col)]) / scale
        out.append((os.path.relpath(f, RES), v))
    return out


def boot(per, B, seed):
    rng = np.random.default_rng(seed)
    b = [rng.choice(per, len(per), replace=True).mean() for _ in range(B)]
    return float(np.quantile(b, .95))


def score(tab, lab, pat, col, scale, D, win):
    rr = runs(os.path.join(RES, pat), col, scale)
    assert rr, f"no runs match results/{pat}"
    per, cnt, ncyc, files = [], [], [], []
    for f, v in rr:
        if win:
            v = v[win[0]:win[1]]
        k = int(np.sum(v > D))
        per.append(100.0 * k / len(v)); cnt.append(k); ncyc.append(len(v)); files.append(f)
    per = np.array(per)
    pooled = 100.0 * sum(cnt) / sum(ncyc)
    srt = np.sort(per)
    cf = (2 * srt[-1] + srt[-2]) / 3 if len(per) == 3 else None
    ub42 = boot(per, 20000, 42)
    ub0 = boot(per, 10000, 0)
    return dict(table=tab, row=lab, n=len(per), D_us=D, window=win, files=files, cycles_per_run=ncyc, misses=cnt,
                per_run_pct=[round(float(x), 4) for x in per], mean_pct=round(float(per.mean()), 4),
                pooled_pct=round(pooled, 4), min_pct=round(float(per.min()), 4), max_pct=round(float(per.max()), 4),
                ub_B20000_seed42=round(ub42, 4), ub_B10000_seed0=round(ub0, 4),
                ub_closed_form_n3=None if cf is None else round(float(cf), 4),
                runs_over_tau=int(np.sum(per > TAU)),
                reported_bound=("max", round(float(per.max()), 4)) if len(per) == 3 else ("UB", round(ub42, 4)))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", action="store_true", help="print the per-row record as JSON instead of the text")
    a = ap.parse_args()
    res = [score(*r) for r in ROWS]
    if a.json:
        print(json.dumps(res, indent=1))
        return
    for r in res:
        per = np.array(r["per_run_pct"])
        cf = r["ub_closed_form_n3"]
        print(f"[{r['table']}] {r['row']}")
        print(f"    n={r['n']} D={r['D_us']}us win={r['window']} cycles/run={sorted(set(r['cycles_per_run']))} misses={r['misses']}")
        print(f"    per-run % = {' '.join(f'{x:.3f}' for x in per)}")
        print(f"    mean {r['mean_pct']:.3f}  pooled {r['pooled_pct']:.3f}  min {r['min_pct']:.3f}  max {r['max_pct']:.3f}  "
              f"UB(B=20000,s42) {r['ub_B20000_seed42']:.3f}  UB(B=10000,s0) {r['ub_B10000_seed0']:.3f}  "
              f"closed-form {'' if cf is None else f'{cf:.3f}'}  runs>tau {r['runs_over_tau']}/{r['n']}")
    print("\n| table | row | n | measured % (pooled) | run range % | bound reported (n >= 8: UB; n = 3: max) % | runs > tau |")
    print("|---|---|---:|---:|---|---|---:|")
    for r in res:
        kind, val = r["reported_bound"]
        print(f"| {r['table']} | {r['row']} | {r['n']} | {r['pooled_pct']:.3f} | {r['min_pct']:.3f}-{r['max_pct']:.3f} | "
              f"{kind} {val:.3f} | {r['runs_over_tau']}/{r['n']} |")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""E3 transfer campaign: launch tally per condition and by setting, recomputed from the trial files.

An independent tally (no campaign imports) that cross-checks the per-condition counts of
campaign/e3_stats.py and sums them by setting (the admission-evidence table, tab:admitevidence).
Read-only over results/ (python3 + numpy).  Run from anywhere:
    python3 campaign/e3_tally.py [--json]

Per condition directory results/e3_<board>/<cond>/:
  launch k exists iff t<k>.done exists (completed launch);
  the deployed decision is t<k>.pick.json field f_b ('INFEASIBLE' = declined);
  realized banded miss m_b = share of t<k>.contract_b.csv rows with response_us > D
     (D from condition.txt D_ms);  a launch violates iff m_b > tau = 2 %;
  pooled = cycles-weighted banded miss over the admitted launches (P2's point value);
  f_oracle = lowest grid clock whose ref_g<f>.csv miss at D is <= tau;
  P1 = admitted launches with f_b < f_oracle;
  band necessity: the band fired iff t<k>.pick.json f_pt != f_b; each removed point admission
     (f_pt feasible) ran as its own contract t<k>.contract_pt.csv, scored like m_b ('pt fail').
The contract's own label suffix _g<f> is checked against f_b and every contract against 1,000 cycles.
'Scored' set = every condition except the superseded first runs A5 and A6 (classified at the time as
protocol-execution failures, re-run as A5r/A6r under the readiness-verified runner; printed for completeness).
Setting lines give, besides the sums, the range of the pooled miss over the conditions, how many
conditions keep it within tau, and how many have P1 = 0.
Usage: e3_tally.py [--json]
"""
import argparse
import csv
import glob
import json
import os
import re

import numpy as np

LAB = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TAU = 0.02
SUPERSEDED = {"A5", "A6"}


def miss(path, D_us):
    v = np.array([float(r["response_us"]) for r in csv.DictReader(open(path)) if r.get("response_us")])
    return int(np.count_nonzero(v > D_us)), int(v.size)


def cond(board, cid):
    d = f"{LAB}/results/e3_{board}/{cid}"
    c = dict(tok.split("=", 1) for tok in open(f"{d}/condition.txt").read().split() if "=" in tok)
    D_us = float(c["D_ms"]) * 1000.0
    grid = [int(x) for x in c["grid"].split(",")]
    ks = sorted(int(re.search(r"/t(\d+)\.done$", p).group(1)) for p in glob.glob(f"{d}/t*.done"))
    ref = {}
    for f in grid:
        m, n = miss(f"{d}/ref_g{f}.csv", D_us)
        ref[f] = m / n
    fo = next((f for f in grid if ref[f] <= TAU), None)
    L = []
    for k in ks:
        pj = json.load(open(f"{d}/t{k}.pick.json"))
        fb, fpt = pj["f_b"], pj["f_pt"]
        # band necessity: the band fired iff the point pick differs; the removed point admission
        # (f_pt feasible) ran as its own contract, t<k>.contract_pt.csv
        band = dict(fired=fpt != fb, pt_fail=None)
        if fpt != fb and fpt != "INFEASIBLE":
            m, n = miss(f"{d}/t{k}.contract_pt.csv", D_us)
            band["pt_fail"] = m / n > TAU
        if fb == "INFEASIBLE":
            L.append(dict(k=k, declined=True, **band))
            continue
        m, n = miss(f"{d}/t{k}.contract_b.csv", D_us)
        lab = json.load(open(f"{d}/t{k}.contract_b.json"))["meta"].get("label", "")
        g = re.search(r"_g(\d+)$", lab)
        ran_at = int(g.group(1)) if g else None
        L.append(dict(k=k, declined=False, fb=int(fb), m=m / n, n=n, misses=m, ran_at=ran_at, **band))
    adm = [x for x in L if not x["declined"]]
    return dict(board=board, cond=cid, co=c.get("co"), co_hz=c.get("co_hz"), primary=c.get("primary"),
                n=len(L), adm=len(adm), declined=[x["k"] for x in L if x["declined"]],
                viol=sum(1 for x in adm if x["m"] > TAU), P1=sum(1 for x in adm if fo is not None and x["fb"] < fo),
                fo=fo, pooled=sum(x["misses"] for x in adm) / max(1, sum(x["n"] for x in adm)),
                mmax=max((x["m"] for x in adm), default=None),
                fired=sum(1 for x in L if x["fired"]), removed=sum(1 for x in L if x["pt_fail"] is not None),
                removed_fail=sum(1 for x in L if x["pt_fail"]),
                ran_mismatch=[x["k"] for x in adm if x["ran_at"] is not None and x["ran_at"] != x["fb"]],
                ncyc_bad=[x["k"] for x in adm if x["n"] != 1000])


def tally(sub, lab):
    n = sum(r["n"] for r in sub); a = sum(r["adm"] for r in sub)
    v = sum(r["viol"] for r in sub); p = sum(r["P1"] for r in sub)
    pooled = [100 * r["pooled"] for r in sub]
    print(f"{lab:62s} conditions {len(sub):2d}  launches {n:3d}  admitted {a:3d}  declined {n - a}  "
          f"violating launches {v:3d}/{a}  P1 {p:3d}/{a}  |  pooled miss {min(pooled):.2f}-{max(pooled):.2f} % "
          f"(within tau {sum(1 for r in sub if r['pooled'] <= TAU)}/{len(sub)})  P1 = 0 {sum(1 for r in sub if r['P1'] == 0)}/{len(sub)}  |  "
          f"band fired {sum(r['fired'] for r in sub)}/{n}, removed point admissions failing "
          f"{sum(r['removed_fail'] for r in sub)}/{sum(r['removed'] for r in sub)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", action="store_true", help="also print the per-condition record as JSON at the end")
    a = ap.parse_args()
    rows = []
    for board in ("nx", "nano"):
        for cid in sorted(os.listdir(f"{LAB}/results/e3_{board}")):
            if os.path.isdir(f"{LAB}/results/e3_{board}/{cid}"):
                rows.append(cond(board, cid))
    print(f"{'cond':5} {'board':5} {'primary':9} {'tenant':12} {'n':>3} {'adm':>4} {'decl':6} {'>tau':>5} {'P1':>4} {'f_o':>5} {'pooled%':>8} {'max%':>6} "
          f"{'band':>5} {'pt fail':>8} scored")
    for r in rows:
        print(f"{r['cond']:5} {r['board']:5} {r['primary']:9} {r['co'] + ' ' + r['co_hz'] + 'Hz':12} {r['n']:>3} {r['adm']:>4} "
              f"{str(r['declined']):6} {r['viol']:>5} {r['P1']:>4} {str(r['fo']):>5} {100 * r['pooled']:8.2f} "
              f"{100 * r['mmax']:6.1f} {r['fired']:>5} {str(r['removed_fail']) + '/' + str(r['removed']):>8} "
              f"{'no (superseded)' if r['cond'] in SUPERSEDED else 'yes'}"
              + (f"  RAN-AT MISMATCH {r['ran_mismatch']}" if r["ran_mismatch"] else "")
              + (f"  CYCLES!=1000 {r['ncyc_bad']}" if r["ncyc_bad"] else ""))
    sc = [r for r in rows if r["cond"] not in SUPERSEDED]
    print()
    tally(sc, "scored (all conditions)")
    tally([r for r in sc if r["board"] == "nx"], "  NX")
    tally([r for r in sc if r["board"] == "nano"], "  Nano")
    inc = [r for r in sc if r["primary"] == "mobilenet" and r["cond"] != "C0"]
    tally(inc, "  phase-sweeping MobileNetV2 transfer (N1-N4, A1-A4)")
    tally([r for r in inc if r["board"] == "nx"], "    NX (N1-N4)")
    tally([r for r in inc if r["board"] == "nano"], "    Nano (A1-A4)")
    tally([r for r in sc if r["primary"] == "mobilenet" and r["board"] == "nx"], "  NX MobileNetV2 incl. C0 (C0, N1-N4)")
    com = [r for r in sc if r["primary"] == "vit" and r["co_hz"] in ("5", "10")]
    tally(com, "  integer-period-multiple ViT-S (N5-N8, A5r, A6r)")
    mech = [r for r in sc if r["co_hz"] == "7"]
    tally(mech, "  7 Hz mechanism tests (N9, A7)")
    tally([r for r in sc if r["cond"] == "C0"], "  control C0")
    tally(inc + com, "  14 transfer conditions (excl. C0, N9, A7)")
    tally([r for r in rows if r["cond"] in SUPERSEDED], "superseded A5/A6 (not scored; for completeness)")
    if a.json:
        print(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()

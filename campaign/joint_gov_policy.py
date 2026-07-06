#!/usr/bin/env python3
"""Joint EMC+horizon+tenancy governor: the pick rule.

T_hat(f)   = k/f' + b            fit per EMC point (2-cell refit @2133;
                                 blind arm fits the full sweep @3199)
T_hat(f,H) = T_hat(f) * (1 + gamma*H)   horizon layer (vision: gamma=0)
feasible   : T_hat(f,H) * r + M_tail <= D
M_tail     = p98 - p50 of a 200-cycle max-clock probe under the co-runner
             (solo policies: M_tail = 0)
Pick the lowest feasible clock; declare INFEASIBLE if none.
"""
import glob, json, re, sys
import numpy as np

LAB = "/home/dk/mobile/jetson-latency-lab"
# deployment candidate set = the live governor's re-pick set (hz33-viable clocks)
GRID = [510, 612, 714, 816, 918, 1020, 1122, 1173]
R = 1.035

def p50(path):
    return json.load(open(path))["response_us"]["p50"]

def fit_2cell(emc):  # aware refit: endpoints only
    pts = {int(re.search(r"gpu(\d+)", f).group(1)): p50(f)
           for f in glob.glob(f"{LAB}/results/nx_validation/nx_est_emc{emc}_gpu*_mobilenet.json")}
    lo, hi = min(pts), max(pts)
    x1, x2 = 1e3 / hi, 1e3 / lo
    k = (pts[lo] - pts[hi]) / (x2 - x1)
    return k, pts[hi] - k * x1

def fit_sweep(emc):  # blind arm: least squares over its own sweep
    pts = {int(re.search(r"gpu(\d+)", f).group(1)): p50(f)
           for f in glob.glob(f"{LAB}/results/nx_validation/nx_est_emc{emc}_gpu*_mobilenet.json")}
    fs = np.array(sorted(pts), float)
    ts = np.array([pts[g] for g in sorted(pts)], float)
    A = np.vstack([1e3 / fs, np.ones_like(fs)]).T
    (k, b), *_ = np.linalg.lstsq(A, ts, rcond=None)
    return k, b

def m_tail():
    j = json.load(open(f"{LAB}/results/joint_gov/probe_co_g1173.json"))
    import csv
    v = np.array([float(x["response_us"]) for x in
                  csv.DictReader(open(f"{LAB}/results/joint_gov/probe_co_g1173.csv"))
                  if x.get("response_us")])
    return float(np.quantile(v, 0.98) - np.median(v)), len(v)

def pick(k, b, D_us, m=0.0, gamma=0.0, H=0):
    for f in GRID:
        t = (k * (1e3 / f) + b) * (1 + gamma * H)
        if t * R + m <= D_us:
            return f, t
    return None, None

if __name__ == "__main__":
    kb, bb = fit_sweep(3199)     # blind: profiled at stock-max EMC
    ka, ba = fit_2cell(2133)     # aware: 2-cell refit at deployment EMC
    M, n = m_tail()
    print(f"tenancy probe: n={n}  M_tail={M/1000:.2f} ms")
    for D in (9000, 12000):
        rows = [("blind",    pick(kb, bb, D)),
                ("emc-solo", pick(ka, ba, D)),
                ("joint",    pick(ka, ba, D, m=M))]
        for name, (f, t) in rows:
            t_s = f"T_hat={t/1000:.2f}ms" if f else "no feasible clock"
            print(f"  D={D//1000}ms  {name:9s} -> {f if f else 'INFEASIBLE'}  ({t_s})")

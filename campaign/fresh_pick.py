#!/usr/bin/env python3
"""Per-trial admission decision from a FRESH probe (runs on the board).
vision mode: guarded pick for D=12ms from a 200-cycle MNv2 probe.
decode mode: first accepted integer deadline from a 200-token probe."""
import csv, glob, json, re, sys
import numpy as np
LAB = "/home/dk/mobile/jetson-latency-lab"
mode, probe_csv = sys.argv[1], sys.argv[2]
R = 1.035
if mode == "vision":
    v = np.array([float(x["response_us"]) for x in csv.DictReader(open(probe_csv)) if x.get("response_us")])
    Mg = v.max() - np.median(v)
    pts = {int(re.search(r"gpu(\d+)", f).group(1)): json.load(open(f))["response_us"]["p50"]
           for f in glob.glob(f"{LAB}/results/nx_validation/nx_est_emc2133_gpu*_mobilenet.json")}
    lo, hi = min(pts), max(pts)
    k = (pts[lo]-pts[hi])/(1e3/lo-1e3/hi); b = pts[hi]-k*(1e3/hi)
    pick = next((g for g in [510,612,714,816,918,1020,1122,1173]
                 if (k*1e3/g+b)*R + Mg*1173/g <= 12000), 0)
    print(f"{Mg:.0f} {pick}")
else:
    v = np.array([float(x["dur_ns"]) for x in csv.DictReader(open(probe_csv))]) / 1e3
    Mg = v.max() - np.median(v)
    t_co = np.median(v)
    solo = []
    for f in sorted(glob.glob(f"{LAB}/results/boundary_stats/q15_g1173_r*.csv"))[:5]:
        solo.append(np.array([float(x["dur_ns"]) for x in csv.DictReader(open(f))]) / 1e3)
    gam = (np.median(np.concatenate([r[1500:2000] for r in solo])) /
           np.median(np.concatenate([r[:500] for r in solo])) - 1) / 1500
    Dstar = (t_co*(1+gam*2000)*R + Mg) / 1000.0
    print(f"{Mg:.1f} {int(np.ceil(Dstar))} {Dstar:.2f}")

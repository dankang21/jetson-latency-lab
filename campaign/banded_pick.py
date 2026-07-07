#!/usr/bin/env python3
"""Dispersion-banded guarded pick (vision, D=12ms).
Band h=400us = 2 x s.d. of the round-1 probe margins (calibrated on the
round-1 probe population after observing its two failures; FIXED before
this held-out validation round). Admit clock f only if feasibility slack
>= h; otherwise promote one bin."""
import csv, glob, json, re, sys
import numpy as np
LAB = "/home/dk/mobile/jetson-latency-lab"
H = 400.0  # us, fixed pre-validation
probe_csv = sys.argv[1]
v = np.array([float(x["response_us"]) for x in csv.DictReader(open(probe_csv)) if x.get("response_us")])
Mg = v.max() - np.median(v)
pts = {int(re.search(r"gpu(\d+)", f).group(1)): json.load(open(f))["response_us"]["p50"]
       for f in glob.glob(f"{LAB}/results/nx_validation/nx_est_emc2133_gpu*_mobilenet.json")}
lo, hi = min(pts), max(pts)
k = (pts[lo]-pts[hi])/(1e3/lo-1e3/hi); b = pts[hi]-k*(1e3/hi)
R = 1.035
pick = 0
for g in [510,612,714,816,918,1020,1122,1173]:
    mstar = (12000 - (k*1e3/g+b)*R) / (1173/g)   # 이 클럭의 Mg 허용 경계
    if Mg <= mstar - H:                           # 슬랙 >= h 요구
        pick = g; break
print(f"{Mg:.0f} {pick}")

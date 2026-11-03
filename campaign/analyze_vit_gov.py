#!/usr/bin/env python3
# Deployed-governor generalization for ViT-Small on the NX.
# Reads the estimator sweep (nx_est_emc{2133,3199}_gpu*_vit) + 3-repeat
# deployment (govrep_vit/g{f}_r{1,2,3}) and computes blind(fit@3199) vs
# aware(2-cell@2133) miss rates + energy across the tight-deadline regime.
# Run with /usr/bin/python3 after copying results/govrep_vit from the NX.
import json, csv, re, os, glob
import numpy as np
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # results/... paths are repo-relative

EST = "results/nx_validation"
def p50us(j):
    if not os.path.exists(j): return None
    d = json.load(open(j)); return d["response_us"]["p50"] if "response_us" in d else None
def rc(f):
    return np.array([float(r["response_us"]) for r in csv.DictReader(open(f)) if r.get("response_us")]) if os.path.exists(f) else None
def mv(t):
    if not os.path.exists(t): return None
    vs = [int(m.group(1)) for ln in open(t, errors="ignore") for m in [re.search(r"VDD_IN (\d+)mW", ln)] if m]
    return np.mean(vs) if vs else None
def fit(emc):
    pts = {}
    for f in glob.glob(f"{EST}/nx_est_emc{emc}_gpu*_vit.json"):
        g = int(re.search(r'gpu(\d+)', f).group(1)); v = p50us(f)
        if v: pts[g] = v
    fs = np.array(sorted(pts), float); ts = np.array([pts[k] for k in sorted(pts)], float)
    A = np.vstack([1/fs, np.ones_like(fs)]).T
    (k, b), *_ = np.linalg.lstsq(A, ts, rcond=None); return k, b, pts

k3, b3, p3 = fit(3199); k2, b2, p2 = fit(2133)
DEPLOY = [612, 714, 816, 918, 1020, 1122, 1173]
G = sorted(f for f in DEPLOY if os.path.exists(f"results/govrep_vit/g{f}_r1.csv"))
r = 1.035
pick = lambda k, b, D: next((f for f in G if (k/f+b)*r <= D), G[-1])
reps = lambda fr: [rc(f"results/govrep_vit/g{fr}_r{i}.csv") for i in (1, 2, 3)]
def vmean(fr):
    vs = [mv(f"results/govrep_vit/g{fr}_r{i}.tegra") for i in (1, 2, 3)]; vs = [v for v in vs if v]
    return np.mean(vs) if vs else None

print("ViT sweep p50 (ms) @2133:", {f: round(p2[f]/1000, 2) for f in sorted(p2)})
print("ViT sweep p50 (ms) @3199:", {f: round(p3[f]/1000, 2) for f in sorted(p3)})
print(f"fit@3199 k={k3:.0f} b={b3:.0f} | fit@2133 k={k2:.0f} b={b2:.0f} | deploy freqs={G}")
print("\nD(ms)  blind_f  blind_miss%(3rep mean±sd)   aware_f  aware_miss%   dE(mJ aware-blind)")
floor = min(np.median(np.concatenate([x for x in reps(G[0]) if x is not None])) for _ in [0]) / 1000
lo = round(min(p2.values())/1000 + 0.1, 1)
for Dms in np.arange(lo, lo+2.6, 0.2):
    D = Dms*1000; fb = pick(k3, b3, D); fa = pick(k2, b2, D)
    mb = [np.mean(x > D)*100 for x in reps(fb) if x is not None]
    ma = [np.mean(x > D)*100 for x in reps(fa) if x is not None]
    eb = vmean(fb)/1000*np.median(np.concatenate([x for x in reps(fb) if x is not None]))/1000
    ea = vmean(fa)/1000*np.median(np.concatenate([x for x in reps(fa) if x is not None]))/1000
    print(f"{Dms:4.1f}  {fb:5d}   {np.mean(mb):5.1f} ± {np.std(mb):3.1f}            {fa:5d}    {np.mean(ma):5.1f}        {ea-eb:+.2f}")

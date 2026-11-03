#!/usr/bin/env python3
# Qwen deployed-governor + calibrated energy on the NX. Token decode is
# saturated; miss@D = P(per-token decode > D); E/token = mean active power x
# p50 decode. blind=fit@3199, aware=fit@2133. Run with /usr/bin/python3.
import json, csv, os, re, glob, statistics as st
import numpy as np
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # results/... paths are repo-relative
Q = "results/qwen_deploy"
def p50_decode(tag):  # ms/token from .json compute_us p50
    f = f"{Q}/{tag}.json"
    return json.load(open(f))["compute_us"]["p50"]/1000.0 if os.path.exists(f) else None
def tok_ms(tag):       # per-token decode times (ms) from .csv dur_ns
    f = f"{Q}/{tag}.csv"
    if not os.path.exists(f): return None
    return np.array([int(r["dur_ns"])/1e6 for r in csv.DictReader(open(f)) if r.get("dur_ns")])
def meanpow(tag):
    f = f"{Q}/{tag}.power.csv"
    if not os.path.exists(f): return None
    p = [float(r["vdd_in_mw"]) for r in csv.DictReader(open(f)) if r.get("vdd_in_mw")]
    n=len(p); p=p[n//10:-n//10] if n>20 else p
    return st.mean(p) if p else None
def reps(prefix):
    return sorted(set(re.search(r'_r(\d+)$', os.path.basename(x)[:-4]).group(1)
                  for x in glob.glob(f"{Q}/{prefix}_r*.csv") if '.power.' not in x))
def fit(emc):
    pts={}
    for f in glob.glob(f"{Q}/qsweep_emc{emc}_gpu*.json"):
        g=int(re.search(r'gpu(\d+)',f).group(1)); v=p50_decode(os.path.basename(f)[:-5])
        if v: pts[g]=v
    fs=np.array(sorted(pts),float); ts=np.array([pts[k] for k in sorted(pts)],float)
    A=np.vstack([1/fs,np.ones_like(fs)]).T;(k,b),*_=np.linalg.lstsq(A,ts,rcond=None);return k,b,pts
k3,b3,p3=fit(3199); k2,b2,p2=fit(2133)
DEP=[714,816,918,1020,1122,1173]; G=[f for f in DEP if os.path.exists(f"{Q}/qdep_g{f}_r1.csv")]
r=1.035
pick=lambda k,b,D: next((f for f in G if (k/f+b)*r<=D),G[-1])
print("Qwen sweep p50 ms/token @2133:", {f:round(p2[f],1) for f in sorted(p2)})
print("Qwen sweep p50 ms/token @3199:", {f:round(p3[f],1) for f in sorted(p3)})
print(f"deploy freqs={G}")
def dep_miss(freq,D):
    ms=[np.mean(x>D)*100 for r_ in reps(f"qdep_g{freq}") for x in [tok_ms(f"qdep_g{freq}_r{r_}")] if x is not None]
    return ms
def dep_energy(freq):
    es=[meanpow(f"qdep_g{freq}_r{r_}")*p50_decode(f"qdep_g{freq}_r{r_}")/1000.0
        for r_ in reps(f"qdep_g{freq}") if meanpow(f"qdep_g{freq}_r{r_}") and p50_decode(f"qdep_g{freq}_r{r_}")]
    return (st.mean(es),st.pstdev(es)) if es else None
print("\nD(ms) blind_f blind_miss%(3rep)   aware_f aware_miss%   dE(mJ/token aware-blind)")
lo=round(min(p2.values())+1.0)
for D in range(int(lo), int(lo)+13, 2):
    fb=pick(k3,b3,D); fa=pick(k2,b2,D)
    mb=dep_miss(fb,D); ma=dep_miss(fa,D)
    eb=dep_energy(fb); ea=dep_energy(fa)
    if mb and ma and eb and ea:
        print(f"{D:4d}  {fb:5d}  {np.mean(mb):5.1f}±{np.std(mb):4.1f}        {fa:5d}   {np.mean(ma):5.1f}      {ea[0]-eb[0]:+.1f}  (aware {ea[0]:.0f}mJ)")

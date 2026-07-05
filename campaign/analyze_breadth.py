#!/usr/bin/env python3
# Deployed-governor analysis for the breadth models (ResNet-50, YOLOv8s, Qwen-3B)
# on the NX. Mirrors analyze_vit_gov.py / analyze_qwen_deploy.py: blind = k/f+b
# fit at EMC 3199, aware = refit at the 2133 deployment point; pick = lowest f
# with T*1.035 <= D; miss measured over 3x1000-cycle deployments; energy =
# calibrated 1 ms-sampled VDD_IN (power.csv) / release rate.
import json, csv, re, os, glob, statistics as st
import numpy as np

EST = "results/nx_validation"

def p50us(j, key="response_us"):
    if not os.path.exists(j): return None
    d = json.load(open(j))
    return d[key]["p50"] if key in d else None

def rc(f):
    return np.array([float(r["response_us"]) for r in csv.DictReader(open(f))
                     if r.get("response_us")]) if os.path.exists(f) else None

def meanpow(f):
    if not os.path.exists(f): return None
    p = [float(r["vdd_in_mw"]) for r in csv.DictReader(open(f)) if r.get("vdd_in_mw")]
    if len(p) < 20: return None
    n = len(p); return st.mean(p[n//10:-n//10])

def fit(pts):
    fs = np.array(sorted(pts), float); ts = np.array([pts[k] for k in sorted(pts)], float)
    A = np.vstack([1/fs, np.ones_like(fs)]).T
    (k, b), *_ = np.linalg.lstsq(A, ts, rcond=None); return k, b

def vision(name, rate, dgrid):
    sw = {}
    for emc in (2133, 3199):
        pts = {}
        for f in glob.glob(f"{EST}/nx_est_emc{emc}_gpu*_{name}.json"):
            g = int(re.search(r"gpu(\d+)", f).group(1)); v = p50us(f)
            if v: pts[g] = v
        sw[emc] = pts
    k3, b3 = fit(sw[3199]); k2, b2 = fit(sw[2133])
    G = sorted(f for f in [612,714,816,918,1020,1122,1173]
               if os.path.exists(f"results/govrep_{name}/g{f}_r1.csv"))
    r = 1.035
    pick = lambda k, b, D: next((f for f in G if (k/f+b)*r <= D), G[-1])
    reps = lambda fr: [rc(f"results/govrep_{name}/g{fr}_r{i}.csv") for i in (1,2,3)]
    def cal_e(fr):
        es = [meanpow(f"results/govrep_{name}/g{fr}_r{i}.power.csv") for i in (1,2,3)]
        es = [e/rate for e in es if e]
        return (st.mean(es), (max(es)-min(es)) if len(es) > 1 else 0) if es else (None, None)
    print(f"\n===== {name} (NX, rate {rate} Hz) =====")
    print(f"  p50@2133: {{ {', '.join(f'{f}:{sw[2133][f]/1000:.2f}ms' for f in sorted(sw[2133]))} }}")
    print(f"  p50@3199: {{ {', '.join(f'{f}:{sw[3199][f]/1000:.2f}ms' for f in sorted(sw[3199]))} }}")
    print("  D(ms)  blind_f miss%(mean±sd)   aware_f miss%   E_blind    E_aware    dE(mJ/inf)")
    for Dms in dgrid:
        D = Dms*1000; fb = pick(k3,b3,D); fa = pick(k2,b2,D)
        if fb == fa: continue
        mb = [np.mean(x>D)*100 for x in reps(fb) if x is not None]
        ma = [np.mean(x>D)*100 for x in reps(fa) if x is not None]
        eb, _ = cal_e(fb); ea, _ = cal_e(fa)
        de = f"{ea-eb:+7.1f}" if (ea and eb) else "    n/a"
        print(f"  {Dms:5.1f}  {fb:5d}  {np.mean(mb):5.1f} ± {np.std(mb):4.1f}     "
              f"{fa:5d}  {np.mean(ma):5.2f}   {eb or 0:7.1f}   {ea or 0:7.1f}   {de}")

def qwen3b():
    D3 = "results/qwen3b_deploy"
    sw = {}
    for emc in (2133, 3199):
        pts = {}
        for f in glob.glob(f"{D3}/qsweep_emc{emc}_gpu*.json"):
            g = int(re.search(r"gpu(\d+)", f).group(1))
            v = p50us(f, "compute_us")
            if v: pts[g] = v
        sw[emc] = pts
    k3, b3 = fit(sw[3199]); k2, b2 = fit(sw[2133])
    G = sorted(int(re.search(r"g(\d+)_r1", f).group(1))
               for f in glob.glob(f"{D3}/q3dep_g*_r1.json"))
    r = 1.035
    pick = lambda k, b, D: next((f for f in G if (k/f+b)*r <= D), G[-1])
    def toks(fr):
        out = []
        for i in (1,2,3):
            f = f"{D3}/q3dep_g{fr}_r{i}.csv"
            if os.path.exists(f):
                out.append(np.array([int(x["dur_ns"])/1e6 for x in csv.DictReader(open(f))]))
        return out
    def e_tok(fr):
        es = []
        for i in (1,2,3):
            mp = meanpow(f"{D3}/q3dep_g{fr}_r{i}.power.csv")
            c = p50us(f"{D3}/q3dep_g{fr}_r{i}.json", "compute_us")
            if mp and c: es.append(mp*c/1e6)  # mW * ms /1000 = mJ
        return st.mean(es) if es else None
    print("\n===== Qwen2.5-3B decode (NX, saturated) =====")
    print(f"  ms/token p50@2133: {{ {', '.join(f'{f}:{sw[2133][f]/1000:.1f}' for f in sorted(sw[2133]))} }}")
    print(f"  ms/token p50@3199: {{ {', '.join(f'{f}:{sw[3199][f]/1000:.1f}' for f in sorted(sw[3199]))} }}")
    lo = min(sw[2133].values())/1000
    print(f"  floor@2133 = {lo:.1f} ms/token")
    print("  D(ms/tok)  blind_f miss%   aware_f miss%   E_blind   E_aware  dE(mJ/tok)")
    for Dms in [round(lo+d,1) for d in (0.5, 1.5, 3.0, 5.0)]:
        D = Dms*1000; fb = pick(k3,b3,D); fa = pick(k2,b2,D)
        if fb == fa: continue
        mb = [np.mean(t>Dms)*100 for t in toks(fb)]
        ma = [np.mean(t>Dms)*100 for t in toks(fa)]
        eb, ea = e_tok(fb), e_tok(fa)
        de = f"{ea-eb:+7.1f}" if (ea and eb) else "n/a"
        print(f"  {Dms:7.1f}   {fb:5d} {np.mean(mb):6.1f}   {fa:5d} {np.mean(ma):6.2f}   "
              f"{eb or 0:7.1f}  {ea or 0:7.1f}  {de}")

vision("resnet50", 10, np.arange(8.0, 16.1, 0.5))
vision("yolov8s", 5, np.arange(20.0, 45.1, 1.0))
qwen3b()

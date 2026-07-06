#!/usr/bin/env python3
"""Score the joint-governor deployments and the 4k-token horizon runs.

Vision+tenancy: map each policy's pick (results/joint_gov/picks.txt) to its
deployed cells and score misses at that policy's deadline; per-run miss rates
for run-to-run spread. Horizon: rescore 2k and 4k traces at (D, H) contracts,
compute the quoted horizon cap H* = ((D / (R * T_short)) - 1) / gamma.
"""
import csv, glob, json, re
import numpy as np

LAB = "/home/dk/mobile/jetson-latency-lab"
R = 1.035

def cells(pat, col="response_us"):
    out = {}
    for f in sorted(glob.glob(pat)):
        g = int(re.search(r"g(\d+)_r", f).group(1))
        v = np.array([float(x[col]) for x in csv.DictReader(open(f)) if x.get(col)])
        out.setdefault(g, []).append(v)
    return out

print("=== vision + tenancy: policy picks vs measured miss (ViT@10Hz co-runner) ===")
dep = cells(f"{LAB}/results/joint_gov/dep_co_g*_r*.csv")
picks = {}
for ln in open(f"{LAB}/results/joint_gov/picks.txt"):
    m = re.match(r"\s*D=(\d+)ms\s+(\S+)\s+->\s+(\d+|INFEASIBLE)", ln)
    if m:
        picks[(int(m.group(1)), m.group(2))] = None if m.group(3) == "INFEASIBLE" else int(m.group(3))
for (D, pol), g in sorted(picks.items()):
    if g is None:
        print(f"  D={D}ms {pol:9s}: INFEASIBLE declared"); continue
    runs = dep.get(g, [])
    per = [np.mean(v > D * 1000) * 100 for v in runs]
    allv = np.concatenate(runs)
    print(f"  D={D}ms {pol:9s} -> {g:4d}: miss {np.mean(allv > D*1000)*100:5.2f}%  "
          f"(runs: {' '.join(f'{p:.1f}' for p in per)})  [{'PASS' if np.mean(allv > D*1000) <= 0.02 else 'VIOLATION'} vs tau=2%]")

print("\n  per-clock reference (all deployed cells, D=9/12):")
for g, runs in sorted(dep.items()):
    allv = np.concatenate(runs)
    print(f"    g{g:4d}: n={len(allv)}  D9 {np.mean(allv>9000)*100:5.2f}%  D12 {np.mean(allv>12000)*100:5.2f}%  "
          f"p50 {np.median(allv)/1e3:.2f}ms  p98 {np.quantile(allv,.98)/1e3:.2f}ms")

print("\n=== horizon: H=2000 vs H=4000 at (D, clock) contracts ===")
h2 = cells(f"{LAB}/results/boundary_stats/q15_g*_r*.csv", col=None)
def dur_cells(pat):
    out = {}
    for f in sorted(glob.glob(pat)):
        g = int(re.search(r"g(\d+)_r", f).group(1))
        v = np.array([float(x["dur_ns"]) for x in csv.DictReader(open(f))]) / 1e3
        out.setdefault(g, []).append(v)
    return out
h2 = dur_cells(f"{LAB}/results/boundary_stats/q15_g*_r*.csv")
h4 = dur_cells(f"{LAB}/results/horizon_4k/q15h4k_g*_r*.csv")

for g in sorted(set(h2) & set(h4)):
    t_short = np.median(np.concatenate([r[:300] for r in h2[g]]))
    drift = (np.median(np.concatenate([r[1500:2000] for r in h2[g]])) /
             np.median(np.concatenate([r[:500] for r in h2[g]])) - 1)
    gamma = drift / 1500
    for D in (28000,):
        acc_blind = t_short * R <= D
        hstar = ((D / (R * t_short)) - 1) / gamma
        for H, data in ((2000, h2[g]), (4000, h4[g])):
            allv = np.concatenate(data)
            last = np.concatenate([r[H - 500:H] for r in data])
            acc_joint = t_short * (1 + gamma * H) * R <= D
            print(f"  g{g} D={D//1000} H={H}: blind={'ACC' if acc_blind else 'REJ'} "
                  f"joint={'ACC' if acc_joint else 'REJ'} (H*={hstar:.0f}) | "
                  f"miss all {np.mean(allv>D)*100:5.2f}%  last500 {np.mean(last>D)*100:5.2f}%")
    # 4k drift check (is drift ~linear to 4000?)
    d4 = (np.median(np.concatenate([r[3500:4000] for r in h4[g]])) /
          np.median(np.concatenate([r[:500] for r in h4[g]])) - 1)
    print(f"    g{g}: measured drift @2k {drift*100:+.2f}% -> @4k {d4*100:+.2f}% "
          f"(linear pred {gamma*3500*100:+.2f}%)")

def hxt():
    """Joint-stress cell: 2,000-token decode under the ViT co-runner.
    Composed rule: T_solo(f)*dil*(1+gamma*H)*R + M <= D, with dil and M
    (p98-p50) from the 200-token under-co probe only; gamma from solo."""
    rng = np.random.default_rng(42)
    probe = np.array([float(x["dur_ns"]) for x in
                      csv.DictReader(open(f"{LAB}/results/joint_hxt/probe_co_g1173.csv"))]) / 1e3
    def dur(pat):
        out = {}
        for f in sorted(glob.glob(pat)):
            g = int(re.search(r"g(\d+)", f).group(1))
            out.setdefault(g, []).append(
                np.array([float(x["dur_ns"]) for x in csv.DictReader(open(f))]) / 1e3)
        return out
    co, solo = dur(f"{LAB}/results/joint_hxt/q15co_g*_r*.csv"), dur(f"{LAB}/results/boundary_stats/q15_g*_r*.csv")
    M = np.quantile(probe, 0.98) - np.median(probe)
    dil = np.median(probe) / np.median(np.concatenate([r[:300] for r in solo[1173]]))
    print(f"\n=== joint-stress (horizon x tenancy): probe M={M/1e3:.2f} ms, level x{dil:.3f} ===")
    for g in sorted(co):
        t = np.median(np.concatenate([r[:300] for r in solo[g]]))
        gam = (np.median(np.concatenate([r[1500:2000] for r in solo[g]])) /
               np.median(np.concatenate([r[:500] for r in solo[g]])) - 1) / 1500
        dstar = t * dil * (1 + gam * 2000) * R + M
        print(f"  g{g}: D* = {dstar/1e3:.2f} ms (composed acceptance threshold)")
        for D in (28000, 43000, 44000):
            per = np.array([np.mean(r > D) * 100 for r in co[g]])
            boot = [rng.choice(per, len(per), replace=True).mean() for _ in range(20000)]
            print(f"    D={D//1000}: {'ACC' if dstar <= D else 'REJ'}  miss {np.mean(np.concatenate(co[g])>D)*100:5.2f}%"
                  f"  (runs {' '.join(f'{p:.2f}' for p in per)}; UB {np.quantile(boot,.95):.2f})")

hxt()

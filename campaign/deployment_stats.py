#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Deployment numbers of the article and its supplement that the other analysis
scripts do not print: the one-hour saturated-decode soak (main text Sec. III), the six
directed profile pairs (Supp. Table S2), the maximum clock's energy premium over the
aware picks and the deployed medians of the blind breadth violations (main text Sec. IV),
and the thermal-order figures of the deployment sessions and of the EMC-free re-run
(Supp. Sec. S2, thermal-gate record and E5-B protocol).

Read-only over results/ (python3 + numpy; nothing is written).  Run from anywhere:
    python3 campaign/deployment_stats.py [--section NAME ...] [--results DIR]
Sections: soak, pairs, maxclock, breadth, thermal (default: all).

Data (results/) and definitions:
  soak       soak_qwen/seg<k>.csv (k = 1..60; 2,000 tokens each, dur_ns), soak_qwen/soak.tegra
             (tegrastats every 1 s), campaign/nx_wave2.sh step C4a: Qwen2.5-1.5B Q4_K_M
             decode, NX, GPU 1020 MHz (the aware pick at D = 29 ms/token), EMC 2133 MHz.  A
             token misses when it takes longer than 29 ms.  The INA3221 log
             soak_qwen/soak.power.csv is on Zenodo (soak_power_1khz.csv.gz in the deposit
             10.5281/zenodo.21236659; gunzip it to that path) and is summarized when present.
  pairs      full-sweep fit: least squares T(f) = k/f + b over the ten-clock GPU sweep at the
             source EMC point (nx_validation/nx_est_emc<E>_gpu<g>_mobilenet.json, response_us
             p50), r = 1.035, pick = lowest clock of {612, ..., 1173} with T(f) r <= D; when
             no clock qualifies the model declines and the top clock (1173 MHz) runs.
             Deployments at the target: govrep/g<f>_r{1,2,3}.csv (EMC 2133 MHz) and
             nx_pairs/e<E>_g<f>_r{1,2,3}.csv (EMC 3199 and 665.6 MHz); miss % pooled over the
             three 1,000-cycle repeats [per repeat].
  maxclock   govrep_rev/{nx,nx_mobilenet,nx_mobilenet_s3}/g<f>_p<k>.power.csv: energy per
             inference = mean VDD_IN over the log less its first and last 10 % of samples
             (slice p[n//10:-n//10]) / release rate; dE = E(1173) - E(f) paired by pair index
             over the three sessions; paired-t 95 % interval; relative to the mean E(f).
  breadth    E1b grids and rate-matched two-cell fits: refit33/nx_<m>/emc{3199,2133}_g{306,1173}_
             r{1,2,3}.json (endpoint p50 = median of the three repeats' response_us p50),
             T(f) = k/f + b through the two endpoints, r = 1.035, grid {510, ..., 1173};
             blind = EMC-3199 endpoints.  Runs: govrep_rev/nx_<m>/g<f>_p<1..10>.csv; a run
             is over budget when more than 2 % of its cycles exceed D.  Deployed median of a
             cell: the median of the ten run medians (and, printed alongside, the median of
             all 10,000 pooled cycles).
  thermal    session.log end-of-cell readings ('t=' in millidegrees) of the deployment
             sessions govrep_rev/<s>/ (cells of one pair run back to back in ascending clock
             order) and of the EMC-free re-run e5b_live/nx/ (seven cells per round in a fixed
             order).  Pair rise = last minus first cell of a pair.  'Blind vs aware picks'
             uses the picks of the deployed two-cell rule (as for 'breadth'; NX MobileNetV2
             with the 33 Hz cells refit33/nx and with the 8 Hz cells nx_validation; Nano
             MobileNetV2 with nano_grid/mobilenet; Nano ViT-Small with refit33/nano_vit) on
             each session's pre-stated deadline grid.
"""
import argparse
import csv
import glob
import json
import os
import re
import sys
from collections import Counter

import numpy as np

LAB = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R_MULT = 1.035
TAU = 2.0
# two-sided 97.5 % Student-t quantiles
T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262,
        10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110,
        18: 2.101, 19: 2.093, 20: 2.086, 21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060,
        26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042}
NX_GRID = [510, 612, 714, 816, 918, 1020, 1122, 1173]
NANO_GRID = [306, 408, 510, 612, 714, 816, 918, 1020]
E1B = {"vit": [14.0, 14.5, 15.0, 15.5, 16.0, 17.0, 17.5, 18.5, 19.0],
       "yolov8s": [29.5, 30.0, 31.0, 31.5, 33.0, 33.5, 35.5, 36.0, 38.5, 39.0],
       "resnet50": [14.0, 14.5, 15.0, 15.5, 16.0, 17.0, 17.5, 18.5, 19.0]}
NX_MN_D = [4.4, 4.6, 4.8, 5.0, 5.2, 5.4, 5.6, 5.8, 6.0]
NANO_MN_D = [8.5, 9.0, 9.5, 10.0]
NANO_VIT_D = [14.0, 14.5, 15.0, 15.5, 16.5, 17.0, 18.0, 18.5, 20.0, 20.5]


def p50(path, key="response_us"):
    return float(json.load(open(path))[key]["p50"])


def resp(path):
    return np.loadtxt(path, delimiter=",", skiprows=1)[:, 3]


def runs(pattern):
    out = [p for p in glob.glob(pattern) if not p.endswith(".power.csv")]
    return sorted(out, key=lambda p: int(re.search(r"_[rp](\d+)\.csv$", p).group(1)))


def fit2(t_lo, t_hi, lo, hi):
    k = (t_lo - t_hi) / (1.0 / lo - 1.0 / hi)
    return k, t_hi - k / hi


def pick(kb, D_us, grid):
    for f in grid:
        if (kb[0] / f + kb[1]) * R_MULT <= D_us:
            return f
    return None


def refit_fits(d, hi):
    """Two-cell fits from three-repeat endpoint cells (median p50), or None."""
    out = {}
    for arm, emc in (("blind", 3199), ("aware", 2133)):
        c = {}
        for g in (306, hi):
            fs = [f"{d}/emc{emc}_g{g}_r{k}.json" for k in (1, 2, 3)]
            if not all(os.path.isfile(f) for f in fs):
                return None
            c[g] = float(np.median([p50(f) for f in fs]))
        out[arm] = fit2(c[306], c[hi], 306, hi)
    return out


def cell_fits(path_of, hi):
    out = {}
    for arm, emc in (("blind", 3199), ("aware", 2133)):
        a, b = path_of(emc, 306), path_of(emc, hi)
        if not (os.path.isfile(a) and os.path.isfile(b)):
            return None
        out[arm] = fit2(p50(a), p50(b), 306, hi)
    return out


def trimmed_power(path):
    v = np.array([float(r["vdd_in_mw"]) for r in csv.DictReader(open(path))])
    n = len(v)
    return float(v[n // 10:-n // 10].mean())


# --------------------------------------------------------------------------- soak
def sec_soak(R):
    print("== [main Sec. III] one-hour saturated-decode soak (NX, Qwen2.5-1.5B, GPU 1020 MHz, "
          "EMC 2133 MHz, 60 x 2,000 tokens), D = 29 ms/token ==")
    segs = []
    for k in range(1, 61):
        p = f"{R}/soak_qwen/seg{k}.csv"
        if os.path.isfile(p):
            segs.append(np.array([float(r["dur_ns"]) for r in csv.DictReader(open(p))]) / 1e6)
    if not segs:
        print("  no soak_qwen/seg*.csv\n")
        return
    miss = [np.mean(s > 29.0) * 100 for s in segs]
    cnt = Counter(int(np.count_nonzero(s > 29.0)) for s in segs)
    med = [float(np.median(s)) for s in segs]
    print(f"  windows: {len(segs)} x {len(segs[0])} tokens; per-window miss {min(miss):.2f}-{max(miss):.2f} % "
          f"(missed tokens per window: " + ", ".join(f"{c} in {n}" for c, n in sorted(cnt.items()))
          + f"); pooled {np.mean(np.concatenate(segs) > 29.0) * 100:.3f} %")
    print(f"  per-window median token time: first {med[0]:.3f}, last {med[-1]:.3f}, range {min(med):.3f}-"
          f"{max(med):.3f} ms")
    t = f"{R}/soak_qwen/soak.tegra"
    if os.path.isfile(t):
        T, load, gr, emc, cpu = [], [], [], [], []
        for ln in open(t):
            m = re.search(r"gpu@([\d.]+)C", ln)
            if not m:
                continue
            T.append(float(m.group(1)))
            g = re.search(r"GR3D_FREQ (\d+)%@\[?(\d+)", ln)
            load.append(int(g.group(1)))
            gr.append(int(g.group(2)))
            emc.append(int(re.search(r"EMC_FREQ \d+%@(\d+)", ln).group(1)))
            cpu += [int(x) for x in re.findall(r"@(\d+)", re.search(r"CPU \[([^\]]*)\]", ln).group(1))]
        T = np.array(T)
        act = np.flatnonzero(np.array(load) > 50)
        Ta = T[act[0]:act[-1] + 1]
        print(f"  tegrastats: {len(T)} samples; gpu zone first {T[0]:.2f}, last {T[-1]:.2f}, min {T.min():.2f}, "
              f"max {T.max():.2f} C (max - min {T.max() - T.min():.2f}); while decoding (GR3D load > 50 %) "
              f"{Ta.min():.2f}-{Ta.max():.2f} C ({Ta.max() - Ta.min():.2f})")
        print(f"  clocks in every sample: GR3D {sorted(set(gr))} MHz, EMC {sorted(set(emc))} MHz, "
              f"CPU {sorted(set(cpu))} MHz")
    pw = f"{R}/soak_qwen/soak.power.csv"
    if os.path.isfile(pw):
        a = np.loadtxt(pw, delimiter=",", skiprows=1, usecols=(0, 1))
        n = len(a)
        print(f"  power log: {n} samples over {a[-1, 0] - a[0, 0]:.1f} s, median interval "
              f"{np.median(np.diff(a[:, 0])) * 1e3:.2f} ms, trimmed mean VDD_IN {a[n // 10:-n // 10, 1].mean():.0f} mW")
    else:
        print("  power log soak_qwen/soak.power.csv not present: it is soak_power_1khz.csv.gz in the Zenodo "
              "deposit 10.5281/zenodo.21236659 (gunzip it to that path)")
    print()


# --------------------------------------------------------------------------- pairs
def sweep_fit(R, emc):
    pts, hz = {}, set()
    for f in glob.glob(f"{R}/nx_validation/nx_est_emc{emc}_gpu*_mobilenet.json"):
        g = int(re.search(r"gpu(\d+)_", f).group(1))
        pts[g] = p50(f)
        hz.add(json.load(open(f))["meta"]["hz"])
    fs = np.array(sorted(pts), float)
    A = np.vstack([1.0 / fs, np.ones_like(fs)]).T
    k, b = np.linalg.lstsq(A, np.array([pts[int(x)] for x in fs]), rcond=None)[0]
    return (float(k), float(b)), len(pts), sorted(hz)


def deployed(R, emc, f):
    if emc == 2133:
        return [resp(p) for p in runs(f"{R}/govrep/g{f}_r*.csv")]
    return [resp(p) for p in runs(f"{R}/nx_pairs/e{emc}_g{f}_r*.csv")]


def sec_pairs(R):
    print("== [Supp. Table S2] directed profile pairs, MobileNetV2 on the NX: full-sweep fit at the "
          "source EMC, deployed at the target EMC ==")
    grid = [612, 714, 816, 918, 1020, 1122, 1173]
    fits = {}
    for e in (3199, 2133, 665):
        fits[e] = sweep_fit(R, e)
        kb, n, hz = fits[e]
        print(f"  fit at {'665.6' if e == 665 else e} MHz: k = {kb[0]:.0f} us*MHz, b = {kb[1]:.0f} us "
              f"({n} cells, {'/'.join(f'{h:g}' for h in hz)} Hz); T(1173) r = {(kb[0] / 1173 + kb[1]) * R_MULT / 1e3:.3f} ms")
    for src, tgt, D in ((3199, 2133, 5.0), (2133, 665, 8.6), (3199, 665, 8.6), (2133, 3199, 4.5),
                        (665, 3199, 4.5), (665, 2133, 5.0)):
        out = []
        for arm, e in (("blind", src), ("aware", tgt)):
            f = pick(fits[e][0], D * 1e3, grid)
            run_f = f if f is not None else 1173
            rs = deployed(R, tgt, run_f)
            if not rs:
                out.append(f"{arm} {run_f}: no runs")
                continue
            m = np.mean(np.concatenate(rs) > D * 1e3) * 100
            label = f"{f}" if f is not None else "declines (1173 runs)"
            out.append(f"{arm} {label}: {m:.2f} % [" + " ".join(f"{np.mean(v > D * 1e3) * 100:.1f}" for v in rs) + "]")
        lab = lambda e: "665.6" if e == 665 else str(e)
        print(f"  {lab(src):>5s} -> {lab(tgt):5s} D = {D:3.1f} ms: " + "; ".join(out))
    print()


# --------------------------------------------------------------------------- maxclock
def sec_maxclock(R):
    print("== [main Sec. IV] energy of the maximum clock (1173 MHz) against the aware picks, "
          "NX MobileNetV2, three interleaved sessions, paired by pair index ==")
    sess = ("nx", "nx_mobilenet", "nx_mobilenet_s3")
    E = {}
    for s in sess:
        for p in glob.glob(f"{R}/govrep_rev/{s}/g*_p*.power.csv"):
            m = re.search(r"/g(\d+)_p(\d+)\.power\.csv$", p)
            hz = json.load(open(p[:-10] + ".json"))["meta"]["hz"]
            E[(s, int(m.group(1)), int(m.group(2)))] = trimmed_power(p) / hz
    if not E:
        print("  no power logs\n")
        return
    for f in (816, 918, 1020, 1122):
        d, base, per_s = [], [], {}
        for (s, g, k), e in E.items():
            if g == 1173 and (s, f, k) in E:
                d.append(e - E[(s, f, k)])
                base.append(E[(s, f, k)])
                per_s.setdefault(s, []).append(e - E[(s, f, k)])
        if len(d) < 2:
            continue
        d = np.array(d)
        h = T975.get(len(d) - 1, 1.96) * d.std(ddof=1) / np.sqrt(len(d))
        print(f"  1173 - {f:4d}: n = {len(d)}, mean {d.mean():+.3f} mJ/inf [{d.mean() - h:+.3f}, {d.mean() + h:+.3f}], "
              f"positive {int(np.sum(d > 0))}/{len(d)}, {d.mean() / np.mean(base) * 100:+.2f} % of E({f}) "
              f"= {np.mean(base):.1f} mJ; session means " + ", ".join(f"{np.mean(v):+.2f}" for v in per_s.values()))
    print()


# --------------------------------------------------------------------------- breadth
def sec_breadth(R):
    print("== [main Sec. IV] blind picks over budget on the NX breadth grids: deployed median "
          "against the deadline ==")
    tot, viol = 0, []
    for m, Ds in E1B.items():
        fits = refit_fits(f"{R}/refit33/nx_{m}", 1173)
        if fits is None:
            print(f"  {m}: missing refit33 cells")
            continue
        rs = {}
        for p in runs(f"{R}/govrep_rev/nx_{m}/g*_p*.csv"):
            rs.setdefault(int(re.search(r"/g(\d+)_p", p).group(1)), []).append(resp(p))
        for D in Ds:
            tot += 1
            f = pick(fits["blind"], D * 1e3, NX_GRID)
            if f is None or f not in rs:
                continue
            per = [np.mean(v > D * 1e3) * 100 for v in rs[f]]
            over = sum(x > TAU for x in per)
            if over == 0:
                continue
            mm = float(np.median([np.median(v) for v in rs[f]])) / 1e3
            pm = float(np.median(np.concatenate(rs[f]))) / 1e3
            viol.append((m, D, f, over, len(per), mm, pm))
            print(f"  {m:9s} D = {D:4.1f} ms, blind {f}: runs over tau {over}/{len(per)}; median of run medians "
                  f"{mm:.3f} ms ({(mm - D) * 1e3:+.0f} us); pooled median {pm:.3f} ms ({(pm - D) * 1e3:+.0f} us)")
    every = [v for v in viol if v[3] == v[4]]
    print(f"  blind over budget in {len(viol)} of {tot} cells, {len(every)} of them in every run")
    for i, name in ((5, "median of run medians"), (6, "pooled median")):
        above = [v for v in every if v[i] > v[1]]
        below = [v for v in every if v[i] <= v[1]]
        print(f"  {name}: {len(above)} above D, {len(below)} at or below ("
              + ", ".join(f"{v[0]} {v[1]}: {(v[1] - v[i]) * 1e3:.0f} us" for v in below) + ")")
    print()


# --------------------------------------------------------------------------- thermal
def end_temps(log):
    T = {}
    if not os.path.isfile(log):
        return T
    for ln in open(log):
        m = re.match(r"^\s*(.+?): (?:OK|FAIL) t=(\d+)\s*$", ln)
        if not m:
            continue
        key = m.group(1).strip()
        mm = re.match(r"p(\d+) g(\d+)$", key)
        if mm:
            key = f"g{mm.group(2)}_p{mm.group(1)}"
        T[key] = int(m.group(2)) / 1000.0
    return T


def pairs_of(T):
    P = {}
    for k, t in T.items():
        m = re.match(r"g(\d+)_p(\d+)$", k)
        if m:
            P.setdefault(int(m.group(2)), {})[int(m.group(1))] = t
    return P


def sec_thermal(R):
    print("== [Supp. Sec. S2, thermal-gate record] end-of-cell readings within the interleaved pairs "
          "of the deployment sessions ==")
    nx_mn = {"33 Hz cells": refit_fits(f"{R}/refit33/nx", 1173),
             "8 Hz cells": cell_fits(lambda e, g: f"{R}/nx_validation/nx_est_emc{e}_gpu{g}_mobilenet.json", 1173)}
    nano_mn = {"50 Hz cells": cell_fits(lambda e, g: f"{R}/nano_grid/mobilenet/est_emc{e}_gpu{g}.json", 1020)}
    plan = [("nx", nx_mn, NX_MN_D, NX_GRID), ("nx_mobilenet", nx_mn, NX_MN_D, NX_GRID),
            ("nx_mobilenet_s3", nx_mn, NX_MN_D, NX_GRID)]
    plan += [(f"nx_{m}", {"33 Hz cells": refit_fits(f"{R}/refit33/nx_{m}", 1173)}, E1B[m], NX_GRID)
             for m in ("vit", "yolov8s", "resnet50")]
    plan += [("nano", nano_mn, NANO_MN_D, NANO_GRID), ("nano_mobilenet_s2", nano_mn, NANO_MN_D, NANO_GRID),
             ("nano_vit", {"rate-matched cells": refit_fits(f"{R}/refit33/nano_vit", 1020)}, NANO_VIT_D, NANO_GRID)]
    means, top, near, pickd = [], (0, None), (0, None), (0, None)
    for s, fitsets, Ds, grid in plan:
        P = pairs_of(end_temps(f"{R}/govrep_rev/{s}/session.log"))
        if not P:
            print(f"  {s}: no session.log readings")
            continue
        clocks = sorted({g for v in P.values() for g in v})
        rise = [v[max(v)] - v[min(v)] for v in P.values() if len(v) > 1]
        means.append((s, float(np.mean(rise))))
        mx = max(rise)
        if mx > top[0]:
            top = (mx, s)
        for v in P.values():
            for i, a in enumerate(clocks):
                for b in clocks[i + 1:i + 3]:
                    if a in v and b in v and v[b] - v[a] > near[0]:
                        near = (v[b] - v[a], f"{s} {a}->{b}")
        best = (0, None)
        for name, kb in fitsets.items():
            if kb is None:
                continue
            for D in Ds:
                fb, fa = pick(kb["blind"], D * 1e3, grid), pick(kb["aware"], D * 1e3, grid)
                if fb is None or fa is None or fb == fa or fb not in clocks or fa not in clocks:
                    continue
                for k, v in P.items():
                    if fa in v and fb in v and abs(v[fa] - v[fb]) > best[0]:
                        best = (abs(v[fa] - v[fb]), f"{s} ({name}) D = {D} ms, blind {fb} / aware {fa}, pair {k}")
        if best[0] > pickd[0]:
            pickd = best
        print(f"  {s:18s} pairs {len(P):2d}, clocks {clocks}: mean pair rise {np.mean(rise):.2f} C, max {mx:.2f} C; "
              f"largest |aware - blind| at the picks {best[0]:.2f} C")
    if means:
        print(f"  mean pair rise across sessions {min(m for _, m in means):.2f}-{max(m for _, m in means):.2f} C; "
              f"largest single-pair rise {top[0]:.2f} C ({top[1]}); largest rise between clocks one or two bins "
              f"apart {near[0]:.2f} C ({near[1]}); largest between the blind and aware picks {pickd[0]:.2f} C "
              f"({pickd[1]})")
    print("== [Supp. Sec. S2, EMC-free re-run protocol] e5b_live/nx/session.log ==")
    T = end_temps(f"{R}/e5b_live/nx/session.log")
    rounds = {}
    for k, t in T.items():
        m = re.match(r"e[\d.]+_g\d+_r(\d+)$", k)
        if m:
            rounds.setdefault(int(m.group(1)), []).append(t)
    if not rounds:
        print("  no readings\n")
        return
    late = [t for r, v in rounds.items() if r >= 2 for t in v]
    allv = [t for v in rounds.values() for t in v]
    grad = [v[-1] - v[0] for r, v in sorted(rounds.items()) if r >= 2]
    print(f"  {len(allv)} readings in {len(rounds)} rounds: all {min(allv):.1f}-{max(allv):.1f} C; rounds 2-10 "
          f"{min(late):.1f}-{max(late):.1f} C; in-round gradient (last - first cell), rounds 2-10: mean "
          f"{np.mean(grad):.2f} C, range {min(grad):.2f}-{max(grad):.2f} C")
    print()


SECTIONS = {"soak": sec_soak, "pairs": sec_pairs, "maxclock": sec_maxclock, "breadth": sec_breadth,
            "thermal": sec_thermal}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--section", action="append", choices=sorted(SECTIONS),
                    help="print only this section (repeatable)")
    ap.add_argument("--results", default=os.path.join(LAB, "results"),
                    help="results directory (default: the checkout's results/)")
    a = ap.parse_args()
    R = os.path.abspath(a.results)
    if not os.path.isdir(R):
        sys.exit(f"no results directory at {R}")
    for s in a.section or list(SECTIONS):
        SECTIONS[s](R)


if __name__ == "__main__":
    main()

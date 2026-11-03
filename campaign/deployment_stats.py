#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Deployment numbers of the article and its supplement that the other analysis
scripts do not print: the one-hour saturated-decode soak (main text Sec. III), the six
directed profile pairs (Supp. Table S2), the maximum clock's energy premium over the
aware picks, the deployed medians of the blind breadth violations and the tail state of
the June NX session against the three later sessions (main text Sec. IV), and the
thermal-order figures of the deployment sessions and of the EMC-free re-run (Supp. Sec.
S2, thermal-gate record and E5-B protocol).

Read-only over results/ (python3 + numpy; nothing is written).  Run from anywhere:
    python3 campaign/deployment_stats.py [--section NAME ...] [--results DIR]
Sections: soak, pairs, maxclock, breadth, june, thermal (default: all).

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
  june       govrep/g<f>_r{1,2,3}.csv: the June session (2026-06-17) of NX MobileNetV2 at
             33 Hz and EMC 2133 MHz, three blocked 1,000-cycle runs per GPU clock 510-1173
             MHz; later sessions govrep_rev/{nx,nx_mobilenet,nx_mobilenet_s3} (the
             interleaved sessions of Table II, same cells).  Slow mode of a run: its cycles
             whose response_us exceeds the run's p50 by more than 80 us (in the sparse valley
             between the two modes; the shares under 60, 100 and 120 us cuts are printed
             too); share = their fraction of the run, offset = their median minus the run's
             p50.  'Same p50': per clock, the median of the June run p50s minus the median
             of the later runs' p50s.  Picks: the full-sweep fits of 'pairs' (blind at EMC
             3199, aware at EMC 2133 MHz), r = 1.035, grid {510, ..., 1173}; when no clock
             qualifies the model declines and the top clock (1173 MHz) runs as the fallback.
             Miss % per run = share of cycles with response_us > D; the mean over the three
             runs equals the pooled miss (1,000 cycles each); budget tau = 2 % per run.
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


# --------------------------------------------------------------------------- june
LATER = ("nx", "nx_mobilenet", "nx_mobilenet_s3")
SLOW_CUT = 80.0
JUNE_D = (4.6, 4.8, 5.0)


def slow_mode(x, cut=SLOW_CUT):
    """Share (%) of the cycles more than `cut` us above the run's p50; their median minus the p50 (us)."""
    m = float(np.median(x))
    s = x[x > m + cut]
    return 100.0 * len(s) / len(x), (float(np.median(s)) - m) if len(s) else float("nan")


def mode_band(x):
    """Share (%) of the cycles 120-200 us above the run's p50, where the June slow mode sits."""
    m = float(np.median(x))
    return 100.0 * float(np.mean((x > m + 120) & (x <= m + 200)))


def run_dates(paths):
    out = set()
    for p in paths:
        try:
            out.add(json.load(open(p[:-4] + ".json"))["meta"]["timestamp"][:10])
        except (OSError, KeyError, ValueError):
            pass
    return "..".join(sorted(out)[::max(1, len(out) - 1)]) if out else "date not recorded"


def rng(v, fmt="{:.1f}"):
    return (fmt + "-" + fmt).format(min(v), max(v))


def sec_june(R):
    print("== [main Sec. IV] tail state of the June session against the three later sessions "
          "(NX MobileNetV2, 33 Hz, EMC 2133 MHz) ==")
    jp = {f: runs(f"{R}/govrep/g{f}_r*.csv") for f in NX_GRID}
    jp = {f: v for f, v in jp.items() if v}
    if not jp:
        print("  no govrep/g*_r*.csv\n")
        return
    J = {f: [resp(p) for p in v] for f, v in jp.items()}
    allj = [x for rs in J.values() for x in rs]
    print(f"  June session: {len(allj)} runs ({run_dates([p for v in jp.values() for p in v])}), clocks "
          f"{min(J)}-{max(J)} MHz.  Slow mode of a run: its cycles more than {SLOW_CUT:.0f} us above the run's "
          "response-time p50; share = their fraction, offset = their median minus the run's p50")
    for f, rs in J.items():
        so = [slow_mode(x) for x in rs]
        print(f"  {f:4d} MHz: p50 " + " ".join(f"{np.median(x) / 1e3:.3f}" for x in rs)
              + " ms, p90 " + " ".join(f"{np.percentile(x, 90) / 1e3:.3f}" for x in rs)
              + " ms; slow-mode share " + " ".join(f"{s:.1f}" for s, _ in so)
              + " %, offset " + " ".join(f"+{o:.0f}" for _, o in so) + " us")
    so = [slow_mode(x) for x in allj]
    off = [o for _, o in so]
    print(f"  all June runs: slow-mode share {rng([s for s, _ in so])} %, offset {rng(off, '{:.0f}')} us (median "
          f"{np.median(off):.0f}); cycles 120-200 us above the p50 {rng([mode_band(x) for x in allj])} %, p90 - p50 "
          f"{rng([np.percentile(x, 90) - np.median(x) for x in allj], '{:.0f}')} us")
    print("  slow-mode share under other cuts: " + ", ".join(
        f"{c:.0f} us {rng([slow_mode(x, c)[0] for x in allj])} %" for c in (60.0, 100.0, 120.0)))
    later = {}
    for s in LATER:
        ps = {f: runs(f"{R}/govrep_rev/{s}/g{f}_p*.csv") for f in NX_GRID}
        ps = {f: v for f, v in ps.items() if v}
        if not ps:
            print(f"  later session {s}: no runs")
            continue
        L = later[s] = {f: [resp(p) for p in v] for f, v in ps.items()}
        xs = [x for rs in L.values() for x in rs]
        print(f"  later session {s} ({run_dates([p for v in ps.values() for p in v])}), clocks {min(L)}-{max(L)} "
              f"MHz, {len(xs)} runs: slow-mode share {rng([slow_mode(x)[0] for x in xs])} % per run, cycles "
              f"120-200 us above the p50 at most {max(mode_band(x) for x in xs):.1f} %, p90 - p50 "
              f"{rng([np.percentile(x, 90) - np.median(x) for x in xs], '{:.0f}')} us")
        if 1122 in L:
            x = L[1122]
            print(f"    1122 MHz: p50 {np.median([np.median(v) for v in x]) / 1e3:.3f} ms (median of the runs), "
                  f"p90 {rng([np.percentile(v, 90) / 1e3 for v in x], '{:.3f}')} ms, cycles above 4.6 ms "
                  f"{np.mean(np.concatenate(x) > 4600) * 100:.2f} % pooled "
                  f"({rng([np.mean(v > 4600) * 100 for v in x])} per run)")
    if later:
        d = []
        for f in J:
            lp = [np.median(x) for L in later.values() for x in L.get(f, [])]
            d.append(f"{f} {np.median([np.median(x) for x in J[f]]) - np.median(lp):+.0f}" if lp
                     else f"{f} (not run later)")
        print("  same p50: June median run p50 minus the later sessions' median run p50, by clock (us): "
              + ", ".join(d))
    fits = {arm: sweep_fit(R, emc) for arm, emc in (("blind", 3199), ("aware", 2133))}
    if min(fits["blind"][1], fits["aware"][1]) < 2:
        print("  no sweep cells nx_validation/nx_est_emc{3199,2133}_gpu*_mobilenet.json for the full-sweep fits\n")
        return
    (kb, bb), nb, hb = fits["blind"]
    (ka, ba), na, ha = fits["aware"]
    print(f"  full-sweep fits (as 'pairs'), r = {R_MULT}: blind (EMC 3199, {nb} cells, "
          f"{'/'.join(f'{h:g}' for h in hb)} Hz) k = {kb:.0f} us*MHz, b = {bb:.0f} us; aware (EMC 2133, {na} cells, "
          f"{'/'.join(f'{h:g}' for h in ha)} Hz) k = {ka:.0f} us*MHz, b = {ba:.0f} us, "
          f"T(1173) r = {(ka / 1173 + ba) * R_MULT / 1e3:.3f} ms")
    bl, picks = [], {}
    for D in JUNE_D:
        out = []
        for arm in ("blind", "aware"):
            f = pick(fits[arm][0], D * 1e3, NX_GRID)
            run_f = picks[(arm, D)] = f if f is not None else NX_GRID[-1]
            if run_f not in J:
                out.append(f"{arm} {run_f}: no June runs")
                continue
            m = [np.mean(x > D * 1e3) * 100 for x in J[run_f]]
            if arm == "blind":
                bl += m
            lab = f"{f}" if f is not None else f"declines, {run_f} MHz runs as the fallback"
            out.append(f"{arm} {lab}: " + " ".join(f"{v:.1f}" for v in m)
                       + f" % (mean {np.mean(m):.2f} %, runs over tau {sum(v > TAU for v in m)}/{len(m)})")
        print(f"  June, D = {D:.1f} ms: " + "; ".join(out))
    if bl:
        print(f"  June blind picks at {JUNE_D[0]}-{JUNE_D[-1]} ms: {rng(bl)} % per run")
    for D in JUNE_D if later else ():
        f = picks[("blind", D)]
        row = []
        for s, L in later.items():
            if f not in L:
                row.append(f"{s} not run")
                continue
            m = [np.mean(x > D * 1e3) * 100 for x in L[f]]
            row.append(f"{s} {sum(v > TAU for v in m)}/{len(m)} runs over tau (pooled "
                       f"{np.mean(np.concatenate(L[f]) > D * 1e3) * 100:.2f} %)")
        print(f"  the June blind clock at D = {D:.1f} ms, {f} MHz, in the later sessions: " + "; ".join(row))
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
            "june": sec_june, "thermal": sec_thermal}


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

#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Co-tenancy numbers of the article that the other analysis scripts do not print:
the MobileNetV2 victim under the 10 Hz ViT-Small co-runner and the decode token
distribution under the same co-runner (main text Sec. V, Sec. VI, Fig. 4b), the drift
under the co-tenant (Sec. VI, VII), the decode probe's autocorrelation (Sec. VI), the
tenants' GPU duty and the primary's inference-occupied fraction against the June
tegrastats GPU load (Sec. VIII-A, Supp. Sec. S2 'Acceptance, occupancy and probe cost'),
and the miss-indicator autocorrelation of the phase-locked launches (Supp. Sec. S2,
'Phase-locking detail').

Read-only over results/ (python3 + numpy; nothing is written).  Run from anywhere:
    python3 campaign/tenancy_stats.py [--section NAME ...] [--results DIR]
Sections: victim, decode, drift, probe, duty, occupancy, phase (default: all).

Data (results/) and definitions:
  victim     tenancy/dep_co_g<f>_r{1,2,3}.csv: MobileNetV2 at 33 Hz, EMC 2133 MHz, NX, under
             the ViT-Small co-runner at 10 Hz (campaign/c3_tenancy.sh, phase 2), response_us;
             tenancy/prof_co_g<f>.json (phase 1, profile under the co-runner).  Solo
             baselines at the same clocks: govrep/g<f>_r{1,2,3}.csv (June session, same
             33 Hz rate, EMC 2133 MHz, campaign/govrep.sh and govrep_hi.sh) and the 8 Hz
             profiling cells nx_validation/nx_est_emc2133_gpu<f>_mobilenet.json.  A cycle
             'queued' is delayed beyond the solo distribution; it is counted with three
             cuts (over 7 ms; over the solo median + 1 ms; over the solo maximum).  Tail
             length = p99 - p50 of the co-tenant runs, and the largest delay beyond the solo
             median.  EMC samples: tenancy/pin_check.tegra (phase 3: tegrastats every
             0.5 s with two GPU clients; c3_tenancy.sh releases the BPMP lock with
             'emc_ctl.sh unlock' before this check).
  decode     boundary_stats/q15_g1173_r*.csv (solo, eight 2,000-token runs) and
             joint_hxt/q15co_g1173_r*.csv (nine runs under the co-runner), dur_ns per token,
             Qwen2.5-1.5B Q4_K_M, GPU 1173 MHz, EMC 2133 MHz.  Medians are pooled over the
             runs' tokens: all 2,000 tokens, and the first 300 (the solo short-window level
             T_short of campaign/analyze_joint.py).  A token is on the collision plateau
             when it exceeds 32 ms, a cut inside the empty gap between the two modes
             (the token counts in that gap are printed).
  drift      the same runs at 1173 MHz and joint_hxt/q15co_g1122_r*, boundary_stats/
             q15_g1122_r*.  Pooled drift (analyze_joint.py; the drift behind gamma):
             median(tokens 1501-2000 of all runs) / median(tokens 1-500 of all runs) - 1.
             Per-run drift: the same ratio within each run; its median over the runs.
  probe      joint_hxt/probe_co_g1173.csv (200-token decode probe under the co-runner):
             sample autocorrelation of dur_ns, sum((x_t - m)(x_t+k - m)) / sum((x_t - m)^2).
  duty       refit33/nx_{vit,yolov8s,resnet50}/emc2133_g1173_r{1,2,3}.json: the solo cell at
             the NX probe clock (1173 MHz, EMC 2133 MHz), median of the three p50s; duty =
             p50 x tenant release rate.
  occupancy  govrep_rev/{nx,nx_mobilenet,nx_mobilenet_s3}/g<f>_p<1..10>.json: median of the
             30 runs' compute_us p50 at each clock, divided by the release period
             (inference-occupied fraction).  June GPU load: govrep/g<f>_r{1,2,3}.tegra
             (tegrastats --interval 100), GR3D_FREQ load % averaged over each whole trace
             and over the samples whose one-second timestamp lies wholly inside the timed
             loop (loop end = the run JSON's timestamp, start = end - iters / hz; the
             trace's local-time offset is the whole number of hours that aligns its first
             sample with the loop start), then averaged over the three runs.
  phase      e3_nano/{A5r,A6r,A7}, e3_nx/{N5,...,N9}: every banded and point contract
             (t<k>.contract_{b,pt}.csv) with misses at its condition's deadline
             (condition.txt D_ms); lag-1 autocorrelation of the per-cycle miss indicator
             (definition as for 'probe'), launches grouped by the gaps between misses.
"""
import argparse
import csv
import glob
import json
import os
import re
import sys
from datetime import datetime, timedelta, timezone

import numpy as np

LAB = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def resp_csv(path, col=3):
    return np.loadtxt(path, delimiter=",", skiprows=1)[:, col]


def dur_csv(path):
    """Per-token decode time in ms."""
    return np.array([float(r["dur_ns"]) for r in csv.DictReader(open(path))]) / 1e6


def runs(pattern):
    return [p for p in sorted(glob.glob(pattern), key=_runkey) if not p.endswith(".power.csv")]


def _runkey(p):
    m = re.search(r"_[rp](\d+)\.", os.path.basename(p))
    return (re.sub(r"_[rp]\d+\.", "_.", p), int(m.group(1)) if m else 0)


def acf(x, k):
    x = np.asarray(x, float) - np.mean(x)
    v = float(np.sum(x * x))
    return float("nan") if v == 0 else float(np.sum(x[:-k] * x[k:]) / v)


def p50_json(path, key="response_us"):
    return float(json.load(open(path))[key]["p50"])


# --------------------------------------------------------------------------- victim
def sec_victim(R):
    print("== [main Sec. V] MobileNetV2 victim (33 Hz, NX, EMC 2133 MHz) under the 10 Hz "
          "ViT-Small co-runner, against solo runs ==")
    clocks = (918, 1020, 1122, 1173)
    print(f"  {'clock':>5s} {'co p50':>7s} {'solo33':>7s} {'shift':>7s} {'solo 8Hz':>8s} {'shift':>7s} "
          f"{'prof_co':>7s} {'>7ms':>6s} {'>s50+1':>6s} {'>smax':>6s} {'p99-p50':>7s} {'maxdly':>6s}")
    for f in clocks:
        co = [resp_csv(p) / 1e3 for p in runs(f"{R}/tenancy/dep_co_g{f}_r*.csv")]
        so = [resp_csv(p) / 1e3 for p in runs(f"{R}/govrep/g{f}_r*.csv")]
        s8 = f"{R}/nx_validation/nx_est_emc2133_gpu{f}_mobilenet.json"
        pc = f"{R}/tenancy/prof_co_g{f}.json"
        if not co or not so:
            print(f"  {f:5d} missing runs")
            continue
        c, s = np.concatenate(co), np.concatenate(so)
        cm, sm = float(np.median(c)), float(np.median(s))
        s8m = p50_json(s8) / 1e3 if os.path.isfile(s8) else float("nan")
        pcm = p50_json(pc) / 1e3 if os.path.isfile(pc) else float("nan")
        print(f"  {f:5d} {cm:7.3f} {sm:7.3f} {(cm / sm - 1) * 100:+6.2f}% {s8m:8.3f} {(cm / s8m - 1) * 100:+6.2f}% "
              f"{pcm:7.3f} {np.mean(c > 7.0) * 100:5.2f}% {np.mean(c > sm + 1.0) * 100:5.2f}% "
              f"{np.mean(c > s.max()) * 100:5.2f}% {np.percentile(c, 99) - cm:7.2f} {c.max() - sm:6.2f}")
    print("  (ms; shifts are co-tenant pooled median against the solo pooled median (33 Hz runs) and "
          "against the 8 Hz profiling cell's p50; prof_co = p50 of the profile under the co-runner)")
    print("  median of the three run p50s, co vs solo33:")
    for f in clocks:
        co = [float(np.median(resp_csv(p))) for p in runs(f"{R}/tenancy/dep_co_g{f}_r*.csv")]
        so = [float(np.median(resp_csv(p))) for p in runs(f"{R}/govrep/g{f}_r*.csv")]
        if co and so:
            print(f"    {f}: {np.median(co) / 1e3:.3f} vs {np.median(so) / 1e3:.3f} ms "
                  f"({(np.median(co) / np.median(so) - 1) * 100:+.2f} %)")
    print("  deadline misses under the co-runner (%), pooled over the three runs [per run]:")
    for D in (7.0, 9.0, 12.0):
        parts = []
        for f in clocks:
            co = [resp_csv(p) / 1e3 for p in runs(f"{R}/tenancy/dep_co_g{f}_r*.csv")]
            if co:
                parts.append(f"{f}: {np.mean(np.concatenate(co) > D) * 100:.2f} ["
                             + " ".join(f"{np.mean(v > D) * 100:.1f}" for v in co) + "]")
        print(f"    D = {D:4.1f} ms  " + "; ".join(parts))
    t = f"{R}/tenancy/pin_check.tegra"
    if os.path.isfile(t):
        lines = [ln for ln in open(t) if "EMC_FREQ" in ln]
        emc = [m.group(1) for m in (re.search(r"EMC_FREQ \d+%@(\d+)", ln) for ln in lines) if m]
        vals = {v: emc.count(v) for v in sorted(set(emc))}
        print(f"  EMC readings during the two-client check (tenancy/pin_check.tegra, 0.5 s samples): "
              + ", ".join(f"{v} MHz in {n}/{len(emc)}" for v, n in vals.items()))
    print()


# --------------------------------------------------------------------------- decode
def sec_decode(R):
    print("== [main Sec. V, Sec. VI; Fig. 4b] Qwen2.5-1.5B decode at 1173 MHz, solo and under the "
          "10 Hz ViT-Small co-runner ==")
    so = [dur_csv(p) for p in runs(f"{R}/boundary_stats/q15_g1173_r*.csv")]
    co = [dur_csv(p) for p in runs(f"{R}/joint_hxt/q15co_g1173_r*.csv")]
    if not so or not co:
        print("  missing runs\n")
        return
    S, C = np.concatenate(so), np.concatenate(co)
    s_all, c_all = float(np.median(S)), float(np.median(C))
    s300 = float(np.median(np.concatenate([r[:300] for r in so])))
    c300 = float(np.median(np.concatenate([r[:300] for r in co])))
    print(f"  runs: solo {len(so)}, co-tenant {len(co)}, tokens per run {len(so[0])}")
    print(f"  all-token median: solo {s_all:.3f} -> co-tenant {c_all:.3f} ms ({(c_all / s_all - 1) * 100:+.2f} %)")
    print(f"  first-300-token median: solo {s300:.3f} (T_short) -> co-tenant {c300:.3f} ms "
          f"({(c300 / s300 - 1) * 100:+.2f} %)")
    print(f"  solo T_short against the co-tenant all-token median: {s300:.3f} -> {c_all:.3f} ms "
          f"({(c_all / s300 - 1) * 100:+.2f} %)")
    for k in (3,):
        c_k = float(np.median(np.concatenate(co[:k])))
        print(f"  co-tenant all-token median over its first {k} runs only: {c_k:.3f} ms")
    gap = [np.count_nonzero((C > lo) & (C <= lo + 1.0)) for lo in range(30, 39)]
    print("  co-tenant tokens per 1 ms bin from 30 to 39 ms (the gap between the modes): "
          + " ".join(map(str, gap)))
    cut = 32.0
    hi = C[C > cut]
    print(f"  plateau (tokens > {cut:.0f} ms): {len(hi)} of {len(C)} = {len(hi) / len(C) * 100:.2f} %; "
          f"per run {min(np.mean(r > cut) for r in co) * 100:.2f}-{max(np.mean(r > cut) for r in co) * 100:.2f} %")
    print(f"  plateau location: median {np.median(hi):.2f} ms, IQR {np.percentile(hi, 25):.2f}-"
          f"{np.percentile(hi, 75):.2f}, p5-p95 {np.percentile(hi, 5):.2f}-{np.percentile(hi, 95):.2f} ms; "
          "share of all tokens per 1 ms bin: " + ", ".join(
              f"{b}-{b + 1}: {np.mean((C > b) & (C <= b + 1)) * 100:.1f}" for b in range(39, 44)))
    for c2 in (30.0, 35.0, 38.0):
        print(f"    (cut at {c2:.0f} ms: {np.mean(C > c2) * 100:.2f} %)")
    print(f"  tokens over 28 ms under the co-runner (the tenancy-blind D = 28 ms contract): "
          f"{np.mean(C > 28.0) * 100:.2f} %")
    print()


# --------------------------------------------------------------------------- drift
def drift_pooled(rs):
    return float(np.median(np.concatenate([r[1500:2000] for r in rs]))
                 / np.median(np.concatenate([r[:500] for r in rs])) - 1)


def sec_drift(R):
    print("== [main Sec. VI, Sec. VII] decode drift by token 2,000, under the co-runner and solo, "
          "at the same two clocks ==")
    out = {}
    for f in (1173, 1122):
        for tag, pat in (("co-tenant", f"{R}/joint_hxt/q15co_g{f}_r*.csv"),
                         ("solo", f"{R}/boundary_stats/q15_g{f}_r*.csv")):
            rs = [dur_csv(p) for p in runs(pat)]
            if not rs:
                continue
            per = [float(np.median(r[1500:2000]) / np.median(r[:500]) - 1) for r in rs]
            out[(tag, f)] = (drift_pooled(rs), float(np.median(per)))
            print(f"  {f} {tag:9s} runs={len(rs)}: pooled {drift_pooled(rs) * 100:+.3f} %, per-run median "
                  f"{np.median(per) * 100:+.3f} % (runs " + " ".join(f"{x * 100:+.2f}" for x in per) + ")")
    for i, name in ((0, "pooled"), (1, "per-run median")):
        for tag in ("co-tenant", "solo"):
            v = [out[(tag, f)][i] * 100 for f in (1173, 1122) if (tag, f) in out]
            if v:
                print(f"  {name:14s} {tag:9s}: {min(v):+.2f} to {max(v):+.2f} %")
    print()


# --------------------------------------------------------------------------- probe
def sec_probe(R):
    print("== [main Sec. VI] decode probe under the co-runner (200 tokens, 1173 MHz): "
          "autocorrelation by lag ==")
    p = f"{R}/joint_hxt/probe_co_g1173.csv"
    if not os.path.isfile(p):
        print("  missing joint_hxt/probe_co_g1173.csv\n")
        return
    x = dur_csv(p)
    print(f"  n = {len(x)}: " + "  ".join(f"lag {k}: {acf(x, k):+.3f}" for k in range(1, 11)))
    print()


# --------------------------------------------------------------------------- duty
def sec_duty(R):
    print("== [main Sec. VIII-A] tenant GPU duty at the NX probe clock: solo p50 (1173 MHz, "
          "EMC 2133 MHz) x release rate ==")
    rows = []
    for m, label, rates in (("yolov8s", "YOLOv8s", (5, 7, 10)), ("resnet50", "ResNet-50", (5, 10)),
                            ("vit", "ViT-Small", (10,))):
        fs = [f"{R}/refit33/nx_{m}/emc2133_g1173_r{k}.json" for k in (1, 2, 3)]
        if not all(os.path.isfile(f) for f in fs):
            print(f"  {label}: missing refit33/nx_{m} cells")
            continue
        pr = float(np.median([p50_json(f) for f in fs])) / 1e3
        pc = float(np.median([p50_json(f, "compute_us") for f in fs])) / 1e3
        for hz in rates:
            rows.append((label, hz, pr * hz / 10, pc * hz / 10))
            print(f"  {label:9s} {hz:2d} Hz: p50 resp {pr:.3f} ms -> {pr * hz / 10:5.2f} %; "
                  f"comp {pc:.3f} ms -> {pc * hz / 10:5.2f} %")
    tr = [r for r in rows if r[0] != "ViT-Small" and r[1] in (5, 10)]
    if tr:
        print(f"  transfer tenants at 5 and 10 Hz: {min(r[2] for r in tr):.1f}-{max(r[2] for r in tr):.1f} % "
              f"(resp), {min(r[3] for r in tr):.1f}-{max(r[3] for r in tr):.1f} % (comp)")
    print()


# --------------------------------------------------------------------------- occupancy
def tegra_samples(path):
    out = []
    for ln in open(path):
        m = re.match(r"(\d\d-\d\d-\d{4} \d\d:\d\d:\d\d) .*GR3D_FREQ (\d+)%", ln)
        if m:
            out.append((datetime.strptime(m.group(1), "%m-%d-%Y %H:%M:%S"), int(m.group(2))))
    return out


def sec_occupancy(R):
    print("== [main Sec. VIII-A; Supp. Sec. S2, occupancy] primary inference-occupied fraction "
          "(median compute p50 / release period) at the NX MobileNetV2 picks, against the June "
          "tegrastats GPU load ==")
    sessions = ("nx", "nx_mobilenet", "nx_mobilenet_s3")
    print(f"  {'clock':>5s} {'runs':>4s} {'comp p50':>8s} {'occupied':>8s} {'GR3D whole':>10s} "
          f"{'timed loop':>10s} {'dt(s)':>6s}")
    d_whole, d_loop = [], []
    for f in (1173, 1122, 1020, 918, 816):
        js = []
        for s in sessions:
            js += [p for p in glob.glob(f"{R}/govrep_rev/{s}/g{f}_p*.json")]
        if not js:
            continue
        per = json.load(open(js[0]))["meta"]["period_us"] / 1e3
        cp = float(np.median([p50_json(p, "compute_us") for p in js])) / 1e3
        occ = cp / per * 100
        whole, loop, dts = [], [], []
        for k in (1, 2, 3):
            t = f"{R}/govrep/g{f}_r{k}.tegra"
            j = f"{R}/govrep/g{f}_r{k}.json"
            if not (os.path.isfile(t) and os.path.isfile(j)):
                continue
            smp = tegra_samples(t)
            meta = json.load(open(j))["meta"]
            end = datetime.fromisoformat(meta["timestamp"]).astimezone(timezone.utc).replace(tzinfo=None)
            start = end - timedelta(seconds=meta["iters"] / meta["hz"])
            off = timedelta(hours=round((smp[0][0] - start).total_seconds() / 3600.0))
            s_loc, e_loc = start + off, end + off
            whole.append(np.mean([v for _, v in smp]))
            inside = [v for ts, v in smp if ts >= s_loc and ts + timedelta(seconds=1) <= e_loc]
            loop.append(np.mean(inside))
            span = (smp[-1][0] - smp[0][0]).total_seconds() + 1.0
            dts.append(span / len(smp))
        w, lo = float(np.mean(whole)), float(np.mean(loop))
        d_whole.append(occ - w)
        d_loop.append(occ - lo)
        print(f"  {f:5d} {len(js):4d} {cp:8.3f} {occ:7.3f}% {w:9.3f}% {lo:9.3f}% {np.median(dts):6.3f}")
    if d_whole:
        print(f"  occupied fraction minus GR3D load: whole trace {min(d_whole):+.3f} to {max(d_whole):+.3f} pp; "
              f"timed loop {min(d_loop):+.3f} to {max(d_loop):+.3f} pp")
    print()


# --------------------------------------------------------------------------- phase
def pattern(miss):
    """Miss pattern of a launch from the gaps between consecutive misses (95 % of the gaps)."""
    idx = np.flatnonzero(miss)
    if len(idx) < 3:
        return "isolated"
    g = np.diff(idx)
    if np.all(g == 1):
        return "every cycle"
    for gap, name in ((2, "alternate cycles"), (4, "every fourth cycle")):
        if np.mean(g == gap) >= 0.95:
            return name
    if np.mean((g == 1) | (g == 3)) >= 0.95 and abs(np.mean(g) - 2) < 0.05:
        return "two of every four"
    return "irregular"


def natural(tag):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", tag)]


def sec_phase(R):
    print("== [Supp. Sec. S2, phase-locking detail] lag-1 autocorrelation of the per-cycle miss "
          "indicator, ViT-Small primary launches with misses ==")
    groups = {}
    for d in [f"{R}/e3_nano/{c}" for c in ("A5r", "A6r", "A7")] + [f"{R}/e3_nx/N{k}" for k in range(5, 10)]:
        cf = f"{d}/condition.txt"
        if not os.path.isfile(cf):
            continue
        cond = dict(kv.split("=", 1) for kv in open(cf).read().split())
        D = float(cond["D_ms"]) * 1e3
        for p in sorted(glob.glob(f"{d}/t*.contract_*.csv")):
            miss = resp_csv(p) > D
            if not miss.any():
                continue
            pat = pattern(miss)
            r = acf(miss.astype(float), 1)
            tag = f"{os.path.basename(d)} {os.path.basename(p).split('.')[0]}" + (
                " (pt)" if "contract_pt" in p else "")
            groups.setdefault(pat, []).append((tag, np.mean(miss) * 100, r))
    for pat in ("alternate cycles", "every fourth cycle", "two of every four", "irregular", "isolated",
                "every cycle"):
        g = groups.get(pat, [])
        if not g:
            continue
        rr = [x[2] for x in g if not np.isnan(x[2])]
        rng = f"lag-1 {min(rr):+.3f} to {max(rr):+.3f}" if rr else "lag-1 undefined (no variance)"
        print(f"  {pat:19s} {len(g):3d} launches, miss {min(x[1] for x in g):.1f}-{max(x[1] for x in g):.1f} %, "
              f"{rng}: " + ", ".join(x[0] for x in sorted(g, key=lambda x: natural(x[0]))))
    print("  (an alternating miss pattern has lag-1 autocorrelation -1, a one-in-four pattern -1/3, "
          "a two-in-four pattern 0)")
    print()


SECTIONS = {"victim": sec_victim, "decode": sec_decode, "drift": sec_drift, "probe": sec_probe,
            "duty": sec_duty, "occupancy": sec_occupancy, "phase": sec_phase}


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

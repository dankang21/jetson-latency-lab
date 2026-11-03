#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""E5: memory-aware latency-model baseline -- the Han et al. monotone power-law
estimator, profiled at OTHER memory points, deployed through the paper's pick
rule and scored on the same interleaved sessions and pre-stated deadline grids
as the blind / aware arms (docs/protocols/E1_E2_protocol.md, E5 section,
frozen 2026-09-16 before scoring).

Read-only, offline (python3 + numpy over results/ only).  Run pinned:
    taskset -c 1 nice -n 19 python3 campaign/e5_han_baseline.py \
        [--md OUT.md] [--tex OUT.tex] [--weight abs|rel] [--model mobilenet,...] [--no-gemm]

Estimator (campaign/han_fit.py):  T(f_gpu, f_emc) = A f_gpu^-alpha + B f_emc^-beta [+ C]   [us]
fitted per model on the rate-matched sweeps results/nx_validation_r33/
nx_est_emc{665,2133,3199}_gpu{306..1173}_<model>.json (three EMC levels x ten GPU
clocks, 800 cycles at the deployment release rate; response_us p50), A, B >= 0.
  H1 (realistic)   train on the profiling points {665.6, 3199} only; the deployment
                   point 2133 is held out.  With the offset C free, beta is NOT
                   identifiable from two EMC levels (han_fit.py refuses), so H1 is
                   the source model's pure form (C = 0), which is identifiable.
                   Sensitivity H1c: C free with beta pinned at 1.0 (printed; a
                   different member of the same unidentified family).
  H2 (in-sample)   train on all three levels, C free, in-sample at 2133.
Pick rule (identical to both arms): lowest grid clock f with T_hat(f, 2133) * r <= D,
r = 1.035, else the policy DECLINES and the deployment falls back to the top clock
(1173 MHz; its measured outcome is shown in parentheses, not counted as a pick).
Blind / aware arms: the deployed two-cell endpoint refit k/f + b, rate-matched cells
(results/refit33/nx*, median of three repeat p50s) -- the headline panels of
Table II and the breadth table; recomputed here with the same functions
(campaign/govdeploy_common.py, campaign/e1b_stats.py).
Scoring (the Table II convention): miss = response_us > D per 1000-cycle run;
per (arm, D): median and min-max of the per-run miss %, number of runs with miss > tau
= 2 % out of 10 per session, pooled miss over all runs.  Sessions: NX MobileNetV2
results/govrep_rev/{nx, nx_mobilenet[, _s2, _s3]} (pooled, violations per session);
ViT-S / YOLOv8s / ResNet-50 results/govrep_rev/nx_<model> (one session each).
Summary per arm and model: deadlines with >= 1 violating run and deadlines failing
the pooled 2 % budget, out of the admitted (non-declined) evaluable deadlines; declines
and pick differences vs the aware arm are listed separately.
GEMM proxy: the only GPU x EMC sweep of the L2-resident GEMM is the Nano
results/estimator_break grid (3 EMC x 8 GPU); H1 / H2 in-sample and held-out errors,
the inversion cell's predicted sign, and the picks under the same rule are reported
there (no deployment session exists to score them on).
"""
import argparse
import csv
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LAB = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from han_fit import (load_sweep_dir, cells_dict, fit_han, predict, score, err_stats,   # noqa: E402
                     two_cell_kb, _load_estimator_break, HAVE_SCIPY, RES)
from e1b_stats import (fit_2cell, miss_pct, R, TAU, GRID, FMAX, refit33_fits,          # noqa: E402
                       original_fits as e1b_original_fits, MODELS as E1B)
from govdeploy_common import (BOARDS, refit_fits as mnv2_refit_fits,                   # noqa: E402
                              original_fits as mnv2_original_fits)

R33 = os.path.join(RES, "nx_validation_r33")
DEPLOY_EMC = 2133.0
PROFILE_EMC = (3199.0, 665.6)
MODELS = ["mobilenet", "vit", "yolov8s", "resnet50"]
ARMS = ["blind", "aware", "H1", "H2"]
ARM_LABEL = {"blind": "blind", "aware": "aware", "H1": "Han-H1", "H2": "Han-H2", "H1c": "Han-H1c"}
NANO_GRID = [306, 408, 510, 612, 714, 816, 918, 1020]


# ------------------------------------------------------------- config ----
def model_cfg(model):
    if model == "mobilenet":
        B = BOARDS["nx"]
        return dict(name="MobileNetV2", hz=B["hz"], grid=list(B["grid"]), D=list(B["D"]), fmax=B["fmax"],
                    sessions=list(B["sessions"]), table="Table II (tab:govdeploy)")
    M = E1B[model]
    return dict(name=M["name"], hz=float(M["hz"]), grid=list(GRID), D=list(M["grid"]), fmax=FMAX,
                sessions=[f"nx_{model}"], table="breadth table (tab:breadthgov)")


# --------------------------------------------------------------- data ----
def load_runs_dir(name, hz_expected):
    """{clock: {pair: dict(v=response_us array)}} for results/govrep_rev/<name>, or None.
    Same files, rate and length checks as govdeploy_common.load_session / e1b_stats.load_runs
    (energy is not needed here)."""
    root = os.path.join(RES, "govrep_rev", name)
    files = [f for f in sorted(glob.glob(os.path.join(root, "g*_p*.csv"))) if not f.endswith(".power.csv")]
    if not files:
        return None
    runs = {}
    for csvp in files:
        g, p = os.path.basename(csvp)[:-4].split("_p")
        f, pair = int(g[1:]), int(p)
        v = np.array([float(r["response_us"]) for r in csv.DictReader(open(csvp))])
        meta = json.load(open(csvp[:-4] + ".json"))["meta"]
        assert abs(float(meta["hz"]) - hz_expected) < 1e-6, (csvp, meta["hz"])
        assert len(v) == 1000, (csvp, len(v))
        runs.setdefault(f, {})[pair] = dict(v=v, ts=meta.get("timestamp", ""))
    return dict(name=name, runs=runs, pairs={f: len(runs[f]) for f in sorted(runs)})


def load_sessions(cfg):
    return [s for s in (load_runs_dir(n, cfg["hz"]) for n in cfg["sessions"]) if s]


# --------------------------------------------------------------- fits ----
def arm_fits(model):
    """Rate-matched two-cell fits of the two arms (headline), original fits as fallback."""
    if model == "mobilenet":
        fits, info = mnv2_refit_fits(BOARDS["nx"])
        if fits is not None:
            cells = {arm: (info[arm]["cells"][306], info[arm]["cells"][FMAX], info[arm]["hz"]) for arm in fits}
            return fits, cells, "rate-matched (results/refit33/nx)"
        fits, info = mnv2_original_fits(BOARDS["nx"])
        cells = {arm: (info[arm]["cells"][306], info[arm]["cells"][FMAX], info[arm]["hz"]) for arm in fits}
        return fits, cells, "original 8 Hz cells (results/nx_validation)"
    fits, info = refit33_fits(model)
    if fits is not None:
        cells = {arm: (info[(arm, 306)]["p50"], info[(arm, FMAX)]["p50"], info[(arm, 306)]["hz"]) for arm in fits}
        return fits, cells, f"rate-matched (results/refit33/nx_{model})"
    fits, info = e1b_original_fits(model)
    cells = {arm: (info[(arm, 306)]["p50"], info[(arm, FMAX)]["p50"], info[(arm, 306)]["hz"]) for arm in fits}
    return fits, cells, "original cells (results/nx_validation)"


def han_fits(d, weight, deploy=DEPLOY_EMC):
    """H1 (pure Han, profiling levels only), H1c (+C, beta pinned 1.0), H2 (all levels, +C)."""
    train1 = sorted(k for k in d if abs(k[0] - deploy) > 1e-6)
    hold = sorted(k for k in d if abs(k[0] - deploy) < 1e-6)
    P = {
        "H1": fit_han(d, train=train1, with_c=False, weight=weight),
        "H1c": fit_han(d, train=train1, with_c=True, weight=weight, fix_beta=1.0),
        "H2": fit_han(d, train=None, with_c=True, weight=weight),
    }
    return P, train1, hold


def fmt_params(p):
    s = f"{p['A']:.1f}*g^-{p['alpha']:.3f} + {p['B']:.1f}*e^-{p['beta']:.3f}"
    if p.get("with_c", True):
        s += f" + {p['C']:.1f}"
    s += f"   [us; g = f_gpu/1 GHz, e = f_emc/1 GHz; {p['n_train']} cells, {p['fitter']}"
    if p.get("fix_beta") is not None:
        s += f"; beta PINNED {p['fix_beta']}"
    return s + "]"


# --------------------------------------------------------------- rule ----
def T2(kb, f):
    return (kb[0] / f + kb[1]) * R


def pick_2cell(kb, D_us, grid, fmax):
    for f in grid:
        if T2(kb, f) <= D_us:
            return f, True
    return fmax, False


def pick_han(params, D_us, grid, fmax, emc=DEPLOY_EMC):
    for f in grid:
        if predict(params, f, emc) * R <= D_us:
            return f, True
    return fmax, False


# ------------------------------------------------------------ scoring ----
_R33_CSV = {}


def r33_miss(model, f, D_us):
    """Indicative only: miss % of the 800-cycle r33 sweep cell (EMC 2133, GPU f) from the profiling
    session, for picks the interleaved sessions did not run (1173 MHz in the breadth sessions)."""
    key = (model, f)
    if key not in _R33_CSV:
        p = os.path.join(R33, f"nx_est_emc2133_gpu{f}_{model}.csv")
        _R33_CSV[key] = (np.array([float(r["response_us"]) for r in csv.DictReader(open(p))])
                         if os.path.isfile(p) else None)
    v = _R33_CSV[key]
    return None if v is None else miss_pct(v, D_us)


def cell(sessions, f, feasible, D_us, indic=None):
    per = []
    for s in sessions:
        per.append([miss_pct(s["runs"][f][p]["v"], D_us) for p in sorted(s["runs"][f])]
                   if f in s["runs"] else None)
    allm = [m for l in per if l for m in l]
    c = dict(f=f, feasible=feasible, per=per, has_data=bool(allm))
    if not allm and indic is not None:
        c["indic"] = indic(f, D_us)
    if allm:
        misses = sum(int(np.count_nonzero(s["runs"][f][p]["v"] > D_us))
                     for s in sessions if f in s["runs"] for p in s["runs"][f])
        cycles = sum(len(s["runs"][f][p]["v"]) for s in sessions if f in s["runs"] for p in s["runs"][f])
        viol = [None if l is None else sum(1 for m in l if m > TAU) for l in per]
        c.update(med=float(np.median(allm)), mn=min(allm), mx=max(allm), viol=viol,
                 nviol=sum(v for v in viol if v), nruns=len(allm), pooled=100.0 * misses / cycles,
                 all10=all(v == len(l) for v, l in zip(viol, per) if l is not None))
    return c


def evaluate(cfg, fits2, P, sessions, model):
    rows = []
    indic = lambda f, D_us: r33_miss(model, f, D_us)  # noqa: E731
    for D in cfg["D"]:
        D_us = D * 1000.0
        row = dict(D=D, T={})
        for arm in ("blind", "aware"):
            f, ok = pick_2cell(fits2[arm], D_us, cfg["grid"], cfg["fmax"])
            row[arm] = cell(sessions, f, ok, D_us, indic)
            row["T"][arm] = T2(fits2[arm], f)
        for arm in ("H1", "H1c", "H2"):
            f, ok = pick_han(P[arm], D_us, cfg["grid"], cfg["fmax"])
            row[arm] = cell(sessions, f, ok, D_us, indic)
            row["T"][arm] = predict(P[arm], f, DEPLOY_EMC) * R
        rows.append(row)
    return rows


def summary(rows, arm):
    """Counts over the grid: admitted+evaluable cells with >=1 violating run / failing the pooled
    budget / all runs violating; declines; no-data; picks vs the aware arm."""
    S = dict(N=len(rows), decl=0, na=0, eval=0, viol=0, pool=0, all10=0, same=0, higher=0, lower=0,
             decl_aware_admits=0, admits_aware_declines=0, na_indic_max=None)
    for r in rows:
        c, a = r[arm], r["aware"]
        if c["feasible"] and a["feasible"]:
            S["same" if c["f"] == a["f"] else ("higher" if c["f"] > a["f"] else "lower")] += 1
        if not c["feasible"]:
            S["decl"] += 1
            if a["feasible"]:
                S["decl_aware_admits"] += 1
            continue
        if not a["feasible"]:
            S["admits_aware_declines"] += 1
        if not c["has_data"]:
            S["na"] += 1
            if c.get("indic") is not None:
                S["na_indic_max"] = max(c["indic"], S["na_indic_max"] or 0.0)
            continue
        S["eval"] += 1
        S["viol"] += c["nviol"] >= 1
        S["pool"] += c["pooled"] > TAU
        S["all10"] += c["all10"]
    return S


# ------------------------------------------------------------- output ----
def pm(x):
    return "100" if x >= 99.95 else f"{x:.1f}"


def fmt_cell(c, tex=False):
    if not c["has_data"]:
        body = "n/a" + (f" [r33 cell {pm(c['indic'])}]" if c.get("indic") is not None else "")
    else:
        ks = []
        for k, l in zip(c["viol"], c["per"]):
            if l is None:
                ks.append("--" if tex else "–")
            elif tex:
                ks.append(f"\\textbf{{{k}/{len(l)}}}" if k >= 1 else f"{k}/{len(l)}")
            else:
                ks.append(f"**{k}/{len(l)}**" if k >= 1 else f"{k}/{len(l)}")
        rng = "" if c["mn"] == c["mx"] else (f" [{pm(c['mn'])}--{pm(c['mx'])}]" if tex else f" [{pm(c['mn'])}–{pm(c['mx'])}]")
        body = f"{pm(c['med'])}{rng} ({', '.join(ks)})"
    if c["feasible"]:
        return f"{c['f']} → {body}" if not tex else f"{c['f']}: {body}"
    return f"declines ({c['f']}: {body})"


def fmt_summary(S):
    return (f"{S['viol']}/{S['eval']} admitted deadlines with >=1 violating run, {S['pool']}/{S['eval']} fail the "
            f"pooled {TAU:.0f} % budget, {S['all10']} all-runs-violating; declines {S['decl']}"
            + (f" (of which the aware arm admits {S['decl_aware_admits']})" if S["decl_aware_admits"] else "")
            + (f"; admits {S['admits_aware_declines']} deadline(s) the aware arm declines" if S["admits_aware_declines"] else "")
            + (f"; no data {S['na']} (1173 not run; r33 sweep cell max {S['na_indic_max']:.1f} %)"
               if S["na"] and S["na_indic_max"] is not None else (f"; no data {S['na']}" if S["na"] else ""))
            + f"; pick vs aware: same {S['same']}, higher {S['higher']}, lower {S['lower']}")


def print_model(model, cfg, sessions, fits2, fcells, ftag, d, P, train1, hold, rows):
    print(f"\n##### NX {cfg['name']} {cfg['hz']:.0f} Hz -- {cfg['table']}; deadlines {cfg['D']} ms; grid {cfg['grid']}")
    for s in sessions:
        print(f"  session {s['name']}: clocks/pairs {s['pairs']}")
    print(f"  arms: two-cell {ftag}: " + "; ".join(
        f"{arm} k={fits2[arm][0]:.0f} b={fits2[arm][1]:.0f} (cells {fcells[arm][0]:.0f}/{fcells[arm][1]:.0f} us @{fcells[arm][2]:.0f} Hz)"
        for arm in ("blind", "aware")))
    hz = {c.hz for c in load_sweep_dir(R33, model)}
    print(f"  Han fits on results/nx_validation_r33 ({len(d)} cells, {sorted({k[0] for k in d}, reverse=True)} MHz x "
          f"{len({k[1] for k in d})} GPU, rate {sorted(hz)} Hz; weight {P['H1']['weight']}):")
    for arm in ("H1", "H1c", "H2"):
        print(f"    {ARM_LABEL[arm]:7s} T_hat = {fmt_params(P[arm])}")
    # prediction table at the deployment point
    gpus = sorted(k[1] for k in hold)
    print(f"  p50 at EMC 2133 [ms]: measured r33 sweep | H1 (err %) | H1c (err %) | H2 (err %) | two-cell aware chord (err %)")
    for g in gpus:
        m = d[(DEPLOY_EMC, g)]
        line = f"    {g:>5}: {m / 1e3:6.2f} |"
        for arm in ("H1", "H1c", "H2"):
            t = predict(P[arm], g, DEPLOY_EMC)
            line += f" {t / 1e3:6.2f} ({(t - m) / m * 100:+5.1f}) |"
        t = fits2["aware"][0] / g + fits2["aware"][1]
        line += f" {t / 1e3:6.2f} ({(t - m) / m * 100:+5.1f})"
        print(line)
    stats = {}
    for arm in ("H1", "H1c", "H2"):
        ins = err_stats(score(P[arm], d, train1 if arm != "H2" else sorted(d)))
        out = err_stats(score(P[arm], d, hold))
        sgn = np.mean([r[3] for r in score(P[arm], d, hold)]) * 100
        stats[arm] = dict(in_med=ins[0], in_max=ins[1], out_med=out[0], out_max=out[1], out_mean_signed=sgn)
        print(f"  {ARM_LABEL[arm]:7s} |err| %: {'in-sample' if arm == 'H2' else 'train {3199, 665.6}'} median {ins[0]:.1f} max {ins[1]:.1f}; "
              f"2133 row {'(in-sample)' if arm == 'H2' else '(held-out)'} median {out[0]:.1f} max {out[1]:.1f} "
              f"(mean signed {sgn:+.1f})")
    top = max(gpus)
    dm = (d[(DEPLOY_EMC, top)] - d[(3199.0, top)]) / 1e3
    print(f"  T(2133)-T(3199) at GPU {top}: measured {dm:+.2f} ms; " + "; ".join(
        f"{ARM_LABEL[a]} {(predict(P[a], top, DEPLOY_EMC) - predict(P[a], top, 3199.0)) / 1e3:+.2f}" for a in ("H1", "H1c", "H2")))
    # deadline table
    print(f"  {'D':>5} | " + " | ".join(f"{ARM_LABEL[a]:^44}" for a in ("blind", "aware", "H1", "H2", "H1c")))
    for r in rows:
        print(f"  {r['D']:5.1f} | " + " | ".join(f"{fmt_cell(r[a]):44}" for a in ("blind", "aware", "H1", "H2", "H1c")))
    print("  T_hat(pick)*r [ms]: " + "; ".join(
        f"D={r['D']:.1f}: " + "/".join(f"{r['T'][a] / 1e3:.2f}" for a in ("blind", "aware", "H1", "H2", "H1c")) for r in rows))
    print("  per-run miss % (pairs 1..10, per session) for the Han picks:")
    for r in rows:
        for arm in ("H1", "H2", "H1c"):
            c = r[arm]
            for s, l in zip(sessions, c["per"]):
                if l is not None:
                    print(f"    D={r['D']:5.1f} {ARM_LABEL[arm]:7s} {c['f']:4d} {s['name']:14s}: " + " ".join(f"{x:5.1f}" for x in l))
    S = {a: summary(rows, a) for a in ("blind", "aware", "H1", "H2", "H1c")}
    for a in ("blind", "aware", "H1", "H2", "H1c"):
        print(f"  {ARM_LABEL[a]:7s} summary: {fmt_summary(S[a])}")
    return stats, S


def md_model(cfg, sessions, fits2, fcells, ftag, P, stats, rows, S):
    L = [f"### NX {cfg['name']} {cfg['hz']:.0f} Hz — {cfg['table']}; "
         f"{len(sessions)} interleaved session{'s' if len(sessions) > 1 else ''} ({', '.join(s['name'] for s in sessions)})", ""]
    L.append("- Two-cell arms, " + ftag + ": " + "; ".join(
        f"{arm} k={fits2[arm][0]:.0f} b={fits2[arm][1]:.0f} (cells {fcells[arm][0]:.0f}/{fcells[arm][1]:.0f} µs @{fcells[arm][2]:.0f} Hz)"
        for arm in ("blind", "aware")) + ".")
    for arm in ("H1", "H1c", "H2"):
        L.append(f"- {ARM_LABEL[arm]}: T̂ = {fmt_params(P[arm])}; |err| at 2133 "
                 f"{'in-sample' if arm == 'H2' else 'held-out'} median {stats[arm]['out_med']:.1f} % max {stats[arm]['out_max']:.1f} % "
                 f"(mean signed {stats[arm]['out_mean_signed']:+.1f} %); training rows median {stats[arm]['in_med']:.1f} % max {stats[arm]['in_max']:.1f} %.")
    L.append("")
    L.append("| D (ms) | blind | aware | Han-H1 | Han-H2 |")
    L.append("|---:|---|---|---|---|")
    for r in rows:
        L.append(f"| {r['D']:.1f} | " + " | ".join(fmt_cell(r[a]) for a in ("blind", "aware", "H1", "H2")) + " |")
    L.append("")
    h1c_same = all(r["H1c"]["f"] == r["H1"]["f"] and r["H1c"]["feasible"] == r["H1"]["feasible"] for r in rows)
    if h1c_same:
        L.append("- Han-H1c (C free, β pinned 1.0) picks the same clock as Han-H1 at every deadline.")
    else:
        diff = [f"{r['D']:.1f}: {fmt_cell(r['H1c'])}" for r in rows
                if not (r["H1c"]["f"] == r["H1"]["f"] and r["H1c"]["feasible"] == r["H1"]["feasible"])]
        L.append("- Han-H1c (C free, β pinned 1.0) differs from Han-H1 at: " + "; ".join(diff) + ".")
    for a in ("blind", "aware", "H1", "H2", "H1c"):
        L.append(f"- {ARM_LABEL[a]} summary: {fmt_summary(S[a])}.")
    L.append("")
    return L


# ---------------------------------------------------------------- GEMM ----
def gemm_inventory():
    lines = []
    eb = sorted(glob.glob(os.path.join(RES, "estimator_break", "emc*_gpu*_cproxyv2.json")))
    per = defaultdict(set)
    for f in eb:
        b = os.path.basename(f)
        per[b.split("_")[0]].add(b.split("_")[1])
    lines.append(f"results/estimator_break: {len(eb)} L2-resident GEMM (cproxyv2) cells, "
                 + "; ".join(f"{e}: {len(g)} GPU" for e, g in sorted(per.items(), reverse=True))
                 + " (Nano, 300 iters @ 3 Hz, root) -> the only GPU x EMC GEMM sweep; fittable")
    pe = sorted(glob.glob(os.path.join(RES, "pilot_emc", "emc*_cproxy*.json")))
    lines.append(f"results/pilot_emc: {len(pe)} GEMM cells ({', '.join(os.path.basename(f) for f in pe)}) -- "
                 "GPU 1020 only (no GPU sweep: alpha not identifiable)")
    nxc = sorted(glob.glob(os.path.join(RES, "nx_validation", "nx_emc*_cproxyv2.json")))
    nxr = sorted(glob.glob(os.path.join(RES, "parta_reps_nx", "*cproxyv2.json")))
    lines.append(f"NX: results/nx_validation catalogue {len(nxc)} cells @GPU 1173 only; results/parta_reps_nx {len(nxr)} "
                 "cells @1173 only; results/nx_validation_r33 has no GEMM -> no NX GEMM sweep exists")
    return lines


def gemm_section(weight):
    print("\n##### L2-resident GEMM proxy (cproxyv2): sweep inventory and Han fits")
    for l in gemm_inventory():
        print("  " + l)
    cells = _load_estimator_break("cproxyv2", "response_us")
    d = cells_dict(cells)
    P, train1, hold = han_fits(d, weight)
    L = ["### L2-resident GEMM proxy (the inversion cell; Nano results/estimator_break, 3 EMC × 8 GPU, 300 iters @ 3 Hz)", ""]
    L += [f"- {l}" for l in gemm_inventory()]
    top = max(k[1] for k in d)
    dm = (d[(DEPLOY_EMC, top)] - d[(3199.0, top)]) / 1e3
    stats = {}
    for arm in ("H1", "H1c", "H2"):
        print(f"  {ARM_LABEL[arm]:7s} T_hat = {fmt_params(P[arm])}")
        ins = err_stats(score(P[arm], d, train1 if arm != "H2" else sorted(d)))
        out = err_stats(score(P[arm], d, hold))
        dp = (predict(P[arm], top, DEPLOY_EMC) - predict(P[arm], top, 3199.0)) / 1e3
        stats[arm] = (ins, out, dp)
        print(f"          |err| %: {'in-sample (24)' if arm == 'H2' else 'train {3199, 665.6} (16)'} median {ins[0]:.1f} max {ins[1]:.1f}; "
              f"2133 row {'(in-sample)' if arm == 'H2' else '(held-out)'} median {out[0]:.1f} max {out[1]:.1f}; "
              f"T(2133)-T(3199)@{top}: Han {dp:+.2f} ms vs measured {dm:+.2f} ms ({'sign OK' if np.sign(dp) == np.sign(dm) else 'WRONG SIGN'})")
        L.append(f"- {ARM_LABEL[arm]}: T̂ = {fmt_params(P[arm])}; |err| training rows median {ins[0]:.1f} % max {ins[1]:.1f} %, "
                 f"2133 row {'in-sample' if arm == 'H2' else 'held-out'} median {out[0]:.1f} % max {out[1]:.1f} %; "
                 f"Δ̂(2133−3199)@{top} {dp:+.2f} ms vs measured {dm:+.2f} ms ({'sign reproduced' if np.sign(dp) == np.sign(dm) else 'wrong sign'}).")
    gpus = sorted(k[1] for k in hold)
    print("  p50 at EMC 2133 [ms]: measured | H1 | H1c | H2 | two-cell aware (2133 endpoints 306/1020) | two-cell blind (3199)")
    # two_cell_kb returns k for T = k*1e3/f + b (GHz form); convert to the e1b (k/f + b, f in MHz) form used by T2/pick_2cell
    kb_a = tuple(x * y for x, y in zip(two_cell_kb(d, DEPLOY_EMC), (1e3, 1.0)))
    kb_b = tuple(x * y for x, y in zip(two_cell_kb(d, 3199.0), (1e3, 1.0)))
    L += ["", "| GPU | measured 2133 | Han-H1 | Han-H2 | 2-cell aware | 2-cell blind |", "|---:|---:|---:|---:|---:|---:|"]
    for g in gpus:
        m = d[(DEPLOY_EMC, g)]
        vals = [predict(P[a], g, DEPLOY_EMC) for a in ("H1", "H1c", "H2")] + [kb_a[0] / g + kb_a[1], kb_b[0] / g + kb_b[1]]
        print(f"    {g:>5}: {m / 1e3:6.2f} | " + " | ".join(f"{v / 1e3:6.2f} ({(v - m) / m * 100:+5.1f})" for v in vals))
        L.append(f"| {g} | {m / 1e3:.2f} | " + " | ".join(f"{v / 1e3:.2f} ({(v - m) / m * 100:+.1f} %)" for v in (vals[0], vals[2], vals[3], vals[4])) + " |")
    # picks under the same rule (no deployment session: compared with the measured-curve oracle);
    # deadlines by the protocol's grid formula: T_aware(f)*r rounded up to 0.5 ms, plus each +0.5 ms
    base = {float(np.ceil(T2(kb_a, g) / 500.0) * 0.5) for g in NANO_GRID if g >= 510}
    Ds = sorted(base | {b + 0.5 for b in base})
    print(f"  picks at EMC 2133, r={R}, Nano grid {NANO_GRID}, D = T_aware(f)*r rounded up to 0.5 ms plus each +0.5 (no session "
          "to score; oracle = lowest clock whose MEASURED 2133 p50*r <= D):")
    print(f"    {'D ms':>6} | oracle | blind | aware | H1 | H1c | H2")
    L += ["", f"Picks under the same rule (r = {R}, Nano grid 306–1020 MHz, deadlines = T̂_aware(f)·r rounded up to 0.5 ms plus "
              "each +0.5 ms, the protocol's grid formula; no deployment session exists, so the comparison is against the oracle "
              "pick from the measured 2133 cells):", "", "| D (ms) | oracle (measured 2133) | blind | aware | Han-H1 | Han-H2 |", "|---:|---:|---:|---:|---:|---:|"]
    for D in Ds:
        D_us = D * 1000.0
        orc = next((g for g in NANO_GRID if d[(DEPLOY_EMC, g)] * R <= D_us), None)
        pb, okb = pick_2cell(kb_b, D_us, NANO_GRID, 1020)
        pa, oka = pick_2cell(kb_a, D_us, NANO_GRID, 1020)
        hp = {a: pick_han(P[a], D_us, NANO_GRID, 1020) for a in ("H1", "H1c", "H2")}
        def fp(x):
            return f"{x[0]}" if x[1] else "declines"
        print(f"    {D:6.1f} | {orc or 'none':>6} | {fp((pb, okb)):>8} | {fp((pa, oka)):>8} | {fp(hp['H1']):>8} | {fp(hp['H1c']):>8} | {fp(hp['H2']):>8}")
        L.append(f"| {D:.1f} | {orc or 'none'} | {fp((pb, okb))} | {fp((pa, oka))} | {fp(hp['H1'])} | {fp(hp['H2'])} |")
    L.append("")
    return L, stats


# ---------------------------------------------------------------- TeX ----
def tex_table(cfgs, SS, stats_all, path):
    """Compact one-column summary: per model and arm, 'v(p)/a' = deadlines with >=1 violating run
    (pooled-budget failures when different) out of the a admitted, evaluable deadlines; superscripts:
    dagger d = grid deadlines the arm declines, ddagger m = admitted picks the session did not run."""
    def cnt(S):
        s = f"{S['viol']}"
        if S["pool"] != S["viol"]:
            s += f"\\,({S['pool']})"
        s += f"/{S['eval']}"
        sup = ""
        if S["decl"]:
            sup += f"\\dagger{S['decl']}"
        if S["na"]:
            sup += f"\\ddagger{S['na']}"
        return s + (f"$^{{{sup}}}$" if sup else "")
    tot = {a: dict(viol=0, pool=0, decl=0, na=0, N=0, eval=0, same=0, higher=0, lower=0, decl_aware_admits=0)
           for a in ("blind", "aware", "H1", "H2")}
    for model in MODELS:
        for a in tot:
            for k in tot[a]:
                tot[a][k] += SS[model][a][k]
    N = tot["blind"]["N"]
    h1, h2 = tot["H1"], tot["H2"]
    both1, both2 = h1["same"] + h1["higher"] + h1["lower"], h2["same"] + h2["higher"] + h2["lower"]
    indic = [SS[m][a]["na_indic_max"] for m in MODELS for a in ("H1", "H2") if SS[m][a]["na_indic_max"] is not None]
    indic_txt = (f"misses at most {max(indic):.1f}\\,\\% in every such case" if indic else "n/a")
    cap = (
        "Memory-aware latency-model baseline (E5): a Han-style monotone power law "
        "$\\hat T = A f_{\\mathrm{gpu}}^{-\\alpha} + B f_{\\mathrm{emc}}^{-\\beta}\\,[+C]$ fitted per model on the "
        "rate-matched GPU$\\times$EMC sweeps (EMC $\\{665.6, 2133, 3199\\}$\\,MHz $\\times$ ten GPU clocks, 800 cycles at "
        "the deployment release rate) and deployed through the same rule as both arms ($f^\\star$ = lowest grid clock with "
        "$\\hat T(f, 2133)\\,r\\le D$, $r{=}1.035$, else the deadline is declined). H1 (realistic): fitted on the "
        "profiling points $\\{665.6, 3199\\}$\\,MHz only, the deployment point 2133\\,MHz held out (pure Han form, $C{=}0$: "
        "with a free offset, $\\beta$ is not identifiable from two memory levels); H2 (in-sample reference): all three levels "
        "in-sample, $C$ free. Every arm is scored offline on the same interleaved sessions and pre-stated deadline grids "
        "as main-text Table~II (MobileNetV2, the three sessions) and Table~\\ref{tab:breadthgov} (one session per model). "
        "Entries: deadlines with ${\\ge}1$ run above the $\\tau{=}2\\,\\%$ budget (in parentheses, when different: "
        "deadlines failing the pooled budget) out of the admitted, evaluable deadlines; superscripts $\\dagger d$ = the arm "
        "declines $d$ of the $N$ grid deadlines (the aware arm declines only 4.4\\,ms, at which no tested clock at the "
        "2133\\,MHz deployment point kept the budget in every run), "
        "$\\ddagger m$ = $m$ further admitted picks of 1173\\,MHz that the breadth sessions did not run, excluded from "
        f"the counts (the 800-cycle profiling cell at that clock {indic_txt}). Right: median\\,/\\,max $|$error$|$ (\\%) "
        "of the Han p50 "
        f"prediction at EMC 2133\\,MHz over the ten GPU clocks (H1 held out, H2 in-sample). Over the {N} deadlines the "
        f"aware arm admits {tot['aware']['eval'] + tot['aware']['na']} with no violating run; H1 admits "
        f"{h1['eval'] + h1['na']}, declining {h1['decl_aware_admits']} that the aware arm meets, and picks a higher clock "
        f"than the aware arm at {h1['higher']} of the {both1} deadlines both admit; H2 admits {h2['eval'] + h2['na']} and "
        f"picks the same clock as the aware arm at {h2['same']}, a higher one at {h2['higher']} and a lower one at "
        f"{h2['lower']} of {both2}.")
    L = [r"\begin{table}[t]", r"  \centering", "  \\caption{" + cap + "}",
         r"  \label{tab:e5han}", r"  \footnotesize", r"  \setlength{\tabcolsep}{3.5pt}",
         r"  \resizebox{\columnwidth}{!}{%",
         r"  \begin{tabular}{@{}lrllllrr@{}}", r"    \toprule",
         r"    & & \multicolumn{4}{c}{violating / admitted deadlines} & \multicolumn{2}{c}{$|$err$|$ @2133 (\%)} \\",
         r"    \cmidrule(lr){3-6}\cmidrule(lr){7-8}",
         r"    model (rate) & $N$ & blind & aware & Han-H1 & Han-H2 & H1 med/max & H2 med/max \\",
         r"    \midrule"]
    for model in MODELS:
        cfg, S, st = cfgs[model], SS[model], stats_all[model]
        row = f"    {cfg['name']} ({cfg['hz']:.0f}\\,Hz) & {S['blind']['N']} & " + " & ".join(cnt(S[a]) for a in ARMS)
        row += f" & {st['H1']['out_med']:.1f}\\,/\\,{st['H1']['out_max']:.1f} & {st['H2']['out_med']:.1f}\\,/\\,{st['H2']['out_max']:.1f} \\\\"
        L.append(row)
    L.append(r"    \midrule")
    L.append(f"    all & {N} & " + " & ".join(cnt(tot[a]) for a in ARMS) + " & & \\\\")
    L += [r"    \bottomrule", r"  \end{tabular}}", r"\end{table}"]
    tex = "\n".join(L) + "\n"
    open(path, "w").write(tex)
    return tex


# ---------------------------------------------------------------- main ----
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=",".join(MODELS))
    ap.add_argument("--weight", default="abs", choices=["abs", "rel"], help="han_fit residual weighting")
    ap.add_argument("--md", default=None, help="write the markdown tables here")
    ap.add_argument("--tex", default=None, help="write the compact LaTeX summary table here")
    ap.add_argument("--no-gemm", action="store_true")
    a = ap.parse_args()
    print(f"# E5 Han baseline: rule r={R}, tau={TAU} %, deploy EMC {DEPLOY_EMC:g}, profiling EMC {PROFILE_EMC}; "
          f"fitter numpy-grid{' + scipy polish' if HAVE_SCIPY else ''}; weight {a.weight}; sweeps {os.path.relpath(R33, LAB)}")
    md = ["## Generated tables (campaign/e5_han_baseline.py --md; do not edit by hand)", "",
          f"Rule r = {R}, τ = {TAU:.0f} %; fitter numpy-grid{' + scipy polish' if HAVE_SCIPY else ''}, weight {a.weight}; "
          f"sweeps results/nx_validation_r33 (response_us p50). Cell format: pick → per-run miss % median [min–max] "
          f"(runs > τ per session, bold when ≥ 1); 'declines (1173: …)' = no feasible clock, top-clock fallback measured; "
          f"'n/a [r33 cell x]' = the picked clock was not run in that session (1173 in the breadth sessions), x = miss % "
          f"of the 800-cycle r33 profiling cell at (2133, that clock) — indicative only, not a scored cell.", ""]
    cfgs, SS, stats_all = {}, {}, {}
    for model in a.model.split(","):
        cfg = model_cfg(model)
        sessions = load_sessions(cfg)
        assert sessions, f"no session for {model}"
        fits2, fcells, ftag = arm_fits(model)
        d = cells_dict(load_sweep_dir(R33, model))
        assert len(d) == 30, (model, len(d))
        P, train1, hold = han_fits(d, a.weight)
        rows = evaluate(cfg, fits2, P, sessions, model)
        stats, S = print_model(model, cfg, sessions, fits2, fcells, ftag, d, P, train1, hold, rows)
        cfgs[model], SS[model], stats_all[model] = cfg, S, stats
        md += md_model(cfg, sessions, fits2, fcells, ftag, P, stats, rows, S)
    if not a.no_gemm:
        gl, _ = gemm_section(a.weight)
        md += gl
    print("\n##### Summary (admitted deadlines with >=1 violating run / failing the pooled budget, out of N; declines)")
    for model in cfgs:
        S = SS[model]
        print(f"  {cfgs[model]['name']:12s} N={S['blind']['N']:2d}: " + "; ".join(
            f"{ARM_LABEL[a]} {S[a]['viol']}/{S[a]['pool']}" + (f" decl {S[a]['decl']}" if S[a]["decl"] else "") + (f" n/a {S[a]['na']}" if S[a]["na"] else "")
            for a in ("blind", "aware", "H1", "H2", "H1c")))
    if a.md:
        open(a.md, "w").write("\n".join(md) + "\n")
        print(f"\nwrote {a.md}")
    if a.tex:
        assert set(cfgs) == set(MODELS), "the LaTeX table needs all four models"
        tex_table(cfgs, SS, stats_all, a.tex)
        print(f"wrote {a.tex}")


if __name__ == "__main__":
    main()

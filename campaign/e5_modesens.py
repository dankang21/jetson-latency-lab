#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""E5 / E5-B post-hoc sensitivity: the tail state of the rate-matched sweeps that train the Han-style fits.

Post hoc (2026-11-11): this analysis was specified in neither the E5 nor the E5-B protocol; a dated
post-hoc entry of docs/protocols/E1_E2_protocol.md records it.  It changes no reported table.
Read-only over results/.

The E5 fits (campaign/e5_han_baseline.py, campaign/e5b_emcfree.py) are trained on the p50 of each
cell of results/nx_validation_r33 (three EMC levels x ten GPU clocks, 800 cycles per cell).  The
cycles of a cell fall into two latency modes, a fast and a slow one, and the cell p50 lands on
either, depending on which mode holds more cycles.  The scoring sessions of Table II
(results/govrep_rev/{nx, nx_mobilenet, nx_mobilenet_s3}) are unimodal.  This script

  1. splits every sweep cell into its two modes and reports, per cell, the cycles in each mode,
     the two mode medians and the mode the cell p50 lies on, and compares the cell p50 and the
     fast-mode median with the p50 of the scoring-session runs at the same (2133 MHz, GPU) cell;
  2. fits H1 / H2 (and the contrast fit H1a) again with the released fitter, on the fast-mode
     medians instead of the cell p50s -- nothing else changes -- and
  3. re-scores E5 (memory point fixed at 2133 MHz; Supp. Tables S5 and S6) and E5-B (memory
     point free; main-text Table IV, Supp. Table S8) with the released pick rules on the recorded runs,
     the fits on the fast-mode medians ("fast-mode fits") side by side with the fits on the cell
     p50s ("reported fits"), and
  4. reports what a reader needs to weigh the comparison: the picks admitted with a predicted
     margin D - T-hat*r under 10 us (a pick that close to the rule's boundary can move with the
     optimiser), the contrast fit H1a with the memory point free, and the per-repeat spread of the
     endpoint cells that the two-cell refit is fitted on.

Mode split.  Primary: the two-cluster split of the cell's cycles that minimises the within-cluster
sum of squares (exact, one-dimensional; threshold searched on the central 98 % of the cycles, all
cycles then assigned by the threshold).  Checks: (b) threshold at the midpoint of the 5th and 95th
percentiles; (c) threshold at the midpoint of the widest gap between consecutive sorted cycles
within the 10th-90th percentile range.  The script reports whether the three give the same picks.

    /usr/bin/python3 -B campaign/e5_modesens.py [--model mobilenet] [--md OUT.md] [--json OUT.json] [--tex OUT.tex]
"""
import argparse
import csv
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from han_fit import load_sweep_dir, cells_dict, fit_han, predict, RES          # noqa: E402
from e5_han_baseline import (model_cfg, load_sessions, arm_fits, han_fits, summary, cell,      # noqa: E402
                             pick_2cell, pick_han, r33_miss, T2, R, TAU, R33, DEPLOY_EMC)
import e5b_emcfree as EB                                                        # noqa: E402

EMC_TOK = {665.6: "665", 2133.0: "2133", 3199.0: "3199"}
GPUS = (306, 408, 510, 612, 714, 816, 918, 1020, 1122, 1173)
METHODS = ("twomeans", "midrange", "gap")
NEAR_US = 10.0        # picks admitted with a predicted margin below this are listed


# ------------------------------------------------------------------ data ----
def cell_cycles(model, emc, g):
    p = os.path.join(R33, f"nx_est_emc{EMC_TOK[emc]}_gpu{g}_{model}.csv")
    return np.array([float(r["response_us"]) for r in csv.DictReader(open(p))])


def threshold(v, method):
    s = np.sort(v)
    if method == "twomeans":
        lo, hi = np.percentile(s, [1, 99])
        c = s[(s >= lo) & (s <= hi)]
        n = len(c); c1 = np.cumsum(c); c2 = np.cumsum(c * c)
        best = None
        for i in range(2, n - 1):
            a = c2[i - 1] - c1[i - 1] ** 2 / i
            b = (c2[-1] - c2[i - 1]) - (c1[-1] - c1[i - 1]) ** 2 / (n - i)
            if best is None or a + b < best[0]:
                best = (a + b, i)
        i = best[1]
        return float((c[i - 1] + c[i]) / 2.0)
    if method == "midrange":
        p5, p95 = np.percentile(s, [5, 95])
        return float((p5 + p95) / 2.0)
    if method == "gap":
        lo, hi = np.percentile(s, [10, 90])
        c = s[(s >= lo) & (s <= hi)]
        k = int(np.argmax(np.diff(c)))
        return float((c[k] + c[k + 1]) / 2.0)
    raise ValueError(method)


def split(v, method="twomeans"):
    thr = threshold(v, method)
    fast = v[v < thr]; slow = v[v >= thr]
    p50 = float(np.median(v))
    fm = float(np.median(fast)); sm = float(np.median(slow))
    w = max(float(np.percentile(fast, 90) - np.percentile(fast, 10)), float(np.percentile(slow, 90) - np.percentile(slow, 10)))
    return dict(n=len(v), n_fast=len(fast), n_slow=len(slow), fast=fm, slow=sm, sep=sm - fm, within=w, thr=thr, p50=p50,
                on_slow=abs(p50 - sm) < abs(p50 - fm))


def session_p50(model, g):
    """p50 of every scoring-session run at (2133 MHz, g): (list of run p50s, list of per-run p10-p90 widths)."""
    cfg = model_cfg(model)
    p50s, widths = [], []
    for s in load_sessions(cfg):
        for pair, run in sorted(s["runs"].get(g, {}).items()):
            v = run["v"]
            p50s.append(float(np.median(v))); widths.append(float(np.percentile(v, 90) - np.percentile(v, 10)))
    return p50s, widths


def refit_cell_widths(model):
    """p10-p90 width of every repeat of the endpoint cells behind the two-cell refit
    (results/refit33/nx[_<model>]/emc<E>_g<f>_r<k>.csv, the cells campaign/e5_han_baseline.py arm_fits() uses)."""
    d = os.path.join(RES, "refit33", "nx" if model == "mobilenet" else f"nx_{model}")
    w = []
    for p in sorted(glob.glob(os.path.join(d, "emc*_g*_r*.csv"))):
        if p.endswith(".power.csv"):
            continue
        v = np.array([float(r["response_us"]) for r in csv.DictReader(open(p))])
        w.append(float(np.percentile(v, 90) - np.percentile(v, 10)))
    return w


# ------------------------------------------------------------------ fits ----
def fits_of(d):
    P, train1, hold = han_fits(d, "abs")
    P["H1a"] = fit_han(d, train=None, with_c=False, weight="abs")      # contrast fit (campaign/e5b_contrast.py)
    return P


def errs_at_deploy(P, ref):
    out = {}
    for arm in ("H1", "H1a", "H2"):
        e = [100.0 * abs(predict(P[arm], g, DEPLOY_EMC) - ref[(DEPLOY_EMC, g)]) / ref[(DEPLOY_EMC, g)] for g in GPUS]
        out[arm] = (float(np.median(e)), float(max(e)))
    return out


# -------------------------------------------------------- E5: memory fixed ----
def e5_rows(model, P):
    """The released evaluate() loop (pick rule, scoring cell) for the arms H1, H1a and H2."""
    cfg = model_cfg(model)
    sessions = load_sessions(cfg)
    fits2, fcells, ftag = arm_fits(model)
    indic = lambda f, D_us: r33_miss(model, f, D_us)  # noqa: E731
    rows = []
    for D in cfg["D"]:
        D_us = D * 1000.0
        row = dict(D=D)
        f, ok = pick_2cell(fits2["aware"], D_us, cfg["grid"], cfg["fmax"])
        row["aware"] = cell(sessions, f, ok, D_us, indic)
        for arm in ("H1", "H1a", "H2"):
            f, ok = pick_han(P[arm], D_us, cfg["grid"], cfg["fmax"])
            row[arm] = cell(sessions, f, ok, D_us, indic)
            row[arm]["margin"] = (D_us - predict(P[arm], f, DEPLOY_EMC) * R) if ok else None
        rows.append(row)
    return cfg, rows


def e5_cell(c):
    if not c["feasible"]:
        return "declines"
    if not c["has_data"]:
        return f"{c['f']} (not run)"
    return f"{c['f']} ({c['nviol']}/{c['nruns']})"


# --------------------------------------------------------- E5-B: memory free ----
def e5b_rows(P, runs, live, PW):
    fits2, cells2, src2 = arm_fits("mobilenet")
    rows = []
    for D in EB.DEADLINES:
        D_us = D * 1000.0
        ga, oka = pick_2cell(fits2["aware"], D_us, list(EB.TABLE2_GRID), 1173)
        row = dict(D=D, aware=(ga if oka else None), arms={}, pw_same=True)
        for arm in ("H1", "H1a", "H2"):
            pk = EB.joint_pick(P[arm], PW["H1" if arm == "H1" else "H2"], D_us)
            if arm == "H1a":        # as campaign/e5b_contrast.py: the pick must not depend on which power fit is used
                alt = EB.joint_pick(P[arm], PW["H1"], D_us)
                row["pw_same"] = (pk is None) == (alt is None) and (pk is None or (pk["emc"], pk["g"]) == (alt["emc"], alt["g"]))
            if pk is None:
                row["arms"][arm] = None
                continue
            key = (pk["emc"], pk["g"])
            sc = EB.score(runs.get(key, []), D_us)
            lv = EB.score(live.get(key, []), D_us) if live.get(key) else None
            prem = None        # round-paired energy premium over the aware refit's pick, from the interleaved re-run
            if oka and key != (DEPLOY_EMC, ga) and live.get(key) and live.get((DEPLOY_EMC, ga)):
                la = {r["round"]: r for r in live[(DEPLOY_EMC, ga)]}; lp = {r["round"]: r for r in live[key]}
                common = sorted(set(la) & set(lp))
                if common:
                    pr = EB.paired([lp[k]["E"] for k in common], [la[k]["E"] for k in common])
                    base = float(np.mean([la[k]["E"] for k in common]))
                    prem = dict(pct=100.0 * pr["mean"] / base, lo=100.0 * pr["lo"] / base, hi=100.0 * pr["hi"] / base, n=pr["n"])
            row["arms"][arm] = dict(emc=pk["emc"], g=pk["g"], That_r=pk["That_r"], margin=D_us - pk["That_r"], rec=sc, live=lv, prem=prem)
        rows.append(row)
    return rows


def cellname(x):
    return "declines" if x is None else f"{int(x['emc']) if x['emc'] != 665.6 else '665.6'}/{x['g']}"


def cellv(x):
    """Pick with its runs over tau; a declined deadline has neither."""
    return cellname(x) if x is None else f"{cellname(x)} ({viol(x)})"


def viol(x):
    if x is None:
        return ""
    parts = []
    if x["rec"]:
        parts.append(f"{x['rec']['viol']}/{x['rec']['n']}")
    if x["live"]:
        parts.append(f"live {x['live']['viol']}/{x['live']['n']}")
    return "; ".join(parts) if parts else "no recorded run"


# ------------------------------------------------------------------ main ----
def analyse(model, L):
    d_json = cells_dict(load_sweep_dir(R33, model))
    assert len(d_json) == 30, (model, len(d_json))
    S = {m: {} for m in METHODS}
    for emc in (665.6, 2133.0, 3199.0):
        for g in GPUS:
            v = cell_cycles(model, emc, g)
            assert len(v) == 800, (model, emc, g, len(v))
            assert abs(float(np.median(v)) - d_json[(emc, g)]) < 1.0, (model, emc, g, float(np.median(v)), d_json[(emc, g)])
            for m in METHODS:
                S[m][(emc, g)] = split(v, m)
    cfg = model_cfg(model)
    L.append(f"## {cfg['name']} (sweeps results/nx_validation_r33, {int(cfg['hz'])} Hz, 800 cycles per cell)")
    L.append("")
    prim = S["twomeans"]
    L.append("| EMC (MHz) | mode separation (us) | widest within-mode p10-p90 (us) | cycles in the slow mode | cells with the p50 on the slow mode (GPU MHz) |")
    L.append("|---:|---:|---:|---:|---|")
    for emc in (3199.0, 2133.0, 665.6):
        cs = [prim[(emc, g)] for g in GPUS]
        on = [g for g in GPUS if prim[(emc, g)]["on_slow"]]
        L.append(f"| {emc:g} | {min(c['sep'] for c in cs):.0f}-{max(c['sep'] for c in cs):.0f} | {max(c['within'] for c in cs):.0f} | "
                 f"{min(c['n_slow'] for c in cs)}-{max(c['n_slow'] for c in cs)} of 800 | {len(on)} of 10: {', '.join(map(str, on)) or 'none'} |")
    n_on = sum(1 for c in prim.values() if c["on_slow"])
    L.append("")
    L.append(f"Cells with the p50 on the slow mode: {n_on} of 30. Per cell (EMC/GPU: cycles fast+slow, fast median, slow median, p50): "
             + "; ".join(f"{emc:g}/{g}: {prim[(emc, g)]['n_fast']}+{prim[(emc, g)]['n_slow']}, {prim[(emc, g)]['fast']:.0f}, {prim[(emc, g)]['slow']:.0f}, {prim[(emc, g)]['p50']:.0f}"
                         for emc in (3199.0, 2133.0, 665.6) for g in GPUS) + ".")
    L.append("")
    L.append("Sweep cell against the scoring-session runs at the same (2133 MHz, GPU) cell:")
    L.append("")
    L.append("| GPU (MHz) | session runs | session run p50 (us) | per-run p10-p90 (us) | sweep p50 - session median (us, %) | fast-mode median - session median (us) |")
    L.append("|---:|---:|---:|---:|---:|---:|")
    cmp_rows = []
    for g in GPUS:
        p50s, widths = session_p50(model, g)
        if not p50s:
            continue
        sm = float(np.median(p50s)); c = prim[(DEPLOY_EMC, g)]
        cmp_rows.append(dict(g=g, n=len(p50s), lo=min(p50s), hi=max(p50s), med=sm, d_p50=c["p50"] - sm, d_fast=c["fast"] - sm))
        L.append(f"| {g} | {len(p50s)} | {min(p50s):.0f}-{max(p50s):.0f} | {min(widths):.0f}-{max(widths):.0f} | {c['p50'] - sm:+.0f} ({100 * (c['p50'] - sm) / sm:+.1f} %) | {c['fast'] - sm:+.0f} |")
    L.append("")
    rw = refit_cell_widths(model)
    sw = [float(np.percentile(cell_cycles(model, emc, g), 90) - np.percentile(cell_cycles(model, emc, g), 10)) for emc in (665.6, 2133.0, 3199.0) for g in GPUS]
    L.append(f"Endpoint cells of the two-cell refit ({len(rw)} repeats under results/refit33): per-repeat p10-p90 {min(rw):.0f}-{max(rw):.0f} us; "
             f"the 30 sweep cells above: p10-p90 {min(sw):.0f}-{max(sw):.0f} us.")
    L.append("")

    P0 = fits_of(d_json)
    PF = {m: fits_of({k: S[m][k]["fast"] for k in S[m]}) for m in METHODS}
    e0 = errs_at_deploy(P0, d_json)
    L.append("Fits (released fitter; |error| at 2133 MHz over the ten GPU clocks against the cell p50s, median/max %): "
             + "; ".join(f"{a} {e0[a][0]:.1f}/{e0[a][1]:.1f}" for a in ("H1", "H1a", "H2")) + ". Fast-mode fit coefficients: "
             + "; ".join(f"{a}: A={PF['twomeans'][a]['A']:.1f} alpha={PF['twomeans'][a]['alpha']:.3f} B={PF['twomeans'][a]['B']:.1f} beta={PF['twomeans'][a]['beta']:.3f} C={PF['twomeans'][a]['C']:.1f}"
                         for a in ("H1", "H1a", "H2")) + ".")
    L.append("")

    # ---- E5, memory point fixed ----
    cfg, rows0 = e5_rows(model, P0)
    rowsF = {m: e5_rows(model, PF[m])[1] for m in METHODS}
    L.append(f"### E5, memory point fixed at 2133 MHz ({len(cfg['D'])} deadlines; pick (runs over tau / recorded runs))")
    L.append("")
    L.append("| D (ms) | aware refit | H1 reported | H1 fast-mode | H1a reported | H1a fast-mode | H2 reported | H2 fast-mode |")
    L.append("|---:|---|---|---|---|---|---|---|")
    out = dict(model=model, e5=[], same_picks_all_methods=True)
    for i, r0 in enumerate(rows0):
        rf = rowsF["twomeans"][i]
        L.append(f"| {r0['D']:.1f} | {e5_cell(r0['aware'])} | " + " | ".join(f"{e5_cell(r0[a])} | {e5_cell(rf[a])}" for a in ("H1", "H1a", "H2")) + " |")
        out["e5"].append(dict(D=r0["D"], **{a: dict(rep=e5_cell(r0[a]), fast=e5_cell(rf[a])) for a in ("H1", "H1a", "H2")}))
        for m in METHODS[1:]:
            for a in ("H1", "H1a", "H2"):
                if (rowsF[m][i][a]["f"], rowsF[m][i][a]["feasible"]) != (rf[a]["f"], rf[a]["feasible"]):
                    out["same_picks_all_methods"] = False
    cnt = {}
    for a in ("H1", "H1a", "H2"):
        s0 = summary(rows0, a); sf = summary(rowsF["twomeans"], a)
        cnt[a] = (s0, sf)
        L.append(f"- {a}: admitted {s0['N'] - s0['decl']} -> {sf['N'] - sf['decl']} of {s0['N']}; admitted picks with a run over tau {s0['viol']} -> {sf['viol']}; "
                 f"admitted picks not run in the sessions {s0['na']} -> {sf['na']}; same clock as the aware refit {s0['same']} -> {sf['same']}, higher {s0['higher']} -> {sf['higher']}, lower {s0['lower']} -> {sf['lower']}")
    def near_fixed(rows):
        return [(a, r["D"], r[a]["f"], r[a]["margin"]) for r in rows for a in ("H1", "H1a", "H2")
                if r[a]["feasible"] and r[a]["margin"] < NEAR_US]
    out["e5_near"] = dict(rep=near_fixed(rows0), fast=near_fixed(rowsF["twomeans"]))
    for tag, key in (("reported", "rep"), ("fast-mode", "fast")):
        L.append(f"- {tag} fits, picks admitted with a predicted margin D - T-hat*r under {NEAR_US:g} us: "
                 + ("; ".join(f"{a} {D:.1f} ms at {f} MHz ({m:.1f} us)" for a, D, f, m in out["e5_near"][key]) or "none"))
    out["e5_counts"] = {a: dict(rep=dict(admitted=cnt[a][0]["N"] - cnt[a][0]["decl"], viol=cnt[a][0]["viol"], na=cnt[a][0]["na"], same=cnt[a][0]["same"], higher=cnt[a][0]["higher"], lower=cnt[a][0]["lower"]),
                               fast=dict(admitted=cnt[a][1]["N"] - cnt[a][1]["decl"], viol=cnt[a][1]["viol"], na=cnt[a][1]["na"], same=cnt[a][1]["same"], higher=cnt[a][1]["higher"], lower=cnt[a][1]["lower"]),
                               N=cnt[a][0]["N"]) for a in cnt}
    L.append("")

    # ---- E5-B, memory point free (MobileNetV2 only: the only model with recorded cells off 2133 MHz) ----
    if model == "mobilenet":
        runs = EB.load_recorded(); live = EB.load_live()
        cell_power = {k: float(np.mean([r["E"] * EB.HZ for r in v])) for k, v in runs.items()}
        keys_all = sorted(k for k in cell_power if k[0] in EB.EMC_LEVELS and k[1] in EB.GPU_GRID)
        keys_h1 = [k for k in keys_all if abs(k[0] - DEPLOY_EMC) > 1e-6]
        PW = {"H1": EB.fit_power(cell_power, keys_h1), "H2": EB.fit_power(cell_power, keys_all)}
        b0 = e5b_rows(P0, runs, live, PW)
        bF = {m: e5b_rows(PF[m], runs, live, PW) for m in METHODS}
        L.append("### E5-B, memory point free (nine deadlines; pick EMC/GPU (runs over tau / recorded runs; live = the interleaved re-run))")
        L.append("")
        L.append("| D (ms) | aware refit (2133) | H1 reported | H1 fast-mode | H2 reported | H2 fast-mode |")
        L.append("|---:|---|---|---|---|---|")
        changed = {"H1": [], "H2": []}; out["e5b"] = []
        for i, r0 in enumerate(b0):
            rf = bF["twomeans"][i]
            L.append(f"| {r0['D']:.1f} | {r0['aware'] if r0['aware'] else 'declines'} | " + " | ".join(
                f"{cellv(r0['arms'][a])} | {cellv(rf['arms'][a])}" for a in ("H1", "H2")) + " |")
            rec = dict(D=r0["D"])
            for a in ("H1", "H2"):
                if cellname(r0["arms"][a]) != cellname(rf["arms"][a]):
                    changed[a].append(r0["D"])
                rec[a] = dict(rep=cellname(r0["arms"][a]), fast=cellname(rf["arms"][a]), rep_viol=viol(r0["arms"][a]), fast_viol=viol(rf["arms"][a]))
                for m in METHODS[1:]:
                    if cellname(bF[m][i]["arms"][a]) != cellname(rf["arms"][a]):
                        out["same_picks_all_methods"] = False
            out["e5b"].append(rec)

        def tally(rows, a):
            adm = [r for r in rows if r["arms"][a] is not None]
            return dict(admitted=len(adm), at3199=sum(1 for r in adm if abs(r["arms"][a]["emc"] - 3199.0) < 1e-6),
                        escapes=sum(1 for r in adm if abs(r["arms"][a]["emc"] - 3199.0) < 1e-6 and r["aware"]),
                        viol_picks=sum(1 for r in adm if (r["arms"][a]["rec"] and r["arms"][a]["rec"]["viol"] > 0) or (r["arms"][a]["live"] and r["arms"][a]["live"]["viol"] > 0)),
                        unrecorded=sum(1 for r in adm if not r["arms"][a]["rec"] and not r["arms"][a]["live"]))
        def premiums(rows, a):
            v = [(r["D"], r["arms"][a]["prem"]) for r in rows if r["arms"][a] is not None and abs(r["arms"][a]["emc"] - 3199.0) < 1e-6 and r["arms"][a]["prem"]]
            return v
        out["e5b_counts"] = {}
        for a in ("H1", "H2"):
            for tag, rows in (("reported", b0), ("fast-mode", bF["twomeans"])):
                pv = premiums(rows, a)
                if pv:
                    L.append(f"- {a} {tag}: round-paired energy premium of the 3199 MHz picks over the aware refit's pick (interleaved re-run): "
                             + "; ".join(f"{D:.1f} ms {x['pct']:+.1f} % [{x['lo']:+.1f}, {x['hi']:+.1f}] (n={x['n']})" for D, x in pv))
            out.setdefault("e5b_premium", {})[a] = dict(rep=[(D, x) for D, x in premiums(b0, a)], fast=[(D, x) for D, x in premiums(bF["twomeans"], a)])
        for a in ("H1", "H2"):
            t0, tf = tally(b0, a), tally(bF["twomeans"], a)
            out["e5b_counts"][a] = dict(rep=t0, fast=tf, changed=changed[a])
            L.append(f"- {a}: admitted {t0['admitted']} -> {tf['admitted']} of 9; picks at 3199 MHz {t0['at3199']} -> {tf['at3199']} "
                     f"(where the aware refit admits a 2133 MHz clock: {t0['escapes']} -> {tf['escapes']}); picks with a recorded or live run over tau {t0['viol_picks']} -> {tf['viol_picks']}; "
                     f"picks without a recorded run {t0['unrecorded']} -> {tf['unrecorded']}; cells that change: {len(changed[a])} ({', '.join(f'{D:.1f}' for D in changed[a]) or 'none'} ms)")
        L.append(f"- cells that change: {len(changed['H1']) + len(changed['H2'])} of 18")

        def near_free(rows):
            return [(a, r["D"], cellname(r["arms"][a]), r["arms"][a]["margin"]) for r in rows for a in ("H1", "H1a", "H2")
                    if r["arms"][a] is not None and r["arms"][a]["margin"] < NEAR_US]
        out["e5b_near"] = dict(rep=near_free(b0), fast=near_free(bF["twomeans"]))
        for tag, key in (("reported", "rep"), ("fast-mode", "fast")):
            L.append(f"- {tag} fits, picks admitted with a predicted margin D - T-hat*r under {NEAR_US:g} us: "
                     + ("; ".join(f"{a} {D:.1f} ms at {c} MHz ({m:.1f} us)" for a, D, c, m in out["e5b_near"][key]) or "none"))
        # the contrast fit H1a (pure form, deployment point in-sample), not part of Table IV
        t0, tf = tally(b0, "H1a"), tally(bF["twomeans"], "H1a")
        ch = [r0["D"] for r0, rf in zip(b0, bF["twomeans"]) if cellname(r0["arms"]["H1a"]) != cellname(rf["arms"]["H1a"])]
        for m in METHODS[1:]:
            for i, rf in enumerate(bF["twomeans"]):
                if cellname(bF[m][i]["arms"]["H1a"]) != cellname(rf["arms"]["H1a"]):
                    out["same_picks_all_methods"] = False
        pw_ok = all(r["pw_same"] for r in b0) and all(r["pw_same"] for r in bF["twomeans"])

        def shared(rows):        # deadlines at which H1a and H2 both pick a 3199 MHz cell although the aware refit admits a 2133 MHz clock
            return [r["D"] for r in rows if r["aware"] and all(r["arms"][a] is not None and abs(r["arms"][a]["emc"] - 3199.0) < 1e-6 for a in ("H1a", "H2"))]
        out["e5b_h1a"] = dict(rep=[cellname(r["arms"]["H1a"]) for r in b0], fast=[cellname(r["arms"]["H1a"]) for r in bF["twomeans"]],
                              rep_viol=[viol(r["arms"]["H1a"]) for r in b0], fast_viol=[viol(r["arms"]["H1a"]) for r in bF["twomeans"]],
                              counts=dict(rep=t0, fast=tf), changed=ch, shared_3199=dict(rep=shared(b0), fast=shared(bF["twomeans"])), pw_same=pw_ok)
        L.append("- H1a (contrast fit: pure form, deployment point in-sample; memory point free, pick reported -> fast-mode (runs over tau / recorded runs)): "
                 + "; ".join(f"{r0['D']:.1f} ms {cellv(r0['arms']['H1a'])} -> {cellv(rf['arms']['H1a'])}"
                             for r0, rf in zip(b0, bF["twomeans"])))
        L.append(f"- H1a: admitted {t0['admitted']} -> {tf['admitted']} of 9; picks at 3199 MHz where the aware refit admits a 2133 MHz clock {t0['escapes']} -> {tf['escapes']}; "
                 f"picks with a recorded or live run over tau {t0['viol_picks']} -> {tf['viol_picks']}; picks without a recorded run {t0['unrecorded']} -> {tf['unrecorded']}; "
                 f"cells that change: {len(ch)} ({', '.join(f'{D:.1f}' for D in ch) or 'none'} ms); same pick under either power fit: {'yes' if pw_ok else 'NO'}")
        for tag, rows in (("reported", b0), ("fast-mode", bF["twomeans"])):
            pv = premiums(rows, "H1a")
            if pv:
                L.append(f"- H1a {tag}: round-paired energy premium of the 3199 MHz picks over the aware refit's pick (interleaved re-run): "
                         + "; ".join(f"{D:.1f} ms {x['pct']:+.1f} % [{x['lo']:+.1f}, {x['hi']:+.1f}] (n={x['n']})" for D, x in pv))
        out["e5b_premium"]["H1a"] = dict(rep=[(D, x) for D, x in premiums(b0, "H1a")], fast=[(D, x) for D, x in premiums(bF["twomeans"], "H1a")])
        L.append(f"- deadlines at which H1a and H2 both pick a 3199 MHz cell although the aware refit admits a 2133 MHz clock: "
                 f"{', '.join(f'{D:.1f}' for D in out['e5b_h1a']['shared_3199']['rep']) or 'none'} ms -> "
                 f"{', '.join(f'{D:.1f}' for D in out['e5b_h1a']['shared_3199']['fast']) or 'none'} ms")
        L.append("")
    L.append(f"Same picks under the three split methods ({', '.join(METHODS)}): {'yes' if out['same_picks_all_methods'] else 'NO'}.")
    L.append("")
    out["refit_cell_p10p90"] = rw
    out["sweep_cell_p10p90"] = sw
    out["cells"] = {f"{emc:g}/{g}": {k: (float(x) if not isinstance(x, (bool, np.bool_)) else bool(x)) for k, x in prim[(emc, g)].items()} for (emc, g) in prim}
    out["cmp"] = cmp_rows
    out["n_on_slow"] = n_on
    return out


def tex_table(res, path):
    """Compact supplement table for MobileNetV2: E5 and E5-B picks, the reported fits against the fast-mode fits."""
    e5 = res["e5"]; e5b = res["e5b"]
    T = [r"  \begin{tabular}{@{}r ll ll ll ll@{}}", r"    \toprule",
         r"    & \multicolumn{4}{c}{memory point fixed at 2133\,MHz} & \multicolumn{4}{c}{memory point free} \\",
         r"    \cmidrule(lr){2-5}\cmidrule(lr){6-9}",
         r"    & \multicolumn{2}{c}{H1} & \multicolumn{2}{c}{H2} & \multicolumn{2}{c}{H1} & \multicolumn{2}{c}{H2} \\",
         r"    \cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}",
         r"    $D$ (ms) & cell p50 & fast mode & cell p50 & fast mode & cell p50 & fast mode & cell p50 & fast mode \\", r"    \midrule"]
    for a, b in zip(e5, e5b):
        def fx(x):
            return x.split(" (")[0].replace("declines", "decl.")
        def mark(rep, fast):
            return fast if rep == fast else r"\textbf{" + fast + "}"
        cells = []
        for arm in ("H1", "H2"):
            cells += [fx(a[arm]["rep"]), mark(fx(a[arm]["rep"]), fx(a[arm]["fast"]))]
        for arm in ("H1", "H2"):
            cells += [fx(b[arm]["rep"]), mark(fx(b[arm]["rep"]), fx(b[arm]["fast"]))]
        T.append(f"    {a['D']:.1f} & " + " & ".join(cells) + r" \\")
    T += [r"    \bottomrule", r"  \end{tabular}"]
    open(path, "w").write("\n".join(T) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="mobilenet,vit,yolov8s,resnet50")
    ap.add_argument("--md"); ap.add_argument("--json"); ap.add_argument("--tex")
    a = ap.parse_args()
    L = ["# E5 / E5-B post-hoc sensitivity to the tail state of the training sweeps", "",
         f"Rule r = {R}, tau = {TAU:g} %; fits and pick rules from campaign/han_fit.py, e5_han_baseline.py and e5b_emcfree.py, unchanged. "
         "'reported' = fits on the cell p50s (the reported tables); 'fast-mode' = the same fits on the fast-mode medians.", ""]
    res = {}
    for model in a.model.split(","):
        res[model] = analyse(model, L)
    md = "\n".join(L) + "\n"
    print(md)
    if a.md:
        open(a.md, "w").write(md)
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1, default=float)
    if a.tex:
        if "mobilenet" not in res:
            raise SystemExit("--tex writes the MobileNetV2 table; include mobilenet in --model")
        tex_table(res["mobilenet"], a.tex)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""E5-B: EMC-free replay of the Han et al. joint scaling policy on the recorded NX MobileNetV2 cells.

Protocol: docs/protocols/E1_E2_protocol.md §E5-B (pre-stated 2026-10-28 before scoring) and
§E5-B-live (the interleaved re-run of the 3199 MHz picks).  Read-only over results/.  The
markdown record goes to stdout; --md also writes it to a file, --tex writes the LaTeX table and
--json the JSON that campaign/e5b_live_table.py formats.

    /usr/bin/python3 campaign/e5b_emcfree.py [--md OUT.md] [--tex OUT.tex] [--json OUT.json]

Policy (Han et al., arXiv 2509.17970, Sec. IV-A "joint scaling policy", no controller): among the
candidate cells (f_emc, f_gpu) with T_hat(f_gpu, f_emc)*r <= D choose the one with the lowest
predicted energy per period; the latency model is E5's H1 / H2 (campaign/han_fit.py on
results/nx_validation_r33), the power model is Han's Eq. (2) p = k_mem f_mem^3 + k_com f_com^3 + sigma
fitted (non-negative least squares) to the per-cell mean of the trimmed-mean VDD_IN power of the
recorded runs (H1-power: the 14 cells at EMC {665.6, 3199}; H2-power: all 21 cells).  Energy per
period = power / 33 Hz, the Table II E/inf metric (campaign/e1b_stats.energy convention).  Han's
e = p*t (busy energy) is not derivable from the 3.6 ms INA logs and is not evaluated.
Candidate cells: EMC {665.6, 2133, 3199} x GPU {612..1173} (exclusions: protocol).  Recorded runs:
EMC 2133 -> the three E1 sessions results/govrep_rev/{nx, nx_mobilenet, nx_mobilenet_s3};
EMC 3199 / 665.6 -> results/nx_pairs (2026-07-03, 3 runs per cell); live re-run (if present) ->
results/e5b_live/nx (E5-B-live).  The aware arm = Table II's per-EMC two-cell refit at 2133 MHz.
"""
import argparse
import csv
import glob
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LAB = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from han_fit import load_sweep_dir, cells_dict, fit_han, predict, RES          # noqa: E402
from e5_han_baseline import (arm_fits, pick_2cell, han_fits, T2, R, TAU, R33,  # noqa: E402
                             DEPLOY_EMC)
from e1b_stats import GRID as TABLE2_GRID, tcrit  # noqa: E402
try:
    from scipy.optimize import nnls as _nnls
except Exception:  # pragma: no cover
    _nnls = None

HZ = 33.0
EMC_LEVELS = (665.6, 2133.0, 3199.0)
GPU_GRID = (612, 714, 816, 918, 1020, 1122, 1173)
DEADLINES = [4.4, 4.6, 4.8, 5.0, 5.2, 5.4, 5.6, 5.8, 6.0]
E1_SESSIONS = ("nx", "nx_mobilenet", "nx_mobilenet_s3")
EMC_TOK = {"665": 665.6, "2133": 2133.0, "3199": 3199.0}
PUBLISHED_ERR = {"H1": (6.6, 15.1), "H2": (2.3, 6.3)}   # supplementary tab:e5han, MobileNetV2 row
T975_9 = 2.262   # paired-t, 9 d.o.f.


# ------------------------------------------------------------------ runs ----
def _read_run(stem, hz_expected=HZ):
    v = np.array([float(r["response_us"]) for r in csv.DictReader(open(stem + ".csv"))])
    meta = json.load(open(stem + ".json"))["meta"]
    assert abs(float(meta["hz"]) - hz_expected) < 1e-6, (stem, meta["hz"])
    assert len(v) == 1000, (stem, len(v))
    p = np.array([float(r["vdd_in_mw"]) for r in csv.DictReader(open(stem + ".power.csv")) if r.get("vdd_in_mw")])
    n = len(p)
    assert n >= 20, stem
    e_inf = float(p[n // 10:-n // 10].mean()) / hz_expected      # mJ per inference (Table II convention)
    return dict(stem=stem, v=v, p50=float(np.median(v)), E=e_inf, ts=meta.get("timestamp", ""),
                warmup=meta.get("warmup"), host=meta.get("host"))


def load_recorded():
    """{(emc, gpu): [run, ...]} for the pre-stated replay sources; also the per-session index."""
    runs = {}
    for name in E1_SESSIONS:
        for csvp in sorted(glob.glob(os.path.join(RES, "govrep_rev", name, "g*_p*.csv"))):
            if csvp.endswith(".power.csv"):
                continue
            g = int(os.path.basename(csvp)[:-4].split("_p")[0][1:])
            r = _read_run(csvp[:-4]); r["src"] = name
            runs.setdefault((DEPLOY_EMC, g), []).append(r)
    for csvp in sorted(glob.glob(os.path.join(RES, "nx_pairs", "e*_g*_r*.csv"))):
        if csvp.endswith(".power.csv"):
            continue
        m = re.match(r"e(\d+)_g(\d+)_r(\d+)\.csv$", os.path.basename(csvp))
        if not m or m.group(1) not in EMC_TOK:
            continue
        r = _read_run(csvp[:-4]); r["src"] = "nx_pairs"; r["round"] = int(m.group(3))
        runs.setdefault((EMC_TOK[m.group(1)], int(m.group(2))), []).append(r)
    return runs


def load_live():
    live = {}
    for csvp in sorted(glob.glob(os.path.join(RES, "e5b_live", "nx", "e*_g*_r*.csv"))):
        if csvp.endswith(".power.csv"):
            continue
        m = re.match(r"e(\d+)_g(\d+)_r(\d+)\.csv$", os.path.basename(csvp))
        if (not m or m.group(1) not in EMC_TOK or not os.path.isfile(csvp[:-4] + ".json")
                or not os.path.isfile(csvp[:-4] + ".power.csv")):
            continue
        r = _read_run(csvp[:-4]); r["src"] = "e5b_live"; r["round"] = int(m.group(3))
        live.setdefault((EMC_TOK[m.group(1)], int(m.group(2))), []).append(r)
    return live


# ---------------------------------------------------------------- models ----
def han_latency():
    d = cells_dict(load_sweep_dir(R33, "mobilenet"))
    P, train1, hold = han_fits(d, "abs")
    # self-check against the published held-out / in-sample errors at 2133 MHz over the ten GPU clocks
    errs = {}
    for arm in ("H1", "H2"):
        e = [100.0 * abs(predict(P[arm], g, DEPLOY_EMC) - d[(DEPLOY_EMC, g)]) / d[(DEPLOY_EMC, g)]
             for (emc, g) in sorted(d) if abs(emc - DEPLOY_EMC) < 1e-6]
        errs[arm] = (float(np.median(e)), float(max(e)))
        pm, px = PUBLISHED_ERR[arm]
        assert abs(errs[arm][0] - pm) < 0.06 and abs(errs[arm][1] - px) < 0.06, (arm, errs[arm], PUBLISHED_ERR[arm])
    return P, errs, d


def fit_power(cell_power, keys):
    """Han Eq. (2): p = k_mem f_mem^3 + k_com f_com^3 + sigma [mW], f in GHz, all coefficients >= 0."""
    X = np.array([[(k[0] / 1000.0) ** 3, (k[1] / 1000.0) ** 3, 1.0] for k in keys])
    y = np.array([cell_power[k] for k in keys])
    if _nnls is not None:
        coef, _ = _nnls(X, y)
    else:  # pragma: no cover
        coef = np.clip(np.linalg.lstsq(X, y, rcond=None)[0], 0.0, None)
    pred = X @ coef
    err = 100.0 * np.abs(pred - y) / y
    return dict(k_mem=float(coef[0]), k_com=float(coef[1]), sigma=float(coef[2]), n=len(keys),
                max_err_pct=float(err.max()), med_err_pct=float(np.median(err)))


def pred_power(pw, emc, g):
    return pw["k_mem"] * (emc / 1000.0) ** 3 + pw["k_com"] * (g / 1000.0) ** 3 + pw["sigma"]


def pred_e_inf(pw, emc, g):
    return pred_power(pw, emc, g) / HZ


# ------------------------------------------------------------------ rule ----
def joint_pick(P_lat, pw, D_us):
    cands = []
    for emc in EMC_LEVELS:
        for g in GPU_GRID:
            t = predict(P_lat, g, emc) * R
            if t <= D_us:
                cands.append((pred_e_inf(pw, emc, g), g, emc, t))
    if not cands:
        return None
    e, g, emc, t = sorted(cands)[0]
    return dict(emc=emc, g=g, pred_E=e, That_r=t)


def score(runs, D_us):
    if not runs:
        return None
    miss = [100.0 * float(np.count_nonzero(r["v"] > D_us)) / len(r["v"]) for r in runs]
    viol = sum(1 for m in miss if m > TAU)
    pooled = 100.0 * sum(int(np.count_nonzero(r["v"] > D_us)) for r in runs) / sum(len(r["v"]) for r in runs)
    E = [r["E"] for r in runs]
    return dict(n=len(runs), viol=viol, pooled=pooled, max_miss=max(miss), E=float(np.mean(E)),
                E_min=float(min(E)), E_max=float(max(E)), p50=float(np.median([r["p50"] for r in runs])))


def frontier(runs, D_us):
    """Lowest-energy candidate cell with no recorded run over tau at this deadline."""
    best = None
    for emc in EMC_LEVELS:
        for g in GPU_GRID:
            s = score(runs.get((emc, g), []), D_us)
            if s and s["viol"] == 0 and (best is None or s["E"] < best[0]):
                best = (s["E"], emc, g, s["n"])
    return None if best is None else dict(E=best[0], emc=best[1], g=best[2], n=best[3])


# ------------------------------------------------------------------ main ----
def paired(xs, ys):
    d = np.array(xs) - np.array(ys)
    m = float(d.mean()); sd = float(d.std(ddof=1)) if len(d) > 1 else float("nan")
    h = tcrit(len(d) - 1) * sd / np.sqrt(len(d)) if len(d) > 1 else float("nan")
    return dict(mean=m, lo=m - h, hi=m + h, n=len(d))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--md"); ap.add_argument("--tex"); ap.add_argument("--json")
    args = ap.parse_args()

    P, errs, d_lat = han_latency()
    runs = load_recorded(); live = load_live()
    cell_power = {k: float(np.mean([r["E"] * HZ for r in v])) for k, v in runs.items()}
    keys_all = sorted(k for k in cell_power if k[0] in EMC_LEVELS and k[1] in GPU_GRID)
    keys_h1 = [k for k in keys_all if abs(k[0] - DEPLOY_EMC) > 1e-6]
    PW = {"H1": fit_power(cell_power, keys_h1), "H2": fit_power(cell_power, keys_all)}
    fits2, cells2, src2 = arm_fits("mobilenet")

    rows = []
    for D in DEADLINES:
        D_us = D * 1000.0
        ga, oka = pick_2cell(fits2["aware"], D_us, list(TABLE2_GRID), 1173)   # Table II rule, grid 510-1173
        aware = dict(g=ga if oka else None, feasible=oka, That_r=T2(fits2["aware"], ga) if oka else None,
                     score=score(runs.get((DEPLOY_EMC, ga), []), D_us) if oka else None,
                     live=score(live.get((DEPLOY_EMC, ga), []), D_us) if (oka and live.get((DEPLOY_EMC, ga))) else None)
        fr = frontier(runs, D_us)
        row = dict(D=D, aware=aware, frontier=fr, arms={})
        for arm in ("H1", "H2"):
            pk = joint_pick(P[arm], PW[arm], D_us)
            if pk is None:
                row["arms"][arm] = dict(feasible=False)
                continue
            sc = score(runs.get((pk["emc"], pk["g"]), []), D_us)
            dE = (100.0 * (sc["E"] - aware["score"]["E"]) / aware["score"]["E"]) if (sc and aware["score"]) else None
            lv = live.get((pk["emc"], pk["g"]), [])
            lsc = score(lv, D_us) if lv else None
            lpair = None
            if lv and aware["feasible"] and live.get((DEPLOY_EMC, ga)):
                la = {r["round"]: r for r in live[(DEPLOY_EMC, ga)]}; lp = {r["round"]: r for r in lv}
                common = sorted(set(la) & set(lp))
                if common:
                    pr = paired([lp[k]["E"] for k in common], [la[k]["E"] for k in common])
                    pr["aware_E"] = float(np.mean([la[k]["E"] for k in common]))
                    pr["pct"] = 100.0 * pr["mean"] / pr["aware_E"]
                    pr["lo_pct"] = 100.0 * pr["lo"] / pr["aware_E"]; pr["hi_pct"] = 100.0 * pr["hi"] / pr["aware_E"]
                    pr["aware_viol"] = sum(1 for k in common if 100.0 * np.count_nonzero(la[k]["v"] > D_us) / 1000 > TAU)
                    lpair = pr
            row["arms"][arm] = dict(feasible=True, **pk, score=sc, dE_pct=dE,
                                    same_as_aware=(aware["feasible"] and abs(pk["emc"] - DEPLOY_EMC) < 1e-6 and pk["g"] == ga),
                                    is_frontier=(fr is not None and abs(pk["emc"] - fr["emc"]) < 1e-6 and pk["g"] == fr["g"]),
                                    live=lsc, live_paired=lpair)
        rows.append(row)

    summ = {}
    for arm in ("H1", "H2"):
        adm = [r for r in rows if r["arms"][arm]["feasible"]]
        summ[arm] = dict(admitted=len(adm), declined=len(rows) - len(adm),
                         viol_picks=sum(1 for r in adm if r["arms"][arm]["score"] and r["arms"][arm]["score"]["viol"] > 0),
                         at_3199=sum(1 for r in adm if abs(r["arms"][arm]["emc"] - 3199.0) < 1e-6),
                         at_665=sum(1 for r in adm if abs(r["arms"][arm]["emc"] - 665.6) < 1e-6),
                         same_as_aware=sum(1 for r in adm if r["arms"][arm]["same_as_aware"]),
                         frontier_hits=sum(1 for r in adm if r["arms"][arm]["is_frontier"]),
                         escapes_3199=sum(1 for r in adm if abs(r["arms"][arm]["emc"] - 3199.0) < 1e-6 and r["aware"]["feasible"]),
                         dE_range=([min(v) for v in [[r["arms"][arm]["dE_pct"] for r in adm if r["arms"][arm]["dE_pct"] is not None]] if v] + [max(v) for v in [[r["arms"][arm]["dE_pct"] for r in adm if r["arms"][arm]["dE_pct"] is not None]] if v]) or None,
                         live_viol_picks=sum(1 for r in adm if r["arms"][arm]["live"] and r["arms"][arm]["live"]["viol"] > 0),
                         live_cells=sum(1 for r in adm if r["arms"][arm]["live"]))
    aware_viol = sum(1 for r in rows if r["aware"]["score"] and r["aware"]["score"]["viol"] > 0)
    fr_only_emcfree = [r["D"] for r in rows if r["frontier"] and abs(r["frontier"]["emc"] - DEPLOY_EMC) > 1e-6]

    # live-vs-July comparison per 3199 cell
    live_cmp = []
    for (emc, g), lv in sorted(live.items()):
        jul = runs.get((emc, g), [])
        if emc == DEPLOY_EMC or not jul:
            continue
        live_cmp.append(dict(emc=emc, g=g, n_live=len(lv), p50_live=float(np.median([r["p50"] for r in lv])),
                             p50_july=float(np.median([r["p50"] for r in jul])),
                             E_live=float(np.mean([r["E"] for r in lv])), E_july=float(np.mean([r["E"] for r in jul]))))

    def cellname(emc, g):
        return f"{int(emc) if emc != 665.6 else '665.6'}/{g}"

    L = ["# E5-B: EMC-free replay of the Han joint scaling policy (NX MobileNetV2, 33 Hz)", "",
         f"Latency fits (han_fit, results/nx_validation_r33): H1 held-out |err| at 2133 MHz median/max = {errs['H1'][0]:.1f}/{errs['H1'][1]:.1f} %, "
         f"H2 in-sample {errs['H2'][0]:.1f}/{errs['H2'][1]:.1f} % (published 6.6/15.1 and 2.3/6.3). Aware arm: {src2}.",
         f"Power model (Han Eq. 2, nnls on per-cell mean trimmed VDD_IN): H1-power on {PW['H1']['n']} cells {{665.6, 3199}}: "
         f"k_mem={PW['H1']['k_mem']:.1f} k_com={PW['H1']['k_com']:.1f} sigma={PW['H1']['sigma']:.0f} mW (max err {PW['H1']['max_err_pct']:.1f} %, "
         f"2133 MHz predicted vs measured: " + ", ".join(f"{g}: {pred_power(PW['H1'], DEPLOY_EMC, g):.0f}/{cell_power[(DEPLOY_EMC, g)]:.0f}" for g in GPU_GRID) + "); "
         f"H2-power on {PW['H2']['n']} cells: k_mem={PW['H2']['k_mem']:.1f} k_com={PW['H2']['k_com']:.1f} sigma={PW['H2']['sigma']:.0f} mW (max err {PW['H2']['max_err_pct']:.1f} %).",
         "Measured E/inf (mJ, mean over recorded runs; n): " + "; ".join(
             f"{cellname(emc, g)} {cell_power[(emc, g)] / HZ:.1f} (n={len(runs[(emc, g)])})" for emc in EMC_LEVELS for g in GPU_GRID if (emc, g) in runs), "",
         "ΔE in the table = unpaired difference of mean E/inf versus the aware pick (different sessions where the EMC differs); 'paired' = E5-B-live, by round.", "",
         "| D (ms) | aware pick: g; runs>τ/n; E [live runs>τ/n] | H1 pick: emc/g; T̂r; runs>τ/n; pooled; E; ΔE | H2 pick: emc/g; T̂r; runs>τ/n; pooled; E; ΔE | EMC-free frontier: emc/g; E (n) |",
         "|---:|---|---|---|---|"]
    for r in rows:
        a = r["aware"]
        acell = (f"{a['g']}; {a['score']['viol']}/{a['score']['n']}; {a['score']['E']:.1f}" + (f" [live {a['live']['viol']}/{a['live']['n']}, {a['live']['E']:.1f} mJ]" if a.get("live") else "") if a["feasible"] else "declines")
        parts = []
        for arm in ("H1", "H2"):
            x = r["arms"][arm]
            if not x["feasible"]:
                parts.append("declines"); continue
            s = x["score"]
            if s is None:
                parts.append(f"{cellname(x['emc'], x['g'])}; {x['That_r']/1000:.2f}; no recorded runs"); continue
            parts.append(f"{cellname(x['emc'], x['g'])}; {x['That_r']/1000:.2f}; {s['viol']}/{s['n']}; {s['pooled']:.2f} %; {s['E']:.1f}; "
                         + (f"{x['dE_pct']:+.1f} %" if x["dE_pct"] is not None else "—")
                         + (f"; live {x['live']['viol']}/{x['live']['n']}, {x['live']['E']:.1f} mJ" if x["live"] else "")
                         + (f", paired ΔE {x['live_paired']['pct']:+.1f} % [{x['live_paired']['lo_pct']:+.1f}, {x['live_paired']['hi_pct']:+.1f}] (n={x['live_paired']['n']}; aware live {x['live_paired']['aware_viol']}/{x['live_paired']['n']} runs>τ)" if x["live_paired"] else ""))
        fr = r["frontier"]
        L.append(f"| {r['D']:.1f} | {acell} | {parts[0]} | {parts[1]} | " + (f"{cellname(fr['emc'], fr['g'])}; {fr['E']:.1f} ({fr['n']})" if fr else "none") + " |")
    L += ["", "Summary: aware violating picks " + str(aware_viol) + "; EMC-free frontier sits at a non-2133 cell at D = " + (", ".join(f"{D:.1f}" for D in fr_only_emcfree) or "none") + "."]
    for r in rows:
        fr = r["frontier"]
        if fr and live.get((fr["emc"], fr["g"])):
            ls = score(live[(fr["emc"], fr["g"])], r["D"] * 1000.0)
            L.append(f"- frontier cell {cellname(fr['emc'], fr['g'])} at D = {r['D']:.1f} ms, live: {ls['viol']}/{ls['n']} runs>τ, max run {ls['max_miss']:.1f} %, pooled {ls['pooled']:.2f} %, E {ls['E']:.1f} mJ")
    for arm in ("H1", "H2"):
        s = summ[arm]
        L.append(f"- {arm}: admitted {s['admitted']}/9, declined {s['declined']}; picks with ≥1 violating run {s['viol_picks']}; "
                 f"picks at 3199 MHz {s['at_3199']} (escapes where aware admits 2133 MHz: {s['escapes_3199']}), at 665.6 MHz {s['at_665']}; same cell as aware {s['same_as_aware']}; frontier hits {s['frontier_hits']}; "
                 + (f"ΔE vs aware (unpaired) {s['dE_range'][0]:+.1f}..{s['dE_range'][1]:+.1f} %" if s['dE_range'] else "ΔE n/a") + (f"; live cells {s['live_cells']}, live violating picks {s['live_viol_picks']}" if s["live_cells"] else ""))
    if live_cmp:
        L += ["", "Live (E5-B-live) vs July cells at 3199 MHz: " + "; ".join(
            f"{cellname(c['emc'], c['g'])}: p50 {c['p50_live']:.0f} vs {c['p50_july']:.0f} µs, E {c['E_live']:.1f} vs {c['E_july']:.1f} mJ (n_live={c['n_live']})" for c in live_cmp)]
    L += ["", "Han's e = p·t (busy energy) not evaluated: the 3.6 ms INA samples do not resolve the 4–6 ms busy windows at 33 Hz."]
    md = "\n".join(L) + "\n"
    print(md)
    if args.md:
        open(args.md, "w").write(md)
    if args.json:
        json.dump(dict(errs=errs, power=PW, rows=rows, summary=summ, live_cmp=live_cmp, cell_E=
                       {cellname(k[0], k[1]): dict(E=cell_power[k] / HZ, n=len(runs[k])) for k in keys_all}),
                  open(args.json, "w"), indent=1, default=float)
    if args.tex:
        def tcell(arm, r):
            x = r["arms"][arm]
            if not x["feasible"]:
                return "declines & & & "
            s = x["score"]; lv = x["live"]
            viol = f"{s['viol']}/{s['n']}" + (f" ({lv['viol']}/{lv['n']})" if lv else "")
            return f"{cellname(x['emc'], x['g'])} & {viol} & {s['E']:.0f} & " + (f"${x['dE_pct']:+.1f}$" if x["dE_pct"] is not None else "--")
        T = [r"\begin{table}[t]", r"  \centering",
             r"  \caption{EMC-free replay of the Han-derived joint-scaling selector adapted to the energy-per-period objective (E5-B; protocol, disclosure and re-run in \S\ref{sec:e5bprot}): the E5 latency fits (H1: the twenty rate-matched sweep cells at the profiling levels $\{665.6, 3199\}$\,MHz, deployment point held out, pure form ($C{=}0$); H2: all thirty, with the free offset $C$) with Han et al.'s cubic power model fitted on the recorded cells (H1-power on the 14 at the profiling levels, H2-power on all 21), choosing the $(f_{\mathrm{emc}}, f_{\mathrm{gpu}})$ cell of lowest predicted energy per period at 33\,Hz among the cells whose $\hat T\,r \le D$ ($r{=}1.035$; ties toward the lower GPU clock, then the lower EMC; candidates EMC $\{665.6, 2133, 3199\}$\,MHz $\times$ GPU 612--1173\,MHz); the per-EMC refit it is compared with uses two cells at the deployment point. Each pick is scored on the recorded runs at its cell: EMC 2133\,MHz from the three interleaved sessions of Table~II, 3199 and 665.6\,MHz from the July 2026 cells (three runs each, warm-up 60 cycles, no readback file); parenthesised counts are from the interleaved, readback-verified re-run of the seven cells the selector and the refit pick at 4.6--5.2\,ms (E5-B-live, ten rounds, EMC re-locked and read back before every cell, the $\le 55\,^\circ$C gate timing out before rounds 2--10, end-of-cell readings 53.2--58.1\,$^\circ$C over the ten rounds, Table~\ref{tab:thermal}) and appear wherever a pick falls on one of those cells. E = mean energy per inference (mJ) over the recorded runs; $\Delta E$ = unpaired difference versus the aware arm's pick (different sessions where the EMC differs; the paired, same-session premiums of the re-run are in Table~\ref{tab:e5blive}); frontier = the lowest-energy candidate cell with no recorded run over $\tau$. Han's own $e = p\,t$ objective is not derivable from the power logs and is not evaluated.}",
             r"  \label{tab:e5bemcfree}", r"  \footnotesize", r"  \setlength{\tabcolsep}{3pt}", r"  \resizebox{\columnwidth}{!}{%",
             r"  \begin{tabular}{@{}rrrr rrrr rrrr rr@{}}", r"    \toprule",
             r"    & \multicolumn{3}{c}{aware (2133\,MHz)} & \multicolumn{4}{c}{Han-H1, EMC free} & \multicolumn{4}{c}{Han-H2, EMC free} & \multicolumn{2}{c}{frontier} \\",
             r"    \cmidrule(lr){2-4}\cmidrule(lr){5-8}\cmidrule(lr){9-12}\cmidrule(lr){13-14}",
             r"    $D$ (ms) & $f$ & runs${>}\tau$ & E & emc/$f$ & runs${>}\tau$ & E & $\Delta E$\,\% & emc/$f$ & runs${>}\tau$ & E & $\Delta E$\,\% & emc/$f$ & E \\",
             r"    \midrule"]
        for r in rows:
            a_ = r["aware"]
            ac = (f"{a_['g']} & {a_['score']['viol']}/{a_['score']['n']}" + (f" ({a_['live']['viol']}/{a_['live']['n']})" if a_.get("live") else "") + f" & {a_['score']['E']:.0f}" if a_["feasible"] else "declines & & ")
            fr = r["frontier"]
            fc = (f"{cellname(fr['emc'], fr['g'])} & {fr['E']:.0f}" if fr else "none & ")
            T.append(f"    {r['D']:.1f} & {ac} & {tcell('H1', r)} & {tcell('H2', r)} & {fc} \\\\")
        T += [r"    \bottomrule", r"  \end{tabular}}", r"\end{table}"]
        open(args.tex, "w").write("\n".join(T) + "\n")


if __name__ == "__main__":
    main()

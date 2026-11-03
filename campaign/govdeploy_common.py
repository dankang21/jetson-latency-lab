# SPDX-License-Identifier: Apache-2.0
"""Deployed MobileNetV2 governor (Table II): per-board configuration, the two-cell
endpoint fits of both arms, the deployed pick rule, the loader of the interleaved
E1 sessions and the scoring of a deployed (arm, deadline) cell.  Shared by
campaign/e5_han_baseline.py (and through it e5b_emcfree.py and e5b_contrast.py),
campaign/make_fig_govdeploy_rev.py, selector_cmp.py and the Table II, Table III and
Supp. Nano ViT-Small printers (govdeploy_stats.py, solo_stats.py, nano_vit_grid.py).

Read-only on results/ (python3 + numpy); nothing here writes a file.

Rule (both arms): T(f) = k/f + b through the GPU 306 and top-clock endpoint cells
(response_us p50), r = 1.035, pick = lowest f in the SKU's grid with T(f)*r <= D,
else the policy DECLINES (fallback to the top clock).
  blind : endpoints at EMC 3199 (profiling point);  aware : endpoints at EMC 2133.
  NX  original fit  : results/nx_validation/nx_est_emc{3199,2133}_gpu{306,1173}_mobilenet.json (8 Hz cells)
      rate-matched  : results/refit33/nx/emc{3199,2133}_g{306,1173}_r{1,2,3}.json (33 Hz, endpoint
                      p50 = median of the three repeat p50s)
  Nano original fit : results/nano_grid/mobilenet/est_emc{3199,2133}_gpu{306,1020}.json (50 Hz =
                      deployment rate, so rate-matched by construction)
      refit dir     : results/refit33/nano holds the 33 Hz re-profile of 2026-10-28 (not the
                      50 Hz deployment rate); refit_fits() reads it only when a caller asks.
Deployment data: results/govrep_rev/<session>/g<f>_p<1..10>.{csv,json,power.csv}, 1000 cycles per
run, EMC locked 2133; load_session() returns None for a session directory that does not exist.

Scoring of one (arm, D) cell (arm_cell, evaluate): miss = response_us > D per 1000-cycle run;
median and min-max of the per-run miss % over every pooled run and, per session, the number of
runs with miss > tau = 2 %; a picked clock outside a session's run set has no runs there.
Energy (delta_E, paired_stats): E/inf [mJ] = trimmed-mean VDD_IN [mW] / rate [Hz]
(e1b_stats.energy); dE_i = E_aware,i - E_blind,i paired by interleaved pair index, pairs pooled
over sessions; mean, paired-t 95 % CI (two-sided 97.5 % Student-t quantiles from the table below,
numpy only, so the interval does not depend on scipy), paired bootstrap 95 % CI (RandomState
seed 20260915, B = 10000), positive pairs with the exact two-sided sign-test p, and
rel = mean dE / mean E_blind.
"""
import csv
import glob
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LAB = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from e1b_stats import fit_2cell, energy, R, miss_pct, sign_test_p, TAU, SEED, NBOOT   # noqa: E402

BOARDS = {
    "nx": dict(name="NX", hz=33.0, lo=306, fmax=1173,
               grid=[510, 612, 714, 816, 918, 1020, 1122, 1173],
               D=[4.4, 4.6, 4.8, 5.0, 5.2, 5.4, 5.6, 5.8, 6.0],
               sessions=["nx", "nx_mobilenet", "nx_mobilenet_s2", "nx_mobilenet_s3"],
               orig=lambda emc, g: f"{LAB}/results/nx_validation/nx_est_emc{emc}_gpu{g}_mobilenet.json",
               refit=f"{LAB}/results/refit33/nx"),
    "nano": dict(name="Nano", hz=50.0, lo=306, fmax=1020,
                 grid=[306, 408, 510, 612, 714, 816, 918, 1020],
                 D=[8.5, 9.0, 9.5, 10.0],
                 sessions=["nano", "nano_mobilenet", "nano_mobilenet_s2", "nano_mobilenet_s3"],
                 orig=lambda emc, g: f"{LAB}/results/nano_grid/mobilenet/est_emc{emc}_gpu{g}.json",
                 refit=f"{LAB}/results/refit33/nano"),
}

# Two-sided 97.5 % Student-t quantiles (numpy-only; no scipy in the venv).
T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306,
        9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131, 16: 2.120,
        17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086, 21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064,
        25: 2.060, 26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042, 40: 2.021, 60: 2.000,
        120: 1.980}


def tcrit(df):
    if df in T975:
        return T975[df]
    ks = sorted(T975)
    if df > ks[-1]:
        return 1.960
    hi = min(k for k in ks if k > df)
    lo = max(k for k in ks if k < df)
    w = (df - lo) / (hi - lo)
    return T975[lo] + w * (T975[hi] - T975[lo])


# ------------------------------------------------------------------ fits ----
def p50_json(path):
    d = json.load(open(path))
    return float(d["response_us"]["p50"]), float(d["meta"]["hz"])


def original_fits(B):
    out, info = {}, {}
    for arm, emc in (("blind", 3199), ("aware", 2133)):
        cells, hz = {}, None
        for g in (B["lo"], B["fmax"]):
            cells[g], hz = p50_json(B["orig"](emc, g))
        out[arm] = fit_2cell(cells[B["lo"]], cells[B["fmax"]], B["lo"], B["fmax"])
        info[arm] = dict(cells=cells, hz=hz)
    return out, info


def refit_fits(B):
    d = B["refit"]
    out, info = {}, {}
    for arm, emc in (("blind", 3199), ("aware", 2133)):
        cells, hz = {}, None
        for g in (B["lo"], B["fmax"]):
            files = [f"{d}/emc{emc}_g{g}_r{r}.json" for r in (1, 2, 3)]
            if not all(os.path.isfile(f) for f in files):
                return None, None
            ps = [p50_json(f) for f in files]
            hz = ps[0][1]
            cells[g] = float(np.median([p for p, _ in ps]))
        out[arm] = fit_2cell(cells[B["lo"]], cells[B["fmax"]], B["lo"], B["fmax"])
        info[arm] = dict(cells=cells, hz=hz)
    return out, info


def that(kb, f):
    k, b = kb
    return (k / f + b) * R


def pick(kb, D_us, B):
    """(clock, feasible): lowest f in the grid with T(f)*r <= D, else (fmax, False) = declined."""
    for f in B["grid"]:
        if that(kb, f) <= D_us:
            return f, True
    return B["fmax"], False


# ------------------------------------------------------------------ data ----
def load_session(name, hz_expected):
    root = f"{LAB}/results/govrep_rev/{name}"
    files = sorted(glob.glob(f"{root}/g*_p*.csv"))
    files = [f for f in files if not f.endswith(".power.csv")]
    if not os.path.isdir(root) or not files:
        return None
    runs = {}
    for csvp in files:
        g, p = os.path.basename(csvp)[:-4].split("_p")
        f, pair = int(g[1:]), int(p)
        v = np.array([float(r["response_us"]) for r in csv.DictReader(open(csvp))])
        meta = json.load(open(csvp[:-4] + ".json"))["meta"]
        assert abs(float(meta["hz"]) - hz_expected) < 1e-6, (csvp, meta["hz"])
        assert len(v) == 1000, (csvp, len(v))
        runs.setdefault(f, {})[pair] = dict(v=v, stem=csvp[:-4], hz=float(meta["hz"]),
                                            ts=meta.get("timestamp", ""))
    E = {f: {p: energy(runs[f][p]) for p in runs[f]} for f in runs}
    ts = [runs[f][p]["ts"] for f in runs for p in runs[f]]
    return dict(name=name, runs=runs, E=E, t0=min(ts)[:16], t1=max(ts)[:16],
                pairs={f: len(runs[f]) for f in sorted(runs)})


# --------------------------------------------------------------- scoring ----
def arm_cell(sessions, f, D_us):
    """Per-session miss lists (None if the clock was not run) + pooled median/min/max."""
    per = []
    for s in sessions:
        if f in s["runs"]:
            per.append([miss_pct(s["runs"][f][p]["v"], D_us) for p in sorted(s["runs"][f])])
        else:
            per.append(None)
    allm = [m for l in per if l for m in l]
    cell = dict(f=f, per=per, has_data=bool(allm))
    if allm:
        cell.update(med=float(np.median(allm)), mn=min(allm), mx=max(allm),
                    viol=[None if l is None else sum(1 for m in l if m > TAU) for l in per],
                    nruns=len(allm))
    return cell


def paired_stats(aware, blind):
    d = np.asarray(aware, float) - np.asarray(blind, float)
    n = len(d)
    out = dict(n=n, mean=float(d.mean()), pos=int((d > 0).sum()))
    out["sign_p"] = sign_test_p(out["pos"], n)
    sd = float(d.std(ddof=1)) if n > 1 else float("nan")
    se = sd / math.sqrt(n)
    tc = tcrit(n - 1)
    out.update(sd=sd, t_lo=out["mean"] - tc * se, t_hi=out["mean"] + tc * se)
    rng = np.random.RandomState(SEED)
    boots = d[rng.randint(0, n, size=(NBOOT, n))].mean(axis=1)
    out["b_lo"], out["b_hi"] = (float(x) for x in np.percentile(boots, [2.5, 97.5]))
    out["Eb"], out["Ea"] = float(np.mean(blind)), float(np.mean(aware))
    out["rel"] = 100.0 * out["mean"] / out["Eb"]
    return out


def delta_E(sessions, fb, fa):
    if fb == fa:
        return "same"
    eb, ea = [], []
    for s in sessions:
        if fb in s["runs"] and fa in s["runs"]:
            for p in sorted(set(s["runs"][fb]) & set(s["runs"][fa])):
                eb.append(s["E"][fb][p])
                ea.append(s["E"][fa][p])
    return paired_stats(ea, eb) if eb else None


def evaluate(B, fits, sessions):
    rows = []
    for D in B["D"]:
        D_us = D * 1000.0
        fb, okb = pick(fits["blind"], D_us, B)
        fa, oka = pick(fits["aware"], D_us, B)
        rows.append(dict(D=D, fb=fb, okb=okb, fa=fa, oka=oka,
                         blind=arm_cell(sessions, fb, D_us), aware=arm_cell(sessions, fa, D_us),
                         dE=delta_E(sessions, fb, fa)))
    return rows


def session_dE(sessions, fb, fa):
    """Per-session view of a contrasting cell's paired energy differences (aware - blind, mJ/inf):
    one dict per session that ran both clocks (name, n pairs, mean, positive pairs), and over
    those session means: their spread (max - min) and the t interval over the k session means
    (k - 1 df, same quantile table as the pooled interval).  None when fb == fa or no session
    ran both clocks."""
    if fb == fa:
        return None
    per = []
    for s in sessions:
        if fb in s["runs"] and fa in s["runs"]:
            ps = sorted(set(s["runs"][fb]) & set(s["runs"][fa]))
            d = np.array([s["E"][fa][p] - s["E"][fb][p] for p in ps])
            per.append(dict(name=s["name"], n=len(d), mean=float(d.mean()), pos=int((d > 0).sum())))
    if not per:
        return None
    m = np.array([x["mean"] for x in per])
    out = dict(sessions=per, k=len(per), spread=float(m.max() - m.min()), mean_of_means=float(m.mean()),
               all_pos=all(x["pos"] == x["n"] for x in per))
    if len(per) > 1:
        se = float(m.std(ddof=1)) / math.sqrt(len(per))
        tc = tcrit(len(per) - 1)
        out.update(df=len(per) - 1, lo=out["mean_of_means"] - tc * se, hi=out["mean_of_means"] + tc * se)
    return out

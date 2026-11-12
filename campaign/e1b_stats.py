#!/usr/bin/env python3
"""E1b statistics: NX breadth cells (ViT-Small 20 Hz, YOLOv8s 10 Hz, ResNet-50 20 Hz)
scored under the deployed two-cell rule over the pre-stated deadline grids
(docs/protocols/E1_E2_protocol.md, E1b addendum, frozen 2026-09-15).

Read-only, offline (python3 over results/ only).  Run pinned:
    taskset -c 1 nice -n 19 python3 campaign/e1b_stats.py \
        [--model vit,yolov8s,resnet50] [--fits original|refit|both] [--md out.md] [--offgrid]
The repository root defaults to the parent of campaign/ (override: LAB=/path/to/checkout).

Rule (both arms, docs/protocols/E1_E2_protocol.md): T(f) = k/f + b through the GPU 306 and 1173 endpoint
cells (response_us p50), r = 1.035, pick = lowest f in {510, ..., 1173} with T(f)*r <= D,
else INFEASIBLE -> 1173 fallback (reported as a fallback, not a pick).
  blind : endpoints at EMC 3199  results/nx_validation/nx_est_emc3199_gpu{306,1173}_<model>.json
  aware : endpoints at EMC 2133  results/nx_validation/nx_est_emc2133_gpu{306,1173}_<model>.json
  rate-matched refit (when present): results/refit33/nx_<model>/emc{2133,3199}_g{306,1173}_r{1,2,3}.json,
          endpoint p50 = median of the three per-repeat p50s (all three are printed).
Data: results/govrep_rev/nx_<model>/g<f>_p<1..10>.{csv,json,power.csv}; one interleaved,
thermally gated session per model, EMC locked 2133, clocks {612,...,1122} (govrep_rev3.sh).

Scoring (Table II convention, fixed 2026-09-15): miss = response_us > D per
1000-cycle run; per (arm, D): the 10 per-run miss %, median and min-max, number of runs with
miss > tau = 2 % (k/10), pooled miss over the 10 runs.  A picked clock outside the run set
is reported as 'no data'.

Energy (campaign/analyze_energy_cal.py convention):
    E/inf [mJ] = mean(VDD_IN [mW] over samples p[n//10 : -n//10]) / release rate [Hz]
dE_i = E_aware,i - E_blind,i paired by interleaved pair index i = 1..10; reported: mean, s.d.,
paired-t 95 % CI, paired-bootstrap percentile 95 % CI (seed 20260915, B = 10000), sign count
with exact two-sided sign-test p, and relative dE = mean dE / mean E_blind.  Sensitivity: the
same with a steady-state window t >= 8 s (the 40 s INA log starts at process launch and its
first ~4 s are the ORT model-load phase, which the 10 % trim removes).
"""
import argparse
import csv
import glob
import json
import math
import os
import sys

import numpy as np

try:
    from scipy import stats as sps
except Exception:  # fall back to a small t table
    sps = None

LAB = os.environ.get("LAB", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
R = 1.035
GRID = [510, 612, 714, 816, 918, 1020, 1122, 1173]
FMAX = 1173
TAU = 2.0                     # per-run miss budget, %
SEED = 20260915
NBOOT = 10000
STEADY_T0 = 8.0               # sensitivity window start, s
# Two-sided 95 % Student t critical values t_{0.975, df} for df 1-60 (scipy.stats.t.ppf(0.975, df),
# scipy 1.8.0, rounded to 3 decimals) and the large-df limit z_{0.975} (scipy.stats.norm.ppf(0.975));
# used only when scipy is not importable (tcrit()).
T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306,
        9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131, 16: 2.120,
        17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086, 21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064,
        25: 2.060, 26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042, 31: 2.040, 32: 2.037,
        33: 2.035, 34: 2.032, 35: 2.030, 36: 2.028, 37: 2.026, 38: 2.024, 39: 2.023, 40: 2.021,
        41: 2.020, 42: 2.018, 43: 2.017, 44: 2.015, 45: 2.014, 46: 2.013, 47: 2.012, 48: 2.011,
        49: 2.010, 50: 2.009, 51: 2.008, 52: 2.007, 53: 2.006, 54: 2.005, 55: 2.004, 56: 2.003,
        57: 2.002, 58: 2.002, 59: 2.001, 60: 2.000}
T975_INF = 1.960

MODELS = {
    "vit": dict(name="ViT-Small", hz=20,
                grid=[14.0, 14.5, 15.0, 15.5, 16.0, 17.0, 17.5, 18.5, 19.0],
                proto_aware=(8283856, 5858)),
    "yolov8s": dict(name="YOLOv8s", hz=10,
                    grid=[29.5, 30.0, 31.0, 31.5, 33.0, 33.5, 35.5, 36.0, 38.5, 39.0],
                    proto_aware=(17627925, 12354)),
    "resnet50": dict(name="ResNet-50", hz=20,
                     grid=[14.0, 14.5, 15.0, 15.5, 16.0, 17.0, 17.5, 18.5, 19.0],
                     proto_aware=(9130786, 4941)),
}

# Submitted rows (first-version supplement, tables tab:vitgov and tab:breadthgov):
# D -> (blind f, blind miss, aware f, aware miss, dE mJ/inf).  Energies there: ViT from
# results/energy_cal (20 Hz, n=2); ResNet/YOLO from results/govrep_{resnet50,yolov8s}
# (10 Hz / 5 Hz, n=3).  Note that both rates differ from E1b (ResNet-50 20 Hz, YOLOv8s 10 Hz), so the dE of those rows is not comparable with E1b's.
SUBMITTED = {
    "vit": {13.6: (918, "100", 1122, "0.1", 12.6),
            14.2: (816, "100", 1020, "0.0", 10.9),
            15.0: (816, "28.0+-10.7", 918, "0.2", 4.6)},
    "resnet50": {13.0: (1020, "100", 1173, "0.03", 17.3),
                 14.0: (918, "95.8+-1.1", 1020, "0.07", 11.8),
                 15.0: (816, "95.5+-1.2", 918, "0.00", 11.2)},
    "yolov8s": {29.0: (918, "100", 1122, "0.00", 48.7),
                30.0: (918, "86.3+-1.5", 1020, "0.00", 24.0),
                32.0: (816, "59.0+-9.5", 918, "0.03", 23.5)},
}
# Two-cell re-derivation of those rows (2026-09-15; original endpoint cells, deployed two-cell rule):
# D -> (blind f, aware f or 'INF', dE from the old blocked energy repeats).
REDERIVED2C = {
    "vit": {13.6: (918, 1173, 15.4), 14.2: (918, 1122, 12.6), 15.0: (816, 1020, 10.9)},
    "resnet50": {13.0: (1020, "INF", None), 14.0: (918, 1122, 23.6), 15.0: (816, 1020, 23.1)},
    "yolov8s": {29.0: (918, 1173, 59.3), 30.0: (918, 1122, 48.7), 32.0: (816, 1020, 47.5)},
}


# ------------------------------------------------------------------ fits ----
def fit_2cell(t_lo, t_hi, lo=306, hi=FMAX):
    """Endpoint refit T(f) = k/f + b, f in MHz -> k in us*MHz, b in us
    (same arithmetic as campaign/admit_pick.py::fit_2cell, which uses GHz)."""
    k = (t_lo - t_hi) / (1.0 / lo - 1.0 / hi)
    b = t_hi - k / hi
    return k, b


def p50_json(path):
    return float(json.load(open(path))["response_us"]["p50"])


def original_fits(model):
    """(blind (k,b), aware (k,b), info) from the committed nx_validation endpoint cells."""
    out, info = {}, {}
    for arm, emc in (("blind", 3199), ("aware", 2133)):
        cells = {}
        for g in (306, FMAX):
            p = f"{LAB}/results/nx_validation/nx_est_emc{emc}_gpu{g}_{model}.json"
            d = json.load(open(p))
            cells[g] = float(d["response_us"]["p50"])
            info[(arm, g)] = dict(p50=cells[g], hz=d["meta"]["hz"], iters=d["meta"]["iters"],
                                  file=os.path.relpath(p, LAB))
        out[arm] = fit_2cell(cells[306], cells[FMAX])
    return out, info


def refit33_fits(model, refit_dir=None):
    """Rate-matched endpoint fits from results/refit33/nx_<model>, or None if incomplete."""
    d = refit_dir or f"{LAB}/results/refit33/nx_{model}"
    out, info = {}, {}
    for arm, emc in (("blind", 3199), ("aware", 2133)):
        cells = {}
        for g in (306, FMAX):
            files = [f"{d}/emc{emc}_g{g}_r{r}.json" for r in (1, 2, 3)]
            if not all(os.path.isfile(f) for f in files):
                return None, dict(missing=[os.path.relpath(f, LAB) for f in files if not os.path.isfile(f)])
            ps = [p50_json(f) for f in files]
            hz = [json.load(open(f))["meta"]["hz"] for f in files]
            cells[g] = float(np.median(ps))
            info[(arm, g)] = dict(p50=cells[g], reps=ps, hz=hz[0], file=os.path.relpath(files[0], LAB))
        out[arm] = fit_2cell(cells[306], cells[FMAX])
    return out, info


def that(kb, f):
    k, b = kb
    return (k / f + b) * R


def pick(kb, D_us):
    """(clock, feasible).  Lowest f in GRID with T(f)*r <= D, else (FMAX, False)."""
    for f in GRID:
        if that(kb, f) <= D_us:
            return f, True
    return FMAX, False


# ------------------------------------------------------------------ data ----
def load_runs(model):
    """{clock: {pair: dict(v=response_us array, stem=..., hz=...)}} for one model."""
    root = f"{LAB}/results/govrep_rev/nx_{model}"
    runs = {}
    for csvp in sorted(glob.glob(f"{root}/g*_p*.csv")):
        base = os.path.basename(csvp)
        if base.endswith(".power.csv"):
            continue
        g, p = base[:-4].split("_p")
        f, pair = int(g[1:]), int(p)
        v = np.array([float(r["response_us"]) for r in csv.DictReader(open(csvp))])
        meta = json.load(open(csvp[:-4] + ".json"))["meta"]
        runs.setdefault(f, {})[pair] = dict(v=v, stem=csvp[:-4], hz=float(meta["hz"]),
                                            ts=meta.get("timestamp", ""))
    return runs


RAILS = ("vdd_in_mw", "cpugpu_mw", "soc_mw")


def read_power(path):
    cols = {k: [] for k in ("t",) + RAILS}
    for r in csv.DictReader(open(path)):
        if r.get("vdd_in_mw"):
            for k in cols:
                cols[k].append(float(r[k]))
    return {k: np.array(v) for k, v in cols.items()}


def energy(run, rail="vdd_in_mw", steady=False):
    """E/inf in mJ on one rail.  Default: campaign trim p[n//10:-n//10]; steady: t >= STEADY_T0 s."""
    if "power" not in run:
        run["power"] = read_power(run["stem"] + ".power.csv")
    t, p = run["power"]["t"], run["power"][rail]
    n = len(p)
    if n < 20:
        raise RuntimeError(run["stem"])
    if steady:
        q = p[t >= STEADY_T0]
    else:
        q = p[n // 10:-n // 10]
    return float(q.mean()) / run["hz"]


def miss_pct(v, D_us):
    return 100.0 * float(np.count_nonzero(v > D_us)) / len(v)


def tcrit(df):
    if sps is not None:
        return float(sps.t.ppf(0.975, df))
    if df < 1:
        return float("nan")                    # as scipy for df = 0
    if df in T975:
        return T975[df]
    # df > 60: interpolate linearly in 1/df between df = 60 and the limit (within 0.02 % of scipy)
    return T975_INF + (T975[60] - T975_INF) * 60.0 / df


def sign_test_p(pos, n):
    k = min(pos, n - pos)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def paired_stats(aware, blind, seed=SEED):
    d = np.asarray(aware, float) - np.asarray(blind, float)
    n = len(d)
    out = dict(n=n, mean=float(d.mean()), diffs=d.tolist(), pos=int((d > 0).sum()))
    out["sign_p"] = sign_test_p(out["pos"], n)
    sd = float(d.std(ddof=1))
    se = sd / math.sqrt(n)
    tc = tcrit(n - 1)
    out.update(sd=sd, se=se, t_lo=out["mean"] - tc * se, t_hi=out["mean"] + tc * se)
    rng = np.random.RandomState(seed)
    idx = rng.randint(0, n, size=(NBOOT, n))
    boots = d[idx].mean(axis=1)
    out["b_lo"], out["b_hi"] = (float(x) for x in np.percentile(boots, [2.5, 97.5]))
    return out


# ------------------------------------------------------------- evaluate ----
def arm_cell(runs, f, feasible, D_us):
    """Miss statistics for one (arm, D) cell at clock f (or 'no data')."""
    cell = dict(f=f, feasible=feasible, has_data=f in runs)
    if f not in runs:
        return cell
    pairs = sorted(runs[f])
    m = [miss_pct(runs[f][p]["v"], D_us) for p in pairs]
    misses = sum(int(np.count_nonzero(runs[f][p]["v"] > D_us)) for p in pairs)
    cycles = sum(len(runs[f][p]["v"]) for p in pairs)
    cell.update(pairs=pairs, miss=m, med=float(np.median(m)), mn=min(m), mx=max(m),
                viol=sum(1 for x in m if x > TAU), pooled=100.0 * misses / cycles)
    return cell


def evaluate(model, fits, runs, E, Es, Ec, Dlist):
    rows = []
    for D in Dlist:
        D_us = D * 1000.0
        fb, okb = pick(fits["blind"], D_us)
        fa, oka = pick(fits["aware"], D_us)
        row = dict(D=D, blind=arm_cell(runs, fb, okb, D_us), aware=arm_cell(runs, fa, oka, D_us),
                   Tb=that(fits["blind"], fb), Ta=that(fits["aware"], fa))
        if fb == fa:
            row["dE"] = "same"
        elif fb in runs and fa in runs:
            pairs = sorted(set(runs[fb]) & set(runs[fa]))
            eb = [E[fb][p] for p in pairs]
            ea = [E[fa][p] for p in pairs]
            st = paired_stats(ea, eb)
            st["Eb"], st["Ea"] = float(np.mean(eb)), float(np.mean(ea))
            st["Eb_sd"], st["Ea_sd"] = float(np.std(eb, ddof=1)), float(np.std(ea, ddof=1))
            st["rel"] = 100.0 * st["mean"] / st["Eb"]
            sts = paired_stats([Es[fa][p] for p in pairs], [Es[fb][p] for p in pairs])
            st["steady_mean"], st["steady_lo"], st["steady_hi"] = sts["mean"], sts["t_lo"], sts["t_hi"]
            stc = paired_stats([Ec[fa][p] for p in pairs], [Ec[fb][p] for p in pairs])
            st["cg_mean"], st["cg_lo"], st["cg_hi"], st["cg_sd"] = stc["mean"], stc["t_lo"], stc["t_hi"], stc["sd"]
            row["dE"] = st
        else:
            row["dE"] = None
        rows.append(row)
    return rows


# --------------------------------------------------------------- output ----
def fpick(cell):
    return f"{cell['f']}" if cell["feasible"] else f"INFEASIBLE->{cell['f']}"


def fmiss(cell):
    if not cell["has_data"]:
        return "no data"
    return (f"{cell['med']:.1f} % med ({cell['mn']:.1f}-{cell['mx']:.1f}), "
            f"{cell['viol']}/10 >{TAU:.0f} %, pooled {cell['pooled']:.2f} %")


def fdE(dE):
    if dE == "same":
        return "same clock (dE = 0)"
    if dE is None:
        return "no data"
    return (f"{dE['mean']:+.2f} mJ [t {dE['t_lo']:+.2f}, {dE['t_hi']:+.2f}] "
            f"[boot {dE['b_lo']:+.2f}, {dE['b_hi']:+.2f}] {dE['pos']}/{dE['n']} (p={dE['sign_p']:.3f}) "
            f"= {dE['rel']:+.2f} % of E_blind {dE['Eb']:.1f}")


def print_model(model, tag, fits, finfo, rows, runs, E, hz):
    M = MODELS[model]
    print(f"\n=== NX {M['name']} {hz:.0f} Hz  [{tag} fits]  results/govrep_rev/nx_{model}")
    for arm in ("blind", "aware"):
        k, b = fits[arm]
        cells = "; ".join(f"g{g}: {finfo[(arm, g)]['p50']:.1f} us"
                          + (f" (reps {', '.join(f'{x:.1f}' for x in finfo[(arm, g)]['reps'])})"
                             if "reps" in finfo[(arm, g)] else "")
                          + f" @{finfo[(arm, g)]['hz']:.0f} Hz" for g in (306, FMAX))
        print(f"  {arm:5s} fit: k={k:.0f} us*MHz b={b:.0f} us  [{cells}]")
        print("        T(f)*r us: " + " ".join(f"{f}:{that(fits[arm], f):.0f}" for f in GRID))
    print(f"  {'D ms':>6} | {'blind':>16} {'miss':52} | {'aware':>16} {'miss':52} | dE aware-blind")
    for r in rows:
        print(f"  {r['D']:6.1f} | {fpick(r['blind']):>16} {fmiss(r['blind']):52} | "
              f"{fpick(r['aware']):>16} {fmiss(r['aware']):52} | {fdE(r['dE'])}")
    print("  per-run miss % (pairs 1..10):")
    for r in rows:
        for arm in ("blind", "aware"):
            c = r[arm]
            if c["has_data"]:
                print(f"    D={r['D']:5.1f} {arm:5s} {c['f']:4d}: " + " ".join(f"{x:5.1f}" for x in c["miss"]))
    for r in rows:
        if isinstance(r["dE"], dict):
            d = r["dE"]
            print(f"    D={r['D']:5.1f} dE per pair: " + " ".join(f"{x:+.2f}" for x in d["diffs"])
                  + f"  | steady-window (t>={STEADY_T0:.0f}s) mean {d['steady_mean']:+.2f} "
                    f"[{d['steady_lo']:+.2f}, {d['steady_hi']:+.2f}]"
                  + f"  | CPU/GPU rail only {d['cg_mean']:+.2f} sd {d['cg_sd']:.2f} [{d['cg_lo']:+.2f}, {d['cg_hi']:+.2f}]")


def print_curves(model, runs, E, Dlist):
    clocks = sorted(runs)
    print("  per-clock p50 response us / E per inf mJ (mean +- sd over 10 pairs) / SOC-rail mW per pair / pooled miss %/violating runs at each grid D:")
    print("  " + f"{'f':>5} {'p50':>8} {'E/inf':>14} | " + " ".join(f"{D:>6.1f}" for D in Dlist))
    for f in clocks:
        pairs = sorted(runs[f])
        soc = [runs[f][p]["power"]["soc_mw"][len(runs[f][p]["power"]["soc_mw"]) // 10:-len(runs[f][p]["power"]["soc_mw"]) // 10].mean() for p in pairs]
        p50 = float(np.median(np.concatenate([runs[f][p]["v"] for p in pairs])))
        e = np.array([E[f][p] for p in pairs])
        misses = []
        for D in Dlist:
            D_us = D * 1000.0
            tot = sum(int(np.count_nonzero(runs[f][p]["v"] > D_us)) for p in pairs)
            n = sum(len(runs[f][p]["v"]) for p in pairs)
            viol = sum(1 for p in pairs if miss_pct(runs[f][p]["v"], D_us) > TAU)
            misses.append(f"{100.0 * tot / n:5.1f}/{viol}")
        print("  " + f"{f:>5} {p50:8.0f} {e.mean():7.1f}+-{e.std(ddof=1):5.1f} | " + " ".join(f"{m:>7}" for m in misses)
              + "  | SOC " + " ".join(f"{s:.0f}" for s in soc))


def md_model(model, tag, fits, finfo, rows, runs, E, hz, offgrid_rows=None, refit_note=None):
    M = MODELS[model]
    L = []
    L.append(f"## NX {M['name']} {hz:.0f} Hz (results/govrep_rev/nx_{model}), {tag}")
    fit_line = []
    for arm in ("blind", "aware"):
        k, b = fits[arm]
        g306, g1173 = finfo[(arm, 306)], finfo[(arm, FMAX)]
        fit_line.append(f"{arm} k={k:.0f} b={b:.0f} (cells {g306['p50']:.0f}/{g1173['p50']:.0f} µs @{g306['hz']:.0f} Hz)")
    L.append("- Fits (two-cell 306/1173, r=1.035): " + "; ".join(fit_line)
             + (f". Fit-cell rate {finfo[('aware', 306)]['hz']:.0f} Hz vs deployment {hz:.0f} Hz."
                if abs(finfo[("aware", 306)]["hz"] - hz) > 0.01 else ". Fit-cell rate matches deployment."))
    for arm in ("blind", "aware"):
        parts = []
        for r in rows:
            c = r[arm]
            if not c["has_data"]:
                parts.append(f"{fpick(c)}@{r['D']:.1f} → no data")
            elif c["viol"] == 0:
                parts.append(f"{fpick(c)}@{r['D']:.1f} → 0 (max {c['mx']:.1f} %)" if c["mx"] > 0 else f"{fpick(c)}@{r['D']:.1f} → 0")
            else:
                parts.append(f"{fpick(c)}@{r['D']:.1f} → {c['med']:.1f} % med ({c['mn']:.1f}–{c['mx']:.1f}; {c['viol']}/10; pooled {c['pooled']:.1f} %)")
        L.append(f"- {arm}: " + "; ".join(parts) + ".")
    # violation summary
    for arm in ("blind", "aware"):
        cells = [r[arm] for r in rows if r[arm]["has_data"]]
        nv = sum(1 for c in cells if c["viol"] >= 1)
        nf = sum(1 for c in cells if c["pooled"] > TAU)
        n10 = sum(1 for c in cells if c["viol"] == 10)
        L.append(f"- {arm} summary: {nv}/{len(cells)} evaluable cells with ≥1 violating run, "
                 f"{nf}/{len(cells)} fail the pooled 2 % budget, {n10} cells 10/10.")
    L.append("")
    L.append("| D (ms) | blind f | aware f | E_blind (mJ/inf) | E_aware | ΔE mean | s.d. | 95 % CI (t) | 95 % CI (boot) | sign (p) | ΔE/E_blind |")
    L.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        d = r["dE"]
        if d == "same":
            L.append(f"| {r['D']:.1f} | {fpick(r['blind'])} | {fpick(r['aware'])} | — | — | 0 (same clock) | | | | | 0 |")
        elif d is None:
            L.append(f"| {r['D']:.1f} | {fpick(r['blind'])} | {fpick(r['aware'])} | no data | | | | | | | |")
        else:
            L.append(f"| {r['D']:.1f} | {fpick(r['blind'])} | {fpick(r['aware'])} | {d['Eb']:.1f} ± {d['Eb_sd']:.1f} | "
                     f"{d['Ea']:.1f} ± {d['Ea_sd']:.1f} | {d['mean']:+.2f} | {d['sd']:.2f} | "
                     f"[{d['t_lo']:+.2f}, {d['t_hi']:+.2f}] | [{d['b_lo']:+.2f}, {d['b_hi']:+.2f}] | "
                     f"{d['pos']}/{d['n']} ({d['sign_p']:.3f}) | {d['rel']:+.1f} % |")
    L.append("")
    # per-clock energy line
    clocks = sorted(runs)
    L.append("- Per-clock E/inf (mJ, mean ± sd, 10 runs): " + ", ".join(
        f"{f}: {np.mean([E[f][p] for p in runs[f]]):.1f} ± {np.std([E[f][p] for p in runs[f]], ddof=1):.1f}"
        for f in clocks) + f". Steady-window (t ≥ {STEADY_T0:.0f} s) sensitivity: "
        + ", ".join(f"{r['D']:.1f}: {r['dE']['steady_mean']:+.2f}" for r in rows if isinstance(r["dE"], dict))
        + ". CPU/GPU-rail-only ΔE (t-CI): "
        + ", ".join(f"{r['D']:.1f}: {r['dE']['cg_mean']:+.2f} [{r['dE']['cg_lo']:+.2f}, {r['dE']['cg_hi']:+.2f}]"
                    for r in rows if isinstance(r["dE"], dict)) + ".")
    if offgrid_rows:
        L.append("- Submitted (off-grid, comparison only) deadlines re-scored on this session under the two-cell rule: "
                 + "; ".join(f"D={r['D']:.1f}: blind {fpick(r['blind'])} → {fmiss(r['blind'])}; aware {fpick(r['aware'])} → {fmiss(r['aware'])}; ΔE {fdE(r['dE'])}"
                             for r in offgrid_rows) + ".")
    if refit_note:
        L.append(refit_note)
    return L


# ----------------------------------------------------------------- main ----
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="vit,yolov8s,resnet50")
    ap.add_argument("--fits", default="both", choices=["original", "refit", "both"])
    ap.add_argument("--refit-dir", default=None, help="override results/refit33/nx_<model>")
    ap.add_argument("--md", default=None, help="write markdown sections here")
    ap.add_argument("--offgrid", action="store_true", help="also re-score the submitted (off-grid) deadlines")
    a = ap.parse_args()
    print(f"# E1b stats: rule two-cell r={R}, grid {GRID}, tau={TAU} %, seed={SEED}, B={NBOOT}, "
          f"scipy={'yes' if sps else 'no'}")
    md = []
    for model in a.model.split(","):
        M = MODELS[model]
        runs = load_runs(model)
        clocks = sorted(runs)
        npairs = {f: len(runs[f]) for f in clocks}
        hz = {runs[f][p]["hz"] for f in clocks for p in runs[f]}
        assert len(hz) == 1, hz
        hz = hz.pop()
        assert abs(hz - M["hz"]) < 1e-6, (model, hz)
        E = {f: {p: energy(runs[f][p]) for p in runs[f]} for f in clocks}
        Es = {f: {p: energy(runs[f][p], steady=True) for p in runs[f]} for f in clocks}
        Ec = {f: {p: energy(runs[f][p], rail="cpugpu_mw") for p in runs[f]} for f in clocks}
        print(f"\n##### {M['name']}: clocks {clocks}, pairs/clock {npairs}, rate {hz:.0f} Hz, "
              f"session {min(runs[f][p]['ts'] for f in clocks for p in runs[f])[:19]} .. "
              f"{max(runs[f][p]['ts'] for f in clocks for p in runs[f])[:19]} UTC")

        variants = []
        ofits, oinfo = original_fits(model)
        kp, bp = M["proto_aware"]
        ka, ba = ofits["aware"]
        if abs(ka - kp) > 1 or abs(ba - bp) > 1:
            print(f"  WARNING: aware fit k={ka:.0f} b={ba:.0f} differs from protocol k={kp} b={bp}")
        if a.fits in ("original", "both"):
            variants.append((f"original {oinfo[('aware', 306)]['hz']:.0f} Hz-cell fits", ofits, oinfo))
        rfits, rinfo = refit33_fits(model, a.refit_dir)
        refit_note = None
        if a.fits in ("refit", "both"):
            if rfits is None:
                refit_note = (f"- Rate-matched ({hz:.0f} Hz) refit: **pending** (results/refit33/nx_{model} incomplete; "
                              f"missing e.g. {rinfo['missing'][0]}) → re-run `campaign/e1b_stats.py --model {model}` when present.")
                print(f"  rate-matched refit: pending ({len(rinfo['missing'])} files missing, e.g. {rinfo['missing'][0]})")
            else:
                variants.append((f"rate-matched {rinfo[('aware', 306)]['hz']:.0f} Hz-cell fits (results/refit33/nx_{model})", rfits, rinfo))

        print_curves(model, runs, E, M["grid"])
        for (tag, fits, finfo) in variants:
            rows = evaluate(model, fits, runs, E, Es, Ec, M["grid"])
            print_model(model, tag, fits, finfo, rows, runs, E, hz)
            off = None
            if a.offgrid:
                offD = sorted(set(SUBMITTED[model]) - set(M["grid"]))
                off = evaluate(model, fits, runs, E, Es, Ec, offD)
                print("  -- submitted off-grid deadlines (comparison only):")
                for r in off:
                    print(f"  {r['D']:6.1f} | {fpick(r['blind']):>16} {fmiss(r['blind']):52} | "
                          f"{fpick(r['aware']):>16} {fmiss(r['aware']):52} | {fdE(r['dE'])}")
            # comparison with the original table and its two-cell re-derivation
            print("  -- vs original table and its two-cell re-derivation:")
            allrows = {r["D"]: r for r in rows}
            if off:
                allrows.update({r["D"]: r for r in off})
            for D, (sb, smb, sa, sma, sdE) in sorted(SUBMITTED[model].items()):
                ab, aa, adE = REDERIVED2C[model][D]
                r = allrows.get(D)
                got = (f"blind {fpick(r['blind'])} → {('%.1f%% med' % r['blind']['med']) if r['blind']['has_data'] else 'no data'}; "
                       f"aware {fpick(r['aware'])} → {('%.2f%% med' % r['aware']['med']) if r['aware']['has_data'] else 'no data'}; "
                       f"dE {fdE(r['dE']).split(' = ')[0]}") if r else "not evaluated (off-grid; use --offgrid)"
                print(f"    D={D:5.1f}: submitted blind {sb} ({smb}) aware {sa} ({sma}) dE {sdE:+.1f} | "
                      f"re-derived 2-cell blind {ab} aware {aa} dE {adE if adE is None else '%+.1f' % adE} | E1b: {got}")
            md += md_model(model, tag, fits, finfo, rows, runs, E, hz, off, refit_note if tag.startswith("original") else None)
            md.append("")
    if a.md:
        open(a.md, "w").write("\n".join(md) + "\n")
        print(f"\nwrote {a.md}")


if __name__ == "__main__":
    main()

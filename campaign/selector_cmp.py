#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Selector comparison on recorded runs (Supp. tab:memawarecmp, main text Sec. IV): five latency-model
selectors scored on the measured settings only, for admission coverage, the admitted picks (measured
feasible / measured violating / clock not measured), over-provisioning against the measured frontier,
and the energy premium there.

A comparison of latency models and the selectors built on them, not a re-implementation of DVFO,
SparseDVFS or PredJoule: only the estimator behind the deployed rule changes.  Read-only, offline
(python3 + numpy over results/ only).  Run from anywhere:
    taskset -c 1 nice -n 19 python3 campaign/selector_cmp.py [--mnv2-sessions nx,nx_mobilenet] [--json]

Deadlines (37, all pre-stated): NX MobileNetV2 Table II grid at 33 Hz (sessions results/govrep_rev/
{nx, nx_mobilenet, nx_mobilenet_s3}) and the ViT-S / YOLOv8s / ResNet-50 breadth grids (one session
each, results/govrep_rev/nx_<model>), EMC locked at 2133 MHz.
Selectors, all through the deployed rule (f* = lowest grid clock with T_hat(f, 2133) * r <= D, r = 1.035,
else decline and fall back to 1173 MHz):
  blind  two-cell k/f+b through the EMC-3199 endpoint cells (rate-matched refit33), deployed at 2133
  H1     Han-style A g^-a + B e^-b (C = 0), fitted on the rate-matched cells at EMC {665.6, 3199}, read at 2133
  H2     Han-style + C, all three EMC levels in-sample (2133 included)
  aware  two-cell k/f+b through the EMC-2133 endpoint cells (rate-matched refit33)
  front  measured frontier (an oracle on the scoring runs, not a deployable selector)
  H1c    sensitivity only: C free with beta pinned at 1.0 (printed, not in the table)
The picks of blind/aware/H1/H2/H1c come from the unmodified functions of campaign/e5_han_baseline.py
(arm_fits, han_fits, pick_2cell, pick_han), so they are the E5 picks by construction.
Scoring: miss = response_us > D per 1000-cycle run; a run violates if miss % > tau = 2 %, counted over
every run of that clock in every session that ran it.  Frontier f_F(D): the lowest grid clock that was
run in at least one session and has no violating run over all of its runs; clocks never run (510 MHz
everywhere, 1173 MHz in the breadth sessions) are not candidates.  An admitted pick is 'measured
feasible' (clock run, no violating run), 'measured violating' (>= 1 violating run) or 'clock not
measured'; bins = grid-index distance pick - f_F.
Energy (campaign/e1b_stats.energy, the Table II method): E/inf [mJ] = mean VDD_IN over samples
p[n//10:-n//10] / rate [Hz]; dE = E(selector clock) - E(frontier clock), paired by pair index within a
session and pooled over sessions; % of the mean E_frontier over the same pairs ('E n/a' when no session
ran both clocks).  At a decline the 1173 MHz fallback is priced the same way against the frontier.
Checks printed after the per-model rows: the frontier is monotone at every deadline, per-clock energy
rises with the clock, and the aware - blind dE means with the same pairing (the energy column of
Table II and the breadth table).
Han fits use campaign/han_fit.py (numpy grid, scipy polish when scipy is importable, as in E5).
Usage: selector_cmp.py [--mnv2-sessions LIST] [--json]
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LAB = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import e5_han_baseline as E5                                 # noqa: E402
from e1b_stats import miss_pct, TAU                          # noqa: E402
from govdeploy_common import load_session                    # noqa: E402
from han_fit import load_sweep_dir, cells_dict, HAVE_SCIPY   # noqa: E402

MODELS = ["mobilenet", "vit", "yolov8s", "resnet50"]
SEL = ["blind", "H1", "H2", "aware", "front"]
NAME = {"blind": "EMC-blind 2-cell", "H1": "Han-H1 (held-out)", "H2": "Han-H2 (in-sample)",
        "aware": "per-EMC 2-cell refit", "front": "measured frontier", "H1c": "Han-H1c (beta pinned 1.0)"}


# ------------------------------------------------------------- scoring ----
def sessions_for(cfg, which):
    names = cfg["sessions"] if which is None else which
    out = []
    for n in names:
        s = load_session(n, cfg["hz"])
        if s:
            out.append(s)
    return out


def clock_runs(sessions, f):
    """List of (session index, pair, response array) for clock f across sessions."""
    return [(i, p, s["runs"][f][p]["v"]) for i, s in enumerate(sessions) if f in s["runs"]
            for p in sorted(s["runs"][f])]


def viol_count(sessions, f, D_us):
    rr = clock_runs(sessions, f)
    if not rr:
        return None
    m = [miss_pct(v, D_us) for _, _, v in rr]
    per = []
    for i, s in enumerate(sessions):
        if f in s["runs"]:
            per.append(sum(1 for (j, _, v) in rr if j == i and miss_pct(v, D_us) > TAU))
        else:
            per.append(None)
    return dict(n=len(rr), viol=sum(1 for x in m if x > TAU), per=per, med=float(np.median(m)),
                mx=max(m), pooled=100.0 * sum(int(np.count_nonzero(v > D_us)) for _, _, v in rr)
                / sum(len(v) for _, _, v in rr))


def frontier(sessions, grid, D_us):
    """Lowest grid clock that was run (>= 1 session) and has 0 violating runs over every run of it in every
    session that ran it.  Clocks not run in any session are skipped (not candidates).  Returns
    (f or None, list of unrun grid clocks below f, monotone flag = every higher RUN clock also clean)."""
    f_star, unrun_below = None, []
    for f in grid:
        vc = viol_count(sessions, f, D_us)
        if vc is None:
            unrun_below.append(f)
            continue
        if vc["viol"] == 0:
            f_star = f
            break
    if f_star is None:
        return None, unrun_below, None
    higher = [f for f in grid if f > f_star]
    mono = all((viol_count(sessions, f, D_us) is None) or viol_count(sessions, f, D_us)["viol"] == 0 for f in higher)
    viol_above = [f for f in higher if viol_count(sessions, f, D_us) is not None and viol_count(sessions, f, D_us)["viol"] > 0]
    return f_star, unrun_below, dict(mono=mono, viol_above=viol_above)


def paired(sel, ref):
    """Paired E(selector) - E(reference): n, mean, sign count, means, % of the mean reference energy."""
    d = np.asarray(sel, float) - np.asarray(ref, float)
    Eref = float(np.mean(ref))
    return dict(same=False, n=len(d), mean=float(d.mean()), pos=int((d > 0).sum()), Eref=Eref,
                Esel=float(np.mean(sel)), rel=100.0 * float(d.mean()) / Eref)


def energy_delta(sessions, f_sel, f_ref):
    """Paired E(f_sel) - E(f_ref) by pair index within session, pooled; None if no common pair."""
    if f_sel == f_ref:
        n = sum(len(s["runs"][f_ref]) for s in sessions if f_ref in s["runs"])
        Eref = [s["E"][f_ref][p] for s in sessions if f_ref in s["runs"] for p in s["runs"][f_ref]]
        return dict(same=True, n=n, mean=0.0, rel=0.0, Eref=float(np.mean(Eref)) if Eref else None)
    a, b, per = [], [], []
    for s in sessions:
        if f_sel in s["runs"] and f_ref in s["runs"]:
            pp = sorted(set(s["runs"][f_sel]) & set(s["runs"][f_ref]))
            per.append((s["name"], len(pp)))
            for p in pp:
                a.append(s["E"][f_sel][p])
                b.append(s["E"][f_ref][p])
    if not a:
        return None
    return dict(paired(a, b), per=per)


def bins(grid, f, g):
    return grid.index(f) - grid.index(g)


def run(which_mnv2=None):
    out = {}
    for model in MODELS:
        cfg = E5.model_cfg(model)
        sessions = sessions_for(cfg, which_mnv2 if model == "mobilenet" else None)
        fits2, fcells, ftag = E5.arm_fits(model)
        d = cells_dict(load_sweep_dir(E5.R33, model))
        P, _, _ = E5.han_fits(d, "abs")
        grid, fmax = cfg["grid"], cfg["fmax"]
        ran = sorted({f for s in sessions for f in s["runs"]})
        rows = []
        for D in cfg["D"]:
            D_us = D * 1000.0
            fF, unrun_below, minfo = frontier(sessions, grid, D_us)
            r = dict(D=D, front=dict(f=fF, feasible=fF is not None, unrun_below=unrun_below, minfo=minfo))
            picks = {}
            picks["blind"] = E5.pick_2cell(fits2["blind"], D_us, grid, fmax)
            picks["aware"] = E5.pick_2cell(fits2["aware"], D_us, grid, fmax)
            picks["H1"] = E5.pick_han(P["H1"], D_us, grid, fmax)
            picks["H2"] = E5.pick_han(P["H2"], D_us, grid, fmax)
            picks["H1c"] = E5.pick_han(P["H1c"], D_us, grid, fmax)
            for k, (f, ok) in picks.items():
                c = dict(f=f, feasible=ok)
                c["vc"] = viol_count(sessions, f, D_us)            # None = clock not run in any session
                if not c["vc"]:
                    c["r33"] = E5.r33_miss(model, f, D_us)          # indicative only
                if ok:
                    if fF is not None:
                        c["bins"] = bins(grid, f, fF)
                        c["dE"] = energy_delta(sessions, f, fF)
                else:                                               # declined: top-clock fallback
                    c["fb_dE_front"] = energy_delta(sessions, f, fF) if fF is not None else None
                    fa, oka = picks["aware"]
                    c["fb_dE_aware"] = energy_delta(sessions, f, fa) if oka else None
                r[k] = c
            # method check: aware - blind paired dE, the Table II / breadth-table energy column
            r["check_aware_minus_blind"] = energy_delta(sessions, picks["aware"][0], picks["blind"][0])
            r["front"]["vc"] = viol_count(sessions, fF, D_us) if fF is not None else None
            r["matrix"] = {f: viol_count(sessions, f, D_us) for f in ran}
            rows.append(r)
        # per-clock energy means
        Ec = {}
        for f in ran:
            e = [s["E"][f][p] for s in sessions if f in s["runs"] for p in s["runs"][f]]
            Ec[f] = (float(np.mean(e)), float(np.std(e, ddof=1)), len(e))
        out[model] = dict(cfg=dict(name=cfg["name"], hz=cfg["hz"], grid=grid, D=cfg["D"], fmax=fmax,
                                   sessions=[s["name"] for s in sessions]),
                          pairs={s["name"]: s["pairs"] for s in sessions}, ran=ran, rows=rows, Ec=Ec, ftag=ftag)
    return out


# ----------------------------------------------------------- aggregate ----
def cell_of(r, s):
    if s != "front":
        return r[s]
    fr = r["front"]
    if fr["f"] is None:
        return dict(f=None, feasible=False, vc=None)
    return dict(f=fr["f"], feasible=True, vc=fr["vc"], bins=0, dE=dict(same=True, n=fr["vc"]["n"], mean=0.0, rel=0.0))


def aggregate(R, sels, models=MODELS):
    A = {}
    for s in sels:
        a = dict(N=0, adm=0, decl=0, decl_front_adm=0, adm_na=0, adm_eval=0, viol=0, false_adm=0, below=0, at=0,
                 above=0, bins_above=[], dE_above=[], dE_above_n=[], dE_above_na=0, dE_below=[], below_viol=0,
                 fb=[], all_bins=[])
        for m in models:
            for r in R[m]["rows"]:
                c = cell_of(r, s)
                fF = r["front"]["f"]
                a["N"] += 1
                if not c["feasible"]:
                    a["decl"] += 1
                    if fF is not None:
                        a["decl_front_adm"] += 1
                    if s != "front":
                        a["fb"].append((m, r["D"], c["f"], c.get("fb_dE_front"), c.get("fb_dE_aware"), c["vc"], fF))
                    continue
                a["adm"] += 1
                if c["vc"] is None:
                    a["adm_na"] += 1
                else:
                    a["adm_eval"] += 1
                    if c["vc"]["viol"] >= 1:
                        a["viol"] += 1
                if fF is None:
                    a["false_adm"] += 1
                    continue
                b = c["bins"]
                a["all_bins"].append(b)
                dE = c.get("dE")
                if b < 0:
                    a["below"] += 1
                    if c["vc"] and c["vc"]["viol"] >= 1:
                        a["below_viol"] += 1
                    if dE:
                        a["dE_below"].append(dE["rel"])
                elif b == 0:
                    a["at"] += 1
                else:
                    a["above"] += 1
                    a["bins_above"].append(b)
                    if dE is None:
                        a["dE_above_na"] += 1
                    else:
                        a["dE_above"].append(dE["rel"])
                        a["dE_above_n"].append(dE["n"])
        A[s] = a
    return A


def med(x):
    return float(np.median(x)) if x else None


# -------------------------------------------------------------- output ----
def row_md(s, a):
    bmed = med(a["bins_above"])
    if a["dE_above"]:
        rng = f" [{min(a['dE_above']):+.1f}, {max(a['dE_above']):+.1f}]" if len(a["dE_above"]) > 1 else ""
        de = f"{med(a['dE_above']):+.1f}{rng} ({len(a['dE_above'])}; {sum(a['dE_above_n'])})"
    else:
        de = "--"
    dec = f"{a['decl']}" + (f" ({a['decl_front_adm']})" if a["decl"] else "")
    return (f"| {NAME[s]} | {a['adm']} | {dec} | {a['adm_eval'] - a['viol']} | {a['viol']} | {a['adm_na']} | "
            f"{a['below']}/{a['at']}/{a['above']} | {'--' if bmed is None else f'{bmed:.0f}'} | {de} |")


def fmt_vc(vc):
    if vc is None:
        return "not run"
    per = "/".join("-" if k is None else str(k) for k in vc["per"])
    return f"{vc['viol']}/{vc['n']}" + (f" ({per})" if len(vc["per"]) > 1 else "")


def fmt_dE(dE):
    if dE is None:
        return "E n/a"
    if dE.get("same"):
        return "dE 0 (same clock)"
    return f"dE {dE['mean']:+.2f} mJ ({dE['rel']:+.1f} %, n={dE['n']}, {dE['pos']}/{dE['n']} pairs > 0)"


def fmt_cell(r, s):
    c = cell_of(r, s)
    fF = r["front"]["f"]
    if s == "front":
        return "none (no run clock clean)" if fF is None else f"{fF} (0/{c['vc']['n']} viol.)"
    if not c["feasible"]:
        if c["vc"] is None:
            return f"declines -> 1173 fallback not run (r33 cell {c.get('r33', float('nan')):.1f} % miss, indicative); E n/a"
        out = f"declines -> 1173 fallback: {fmt_vc(c['vc'])} viol."
        return out + ("; no frontier" if fF is None else "; fallback " + fmt_dE(c.get("fb_dE_front")) + " vs frontier")
    b = c.get("bins")
    bs = "" if b is None else f" ({b:+d})"
    if c["vc"] is None:
        return f"{c['f']}{bs}: not run (r33 cell {c.get('r33', float('nan')):.1f} % miss, indicative); E n/a"
    out = f"{c['f']}{bs}: {fmt_vc(c['vc'])} viol."
    return out + ("; no frontier (false admission)" if fF is None else "; " + fmt_dE(c.get("dE")))


def report(R):
    A = aggregate(R, SEL + ["H1c"])
    Apm = {m: aggregate(R, SEL + ["H1c"], [m]) for m in MODELS}
    hdr = ("| selector | adm. | decl. (front.) | meas. feasible | meas. violating | not measured | below/at/above | "
           "med. bins | dE % med [range] (deadl.; pairs) |")
    sep = "|---|---:|---:|---:|---:|---:|---:|---:|---|"
    print(f"\n## Totals over the {A['front']['N']} deadlines")
    print(hdr)
    print(sep)
    for s in SEL:
        print(row_md(s, A[s]))
    print("sensitivity (not in the table): " + row_md("H1c", A["H1c"]))
    print("bins above the frontier (distribution): " + "; ".join(
        f"{s} " + ", ".join(f"+{k}: {A[s]['bins_above'].count(k)}" for k in sorted(set(A[s]["bins_above"])))
        for s in SEL if A[s]["bins_above"]))
    print("median bins vs frontier over all admitted picks with a frontier (negative = below): " + "; ".join(
        f"{s} {med(A[s]['all_bins']):+.0f} (n={len(A[s]['all_bins'])})" for s in SEL if A[s]["all_bins"]))
    # blind below the frontier, false admissions
    b = A["blind"]
    print(f"\nblind: below the frontier at {b['below']} deadlines ({b['below_viol']} with violating runs), dE vs frontier there "
          f"{min(b['dE_below']):+.1f} to {max(b['dE_below']):+.1f} % (median {med(b['dE_below']):+.1f} %); "
          f"false admissions (no measured-feasible clock) {b['false_adm']}")
    for s in ("H1", "H2", "aware", "H1c"):
        a = A[s]
        meas = [x for x in a["fb"] if x[3] is not None and not x[3].get("same", False)]
        zero = [x for x in a["fb"] if x[3] is not None and x[3].get("same", False)]
        unrun = [x for x in a["fb"] if x[5] is None]
        nofront = [x for x in a["fb"] if x[6] is None]
        fbviol = sum(x[5]["viol"] for x in meas)
        line = (f"{s}: {a['decl']} declines ({a['decl_front_adm']} the frontier admits); 1173 fallback run and priced at "
                f"{len(meas)}")
        if meas:
            line += (f": {min(x[3]['rel'] for x in meas):+.1f} to {max(x[3]['rel'] for x in meas):+.1f} % vs the frontier "
                     f"({min(x[3]['mean'] for x in meas):+.2f} to {max(x[3]['mean'] for x in meas):+.2f} mJ), violating fallback runs {fbviol}")
        line += (f"; fallback at the frontier clock {len(zero)}; fallback clock not run {len(unrun)} (E n/a); "
                 f"no frontier {len(nofront)}")
        print(line)
    print("\n## Per model")
    print(hdr.replace("| selector |", "| model | selector |"))
    print(sep.replace("|---|", "|---|---|", 1))
    for m in MODELS:
        for s in SEL:
            print(f"| {R[m]['cfg']['name']} " + row_md(s, Apm[m][s]))
    print("\n## Checks")
    mono = all(r["front"]["minfo"] is None or r["front"]["minfo"]["mono"] for m in MODELS for r in R[m]["rows"])
    print(f"- frontier monotone at every deadline (no run clock above f_F has a violating run): {mono}")
    for m in MODELS:
        e = [v[0] for f, v in R[m]["Ec"].items()]
        print(f"- {m}: per-clock mean E/inf strictly increasing over measured clocks: {bool(np.all(np.diff(e) > 0))} "
              f"({', '.join(f'{f}:{v[0]:.1f}' for f, v in R[m]['Ec'].items())})")
    meth = []
    for m in MODELS:
        for r in R[m]["rows"]:
            c = r["check_aware_minus_blind"]
            if c and not c.get("same"):
                meth.append(f"{m} {r['D']:.1f}: {c['mean']:+.2f} mJ ({c['pos']}/{c['n']} pairs > 0)")
    print("- aware - blind dE with the same pairing (the Table II / breadth-table energy column): " + "; ".join(meth))
    for m in MODELS:
        M = R[m]
        cfg = M["cfg"]
        print(f"\n### {cfg['name']} ({cfg['hz']:.0f} Hz), sessions {', '.join(cfg['sessions'])}; two-cell arms {M['ftag']}")
        print("runs per clock and session: " + "; ".join(f"{s}: " + ", ".join(f"{f}x{n}" for f, n in M["pairs"][s].items())
                                                        for s in cfg["sessions"])
              + "; clocks never run: " + ", ".join(str(g) for g in cfg["grid"] if g not in M["ran"]))
        print("violating runs / runs per measured clock (all sessions pooled; frontier = first 0 from the left):")
        print("| D (ms) | " + " | ".join(str(f) for f in M["ran"]) + " | frontier |")
        print("|---:|" + "---:|" * len(M["ran"]) + "---:|")
        for r in M["rows"]:
            print(f"| {r['D']:.1f} | " + " | ".join(f"{r['matrix'][f]['viol']}/{r['matrix'][f]['n']}" for f in M["ran"])
                  + f" | {r['front']['f'] if r['front']['f'] else 'none'} |")
        print("| D (ms) | frontier | EMC-blind | Han-H1 | Han-H2 | aware | Han-H1c |")
        print("|---:|---|---|---|---|---|---|")
        for r in M["rows"]:
            print(f"| {r['D']:.1f} | " + " | ".join(fmt_cell(r, s) for s in ["front", "blind", "H1", "H2", "aware", "H1c"]) + " |")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mnv2-sessions", default=None, help="comma list overriding the MobileNetV2 sessions")
    ap.add_argument("--json", action="store_true", help="print the per-deadline record as JSON instead of the tables")
    a = ap.parse_args()
    R = run(a.mnv2_sessions.split(",") if a.mnv2_sessions else None)
    if a.json:
        def conv(o):
            if isinstance(o, dict):
                return {str(k): conv(v) for k, v in o.items()}
            if isinstance(o, (list, tuple)):
                return [conv(x) for x in o]
            if isinstance(o, np.floating):
                return float(o)
            if isinstance(o, np.integer):
                return int(o)
            if isinstance(o, np.bool_):
                return bool(o)
            return o
        print(json.dumps(conv(R), indent=1))
        return
    print(f"# Selector comparison: rule r={E5.R}, tau={TAU} %, deploy EMC {E5.DEPLOY_EMC:g}; Han fitter numpy-grid"
          f"{' + scipy polish' if HAVE_SCIPY else ' (scipy not importable)'}; MobileNetV2 sessions "
          f"{', '.join(R['mobilenet']['cfg']['sessions'])}")
    report(R)


if __name__ == "__main__":
    main()

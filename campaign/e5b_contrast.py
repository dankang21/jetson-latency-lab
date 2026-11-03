#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""E5 contrast fits: what the H1 -> H2 change is made of (2026-10-29).

H2 differs from H1 in two respects at once -- the deployment-point sweep cells enter the training set and the latency
model gains a free additive offset C -- so the H1-to-H2 difference alone does not separate the two.  The sweeps
identify one intermediate fit:

  H1   pure Han form (C = 0)   trained on the 20 rate-matched sweep cells at EMC {665.6, 3199} MHz (deployment point held out)
  H1a  pure Han form (C = 0)   trained on all 30 cells (deployment point in-sample)      <- form fixed, training set changed
  H2   + free offset C         trained on all 30 cells                                  <- offset added to H1a's training set

The converse step (offset added, training set held at the 20 profiling-level cells) is NOT identifiable: with two
memory levels the term B f_emc^-beta + C fits the two level constants exactly for any beta, so the training residual
is the same for every pinned beta and the held-out behaviour at 2133 MHz is set by the pin.  That family (H1c, the
pinned-beta sensitivity fit that campaign/e5_han_baseline.py prints) is scored here over a sweep of beta and reported
for disclosure only.  All fits are scored through the unmodified E5 / E5-B code paths (same sweeps, sessions, rule,
power models, deadline grids); nothing is re-measured.

Outputs: a markdown record with the beta sweep (stdout; --md also writes it to a file) and, with --tex, the
supplementary float (label tab:e5contrast; table*, rows = metrics, columns = fits).
    python3 campaign/e5b_contrast.py [--tex OUT.tex] [--md OUT.md]
Read-only over results/.
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LAB = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from han_fit import load_sweep_dir, cells_dict, fit_han, predict, score, err_stats          # noqa: E402
from e5_han_baseline import (model_cfg, load_sessions, arm_fits, han_fits, pick_2cell,     # noqa: E402
                             pick_han, cell, summary, r33_miss, R33, DEPLOY_EMC, MODELS, fmt_params)
import e5b_emcfree as EF                                                                     # noqa: E402
from e1b_stats import GRID as TABLE2_GRID                                                    # noqa: E402

ARMS = ["H1", "H1a", "H1c", "H2"]
BETAS = [0.523, 0.681, 1.0, 1.268, 1.5, 2.0]     # H1's own beta, H1a's, the printed pin, H2's, and two larger pins
PRETTY = {"mobilenet": "MobileNetV2", "vit": "ViT-Small", "yolov8s": "YOLOv8s", "resnet50": "ResNet-50"}


def fits_for(model, beta_pin=1.0):
    d = cells_dict(load_sweep_dir(R33, model))
    assert len(d) == 30, (model, len(d))
    P, train1, hold = han_fits(d, "abs")
    P["H1a"] = fit_han(d, train=None, with_c=False, weight="abs")
    if beta_pin != 1.0:
        P["H1c"] = fit_han(d, train=train1, with_c=True, weight="abs", fix_beta=beta_pin)
    assert P["H1a"]["C"] == 0.0 and P["H1a"]["n_train"] == 30 and P["H1"]["n_train"] == 20 and P["H1"]["C"] == 0.0
    return d, P, train1, hold


def fixed_emc(beta_pin=1.0):
    tot = {a: dict(N=0, eval=0, na=0, viol=0, pool=0, decl=0, same=0, higher=0, lower=0, decl_aware_admits=0)
           for a in ARMS + ["aware"]}
    errs, declined = {}, {a: [] for a in ARMS}
    for model in MODELS:
        cfg = model_cfg(model)
        sessions = load_sessions(cfg)
        fits2, _, _ = arm_fits(model)
        d, P, train1, hold = fits_for(model, beta_pin)
        errs[model] = {a: err_stats(score(P[a], d, hold)) for a in ARMS}
        errs[model]["signed"] = {a: float(np.mean([r[3] for r in score(P[a], d, hold)]) * 100) for a in ARMS}
        indic = lambda f, D_us, model=model: r33_miss(model, f, D_us)  # noqa: E731
        rows = []
        for D in cfg["D"]:
            D_us = D * 1000.0
            row = dict(D=D)
            f, ok = pick_2cell(fits2["aware"], D_us, cfg["grid"], cfg["fmax"])
            row["aware"] = cell(sessions, f, ok, D_us, indic)
            for a in ARMS:
                f, ok = pick_han(P[a], D_us, cfg["grid"], cfg["fmax"])
                row[a] = cell(sessions, f, ok, D_us, indic)
                if not ok:
                    declined[a].append((model, D))
            rows.append(row)
        for a in ["aware"] + ARMS:
            S = summary(rows, a)
            for k in tot[a]:
                tot[a][k] += S[k]
    return tot, errs, declined


def emc_free(beta_pin=1.0):
    P, _, _ = EF.han_latency()
    _, Pm, _, _ = fits_for("mobilenet", beta_pin)
    P["H1a"] = Pm["H1a"]; P["H1c"] = Pm["H1c"]
    runs = EF.load_recorded(); live = EF.load_live()
    cell_power = {k: float(np.mean([r["E"] * EF.HZ for r in v])) for k, v in runs.items()}
    keys_all = sorted(k for k in cell_power if k[0] in EF.EMC_LEVELS and k[1] in EF.GPU_GRID)
    keys_h1 = [k for k in keys_all if abs(k[0] - DEPLOY_EMC) > 1e-6]
    PW = {"H1": EF.fit_power(cell_power, keys_h1), "H2": EF.fit_power(cell_power, keys_all)}
    fits2, _, _ = arm_fits("mobilenet")
    out = {a: dict(picks={}, admitted=0, escapes=[], same_as_aware=0, live_ok=True, viol=0, pw_same=True, live_max={}) for a in ARMS}
    for D in EF.DEADLINES:
        D_us = D * 1000.0
        ga, oka = pick_2cell(fits2["aware"], D_us, list(TABLE2_GRID), 1173)
        for a in ARMS:
            pk1 = EF.joint_pick(P[a], PW["H1"], D_us); pk2 = EF.joint_pick(P[a], PW["H2"], D_us)
            if (pk1 is None) != (pk2 is None) or (pk1 and (pk1["emc"], pk1["g"]) != (pk2["emc"], pk2["g"])):
                out[a]["pw_same"] = False
            pk = pk2
            if pk is None:
                out[a]["picks"][D] = None
                continue
            sc = EF.score(runs.get((pk["emc"], pk["g"]), []), D_us)
            assert sc is not None, (a, D, pk)
            lv = live.get((pk["emc"], pk["g"]), [])
            lsc = EF.score(lv, D_us) if lv else None
            out[a]["admitted"] += 1
            out[a]["viol"] += int(sc["viol"] > 0) + int(bool(lsc and lsc["viol"] > 0))
            if abs(pk["emc"] - 3199.0) < 1e-6:
                out[a]["escapes"].append(D)
                out[a]["live_ok"] &= lsc is not None
                if lsc:
                    out[a]["live_max"][D] = lsc["max_miss"]
            if oka and abs(pk["emc"] - DEPLOY_EMC) < 1e-6 and pk["g"] == ga:
                out[a]["same_as_aware"] += 1
            out[a]["picks"][D] = dict(emc=pk["emc"], g=pk["g"], rec=(sc["viol"], sc["n"]), live=(lsc["viol"], lsc["n"]) if lsc else None)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tex", default=None, help="write the LaTeX table (tab:e5contrast) here")
    ap.add_argument("--md", default=None, help="write the markdown record here as well")
    a = ap.parse_args()
    tot, errs, declined = fixed_emc()
    ef = emc_free()
    for arm in ARMS:
        assert ef[arm]["pw_same"], f"{arm}: picks differ between the two power models"
        assert ef[arm]["viol"] == 0, f"{arm}: a pick has a violating recorded or re-run run"
        assert ef[arm]["live_ok"], f"{arm}: a 3199 MHz pick lies on a cell the re-run did not cover"
        assert tot[arm]["viol"] == 0, f"{arm}: a fixed-EMC pick has a violating run"
    assert sorted(declined["H1a"]) == sorted(declined["H2"]), "H1a and H2 do not decline the same deadlines"
    # beta sweep of the pinned family
    sweep = []
    for b in BETAS:
        t, e, _ = fixed_emc(b); f = emc_free(b)
        sweep.append(dict(beta=b, admitted=t["H1c"]["eval"] + t["H1c"]["na"], same=t["H1c"]["same"], lower=t["H1c"]["lower"],
                          viol=t["H1c"]["viol"], err=e["mobilenet"]["H1c"], escapes=f["H1c"]["escapes"], adm_free=f["H1c"]["admitted"]))
    sw_adm = (min(s["admitted"] for s in sweep), max(s["admitted"] for s in sweep)); sw_same = (min(s["same"] for s in sweep), max(s["same"] for s in sweep))
    _, Pm, _, _ = fits_for("mobilenet")
    h1a44 = ef["H1a"]["live_max"].get(4.4)
    # ---------------------------------------------------------------- table (rows = metrics, columns = fits)
    def ferr(model, arm):
        e = errs[model][arm]; return f"{e[0]:.1f}\\,/\\,{e[1]:.1f}"
    rows = [
        ("form", ["pure ($C{=}0$)", "pure ($C{=}0$)", "$+C$, $\\beta$ pinned at $1.0$", "$+C$"]),
        ("training cells", ["20, 2133 held out", "30, 2133 in", "20, 2133 held out", "30, 2133 in"]),
        ("MobileNetV2 $|$err$|$ at 2133\\,MHz, median\\,/\\,max (\\%)", [ferr("mobilenet", a) for a in ARMS]),
        ("mean signed error at 2133\\,MHz (\\%)", [f"${errs['mobilenet']['signed'][a]:+.1f}$" for a in ARMS]),
        ("median $|$err$|$, ViT-S\\,/\\,YOLOv8s\\,/\\,ResNet-50 (\\%)", [f"{errs['vit'][a][0]:.1f}\\,/\\,{errs['yolov8s'][a][0]:.1f}\\,/\\,{errs['resnet50'][a][0]:.1f}" for a in ARMS]),
        ("fixed EMC, 37 deadlines: admitted (measured feasible\\,/\\,not measured)", [f"{tot[a]['eval'] + tot[a]['na']} ({tot[a]['eval'] - tot[a]['viol']}\\,/\\,{tot[a]['na']})" for a in ARMS]),
        ("declined (of which the refit admits)", [f"{tot[a]['decl']} ({tot[a]['decl_aware_admits']})" for a in ARMS]),
        ("admitted picks at\\,/\\,above\\,/\\,below the refit's clock", [f"{tot[a]['same']}\\,/\\,{tot[a]['higher']}\\,/\\,{tot[a]['lower']}" for a in ARMS]),
        ("EMC free, MobileNetV2, 9 deadlines: admitted", [f"{ef[a]['admitted']}" for a in ARMS]),
        ("deadlines picked at 3199\\,MHz ($D$, ms)", [", ".join(f"{D:.1f}" for D in ef[a]["escapes"]) or "none" for a in ARMS]),
        ("picks on the refit's cell", [f"{ef[a]['same_as_aware']}" for a in ARMS]),
    ]
    md = ["## E5 contrast fits (campaign/e5b_contrast.py; do not edit by hand)", "", "| metric | " + " | ".join(ARMS) + " |", "|---|" + "---|" * len(ARMS)]
    for lab, vals in rows:
        md.append(f"| {lab} | " + " | ".join(vals) + " |")
    md += ["", "Coefficients (us; f in GHz): " + "; ".join(f"{arm}: {fmt_params(Pm[arm])}" for arm in ARMS), "",
           "Pinned-beta family (H1c, 20 cells at {665.6, 3199} MHz, C free, beta pinned): training residual identical for every pin;",
           "| beta | C (us) | MNv2 held-out med/max % | fixed-EMC admitted | at refit's clock | below | EMC-free admitted | 3199 picks |", "|---:|---:|---|---:|---:|---:|---:|---|"]
    for s in sweep:
        md.append(f"| {s['beta']} | {fits_for('mobilenet', s['beta'])[1]['H1c']['C']:.1f} | {s['err'][0]:.1f}/{s['err'][1]:.1f} | {s['admitted']} | {s['same']} | {s['lower']} | {s['adm_free']}/9 | {', '.join(f'{D:.1f}' for D in s['escapes'])} |")
    md += ["", "EMC-free picks (MobileNetV2, 33 Hz; cell = EMC/GPU MHz, recorded runs over tau / n, re-run runs over tau / n where the cell was re-run):"]
    for D in EF.DEADLINES:
        md.append(f"- D = {D:.1f} ms: " + "; ".join(
            f"{arm} " + ("declines" if ef[arm]["picks"][D] is None else
                         f"{int(ef[arm]['picks'][D]['emc']) if ef[arm]['picks'][D]['emc'] != 665.6 else '665.6'}/{ef[arm]['picks'][D]['g']} "
                         f"({ef[arm]['picks'][D]['rec'][0]}/{ef[arm]['picks'][D]['rec'][1]}" + (f", live {ef[arm]['picks'][D]['live'][0]}/{ef[arm]['picks'][D]['live'][1]}" if ef[arm]['picks'][D]['live'] else "") + ")")
            for arm in ARMS))
    md.append(f"\nH1a's 4.4 ms cell 3199/1122 MHz in the re-run: largest run miss {h1a44:.2f} % (0/10 over tau)")
    # error reduction H1 -> H1a at 2133 MHz: the stated range covers both the unrounded medians and the one-decimal
    # medians the tables print (2026-10-30: 44.6-60.2 % exact, 44.4-60.6 % from printed values)
    _red = [100.0 * (1.0 - errs[m]["H1a"][0] / errs[m]["H1"][0]) for m in MODELS]
    _red += [100.0 * (1.0 - round(errs[m]["H1a"][0], 1) / round(errs[m]["H1"][0], 1)) for m in MODELS]
    red_lo, red_hi = int(np.floor(min(_red))), int(np.ceil(max(_red)))
    n_h1 = tot['H1']['eval'] + tot['H1']['na']; n_h1a = tot['H1a']['eval'] + tot['H1a']['na']; n_h2 = tot['H2']['eval'] + tot['H2']['na']
    na = {a: tot[a]['na'] for a in ARMS}
    cap = (
        "Contrast fits behind the H1-to-H2 comparison (NX, the rate-matched sweeps of Table~\\ref{tab:e5han}; rows are "
        "metrics, columns the fits). H2 differs from H1 in two respects at once --- the deployment-point cells enter the "
        "training set and the latency model gains a free offset $C$ --- and the sweeps identify one intermediate fit: H1a "
        "keeps the pure form and adds the deployment point, so H1${\\to}$H1a is the training-set step and H1a${\\to}$H2 the "
        "offset step at the same data. The converse step is not identifiable: with two memory levels the term "
        "$B f_{\\mathrm{emc}}^{-\\beta} + C$ fits the two level constants exactly for any $\\beta$, so the training residual is "
        "the same for every pinned $\\beta$ and the H1c column is set by the pin (the pinned-$\\beta$ sensitivity variant the "
        f"released E5 script prints; over a common pin $\\beta = 0.52$--$2.0$ for all four models it admits {sw_adm[0]}--{sw_adm[1]} "
        f"of the 37 deadlines and matches the refit at {sw_same[0]}--{sw_same[1]}, with no violating pick at any pin); it is shown "
        "for disclosure, not as a contrast. All four are scored through the unchanged E5 rule (fixed memory point, 37 "
        "pre-stated NX deadlines against the per-EMC refit; the MobileNetV2 picks on the three Table~II sessions) and the "
        "unchanged E5-B selector (memory point free, nine MobileNetV2 deadlines; both power models give the same picks); "
        "nothing was re-measured, every EMC-free pick lies on a recorded cell and every 3199\\,MHz pick on a re-run cell, and "
        "no measured pick of any fit has a run over $\\tau$ (the 1173\\,MHz picks the breadth sessions did not run --- "
        f"{na['H1']}, {na['H1a']}, {na['H1c']} and {na['H2']} --- count as not measured; H1a's $4.4$\\,ms cell, 3199/1122\\,MHz, "
        f"a deadline the refit declines: largest per-run miss {h1a44:.1f}\\% in the EMC-free re-run of "
        f"Table~\\ref{{tab:e5blive}}, which had run that cell as an H1/H2 pick before these fits were made). The identified path is the basis "
        f"of ``chiefly conditioning'' in the main text, stated for admission coverage: the training-set step raises the "
        f"admitted deadlines from {n_h1} to {n_h1a} and the picks at the refit's clock from {tot['H1']['same']} to "
        f"{tot['H1a']['same']}, and cuts the median error at 2133\\,MHz by {red_lo}--{red_hi}\\% on every model; the offset "
        f"step keeps the {n_h2} admitted deadlines and moves nine picks one bin, seven onto the refit's clock "
        f"({tot['H2']['same']}) and two below it, so in pick agreement with the refit ({tot['H1']['same']}, {tot['H1a']['same']} "
        f"and {tot['H2']['same']} of its 36) the offset and the two-cell form account for about as much as the training set; "
        "in the EMC-free replay the offset step gives up the $4.4$\\,ms admission, moves the $4.6$\\,ms pick one GPU bin up "
        "and the $5.2$ and $5.8$\\,ms picks onto the refit's cell. Rows: median\\,/\\,max $|$error$|$ and mean signed error "
        "(positive: predicted above measured) of the MobileNetV2 fit at EMC 2133\\,MHz over the ten GPU clocks, and the "
        "median $|$error$|$ for the other three models; fixed-EMC deadlines admitted (measured feasible\\,/\\,pick at the "
        "1173\\,MHz clock the breadth sessions did not run) and declined (of which the refit admits); admitted picks at, above "
        "and below the refit's clock; EMC-free deadlines admitted, the deadlines at which the selector picks a 3199\\,MHz cell, "
        "and the picks that coincide with the refit's cell.")
    assert (n_h1, n_h1a, n_h2) == (20, 33, 33) and (tot['H1']['same'], tot['H1a']['same'], tot['H2']['same']) == (1, 18, 23), "contrast totals moved"
    assert (red_lo, red_hi) == (44, 61), (red_lo, red_hi)
    L = [r"\begin{table*}[!t]", r"  \centering", "  \\caption{" + cap + "}", r"  \label{tab:e5contrast}",
         r"  \footnotesize", r"  \setlength{\tabcolsep}{5pt}",
         r"  \begin{tabular}{@{}lllll@{}}", r"    \toprule",
         r"    & H1 & H1a & H1c (pinned $\beta$, disclosure) & H2 \\", r"    \midrule"]
    for lab, vals in rows:
        L.append(f"    {lab} & " + " & ".join(vals) + " \\\\")
    L += [r"    \bottomrule", r"  \end{tabular}", r"\end{table*}"]
    if a.tex:
        open(a.tex, "w").write("\n".join(L) + "\n")
    if a.md:
        open(a.md, "w").write("\n".join(md) + "\n")
    print("\n".join(md))
    written = [p for p in (a.tex, a.md) if p]
    if written:
        print(f"\nwrote {' and '.join(written)}")


if __name__ == "__main__":
    main()

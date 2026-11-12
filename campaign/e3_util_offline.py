#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Offline evaluation of a proposed utilization guard u(f) <= 1 on the E3 Nano picks, and of the
response-time stretch of the integer-period-multiple conditions (main text Sec. VIII-A, Supp. S2
phase-locking detail).  A post-hoc analysis of a proposed extension: not part of the frozen rule,
not deployed, no new runs.

Read-only over results/ (python3 + numpy).  Run from anywhere:
    python3 campaign/e3_util_offline.py

  u(f)  = [ T_p(f) * rate_p  +  T_t(f) * rate_t ] / 1e6        (T in us, rate in Hz)
  f_u   = lowest f in G = {306, ..., 1020} with u(f) <= theta (theta = 1 unless swept)
  guard pick = max(f_b, f_u)        (f_b = the committed banded pick, t<k>.pick.json)
T_p(f) = k/f + b is the two-cell endpoint refit the pick uses (campaign/admit_pick.py fit_2cell, p50 of
the two extreme GPU cells at EMC 2133: results/nano_grid/<primary>/est_emc2133_gpu{306,1020}.json).
No Nano sweep exists for either tenant, so T_t(f) is the same form fitted to the tenant's NX cells at
EMC 2133 (results/nx_validation/nx_est_emc2133_gpu{306,1173}_<tenant>.json) at scale S = 1.00: ViT-S,
the one model with like-for-like sweeps on both boards, gives Nano/NX p50 ratios of 0.987-1.003
(printed); +-10 % and a 306/1020 tenant refit are printed as sensitivities.  No safety multiplier and
no band enter u(f) (the R = 1.035 variant is printed).  Picks, contract misses (response_us > D,
tau = 2 %) and reference sweeps are those of results/e3_nano/<cond>/ (t<k>.pick.json,
t<k>.contract_b.csv, ref_g<f>.csv); A5/A6 are the superseded first runs of A5r/A6r, shown apart.
Also printed: the range of u over the stretched picks (failing, not raised by the guard) and at the
reference frontier, alternative response-time idealisations per condition with the bin distance of the
2 T_p(f) R <= D term from the reference frontier, and the stretch itself: 2 T_p(f) of the ViT-S primary
against the median response of each A5/A6/A5r/A6r reference contract at f.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LAB = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from admit_pick import load_probe, load_cells, fit_2cell  # noqa: E402

GRID = [306, 408, 510, 612, 714, 816, 918, 1020]
TAU = 0.02
R = 1.035
COND = {  # cond: (primary, pri_hz, tenant, ten_hz, D_us, ref_dir)
    "A1": ("mobilenet", 33, "yolov8s", 5, 13000, "A1"),
    "A2": ("mobilenet", 33, "yolov8s", 10, 13000, "A2"),
    "A3": ("mobilenet", 33, "resnet50", 5, 13000, "A3"),
    "A4": ("mobilenet", 33, "resnet50", 10, 13000, "A4"),
    "A5": ("vit", 20, "yolov8s", 5, 37500, "A5"),
    "A6": ("vit", 20, "yolov8s", 10, 37500, "A6"),
    "A5r": ("vit", 20, "yolov8s", 5, 37500, "A5r"),   # own reference sweep
    "A6r": ("vit", 20, "yolov8s", 10, 37500, "A6r"),  # own reference sweep
}
PRIM_CELLS = {
    "mobilenet": [f"{LAB}/results/nano_grid/mobilenet/est_emc2133_gpu306.json", f"{LAB}/results/nano_grid/mobilenet/est_emc2133_gpu1020.json"],
    "vit": [f"{LAB}/results/nano_grid/vit/est_emc2133_gpu306.json", f"{LAB}/results/nano_grid/vit/est_emc2133_gpu1020.json"],
}
TEN_CELLS = {  # NX stand-in (no Nano sweep exists for either tenant)
    "yolov8s": [f"{LAB}/results/nx_validation/nx_est_emc2133_gpu306_yolov8s.json", f"{LAB}/results/nx_validation/nx_est_emc2133_gpu1173_yolov8s.json"],
    "resnet50": [f"{LAB}/results/nx_validation/nx_est_emc2133_gpu306_resnet50.json", f"{LAB}/results/nx_validation/nx_est_emc2133_gpu1173_resnet50.json"],
}
TEN_CELLS_1020 = {k: [v[0], v[1].replace("gpu1173", "gpu1020")] for k, v in TEN_CELLS.items()}
S = 1.0


def model(files, scale=1.0):
    c = load_cells(files)
    k, b, lo, hi = fit_2cell(c)
    return dict(k=k * scale, b=b * scale, lo=lo, hi=hi, cells={g: c[g] * scale for g in c})


def That(m, f):
    return m["k"] * 1e3 / f + m["b"]


PM = {p: model(PRIM_CELLS[p]) for p in PRIM_CELLS}
TM = {t: model(TEN_CELLS[t], S) for t in TEN_CELLS}


def u_of(cond, f, tm=None, pm=None, rmult=1.0):
    p, rp, t, rt, D, _ = COND[cond]
    pm = pm or PM[p]; tm = tm or TM[t]
    return (That(pm, f) * rmult * rp + That(tm, f) * rmult * rt) / 1e6


def f_u(cond, theta=1.0, **kw):
    for f in GRID:
        if u_of(cond, f, **kw) <= theta:
            return f
    return None


def score(path, D):
    v = load_probe(path)
    return dict(n=len(v), miss=float(np.mean(v > D)), p50=float(np.median(v)), p90=float(np.percentile(v, 90)), max=float(v.max()))


def f_oracle(refc):
    for f in GRID:
        if f in refc and refc[f]["miss"] <= TAU:
            return f
    return None


def main():
    argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    # ---- Nano/NX scale from ViT (same model, same EMC, both boards)
    nx_vit = load_cells(glob.glob(f"{LAB}/results/nx_validation/nx_est_emc2133_gpu*_vit.json"))
    na_vit = load_cells(glob.glob(f"{LAB}/results/nano_grid/vit/est_emc2133_gpu*.json"))
    nx_mn = load_cells(glob.glob(f"{LAB}/results/nx_validation/nx_est_emc2133_gpu*_mobilenet.json"))
    na_mn = load_cells(glob.glob(f"{LAB}/results/nano_grid/mobilenet/est_emc2133_gpu*.json"))
    ratios_vit = {g: na_vit[g] / nx_vit[g] for g in GRID}
    ratios_mn = {g: na_mn[g] / nx_mn[g] for g in GRID}
    S_VIT = float(np.exp(np.mean(np.log([ratios_vit[306], ratios_vit[1020]]))))
    print("Nano/NX p50 ratio ViT-S per clock:", {g: round(r, 4) for g, r in ratios_vit.items()})
    print("Nano/NX p50 ratio MNv2 per clock (NX cells 8 Hz/300 iters, not like-for-like):", {g: round(r, 4) for g, r in ratios_mn.items()})
    print(f"S (geo-mean of ViT endpoint ratios 306/1020) = {S_VIT:.4f}; using S = 1.00 nominal, +-10 % sensitivity")
    for name, m in list(PM.items()) + list(TM.items()):
        print(f"model {name:10s} cells {m['lo']}/{m['hi']} p50 {m['cells'][m['lo']]:.0f}/{m['cells'][m['hi']]:.0f} us  k={m['k']:.0f} us*GHz  b={m['b']:.0f} us")

    refs = {}
    for c in ["A1", "A2", "A3", "A4", "A5", "A6", "A5r", "A6r"]:
        D = COND[c][4]
        refs[c] = {}
        for f in GRID:
            p = f"{LAB}/results/e3_nano/{c}/ref_g{f}.csv"
            if os.path.exists(p):
                refs[c][f] = score(p, D)

    # ---- per-condition u(f) table
    print("\n== u(f) per condition (solo-additive, S=1, no R) ==")
    hdr = "cond  prim/rate     tenant/rate   " + " ".join(f"{f:>6d}" for f in GRID) + "   f_u(1.0)  f_oracle"
    print(hdr)
    for c in ["A1", "A2", "A3", "A4", "A5r", "A6r"]:
        p, rp, t, rt, D, rd = COND[c]
        row = " ".join(f"{u_of(c, f):6.3f}" for f in GRID)
        print(f"{c:5s} {p:9s}{rp:3d}Hz  {t:9s}{rt:3d}Hz  {row}   {f_u(c)!s:>7}  {f_oracle(refs[rd])}")

    # ---- per-trial table
    print("\n== per-trial ==")
    rows = []
    for c in ["A1", "A2", "A3", "A4", "A5r", "A6r", "A5", "A6"]:
        p, rp, t, rt, D, rd = COND[c]
        fu = f_u(c)
        for k in range(1, 11):
            pk = f"{LAB}/results/e3_nano/{c}/t{k}.pick.json"
            if not os.path.exists(pk):
                continue
            j = json.load(open(pk))
            fb = j["f_b"]
            cb = f"{LAB}/results/e3_nano/{c}/t{k}.contract_b.csv"
            sc = score(cb, D) if (fb != "INFEASIBLE" and os.path.exists(cb)) else None
            if fb == "INFEASIBLE":
                rows.append(dict(cond=c, t=k, Mg=j["Mg_us"], fb=fb, u_fb=None, fu=fu, m_b=None, p50=None, raised=None, ref_fb=None, ref_raised=None))
                continue
            raised = max(fb, fu) if fu else None
            rows.append(dict(cond=c, t=k, Mg=j["Mg_us"], fb=fb, u_fb=u_of(c, fb), fu=fu, m_b=sc["miss"], p50=sc["p50"], max=sc["max"],
                             raised=raised, changed=(raised != fb),
                             ref_fb=refs[rd].get(fb, {}).get("miss"), ref_raised=refs[rd].get(raised, {}).get("miss") if raised else None))
    for r in rows:
        if r["fb"] == "INFEASIBLE":
            print(f"{r['cond']:4s} t{r['t']:2d} Mg={r['Mg']:6.0f} f_b=INFEASIBLE"); continue
        print(f"{r['cond']:4s} t{r['t']:2d} Mg={r['Mg']:6.0f} f_b={r['fb']:4d} u={r['u_fb']:.3f} f_u={r['fu']} m_b={r['m_b']:.3f} p50={r['p50']/1e3:8.1f}ms "
              f"raised->{r['raised']} {'CHANGED' if r['changed'] else 'same'} ref@f_b={r['ref_fb']} ref@raised={r['ref_raised']}")

    # ---- summary
    print("\n== summary (theta = 1) ==")
    for grp, cs in [("A5r+A6r", ["A5r", "A6r"]), ("A1-A4", ["A1", "A2", "A3", "A4"]), ("A5+A6 (superseded first runs)", ["A5", "A6"])]:
        rr = [r for r in rows if r["cond"] in cs and r["fb"] != "INFEASIBLE"]
        fail = [r for r in rr if r["m_b"] > TAU]
        raised = [r for r in rr if r["changed"]]
        fail_raised = [r for r in fail if r["changed"]]
        fail_raised_pass = [r for r in fail_raised if r["ref_raised"] is not None and r["ref_raised"] <= TAU]
        pass_raised = [r for r in raised if r["m_b"] <= TAU]
        print(f"{grp}: admitted {len(rr)}, failing (m_b>tau) {len(fail)}, guard raises {len(raised)} picks "
              f"(of which failing {len(fail_raised)}, raised to a ref-passing clock {len(fail_raised_pass)}; passing picks raised (cost) {len(pass_raised)})")
        untouched = [r for r in fail if not r["changed"]]
        print("   failing picks left untouched:", [(r["cond"], r["t"], r["fb"], round(r["u_fb"], 3), r["m_b"]) for r in untouched])
        if untouched:
            print(f"   u over the failing picks left untouched: {min(r['u_fb'] for r in untouched):.3f}-{max(r['u_fb'] for r in untouched):.3f} "
                  f"({len(untouched)} picks); over the raised ones: "
                  + (f"{min(r['u_fb'] for r in fail_raised):.3f}-{max(r['u_fb'] for r in fail_raised):.3f}" if fail_raised else "none"))

    # ---- threshold sweep
    print("\n== threshold sweep ==")
    print("theta   " + "  ".join(f"{c}:f_u" for c in ["A1", "A2", "A3", "A4", "A5r", "A6r"]) + "   A5r+A6r: fail raised / to-pass-ref | A1-A4: picks changed (of which passing)")
    for th in [1.0, 0.9, 0.8, 0.7, 0.6, 0.55, 0.5, 0.45, 0.4, 0.35]:
        fus = {c: f_u(c, th) for c in COND}
        line = f"{th:5.2f}   " + "  ".join(f"{fus[c]!s:>6}" for c in ["A1", "A2", "A3", "A4", "A5r", "A6r"])
        rr = [r for r in rows if r["cond"] in ("A5r", "A6r") and r["fb"] != "INFEASIBLE"]
        fail = [r for r in rr if r["m_b"] > TAU]
        fr = [r for r in fail if fus[r["cond"]] and fus[r["cond"]] > r["fb"]]
        frp = [r for r in fr if refs[COND[r["cond"]][5]].get(fus[r["cond"]], {}).get("miss", 1) <= TAU]
        r14 = [r for r in rows if r["cond"] in ("A1", "A2", "A3", "A4") and r["fb"] != "INFEASIBLE"]
        ch = [r for r in r14 if fus[r["cond"]] and fus[r["cond"]] > r["fb"]]
        chp = [r for r in ch if r["m_b"] <= TAU]
        print(f"{line}   {len(fr)}/{len(fail)} / {len(frp)} | {len(ch)}/{len(r14)} ({len(chp)})")

    # ---- u at empirical frontier
    print("\n== u at the reference frontier ==")
    uo = []
    for c in ["A1", "A2", "A3", "A4", "A5r", "A6r"]:
        rd = COND[c][5]; fo = f_oracle(refs[rd])
        i = GRID.index(fo); fm = GRID[i - 1]
        uo.append(u_of(c, fo))
        print(f"{c:4s} f_oracle={fo} u={u_of(c, fo):.3f} ref_miss={refs[rd][fo]['miss']:.3f} | one bin below f={fm} u={u_of(c, fm):.3f} ref_miss={refs[rd][fm]['miss']:.3f} ref_p50={refs[rd][fm]['p50']/1e3:.1f}ms")
    print(f"u at the reference frontier over the six conditions: {min(uo):.3f}-{max(uo):.3f}")

    # ---- stability check: u vs contract/ref p50 at low clocks
    print("\n== queue stability vs u (ref sweeps A1-A4, A5r, A6r at 306..612) ==")
    for c in ["A1", "A2", "A3", "A4", "A5r", "A6r"]:
        rd = COND[c][5]
        for f in [306, 408, 510, 612]:
            s = refs[rd][f]
            print(f"{c:4s} f={f} u={u_of(c, f):.3f} ref p50={s['p50']/1e3:9.1f} ms max={s['max']/1e3:9.1f} ms miss={s['miss']:.3f}")

    # ---- the stretch: an overlapped ViT-S launch runs at about twice its solo time
    print("\n== stretch: 2*T_p(f) of the ViT-S primary vs the reference-contract median at f (ms) ==")
    print("   f   T_p   2*T_p  " + "  ".join(f"{c:>4s} p50" for c in ["A5", "A6", "A5r", "A6r"]))
    for f in [510, 612, 714, 816]:
        t = That(PM["vit"], f)
        print(f"{f:4d} {t/1e3:5.1f}  {2*t/1e3:5.1f}  " + "  ".join(
            f"{refs[c][f]['p50']/1e3:8.1f}" if f in refs[c] else f"{'-':>8s}" for c in ["A5", "A6", "A5r", "A6r"]))

    # ---- alternative idealisations (exploratory)
    print("\n== alternative response-time idealisations (exploratory, per condition) ==")
    print("cond  f_oracle  u<=1  PS: T_p*R/(1-rho_t)<=D  stretch2: 2*T_p*R<=D  block: T_p*R+T_t<=D")
    st_bins = []
    for c in ["A1", "A2", "A3", "A4", "A5r", "A6r"]:
        p, rp, t, rt, D, rd = COND[c]
        def first(pred):
            for f in GRID:
                if pred(f): return f
            return None
        ps = first(lambda f: That(PM[p], f) * R / (1 - That(TM[t], f) * rt / 1e6) <= D if That(TM[t], f) * rt / 1e6 < 1 else False)
        st = first(lambda f: 2 * That(PM[p], f) * R <= D)
        bl = first(lambda f: That(PM[p], f) * R + That(TM[t], f) <= D)
        fo = f_oracle(refs[rd])
        print(f"{c:4s}  {fo!s:>7}  {f_u(c)!s:>4}  {ps!s:>22}  {st!s:>20}  {bl!s:>18}")
        st_bins.append((c, None if (st is None or fo is None) else GRID.index(st) - GRID.index(fo)))
    sb = [b for _, b in st_bins if b is not None]
    print("stretch2 - f_oracle (bins): " + ", ".join(f"{c} {b:+d}" if b is not None else f"{c} -" for c, b in st_bins)
          + f"; within one bin {sum(1 for b in sb if abs(b) <= 1)}/{len(st_bins)}: at the frontier {sb.count(0)}, "
            f"one bin optimistic (below) {sb.count(-1)}, one bin conservative (above) {sb.count(1)}")

    # ---- sensitivity
    print("\n== sensitivity of f_u(theta=1) ==")
    for label, kw in [("S=0.9", dict(scale=0.9)), ("S=1.1", dict(scale=1.1)), ("tenant endpoints 306/1020", dict(files=TEN_CELLS_1020)), ("with R=1.035 on both terms", dict(rmult=R))]:
        out = {}
        for c in ["A1", "A2", "A3", "A4", "A5r", "A6r"]:
            t = COND[c][2]
            if "scale" in kw:
                tm = model(TEN_CELLS[t], kw["scale"]); out[c] = f_u(c, tm=tm)
            elif "files" in kw:
                tm = model(kw["files"][t], S); out[c] = f_u(c, tm=tm)
            else:
                out[c] = f_u(c, rmult=kw["rmult"])
        print(f"{label:28s}", out)
        if "files" in kw:
            for t in TEN_CELLS:
                m = model(kw["files"][t], S); print(f"    {t} 306/1020 fit k={m['k']:.0f} b={m['b']:.0f}")
    print("u(f) with R on both terms:", {c: {f: round(u_of(c, f, rmult=R), 3) for f in [306, 408, 510, 612]} for c in ["A5r", "A6r"]})
    print("two-cell vs measured NX yolov8s cell at 510/612:", That(TM["yolov8s"], 510), That(TM["yolov8s"], 612),
          load_cells([f"{LAB}/results/nx_validation/nx_est_emc2133_gpu510_yolov8s.json", f"{LAB}/results/nx_validation/nx_est_emc2133_gpu612_yolov8s.json"]))


if __name__ == "__main__":
    main()

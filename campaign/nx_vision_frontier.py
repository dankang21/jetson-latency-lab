#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""NX vision deadlines: admission of the per-EMC two-cell refit, its run-level compliance and its
position against the measured frontier, recomputed from the per-run CSVs.

An independent re-implementation (no campaign imports) of the pick and the scoring of main-text
Table II and the breadth table, so it cross-checks campaign/e1b_stats.py and the Table II picks.
Read-only over results/ (python3 + numpy).  Run from anywhere:
    python3 campaign/nx_vision_frontier.py [--json]

Inputs:
  deployment runs  results/govrep_rev/<session>/g<f>_p<k>.csv  (column response_us, 1000 rows)
     MobileNetV2 33 Hz: sessions nx, nx_mobilenet, nx_mobilenet_s3
     ViT-S 20 Hz / YOLOv8s 10 Hz / ResNet-50 20 Hz: nx_vit, nx_yolov8s, nx_resnet50 (one each)
  rate-matched endpoint cells  results/refit33/<dir>/emc{2133,3199}_g{306,1173}_r{1,2,3}.json
     field response_us.p50; endpoint = median of the three repeat p50s
Rule: T(f) = k/f + b through (306, 1173), r = 1.035, pick = lowest f in
  G = [510, 612, 714, 816, 918, 1020, 1122, 1173] with T(f)*r <= D, else decline (the
  deployment falls back to 1173).  aware: EMC-2133 cells; blind: EMC-3199 cells.
Scoring: miss = response_us > D (per 1000-cycle run); a run violates iff miss % > tau = 2 %.
Frontier at D: lowest clock that was run (in at least one session) and has no violating run
  in any session that ran it ("any"); variant "strict": the clock must have been run in every
  session of that model.  Bins are index distances in G.  A frontier equal to the lowest clock a
  session ran is censored (the true frontier could be lower).
Second part: per-session frontiers of the three MobileNetV2 sessions, the blind arm on the same
  37 deadlines, and one-sided 95 % Clopper-Pearson upper bounds for the zero counts.
Usage: nx_vision_frontier.py [--json]
"""
import argparse
import csv
import glob
import json
import os
from collections import Counter

import numpy as np

LAB = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
G = [510, 612, 714, 816, 918, 1020, 1122, 1173]
R = 1.035
TAU = 2.0
MODELS = [
    ("MobileNetV2", 33.0, ["nx", "nx_mobilenet", "nx_mobilenet_s3"], "nx",
     [4.4, 4.6, 4.8, 5.0, 5.2, 5.4, 5.6, 5.8, 6.0]),
    ("ViT-S", 20.0, ["nx_vit"], "nx_vit", [14.0, 14.5, 15.0, 15.5, 16.0, 17.0, 17.5, 18.5, 19.0]),
    ("YOLOv8s", 10.0, ["nx_yolov8s"], "nx_yolov8s",
     [29.5, 30.0, 31.0, 31.5, 33.0, 33.5, 35.5, 36.0, 38.5, 39.0]),
    ("ResNet-50", 20.0, ["nx_resnet50"], "nx_resnet50", [14.0, 14.5, 15.0, 15.5, 16.0, 17.0, 17.5, 18.5, 19.0]),
]


def fit(refdir, emc):
    cells = {}
    for g in (306, 1173):
        ps = []
        for r in (1, 2, 3):
            d = json.load(open(f"{LAB}/results/refit33/{refdir}/emc{emc}_g{g}_r{r}.json"))
            ps.append(float(d["response_us"]["p50"]))
            hz = float(d["meta"]["hz"])
        cells[g] = (float(np.median(ps)), ps, hz)
    t_lo, t_hi = cells[306][0], cells[1173][0]
    k = (t_lo - t_hi) / (1 / 306 - 1 / 1173)
    b = t_hi - k / 1173
    return k, b, cells


def pick(kb, D_us):
    k, b = kb
    for f in G:
        if (k / f + b) * R <= D_us:
            return f
    return None


def load(session, hz):
    runs = {}
    for p in sorted(glob.glob(f"{LAB}/results/govrep_rev/{session}/g*_p*.csv")):
        base = os.path.basename(p)
        if base.endswith(".power.csv"):
            continue
        g, k = base[:-4].split("_p")
        v = np.array([float(r["response_us"]) for r in csv.DictReader(open(p))])
        meta = json.load(open(p[:-4] + ".json"))["meta"]
        assert len(v) == 1000, p
        assert abs(float(meta["hz"]) - hz) < 1e-9, p
        runs.setdefault(int(g[1:]), {})[int(k)] = v
    return runs


def cp_upper0(n, alpha=0.05):
    return 1.0 - alpha ** (1.0 / n)          # exact one-sided CP UB for 0 events in n


def frontier_rows():
    out = dict(rows=[], fits={})
    print(f"rule: two-cell (306,1173) r={R}, grid {G}, tau={TAU} % per 1000-cycle run")
    for name, hz, sessions, refdir, Ds in MODELS:
        ka, ba, ca = fit(refdir, 2133)
        kbl, bbl, cb = fit(refdir, 3199)
        out["fits"][name] = dict(aware=(ka, ba), blind=(kbl, bbl),
                                 aware_cells={g: ca[g][0] for g in ca}, aware_reps={g: ca[g][1] for g in ca},
                                 cell_hz=ca[306][2])
        S = {s: load(s, hz) for s in sessions}
        print(f"\n## {name} {hz:.0f} Hz; sessions " + ", ".join(f"{s}: clocks {sorted(S[s])}" for s in sessions))
        print(f"   aware refit (cells @{ca[306][2]:.0f} Hz): p50 306 {ca[306][0]:.1f} (reps {ca[306][1]}), "
              f"1173 {ca[1173][0]:.1f} (reps {ca[1173][1]}); k={ka:.0f} b={ba:.0f}")
        allclk = sorted({f for s in S for f in S[s]})
        for D in Ds:
            Dus = D * 1000.0
            # violating runs per (clock, session)
            V = {}
            for f in allclk:
                V[f] = {}
                for s in sessions:
                    if f in S[s]:
                        m = [100.0 * np.count_nonzero(S[s][f][p] > Dus) / 1000.0 for p in sorted(S[s][f])]
                        V[f][s] = dict(nv=sum(1 for x in m if x > TAU), n=len(m), mx=max(m),
                                       pooled=100.0 * sum(np.count_nonzero(S[s][f][p] > Dus) for p in S[s][f]) / (1000.0 * len(m)))
            clean = [f for f in allclk if all(V[f][s]["nv"] == 0 for s in V[f])]
            clean_strict = [f for f in clean if len(V[f]) == len(sessions)]
            fr = min(clean) if clean else None
            fr_s = min(clean_strict) if clean_strict else None
            # non-monotone: a violating clock above the frontier
            above_viol = [f for f in allclk if fr is not None and f > fr and f not in clean]
            fa = pick((ka, ba), Dus)
            fbl = pick((kbl, bbl), Dus)
            row = dict(model=name, D=D, aware=fa, blind=fbl, frontier=fr, frontier_strict=fr_s,
                       lowest_run=min(allclk), censored=(fr is not None and fr == min(allclk)),
                       nonmono_above=above_viol,
                       frontier_cell={s: V[fr][s] for s in V[fr]} if fr else None)
            if fa is not None:
                row["bins"] = (G.index(fa) - G.index(fr)) if fr is not None else None
                row["bins_strict"] = (G.index(fa) - G.index(fr_s)) if fr_s is not None else None
                row["aware_cell"] = {s: V[fa][s] for s in V.get(fa, {})} if fa in V else None
            else:
                # declined: the fallback 1173 is what would run
                row["bins"] = None
                row["fallback_cell"] = {s: V[1173][s] for s in V[1173]} if 1173 in V else None
            out["rows"].append(row)
            ac = row.get("aware_cell") or row.get("fallback_cell")
            acs = ("; ".join(f"{s} {c['nv']}/{c['n']} (max {c['mx']:.1f} %)" for s, c in ac.items()) if ac else "no data")
            frs = ("; ".join(f"{s} {c['nv']}/{c['n']} (max {c['mx']:.1f} %)" for s, c in row["frontier_cell"].items())
                   if row["frontier_cell"] else "-")
            print(f"   D={D:5.1f} blind {fbl}  aware {fa if fa else 'DECLINE(->1173)'}  frontier {fr} "
                  f"(strict {fr_s}{', censored at lowest run clock ' + str(min(allclk)) if row['censored'] else ''})  "
                  f"bins {row['bins']}  | aware-pick runs>tau: {acs} | frontier runs>tau: {frs}"
                  + (f" | NON-MONOTONE violating clocks above frontier: {above_viol}" if above_viol else ""))
    return out


def summary(rows):
    adm = [r for r in rows if r["aware"] is not None]
    dec = [r for r in rows if r["aware"] is None]
    print("\n# summary")
    print(f"deadlines {len(rows)}; aware admits {len(adm)}; declines {[(r['model'], r['D']) for r in dec]}")
    for lab, key in (("any-session frontier", "bins"), ("strict frontier", "bins_strict")):
        c = Counter(r[key] for r in adm)
        by = {}
        for r in adm:
            grp = "MobileNetV2" if r["model"] == "MobileNetV2" else "breadth"
            by.setdefault(grp, Counter())[r[key]] += 1
        print(f"{lab}: aware - frontier bins over admitted: {dict(sorted(c.items(), key=lambda x: (x[0] is None, x[0])))}; "
              f"by group {{{', '.join(f'{g}: {dict(v)}' for g, v in by.items())}}}")
    cens = [(r["model"], r["D"]) for r in adm if r["censored"]]
    print(f"frontier censored (equals the lowest run clock, true frontier could be lower): {len(cens)} {cens}")
    nm = [(r["model"], r["D"], r["nonmono_above"]) for r in rows if r["nonmono_above"]]
    print(f"deadlines with a violating clock above the frontier: {len(nm)} {nm}")
    frv = sum(c["nv"] for r in rows if r["frontier_cell"] for c in r["frontier_cell"].values())
    print(f"violating runs at the frontier clock, summed over deadlines and sessions: {frv}")
    # aware-pick run-level false admissions
    tot_runs = sum(c["n"] for r in adm for c in r["aware_cell"].values())
    tot_viol = sum(c["nv"] for r in adm for c in r["aware_cell"].values())
    dl_viol = sum(1 for r in adm if any(c["nv"] > 0 for c in r["aware_cell"].values()))
    mx = max(c["mx"] for r in adm for c in r["aware_cell"].values())
    distinct = {}
    for r in adm:
        for s, c in r["aware_cell"].items():
            distinct.setdefault((r["model"], s, r["aware"]), []).append(c["nv"])
    print(f"aware pick over the {len(adm)} admitted deadlines: violating runs {tot_viol}/{tot_runs} "
          f"(deadline-run evaluations); deadlines with >=1 violating run {dl_viol}/{len(adm)}; "
          f"max single-run miss {mx:.2f} %; distinct (model, session, clock) cells used: {len(distinct)} "
          f"(= {10 * len(distinct)} distinct runs; deadlines sharing a pick share runs)")
    for grp in ("MobileNetV2", "breadth"):
        sub = [r for r in adm if (r["model"] == "MobileNetV2") == (grp == "MobileNetV2")]
        n = sum(c["n"] for r in sub for c in r["aware_cell"].values())
        v = sum(c["nv"] for r in sub for c in r["aware_cell"].values())
        m = max(c["mx"] for r in sub for c in r["aware_cell"].values())
        print(f"   {grp}: {len(sub)} admitted deadlines, violating runs {v}/{n}, max single-run miss {m:.2f} %")
    dc = dec[0].get("fallback_cell") if dec else None
    if dc:
        print("declined deadline fallback (1173) runs>tau: " + "; ".join(f"{s} {c['nv']}/{c['n']} max {c['mx']:.1f} %" for s, c in dc.items()))


def extra():
    """per-session MobileNetV2 frontiers, the blind arm on the same deadlines, CP bounds for the zero counts"""
    print("\n# per-session frontiers (MobileNetV2) and the blind arm on the same deadlines")
    blind_dl, blind_runs, blind_n, n_dl = 0, 0, 0, 0
    for name, hz, sessions, refdir, Ds in MODELS:
        ka, ba, _ = fit(refdir, 2133)
        kb, bb, _ = fit(refdir, 3199)
        S = {s: load(s, hz) for s in sessions}
        for D in Ds:
            n_dl += 1
            Dus = D * 1000.0
            fa, fbl = pick((ka, ba), Dus), pick((kb, bb), Dus)
            # blind
            nv = 0; n = 0
            for s in sessions:
                if fbl in S[s]:
                    m = [100.0 * np.count_nonzero(S[s][fbl][p] > Dus) / 1000 for p in S[s][fbl]]
                    nv += sum(1 for x in m if x > TAU); n += len(m)
            blind_runs += nv; blind_n += n; blind_dl += (nv > 0)
            if name == "MobileNetV2":
                per = []
                for s in sessions:
                    clean = [f for f in sorted(S[s]) if all(100.0 * np.count_nonzero(S[s][f][p] > Dus) / 1000 <= TAU for p in S[s][f])]
                    fr = min(clean) if clean else None
                    cens = fr is not None and fr == min(S[s])
                    per.append(f"{s}: frontier {fr}{'(censored)' if cens else ''} bins "
                               f"{(G.index(fa) - G.index(fr)) if (fa and fr) else '-'}")
                print(f"MNv2 D={D}: aware {fa}; " + "; ".join(per))
    print(f"\nblind arm on the {n_dl} deadlines: deadlines with >=1 violating run {blind_dl}/{n_dl}; "
          f"violating runs {blind_runs}/{blind_n} deadline-run evaluations")
    for n in (10, 30, 36, 240, 280, 300, 520):
        print(f"CP one-sided 95 % UB for 0/{n}: {100 * cp_upper0(n):.2f} %")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", action="store_true", help="also print the per-deadline record as JSON at the end")
    a = ap.parse_args()
    out = frontier_rows()
    summary(out["rows"])
    extra()
    if a.json:
        print(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()

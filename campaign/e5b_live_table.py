#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Per-cell / per-pairing LaTeX table of the E5-B-live re-run (label tab:e5blive) from the JSON that
campaign/e5b_emcfree.py --json writes; the frontier cell's 4.4 ms row is re-scored from the raw re-run CSVs
(results/e5b_live/nx, or $E5B_LIVE_DIR).
Usage: e5b_live_table.py IN.json [OUT.tex]   (OUT.tex omitted: the LaTeX goes to stdout; nothing else is written)."""
import json, os, sys
J = json.load(open(sys.argv[1]))
rows = J["rows"]; live_cmp = {(c["emc"], c["g"]): c for c in J["live_cmp"]}

def cn(emc, g):
    return f"{int(emc) if emc != 665.6 else '665.6'}/{g}"

# ---- panel A: per live cell
cells = {}
for r in rows:
    D = r["D"]; a = r["aware"]
    if a.get("live"):
        c = cells.setdefault((2133.0, a["g"]), dict(served=[], live=a["live"], rec=a["score"], tight=None))
        c["served"].append(f"refit {D:.1f}")
        if c["tight"] is None or D < c["tight"][0]:
            c["tight"] = (D, a["live"])
    for arm in ("H1", "H2"):
        x = r["arms"][arm]
        if x.get("feasible") and x.get("live"):
            c = cells.setdefault((x["emc"], x["g"]), dict(served=[], live=x["live"], rec=x["score"], tight=None))
            c["served"].append(f"{arm} {D:.1f}")
            if c["tight"] is None or D < c["tight"][0]:
                c["tight"] = (D, x["live"])
# frontier cell at 4.4 (3199/1020) is a live cell too, but the JSON rows hold no live score for it at 4.4
# (per-run data not stored): when the frontier row names it, its live runs are re-scored at 4.4 ms from the CSVs.
fr44 = next((r for r in rows if abs(r["D"] - 4.4) < 1e-9), None)
A = []
for (emc, g), c in sorted(cells.items(), key=lambda kv: (-kv[0][0], -kv[0][1])):
    lv, rec = c["live"], c["rec"]
    cmpr = live_cmp.get((emc, g))
    p50_rec = cmpr["p50_july"] if cmpr else rec["p50"]
    E_rec = cmpr["E_july"] if cmpr else rec["E"]
    n_rec = 3 if cmpr else rec["n"]
    tD, tl = c["tight"]
    _grp = {}
    for _s in c["served"]:
        _a, _d = _s.split(" ", 1); _grp.setdefault(_a, []).append(_d)
    served = "; ".join(f"{_a} {', '.join(_ds)}" for _a, _ds in _grp.items())
    if (emc, g) == (3199.0, 1020) and fr44 and fr44["frontier"] and fr44["frontier"]["g"] == 1020:
        served = "frontier 4.4; " + served
        # the frontier role at 4.4 ms: score the live runs at D = 4.4 ms from the raw per-cycle logs
        import glob, csv
        LIVE = os.environ.get("E5B_LIVE_DIR", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                                           "results", "e5b_live", "nx"))
        miss = []
        for fn in sorted(glob.glob(f"{LIVE}/e3199_g1020_r*.csv")):
            if fn.endswith(".power.csv"):
                continue
            v = [float(r["response_us"]) for r in csv.DictReader(open(fn))]
            miss.append(100.0 * sum(1 for x in v if x > 4400.0) / len(v))
        assert len(miss) == 10, f"frontier cell: {len(miss)} live runs found"
        tD, tl = 4.4, dict(viol=sum(1 for m in miss if m > 2.0), n=len(miss), max_miss=max(miss))
    A.append(f"    {cn(emc, g)} & {lv['n']} & {lv['p50']:.0f} & {p50_rec:.0f} ({n_rec}) & {lv['E']:.1f} [{lv['E_min']:.1f}, {lv['E_max']:.1f}] & {E_rec:.1f} & "
             f"{tl['viol']}/{tl['n']} at {tD:.1f} (max {tl['max_miss']:.1f}) & {served} \\\\")

# ---- panel B: per (D, fit) pairing
B = []
for r in rows:
    for arm in ("H1", "H2"):
        x = r["arms"][arm]
        lp = x.get("live_paired") if x.get("feasible") else None
        if not lp:
            continue
        a = r["aware"]
        B.append(f"    {r['D']:.1f} & {arm} & {cn(x['emc'], x['g'])} & 2133/{a['g']} & {x['live']['viol']}/{x['live']['n']} / {lp['aware_viol']}/{lp['n']} & "
                 f"${lp['mean']:+.1f}$ [${lp['lo']:+.1f}$, ${lp['hi']:+.1f}$] & ${lp['pct']:+.1f}$ [${lp['lo_pct']:+.1f}$, ${lp['hi_pct']:+.1f}$] \\\\")

T = [r"\begin{table*}[t]", r"  \centering",
     r"  \caption{Interleaved re-run of the EMC-free picks (E5-B-live; NX MobileNetV2 at 33\,Hz, $\tau{=}2\%$; seven cells $\times$ ten rounds, the EMC re-locked and read back before every cell; protocol, disclosure and thermal-gate record in \S\ref{sec:e5bprot}). Top: per cell, the live p50 and mean energy per inference [min, max over the ten runs] against the recorded cells the replay scored (3199\,MHz: the July 2026 cells, $n{=}3$; 2133\,MHz: the three interleaved sessions of main-text Table~II, $n{=}30$), runs over $\tau$ out of ten at the tightest deadline the cell serves (largest run miss, \%), and which pick each cell is. Bottom: per deadline and latency fit, the selector's cell against the per-EMC refit's, runs over $\tau$ out of ten for each, and the round-paired premium in energy per inference (selector minus refit, mJ and \% of the refit's live mean) with its paired-$t$ $95\%$ interval over the ten rounds.}",
     r"  \label{tab:e5blive}", r"  \footnotesize", r"  \setlength{\tabcolsep}{3pt}",
     r"  \resizebox{\textwidth}{!}{%", r"  \begin{tabular}{@{}l r rr rr r l@{}}", r"    \toprule",
     r"    cell (EMC/GPU, MHz) & $n$ & p50 live ($\mu$s) & p50 recorded ($n$) & E live (mJ) [min, max] & E recorded & runs${>}\tau$ at $D$ (ms) & picks served \\",
     r"    \midrule"] + A + [
     r"    \midrule",
     r"    \multicolumn{8}{@{}l}{\textit{Round-paired premiums (ten rounds)}} \\",
     r"    $D$ (ms) & fit & selector cell & refit cell & runs${>}\tau$ sel.\,/\,refit & $\Delta E$ (mJ) [$95\%$ CI] & \multicolumn{2}{l}{$\Delta E$ (\%) [$95\%$ CI]} \\",
     r"    \midrule"] + [b.replace(" \\\\", " & \\\\") for b in B] + [
     r"    \bottomrule", r"  \end{tabular}}", r"\end{table*}"]
if len(sys.argv) > 2:
    open(sys.argv[2], "w").write("\n".join(T) + "\n")
    print(f"wrote {sys.argv[2]}: {len(A)} cells, {len(B)} pairings")
else:
    print("\n".join(T))

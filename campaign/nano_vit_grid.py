#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Supp. Nano ViT-Small grid table: the full Nano ViT-Small deployment grid at EMC 2133 MHz (ten
pre-stated deadlines, one interleaved session, 20 Hz), both arms under the deployed two-cell rule --
the Nano counterpart of the NX breadth table and the full grid behind the ViT-Small (Nano) row of
Table III (entry 'vit_nano' of campaign/solo_stats.py; rule, fits, loader and cell scoring:
campaign/govdeploy_common.py).

Deadlines: the pre-stated grid of the protocol addendum of 2026-09-15 (docs/protocols/E1_E2_protocol.md):
D = T_aware(f) r rounded up to 0.5 ms for f in {612, ..., 1020} MHz plus each value + 0.5 ms, from the
original-rate (10 Hz) aware fit; the rate-matched fit yields the same ten values (asserted).  Fits:
rate-matched 20 Hz endpoint cells results/refit33/nano_vit (median of three repeat p50s), GPU 306 and
1020 MHz; blind at EMC 3199, aware at 2133.  Data: results/govrep_rev/nano_vit, 10 pairs x 1,000 cycles.
Per deadline: both picks; per-run miss % median [min-max] and runs over tau = 2 % (deadlines that share
a pick share the same ten runs); dE = aware - blind energy per inference paired by pair index: mean,
paired-t 95 % CI and the pairs in which the aware arm cost more.  Energy per inference is the Table II
convention (trimmed-mean VDD_IN over the INA log / 20 Hz).  Also printed:
  window    the scored energy window of the runs: where the power log ends, the run spacing (from
            consecutive run timestamps), the cycles inside the trimmed window, and the end of the
            back-to-back warm-up plateau detected in the trace
  baseline  the VDD_IN baseline levels of the session: per clock, each run's trimmed-window mean and its
            2 s segment means relative to the clock's lowest run mean; the step between the two
            run-mean clusters (median over clocks of median(upper) - median(lower), upper = runs more
            than 0.4 W above the lowest) and the runs whose segments reach a third level (> 1.4 W)
  the grid facts the Table III note and the main text quote (deadlines with a violating run per arm,
  the aware knife-edge at the top clock).

Read-only on results/ (python3 + numpy; no scipy); prints to stdout, writes nothing.
Usage: nano_vit_grid.py [--json]
"""
import argparse
import csv
import json
import math
import os
import sys
from datetime import datetime

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from govdeploy_common import load_session, refit_fits, original_fits, that, evaluate   # noqa: E402
from solo_stats import ENTRIES, viol0, nano_vit_facts                                 # noqa: E402


def grid_from(fits, clocks=(612, 714, 816, 918, 1020)):
    base = {math.ceil(that(fits["aware"], f) / 1000.0 / 0.5) * 0.5 for f in clocks}
    return sorted(base | {b + 0.5 for b in base})


def ina_window(s):
    """Scored energy window of each run: (log end s, run spacing s from consecutive run ends, cycles inside the trimmed
    window, warm-up end s). The periodic phase starts when the back-to-back warm-up plateau ends (detected in the trace)."""
    ends, cyc, wend, ts = [], [], [], []
    for f in s["runs"]:
        for p in s["runs"][f]:
            stem = s["runs"][f][p]["stem"]; hz = s["runs"][f][p]["hz"]
            rows = [(float(r["t"]), float(r["vdd_in_mw"])) for r in csv.DictReader(open(stem + ".power.csv")) if r.get("vdd_in_mw")]
            t = np.array([a for a, b in rows]); pw = np.array([b for a, b in rows]); n = len(t)
            tlo, thi = t[n // 10], t[-n // 10 - 1]
            edges = np.arange(0, 8.0001, 0.25)
            bins = np.array([pw[(t >= x) & (t < x + 0.25)].mean() if np.any((t >= x) & (t < x + 0.25)) else np.nan for x in edges[:-1]])
            base = np.median(pw[t > 10]); thr = (np.nanmax(bins) + base) / 2
            hi = [edges[k] + 0.25 for k, b in enumerate(bins) if b > thr]
            ps = max(hi) if hi else 0.0
            ends.append(t[-1]); wend.append(ps); cyc.append((thi - max(tlo, ps)) * hz)
            ts.append(datetime.fromisoformat(json.load(open(stem + ".json"))["meta"]["timestamp"]))
    ts.sort(); gaps = [(b - a).total_seconds() for a, b in zip(ts, ts[1:])]
    return dict(log=(min(ends), max(ends)), spacing=(min(gaps), max(gaps)), cycles=(min(cyc), max(cyc)), warm=(min(wend), max(wend)))


def baseline_levels(s):
    """VDD_IN baseline levels of the session (check of 2026-10-30): per clock, each run's trimmed-window mean and
    its 2 s segment means relative to the clock's lowest run mean.  Returns the step between the two run-mean clusters
    (median over clocks of median(upper) - median(lower), upper = runs > lowest + 0.4 W) and the runs whose segments reach a
    third level (> lowest + 1.4 W), with the range of their highest segment."""
    steps, third = [], []
    for f in s["runs"]:
        lv = {}
        for p in s["runs"][f]:
            stem = s["runs"][f][p]["stem"]
            rows = [(float(r["t"]), float(r["vdd_in_mw"])) for r in csv.DictReader(open(stem + ".power.csv")) if r.get("vdd_in_mw")]
            t = np.array([a for a, b in rows]); pw = np.array([b for a, b in rows]); n = len(t)
            tw, w = t[n // 10:n - n // 10], pw[n // 10:n - n // 10]
            seg = [w[(tw >= x) & (tw < x + 2)].mean() for x in np.arange(math.ceil(tw[0]), tw[-1] - 2, 2.0)]
            lv[p] = (float(w.mean()), float(max(seg)))
        base = min(m for m, _ in lv.values())
        lo = [m - base for m, _ in lv.values() if m - base <= 400]; hi = [m - base for m, _ in lv.values() if m - base > 400]
        if lo and hi:
            steps.append(np.median(hi) - np.median(lo))
        third += [(f, p, mx - base) for p, (_, mx) in lv.items() if mx - base > 1400]
    return dict(step=float(np.median(steps)), third=third)


def build():
    """Score the Nano ViT-Small grid with its structural checks; returns the rows and every quantity the
    supplementary table and its caption state."""
    B = next(e for e in ENTRIES if e["key"] == "vit_nano")
    sessions = [s for s in (load_session(n, B["hz"]) for n in B["sessions"]) if s]
    assert len(sessions) == 1, "expected exactly one Nano ViT-Small session"
    s = sessions[0]
    clocks = sorted(s["runs"])
    assert all(len(s["runs"][f]) == 10 for f in clocks), s["pairs"]
    fits, info = refit_fits(B)
    ofits, oinfo = original_fits(B)
    assert fits is not None, "rate-matched Nano ViT-Small refit cells missing (results/refit33/nano_vit)"
    assert grid_from(ofits) == list(B["D"]) == grid_from(fits), (grid_from(ofits), grid_from(fits), B["D"])
    rows = evaluate(B, fits, sessions)
    assert len(rows) == 10 and all(r["okb"] and r["oka"] for r in rows), "every grid deadline should be admitted by both arms"
    facts = nano_vit_facts(B, rows)
    bviol = sum(1 for r in rows if viol0(r["blind"]) >= 1)
    aviol = [r for r in rows if viol0(r["aware"]) >= 1]
    k15 = next(r for r in rows if abs(r["D"] - 15.0) < 1e-9)
    W = ina_window(s)
    BL = baseline_levels(s)
    assert 800 <= BL['step'] <= 950 and len(BL['third']) == 4, BL   # 2026-10-30 check: step ~0.87 W, four third-level runs
    pos = [(r["D"], r["dE"]["pos"], r["dE"]["n"]) for r in rows if isinstance(r["dE"], dict)]
    neg_means = [r["D"] for r in rows if isinstance(r["dE"], dict) and r["dE"]["mean"] < 0]
    spans = all(r["dE"]["t_lo"] < 0 < r["dE"]["t_hi"] for r in rows if isinstance(r["dE"], dict))
    negr = [r for r in rows if isinstance(r["dE"], dict) and r["dE"]["mean"] < 0]
    assert len({(r["fb"], r["fa"]) for r in negr}) <= 1, "negative means from more than one contrast"
    neg_from, neg_to = (negr[0]["fb"], negr[0]["fa"]) if negr else (None, None)
    assert W["warm"][1] <= 3.0 + 1e-9 and W["log"][1] - W["log"][0] < 0.05, W
    return dict(B=B, s=s, clocks=clocks, fits=fits, info=info, ofits=ofits, oinfo=oinfo, rows=rows, facts=facts,
                bviol=bviol, aviol=aviol, k15=k15, W=W, BL=BL, pos=pos, neg_means=neg_means, spans=spans,
                neg_from=neg_from, neg_to=neg_to, hzc=info["aware"]["hz"])


def pm(x):
    return "100" if x >= 99.95 else f"{x:.1f}"


def miss_txt(c):
    if not c["has_data"]:
        return "n/a"
    rng = "" if c["mn"] == c["mx"] else f" [{pm(c['mn'])}-{pm(c['mx'])}]"
    return f"{pm(c['med'])}{rng} ({viol0(c)}/10)"


def dE_txt(dE):
    if dE == "same":
        return "0 (same f)"
    if dE is None:
        return "n/a"
    return f"{dE['mean']:+.1f} [{dE['t_lo']:.1f}, {dE['t_hi']:.1f}] ({dE['pos']}/{dE['n']})"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", action="store_true", help="print the rows and the caption quantities as JSON")
    a = ap.parse_args()
    g = build()
    rows, W, BL, s = g["rows"], g["W"], g["BL"], g["s"]
    if a.json:
        def cell(c):
            return {k: c.get(k) for k in ("f", "has_data", "med", "mn", "mx", "per", "viol")}
        out = dict(session=s["name"], clocks=g["clocks"], pairs=s["pairs"], t0=s["t0"], t1=s["t1"],
                   fit_hz=g["hzc"], original_fit_hz=g["oinfo"]["aware"]["hz"], deadlines=[r["D"] for r in rows],
                   rows=[dict(D=r["D"], fb=r["fb"], fa=r["fa"], blind=cell(r["blind"]), aware=cell(r["aware"]),
                              dE=r["dE"] if not isinstance(r["dE"], dict) else
                              {k: r["dE"][k] for k in ("n", "mean", "sd", "t_lo", "t_hi", "b_lo", "b_hi", "pos", "sign_p", "Eb", "Ea", "rel")})
                         for r in rows],
                   window=W, baseline=dict(step_mW=BL["step"], third_level=[dict(f=f, pair=p, above_lowest_mW=x) for f, p, x in BL["third"]]),
                   blind_violating_deadlines=g["bviol"], aware_violating_deadlines=[r["D"] for r in g["aviol"]],
                   negative_mean_deadlines=g["neg_means"], all_intervals_span_zero=g["spans"])
        print(json.dumps(out, indent=1))
        return
    fb, fa = g["fits"]["blind"], g["fits"]["aware"]
    print(f"Nano ViT-Small grid: session {s['name']} ({s['t0']} .. {s['t1']} UTC), clocks {g['clocks'][0]}-{g['clocks'][-1]} MHz "
          f"{g['clocks']}, pairs {s['pairs']}")
    print(f"fits: rate-matched {g['hzc']:.0f} Hz cells, blind k={fb[0]:.0f} b={fb[1]:.0f}, aware k={fa[0]:.0f} b={fa[1]:.0f}; "
          f"deadlines from the original {g['oinfo']['aware']['hz']:.0f} Hz aware fit = from the rate-matched fit = "
          f"{', '.join(f'{d:.1f}' for d in g['B']['D'])} ms")
    print("\n| D (ms) | blind f* | blind miss % med [min-max] (runs > tau) | aware f* | aware miss % med [min-max] (runs > tau) | "
          "dE mJ/inf [paired-t 95 % CI] (pairs > 0) |")
    print("|---:|---:|---|---:|---|---|")
    for r in rows:
        print(f"| {r['D']:.1f} | {r['fb']} | {miss_txt(r['blind'])} | {r['fa']} | {miss_txt(r['aware'])} | {dE_txt(r['dE'])} |")
    print("\nper-run miss % (pairs 1..10):")
    for r in rows:
        for arm in ("blind", "aware"):
            c = r[arm]
            print(f"  D={r['D']:5.1f} {arm:5s} {c['f']:4d}: " + " ".join(f"{x:5.1f}" for x in c["per"][0]))
    print("\npaired dE (aware - blind, mJ/inf) at contrasting cells:")
    for r in rows:
        d = r["dE"]
        if isinstance(d, dict):
            print(f"  D={r['D']:5.1f} {r['fb']}->{r['fa']}: mean {d['mean']:+.2f} sd {d['sd']:.2f} t95 [{d['t_lo']:+.2f}, {d['t_hi']:+.2f}] "
                  f"boot95 [{d['b_lo']:+.2f}, {d['b_hi']:+.2f}] pairs > 0 {d['pos']}/{d['n']} (sign p={d['sign_p']:.3f}); "
                  f"E_blind {d['Eb']:.1f} E_aware {d['Ea']:.1f}")
    print(f"every interval at a contrasting cell spans zero: {'yes' if g['spans'] else 'no'}; negative means at "
          f"{', '.join(f'{d:.1f}' for d in g['neg_means']) or 'none'} ms"
          + (f" (the {g['neg_from']}->{g['neg_to']} MHz contrast)" if g["neg_means"] else ""))
    print(f"\nwindow: the power log ends at {W['log'][0]:.2f}-{W['log'][1]:.2f} s; runs start every "
          f"{W['spacing'][0]:.1f}-{W['spacing'][1]:.1f} s (mean {(W['spacing'][0] + W['spacing'][1]) / 2:.1f} s); the trimmed "
          f"window covers {W['cycles'][0]:.0f}-{W['cycles'][1]:.0f} of the 1,000 cycles; warm-up plateau ends by "
          f"{W['warm'][0]:.2f}-{W['warm'][1]:.2f} s")
    th = [x for *_, x in BL["third"]]
    print(f"baseline: step between the two run-mean clusters {BL['step']:.0f} mW (to 0.05 W: "
          f"{round(BL['step'] / 50) * 50 / 1000:.2f} W); {len(th)} runs reach a third level, "
          f"{min(th) / 1000:.1f}-{max(th) / 1000:.1f} W above the lowest "
          f"({', '.join(f'{f} MHz pair {p}: {x:.0f} mW' for f, p, x in BL['third'])})")
    F, k, k15 = g["facts"], g["facts"]["knife"], g["k15"]
    full_d = ", ".join("%.1f" % r["D"] for r in F["bfull"])
    print(f"\nblind arm: a violating run at {g['bviol']} of 10 deadlines (every run at "
          f"{full_d} ms); aware arm at {len(g['aviol'])}: "
          f"{k['D']:.1f} ms at the top clock {k['fa']} MHz (median {pm(k['aware']['med'])} %, max {pm(k['aware']['mx'])} %, "
          f"{viol0(k['aware'])}/10 runs > tau) and {k15['D']:.1f} ms ({viol0(k15['aware'])}/10, max {pm(k15['aware']['mx'])} %)")


if __name__ == "__main__":
    main()

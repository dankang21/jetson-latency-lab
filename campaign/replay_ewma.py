#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""EWMA level calibrator (the blind k/f+b model plus an online offset, deployed
live by nx_live_gov.py --mode calib): stationary replay over the recorded
per-clock traces at several gains and update cadences, the live runs it is
compared with, and the converged constant offset's held-out error (supplement,
online-calibrator paragraph, and the offset table, label tab:offset).
No board runs.

Controller (nx_live_gov.py, mode calib):
  T(f) = k/f + b, k = 3207403 us*MHz, b = 1325 us (its --k/--b defaults: the blind
  least-squares fit at EMC 3199, joint_gov_policy.fit_sweep, rounded)
  pick(off) = lowest f in {612, 714, 816, 918, 1020, 1122, 1173} with
              (T(f) + off) * 1.035 <= D, else 1173;  off = 0 and cur = pick(0) at start
  per cycle:  off <- (1 - alpha) off + alpha (resp - T(cur)), re-pick every cycle,
              a switch decided in cycle i takes effect in cycle i + 1 (no hysteresis)
  windowed cadence (FLAME's online adaptation: a ten-estimate window, alpha = 0.6,
  one update per window): off <- (1 - alpha) off + alpha mean(window of w residuals),
  non-overlapping windows; the pick is re-evaluated only when off updates.
Stationary replay: the response in cycle i at clock f is cycle (i mod 1000) of a
recorded locked-clock run at f, EMC 2133, 33 Hz -- the same run index at every
clock -- so a trace cannot drift while the controller sits on it.
  June       results/govrep/g<f>_r<1..3>.csv                   (per-clock runs)
  September  results/govrep_rev/nx_mobilenet/g<f>_p<1..10>.csv (interleaved session)
Variants: per cycle at alpha 0.02 / 0.1 / 0.3 (the gain sweep; 0.1 is deployed),
  w = 10 at alpha 0.6 (FLAME's cadence), per cycle at alpha 0.6 (its gain alone)
  and w = 10 at alpha 0.1 (its cadence alone); D = 4.6, 4.8, 5.0 ms; 2,000 cycles.
Escape = the number of cycles executed at the initial pick, i.e. 1 + the index of
  the cycle whose update first moves the pick; misses = deadline misses in those
  cycles.  A clock is feasible at D in a run when its trace misses <= tau = 2 %;
  'one switch' = the first switch away from an infeasible initial pick lands on
  a feasible clock.
Live runs (results/live_gov/calib_D<D>_r<1..3>.csv, alpha 0.1, 2,000 cycles):
  escape = first 'switched' row's cycle + 1; misses up to and including it; switch
  count; cycle of the last switch.
Converged offset (tab:offset): the blind GPU-only fit at EMC 3199 (analyze_estimator.py
  model A) plus the constant offset that re-centres its median residual over the
  eight held-out EMC-2133 cells; median/max |relative error| over those cells.

Read-only on results/ (python3 + numpy); prints to stdout, writes nothing.
Usage: replay_ewma.py [--n-cycles 2000] [--index-mode global|since_switch] [--tau 2]
                      [--k 3207403] [--b 1325] [--per-run] [--json]
"""
import argparse, csv, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LAB = os.path.dirname(HERE)
sys.path.insert(0, HERE)

GRID = [612, 714, 816, 918, 1020, 1122, 1173]       # nx_live_gov.py G_LIST
MARGIN = 1.035
D_MS = [4.6, 4.8, 5.0]
SOURCES = {
    "june": dict(label="June per-clock traces (results/govrep, r1-r3)",
                 path=LAB + "/results/govrep/g{f}_r{r}.csv", runs=[1, 2, 3]),
    "sept": dict(label="September interleaved traces (results/govrep_rev/nx_mobilenet, p1-p10)",
                 path=LAB + "/results/govrep_rev/nx_mobilenet/g{f}_p{r}.csv", runs=list(range(1, 11))),
}
VARIANTS = [  # key, alpha, window, label
    ("a0.02", 0.02, 1, "per cycle, alpha 0.02"),
    ("a0.1", 0.1, 1, "per cycle, alpha 0.1 (deployed)"),
    ("a0.3", 0.3, 1, "per cycle, alpha 0.3"),
    ("flame", 0.6, 10, "w=10, alpha 0.6 (FLAME cadence)"),
    ("gain0.6", 0.6, 1, "per cycle, alpha 0.6 (gain only)"),
    ("w10a0.1", 0.1, 10, "w=10, alpha 0.1 (cadence only)"),
]
GAIN_SWEEP = ("a0.02", "a0.1", "a0.3")


def load_trace(path):
    return [float(x["response_us"]) for x in csv.DictReader(open(path))]


def replay(traces, D_us, alpha, window, n_cycles, k, b, tau, index_mode):
    T = lambda f: k / f + b
    pick = lambda off: next((f for f in GRID if (T(f) + off) * MARGIN <= D_us), GRID[-1])
    feasible = {f: sum(v > D_us for v in tr) / len(tr) <= tau / 100 for f, tr in traces.items()}
    off, cur, arrived, buf = 0.0, pick(0.0), 0, []
    init = cur
    esc = miss_esc = first_to = last = None
    misses = switches = 0
    for i in range(n_cycles):
        tr = traces[cur]
        resp = tr[(i if index_mode == "global" else i - arrived) % len(tr)]
        misses += resp > D_us
        buf.append(resp - T(cur))
        if len(buf) < window:
            continue
        off = (1 - alpha) * off + alpha * sum(buf) / len(buf)
        buf = []
        new = pick(off)
        if new != cur:
            switches += 1
            last = i
            if esc is None:
                esc, miss_esc, first_to = i + 1, misses, new
            cur, arrived = new, i + 1
    return dict(init=init, init_feasible=feasible[init], escape=esc, misses_before=miss_esc, first_to=first_to,
                lands_feasible=None if first_to is None else feasible[first_to],
                switches=switches, last_switch=last, final=cur)


def live(D, k, b):
    """The logged gpu_mhz is the clock after that cycle's decision, so the initial pick is added."""
    init = next((f for f in GRID if (k / f + b) * MARGIN <= D * 1000), GRID[-1])
    out = []
    for r in (1, 2, 3):
        rows = list(csv.DictReader(open(f"{LAB}/results/live_gov/calib_D{D}_r{r}.csv")))
        sw = [int(x["cycle"]) for x in rows if x["switched"] == "1"]
        first = sw[0] if sw else None
        out.append(dict(run=r, init=init, escape=None if first is None else first + 1,
                        misses_before=sum(int(x["miss"]) for x in rows[:(first + 1 if sw else len(rows))]),
                        switches=len(sw), last_switch=sw[-1] if sw else None, cycles=len(rows),
                        clocks=sorted({init} | {int(x["gpu_mhz"]) for x in rows})))
    return out


def offset_table():
    from analyze_estimator import load as load_cells_est, fit_A, GPUS, WLS
    data = load_cells_est()
    out = {}
    for wl in WLS:
        cells = data[wl]
        k, b = fit_A(cells)
        meas = np.array([cells[(2133, g)] for g in GPUS])
        pred = np.array([k * 1e3 / g + b for g in GPUS])
        off = float(np.median(meas - pred))
        eb = np.abs(pred - meas) / meas * 100
        eo = np.abs(pred + off - meas) / meas * 100
        out[wl] = dict(offset_us=off, blind=(float(np.median(eb)), float(eb.max())),
                       offset=(float(np.median(eo)), float(eo.max())))
    return out


def rng_of(xs):
    xs = [x for x in xs if x is not None]
    return (min(xs), max(xs)) if xs else None


def fr(t):
    return "-" if t is None else f"{t[0]}-{t[1]}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-cycles", type=int, default=2000)
    ap.add_argument("--index-mode", choices=["global", "since_switch"], default="global",
                    help="trace index: global cycle mod 1000, or restarted at each switch")
    ap.add_argument("--tau", type=float, default=2.0, help="per-run miss budget, %% (feasible clock)")
    ap.add_argument("--k", type=float, default=3207403.0, help="us*MHz (nx_live_gov.py default)")
    ap.add_argument("--b", type=float, default=1325.0, help="us (nx_live_gov.py default)")
    ap.add_argument("--per-run", action="store_true", help="also print every (source, D, run, variant) cell")
    ap.add_argument("--json", action="store_true", help="print the numbers as JSON instead of tables")
    a = ap.parse_args()

    res = {}
    for src, s in SOURCES.items():
        res[src] = {}
        for r in s["runs"]:
            traces = {f: load_trace(s["path"].format(f=f, r=r)) for f in GRID}
            for D in D_MS:
                for key, alpha, w, _ in VARIANTS:
                    res[src][(D, r, key)] = replay(traces, D * 1000, alpha, w, a.n_cycles, a.k, a.b, a.tau,
                                                   a.index_mode)
    lv = {D: live(D, a.k, a.b) for D in D_MS}
    off = offset_table()

    def esc(src, keys, Ds, runs=None):
        runs = runs or SOURCES[src]["runs"]
        return rng_of([res[src][(D, r, kk)]["escape"] for D in Ds for r in runs for kk in keys])

    one_switch = {src: [(D, r, kk, x["lands_feasible"]) for (D, r, kk), x in res[src].items()
                        if not x["init_feasible"] and x["escape"] is not None] for src in SOURCES}
    no_escape = {src: [(D, r, kk) for (D, r, kk), x in res[src].items() if x["escape"] is None] for src in SOURCES}
    if a.json:
        print(json.dumps(dict(
            replay={src: [dict(D_ms=D, run=r, variant=kk, **x) for (D, r, kk), x in v.items()] for src, v in res.items()},
            live={str(D): v for D, v in lv.items()}, offset_table=off), indent=1))
        return

    print(f"=== EWMA calibrator, stationary replay: k={a.k:.0f} us*MHz, b={a.b:.0f} us, r={MARGIN}, "
          f"{a.n_cycles} cycles, index {a.index_mode}, feasible = trace miss <= {a.tau:g} % ===")
    for src, s in SOURCES.items():
        print(f"\n--- {s['label']}")
        x0 = {D: res[src][(D, s["runs"][0], "a0.1")] for D in D_MS}
        print("  initial pick: " + ", ".join(f"D={D} -> {x0[D]['init']}" for D in D_MS)
              + "; infeasible in " + ", ".join(
                  f"D={D}: {sum(not res[src][(D, r, 'a0.1')]['init_feasible'] for r in s['runs'])}/{len(s['runs'])} runs"
                  for D in D_MS))
        print("  escape (cycles at the initial pick), min-max over runs:")
        print(f"  {'variant':<34}" + "".join(f"{'D=' + str(D):>11}" for D in D_MS) + f"{'all D':>11}"
              + f"{'misses':>9}{'switches':>10}")
        for key, _, _, lab in VARIANTS:
            row = f"  {lab:<34}" + "".join(f"{fr(esc(src, [key], [D])):>11}" for D in D_MS) + f"{fr(esc(src, [key], D_MS)):>11}"
            xs = [res[src][(D, r, key)] for D in D_MS for r in s["runs"]]
            row += f"{fr(rng_of([x['misses_before'] for x in xs])):>9}{fr(rng_of([x['switches'] for x in xs])):>10}"
            print(row)
        print(f"  gain sweep (alpha 0.02/0.1/0.3), D=4.6-4.8: {fr(esc(src, GAIN_SWEEP, [4.6, 4.8]))} cycles"
              f" (run {s['runs'][0]} alone: {fr(esc(src, GAIN_SWEEP, [4.6, 4.8], [s['runs'][0]]))})")
        allk = [kk for kk, _, _, _ in VARIANTS]
        mx = max(res[src][(D, r, kk)]["escape"] for D in D_MS for r in s["runs"] for kk in allk
                 if res[src][(D, r, kk)]["escape"] is not None)
        at = sorted({kk for D in D_MS for r in s["runs"] for kk in allk if res[src][(D, r, kk)]["escape"] == mx})
        print(f"  longest escape over every variant: {mx} cycles ({', '.join(at)})")
        os_ = one_switch[src]
        print(f"  infeasible initial pick left in one switch (first switch lands on a feasible clock): "
              f"{sum(1 for x in os_ if x[3])}/{len(os_)} (D x run x variant)"
              + (f"; no escape within {a.n_cycles} cycles: {len(no_escape[src])}" if no_escape[src] else ""))
        inf = [(D, r) for D in D_MS for r in s["runs"] if not res[src][(D, r, "a0.1")]["init_feasible"]]
        if len(inf) < len(D_MS) * len(s["runs"]):
            sub = lambda kk: rng_of([res[src][(D, r, kk)]["escape"] for D, r in inf])
            print(f"  runs whose initial pick is infeasible only ({len(inf)}): FLAME cadence {fr(sub('flame'))} cycles, "
                  f"longest over every variant {max(sub(kk)[1] for kk in allk if sub(kk))} cycles")
        if a.per_run:
            print(f"  {'D':<5}{'run':>4} {'init':>5}{'feas':>5}  " + "".join(f"{kk:>16}" for kk in allk)
                  + "   (escape/misses/switches -> first clock)")
            for D in D_MS:
                for r in s["runs"]:
                    x0 = res[src][(D, r, "a0.1")]
                    cells = []
                    for kk in allk:
                        x = res[src][(D, r, kk)]
                        cells.append("-" if x["escape"] is None else
                                     f"{x['escape']}/{x['misses_before']}/{x['switches']}>{x['first_to']}")
                    print(f"  {D:<5}{r:>4} {x0['init']:>5}{'y' if x0['init_feasible'] else 'n':>5}  "
                          + "".join(f"{c:>16}" for c in cells))

    print("\n=== live runs (results/live_gov, per-cycle alpha 0.1, 2,000 cycles) ===")
    for D in D_MS:
        for x in lv[D]:
            print(f"  D={D} r{x['run']}: initial {x['init']}, escape {x['escape']} cycles, misses {x['misses_before']} "
                  f"in the transient, {x['switches']} switches, last at cycle {x['last_switch']}, clocks {x['clocks']}")
    tail = [x for D in (4.8, 5.0) for x in lv[D]]
    print(f"  D=4.8-5.0: escape {fr(rng_of([x['escape'] for x in tail]))} cycles, "
          f"misses in the transient {fr(rng_of([x['misses_before'] for x in tail]))}")
    print("  D=4.6: switches per run " + ", ".join(f"r{x['run']} {x['switches']} (last at cycle {x['last_switch']})"
                                                   for x in lv[4.6]))

    print("\n=== converged offset: held-out error at EMC 2133 (median/max |rel err| %, eight GPU clocks) ===")
    print(f"  {'workload':<10}{'blind':>12}{'offset (drift)':>16}{'offset us':>11}{'offset median %':>17}")
    for wl, x in off.items():
        print(f"  {wl:<10}{x['blind'][0]:>6.1f}/{x['blind'][1]:<5.1f}{x['offset'][0]:>10.1f}/{x['offset'][1]:<5.1f}"
              f"{x['offset_us']:>11.1f}{x['offset'][0]:>17.2f}")


if __name__ == "__main__":
    main()

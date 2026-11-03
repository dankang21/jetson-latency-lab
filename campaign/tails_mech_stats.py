#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Tail, burst and stall-signature numbers of the supplement that the other analysis
scripts do not print: the replication cells of Supp. Sec. S5 (the Nano worst cell's four
runs, the NX runs of the same cell and the NX Part-B cells), the workload extension and
the tail-cycle signature of Supp. Sec. S6, the per-kernel slow state of Supp. Sec. S6 and
Fig. S9 (Nsight Systems kernel durations), and the live first-miss EMC upshift of Supp.
Sec. S7.

Read-only over results/ (python3 + numpy; nothing is written).  Run from anywhere:
    python3 campaign/tails_mech_stats.py [--section NAME ...] [--results DIR]
Sections: worst, nx, ext, cycles, slow, live (default: all).

Some cells are bulk traces in the Zenodo deposit 10.5281/zenodo.21236659; extract the
archive into results/ first (a missing cell is reported and skipped):
  e4_replication_nano.tar.gz -> results/e4_replication/ (nano_rep_emc2133_adv2_r{2,3,4},
                                and the NX cells nx_pb_*, nx_rep_*)
  partb_nx.tar.gz            -> results/e4_nx/ (the NX cells nx_pb_*, nx_rep_*; either copy
                                is used)
  partb_ext_nano.tar.gz      -> results/partb_ext/ (ViT-Small and L2-resident GEMM cells)

Definitions (one 100k-cycle run, response_us per cycle):
  deadline     the run's own p99.9 (0.1 % of the cycles miss); continuation probability
               P[miss | previous cycle missed]; clustering ratio = that probability /
               P[miss]; bursts = maximal runs of consecutive misses (campaign/analyze_g2.py,
               analyze_bursts.py).
  xi           shape of a generalized Pareto distribution fitted by probability-weighted
               moments to the exceedances over the run's p99 (analyze_bursts.py gpd_pwm).
  tail cycles  the cycles over the p99.9 deadline: mean compute_us and mean
               release_jitter_us of those cycles minus the run's means.
  lag-1        sample autocorrelation sum((x_t - m)(x_t+1 - m)) / sum((x_t - m)^2).
  slow kernel  a GEMM kernel whose duration exceeds 1.1 x the cell's median duration (the
               criterion of campaign/mech_kdur_extract.py); results/mech_dose/kdur_*.txt and
               mech_nx/kdur_*.txt hold the 12,160 durations (ns) of each cell, sorted;
               mech_dose/ts_ts_e<E>_g<f>.csv the same kernels in time order (start_ns,
               dur_ns): 190 inferences of 64 kernels (20 warm-up + 170 timed).
  live         live_gov/emcsw_{static2133,static3199,trigger}_r{1,2,3}.csv (NX, MobileNetV2,
               GPU 918 MHz, D = 5 ms, 1,000 cycles, campaign/nx_live_gov.py --mode emcswitch);
               emc_hz is the BPMP rate read back right after the cycle (after the switch
               on the triggering cycle).  Recovery: the first cycle after the trigger whose
               response is at or under the pooled p99 of the static-3199 runs (and under D).
"""
import argparse
import csv
import glob
import os
import sys

import numpy as np

LAB = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ZENODO = "the Zenodo deposit 10.5281/zenodo.21236659"


def first(R, rels):
    for r in rels:
        p = os.path.join(R, r)
        if os.path.isfile(p):
            return p
    return None


def read_run(path):
    a = np.loadtxt(path, delimiter=",", skiprows=1, usecols=(1, 2, 3))
    return a[:, 0], a[:, 1], a[:, 2]          # jitter, compute, response (us)


def gpd_pwm(excess):
    x = np.sort(excess)
    n = len(x)
    if n < 20:
        return None
    b0 = x.mean()
    p = 1.0 - (np.arange(1, n + 1) - 0.35) / n
    b1 = (x * p).mean()
    den = b0 - 2 * b1
    return None if abs(den) < 1e-12 else 2.0 - b0 / den


def lag1(x):
    x = np.asarray(x, float) - np.mean(x)
    return float(np.sum(x[:-1] * x[1:]) / np.sum(x * x))


def run_stats(path):
    jit, comp, resp = read_run(path)
    p50, q999, q9999 = np.percentile(resp, [50, 99.9, 99.99])
    miss = resp > q999
    pm = miss.mean()
    cont = np.count_nonzero(miss[1:] & miss[:-1]) / max(np.count_nonzero(miss[:-1]), 1)
    lens, k = [], 0
    for m in miss:
        if m:
            k += 1
        elif k:
            lens.append(k)
            k = 0
    if k:
        lens.append(k)
    u = np.percentile(resp, 99.0)
    return dict(n=len(resp), p50=p50 / 1e3, r9999=q9999 / p50, r999=q999 / p50, cont=cont,
                ratio=cont / pm, nb=len(lens), maxb=max(lens), xi=gpd_pwm(resp[resp > u] - u),
                maxms=resp.max() / 1e3, comp_ex=comp[miss].mean() - comp.mean(),
                jit_ex=jit[miss].mean() - jit.mean(), ac1=lag1(comp))


def print_runs(rows):
    if not rows:
        return
    print(f"  {'run':47s} {'p50 ms':>7s} {'p9999/p50':>9s} {'p999/p50':>8s} {'P[cont]':>7s} "
          f"{'ratio':>6s} {'bursts':>6s} {'max':>4s} {'xi':>7s} {'max ms':>7s}")
    for name, s in rows:
        xi = f"{s['xi']:+7.3f}" if s["xi"] is not None else "    n/a"
        print(f"  {name:47s} {s['p50']:7.3f} {s['r9999']:9.4f} {s['r999']:8.4f} {s['cont']:7.3f} "
              f"{s['ratio']:6.0f} {s['nb']:6d} {s['maxb']:4d} {xi} {s['maxms']:7.2f}")


def missing(rel, archive):
    print(f"  {rel}: not present; extract {archive} from {ZENODO} into results/")


# --------------------------------------------------------------------------- worst
def sec_worst(R):
    print("== [Supp. Sec. S5; Fig. S7 caption] the worst cell (Nano, MobileNetV2, EMC 2133 MHz, "
          "two-core write adversary) over its four 100k-cycle runs ==")
    cells = [("run 1  campaign_g2/partB/emc2133_adv2_mobilenet", "campaign_g2/partB/emc2133_adv2_mobilenet.csv")]
    cells += [(f"run {k}  e4_replication/nano_rep_emc2133_adv2_r{k}", f"e4_replication/nano_rep_emc2133_adv2_r{k}.csv")
              for k in (2, 3, 4)]
    rows = []
    for name, rel in cells:
        p = first(R, [rel])
        if p is None:
            missing(rel, "e4_replication_nano.tar.gz")
            continue
        rows.append((name, run_stats(p)))
    print_runs(rows)
    if len(rows) >= 2:
        r = np.array([s["ratio"] for _, s in rows])
        xs = [s["xi"] for _, s in rows if s["xi"] is not None]
        print(f"  over {len(rows)} runs: p99.99/p50 {min(s['r9999'] for _, s in rows):.4f}-"
              f"{max(s['r9999'] for _, s in rows):.4f}; p99.9/p50 max {max(s['r999'] for _, s in rows):.4f}; "
              f"clustering ratio {r.min():.0f}-{r.max():.0f}x, mean {r.mean():.1f}, s.d. {r.std():.1f} "
              f"(n-1: {r.std(ddof=1):.1f}); xi {min(xs):+.3f} to {max(xs):+.3f}")
    print()


# --------------------------------------------------------------------------- nx
def nx_path(R, name):
    return first(R, [f"e4_replication/{name}.csv", f"e4_nx/{name}.csv"])


def sec_nx(R):
    print("== [Supp. Sec. S5, Sec. S6] the same cell on the NX (MobileNetV2, EMC 2133 MHz, two-core "
          "adversary, 50 Hz, 100k cycles) and the NX and Nano Part-B cells ==")
    rows = []
    p = first(R, ["nx_validation/nx_burst_emc2133_adv2_mobilenet.csv"])
    if p:
        rows.append(("nx_validation/nx_burst_emc2133_adv2_mobilenet", run_stats(p)))
    for name in ("nx_pb_emc2133_adv2", "nx_rep_emc2133_adv2_r2", "nx_rep_emc2133_adv2_r3"):
        q = nx_path(R, name)
        if q is None:
            missing(f"e4_replication/{name}.csv (or e4_nx/)", "e4_replication_nano.tar.gz or partb_nx.tar.gz")
        else:
            rows.append((os.path.relpath(q, R)[:-4], run_stats(q)))
    print_runs(rows)
    if rows:
        r = [s["ratio"] for _, s in rows]
        print(f"  NX runs of this cell: {len(rows)}; clustering ratio {min(r):.0f}-{max(r):.0f}x "
              "(" + ", ".join(f"{x:.0f}" for x in r) + ")")
    print("  -- NX Part-B cells (EMC x adversary) --")
    pb = []
    for e in (2133, 3199):
        for adv in (0, 2):
            q = nx_path(R, f"nx_pb_emc{e}_adv{adv}")
            if q is not None:
                pb.append((os.path.relpath(q, R)[:-4], run_stats(q)))
    if pb:
        print_runs(pb)
        allnx = pb + [x for x in rows if "nx_pb_" not in x[0]]
        xs = [s["xi"] for _, s in allnx]
        print(f"  NX xi over these {len(allnx)} runs: {min(xs):+.3f} to {max(xs):+.3f}; largest response "
              f"{max(s['maxms'] for _, s in allnx):.2f} ms")
    else:
        missing("e4_replication/nx_pb_emc*_adv*.csv (or e4_nx/)", "e4_replication_nano.tar.gz or partb_nx.tar.gz")
    print("  -- Nano Part-B MobileNetV2 cells (campaign_g2/partB) --")
    nano = []
    for e in (2133, 3199):
        for adv in (0, 2, 4):
            q = first(R, [f"campaign_g2/partB/emc{e}_adv{adv}_mobilenet.csv"])
            if q:
                nano.append((os.path.relpath(q, R)[:-4], run_stats(q)))
    print_runs(nano)
    if nano:
        xs = [s["xi"] for _, s in nano]
        print(f"  Nano MobileNetV2 xi {min(xs):+.3f} to {max(xs):+.3f}; largest response "
              f"{max(s['maxms'] for _, s in nano):.2f} ms")
    print()


# --------------------------------------------------------------------------- ext
def sec_ext(R):
    print("== [Supp. Sec. S6] the 100k-cycle protocol extended to ViT-Small and the L2-resident GEMM "
          "(Nano, EMC x two-core adversary) ==")
    rows, p50 = [], {}
    for w in ("vit", "cpx"):
        for e in (2133, 3199):
            for adv in (0, 2):
                rel = f"partb_ext/{w}_emc{e}_adv{adv}.csv"
                q = first(R, [rel])
                if q is None:
                    missing(rel, "partb_ext_nano.tar.gz")
                    continue
                s = run_stats(q)
                rows.append((rel[:-4], s))
                p50[(w, e, adv)] = s["p50"]
    print_runs(rows)
    for w, label in (("vit", "ViT-Small"), ("cpx", "L2-resident GEMM")):
        g = [s for n, s in rows if f"/{w}_" in n]
        if g:
            print(f"  {label}: clustering ratio {min(s['ratio'] for s in g):.0f}-{max(s['ratio'] for s in g):.0f}x, "
                  f"longest burst {max(s['maxb'] for s in g)}")
    for adv, tag in ((0, "uncontended"), (2, "two-core adversary")):
        if ("cpx", 2133, adv) in p50 and ("cpx", 3199, adv) in p50:
            print(f"  L2-resident GEMM p50 at 3199 vs 2133 MHz, {tag}: {p50[('cpx', 3199, adv)]:.3f} vs "
                  f"{p50[('cpx', 2133, adv)]:.3f} ms ({(p50[('cpx', 3199, adv)] / p50[('cpx', 2133, adv)] - 1) * 100:+.2f} %)")
    print()


# --------------------------------------------------------------------------- cycles
def sec_cycles(R):
    print("== [Supp. Sec. S6] tail cycles of the worst cell: compute and release-jitter excess over "
          "the run means, compute lag-1 autocorrelation ==")
    for name, rel in (("Nano, run 1", "campaign_g2/partB/emc2133_adv2_mobilenet.csv"),
                      ("NX, nx_validation run", "nx_validation/nx_burst_emc2133_adv2_mobilenet.csv")):
        p = first(R, [rel])
        if p is None:
            print(f"  {name}: missing {rel}")
            continue
        s = run_stats(p)
        print(f"  {name:22s} cycles over p99.9: compute +{s['comp_ex']:.0f} us, release jitter "
              f"+{s['jit_ex']:.0f} us; compute lag-1 {s['ac1']:.3f}; P[cont] {s['cont']:.3f} "
              f"({s['ratio']:.0f}x)")
    print()


# --------------------------------------------------------------------------- slow
def kdur(path):
    return np.loadtxt(path) / 1e3              # us


def sec_slow(R):
    print("== [Supp. Sec. S6; Fig. S9; Table S26] per-kernel GEMM durations (Nano, L2-resident GEMM, "
          "Nsight Systems, 12,160 kernels per cell); slow = over 1.1 x the cell median ==")
    cells = {}
    for p in sorted(glob.glob(f"{R}/mech_dose/kdur_*.txt")) + sorted(glob.glob(f"{R}/mech_nx/kdur_*.txt")):
        cells[os.path.relpath(p, R)[:-4]] = kdur(p)
    if not cells:
        print("  no mech_dose / mech_nx kernel files\n")
        return
    print(f"  {'cell':34s} {'n':>6s} {'median':>7s} {'mean':>7s} {'s.d.':>6s} {'slow %':>7s} "
          f"{'fast med':>8s} {'slow med':>8s}")
    stat = {}
    for name, d in cells.items():
        med = float(np.median(d))
        slow = d > 1.1 * med
        fm = float(np.median(d[~slow]))
        sm = float(np.median(d[slow])) if slow.any() else float("nan")
        stat[name] = (med, d.mean(), d.std(), slow.mean() * 100, fm, sm)
        print(f"  {name:34s} {len(d):6d} {med:7.1f} {d.mean():7.1f} {d.std():6.1f} {slow.mean() * 100:7.2f} "
              f"{fm:8.1f} {sm:8.1f}")
    a, b = stat.get("mech_dose/kdur_e2133_g1020"), stat.get("mech_dose/kdur_e3199_g1020")
    if a and b:
        print(f"  EMC 2133 -> 3199 at GPU 1020 MHz: mean {a[1]:.1f} -> {b[1]:.1f} us "
              f"({(b[1] / a[1] - 1) * 100:+.2f} %), s.d. {a[2]:.1f} -> {b[2]:.1f} us (x{b[2] / a[2]:.1f}); "
              f"3199: fast {100 - b[3]:.1f} % at {b[4]:.1f} us, slow {b[3]:.1f} % at {b[5]:.1f} us "
              f"({(b[5] / b[4] - 1) * 100:+.1f} % over the fast mode); 2133 median {a[0]:.1f} us")
    print("  slow share at EMC 3199 MHz across the GPU clock (repeats: mean, min-max):")
    for g in (612, 816, 918, 1020):
        reps = [v[3] for k, v in stat.items() if k.startswith(f"mech_dose/kdur_e3199_g{g}_rep")]
        single = stat.get(f"mech_dose/kdur_e3199_g{g}")
        parts = []
        if single:
            parts.append(f"cell {single[3]:.2f} %")
        if reps:
            parts.append(f"{len(reps)} repeats mean {np.mean(reps):.2f} % ({min(reps):.2f}-{max(reps):.2f})")
        print(f"    GPU {g:4d}: " + "; ".join(parts))
    nx = {k: v for k, v in stat.items() if k.startswith("mech_nx/")}
    if nx:
        print(f"  NX (same workload, GPU 1173 MHz): slow share {min(v[3] for v in nx.values()):.2f}-"
              f"{max(v[3] for v in nx.values()):.2f} % over {len(nx)} cells at EMC 2133 and 3199 MHz")
    for tag in ("e3199_g1020", "e2133_g1020"):
        p = f"{R}/mech_dose/ts_ts_{tag}.csv"
        if not os.path.isfile(p):
            continue
        t = np.loadtxt(p, delimiter=",", skiprows=1)
        dur = t[:, 1] / 1e3
        slow = dur > 1.1 * np.median(dur)
        lens, k = [], 0
        for s in slow:
            if s:
                k += 1
            elif k:
                lens.append(k)
                k = 0
        if k:
            lens.append(k)
        n_inf = len(dur) // 64
        per = [int(slow[i * 64:(i + 1) * 64].sum()) for i in range(n_inf)]
        dist = {L: lens.count(L) for L in sorted(set(lens))}
        print(f"  time order, ts_ts_{tag}: {len(dur)} kernels = {n_inf} inferences x 64; slow {slow.mean() * 100:.2f} %; "
              f"runs of consecutive slow kernels: {len(lens)}"
              + (f", median {np.median(lens):.0f}, max {max(lens)}, lengths {dist}" if lens else "")
              + f"; slow kernels per inference {min(per)}-{max(per)}")
    print("  (not from released files: the isolated Nsight Compute replay values and the "
          "fine-grained GPU-clock/temperature/power trace of Supp. Sec. S6)")
    print()


# --------------------------------------------------------------------------- live
def read_live(path):
    rows = list(csv.DictReader(open(path)))
    return (np.array([float(r["response_us"]) for r in rows]), np.array([int(r["miss"]) for r in rows]),
            np.array([int(r["switched"]) for r in rows]), np.array([int(r["emc_hz"]) for r in rows]))


def sec_live(R):
    print("== [Supp. Sec. S7; main Sec. VI] live first-miss EMC upshift (NX, MobileNetV2, GPU 918 MHz, "
          "D = 5 ms, 1,000 cycles per run) ==")
    D = 5000.0
    st3 = []
    for arm in ("static2133", "static3199"):
        ms = []
        for k in (1, 2, 3):
            p = f"{R}/live_gov/emcsw_{arm}_r{k}.csv"
            if not os.path.isfile(p):
                continue
            resp, miss, sw, emc = read_live(p)
            ms.append(miss.mean() * 100)
            if arm == "static3199":
                st3.append(resp)
        if ms:
            print(f"  {arm}: miss {', '.join(f'{m:.1f}' for m in ms)} %")
    if not st3:
        print("  missing static runs\n")
        return
    thr = float(np.percentile(np.concatenate(st3), 99))
    print(f"  static-3199 pooled p99 (recovery level): {thr / 1e3:.3f} ms")
    for k in (1, 2, 3):
        p = f"{R}/live_gov/emcsw_trigger_r{k}.csv"
        if not os.path.isfile(p):
            continue
        resp, miss, sw, emc = read_live(p)
        i = int(np.argmax(sw == 1)) if sw.any() else None
        if i is None:
            print(f"  trigger r{k}: no switch")
            continue
        j = i + 1
        while j < len(miss) and miss[j]:
            j += 1
        extra = j - (i + 1)
        rec = next((c for c in range(i + 1, len(resp)) if resp[c] <= thr), None)
        rec_d = next((c for c in range(i + 1, len(resp)) if resp[c] <= D), None)
        later = [int(c) for c in np.flatnonzero(miss) if c > j]
        print(f"  trigger r{k}: first miss at cycle {int(np.argmax(miss == 1))}, switch at cycle {i}, EMC read back "
              f"{emc[i] / 1e6:.0f} MHz on that cycle and {emc[i + 1] / 1e6:.0f} MHz at the next release; "
              f"consecutive misses after the trigger {extra}; back under D after {rec_d - i} cycle(s), at the "
              f"static-3199 p99 after {rec - i}; later misses {len(later)}"
              + (f" (cycles {', '.join(map(str, later))}; {', '.join(f'{resp[c] / 1e3:.2f}' for c in later)} ms)"
                 if later else ""))
    print()


SECTIONS = {"worst": sec_worst, "nx": sec_nx, "ext": sec_ext, "cycles": sec_cycles,
            "slow": sec_slow, "live": sec_live}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--section", action="append", choices=sorted(SECTIONS),
                    help="print only this section (repeatable)")
    ap.add_argument("--results", default=os.path.join(LAB, "results"),
                    help="results directory (default: the checkout's results/)")
    a = ap.parse_args()
    R = os.path.abspath(a.results)
    if not os.path.isdir(R):
        sys.exit(f"no results directory at {R}")
    for s in a.section or list(SECTIONS):
        SECTIONS[s](R)


if __name__ == "__main__":
    main()

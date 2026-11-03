#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Fresh-probe end-to-end trials at the calibration condition (NX, EMC 2133 MHz, the 10 Hz ViT-Small
co-runner): the Supp. fresh-probe summary and trial tables, and the fresh-launch rows of main-text
Table VII (round-1 fresh probes, held-out banded launches, one-bin headroom).

Each trial is a fresh co-runner launch -> a fresh probe -> a fresh admission decision -> a disjoint
production contract.  Series:
  round 1, vision   results/fresh_loop/vprobe_t<k>.csv (200-cycle MobileNetV2 probe at 1173 MHz),
                    vdep_t<k>_g<f>.csv (1,000-cycle contract at the pick, D = 12 ms) and the recorded
                    M_g and pick in decisions.csv (unbanded maximum guard)
  round 1, decode   dprobe_t<k>.csv (200-token Qwen2.5-1.5B probe at 1173 MHz), ddep_t<k>_D<D>.csv
                    (2,000-token contract at 1173 MHz, D = the trial's first admissible deadline) and the
                    recorded M_g, deadline and quote in decisions.csv
  held-out, banded  results/banded_loop/probe_t<k>.csv and dep_t<k>_g<f>.csv (band h = 400 us)
  one-bin headroom  results/launch1020/dep_t1..8 and results/launch_ext/nx/mobilenet_vit10_g1020_D12/t<k>
                    (read with campaign/launch_ext_stats.py load_launch; its own report is that script)
Miss: vision response_us > 12,000 us per 1,000-cycle contract; decode dur_ns > D per token (strict).
Aggregate of a series: mean of the per-launch miss % (equal to the pooled rate: the contracts of a series
have equal length) with its launch-level 95 % upper bound, the 95th percentile of the mean per-launch miss
over B = 20,000 resamples of whole launches (numpy default_rng(42), as campaign/admission_runs.py);
B = 10,000 with seed 0 and campaign/e3_stats.py's pooled bootstrap (B = 10,000, seed 20260915) are
printed as checks.  Launch counts over tau = 2 % carry the one-sided 95 % Clopper-Pearson bound.
Decisions: the vision M_g and picks are re-derived from the probes with campaign/admit_pick.py evaluate
(the transcription admit_pick.py --selftest checks against the deployed code; original 8 Hz cells),
and the vision slack is m*(f) - M_g at the recorded pick, as campaign/replay_bandwidth.py prints it.  The
decode decisions are re-derived with the decode rule of campaign/fresh_pick.py: M_g = max - p50 of the
probe, D* = (t_co (1 + gamma 2000) r + M_g) / 1000 ms with t_co the probe median, gamma from the first
five solo 2,000-token runs at 1173 MHz (results/boundary_stats/q15_g1173_r1..5; windows [0,500) and
[1500,2000), pooled medians, divisor 1500) and r = 1.035, deadline = ceil(D*); both are compared with
decisions.csv.  Also printed: the correlation of the held-out probe margins with the held-out launch
misses, and the descriptive pool of every 918 MHz launch of round 1 and the held-out round.

Read-only on results/ (python3 + numpy); prints to stdout, writes nothing.
Usage: fresh_series.py [--json]
"""
import argparse
import csv
import glob
import json
import math
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LAB = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from admit_pick import load_probe, load_cells, evaluate, CELLS, GRID, FPROBE, R_MULT, H_US   # noqa: E402
from launch_ext_stats import load_launch, cp_upper, COMMITTED_PREFIX, COMMITTED_N, COMMITTED_TAG   # noqa: E402
from e3_stats import bootstrap_pooled                                                        # noqa: E402

RES = os.path.join(LAB, "results")
D_VISION_US = 12000.0
TAU = 2.0


def tnum(path, rx):
    return int(re.search(rx, os.path.basename(path)).group(1))


def numbered(pattern, rx):
    return sorted(glob.glob(pattern), key=lambda p: tnum(p, rx))


def per_token_us(path):
    return np.array([float(r["dur_ns"]) for r in csv.DictReader(open(path)) if r.get("dur_ns")]) / 1e3


def boot_ub(per, B, seed):
    rng = np.random.default_rng(seed)
    return float(np.quantile([rng.choice(per, len(per), replace=True).mean() for _ in range(B)], 0.95))


def aggregate(misses, n):
    """Series aggregate: per-launch miss %, mean, launch-level UB (and its checks), range, launches over tau."""
    misses, n = np.asarray(misses), np.asarray(n)
    per = 100.0 * misses / n
    x = int(np.sum(per > TAU))
    return dict(N=int(len(per)), per_pct=[float(v) for v in per], mean_pct=float(per.mean()),
                pooled_pct=float(100.0 * misses.sum() / n.sum()),
                ub_pct=boot_ub(per, 20000, 42), ub_check_pct=boot_ub(per, 10000, 0),
                ub_e3_pct=100.0 * bootstrap_pooled(misses, n, 10000, 20260915)[1],
                min_pct=float(per.min()), max_pct=float(per.max()), over_tau=x,
                cp95_over_tau_pct=100.0 * cp_upper(x, len(per), 0.05))


# ------------------------------------------------------------------ series ----
def round1_vision(cells):
    dec = {int(x["trial"]): x for x in csv.DictReader(open(f"{RES}/fresh_loop/decisions.csv")) if x["arm"] == "vision"}
    out = []
    for p in numbered(f"{RES}/fresh_loop/vprobe_t*.csv", r"vprobe_t(\d+)\.csv$"):
        k = tnum(p, r"vprobe_t(\d+)\.csv$")
        r = evaluate(load_probe(p), cells, D_VISION_US, GRID["nx"], FPROBE["nx"], 0.0, R_MULT)
        dep = glob.glob(f"{RES}/fresh_loop/vdep_t{k}_g*.csv")
        assert len(dep) == 1, (k, dep)
        f = tnum(dep[0], r"_g(\d+)\.csv$")
        v = load_probe(dep[0])
        pf = {x["f"]: x for x in r["per_f"]}
        out.append(dict(k=k, Mg_rec=float(dec[k]["Mg"]), pick_rec=int(dec[k]["pick"]), Mg=r["Mg_us"], f_pt=r["f_pt"],
                        f=f, slack_us=pf[f]["margin_mg_us"], misses=int(np.sum(v > D_VISION_US)), n=int(v.size),
                        file=os.path.relpath(dep[0], LAB)))
    return out


def solo_gamma():
    solo = [per_token_us(f) for f in sorted(glob.glob(f"{RES}/boundary_stats/q15_g1173_r*.csv"))[:5]]
    return float((np.median(np.concatenate([r[1500:2000] for r in solo])) /
                  np.median(np.concatenate([r[:500] for r in solo])) - 1) / 1500)


def round1_decode():
    dec = {int(x["trial"]): x for x in csv.DictReader(open(f"{RES}/fresh_loop/decisions.csv")) if x["arm"] == "decode"}
    gam = solo_gamma()
    out = []
    for p in numbered(f"{RES}/fresh_loop/dprobe_t*.csv", r"dprobe_t(\d+)\.csv$"):
        k = tnum(p, r"dprobe_t(\d+)\.csv$")
        v = per_token_us(p)
        Mg = float(v.max() - np.median(v))
        Dstar = (float(np.median(v)) * (1 + gam * 2000) * R_MULT + Mg) / 1000.0
        dep = glob.glob(f"{RES}/fresh_loop/ddep_t{k}_D*.csv")
        assert len(dep) == 1, (k, dep)
        D = tnum(dep[0], r"_D(\d+)\.csv$")
        w = per_token_us(dep[0])
        out.append(dict(k=k, Mg_rec_us=float(dec[k]["Mg"]), D_rec=int(dec[k]["pick"]), Dstar_rec=float(dec[k]["extra"]),
                        Mg_us=Mg, Dstar_ms=Dstar, D_rule=int(math.ceil(Dstar)), D=D,
                        misses=int(np.sum(w > D * 1000.0)), n=int(w.size), file=os.path.relpath(dep[0], LAB)))
    return out, gam


def heldout(cells):
    out = []
    for p in numbered(f"{RES}/banded_loop/probe_t*.csv", r"probe_t(\d+)\.csv$"):
        k = tnum(p, r"probe_t(\d+)\.csv$")
        r = evaluate(load_probe(p), cells, D_VISION_US, GRID["nx"], FPROBE["nx"], H_US, R_MULT)
        dep = glob.glob(f"{RES}/banded_loop/dep_t{k}_g*.csv")
        assert len(dep) == 1, (k, dep)
        f = tnum(dep[0], r"_g(\d+)\.csv$")
        v = load_probe(dep[0])
        out.append(dict(k=k, Mg=r["Mg_us"], f_pt=r["f_pt"], f_b=r["f_b"], f=f, misses=int(np.sum(v > D_VISION_US)),
                        n=int(v.size), file=os.path.relpath(dep[0], LAB)))
    return out


def headroom():
    """The committed eight launches and the launch_ext series at the one-bin-headroom design point."""
    ls = []
    for k in range(1, COMMITTED_N + 1):
        r = load_launch(f"{COMMITTED_PREFIX}{k}", D_VISION_US)
        if r:
            r["csv"] = os.path.relpath(r["csv"], LAB)
            ls.append(dict(k=k, src="launch1020", **r))
    ext = os.path.join(RES, "launch_ext", "nx", COMMITTED_TAG)
    for p in sorted(glob.glob(os.path.join(ext, "t*.csv"))):
        m = re.search(r"/t(\d+)\.csv$", p)
        r = load_launch(p[:-4], D_VISION_US) if m else None
        if r:
            r["csv"] = os.path.relpath(r["csv"], LAB)
            ls.append(dict(k=int(m.group(1)), src="launch_ext", **r))
    return ls


def pearson(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    return float(np.corrcoef(x, y)[0, 1])


def build():
    cells = load_cells(CELLS[("nx", "mobilenet")])
    v1 = round1_vision(cells)
    d1, gam = round1_decode()
    ho = heldout(cells)
    hr = headroom()
    agg = dict(
        r1_918=aggregate([t["misses"] for t in v1 if t["f"] == 918], [t["n"] for t in v1 if t["f"] == 918]),
        r1_816=aggregate([t["misses"] for t in v1 if t["f"] == 816], [t["n"] for t in v1 if t["f"] == 816]),
        r1_all=aggregate([t["misses"] for t in v1], [t["n"] for t in v1]),
        heldout=aggregate([t["misses"] for t in ho], [t["n"] for t in ho]),
        headroom=aggregate([t["misses"] for t in hr], [t["n"] for t in hr]),
        decode=aggregate([t["misses"] for t in d1], [t["n"] for t in d1]),
        pool_918=aggregate([t["misses"] for t in v1 + ho if t["f"] == 918], [t["n"] for t in v1 + ho if t["f"] == 918]))
    corr = dict(n=len(ho), pearson=pearson([t["Mg"] for t in ho], [100.0 * t["misses"] / t["n"] for t in ho]))
    return dict(round1_vision=v1, round1_decode=d1, gamma_solo=gam, heldout=ho, headroom=hr, agg=agg, corr=corr)


# -------------------------------------------------------------------- main ----
def pct(x, nd=2):
    return f"{x:.{nd}f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", action="store_true", help="print every trial and aggregate as JSON")
    a = ap.parse_args()
    g = build()
    if a.json:
        print(json.dumps(g, indent=1, default=str))
        return
    v1, d1, ho, hr, A = g["round1_vision"], g["round1_decode"], g["heldout"], g["headroom"], g["agg"]

    print("## Round 1, vision arm (MobileNetV2 33 Hz, D = 12 ms, unbanded maximum guard; original 8 Hz cells)\n")
    print("| trial | M_g recorded (ms) | M_g re-derived (us) | pick recorded | pick re-derived | slack m*(f) - M_g (us) | miss % |")
    print("|---:|---:|---:|---:|---:|---:|---:|")
    for t in v1:
        print(f"| {t['k']} | {t['Mg_rec'] / 1000:.2f} | {t['Mg']:.0f} | {t['pick_rec']} | {t['f_pt']} | "
              f"{t['slack_us']:.0f} | {100.0 * t['misses'] / t['n']:.2f} |")
    print(f"recorded = re-derived on {sum(1 for t in v1 if t['pick_rec'] == t['f_pt'] == t['f'] and round(t['Mg']) == t['Mg_rec'])}"
          f"/{len(v1)} trials (M_g, pick and the contract's clock)")

    print(f"\n## Round 1, decode arm (Qwen2.5-1.5B 2,000 tokens at 1173 MHz under the co-runner; solo gamma "
          f"{g['gamma_solo']:.4e} per token)\n")
    print("| trial | M_g recorded (ms) | M_g re-derived (ms) | D* recorded (ms) | D* re-derived (ms) | decision D (ms) | "
          "deadline = ceil(D*) | miss % |")
    print("|---:|---:|---:|---:|---:|---:|---:|---:|")
    for t in d1:
        print(f"| {t['k']} | {t['Mg_rec_us'] / 1000:.1f} | {t['Mg_us'] / 1000:.1f} | {t['Dstar_rec']:.2f} | {t['Dstar_ms']:.2f} | "
              f"{t['D']} | {t['D_rule']} | {100.0 * t['misses'] / t['n']:.2f} |")
    ok = sum(1 for t in d1 if t["D"] == t["D_rule"] == t["D_rec"] and abs(t["Mg_us"] - t["Mg_rec_us"]) < 0.051
             and abs(t["Dstar_ms"] - t["Dstar_rec"]) < 0.0051)
    print(f"recorded = re-derived on {ok}/{len(d1)} trials (M_g to 0.1 us, D* to 0.01 ms, the deadline and the contract file)")

    print("\n## Held-out banded round (h = 400 us, D = 12 ms; original 8 Hz cells)\n")
    print("| trial | M_g (us) | point pick | banded pick | contract clock | miss % |")
    print("|---:|---:|---:|---:|---:|---:|")
    for t in ho:
        print(f"| {t['k']} | {t['Mg']:.0f} | {t['f_pt']} | {t['f_b']} | {t['f']} | {100.0 * t['misses'] / t['n']:.2f} |")
    c = g["corr"]
    print(f"band fired (point pick != banded pick) on {sum(1 for t in ho if t['f_pt'] != t['f_b'])}/{len(ho)}; "
          f"correlation of M_g with the launch miss over the {c['n']} held-out launches: Pearson {c['pearson']:+.2f}")

    def line(label, key, pick, contract):
        s = A[key]
        return (f"| {label} | {s['N']} | {pick} | {contract} | {pct(s['mean_pct'])} [{pct(s['ub_pct'])}] "
                f"(checks {pct(s['ub_check_pct'])}, {pct(s['ub_e3_pct'])}) | {pct(s['min_pct'])}-{pct(s['max_pct'])} | "
                f"{s['over_tau']}/{s['N']} (CP95 {s['cp95_over_tau_pct']:.1f}) |")
    n1 = len(v1)
    print("\n## Fresh-probe series (summary)\n")
    print("| series | launches | pick | contract | mean % [launch-level UB] (B = 10000 checks) | per-launch range % | launches > tau (CP95 %) |")
    print("|---|---:|---|---|---|---|---|")
    print(line(f"vision, single-probe, {A['r1_918']['N']}/{n1} trials at 918", "r1_918", "918", "D = 12 ms"))
    print(line(f"vision, single-probe, {A['r1_816']['N']}/{n1} trials at 816", "r1_816", "816", "D = 12 ms"))
    print(line("vision, banded (held-out)", "heldout", "/".join(sorted({str(t['f']) for t in ho})), "D = 12 ms"))
    print(line("vision, one-bin headroom", "headroom", "1020", "D = 12 ms"))
    ds = sorted({t["D"] for t in d1})
    print(line("decode co, boundary sweep", "decode", "1173", f"first admissible D = {ds[0]}-{ds[-1]} ms"))
    print(f"\nheadroom launches: {sum(1 for t in hr if t['src'] == 'launch1020')} from launch1020, "
          f"{sum(1 for t in hr if t['src'] == 'launch_ext')} from launch_ext; misses {sum(t['misses'] for t in hr)} in "
          f"{sum(t['n'] for t in hr)} cycles")
    p = A["pool_918"]
    print(f"descriptive pool of every 918 MHz launch (round 1 and held-out, {p['N']} launches): "
          f"{pct(p['mean_pct'])} % [{pct(p['ub_pct'])}] (checks {pct(p['ub_check_pct'])}, {pct(p['ub_e3_pct'])}); "
          f"launches > tau {p['over_tau']}/{p['N']}")
    r1 = A["r1_all"]
    print(f"\nTable VII, fresh-launch rows: round-1 fresh probes, unbanded: n = {r1['N']}, launches > tau {r1['over_tau']}/{r1['N']} "
          f"(at {', '.join(sorted({str(t['f']) for t in v1 if 100.0 * t['misses'] / t['n'] > TAU}))} MHz: "
          f"{', '.join(pct(100.0 * t['misses'] / t['n'], 1) for t in v1 if 100.0 * t['misses'] / t['n'] > TAU)} %); "
          f"held-out banded launches: n = {A['heldout']['N']}, {pct(A['heldout']['mean_pct'])} [{pct(A['heldout']['ub_pct'])}], "
          f"launches > tau {A['heldout']['over_tau']}/{A['heldout']['N']}; one-bin headroom: n = {A['headroom']['N']}, "
          f"{pct(A['headroom']['mean_pct'])}, launches > tau {A['headroom']['over_tau']}/{A['headroom']['N']}, "
          f"one-sided CP bound {A['headroom']['cp95_over_tau_pct']:.1f} %")


if __name__ == "__main__":
    main()

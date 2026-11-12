#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Safety-multiplier sensitivity: the co-tenancy, horizon and joint-stress
admission contracts (main-text admission-outcome table, label tab:admission,
and the co-tenancy ladder, label tab:jointgov) replayed at several values of
the multiplier r over the deployed traces (supplement, 'Margin provenance and
sensitivity', label tab:rsens).  No board runs: r is the only parameter that
changes; every decision is recomputed from the committed fits and probes, and
every outcome is read from the runs already deployed at that (clock, D).

Vision (NX, MobileNetV2 at 33 Hz against the 10 Hz ViT-Small co-runner, EMC 2133;
joint_gov_policy.py, analyze_joint.py guard()):
  T(f)    = k/f + b through the GPU 306/1173 MHz endpoint cells at EMC 2133
            (results/nx_validation, response p50; the original 8 Hz cells)
  probe   = results/joint_gov/probe_co_g1173.csv (200 cycles at f_probe = 1173 MHz)
  point   : lowest f in {510, ..., 1173} with T(f) r + M   f_probe/f <= D,  M   = p98 - p50
  guarded : the same with the maximum guard M_g = max - p50
  ladder rungs outside the pick count: blind (least squares over the EMC-3199
  sweep) and aware with no margin, and the flat margin T(f) r + M <= D.
  D = 9 and 12 ms; outcome = results/joint_gov/dep_co_g<f>_r*.csv.
Horizon (Qwen2.5-1.5B Q4_K_M decode, solo; analyze_joint.py):
  T(f)     = median of tokens 0-299, gamma(f) = (median of tokens 1500-1999 /
             median of tokens 0-499 - 1) / 1500, pooled over the solo 2,000-token
             runs results/boundary_stats/q15_g<f>_r*.csv
  admit (D, H) at the lowest f with T(f) (1 + gamma(f) H) r <= D;
  H*(f) = (D / (r T(f)) - 1) / gamma(f), quoted at the top clock.
  D = 28 ms per token, H = 2,000 (outcome: the same solo runs) and 4,000
  (results/horizon_4k/q15h4k_g<f>_r*.csv).  Ladder rung outside the pick count:
  the same rule run with the co-runner present (tenancy-blind; outcome below).
Joint stress (2,000-token decode against the same co-runner):
  D*(f) = T(f) dil (1 + gamma(f) 2000) r + M_g f_probe/f,  dil = probe p50 / T(1173),
  M_g = max - p50 of results/joint_hxt/probe_co_g1173.csv (200 tokens at 1173 MHz);
  admit D at the lowest f with D*(f) <= D; first accepted deadline = the smallest
  integer ms D that admits; outcome = results/joint_hxt/q15co_g<f>_r*.csv.
  (With M = p98 - p50 in place of M_g, D*(1173) is printed for comparison.)
Outcome: per-run miss % (response or token latency > D) at the picked clock, or,
for a declined contract, at the highest clock deployed for it (the strongest
declined candidate); 'measured' = the mean of the per-run rates with the
run-level bootstrap 95 % upper bound (runs resampled with replacement,
B = 20000, seed 42), as in analyze_joint.py.  Run-level feasible (main text):
the bound at or below tau = 2 % for n >= 8 runs, every run within tau for a
three-run cell.
False admission: a contract of the admission-outcome table (guarded composed
policy; plus the joint-stress contract at its first accepted D) admitted at r
and not run-level feasible.  Picks changed: point and guarded vision picks at
both deadlines that differ from those at the reference r = 1.035.

Read-only on results/ (python3 + numpy); prints to stdout, writes nothing.
Usage: replay_rsens.py [--r 1.02,1.035,1.05] [--tau 2] [--B 20000] [--seed 42] [--json]
"""
import argparse, csv, glob, json, math, os, re, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LAB = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from admit_pick import load_probe, load_cells, fit_2cell, GRID, FPROBE, CELLS   # noqa: E402
from joint_gov_policy import fit_sweep                                            # noqa: E402

R_REF = 1.035
FP = FPROBE["nx"]
VISION_D = (9000.0, 12000.0)
DECODE_D, DECODE_H = 28000.0, (2000, 4000)
JOINT_H = 2000
JOINT_D_TABLE = (28000.0, 43000.0, 44000.0)       # decode co rows of the admission-outcome table


# ----------------------------------------------------------------------------- data
def run_no(path):
    return int(re.search(r"_r(\d+)\.csv$", path).group(1))


def runs_by_clock(pattern, col):
    """{gpu MHz: [per-run array in us]} for files named ..._g<f>_r<k>.csv, runs in k order."""
    out = {}
    for p in sorted(glob.glob(pattern), key=lambda p: (int(re.search(r"_g(\d+)_r", p).group(1)), run_no(p))):
        g = int(re.search(r"_g(\d+)_r", p).group(1))
        if col == "dur_ns":
            v = np.array([float(x["dur_ns"]) for x in csv.DictReader(open(p))]) / 1e3
        else:
            v = load_probe(p)
        out.setdefault(g, []).append(v)
    return out


def dur_probe(path):
    return np.array([float(x["dur_ns"]) for x in csv.DictReader(open(path))]) / 1e3


_SCORED = {}


def outcome(runs, D_us, tau, B, seed):
    """Per-run miss % at D, their mean, the run-level bootstrap 95 % bound and the verdict."""
    if not runs:
        return None
    key = (id(runs), D_us, tau, B, seed)
    if key not in _SCORED:
        per = np.array([np.mean(v > D_us) * 100 for v in runs])
        rng = np.random.default_rng(seed)
        ub = float(np.quantile([rng.choice(per, len(per), replace=True).mean() for _ in range(B)], 0.95))
        ok = bool(np.all(per <= tau)) if len(per) < 8 else ub <= tau
        _SCORED[key] = dict(n=len(per), runs=[float(x) for x in per], mean=float(per.mean()), lo=float(per.min()),
                            hi=float(per.max()), ub=ub, run_level_feasible=ok)
    return _SCORED[key]


def first_feasible(grid, ok):
    return next((f for f in grid if ok(f)), None)


def decision(f, runs, D_us, tau, B, seed):
    """Pick (None = declined) with its outcome; a declined contract is scored at the highest
    deployed clock instead (the strongest declined candidate)."""
    at = f if f is not None else (max(runs) if runs else None)
    return dict(pick=f, at=at, outcome=None if at is None else outcome(runs.get(at, []), D_us, tau, B, seed))


# ----------------------------------------------------------------------------- replay
def replay(r, d, tau, B, seed):
    out = dict(r=r)
    # vision: ladder rungs (only point and guarded enter the pick count)
    T = lambda f, kb: kb[0] * 1e3 / f + kb[1]
    rungs = (("blind (solo)", d["blind"], 0.0, False),
             ("aware (solo)", d["aware"], 0.0, False),
             ("flat M", d["aware"], d["M"], False),
             ("point M f_probe/f", d["aware"], d["M"], True),
             ("guarded M_g f_probe/f", d["aware"], d["Mg"], True))
    vis = []
    for D in VISION_D:
        for name, kb, m, scaled in rungs:
            f = first_feasible(GRID["nx"], lambda g: T(g, kb) * r + (m * FP / g if scaled else m) <= D)
            vis.append(dict(D_us=D, policy=name, counted=scaled, **decision(f, d["dep"], D, tau, B, seed)))
    out["vision"] = vis
    # horizon (solo decode)
    hz = {}
    for H in DECODE_H:
        f = first_feasible(sorted(d["solo"]), lambda g: d["T"][g] * (1 + d["gam"][g] * H) * r <= DECODE_D)
        hz[H] = decision(f, d["solo"] if H == 2000 else d["h4k"], DECODE_D, tau, B, seed)
    top = max(d["solo"])
    out["hstar"] = {g: (DECODE_D / (r * d["T"][g]) - 1) / d["gam"][g] for g in sorted(d["solo"])}
    out["horizon"] = hz
    # ladder rung: the same tenancy-blind horizon rule with the co-runner present
    out["tenancy_blind_decode"] = decision(hz[2000]["pick"], d["co"], DECODE_D, tau, B, seed)
    # joint stress (decode under the co-runner)
    dstar = {g: (d["T"][g] * d["dil"] * (1 + d["gam"][g] * JOINT_H) * r + d["Mg_dec"] * FP / g)
             for g in sorted(d["solo"])}
    out["dstar_us"] = dstar
    out["dstar_p98_top_us"] = d["T"][top] * d["dil"] * (1 + d["gam"][top] * JOINT_H) * r + d["M_dec"]
    jpick = lambda D: first_feasible(sorted(dstar), lambda g: dstar[g] <= D)
    D1 = float(math.ceil(min(dstar.values()) / 1000.0) * 1000)
    joint = {}
    for D in sorted(set(JOINT_D_TABLE) | {D1}):
        joint[D] = decision(jpick(D), d["co"], D, tau, B, seed)
    out["joint"] = joint
    out["joint_first_D_us"] = D1
    # contracts of the admission-outcome table (+ the joint contract at its first accepted D)
    con = [(f"vision co, D={D/1000:g} ms", next(v for v in vis if v["D_us"] == D and v["policy"].startswith("guarded")))
           for D in VISION_D]
    con += [(f"decode, D={DECODE_D/1000:g}, H={H}", hz[H]) for H in DECODE_H]
    con += [(f"decode co, D={D/1000:g}", joint[D]) for D in sorted(joint)]
    out["contracts"] = [dict(contract=n, pick=c["pick"],
                             false_admission=bool(c["pick"] is not None and c["outcome"] is not None
                                                  and not c["outcome"]["run_level_feasible"]),
                             unscored=bool(c["pick"] is not None and c["outcome"] is None))
                        for n, c in con]
    out["false_admissions"] = sum(c["false_admission"] for c in out["contracts"])
    return out


def load():
    d = {}
    d["aware"] = fit_2cell(load_cells(CELLS[("nx", "mobilenet")]))[:2]
    d["blind"] = fit_sweep(3199)
    pv = load_probe(f"{LAB}/results/joint_gov/probe_co_g1173.csv")
    d["M"] = float(np.quantile(pv, 0.98) - np.median(pv))
    d["Mg"] = float(pv.max() - np.median(pv))
    d["dep"] = runs_by_clock(f"{LAB}/results/joint_gov/dep_co_g*_r*.csv", "response_us")
    d["solo"] = runs_by_clock(f"{LAB}/results/boundary_stats/q15_g*_r*.csv", "dur_ns")
    d["h4k"] = runs_by_clock(f"{LAB}/results/horizon_4k/q15h4k_g*_r*.csv", "dur_ns")
    d["co"] = runs_by_clock(f"{LAB}/results/joint_hxt/q15co_g*_r*.csv", "dur_ns")
    pdec = dur_probe(f"{LAB}/results/joint_hxt/probe_co_g1173.csv")
    d["T"] = {g: float(np.median(np.concatenate([v[:300] for v in L]))) for g, L in d["solo"].items()}
    d["gam"] = {g: float((np.median(np.concatenate([v[1500:2000] for v in L])) /
                          np.median(np.concatenate([v[:500] for v in L])) - 1) / 1500) for g, L in d["solo"].items()}
    d["dil"] = float(np.median(pdec)) / d["T"][max(d["solo"])]
    d["Mg_dec"] = float(pdec.max() - np.median(pdec))
    d["M_dec"] = float(np.quantile(pdec, 0.98) - np.median(pdec))
    return d


# ----------------------------------------------------------------------------- report
def ftxt(o, nd=2):
    if o is None:
        return "no runs"
    sp = f"runs {o['lo']:.{nd}f}-{o['hi']:.{nd}f}" if o["n"] > 1 else "1 run"
    return (f"{o['mean']:.{nd}f}% ({sp}; n={o['n']}, UB {o['ub']:.2f}) "
            f"{'run-level feasible' if o['run_level_feasible'] else 'NOT run-level feasible'}")


def dtxt(x):
    if x["pick"] is not None:
        return f"admit {x['pick']}: {ftxt(x['outcome'])}"
    return "DECLINED" + (f" (highest deployed clock {x['at']}: {ftxt(x['outcome'])})" if x["at"] else "")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--r", default="1.02,1.035,1.05", help="comma list of safety multipliers")
    ap.add_argument("--tau", type=float, default=2.0, help="miss budget, %% of cycles per run")
    ap.add_argument("--B", type=int, default=20000, help="bootstrap resamples (run level)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--json", action="store_true", help="print the numbers as JSON instead of tables")
    a = ap.parse_args()

    rs = [float(x) for x in a.r.split(",")]
    d = load()
    res = [replay(r, d, a.tau, a.B, a.seed) for r in rs]
    ref = replay(R_REF, d, a.tau, a.B, a.seed)
    key = lambda v: (v["D_us"], v["policy"])
    refp = {key(v): v["pick"] for v in ref["vision"]}
    for x in res:
        x["picks_changed"] = sum(v["counted"] and v["pick"] != refp[key(v)] for v in x["vision"])
        x["rungs_changed"] = [f"D={v['D_us']/1000:g} {v['policy']}: {refp[key(v)] or 'DECLINED'} -> {v['pick'] or 'DECLINED'}"
                              for v in x["vision"] if not v["counted"] and v["pick"] != refp[key(v)]]
        if x["tenancy_blind_decode"]["pick"] != ref["tenancy_blind_decode"]["pick"]:
            x["rungs_changed"].append(f"D=28 tenancy-blind decode (co): {ref['tenancy_blind_decode']['pick'] or 'DECLINED'}"
                                      f" -> {x['tenancy_blind_decode']['pick'] or 'DECLINED'}")
    top = max(d["solo"])
    if a.json:
        print(json.dumps(dict(inputs=dict(k_us_GHz=d["aware"][0], b_us=d["aware"][1], blind=list(d["blind"]),
                                          M_us=d["M"], Mg_us=d["Mg"], T_us=d["T"], gamma=d["gam"], dil=d["dil"],
                                          Mg_decode_us=d["Mg_dec"], M_decode_us=d["M_dec"]),
                              replay=res), indent=1, default=str))
        return

    print("=== inputs (r-independent) ===")
    print(f"  vision: aware two-cell fit k={d['aware'][0]:.2f} us*GHz b={d['aware'][1]:.2f} us; blind sweep fit "
          f"k={d['blind'][0]:.2f} b={d['blind'][1]:.2f}; probe M={d['M']/1e3:.2f} ms, M_g={d['Mg']/1e3:.2f} ms")
    print("  decode solo: " + "; ".join(f"g{g}: n={len(d['solo'][g])} T={d['T'][g]/1e3:.2f} ms gamma={d['gam'][g]:.3e}"
                                         for g in sorted(d["solo"])))
    print(f"  joint probe: level x{d['dil']:.3f}, M_g={d['Mg_dec']/1e3:.2f} ms (M={d['M_dec']/1e3:.2f} ms)")
    for x in res:
        r = x["r"]
        print(f"\n=== r = {r:g} ===")
        print("  vision co-tenancy (pick: measured miss at D)")
        for v in x["vision"]:
            tag = "" if v["counted"] else "  [ladder rung, not in the pick count]"
            print(f"    D={v['D_us']/1000:>2g} {v['policy']:<22} {dtxt(v)}{tag}")
        print(f"    picks changed vs r={R_REF:g} (point + guarded, D=9/12): {x['picks_changed']}")
        print("  horizon (solo decode, D=28 ms/token): H* = " +
              ", ".join(f"g{g} {h:,.0f}" if h >= 0 else f"g{g} none (T r > D)" for g, h in x["hstar"].items())
              + " tokens")
        for H, h in x["horizon"].items():
            print(f"    H={H}: {dtxt(h)}")
        print(f"    H=2000 with the co-runner present (tenancy-blind): {dtxt(x['tenancy_blind_decode'])}"
              "  [ladder rung, not in the pick count]")
        print("  joint stress: D*(f) = " + ", ".join(f"g{g} {s/1e3:.2f}" for g, s in x["dstar_us"].items())
              + f" ms (with M = p98-p50 at g{top}: {x['dstar_p98_top_us']/1e3:.2f} ms);"
              f" first accepted D = {x['joint_first_D_us']/1e3:g} ms")
        for D, j in x["joint"].items():
            print(f"    D={D/1000:g}: {dtxt(j)}")
        print(f"  ladder rungs moved vs r={R_REF:g}: " + ("; ".join(x["rungs_changed"]) if x["rungs_changed"] else "none"))
        print(f"  false admissions over the admission-outcome contracts: {x['false_admissions']}"
              + "".join(f"  [unscored admission: {c['contract']} at {c['pick']}]" for c in x["contracts"] if c["unscored"]))

    print("\n=== summary (tab:rsens layout) ===")
    print(f"  {'r':<7}{'picks':>6}  {'H*(D=28)':<18}{'joint accept D':<20}{'false adm.':>10}")
    for x in res:
        h = x["hstar"][top]
        acc = "acc." if x["horizon"][2000]["pick"] is not None else "decl."
        D1 = x["joint_first_D_us"]
        o = x["joint"][D1]["outcome"]
        jd = f"{D1/1000:g} ms ({o['mean']:.2f}%)" if o else f"{D1/1000:g} ms (no runs)"
        print(f"  {x['r']:<7g}{x['picks_changed']:>6}  {f'{h:,.0f} ({acc})':<18}{jd:<20}{x['false_admissions']:>10}")


if __name__ == "__main__":
    main()

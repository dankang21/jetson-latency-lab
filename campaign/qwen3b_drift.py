#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Per-token decode drift of Qwen2.5-3B Q4_K_M from its solo 2,000-token runs at EMC
2133 MHz, computed with the drift code of the Qwen2.5-1.5B runs and printed side by
side with Qwen2.5-1.5B Q4_K_M.

Read-only over results/ (python3 + numpy).  Run from anywhere:
    python3 campaign/qwen3b_drift.py [--B 10000] [--seed 42] [--json]

Data: results/boundary_stats/q3_g<f>_r<k>.csv (Qwen2.5-3B, five runs at each of GPU
1020/1122/1173 MHz) and q15_g<f>_r<k>.csv (Qwen2.5-1.5B, five runs at 918/1020/1122,
eight at 1173), column dur_ns per token, 2,000 tokens per run, solo, EMC locked at
2133 MHz by the producing script (campaign/boundary_stats.sh; q15 r6..r8 at 1173 by
campaign/tmc_reps.sh).

Drift definition (unchanged):
  campaign/analyze_joint.py lines 47-53 (dur_cells) and 58-61:
      t_short = median(concat r[:300])
      drift   = median(concat r[1500:2000]) / median(concat r[:500]) - 1
      gamma   = drift / 1500
  campaign/fresh_pick.py lines 24-28: the same pooled-median formula over
      sorted(glob)[:5] (the first five runs) -> gamma_sub, the coefficient the
      admission rule prices the horizon with.
  r = 1.035 (analyze_joint.py line 13 / fresh_pick.py line 9).
The two functions below are copied verbatim from those lines (both files run their
analysis at import, so they cannot be imported); the only change is the glob (q3_*
as well as q15_*) and that every clock is reported (the analyze_joint.py loop at
line 57 keeps only clocks that also have 4,000-token runs).

Printed per (model, clock):
  * pooled drift and gamma over all runs (analyze_joint.py), gamma over the first
    five runs (fresh_pick.py), per-run drifts, a run-level bootstrap interval of
    the pooled gamma (whole runs resampled; descriptive only);
  * margin checks of the admission rule's horizon term (1 + gamma H) r at H = 2000:
    the cell's own gamma priced at H, and gamma_sub of the 1.5B (fresh_pick.py,
    q15_g1173 r1..r5) transferred unchanged: err = med_tail / (med_head (1 + gamma H)) - 1
    (pooled and per run), the centre-to-centre linear variant med_head (1 + gamma 1500),
    and the horizon at which the static margin r - 1 is used up, (r - 1) / gamma;
  * 3B against 1.5B at the same clocks: exact one-sided rank test of the per-run
    drifts (all assignments enumerated) and the gamma ratios;
  * per-token level against the EMC-2133 and EMC-3199 reference runs of the same
    model (results/qwen3b_deploy, results/qwen_deploy: qsweep_emc<e>_gpu<f>.csv and the
    deployment runs q3dep/qdep_g<f>_r<k>.csv at 2133; results/llm_axes q3/q15_e<e>_r<k>.csv
    at 1173) -- the drift runs record no EMC field, so the level is the indirect check;
  * the run inventory (metadata of each JSON, sha256 of each CSV).
--json prints the full record as JSON instead of the tables.
"""
import argparse
import csv
import glob
import hashlib
import itertools
import json
import os
import re

import numpy as np

LAB = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BS = f"{LAB}/results/boundary_stats"
R = 1.035
H = 2000


# ---- verbatim: campaign/analyze_joint.py lines 47-53 -----------------------
def dur_cells(pat):
    out = {}
    for f in sorted(glob.glob(pat)):
        g = int(re.search(r"g(\d+)_r", f).group(1))
        v = np.array([float(x["dur_ns"]) for x in csv.DictReader(open(f))]) / 1e3
        out.setdefault(g, []).append(v)
    return out


# ---- verbatim: campaign/analyze_joint.py lines 58-61 (loop body) -----------
def aj(runs):
    t_short = np.median(np.concatenate([r[:300] for r in runs]))
    drift = (np.median(np.concatenate([r[1500:2000] for r in runs])) /
             np.median(np.concatenate([r[:500] for r in runs])) - 1)
    gamma = drift / 1500
    return float(t_short), float(drift), float(gamma)


# ---- verbatim: campaign/fresh_pick.py lines 24-28 (glob as argument) -------
def fp_gamma(pat):
    solo = []
    for f in sorted(glob.glob(pat))[:5]:
        solo.append(np.array([float(x["dur_ns"]) for x in csv.DictReader(open(f))]) / 1e3)
    gam = (np.median(np.concatenate([r[1500:2000] for r in solo])) /
           np.median(np.concatenate([r[:500] for r in solo])) - 1) / 1500
    return float(gam)


def per_run(v):
    mh, mt = float(np.median(v[:500])), float(np.median(v[1500:2000]))
    return dict(med_head_us=mh, med_tail_us=mt, drift=mt / mh - 1, gamma=(mt / mh - 1) / 1500)


def exact_mwu_less(a, b):
    """exact one-sided P(U_a <= observed) under exchangeability: is a stochastically smaller than b?"""
    allv = np.concatenate([a, b]); n = len(a)
    def u(x, y): return sum((xi > yj) + 0.5 * (xi == yj) for xi in x for yj in y)
    obs = u(a, b); cnt = tot = 0
    for idx in itertools.combinations(range(len(allv)), n):
        m = np.zeros(len(allv), bool); m[list(idx)] = True
        tot += 1; cnt += u(allv[m], allv[~m]) <= obs + 1e-12
    return float(obs), cnt / tot, tot


def boot_gamma(runs, B=10000, seed=42):
    """run-level bootstrap of the pooled gamma (resample whole runs); descriptive only."""
    rng = np.random.default_rng(seed); k = len(runs); g = np.empty(B)
    for b in range(B):
        g[b] = aj([runs[i] for i in rng.integers(0, k, k)])[2]
    return float(np.quantile(g, 0.025)), float(np.quantile(g, 0.975))


def med_file(path, n=None):
    v = np.array([float(x["dur_ns"]) for x in csv.DictReader(open(path))]) / 1e3
    return v if n is None else v[:n]


def analyse(B, seed):
    res = dict(definition="analyze_joint.py 58-61 / fresh_pick.py 24-28", r=R, H=H, boot_B=B, boot_seed=seed)

    # ---------------- run inventory -------------------------------------------
    inv = []
    for f in sorted(glob.glob(f"{BS}/q3_g*_r*.csv")) + sorted(glob.glob(f"{BS}/q15_g*_r*.csv")):
        js = f[:-4] + ".json"; meta = json.load(open(js))["meta"]
        rows = sum(1 for _ in open(f)) - 1
        m = re.search(r"/(q3|q15)_g(\d+)_r(\d+)\.csv$", f)
        rep = int(m.group(3)); model = m.group(1)
        script = ("campaign/tmc_reps.sh (EMC lock 2133)" if model == "q15" and rep >= 6
                  else "campaign/boundary_stats.sh (EMC lock 2133)")
        inv.append(dict(file=os.path.basename(f), model=os.path.basename(meta["model"]), gpu=int(m.group(2)),
                        rep=rep, rows=rows, meta_tokens=meta["tokens"], kv_range=meta["kv_range"],
                        warmup=meta["warmup"], cpu=meta["cpu"], prio=meta["prio"],
                        emc_in_meta=("emc" in json.dumps(meta).lower()),
                        csv_sha256_12=hashlib.sha256(open(f, "rb").read()).hexdigest()[:12],
                        producer=script, emc_mhz_by_script=2133))
    res["inventory"] = inv

    # ---------------- drift with the unchanged code ---------------------------
    gsub5 = fp_gamma(f"{BS}/q15_g1173_r*.csv")           # gamma_sub (fresh_pick.py)
    res["gamma_sub_fresh_pick_r1_5"] = gsub5
    cells = {}
    for tag, pat in (("3B", f"{BS}/q3_g*_r*.csv"), ("1.5B", f"{BS}/q15_g*_r*.csv")):
        dc = dur_cells(pat)
        for g in sorted(dc):
            runs = dc[g]
            assert all(len(r) == 2000 for r in runs)
            t_short, drift, gamma = aj(runs)                                   # analyze_joint (all runs)
            gfp = fp_gamma(pat.replace("g*_", f"g{g}_"))                        # fresh_pick (first 5)
            pr = [per_run(r) for r in runs]
            d = np.array([p["drift"] for p in pr])
            # pooled windows for the margin checks
            mh = float(np.median(np.concatenate([r[:500] for r in runs])))
            mt = float(np.median(np.concatenate([r[1500:2000] for r in runs])))
            lo, hi = boot_gamma(runs, B, seed)
            c = dict(model=tag, gpu=g, n_runs=len(runs), t_short_us=t_short, med_head_us=mh, med_tail_us=mt,
                     drift_all=drift, gamma_all=gamma, gamma_first5_fresh_pick=gfp,
                     drift_first5=gfp * 1500,
                     per_run_drift=[float(x) for x in d], per_run_gamma=[float(x / 1500) for x in d],
                     per_run_drift_median=float(np.median(d)), per_run_drift_min=float(d.min()),
                     per_run_drift_max=float(d.max()),
                     boot95_gamma_runlevel=[lo, hi],
                     # (a) the rule's drift term with the cell's own gamma, priced at H = 2000
                     own_term_gammaH=gamma * H,
                     own_err_rule=mt / (mh * (1 + gamma * H)) - 1,
                     # (b) horizon-blind: drift vs the static r - 1 margin
                     drift_inside_static_r=bool(drift <= R - 1),
                     gammaH_inside_static_r=bool(gamma * H <= R - 1),
                     # (c) the 1.5B gamma_sub transferred unchanged
                     sub_pred_rule_us=mh * (1 + gsub5 * H), sub_err_rule=mt / (mh * (1 + gsub5 * H)) - 1,
                     sub_pred_lin_us=mh * (1 + gsub5 * 1500), sub_err_lin=mt / (mh * (1 + gsub5 * 1500)) - 1,
                     per_run_sub_err_rule=[float(p["med_tail_us"] / (p["med_head_us"] * (1 + gsub5 * H)) - 1) for p in pr],
                     # (d) one-sided coverage: tail median <= head median * (1 + gamma_sub H) * r
                     covered_by_sub_term_times_r=bool(mt <= mh * (1 + gsub5 * H) * R),
                     H_to_exhaust_static_r=(R - 1) / gamma)
            c["sub_within_r_pooled"] = bool(abs(c["sub_err_rule"]) <= R - 1)
            c["sub_within_r_all_runs"] = all(abs(e) <= R - 1 for e in c["per_run_sub_err_rule"])
            cells[f"{tag}@{g}"] = c
    res["cells"] = cells
    # thresholds
    res["gamma_max_static_r_endpriced"] = (R - 1) / H                  # gamma*H <= r-1
    res["gamma_max_static_r_windowdrift"] = (R - 1) / 1500             # drift <= r-1
    res["gamma_max_sub_term_times_r"] = ((1 + gsub5 * H) * R - 1) / H  # 1+gH <= (1+g_sub H)r

    # ---------------- the 1.5B figures of the deployed cells -------------------
    c1173 = cells["1.5B@1173"]
    res["repro"] = dict(
        gamma_r1_8=c1173["gamma_all"], drift_r1_8=c1173["drift_all"],
        gamma_r1_5=c1173["gamma_first5_fresh_pick"],
        drift_1122=cells["1.5B@1122"]["drift_all"],
        drift_range_1p5B=[min(cells[k]["drift_all"] for k in cells if k.startswith("1.5B")),
                          max(cells[k]["drift_all"] for k in cells if k.startswith("1.5B"))])

    # ---------------- 3B vs 1.5B per-run drifts, exact rank test ---------------
    cmp = {}
    for g in (1020, 1122, 1173):
        a = np.array(cells[f"3B@{g}"]["per_run_drift"]); b = np.array(cells[f"1.5B@{g}"]["per_run_drift"])
        U, p, tot = exact_mwu_less(a, b)
        cmp[g] = dict(U_3B_gt_1p5B=U, p_one_sided_exact=p, n_assign=tot,
                      ratio_gamma_all=cells[f"3B@{g}"]["gamma_all"] / cells[f"1.5B@{g}"]["gamma_all"],
                      ratio_gamma_first5=cells[f"3B@{g}"]["gamma_first5_fresh_pick"] / cells[f"1.5B@{g}"]["gamma_first5_fresh_pick"],
                      overlap=bool(a.max() >= b.min()))
    res["compare_3B_vs_1p5B"] = cmp

    # ---------------- indirect EMC check: level vs reference runs --------------
    lv = []
    refs = {"3B": f"{LAB}/results/qwen3b_deploy", "1.5B": f"{LAB}/results/qwen_deploy"}
    for tag, pre, dpre in (("3B", "q3", "q3dep"), ("1.5B", "q15", "qdep")):
        for g in sorted({c["gpu"] for c in cells.values() if c["model"] == tag}):
            runs = [med_file(f) for f in sorted(glob.glob(f"{BS}/{pre}_g{g}_r*.csv"))]
            b300 = float(np.median(np.concatenate([r[:300] for r in runs])))
            b150 = float(np.median(np.concatenate([r[:150] for r in runs])))
            row = dict(model=tag, gpu=g, bs_med_tok0_299_us=b300, bs_med_tok0_149_us=b150)
            dep = sorted(glob.glob(f"{refs[tag]}/{dpre}_g{g}_r[0-9].csv"))
            if dep:
                d300 = float(np.median(np.concatenate([med_file(f) for f in dep])))
                row.update(dep_files=[os.path.basename(f) for f in dep], dep2133_med_us=d300, ratio_bs_dep2133=b300 / d300)
            for emc in (2133, 3199):
                sw = f"{refs[tag]}/qsweep_emc{emc}_gpu{g}.csv"
                if os.path.isfile(sw):
                    s = float(np.median(med_file(sw)))
                    row[f"sweep{emc}_med_us"] = s; row[f"ratio_bs150_sweep{emc}"] = b150 / s
            if g == 1173:
                for emc in (2133, 3199):
                    ax = sorted(glob.glob(f"{LAB}/results/llm_axes/{pre}_e{emc}_r[0-9].csv"))
                    if ax:
                        a300 = float(np.median(np.concatenate([med_file(f) for f in ax])))
                        row[f"llm_axes{emc}_med_us"] = a300; row[f"ratio_bs_llm_axes{emc}"] = b300 / a300
                        row[f"llm_axes{emc}_files"] = [os.path.basename(f) for f in ax]
            lv.append(row)
    res["level_check"] = lv
    return res


def ratios(row, emc):
    """the level ratios of one level-check row against the EMC-<emc> references"""
    keys = (f"ratio_bs_dep{emc}", f"ratio_bs150_sweep{emc}", f"ratio_bs_llm_axes{emc}")
    return [row[k] for k in keys if k in row]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--B", type=int, default=10000, help="bootstrap resamples of the run-level gamma interval")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--json", action="store_true", help="print the full record as JSON instead of the tables")
    a = ap.parse_args()
    res = analyse(a.B, a.seed)
    if a.json:
        print(json.dumps(res, indent=1))
        return
    gsub5, cells, cmp, lv, inv = res["gamma_sub_fresh_pick_r1_5"], res["cells"], res["compare_3B_vs_1p5B"], res["level_check"], res["inventory"]
    c1173 = cells["1.5B@1173"]
    print(f"gamma_sub (fresh_pick.py, q15_g1173 r1..r5) = {gsub5:.4e}  (drift {gsub5*1500*100:+.3f} %)")
    print(f"1.5B@1173 r1..r8 (analyze_joint.py): drift {c1173['drift_all']*100:+.3f} %, gamma {c1173['gamma_all']:.4e}")
    print("\n| model | GPU | n | drift (pooled, all runs) % | gamma all e-5 | gamma first5 e-5 | per-run drift median [min-max] % | boot95 gamma e-5 | gamma*2000 % | t_short ms |")
    for k, c in cells.items():
        print(f"| {c['model']} | {c['gpu']} | {c['n_runs']} | {c['drift_all']*100:+.3f} | {c['gamma_all']*1e5:.3f} | "
              f"{c['gamma_first5_fresh_pick']*1e5:.3f} | {c['per_run_drift_median']*100:+.2f} [{c['per_run_drift_min']*100:+.2f}..{c['per_run_drift_max']*100:+.2f}] | "
              f"{c['boot95_gamma_runlevel'][0]*1e5:.2f}-{c['boot95_gamma_runlevel'][1]*1e5:.2f} | {c['own_term_gammaH']*100:.2f} | {c['t_short_us']/1e3:.3f} |")
    print("\nper-run drift %:")
    for k, c in cells.items():
        print(f"  {k}: " + " ".join(f"{x*100:+.2f}" for x in c["per_run_drift"]))
    print("\nmargin checks (pooled): own err_rule | drift<=3.5 | g*2000<=3.5 | sub err_rule | sub err_lin | sub within r (pooled/all runs) | covered (1+g_sub H) r | H exhausting r-1")
    for k, c in cells.items():
        print(f"  {k}: {c['own_err_rule']*100:+.2f} | {c['drift_inside_static_r']} | {c['gammaH_inside_static_r']} | "
              f"{c['sub_err_rule']*100:+.2f} [runs {min(c['per_run_sub_err_rule'])*100:+.2f}..{max(c['per_run_sub_err_rule'])*100:+.2f}] | "
              f"{c['sub_err_lin']*100:+.2f} | {c['sub_within_r_pooled']}/{c['sub_within_r_all_runs']} | {c['covered_by_sub_term_times_r']} | {c['H_to_exhaust_static_r']:.0f}")
    print(f"\nthresholds: gamma*2000<=r-1 -> gamma<={res['gamma_max_static_r_endpriced']:.4e}; drift<=r-1 -> gamma<={res['gamma_max_static_r_windowdrift']:.4e}; "
          f"(1+gH)<=(1+g_sub H)r -> gamma<={res['gamma_max_sub_term_times_r']:.4e}")
    print("\n3B vs 1.5B per-run drift (exact one-sided rank test, 3B smaller):")
    for g, v in cmp.items():
        print(f"  g{g}: U={v['U_3B_gt_1p5B']:.1f} p={v['p_one_sided_exact']:.4f} (of {v['n_assign']}); overlap={v['overlap']}; "
              f"gamma ratio 3B/1.5B all={v['ratio_gamma_all']:.3f} first5={v['ratio_gamma_first5']:.3f}")
    print("\nlevel check (indirect EMC evidence):")
    for row in lv:
        print("  " + json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}))
    # one line per model over its clocks: the quantities the decode-horizon discussion quotes
    print("\nsummary over clocks (pooled values):")
    for tag in ("3B", "1.5B"):
        cs = [c for c in cells.values() if c["model"] == tag]
        rows = [r for r in lv if r["model"] == tag]
        r2133 = [x for r in rows for x in ratios(r, 2133)]
        r3199 = [x for r in rows for x in ratios(r, 3199)]
        def join(key, scale, fmt):
            return " / ".join(format(c[key] * scale, fmt) for c in cs)
        line = (f"  {tag:4s} at {'/'.join(str(c['gpu']) for c in cs)} MHz (n = {'/'.join(str(c['n_runs']) for c in cs)}): "
                f"drift {join('drift_all', 100, '+.3f')} %; gamma {join('gamma_all', 1e5, '.3f')} e-5; "
                f"err of the (1+gamma_sub H) term {join('sub_err_rule', 100, '+.3f')} %; "
                f"level / EMC-2133 references {min(r2133):.4f}-{max(r2133):.4f}, / EMC-3199 references {min(r3199):.4f}-{max(r3199):.4f}")
        if tag == "3B":
            line += "; gamma ratio 3B/1.5B " + " / ".join(format(cmp[c["gpu"]]["ratio_gamma_all"], ".3f") for c in cs)
        print(line)
    print("\ninventory:")
    for i in inv:
        print(f"  {i['file']:20s} {i['model']:34s} g{i['gpu']} r{i['rep']} rows={i['rows']} meta.tokens={i['meta_tokens']} kv={i['kv_range']} "
              f"warm={i['warmup']} cpu={i['cpu']} prio={i['prio']} emc_in_meta={i['emc_in_meta']} sha={i['csv_sha256_12']} <- {i['producer']}")


if __name__ == "__main__":
    main()

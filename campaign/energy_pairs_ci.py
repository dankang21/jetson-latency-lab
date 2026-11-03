#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Paired uncertainty of the blind-vs-aware calibrated energy deltas of the June/July 2026 governor
cells (the energy column of the first-version Table II): blocked repeats and interleaved A/B pairs.

Offline, read-only: reads results/{energy_cal,ab_energy,nano_d9} power logs and run JSONs, nothing
else.  Run from anywhere:  python3 campaign/energy_pairs_ci.py [--md]

Per-inference energy follows campaign/analyze_energy_cal.py exactly:
    E/inf [mJ] = mean(VDD_IN power [mW], trimmed 10 % each end) / release rate [Hz]
(periodic deployment duty, so the rail integral per period is the per-inference
energy including the idle gap).  The trim is the same integer slice
p[n//10:-n//10] as the campaign script.

For every cell we form paired differences d_i = E_aware,i - E_blind,i and report
    n, mean, s.d. (ddof=1), 95 % t-interval, 95 % paired-bootstrap percentile
    interval (fixed seed), sign count, exact two-sided sign-test p.
Pairing:
  * ab_energy   : true interleaved A/B pairs p1..p10 (nx_wave2.sh B3), paired by p;
                  an order check re-pairs blind(p) with aware(p+1).
  * energy_cal  : blocked design (all clocks in rep 1, then rep 2), paired by rep.
  * nano_d9     : blocked design (408 x3 then 510 x3), paired by rep index.
The blocked pairings are the only pairing the data admit; with n=2 or n=3 the
t-interval is dominated by the t critical value and the bootstrap is degenerate
(3 or 10 distinct resamples).  Both are printed anyway.
--md also prints the summary as a markdown table.
"""
import csv, json, os, sys, math
import numpy as np
try:
    from scipy import stats as sps
except Exception:  # fall back to a small table of t_{0.975,df}
    sps = None

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
SEED = 20260914          # fixed bootstrap seed
NBOOT = 10000
T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
        8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 19: 2.093}


def tcrit(df):
    if sps is not None:
        return float(sps.t.ppf(0.975, df))
    return T975[df]


def meanpow(path, rail="vdd_in_mw", trim=True):
    """Mean rail power (mW) with the campaign's 10 %/10 % edge trim."""
    p = [float(r[rail]) for r in csv.DictReader(open(path)) if r.get(rail)]
    n = len(p)
    if n < 20:
        raise RuntimeError(f"too few samples in {path}")
    if trim:
        p = p[n // 10:-n // 10]
    return float(np.mean(p)), n


def run_energy(stem, trim=True):
    """(E/inf mJ, mean mW, rate Hz, n_samples, miss_rate) for one run stem."""
    d = json.load(open(stem + ".json"))
    rate = float(d["meta"]["hz"])
    mp, ns = meanpow(stem + ".power.csv", trim=trim)
    return mp / rate, mp, rate, ns, d["deadline_miss_rate"]


def sign_test_p(pos, n):
    """Exact two-sided sign test p-value (ties excluded upstream)."""
    k = min(pos, n - pos)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def paired_stats(aware, blind, seed=SEED):
    d = np.asarray(aware, float) - np.asarray(blind, float)
    n = len(d)
    out = dict(n=n, mean=float(d.mean()), diffs=d.tolist())
    out["pos"] = int((d > 0).sum())
    out["sign_p"] = sign_test_p(out["pos"], n)
    if n >= 2:
        sd = float(d.std(ddof=1)); se = sd / math.sqrt(n); tc = tcrit(n - 1)
        out.update(sd=sd, se=se, tcrit=tc, t_lo=out["mean"] - tc * se,
                   t_hi=out["mean"] + tc * se)
        rng = np.random.RandomState(seed)
        idx = rng.randint(0, n, size=(NBOOT, n))
        boots = d[idx].mean(axis=1)
        out["b_lo"], out["b_hi"] = (float(x) for x in np.percentile(boots, [2.5, 97.5]))
        out["n_distinct_resamples"] = math.comb(2 * n - 1, n)
    else:
        out.update(sd=float("nan"), se=float("nan"), tcrit=float("nan"),
                   t_lo=float("nan"), t_hi=float("nan"), b_lo=float("nan"),
                   b_hi=float("nan"), n_distinct_resamples=1)
    return out


# ---------------------------------------------------------------- cells ----
# (label, SKU, D, blind clock, aware clock, subdir, stem pattern, rep ids, design)
CELLS = [
    ("MNv2 NX D=4.6 ms", "NX", "4.6 ms", 1122, 1173, "energy_cal",
     "gov_mobilenet_g{f}_r{r}", [1, 2], "blocked, paired by rep (energy_cal_gov.sh)"),
    ("MNv2 NX D=4.8 ms", "NX", "4.8 ms", 1020, 1173, "energy_cal",
     "gov_mobilenet_g{f}_r{r}", [1, 2], "blocked, paired by rep (energy_cal_gov.sh)"),
    ("MNv2 NX D=5.0 ms", "NX", "5.0 ms", 918, 1122, "energy_cal",
     "gov_mobilenet_g{f}_r{r}", [1, 2], "blocked, paired by rep (energy_cal_gov.sh)"),
    ("MNv2 NX D=5.0 ms", "NX", "5.0 ms", 918, 1122, "ab_energy",
     "mn_p{r}_g{f}", list(range(1, 11)), "interleaved A/B pairs (nx_wave2.sh B3)"),
    ("MNv2 Nano D=9 ms", "Nano", "9 ms", 408, 510, "nano_d9",
     "g{f}_r{r}", [1, 2, 3], "blocked, paired by rep (408 x3 then 510 x3)"),
    # the ViT-Small cells of the same campaigns
    ("ViT NX D=13.6 ms", "NX", "13.6 ms", 918, 1122, "energy_cal",
     "gov_vit_g{f}_r{r}", [1, 2], "blocked, paired by rep (energy_cal_gov.sh)"),
    ("ViT NX D=14.2 ms", "NX", "14.2 ms", 816, 1020, "energy_cal",
     "gov_vit_g{f}_r{r}", [1, 2], "blocked, paired by rep (energy_cal_gov.sh)"),
    ("ViT NX D=15.0 ms", "NX", "15.0 ms", 816, 918, "energy_cal",
     "gov_vit_g{f}_r{r}", [1, 2], "blocked, paired by rep (energy_cal_gov.sh)"),
    ("ViT NX D=15.0 ms", "NX", "15.0 ms", 816, 918, "ab_energy",
     "vit_p{r}_g{f}", list(range(1, 11)), "interleaved A/B pairs (nx_wave2.sh B3)"),
]


def collect(sub, pat, reps, fb, fa, trim=True):
    rows = []
    for r in reps:
        sb = os.path.join(ROOT, sub, pat.format(f=fb, r=r))
        sa = os.path.join(ROOT, sub, pat.format(f=fa, r=r))
        eb, pb, rb, nb, mb = run_energy(sb, trim)
        ea, pa, ra, na, ma = run_energy(sa, trim)
        assert rb == ra, (sb, sa, rb, ra)
        rows.append(dict(rep=r, E_blind=eb, E_aware=ea, P_blind=pb, P_aware=pa,
                         rate=rb, miss_blind=mb, miss_aware=ma,
                         file_blind=os.path.relpath(sb, ROOT) + ".power.csv",
                         file_aware=os.path.relpath(sa, ROOT) + ".power.csv"))
    return rows


def main():
    results = []
    print(f"# Paired energy deltas (aware - blind), method = analyze_energy_cal.py; "
          f"seed={SEED}, B={NBOOT}, scipy={'yes' if sps else 'no'}")
    for (label, sku, D, fb, fa, sub, pat, reps, design) in CELLS:
        rows = collect(sub, pat, reps, fb, fa, trim=True)
        rows0 = collect(sub, pat, reps, fb, fa, trim=False)
        st = paired_stats([x["E_aware"] for x in rows], [x["E_blind"] for x in rows])
        st0 = paired_stats([x["E_aware"] for x in rows0], [x["E_blind"] for x in rows0])
        Eb = np.array([x["E_blind"] for x in rows]); Ea = np.array([x["E_aware"] for x in rows])
        res = dict(label=label, sku=sku, D=D, fb=fb, fa=fa, sub=sub, design=design,
                   rate=rows[0]["rate"], rows=rows, st=st, st_notrim=st0,
                   Eb_mean=float(Eb.mean()), Eb_sd=float(Eb.std(ddof=1)) if len(Eb) > 1 else float("nan"),
                   Ea_mean=float(Ea.mean()), Ea_sd=float(Ea.std(ddof=1)) if len(Ea) > 1 else float("nan"),
                   miss_max=max(max(x["miss_blind"], x["miss_aware"]) for x in rows))
        # order-effect check for the interleaved design: pair blind(p) with aware(p+1)
        if sub == "ab_energy":
            sh = paired_stats([x["E_aware"] for x in rows[1:]], [x["E_blind"] for x in rows[:-1]])
            res["shifted"] = sh
            # trend of d with pair index (drift check)
            d = np.array(st["diffs"]); p = np.arange(1, len(d) + 1)
            slope = float(np.polyfit(p, d, 1)[0]); res["slope_per_pair"] = slope
        results.append(res)

        print(f"\n== {label}  [{sub}]  blind {fb} MHz vs aware {fa} MHz, rate {res['rate']:.0f} Hz, "
              f"{design}")
        for x in rows:
            print(f"   rep {x['rep']:>2}: blind {x['E_blind']:.2f} mJ ({x['P_blind']:.0f} mW)  "
                  f"aware {x['E_aware']:.2f} mJ ({x['P_aware']:.0f} mW)  "
                  f"d={x['E_aware']-x['E_blind']:+.2f}  [{x['file_blind']} | {x['file_aware']}]")
        print(f"   E_blind {res['Eb_mean']:.2f}±{res['Eb_sd']:.2f}  E_aware {res['Ea_mean']:.2f}±{res['Ea_sd']:.2f} mJ/inf")
        print(f"   n={st['n']}  mean d={st['mean']:+.2f}  sd={st['sd']:.2f}  se={st['se']:.2f}  "
              f"t95=[{st['t_lo']:+.2f},{st['t_hi']:+.2f}] (t_crit {st['tcrit']:.3f})  "
              f"boot95=[{st['b_lo']:+.2f},{st['b_hi']:+.2f}] ({st['n_distinct_resamples']} distinct resamples)  "
              f"sign {st['pos']}/{st['n']} p={st['sign_p']:.4f}")
        print(f"   no-trim sensitivity: mean d={st0['mean']:+.2f} t95=[{st0['t_lo']:+.2f},{st0['t_hi']:+.2f}]")
        if "shifted" in res:
            sh = res["shifted"]
            print(f"   order check, blind(p) vs aware(p+1): n={sh['n']} mean {sh['mean']:+.2f} "
                  f"t95=[{sh['t_lo']:+.2f},{sh['t_hi']:+.2f}] sign {sh['pos']}/{sh['n']}; "
                  f"drift slope {res['slope_per_pair']:+.3f} mJ/pair")

    # ------------------------------------------------------------- markdown --
    if "--md" in sys.argv:
        L = ["", "| Cell | Data (design) | n | E_blind (mJ/inf) | E_aware (mJ/inf) | ΔE mean | s.d. | 95% CI (t) | 95% CI (boot) | sign |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for r in results:
            st = r["st"]
            L.append(f"| {r['label']} ({r['fb']}→{r['fa']} MHz) | `{r['sub']}` ({r['design'].split(' (')[0]}) | {st['n']} | "
                     f"{r['Eb_mean']:.1f} ± {r['Eb_sd']:.1f} | {r['Ea_mean']:.1f} ± {r['Ea_sd']:.1f} | "
                     f"{st['mean']:+.2f} | {st['sd']:.2f} | [{st['t_lo']:+.2f}, {st['t_hi']:+.2f}] | "
                     f"[{st['b_lo']:+.2f}, {st['b_hi']:+.2f}] | {st['pos']}/{st['n']} (p={st['sign_p']:.3f}) |")
        print("\n".join(L))
    return results


if __name__ == "__main__":
    main()

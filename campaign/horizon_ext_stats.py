#!/usr/bin/env python3
"""E8 statistics: decode-horizon coverage under co-tenancy (campaign/horizon_ext.sh).

Reads every launch under results/horizon_ext/<board>/H<H>/t<k>.csv -- the
per-token dur_ns of one fresh H-token Qwen2.5-1.5B Q4_K_M decode against the
ViT-Small 10 Hz co-runner (GPU 1173, EMC 2133; docs/protocols/E8_protocol.md) -- and prints,
per H, markdown tables of the three endpoints the protocol pre-states (token miss
against D, the re-estimated drift coefficient, the rule's late-horizon prediction),
of statistics derived from them, and of a replay of the frozen rule (the last item
below), which is not one of the protocol's endpoints:
  * per launch: token miss % at each D (dur > D, strict; analyze_joint.py
    convention) over all H tokens and over the last 500; the median per-token
    latency of tokens 0-499 ("head") and of the last 500 ("tail"); the
    launch's own drift coefficient
        gamma_k = (med_tail / med_head - 1) / (H - 500),
    the fresh_pick.py lines 27-28 / analyze_joint.py lines 59-61, 98-99
    definition (windows [0,500) and [1500,2000) at H = 2000, divisor 1500 =
    the distance between the window starts = between the window centres
    250 -> 1750) generalised to windows [0,500) and [H-500,H), whose start and
    centre distances are both H - 500; the late-horizon median the frozen
    rule predicts from the launch's own head median,
        pred_rule = med_head * (1 + gamma_sub * H)
    (the rule's (1+gamma H) term as the admission rule evaluates it, which
    prices the end of the horizon, 250 tokens past the tail-window centre),
    its signed error vs med_tail, whether |error| <= r - 1 (r = 1.035), and the
    centre-to-centre linear prediction pred_lin = med_head * (1 + gamma_sub
    (H - 500)) with its error;
  * pooled over launches (medians of the concatenated windows, the way
    fresh_pick.py pools the solo runs) and the pooled miss %;
  * launch-level exceedance at each D: x = launches with miss % > tau (2 %) out
    of N and the Clopper-Pearson upper bound on P(launch miss > tau) (one-sided
    95 %, plus the upper end of the two-sided 95 % CI; launch_ext_stats.cp_upper,
    numpy only);
  * the frozen composed rule replayed at each H (analyze_joint.hxt()):
    D*(H) = t_solo * dil * (1 + gamma_sub H) * r + M with t_solo = median of
    tokens 0-299 of the solo runs, dil = probe median / t_solo and M = p98 - p50
    of results/joint_hxt/probe_co_g1173.csv, ACC/REJ at each D, and the quoted
    co-tenant horizon cap H*_co(D) = ((D - M) / (r t_solo dil) - 1) / gamma_sub.
gamma_sub, the solo drift coefficient of the frozen rule: the fresh_pick.py
value, i.e. the first five results/boundary_stats/q15_g1173_r*.csv (r1..r5;
solo 2,000-token decodes at 1173, no co-runner), windows [0,500) vs
[1500,2000), pooled medians, divisor 1500. analyze_joint.py pools r1..r8 for
the same quantity (that variant gives the article's H* = 2,147 (~2,150) tokens
and D* = 43.79 ms); --solo-n 8 reproduces it and the r1..r8 value is printed
alongside either way.
The 2,000-token cell: on the NX with the default --root, the nine 2,000-token
decodes of results/joint_hxt/q15co_g1173_r<k>.csv (the joint-stress cell of
analyze_joint.hxt(): same co-runner, GPU 1173 MHz, EMC 2133 MHz; main-text
Table VI) are scored as H = 2,000 under the same definitions, rows r<k>;
--no-h2000 leaves them out.
Usage: horizon_ext_stats.py [--board nx|nano] [--root DIR] [--H 1000,3000]
         [--D 28,44] [--tau 2] [--r 1.035] [--solo-n 5] [--no-h2000] [--json]
"""
import argparse, glob, json, os, re, sys, warnings
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from launch_ext_stats import cp_upper   # exact binomial inversion, numpy only

LAB = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOLO_GLOB = os.path.join(LAB, "results", "boundary_stats", "q15_g1173_r*.csv")   # fresh_pick.py:25
PROBE_CSV = os.path.join(LAB, "results", "joint_hxt", "probe_co_g1173.csv")      # analyze_joint.py:84
JOINT_DIR = os.path.join(LAB, "results", "joint_hxt")                            # analyze_joint.hxt()
JOINT_H, JOINT_PATTERN, JOINT_RX = 2000, "q15co_g1173_r*.csv", r"/q15co_g1173_r(\d+)\.csv$"
WIN = 500          # head / tail window, tokens (protocol: "tokens 0-500 vs the last 500")
T_SHORT = 300      # analyze_joint.py:58,97 short-cell level window


def load_us(csv, col):
    """one slm_decode_bench CSV (token,dur_ns) -> per-token latency in microseconds (empty file -> empty array)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")            # numpy's "input contained no data" on a partial launch
        return np.loadtxt(csv, delimiter=",", skiprows=1, usecols=col, ndmin=1) / 1e3


def rel(p):
    """path relative to the repo when inside it, absolute otherwise (--root elsewhere)."""
    r = os.path.relpath(p, LAB)
    return r if not r.startswith("..") else os.path.abspath(p)


def frozen_rule(solo_n):
    """gamma_sub, t_solo, dil, M of the frozen rule from the committed solo runs and the reference probe."""
    files = sorted(glob.glob(SOLO_GLOB))
    if len(files) < solo_n:
        sys.exit(f"need {solo_n} solo runs at {SOLO_GLOB}, found {len(files)}")
    solo = [load_us(f, 1) for f in files[:solo_n]]
    gam = (np.median(np.concatenate([r[1500:2000] for r in solo])) /
           np.median(np.concatenate([r[:500] for r in solo])) - 1) / 1500
    t_solo = float(np.median(np.concatenate([r[:T_SHORT] for r in solo])))
    out = dict(gamma=float(gam), solo_files=[rel(f) for f in files[:solo_n]],
               t_solo_us=t_solo, drift_2k_pct=float(gam * 1500 * 100))
    if len(files) >= 8:
        s8 = [load_us(f, 1) for f in files[:8]]
        out["gamma_r1_8"] = float((np.median(np.concatenate([r[1500:2000] for r in s8])) /
                                   np.median(np.concatenate([r[:500] for r in s8])) - 1) / 1500)
    if os.path.isfile(PROBE_CSV):
        p = load_us(PROBE_CSV, 1)
        out.update(probe=rel(PROBE_CSV), M_us=float(np.quantile(p, 0.98) - np.median(p)),
                   dil=float(np.median(p) / t_solo), probe_med_us=float(np.median(p)))
    return out


def launch_stats(v, H, Ds, gam_sub, r):
    n = v.size
    head, tail = v[:WIN], v[max(n - WIN, 0):]
    mh, mt = float(np.median(head)), float(np.median(tail))
    span = H - WIN                     # window-start distance == window-centre distance
    d = dict(n=n, med_head_us=mh, med_tail_us=mt,
             gamma=(mt / mh - 1) / span if span > 0 else float("nan"),
             pred_rule_us=mh * (1 + gam_sub * H), pred_lin_us=mh * (1 + gam_sub * span))
    d["err_rule_pct"] = 100 * (mt / d["pred_rule_us"] - 1)
    d["err_lin_pct"] = 100 * (mt / d["pred_lin_us"] - 1)
    d["within_r"] = abs(d["err_rule_pct"]) <= 100 * (r - 1)
    d["miss"] = {D: dict(misses=int(np.count_nonzero(v > D * 1e3)),
                         miss_pct=100 * float(np.mean(v > D * 1e3)),
                         last_pct=100 * float(np.mean(tail > D * 1e3))) for D in Ds}
    return d


def analyse_H(hdir, H, Ds, sub, tau, r, pattern="t*.csv", rx=r"/t(\d+)\.csv$", label=None):
    """Score every launch CSV of one horizon: hdir/t<k>.csv by default; pattern/rx select other run files
    (k = the run index), label replaces the 't' of the row names."""
    launches, traces, incomplete = [], [], []
    files = [p for p in glob.glob(os.path.join(hdir, pattern)) if re.search(rx, p)]
    for csv in sorted(files, key=lambda p: int(re.search(rx, p).group(1))):
        k = int(re.search(rx, csv).group(1))
        try:
            v = load_us(csv, 1)
        except Exception as e:                       # truncated / malformed file
            incomplete.append((k, f"{e}"))
            continue
        if v.size < 2 * WIN:
            incomplete.append((k, f"{v.size} tokens < {2 * WIN}: head/tail windows overlap"))
            continue
        d = launch_stats(v, H, Ds, sub["gamma"], r)
        d.update(k=k, csv=rel(csv))
        if v.size != H:
            d["warn"] = f"{v.size} tokens in CSV, H={H} from the directory name"
        js = csv[:-4] + ".json"
        if os.path.isfile(js):
            try:
                mt = json.load(open(js))["meta"]
                if mt.get("tokens") != H:
                    d["warn"] = f"json meta.tokens={mt.get('tokens')} != H={H}"
                d["prio"] = mt.get("prio"); d["prompt_tokens"] = mt.get("prompt_tokens")
            except Exception as e:
                d["warn"] = f"unreadable json: {e}"
        else:
            d["warn"] = "no json (CSV complete; summary write failed?)"
        launches.append(d); traces.append(v)
    out = dict(H=H, dir=rel(hdir), N=len(launches), launches=launches,
               incomplete=[dict(k=k, why=w) for k, w in incomplete])
    if label:
        out["label"] = label
    if not launches:
        return out
    heads = np.concatenate([v[:WIN] for v in traces]); tails = np.concatenate([v[-WIN:] for v in traces])
    mh, mt = float(np.median(heads)), float(np.median(tails))
    pooled = dict(n=int(sum(v.size for v in traces)), med_head_us=mh, med_tail_us=mt,
                  gamma=(mt / mh - 1) / (H - WIN), pred_rule_us=mh * (1 + sub["gamma"] * H),
                  pred_lin_us=mh * (1 + sub["gamma"] * (H - WIN)))
    pooled["err_rule_pct"] = 100 * (mt / pooled["pred_rule_us"] - 1)
    pooled["err_lin_pct"] = 100 * (mt / pooled["pred_lin_us"] - 1)
    pooled["within_r"] = abs(pooled["err_rule_pct"]) <= 100 * (r - 1)
    pooled["miss"] = {}
    N = len(launches)
    for D in Ds:
        tot = sum(d["miss"][D]["misses"] for d in launches)
        x = sum(1 for d in launches if d["miss"][D]["miss_pct"] > tau)
        pooled["miss"][D] = dict(misses=tot, miss_pct=100 * tot / pooled["n"],
                                 last_pct=100 * float(np.mean(tails > D * 1e3)),
                                 x_exceed=x, cp_upper_95=cp_upper(x, N, 0.05), cp_upper_975=cp_upper(x, N, 0.025))
    out["pooled"] = pooled
    # frozen composed rule at this H (analyze_joint.hxt():100-105)
    if "M_us" in sub:
        lvl = sub["t_solo_us"] * sub["dil"]
        dstar = lvl * (1 + sub["gamma"] * H) * r + sub["M_us"]
        out["rule"] = dict(Dstar_ms=dstar / 1e3, accept={D: bool(dstar <= D * 1e3) for D in Ds},
                           Hstar_co={D: ((D * 1e3 - sub["M_us"]) / (r * lvl) - 1) / sub["gamma"] for D in Ds})
    return out


def fmt_pct(x):
    return f"{x:.2f}"


def fmt_hstar(h):
    """quoted co-tenant horizon cap; negative = the deadline is infeasible under co even at H = 0."""
    return f"{h:.0f}" if h >= 0 else "<0 (infeasible at any H)"


def rule_str(res, D):
    ru = res["rule"]
    return f"{'ACC' if ru['accept'][D] else 'REJ'} (D*={ru['Dstar_ms']:.2f} ms, H*_co={fmt_hstar(ru['Hstar_co'][D])})"


def print_H(res, Ds, sub, tau, r):
    H, lab = res["H"], res.get("label", "t")
    print(f"\n### H = {H:,} tokens  ({res['dir']}; N = {res['N']} launches)\n")
    for inc in res["incomplete"]:
        print(f"- skipped {lab}{inc['k']}: {inc['why']}")
    if not res["launches"]:
        print("- no complete launches\n")
        return
    if "rule" in res:
        print(f"Frozen rule at H={H}: " + "; ".join(f"D={D:g}: {rule_str(res, D)}" for D in Ds) + "\n")
    cols = ["k", "n"]
    for D in Ds:
        cols += [f"miss%@{D:g}", f"last500@{D:g}"]
    cols += ["med head ms", "med tail ms", "gamma_k e-5", "pred_rule ms", "err_rule %", "pred_lin ms", "err_lin %", f"abs err <= {100*(r-1):g}%"]
    print("| " + " | ".join(cols) + " |")
    print("|" + "|".join("---:" for _ in cols) + "|")

    def row(label, d):
        cells = [label, str(d["n"])]
        for D in Ds:
            cells += [fmt_pct(d["miss"][D]["miss_pct"]), fmt_pct(d["miss"][D]["last_pct"])]
        cells += [f"{d['med_head_us']/1e3:.3f}", f"{d['med_tail_us']/1e3:.3f}", f"{d['gamma']*1e5:.3f}",
                  f"{d['pred_rule_us']/1e3:.3f}", f"{d['err_rule_pct']:+.2f}", f"{d['pred_lin_us']/1e3:.3f}",
                  f"{d['err_lin_pct']:+.2f}", "yes" if d["within_r"] else "NO"]
        print("| " + " | ".join(cells) + " |")
    for d in res["launches"]:
        row(f"{lab}{d['k']}", d)
    row("pooled", res["pooled"])
    for d in res["launches"]:
        if "warn" in d:
            print(f"- WARN {lab}{d['k']}: {d['warn']}")
    print()
    print(f"| D (ms) | launches with miss > {tau:g} % | CP upper bound, one-sided 95 % | upper end of two-sided 95 % CI | pooled miss % | pooled last-500 miss % |")
    print("|---:|---:|---:|---:|---:|---:|")
    for D in Ds:
        m = res["pooled"]["miss"][D]
        print(f"| {D:g} | {m['x_exceed']} / {res['N']} | {100*m['cp_upper_95']:.1f} % | {100*m['cp_upper_975']:.1f} % | "
              f"{m['miss_pct']:.3f} | {m['last_pct']:.2f} |")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--board", choices=["nx", "nano"], default="nx")
    ap.add_argument("--root", help="directory holding H<H>/t<k>.csv (default results/horizon_ext/<board>)")
    ap.add_argument("--H", action="append", help="only these horizons, comma-separated (default: every H<H> dir and, on the NX, the 2,000-token cell)")
    ap.add_argument("--D", action="append", help="deadlines in ms, comma-separated (default 28,44)")
    ap.add_argument("--tau", type=float, default=2.0, help="per-launch exceedance threshold, %% (default 2)")
    ap.add_argument("--r", type=float, default=1.035, help="solo safety multiplier (default 1.035)")
    ap.add_argument("--solo-n", type=int, default=5, help="solo runs pooled for gamma_sub / t_solo (5 = fresh_pick.py, 8 = analyze_joint.hxt)")
    ap.add_argument("--no-h2000", action="store_true",
                    help="leave out the 2,000-token cell of results/joint_hxt (scored by default on the NX with the default --root)")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    Ds = [float(x) for s in (a.D or ["28,44"]) for x in s.split(",") if x.strip()]
    Ds = [int(D) if D == int(D) else D for D in Ds]
    Hs = {int(x) for s in (a.H or []) for x in s.split(",") if x.strip()} or None
    root = a.root or os.path.join(LAB, "results", "horizon_ext", a.board)
    hdirs = sorted(((int(re.search(r"H(\d+)$", d).group(1)), d) for d in glob.glob(os.path.join(root, "H*"))
                    if re.search(r"H(\d+)$", d) and os.path.isdir(d)))
    hdirs = [(H, d) for H, d in hdirs if Hs is None or H in Hs]
    cells = [(H, d, {}) for H, d in hdirs]
    if (a.board == "nx" and not a.root and not a.no_h2000 and (Hs is None or JOINT_H in Hs)
            and JOINT_H not in {H for H, _ in hdirs} and glob.glob(os.path.join(JOINT_DIR, JOINT_PATTERN))):
        cells = sorted(cells + [(JOINT_H, JOINT_DIR, dict(pattern=JOINT_PATTERN, rx=JOINT_RX, label="r"))],
                       key=lambda c: c[0])
    if not cells:
        sys.exit(f"no H<H> directories under {root} (run campaign/horizon_ext.sh on the NX first)")
    sub = frozen_rule(a.solo_n)
    results = [analyse_H(d, H, Ds, sub, a.tau, a.r, **kw) for H, d, kw in cells]
    out = dict(board=a.board, root=rel(root), D_ms=Ds, tau_pct=a.tau, r=a.r, frozen_rule=sub, H=results)
    if a.json:
        print(json.dumps(out, indent=1, default=str))
        return

    print(f"## E8 horizon coverage under co-tenancy -- board {a.board}, {rel(root)}")
    print(f"\nFrozen-rule gamma_sub = {sub['gamma']:.4e} per token (drift {sub['drift_2k_pct']:+.2f} % over 1,500 tokens) from "
          f"{len(sub['solo_files'])} solo runs {sub['solo_files'][0]} .. {sub['solo_files'][-1]} "
          f"(fresh_pick.py windows [0,500) vs [1500,2000), pooled medians, divisor 1500)"
          + (f"; analyze_joint.hxt() r1..r8 variant {sub['gamma_r1_8']:.4e}" if "gamma_r1_8" in sub else "") + ".")
    if "M_us" in sub:
        print(f"Frozen-rule level t_solo = {sub['t_solo_us']/1e3:.3f} ms (tokens 0-{T_SHORT-1}), probe median "
              f"{sub['probe_med_us']/1e3:.3f} ms -> dil = {sub['dil']:.4f}, M = p98 - p50 = {sub['M_us']/1e3:.2f} ms "
              f"({sub['probe']}); r = {a.r}, tau = {a.tau:g} %.")
    print(f"\nColumns: miss % over all tokens and over the last 500 at each D (dur > D); medians of tokens 0-{WIN-1} "
          f"and of the last {WIN}; gamma_k = (med_tail/med_head - 1)/(H - {WIN}); pred_rule = med_head (1 + gamma_sub H); "
          f"pred_lin = med_head (1 + gamma_sub (H - {WIN})); err = med_tail / pred - 1.")
    for res in results:
        print_H(res, Ds, sub, a.tau, a.r)
    print("\n### Summary\n")
    print("| H | D (ms) | N | pooled miss % | last-500 miss % | x (miss > tau) | CP95 UB | frozen rule | gamma_pool e-5 | gamma_pool / gamma_sub | err_rule % (pooled) | within r |")
    print("|---:|---:|---:|---:|---:|---:|---:|:---|---:|---:|---:|:---|")
    for res in results:
        if not res["launches"]:
            print(f"| {res['H']:,} | - | 0 | - | - | - | - | - | - | - | - | - |")
            continue
        p = res["pooled"]
        for D in Ds:
            m = p["miss"][D]
            rule = rule_str(res, D) if "rule" in res else "-"
            print(f"| {res['H']:,} | {D:g} | {res['N']} | {m['miss_pct']:.3f} | {m['last_pct']:.2f} | {m['x_exceed']} / {res['N']} | "
                  f"{100*m['cp_upper_95']:.1f} % | {rule} | {p['gamma']*1e5:.3f} | {p['gamma']/sub['gamma']:.2f} | "
                  f"{p['err_rule_pct']:+.2f} | {'yes' if p['within_r'] else 'NO'} |")


if __name__ == "__main__":
    main()

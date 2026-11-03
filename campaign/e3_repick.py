#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Rate-matched re-pick of the recorded MobileNetV2 admission probes (post hoc, offline).

The MobileNetV2 admission experiments were frozen with endpoint cells profiled at a
release rate other than the primary's 33 Hz: on the NX the 8 Hz June cells
(results/nx_validation/nx_est_emc2133_gpu{306,1173}_mobilenet.json; round-1 fresh
probes, held-out banded round, joint-governor ladder, E3 conditions C0 and N1-N4), on
the Nano the 50 Hz cells (results/nano_grid/mobilenet/est_emc2133_gpu{306,1020}.json;
E3 conditions A1-A4).  govrep_rev2.sh MODE=refit re-profiled the same two endpoint
cells at 33 Hz (EMC 2133 and 3199 MHz locked, three 1000-cycle repeats,
results/refit33/<board>/emc<E>_g<f>_r<k>.json); a rate-matched cell is the median of
the per-repeat response_us p50.

Every recorded probe is re-picked by the deployed rule, admit_pick.evaluate, with only
its cells replaced: D, h, r, the grid and f_probe stay as they were run.  No outcome is
simulated: a re-pick is scored only by a 1000-cycle contract that ran at its clock.

Per board (--board nx|nano, default both):
  cells     both cell sets, T_hat(f) = k/f + b under each, the slack gain
            ds(f) = r (T_hat_orig(f) - T_hat_33(f)) against the band's width h f_probe/f
  E3        every MobileNetV2 condition under results/e3_<board>/: registered picks and
            endpoints from e3_stats.analyse (which cross-checks t<k>.pick.json), the
            re-pick of each probe, and per condition
              moves     banded picks that move down/stay/up (and by how many bins)
              P1        33 Hz banded picks below the registered f_oracle (one-sided 95 % CP UB)
              P2 subst. each 33 Hz banded pick scored on the same trial's contract at its
                        clock: the banded contract when the pick stays, the point contract
                        (run second in that trial, same tenant launch) when it moves; the
                        condition's reference contract only where no same-trial contract
                        ran there.  Run-level bootstrap UB as in e3_stats.py.
            A fresh re-run with the 33 Hz cells, results/e3_<board>_r33/<cond>r33, is
            listed beside (its e3_stats endpoints) when it exists.
NX only, calibration condition (MobileNetV2 33 Hz vs ViT-S 10 Hz tenant, EMC 2133 MHz):
  ladder    joint-governor rungs at D = 9 and 12 ms on the reference probe
            (results/joint_gov/probe_co_g1173.csv), each pick scored on the deployed runs
            results/joint_gov/dep_co_g<f>_r<k>.csv; agreement of the clock-scaled rung's
            feasibility with the measured pooled miss over the 14 deployed (clock, D) cells
  held-out  banded re-pick of results/banded_loop/probe_t<k>.csv (committed: dep_t<k>_g<f>)
  round 1   point re-pick (the deployed rule, h = 0) of results/fresh_loop/vprobe_t<k>.csv
            (committed: decisions.csv)
  runs      every 1000-cycle run on record under that condition, re-scored at D = 12 ms,
            by clock and source
  band      banded picks of the nineteen collected probes (reference, round 1, held-out)
            at h = 0 ... 2.5 sigma (sigma = 200 us, half the band) and the interval of h
            that keeps all nineteen at 918 MHz

Percentages print with Python's format, except the columns marked 'half-up': those
round half-up from the exact ratio of cycle counts (the binary representation error is
removed by rounding to 9 decimals first).
numpy only; read-only on results/, prints to stdout (nothing is written).

Usage: e3_repick.py [--board nx|nano] [--tau 0.02] [--B 10000] [--seed 20260915] [--json]
"""
import argparse, csv, glob, json, os, re, sys
from decimal import Decimal, ROUND_HALF_UP
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LAB = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from admit_pick import load_probe, load_cells, fit_2cell, evaluate, CELLS, GRID, FPROBE, H_US, R_MULT   # noqa: E402
from e3_stats import analyse, bootstrap_pooled, read_condition, cond_key, NAMES                       # noqa: E402
from launch_ext_stats import cp_upper                                                                   # noqa: E402
import joint_gov_policy as jgp                                                                          # noqa: E402

RES = os.path.join(LAB, "results")
INF = "INFEASIBLE"
D_CAL = 12000.0                      # calibration-condition contract deadline, us
D_LADDER = (9000.0, 12000.0)         # joint-governor deadlines, us
SIGMAS = (0.0, 1.0, 1.5, 2.0, 2.5)   # band widths in units of sigma = H_US / 2
assert jgp.GRID == GRID["nx"] and jgp.R == R_MULT


# ----------------------------------------------------------------------------- helpers
def pct(x, nd=2):
    return "-" if x is None else f"{100 * x:.{nd}f}"


def pct_hu(x, nd=2):
    """100*x rounded half-up (x is a ratio of cycle counts; 9-decimal pre-rounding removes binary noise)."""
    if x is None:
        return "-"
    return str(Decimal(repr(round(100 * x, 9))).quantize(Decimal(1).scaleb(-nd), rounding=ROUND_HALF_UP))


def ff(f):
    return "INF" if f == INF else ("-" if f is None else str(f))


def rank(f, grid):
    """Grid index; INFEASIBLE (declined) ranks above the top clock."""
    return len(grid) if f == INF else grid.index(f)


def move(a, b, grid):
    d = rank(b, grid) - rank(a, grid)
    return "stay" if d == 0 else ("down" if d < 0 else "up")


def dist(picks, grid):
    out = {}
    for f in sorted(picks, key=lambda x: rank(x, grid)):
        out[ff(f)] = out.get(ff(f), 0) + 1
    return out


def fdist(d):
    return ", ".join(f"{k}: {v}" for k, v in d.items())


def moves(pairs, grid):
    return {m: sum(1 for a, b in pairs if move(a, b, grid) == m) for m in ("down", "stay", "up")}


def fmoves(m):
    return f"{m['down']}/{m['stay']}/{m['up']}"


def tnum(path, rx):
    return int(re.search(rx, path).group(1))


def numbered(pattern, rx):
    """Files matching pattern, ordered by the integer rx captures."""
    return sorted(glob.glob(pattern), key=lambda p: tnum(p, rx))


def label_clock(js):
    """GPU clock of a harness run from its JSON meta label (..._g<f>), None if absent."""
    try:
        m = re.search(r"_g(\d+)$", json.load(open(js))["meta"].get("label", ""))
        return int(m.group(1)) if m else None
    except Exception:                            # noqa: BLE001
        return None


def scored(path, D_us):
    v = load_probe(path)
    m = int(np.count_nonzero(v > D_us))
    return dict(n=int(v.size), misses=m, miss=m / v.size)


# ----------------------------------------------------------------------------- cells
def cells_reg(board, emc=2133):
    """Registered endpoint cells {f: p50} and their profiling rate (meta.hz)."""
    files = [f.replace("emc2133", f"emc{emc}") for f in CELLS[(board, "mobilenet")]]
    return load_cells(files), {float(json.load(open(f))["meta"]["hz"]) for f in files}, files


def cells_33(board, emc, gpus):
    """33 Hz re-profile at EMC emc: {f: median per-repeat p50}, {f: [per-repeat p50]}, and
    {hz, iters, dates} of the repeats (harness meta)."""
    cells, reps, info = {}, {}, dict(hz=set(), iters=set(), dates=set())
    for g in gpus:
        js = [json.load(open(f)) for f in sorted(glob.glob(f"{RES}/refit33/{board}/emc{emc}_g{g}_r*.json"))]
        if not js:
            sys.exit(f"no 33 Hz re-profile under results/refit33/{board} for EMC {emc}, GPU {g}")
        reps[g] = [float(j["response_us"]["p50"]) for j in js]
        cells[g] = float(np.median(reps[g]))
        info["hz"] |= {float(j["meta"]["hz"]) for j in js}
        info["iters"] |= {int(j["meta"]["iters"]) for j in js}
        info["dates"] |= {j["meta"]["timestamp"][:10] for j in js}
    return cells, reps, {x: sorted(v) for x, v in info.items()}


def model(c_reg, c33, grid, fprobe, h, R):
    kr, br, _, _ = fit_2cell(c_reg)
    k3, b3, _, _ = fit_2cell(c33)
    rows = []
    for f in grid:
        tr, t3 = kr * 1e3 / f + br, k3 * 1e3 / f + b3
        rows.append(dict(f=f, T_reg_us=tr, T_33_us=t3, dT_us=t3 - tr, ds_us=R * (tr - t3), band_us=h * fprobe / f))
    return dict(fit_reg=[kr, br], fit_33=[k3, b3], rows=rows)


def d_rule(c, grid, fprobe, R):
    """E3 deadline rule: 2.12 T_hat(f_k) r, f_k three grid bins below f_probe; and rounded to 0.5 ms."""
    k, b, _, _ = fit_2cell(c)
    fk = grid[grid.index(fprobe) - 3]
    d = 2.12 * (k * 1e3 / fk + b) * R
    return dict(f_k=fk, D_us=d, D_ms_rounded=round(d / 500.0) * 0.5)


# ----------------------------------------------------------------------------- E3 conditions
def verdict(c):
    """Short form of the e3_stats verdict (protocol section 6 as restated 2026-09-15)."""
    P1, P2 = c["P1"]["x"], c["P2"]
    if P2 is None:
        return "no admissions"
    if not P2["passed"]:
        return "not supported (P1, P2)" if P1 else "not supported (P2)"
    if P1 is None:
        return "pending reference"
    if P1 > 0:
        return "not supported (P1)"
    return "band idle" if c["band"]["n_active"] == 0 else "supported"


def repick_condition(cdir, cid, c33, tau, B, seed):
    """Registered endpoints (e3_stats.analyse) and the re-pick of every completed probe with c33."""
    c = analyse(cdir, cid, tau, B, seed)
    if "error" in c:
        return dict(cond=cid, error=c["error"])
    D_us, grid, fp, h, R = c["D_ms"] * 1000.0, c["grid"], c["fprobe"], c["h_us"], c["R"]
    fo, ref = c["reference"]["f_oracle"], c["reference"]["cells"]
    trials = []
    for t in c["trials"]:
        if not t["complete"] or "Mg_us" not in t:
            continue
        r = evaluate(load_probe(os.path.join(cdir, f"t{t['k']}.probe.csv")), c33, D_us, grid, fp, h, R)
        fb, fpt = r["f_b"], r["f_pt"]
        if fb == INF:
            ev = None
        elif fb == t["f_b"] and t.get("m_b") is not None:
            ev = dict(source="banded", n=t["n_b"], misses=t["misses_b"])
        elif fb == t["f_pt"] and t.get("m_pt") is not None:
            ev = dict(source="point", n=t["n_pt"], misses=t["misses_pt"])
        elif str(fb) in ref:
            ev = dict(source="reference", n=ref[str(fb)]["n"], misses=ref[str(fb)]["misses"])
        else:
            ev = dict(source="none", n=0, misses=0)
        if ev is not None and ev["n"]:
            ev["miss"] = ev["misses"] / ev["n"]
        trials.append(dict(k=t["k"], Mg_us=t["Mg_us"], fb_reg=t["f_b"], fpt_reg=t["f_pt"], fb_33=fb, fpt_33=fpt,
                           move_b=move(t["f_b"], fb, grid), bins_b=rank(fb, grid) - rank(t["f_b"], grid),
                           move_pt=move(t["f_pt"], fpt, grid), m_b=t.get("m_b"), m_pt=t.get("m_pt"), ev33=ev))
    adm = [t for t in trials if t["fb_33"] != INF]
    if isinstance(fo, int):
        P1 = sum(1 for t in adm if t["fb_33"] < fo)
    else:                                        # 'none': every admission is below the frontier
        P1 = len(adm) if fo == "none" else None
    scored_ = [t["ev33"] for t in adm if t["ev33"]["n"]]
    mis, nn = [e["misses"] for e in scored_], [e["n"] for e in scored_]
    P2 = None
    if scored_:
        lo, ub95, ub975 = bootstrap_pooled(mis, nn, B, seed)
        P2 = dict(pooled=sum(mis) / sum(nn), boot_ub95=ub95, misses=sum(mis), cycles=sum(nn),
                  sources={s: sum(1 for e in scored_ if e["source"] == s) for s in ("banded", "point", "reference")},
                  over_tau=sum(1 for e in scored_ if e["miss"] > tau),
                  min=min(e["miss"] for e in scored_), max=max(e["miss"] for e in scored_))
    pairs_b = [(t["fb_reg"], t["fb_33"]) for t in trials]
    skipped = [f"t{t['k']}: {t['note'] or 'incomplete'}" for t in c["trials"] if not t["complete"] or "Mg_us" not in t]
    return dict(cond=cid, dir=os.path.relpath(cdir, LAB), co=c["co"], co_hz=c["co_hz"], D_ms=c["D_ms"], grid=grid,
                f_oracle=fo, n=len(trials), skipped=skipped, unscored=sum(1 for t in adm if not t["ev33"]["n"]),
                moves_b=moves(pairs_b, grid), moves_pt=moves([(t["fpt_reg"], t["fpt_33"]) for t in trials], grid),
                bins_moved=sorted({t["bins_b"] for t in trials if t["bins_b"]}),
                bracket=sum(1 for t in trials if rank(t["fpt_reg"], grid) <= rank(t["fb_33"], grid) <= rank(t["fb_reg"], grid)),
                moved_on_fpt_reg=sum(1 for t in trials if t["fb_33"] != t["fb_reg"] and t["fb_33"] == t["fpt_reg"]),
                reg=dict(P1=c["P1"], P2=None if c["P2"] is None else {x: c["P2"][x] for x in ("pooled", "boot_ub95")},
                         over_tau=None if c["violations"] is None else c["violations"]["x"],
                         dist_b=dist([t["fb_reg"] for t in trials], grid), verdict=verdict(c)),
                r33=dict(P1=dict(x=P1, n=len(adm), cp95=cp_upper(P1, len(adm), 0.05) if (P1 is not None and adm) else None),
                         P2=P2, dist_b=dist([t["fb_33"] for t in trials], grid)),
                trials=trials, n_warnings=len(c["warnings"]))


def rerun_block(board, cid, c33, tau, B, seed):
    """e3_stats endpoints of the fresh re-run with the 33 Hz cells, if results/e3_<board>_r33/<cid>r33 exists."""
    d = os.path.join(RES, f"e3_{board}_r33", f"{cid}r33")
    if not os.path.isdir(d):
        return None
    c = analyse(d, f"{cid}r33", tau, B, seed)
    if "error" in c:
        return dict(cond=f"{cid}r33", error=c["error"])
    same = {int(k): v for k, v in c["cells"].items()} == c33
    V, Bn = c["violations"], c["band"]
    return dict(cond=f"{cid}r33", dir=os.path.relpath(d, LAB), n=c["n_complete"], cells_equal_33=same,
                f_oracle=c["reference"]["f_oracle"],
                ref={f: c["reference"]["cells"][str(f)]["miss"] for f in c["grid"] if str(f) in c["reference"]["cells"]},
                P1=c["P1"], P2=None if c["P2"] is None else {x: c["P2"][x] for x in ("pooled", "boot_ub95")},
                launches=None if V is None else dict(x=V["x"], n=V["n"], min=V["min"], max=V["max"], ks=V["ks"]),
                band=dict(n_active=Bn["n_active"], n_pt=Bn["n_pt"], n_pt_fail=Bn["n_pt_fail"]),
                f_b=[t.get("f_b") for t in c["trials"] if t.get("complete")],
                below=[(t["k"], t["f_b"]) for t in c["trials"] if t.get("false_adm")],
                verdict=verdict(c), n_warnings=len(c["warnings"]))


def e3_board(board, c33, tau, B, seed):
    root = os.path.join(RES, f"e3_{board}")
    out = []
    for cid in sorted(os.listdir(root), key=cond_key):
        cdir = os.path.join(root, cid)
        if not os.path.isfile(os.path.join(cdir, "condition.txt")):
            continue
        cond = read_condition(cdir, [])
        if cond is None or cond["primary"] != "mobilenet":
            continue
        if load_cells(cond["cell_files"]) != load_cells(CELLS[(board, "mobilenet")]):
            out.append(dict(cond=cid, error="condition cells are not admit_pick.CELLS[(board, 'mobilenet')]"))
            continue
        r = repick_condition(cdir, cid, c33, tau, B, seed)
        r["rerun"] = rerun_block(board, cid, c33, tau, B, seed)
        out.append(r)
    return out


# ----------------------------------------------------------------------------- calibration condition (NX)
def calibration(c_reg, c33, c33_3199, tau):
    grid, fp, h, R = GRID["nx"], FPROBE["nx"], H_US, R_MULT
    sets = {"reg": c_reg, "33": c33}
    # ---- joint-governor ladder on the reference probe
    probe = load_probe(f"{RES}/joint_gov/probe_co_g1173.csv")
    M_tail, n_probe = jgp.m_tail()
    M_g = float(probe.max() - np.median(probe))
    dep = {}
    for p in sorted(glob.glob(f"{RES}/joint_gov/dep_co_g*_r*.csv")):
        dep.setdefault(tnum(p, r"_g(\d+)_r"), []).append(load_probe(p))
    blind = {"reg": tuple(float(x) for x in jgp.fit_sweep(3199)), "33": fit_2cell(c33_3199)[:2]}
    aware = {s: fit_2cell(c)[:2] for s, c in sets.items()}

    def jp(kb, D, m=0.0):
        f, _ = jgp.pick(*kb, D, m=m)
        return f if f else INF

    rungs = (("blind (solo)", lambda s, D: jp(blind[s], D)),
             ("EMC-aware (solo)", lambda s, D: jp(aware[s], D)),
             ("+tenancy, flat M_tail", lambda s, D: jp(aware[s], D, M_tail)),
             ("+tenancy, M_tail f_probe/f", lambda s, D: evaluate(None, sets[s], D, grid, fp, 0.0, R, M_tail)["f_pt"]),
             ("+max guard, M_g f_probe/f (point, h = 0)", lambda s, D: evaluate(probe, sets[s], D, grid, fp, h, R)["f_pt"]),
             ("+max guard, banded (h = 400 us)", lambda s, D: evaluate(probe, sets[s], D, grid, fp, h, R)["f_b"]))

    def measured(g, D):
        if g == INF or g not in dep:
            return None
        per = [float(np.mean(v > D)) for v in dep[g]]
        return dict(runs=len(per), pooled=float(np.mean(np.concatenate(dep[g]) > D)), min=min(per), max=max(per),
                    over_tau=sum(x > tau for x in per))
    ladder = []
    for D in D_LADDER:
        for name, fn in rungs:
            a, b = fn("reg", D), fn("33", D)
            ladder.append(dict(D_us=D, rung=name, pick_reg=a, pick_33=b, move=move(a, b, grid),
                               meas_reg=measured(a, D), meas_33=measured(b, D)))
    agree = {}
    for s, c in sets.items():
        rows = []
        for g in sorted(dep):
            for D in D_LADDER:
                pred = evaluate(None, c, D, [g], fp, 0.0, R, M_tail)["per_f"][0]["ok_h0"]
                pooled = float(np.mean(np.concatenate(dep[g]) > D))
                rows.append(dict(f=g, D_us=D, rule_feasible=bool(pred), pooled=pooled, agree=bool(pred) == (pooled <= tau)))
        agree[s] = dict(n=len(rows), agree=sum(r["agree"] for r in rows), disagree=[r for r in rows if not r["agree"]])
    # ---- held-out banded round (deployed: banded) and round-1 fresh probes (deployed: point, h = 0)
    held, fresh = [], []
    rx_b, rx_v = r"probe_t(\d+)", r"vprobe_t(\d+)"
    for p in numbered(f"{RES}/banded_loop/probe_t*.csv", rx_b):
        k = tnum(p, rx_b)
        v = load_probe(p)
        a, b = evaluate(v, c_reg, D_CAL, grid, fp, h, R), evaluate(v, c33, D_CAL, grid, fp, h, R)
        d = glob.glob(f"{RES}/banded_loop/dep_t{k}_g*.csv")
        held.append(dict(k=k, Mg_us=a["Mg_us"], f_reg=a["f_b"], f_33=b["f_b"],
                         committed=tnum(d[0], r"_g(\d+)\.csv") if d else None,
                         dep_miss=scored(d[0], D_CAL)["miss"] if d else None))
    dec = {int(x["trial"]): x for x in csv.DictReader(open(f"{RES}/fresh_loop/decisions.csv")) if x["arm"] == "vision"}
    for p in numbered(f"{RES}/fresh_loop/vprobe_t*.csv", rx_v):
        k = tnum(p, rx_v)
        v = load_probe(p)
        a, b = evaluate(v, c_reg, D_CAL, grid, fp, h, R), evaluate(v, c33, D_CAL, grid, fp, h, R)
        d = glob.glob(f"{RES}/fresh_loop/vdep_t{k}_g*.csv")
        fresh.append(dict(k=k, Mg_us=a["Mg_us"], f_reg=a["f_pt"], f_33=b["f_pt"], fb_reg=a["f_b"], fb_33=b["f_b"],
                          committed=int(dec[k]["pick"]) if k in dec else None, committed_Mg=dec[k]["Mg"] if k in dec else None,
                          dep_clock=tnum(d[0], r"_g(\d+)\.csv") if d else None,
                          dep_miss=scored(d[0], D_CAL)["miss"] if d else None))
    # ---- every 1000-cycle run on record under the calibration condition, re-scored at D = 12 ms
    src = []
    for p in glob.glob(f"{RES}/banded_loop/dep_t*_g*.csv"):
        src.append(("banded_loop", p, tnum(p, r"_g(\d+)\.csv")))
    for p in glob.glob(f"{RES}/fresh_loop/vdep_t*_g*.csv"):
        src.append(("fresh_loop", p, tnum(p, r"_g(\d+)\.csv")))
    for d in ("joint_gov", "tenancy"):
        for p in glob.glob(f"{RES}/{d}/dep_co_g*_r*.csv"):
            if not p.endswith(".power.csv"):
                src.append((d, p, tnum(p, r"_g(\d+)_r")))
    for p in glob.glob(f"{RES}/launch1020/dep_t*.csv"):
        src.append(("launch1020", p, 1020))                                   # launch1020.sh locks 1020 MHz
    for p in glob.glob(f"{RES}/launch_ext/nx/mobilenet_vit10_g1020_D12/t*.csv"):
        if not p.endswith(".power.csv"):
            src.append(("launch_ext", p, 1020))
    for d in ("e3_nx/C0", "e3_nx_r33/C0r33"):
        for p in glob.glob(f"{RES}/{d}/ref_g*.csv"):
            src.append((f"{d} reference", p, tnum(p, r"ref_g(\d+)\.csv")))
        for p in glob.glob(f"{RES}/{d}/t*.contract_*.csv"):
            if os.path.getsize(p) > 0:
                src.append((f"{d} contracts", p, label_clock(p[:-4] + ".json")))
    by_clock = {}
    for s, p, g in src:
        by_clock.setdefault(g, {}).setdefault(s, []).append(scored(p, D_CAL)["miss"])
    runs = {g: {s: dict(runs=len(L), over_tau=sum(x > tau for x in L)) for s, L in sorted(by_clock[g].items())}
            for g in sorted(by_clock)}
    # ---- band width over the nineteen collected probes
    probes = [("ref", f"{RES}/joint_gov/probe_co_g1173.csv")] + \
             [("r1_t%d" % tnum(p, rx_v), p) for p in numbered(f"{RES}/fresh_loop/vprobe_t*.csv", rx_v)] + \
             [("ho_t%d" % tnum(p, rx_b), p) for p in numbered(f"{RES}/banded_loop/probe_t*.csv", rx_b)]
    vs = {lab: load_probe(p) for lab, p in probes}
    mg = {lab: float(v.max() - np.median(v)) for lab, v in vs.items()}
    r1 = [mg[lab] for lab, _ in probes if not lab.startswith("ho_")]          # reference + round-1: the band's calibration set
    sigma = H_US / 2.0
    band = dict(n=len(probes), n_r1=len(r1), sd_r1_pop=float(np.std(r1)), sd_r1_n1=float(np.std(r1, ddof=1)), sigma=sigma,
                widths={}, interval={}, ref_margin_918={})
    for s, c in sets.items():
        band["widths"][s] = {}
        for m in SIGMAS:
            picks = {lab: evaluate(v, c, D_CAL, grid, fp, m * sigma, R)["f_b"] for lab, v in vs.items()}
            band["widths"][s][m] = dict(dist=dist(list(picks.values()), grid),
                                        at={ff(f): [lab for lab in picks if picks[lab] == f] for f in set(picks.values())})
        per = {lab: {r["f"]: r["margin_mg_us"] for r in evaluate(v, c, D_CAL, grid, fp, 0.0, R)["per_f"]} for lab, v in vs.items()}
        lo = max(per[lab][816] for lab in per)     # h above this: no probe admits 816 MHz (or below)
        hi = min(per[lab][918] for lab in per)     # h at most this: every probe admits 918 MHz
        band["interval"][s] = dict(gt_us=lo, le_us=hi, gt_sigma=lo / sigma, le_sigma=hi / sigma)
        band["ref_margin_918"][s] = per["ref"][918]
    band["mg"] = mg
    return dict(M_tail_us=M_tail, n_probe=n_probe, M_g_us=M_g, fits=dict(blind=blind, aware=aware), ladder=ladder,
                agreement=agree, held_out=held, round1=fresh, runs=runs, band=band)


# ----------------------------------------------------------------------------- one board
def board_block(board, tau, B, seed):
    grid, fp = GRID[board], FPROBE[board]
    c_reg, hz_reg, files = cells_reg(board)
    c33, reps, info33 = cells_33(board, 2133, sorted(c_reg))
    out = dict(board=board, grid=grid, fprobe=fp, h_us=H_US, R=R_MULT, cell_files=[os.path.relpath(f, LAB) for f in files],
               cells_reg=c_reg, hz_reg=sorted(hz_reg), cells_33=c33, reps_33=reps, info_33=info33,
               model=model(c_reg, c33, grid, fp, H_US, R_MULT),
               d_rule={"reg": d_rule(c_reg, grid, fp, R_MULT), "33": d_rule(c33, grid, fp, R_MULT)})
    derived = sorted(glob.glob(f"{RES}/refit33/{board}/cells_mobilenet/*.json"))
    out["cells_33_files"] = dict(files=[os.path.relpath(f, LAB) for f in derived],
                                 equal=bool(derived) and load_cells(derived) == c33)
    if board == "nx":
        c_reg3199, _, _ = cells_reg(board, 3199)
        c33_3199, reps3199, _ = cells_33(board, 3199, sorted(c_reg))
        out.update(cells_reg_3199=c_reg3199, cells_33_3199=c33_3199, reps_33_3199=reps3199)
    out["e3"] = e3_board(board, c33, tau, B, seed)
    if board == "nx":
        out["calibration"] = calibration(c_reg, c33, out["cells_33_3199"], tau)
    return out


# ----------------------------------------------------------------------------- markdown
def rate(o):
    """Profiling rate of the original cells, e.g. '8 Hz'."""
    return "/".join(f"{x:g}" for x in o["hz_reg"]) + " Hz"


def md_cells(o):
    i33, lr = o["info_33"], rate(o)
    L = [f"## {o['board']}: MobileNetV2 endpoint cells (EMC 2133 MHz) and the two-cell model",
         f"original cells profiled at {lr} ({', '.join(o['cell_files'])}); "
         f"re-profile at {'/'.join(f'{x:g}' for x in i33['hz'])} Hz on {', '.join(i33['dates'])}, "
         f"{len(next(iter(o['reps_33'].values())))} repeats x {'/'.join(map(str, i33['iters']))} cycles per cell, "
         f"median of the per-repeat p50 (results/refit33/{o['board']}/emc2133_g<f>_r<k>.json)", "",
         f"| cell | {lr} p50 (us) | 33 Hz p50 (us) | per-repeat p50 (us) | 33 Hz - {lr} (us) |",
         "|---|---:|---:|---|---:|"]
    for g in sorted(o["cells_reg"]):
        a, b = o["cells_reg"][g], o["cells_33"][g]
        L.append(f"| GPU {g} | {a:.1f} | {b:.1f} | {', '.join(f'{x:.1f}' for x in o['reps_33'][g])} | {b - a:+.1f} |")
    if "cells_reg_3199" in o:
        for g in sorted(o["cells_reg_3199"]):
            a, b = o["cells_reg_3199"][g], o["cells_33_3199"][g]
            L.append(f"| GPU {g}, EMC 3199 | {a:.1f} | {b:.1f} | {', '.join(f'{x:.1f}' for x in o['reps_33_3199'][g])} | {b - a:+.1f} |")
    cf = o["cells_33_files"]
    L.append("")
    L.append("derived 33 Hz cell files (" + (", ".join(cf["files"]) or "none") + "): "
             + ("equal to these medians" if cf["equal"] else ("**differ from these medians**" if cf["files"] else "-")))
    m = o["model"]
    L += ["", f"T_hat(f) = k/f + b: {lr} k = {m['fit_reg'][0]:.1f} us*GHz, b = {m['fit_reg'][1]:.1f} us; "
              f"33 Hz k = {m['fit_33'][0]:.1f} us*GHz, b = {m['fit_33'][1]:.1f} us; "
              f"r = {o['R']}, h = {o['h_us']:g} us, f_probe = {o['fprobe']} MHz", "",
          f"| f (MHz) | T_hat {lr} (us) | T_hat 33 Hz (us) | difference (us) | ds(f) = r (T_hat {lr} - T_hat 33 Hz) (us) "
          "| band h f_probe/f (us) | ds / band |",
          "|---:|---:|---:|---:|---:|---:|---:|"]
    for r in m["rows"]:
        L.append(f"| {r['f']} | {r['T_reg_us']:.1f} | {r['T_33_us']:.1f} | {r['dT_us']:+.1f} | {r['ds_us']:+.1f} | "
                 f"{r['band_us']:.1f} | {r['ds_us'] / r['band_us']:+.3f} |")
    mx = max(m["rows"], key=lambda r: abs(r["dT_us"]))
    L.append("")
    L.append(f"largest |T_hat 33 Hz - T_hat {lr}| over the grid: {abs(mx['dT_us']):.1f} us at {mx['f']} MHz; band h f_probe/f "
             f"{min(r['band_us'] for r in m['rows']):.1f}-{max(r['band_us'] for r in m['rows']):.1f} us")
    for s in ("reg", "33"):
        d = o["d_rule"][s]
        L.append(f"deadline rule 2.12 T_hat({d['f_k']}) r with the {lr if s == 'reg' else '33 Hz'} cells: "
                 f"{d['D_us'] / 1000:.3f} ms -> {d['D_ms_rounded']:.1f} ms (T_hat({d['f_k']}) r = {d['D_us'] / 2.12 / 1000:.3f} ms)")
    L.append("")
    return "\n".join(L)


def md_e3(o):
    L = [f"## {o['board']}: E3 MobileNetV2 conditions re-picked with the 33 Hz cells (D, h, r, grid as registered)", ""]
    E = [c for c in o["e3"] if "error" not in c]
    for c in o["e3"]:
        if "error" in c:
            L.append(f"### {c['cond']} - ERROR: {c['error']}")
            continue
        L += [f"### {c['cond']} - {NAMES.get(c['co'], c['co'])} {c['co_hz']} Hz, D = {c['D_ms']:g} ms, registered f_oracle = {c['f_oracle']}", "",
              "| t | M_g (us) | f_b reg -> 33 Hz | f_pt reg -> 33 Hz | contract at the 33 Hz f_b | miss % |",
              "|---:|---:|---|---|---|---:|"]
        for t in c["trials"]:
            fb = f"{ff(t['fb_reg'])} -> **{ff(t['fb_33'])}**" if t["fb_33"] != t["fb_reg"] else f"{ff(t['fb_reg'])} = {ff(t['fb_33'])}"
            fpt = f"{ff(t['fpt_reg'])} -> **{ff(t['fpt_33'])}**" if t["fpt_33"] != t["fpt_reg"] else f"{ff(t['fpt_reg'])} = {ff(t['fpt_33'])}"
            ev = t["ev33"]
            L.append(f"| {t['k']} | {t['Mg_us']:.0f} | {fb} | {fpt} | {'declined' if ev is None else ev['source']} | "
                     f"{'-' if ev is None or not ev['n'] else pct(ev['miss'])} |")
        L += [f"- WARN not re-picked (as in e3_stats.py): {s}" for s in c["skipped"]]
        L.append("")
    if not E:
        return "\n".join(L)
    L += ["### Summary: offline re-pick of the registered trials", "",
          "| cond | tenant | n | f_oracle | banded moved (down/stay/up) | P1 reg -> 33 Hz (UB %) | P2 reg: miss % [UB] | "
          "P2 subst.: miss % [UB] | P2 subst., half-up | launches > tau reg -> subst. | subst. min-max % | point moved (down/stay/up) |",
          "|---|---|---:|---:|---|---|---|---|---|---|---|---|"]
    for c in E:
        r, s = c["reg"], c["r33"]
        p1r = f"{r['P1']['x']}/{r['P1']['n']}"
        p1s = f"{s['P1']['x']}/{s['P1']['n']} ({pct(s['P1']['cp95'], 1)})"
        p2r = "-" if r["P2"] is None else f"{pct(r['P2']['pooled'])} [{pct(r['P2']['boot_ub95'])}]"
        P2 = s["P2"]
        p2s = "-" if P2 is None else f"{pct(P2['pooled'], 3)} [{pct(P2['boot_ub95'], 3)}]"
        p2h = "-" if P2 is None else f"{pct_hu(P2['pooled'])} [{pct_hu(P2['boot_ub95'])}]"
        L.append(f"| {c['cond']} | {NAMES.get(c['co'], c['co'])} ({c['co_hz']} Hz) | {c['n']} | {c['f_oracle']} | {fmoves(c['moves_b'])} | "
                 f"{p1r} -> {p1s} | {p2r} | {p2s} | {p2h} | {r['over_tau']} -> {'-' if P2 is None else P2['over_tau']} | "
                 f"{'-' if P2 is None else pct(P2['min'], 1) + '-' + pct(P2['max'], 1)} | {fmoves(c['moves_pt'])} |")
    tot = {m: sum(c["moves_b"][m] for c in E) for m in ("down", "stay", "up")}
    totp = {m: sum(c["moves_pt"][m] for c in E) for m in ("down", "stay", "up")}
    n = sum(c["n"] for c in E)
    L.append(f"| all | | {n} | | {fmoves(tot)} | | | | | | | {fmoves(totp)} |")
    L.append("")
    bins = sorted({b for c in E for b in c["bins_moved"]})
    moved = sum(c["moves_b"]["down"] + c["moves_b"]["up"] for c in E)
    L.append(f"banded picks moved: {moved} of {n}; bins moved: {bins if bins else 'none'}")
    L.append(f"f_pt reg <= f_b 33 Hz <= f_b reg on {sum(c['bracket'] for c in E)} of {n} probes; moved banded picks landing on the "
             f"registered f_pt (whose point contract ran in the same trial): {sum(c['moved_on_fpt_reg'] for c in E)} of {moved}")
    src = {s: sum(c["r33"]["P2"]["sources"][s] for c in E if c["r33"]["P2"]) for s in ("banded", "point", "reference")}
    L.append(f"substituted P2 evidence: same-trial banded {src['banded']}, same-trial point {src['point']}, reference contract "
             f"{src['reference']}, none {sum(c['unscored'] for c in E)}; P2 bootstrap as e3_stats.py (B, seed below)")
    L.append("")
    R = [c for c in E if c.get("rerun")]
    if R:
        L += [f"### Fresh re-run with the 33 Hz cells (results/e3_{o['board']}_r33, endpoints from e3_stats.analyse)", "",
              "| cond | n | cells = 33 Hz cells | f_oracle | P1: f_b < f_oracle (UB %) | P2: miss % [UB] | launches > tau (min-max %) | "
              "band fired (failed/removed) | verdict | registered verdict |",
              "|---|---:|---|---:|---|---|---|---|---|---|"]
        for c in R:
            x = c["rerun"]
            if "error" in x:
                L.append(f"| {x['cond']} | ERROR: {x['error']} |")
                continue
            V = x["launches"]
            rng = "-" if V is None else (pct(V["min"], 1) if V["min"] == V["max"] else f"{pct(V['min'], 1)}-{pct(V['max'], 1)}")
            L.append(f"| {x['cond']} | {x['n']} | {'yes' if x['cells_equal_33'] else 'NO'} | {x['f_oracle']} | "
                     f"{x['P1']['x']}/{x['P1']['n']} ({pct(x['P1']['cp95'], 1)}) | "
                     f"{'-' if x['P2'] is None else pct(x['P2']['pooled']) + ' [' + pct(x['P2']['boot_ub95']) + ']'} | "
                     f"{'-' if V is None else str(V['x']) + '/' + str(V['n']) + ' (' + rng + ')'} | "
                     f"{x['band']['n_active']}/{x['n']} ({x['band']['n_pt_fail']}/{x['band']['n_pt']}) | {x['verdict']} | {c['reg']['verdict']} |")
        L.append("")
        for c in R:
            x = c["rerun"]
            if "error" in x:
                continue
            L.append(f"{x['cond']}: reference sweep " + " · ".join(f"{f}: {pct(m, 1)}" for f, m in x["ref"].items())
                     + f"; banded picks {fdist(dist(x['f_b'], c['grid']))}"
                     + (f"; below f_oracle: {', '.join(f't{k} at {f}' for k, f in x['below'])}" if x["below"] else "")
                     + (f"; launches > tau: t{', t'.join(map(str, x['launches']['ks']))}" if x["launches"] and x["launches"]["ks"] else ""))
        L.append("")
    return "\n".join(L)


def md_cal(o):
    k, lr = o["calibration"], rate(o)
    L = ["## nx: calibration condition (MobileNetV2 33 Hz vs ViT-S 10 Hz tenant, EMC 2133 MHz) re-picked with the 33 Hz cells", "",
         f"### Joint-governor ladder on the reference probe (n = {k['n_probe']}: M_tail = p98 - p50 = {k['M_tail_us']:.0f} us, "
         f"M_g = max - p50 = {k['M_g_us']:.0f} us)",
         f"blind fit: least squares over the EMC 3199 sweep (joint_gov_policy.fit_sweep) with the {lr} cells, two-cell through the "
         "33 Hz EMC 3199 endpoints with the 33 Hz cells; aware fit: two-cell at EMC 2133.  Measured: results/joint_gov/dep_co_g<f>_r<k>.csv "
         "re-scored at D, pooled % (per-run min-max %; runs > tau)", "",
         f"| D (ms) | rung | pick, {lr} cells | measured there | pick, 33 Hz cells | measured there | move |",
         "|---:|---|---:|---|---:|---|---|"]

    def meas(m, f):
        if f == INF:
            return "declined"
        if m is None:
            return "not deployed"
        return f"{pct(m['pooled'])} ({pct(m['min'], 1)}-{pct(m['max'], 1)}; {m['over_tau']}/{m['runs']} > tau)"
    for r in k["ladder"]:
        L.append(f"| {r['D_us'] / 1000:g} | {r['rung']} | {ff(r['pick_reg'])} | {meas(r['meas_reg'], r['pick_reg'])} | "
                 f"{ff(r['pick_33'])} | {meas(r['meas_33'], r['pick_33'])} | {r['move']} |")
    L.append("")
    for s, lab in (("reg", f"{lr} cells"), ("33", "33 Hz cells")):
        a = k["agreement"][s]
        dis = "; ".join(f"{d['f']} MHz at D = {d['D_us'] / 1000:g} ms: rule {'feasible' if d['rule_feasible'] else 'infeasible'}, "
                        f"measured {pct(d['pooled'])} %" for d in a["disagree"])
        L.append(f"clock-scaled rung (M_tail f_probe/f) vs measured pooled miss <= tau, deployed (clock, D) cells, {lab}: "
                 f"{a['agree']}/{a['n']} agree" + (f" (disagree: {dis})" if dis else ""))
    L += ["", "### Held-out banded round (results/banded_loop; deployed rule banded, D = 12 ms)"]
    H = k["held_out"]
    L.append(f"banded picks: {lr} cells {fdist(dist([t['f_reg'] for t in H], GRID['nx']))} -> 33 Hz cells "
             f"{fdist(dist([t['f_33'] for t in H], GRID['nx']))} (down/stay/up {fmoves(moves([(t['f_reg'], t['f_33']) for t in H], GRID['nx']))}); "
             f"{lr} pick = committed on {sum(t['f_reg'] == t['committed'] for t in H)}/{len(H)}")
    L += ["", f"| t | M_g (us) | f_b {lr} -> 33 Hz | committed (dep_t<k>_g<f>) | deployed contract miss % |", "|---:|---:|---|---:|---:|"]
    for t in H:
        L.append(f"| {t['k']} | {t['Mg_us']:.0f} | {ff(t['f_reg'])} {'->' if t['f_33'] != t['f_reg'] else '='} {ff(t['f_33'])} | "
                 f"{ff(t['committed'])} | {pct(t['dep_miss'])} |")
    F = k["round1"]
    ok = sum(1 for t in F if t["f_reg"] == t["committed"] and "%.0f" % t["Mg_us"] == t["committed_Mg"])
    L += ["", "### Round-1 fresh probes, vision arm (results/fresh_loop; deployed rule point, h = 0, D = 12 ms)"]
    L.append(f"point picks: {lr} cells {fdist(dist([t['f_reg'] for t in F], GRID['nx']))} -> 33 Hz cells "
             f"{fdist(dist([t['f_33'] for t in F], GRID['nx']))} (down/stay/up {fmoves(moves([(t['f_reg'], t['f_33']) for t in F], GRID['nx']))}); "
             f"banded picks {fdist(dist([t['fb_reg'] for t in F], GRID['nx']))} -> {fdist(dist([t['fb_33'] for t in F], GRID['nx']))}; "
             f"{lr} pick and M_g = decisions.csv on {ok}/{len(F)}")
    mv = [t for t in F if t["f_33"] != t["f_reg"]]
    L.append(f"moved trials whose in-session contract ran at the 33 Hz pick: {sum(t['dep_clock'] == t['f_33'] for t in mv)}/{len(mv)}")
    L += ["", f"| t | M_g (us) | f_pt {lr} -> 33 Hz | committed (decisions.csv) | deployed clock | deployed contract miss % |",
          "|---:|---:|---|---:|---:|---:|"]
    for t in F:
        L.append(f"| {t['k']} | {t['Mg_us']:.0f} | {ff(t['f_reg'])} {'->' if t['f_33'] != t['f_reg'] else '='} {ff(t['f_33'])} | "
                 f"{ff(t['committed'])} | {ff(t['dep_clock'])} | {pct(t['dep_miss'])} |")
    L += ["", "### 1000-cycle runs on record under the calibration condition, re-scored at D = 12 ms", "",
          "| clock (MHz) | runs | > tau | without the 33 Hz re-run session: runs, > tau | per source: runs (> tau) |", "|---:|---:|---:|---|---|"]
    for g, S in k["runs"].items():
        n, x = sum(v["runs"] for v in S.values()), sum(v["over_tau"] for v in S.values())
        n0 = sum(v["runs"] for s, v in S.items() if not s.startswith("e3_nx_r33"))
        x0 = sum(v["over_tau"] for s, v in S.items() if not s.startswith("e3_nx_r33"))
        L.append(f"| {g} | {n} | {x} | {n0}, {x0} | " + "; ".join(f"{s} {v['runs']} ({v['over_tau']})" for s, v in S.items()) + " |")
    b = k["band"]

    def pm(p):
        return "%s (%.0f)" % (p, b["mg"][p])
    L += ["", f"### Band width over the {b['n']} collected probes (reference, round 1, held-out), D = 12 ms",
          f"s.d. of M_g over the {b['n_r1']} probes the band was set on (reference + round 1): {b['sd_r1_pop']:.1f} us "
          f"(population; {b['sd_r1_n1']:.1f} us with n-1); sigma = h/2 = {b['sigma']:g} us", "",
          f"| h | {lr} cells: 816 / 918 / 1020 | 33 Hz cells: 816 / 918 / 1020 |", "|---|---|---|"]
    for m in SIGMAS:
        cols = []
        for s in ("reg", "33"):
            d = b["widths"][s][m]["dist"]
            other = {x: y for x, y in d.items() if x not in ("816", "918", "1020")}
            cols.append(" / ".join(str(d.get(str(f), 0)) for f in (816, 918, 1020)) + (f" (other: {fdist(other)})" if other else ""))
        L.append(f"| {m:g} sigma = {m * b['sigma']:g} us | {cols[0]} | {cols[1]} |")
    L.append("")
    for s, lab in (("reg", f"{lr} cells"), ("33", "33 Hz cells")):
        iv = b["interval"][s]
        L.append(f"{lab}: all {b['n']} probes at 918 MHz iff h in ({iv['gt_sigma']:.2f} sigma, {iv['le_sigma']:.2f} sigma] "
                 f"= ({iv['gt_us']:.0f}, {iv['le_us']:.0f}] us; reference probe m*(918) - M_g = {b['ref_margin_918'][s]:.0f} us")
        for m in SIGMAS:
            at = b["widths"][s][m]["at"]
            low = sorted(at.get("816", []) + at.get("714", []) + at.get("612", []) + at.get("510", []), key=lambda x: b["mg"][x])
            high = sorted(at.get("1020", []) + at.get("1122", []) + at.get("1173", []), key=lambda x: -b["mg"][x])
            if low or high:
                L.append(f"  h = {m:g} sigma: " + "; ".join(x for x in (
                    "below 918: " + ", ".join(map(pm, low)) if low else "",
                    "above 918: " + ", ".join(map(pm, high)) if high else "") if x) + "  [probe (M_g us)]")
    top = sorted(b["mg"], key=lambda x: -b["mg"][x])[:4]
    L.append("largest four M_g: " + ", ".join(map(pm, top)))
    L.append("")
    return "\n".join(L)


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--board", choices=["nx", "nano"], help="default: both")
    ap.add_argument("--tau", type=float, default=0.02)
    ap.add_argument("--B", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=20260915)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    out = [board_block(b, a.tau, a.B, a.seed) for b in ([a.board] if a.board else ["nx", "nano"])]
    if a.json:
        print(json.dumps(dict(tau=a.tau, B=a.B, seed=a.seed, boards=out), indent=1, default=str))
        return
    for o in out:
        print(md_cells(o))
        print(md_e3(o))
        if "calibration" in o:
            print(md_cal(o))
    print(f"tau = {pct(a.tau, 0)} %, bootstrap B = {a.B}, seed = {a.seed}; registered endpoints from e3_stats.analyse "
          f"(e3_stats.py prints its cross-check warnings)")


if __name__ == "__main__":
    main()

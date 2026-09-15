#!/usr/bin/env python3
"""E3 endpoint statistics (protocol docs/protocols/E3_protocol.md §5-§6),
one block per condition directory under results/e3_<board>/.

Every quantity is recomputed from the CSVs written by admit_loop.sh and
admit_refsweep.sh; session.log, t<k>.pick.json and the harness JSONs are read
only to cross-check (a mismatch is printed as WARN, never used).

Per trial (t<k>.probe.csv, 200 cycles at f_probe):  p50, max, M_g = max - p50,
  point pick f_pt (h = 0) and banded pick f_b (h = 400 us) via admit_pick.evaluate
  on the condition's two solo cells; m_b / m_pt = miss rate of the 1000-cycle
  contract CSVs re-scored as response_us > D (admit_pick.py --score convention);
  declined = f_b INFEASIBLE.
Reference sweep (ref_g<f>.csv, one 1000-cycle contract per f in G):  miss at D;
  f_oracle = lowest f with miss <= tau ('none' if the whole grid fails; 'no
  reference yet' if the sweep is absent or has a gap below the first feasible f).
Endpoints (§5):
  P1  #admitted trials with f_b < f_oracle          (one-sided 95 % Clopper-Pearson UB)
  P2  pooled banded miss (cycles-weighted) with a run-level bootstrap upper bound
      (trials resampled with replacement, B, fixed seed; 95th pct one-sided,
      97.5th = upper end of the two-sided CI); pass iff pooled <= tau
  per-launch violations  #admitted with m_b > tau, min/max m_b (CP UB)
  band necessity         #trials with f_pt != f_b; among them #point contracts with m_pt > tau
  acceptance rate        admitted / completed trials
  utilization cost       f_b - f_oracle in grid bins (median, distribution)
  probe -> realization   Pearson and Spearman of (M_g, m_b) over admitted trials
Diagnostic (not a protocol quantity): a probe whose M_g is at most twice the
  solo cell's own max - p50 at f_probe is flagged 'probe~solo' (the probe did
  not see the co-tenant); the threshold is printed in the block header.
Verdict (§6, amended 2026-09-15): 'supported' iff P1 = 0 and P2 holds; 'not
  supported' if P1 > 0 or P2 fails; 'band idle' if f_pt = f_b on every trial.
  Band necessity (removed admissions failed/passed) is reported, not judged;
  without a reference sweep the verdict is
  'pending reference' unless P2 already fails.
numpy only (no scipy in the venv); cp_upper is imported from launch_ext_stats.py.

Usage: e3_stats.py --board nano|nx [--cond A1 A2 ...] [--tau 0.02] [--D ms]
         [--B 10000] [--seed 20260915] [--results DIR] [--json]
"""
import argparse, csv, glob, json, os, re, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LAB = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from admit_pick import load_probe, load_cells, evaluate, GRID, FPROBE, CELLS, H_US, R_MULT   # noqa: E402
from launch_ext_stats import cp_upper                                                       # noqa: E402

INF = "INFEASIBLE"
NAMES = {"mobilenet": "MobileNetV2", "vit": "ViT-S", "yolov8s": "YOLOv8s", "resnet50": "ResNet-50"}


# ----------------------------------------------------------------------------- helpers
def score_csv(path, D_us):
    v = load_probe(path)
    if v.size == 0:
        return None
    m = int(np.count_nonzero(v > D_us))
    return dict(n=int(v.size), misses=m, miss=m / v.size, p99_us=float(np.percentile(v, 99)), max_us=float(v.max()))


def json_miss_check(prefix, D_us, misses, warn, what):
    """Compare the harness's own deadline_miss_count with the CSV re-score (same D only)."""
    js = prefix + ".json"
    if not (os.path.isfile(js) and os.path.getsize(js) > 0):
        warn.append(f"{what}: {os.path.basename(js)} missing or empty (contract may have been interrupted)")
        return
    try:
        meta = json.load(open(js))
    except Exception as e:                       # noqa: BLE001
        warn.append(f"{what}: {os.path.basename(js)} unreadable ({e})")
        return
    if abs(float(meta["meta"].get("deadline_us", -1)) - D_us) < 1e-6 and meta.get("deadline_miss_count") != misses:
        warn.append(f"{what}: harness miss count {meta.get('deadline_miss_count')} != CSV re-score {misses}")


def rankdata(a):
    a = np.asarray(a, float)
    order = np.argsort(a, kind="mergesort")
    sa = a[order]
    ranks = np.empty(a.size)
    i = 0
    while i < a.size:
        j = i
        while j + 1 < a.size and sa[j + 1] == sa[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


def pearson(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.size < 3 or x.std() == 0 or y.std() == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x, y):
    if len(x) < 3:
        return None
    return pearson(rankdata(x), rankdata(y))


def bootstrap_pooled(misses, n, B, seed):
    """Run-level bootstrap of the pooled miss: returns (p2.5, p95, p97.5); same sampling as launch_ext_stats."""
    misses, n = np.asarray(misses), np.asarray(n)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, misses.size, size=(B, misses.size))
    pooled = misses[idx].sum(axis=1) / n[idx].sum(axis=1)
    return tuple(float(np.percentile(pooled, q)) for q in (2.5, 95, 97.5))


def pct(x, nd=2):
    return "-" if x is None else f"{100 * x:.{nd}f} %"


def ff(f):
    return "INF" if f == INF else ("-" if f is None else str(f))


def fkey(f):
    return 0 if f == INF else f


# ----------------------------------------------------------------------------- inputs
def read_condition(cdir, warn):
    """condition.txt (admit_loop.sh SIG) -> dict; falls back to the first pick.json."""
    c = {}
    p = os.path.join(cdir, "condition.txt")
    if os.path.isfile(p):
        for tok in open(p).read().split():
            if "=" in tok:
                k, v = tok.split("=", 1)
                c[k] = v
    if not c:
        picks = sorted(glob.glob(os.path.join(cdir, "t*.pick.json")))
        if not picks:
            return None
        warn.append("condition.txt missing; parameters taken from " + os.path.basename(picks[0]))
        pj = json.load(open(picks[0]))
        c = dict(board=pj.get("board"), primary=pj.get("primary"), D_ms=str(pj["D_us"] / 1000), fprobe=str(pj["fprobe"]),
                 grid=",".join(map(str, pj["grid"])), h_us=str(pj["h_us"]), R=str(pj["R"]), cells=",".join(pj["cell_files"]))
    out = dict(board=c.get("board"), primary=c.get("primary"), co=c.get("co"), co_hz=c.get("co_hz"), pri_hz=c.get("pri_hz"),
               D_ms=float(c["D_ms"]) if "D_ms" in c else None,
               fprobe=int(c["fprobe"]) if "fprobe" in c else FPROBE.get(c.get("board")),
               grid=[int(x) for x in c["grid"].split(",")] if "grid" in c else GRID.get(c.get("board")),
               h_us=float(c.get("h_us", H_US)), R=float(c.get("R", R_MULT)))
    cells = c["cells"].split(",") if "cells" in c else CELLS.get((out["board"], out["primary"]), [])
    out["cell_files"] = [x if os.path.isabs(x) else os.path.join(LAB, x) for x in cells]
    return out


def parse_session_log(path):
    """Last '  t<k>: ...' line per trial, ref lines and the REFSWEEP DONE oracle (cross-check only)."""
    trials, refs, oracle, fails = {}, {}, None, []
    if not os.path.isfile(path):
        return trials, refs, oracle, fails
    for line in open(path):
        m = re.match(r"\s*t(\d+): Mg=(-?\d+)us f_pt=(\d+) f_b=(\d+) m_b=(\S+) m_pt=(\S+)", line)
        if m:
            k = int(m.group(1))
            trials[k] = dict(Mg=int(m.group(2)), f_pt=int(m.group(3)), f_b=int(m.group(4)),
                             m_b=None if m.group(5) in ("-", "FAIL") else float(m.group(5)),
                             m_pt=None if m.group(6) in ("-", "FAIL") else float(m.group(6)))
            continue
        m = re.match(r"\s*t(\d+): (PROBE|PICK) FAIL", line)
        if m:
            fails.append(f"t{m.group(1)} {m.group(2)} FAIL")
            continue
        m = re.match(r"\s*ref g(\d+): m=(\S+)", line)
        if m:
            refs[int(m.group(1))] = None if m.group(2) in ("-", "FAIL") else float(m.group(2))
            continue
        m = re.match(r"REFSWEEP DONE cond=\S+ f_oracle=(\S+)", line)
        if m:
            oracle = m.group(1)
    return trials, refs, oracle, fails


def load_trial(cdir, k, cond, D_us, cells, solo_disp, warn):
    tag = os.path.join(cdir, f"t{k}")
    t = dict(k=k, complete=False, note=[])
    complete = os.path.isfile(tag + ".done") or (os.path.isfile(tag + ".contract_b.json") and os.path.getsize(tag + ".contract_b.json") > 0)
    if not os.path.isfile(tag + ".probe.csv") or os.path.getsize(tag + ".probe.csv") == 0:
        t["note"].append("no probe CSV")
        return t
    v = load_probe(tag + ".probe.csv")
    r = evaluate(v, cells, D_us, cond["grid"], cond["fprobe"], cond["h_us"], cond["R"])
    t.update(n_probe=int(v.size), p50_us=r["p50_us"], max_us=r["max_us"], Mg_us=r["Mg_us"],
             f_pt=r["f_pt"], f_b=r["f_b"], s_pt_us=r["s_at_f_pt_us"], s_b_us=r["s_at_f_b_us"])
    if v.size != 200:
        warn.append(f"t{k}: probe has {v.size} cycles, expected 200")
    # cross-check the deployed pick
    pj = tag + ".pick.json"
    if os.path.isfile(pj):
        try:
            p = json.load(open(pj))
            if p.get("f_pt") != r["f_pt"] or p.get("f_b") != r["f_b"] or abs(float(p.get("Mg_us", np.nan)) - r["Mg_us"]) > 0.5:
                warn.append(f"t{k}: pick.json (Mg={p.get('Mg_us')}, f_pt={p.get('f_pt')}, f_b={p.get('f_b')}) "
                            f"!= recomputed (Mg={r['Mg_us']:.1f}, f_pt={r['f_pt']}, f_b={r['f_b']})")
            if abs(float(p.get("D_us", D_us)) - D_us) > 1e-6:
                warn.append(f"t{k}: pick.json D_us={p.get('D_us')} != D used here {D_us}")
        except Exception as e:                   # noqa: BLE001
            warn.append(f"t{k}: pick.json unreadable ({e})")
    else:
        warn.append(f"t{k}: pick.json missing")
    t["declined"] = r["f_b"] == INF
    t["band_active"] = r["f_pt"] != r["f_b"]
    t["probe_solo"] = bool(solo_disp is not None and r["Mg_us"] <= 2.0 * solo_disp)
    # contracts
    for kind, f in (("b", r["f_b"]), ("pt", r["f_pt"])):
        pre = f"{tag}.contract_{kind}"
        t[f"m_{kind}"] = None
        expected = (f != INF) if kind == "b" else (r["f_pt"] != r["f_b"] and r["f_pt"] != INF)
        if os.path.isfile(pre + ".csv") and os.path.getsize(pre + ".csv") > 0:
            s = score_csv(pre + ".csv", D_us)
            if s is None:
                warn.append(f"t{k}: contract_{kind}.csv has no rows")
                continue
            t.update({f"m_{kind}": s["miss"], f"n_{kind}": s["n"], f"misses_{kind}": s["misses"],
                      f"p99_{kind}_us": s["p99_us"], f"max_{kind}_us": s["max_us"]})
            if s["n"] != 1000:
                warn.append(f"t{k}: contract_{kind} has {s['n']} cycles, expected 1000")
            json_miss_check(pre, D_us, s["misses"], warn, f"t{k} contract_{kind}")
            if not expected:
                warn.append(f"t{k}: contract_{kind} present although not expected (f_pt={r['f_pt']}, f_b={r['f_b']})")
            try:   # the contract must have run at the recomputed pick
                lab = json.load(open(pre + ".json"))["meta"].get("label", "")
                g = re.search(r"_g(\d+)$", lab)
                if g and int(g.group(1)) != f:
                    warn.append(f"t{k}: contract_{kind} ran at {g.group(1)} MHz but the recomputed pick is {f}")
            except Exception:                    # noqa: BLE001
                pass
        elif expected and complete:
            warn.append(f"t{k}: contract_{kind} expected at {f} MHz but missing")
            complete = False
    t["complete"] = complete
    if not complete:
        t["note"].append("incomplete (no .done / contract_b.json)")
    return t


# ----------------------------------------------------------------------------- one condition
def analyse(cdir, cond_id, tau, B, seed, D_override=None):
    warn = []
    cond = read_condition(cdir, warn)
    if cond is None:
        return dict(cond=cond_id, dir=cdir, error="no condition.txt and no pick.json", warnings=warn)
    D_ms = D_override if D_override is not None else cond["D_ms"]
    if D_ms is None:
        return dict(cond=cond_id, dir=cdir, error="D unknown (no condition.txt D_ms, no pick.json, no --D)", warnings=warn)
    D_us = D_ms * 1000.0
    grid, fprobe = cond["grid"], cond["fprobe"]
    for f in cond["cell_files"]:
        if not os.path.isfile(f):
            return dict(cond=cond_id, dir=cdir, error=f"solo cell missing: {f}", warnings=warn)
    cells = load_cells(cond["cell_files"])
    solo_disp = None
    for f in cond["cell_files"]:
        if int(re.search(r"gpu(\d+)", os.path.basename(f)).group(1)) == fprobe:
            ru = json.load(open(f))["response_us"]
            if "max" in ru and "p50" in ru:
                solo_disp = float(ru["max"]) - float(ru["p50"])
    log_trials, log_refs, log_oracle, log_fails = parse_session_log(os.path.join(cdir, "session.log"))

    # ---- trials
    ks = sorted({int(m.group(1)) for p in glob.glob(os.path.join(cdir, "t*.*"))
                 for m in [re.search(r"/t(\d+)\.", p)] if m})
    trials = [load_trial(cdir, k, cond, D_us, cells, solo_disp, warn) for k in ks]
    for t in trials:   # cross-check session.log
        L = log_trials.get(t["k"])
        if L is None or "Mg_us" not in t:
            if t.get("complete"):
                warn.append(f"t{t['k']}: no session.log line")
            continue
        bad = []
        if L["Mg"] != int(round(t["Mg_us"])):
            bad.append(f"Mg {L['Mg']} vs {t['Mg_us']:.0f}")
        if L["f_pt"] != fkey(t["f_pt"]) or L["f_b"] != fkey(t["f_b"]):
            bad.append(f"picks ({L['f_pt']},{L['f_b']}) vs ({fkey(t['f_pt'])},{fkey(t['f_b'])})")
        for kind in ("b", "pt"):
            a, b = L[f"m_{kind}"], t.get(f"m_{kind}")
            if (a is None) != (b is None) or (a is not None and abs(a - round(b, 4)) > 5e-5):
                bad.append(f"m_{kind} {a} vs {None if b is None else round(b, 4)}")
        if bad:
            warn.append(f"t{t['k']}: session.log mismatch: " + "; ".join(bad))
    for s in log_fails:
        warn.append("session.log: " + s)
    done = [t for t in trials if t["complete"]]
    incomplete = [t["k"] for t in trials if not t["complete"]]
    admitted = [t for t in done if not t["declined"] and t.get("m_b") is not None]
    declined = [t for t in done if t["declined"]]

    # ---- reference sweep
    ref, ref_inprogress = {}, []
    for f in grid:
        p = os.path.join(cdir, f"ref_g{f}.csv")
        js = os.path.join(cdir, f"ref_g{f}.json")
        if os.path.isfile(p) and os.path.getsize(p) > 0:
            if not (os.path.isfile(js) and os.path.getsize(js) > 0):     # cell still being written (or aborted)
                ref_inprogress.append(f)
                continue
            s = score_csv(p, D_us)
            if s:
                ref[f] = s
                if s["n"] != 1000:
                    warn.append(f"ref g{f}: {s['n']} cycles, expected 1000")
                json_miss_check(os.path.join(cdir, f"ref_g{f}"), D_us, s["misses"], warn, f"ref g{f}")
                if f in log_refs and log_refs[f] is not None and abs(log_refs[f] - round(s["miss"], 4)) > 5e-5:
                    warn.append(f"ref g{f}: session.log m={log_refs[f]} vs recomputed {s['miss']:.4f}")
    if ref_inprogress:
        warn.append(f"reference cells without harness JSON (in progress or aborted), ignored: {ref_inprogress}")
    if not ref:
        ref_status, f_oracle = "missing", None
    else:
        ref_status = "complete" if len(ref) == len(grid) else f"partial ({len(ref)}/{len(grid)} cells)"
        f_oracle = None
        for f in grid:
            if f not in ref:
                f_oracle = "undetermined"          # a gap below the first feasible cell
                break
            if ref[f]["miss"] <= tau:
                f_oracle = f
                break
        if f_oracle is None:
            f_oracle = "none"                      # every cell fails tau
    if log_oracle is not None and ref_status == "complete":
        lo = INF if log_oracle == "INFEASIBLE" else log_oracle
        if str(lo) != str("none" if f_oracle == "none" else f_oracle).replace("none", INF):
            if not (f_oracle == "none" and log_oracle == "INFEASIBLE"):
                warn.append(f"session.log f_oracle={log_oracle} vs recomputed {f_oracle}")
    have_ref = isinstance(f_oracle, int) or f_oracle == "none"

    # ---- endpoints
    n_adm = len(admitted)
    for t in admitted:
        t["viol"] = t["m_b"] > tau
        t["false_adm"] = (t["f_b"] < f_oracle) if isinstance(f_oracle, int) else (True if f_oracle == "none" else None)
        t["bins"] = (grid.index(t["f_b"]) - grid.index(f_oracle)) if isinstance(f_oracle, int) else None
    P1 = sum(bool(t["false_adm"]) for t in admitted) if have_ref else None
    P1_cp = cp_upper(P1, n_adm, 0.05) if (P1 is not None and n_adm) else None
    if n_adm:
        mis = np.array([t["misses_b"] for t in admitted]); nn = np.array([t["n_b"] for t in admitted])
        pooled = float(mis.sum() / nn.sum())
        lo, ub95, ub975 = bootstrap_pooled(mis, nn, B, seed)
        P2 = dict(pooled=pooled, misses=int(mis.sum()), cycles=int(nn.sum()), boot_lo=lo, boot_ub95=ub95, boot_ub975=ub975,
                  passed=pooled <= tau, ub95_within=ub95 <= tau, degenerate=int(mis.sum()) == 0)
        viol = [t for t in admitted if t["viol"]]
        viols = dict(x=len(viol), n=n_adm, cp95=cp_upper(len(viol), n_adm, 0.05),
                     min=float(min(t["m_b"] for t in admitted)), max=float(max(t["m_b"] for t in admitted)),
                     ks=[t["k"] for t in viol])
    else:
        P2, viols = None, None
    active = [t for t in done if t["band_active"]]
    with_pt = [t for t in active if t.get("m_pt") is not None]
    pt_fail = [t for t in with_pt if t["m_pt"] > tau]
    band = dict(n_active=len(active), n_pt=len(with_pt), n_pt_fail=len(pt_fail),
                cp95_pt_fail=cp_upper(len(pt_fail), len(with_pt), 0.05) if with_pt else None,
                active_ks=[t["k"] for t in active], pt_fail_ks=[t["k"] for t in pt_fail],
                n_band_declined=sum(t["declined"] for t in active))
    accept = dict(n=n_adm, N=len(done), rate=(n_adm / len(done)) if done else None, declined_ks=[t["k"] for t in declined])
    if isinstance(f_oracle, int) and n_adm:
        bins = [t["bins"] for t in admitted]
        util = dict(median=float(np.median(bins)), mean=float(np.mean(bins)),
                    dist={int(b): int(sum(1 for x in bins if x == b)) for b in sorted(set(bins))})
    else:
        util = None
    if n_adm:
        xs, ys = [t["Mg_us"] for t in admitted], [t["m_b"] for t in admitted]
        corr = dict(n=n_adm, pearson=pearson(xs, ys), spearman=spearman(xs, ys))
    else:
        corr = None
    diag = dict(solo_disp_us=solo_disp, probe_solo_ks=[t["k"] for t in done if t.get("probe_solo")])

    # ---- verdict (§6)
    if not done:
        verdict = "no completed trials"
    elif P2 is None:
        verdict = f"no admissions ({len(declined)}/{len(done)} declined) - P2 undefined"
    elif not P2["passed"]:
        verdict = f"NOT SUPPORTED - P2 fails (pooled banded miss {pct(P2['pooled'])} > {pct(tau, 0)})"
        if P1:
            verdict += f"; P1 = {P1}"
    elif P1 is None:
        verdict = f"pending reference sweep (P1 undetermined); P2 holds ({pct(P2['pooled'])})"
        verdict += "; band idle" if not active else f"; band active on {len(active)}/{len(done)}, {len(pt_fail)}/{len(with_pt)} removed point admissions failed"
    elif P1 > 0:
        verdict = f"NOT SUPPORTED - P1 = {P1}/{n_adm} banded picks below f_oracle = {f_oracle}"
    elif not active:
        verdict = f"BAND IDLE - P1 = 0, P2 holds ({pct(P2['pooled'])}), f_pt = f_b on every trial: condition did not exercise the band"
    else:
        # §6 as amended 2026-09-15 (E3_protocol.md §8): supported iff P1 = 0 and P2 holds;
        # band necessity is reported as (removed admissions that failed / that passed) and
        # enters the utilization cost, not the verdict.
        verdict = (f"SUPPORTED - P1 = 0, P2 holds ({pct(P2['pooled'])}); band active on {len(active)}/{len(done)} trials, "
                   f"removed point admissions: {len(pt_fail)} failed / {len(with_pt) - len(pt_fail)} passed")
    if diag["probe_solo_ks"]:
        verdict += f"  [diag: {len(diag['probe_solo_ks'])}/{len(done)} probes indistinguishable from solo]"

    return dict(cond=cond_id, dir=os.path.relpath(cdir, LAB), board=cond["board"], primary=cond["primary"], pri_hz=cond["pri_hz"],
                co=cond["co"], co_hz=cond["co_hz"], D_ms=D_ms, tau=tau, fprobe=fprobe, grid=grid, h_us=cond["h_us"], R=cond["R"],
                cells={str(k): v for k, v in cells.items()}, B=B, seed=seed,
                n_trials_found=len(trials), n_complete=len(done), incomplete=incomplete,
                trials=[{k: v for k, v in t.items() if k != "note"} | {"note": "; ".join(t["note"])} for t in trials],
                reference=dict(status=ref_status, f_oracle=f_oracle, log_oracle=log_oracle,
                               cells={str(f): dict(miss=ref[f]["miss"], misses=ref[f]["misses"], n=ref[f]["n"]) for f in ref}),
                P1=dict(x=P1, n=n_adm, cp95=P1_cp) if have_ref else dict(x=None, n=n_adm, cp95=None, status="no reference yet"),
                P2=P2, violations=viols, band=band, acceptance=accept, utilization=util, corr=corr, diag=diag,
                verdict=verdict, warnings=warn)


# ----------------------------------------------------------------------------- markdown
def md_block(c):
    L = []
    if "error" in c:
        L.append(f"## {c['cond']} - ERROR: {c['error']}\n")
        return "\n".join(L)
    tau = c["tau"]
    title = (f"## {c['cond']} - {c['board']}, {NAMES.get(c['primary'], c['primary'])} {c['pri_hz']} Hz vs "
             f"{NAMES.get(c['co'], c['co'])} {c['co_hz']} Hz, D = {c['D_ms']:g} ms, tau = {pct(tau, 0)}")
    L.append(title)
    L.append(f"{c['n_complete']} completed trials" + (f" (incomplete: {c['incomplete']})" if c["incomplete"] else "")
             + f"; f_probe {c['fprobe']} MHz, h = {c['h_us']:g} us, R = {c['R']}, cells p50 " + ", ".join(f"{k}: {v:.0f} us" for k, v in c["cells"].items())
             + (f"; solo dispersion at f_probe {c['diag']['solo_disp_us']:.0f} us (probe~solo flag: M_g <= {2 * c['diag']['solo_disp_us']:.0f} us)" if c["diag"]["solo_disp_us"] is not None else ""))
    L.append("")
    L.append("| t | probe p50 (us) | probe max (us) | M_g (us) | f_pt | f_b | m_b | m_pt | flags |")
    L.append("|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    for t in c["trials"]:
        if "Mg_us" not in t:
            L.append(f"| {t['k']} | - | - | - | - | - | - | - | {t['note']} |")
            continue
        flags = []
        if not t["complete"]:
            flags.append("INCOMPLETE")
        if t["declined"]:
            flags.append("declined")
        if t["band_active"]:
            flags.append("band" + ("→declined" if t["declined"] else ""))
        if t.get("viol"):
            flags.append("m_b>tau")
        if t.get("false_adm"):
            flags.append("f_b<f_oracle")
        if t.get("m_pt") is not None and t["m_pt"] > tau:
            flags.append("m_pt>tau")
        if t.get("probe_solo"):
            flags.append("probe~solo")
        L.append(f"| {t['k']} | {t['p50_us']:.0f} | {t['max_us']:.0f} | {t['Mg_us']:.0f} | {ff(t['f_pt'])} | {ff(t['f_b'])} | "
                 f"{pct(t.get('m_b'))} | {pct(t.get('m_pt'))} | {', '.join(flags)} |")
    L.append("")
    R = c["reference"]
    if R["status"] == "missing":
        L.append("Reference sweep: **missing - no reference yet** (P1 and utilization cost undetermined)")
    else:
        cellsm = " · ".join(f"{f}: {pct(R['cells'][str(f)]['miss'], 1)}" for f in c["grid"] if str(f) in R["cells"])
        L.append(f"Reference sweep ({R['status']}; miss at D, 1000 cycles per f): {cellsm} → **f_oracle = {R['f_oracle']}**"
                 + (" (whole grid infeasible)" if R["f_oracle"] == "none" else "")
                 + (" (gap below the first feasible cell - no reference yet)" if R["f_oracle"] == "undetermined" else ""))
    L.append("")
    L.append("| endpoint | value |")
    L.append("|---|---|")
    P1, P2, V, Bn, A, U, C = c["P1"], c["P2"], c["violations"], c["band"], c["acceptance"], c["utilization"], c["corr"]
    if P1["x"] is None:
        L.append(f"| P1 low-clock false admission (f_b < f_oracle) | no reference yet ({P1['n']} admitted) |")
    else:
        L.append(f"| P1 low-clock false admission (f_b < f_oracle) | **{P1['x']} / {P1['n']}** admitted; one-sided 95 % CP UB {pct(P1['cp95'], 1)} |")
    if P2 is None:
        L.append("| P2 pooled banded miss | no admitted contracts |")
    else:
        L.append(f"| P2 pooled banded miss (cycles-weighted) | **{pct(P2['pooled'])}** ({P2['misses']}/{P2['cycles']}); "
                 f"run-level bootstrap UB {pct(P2['boot_ub95'])} (one-sided 95 %), {pct(P2['boot_ub975'])} (two-sided 97.5 %)"
                 + (" - degenerate, 0 events" if P2["degenerate"] else "")
                 + f" → **{'holds' if P2['passed'] else 'FAILS'}** vs tau = {pct(tau, 0)}"
                 + (" (UB also ≤ tau)" if P2["passed"] and P2["ub95_within"] else (" (UB > tau)" if P2["passed"] else "")) + " |")
    if V is None:
        L.append("| per-launch violations (m_b > tau) | - |")
    else:
        L.append(f"| per-launch violations (m_b > tau; layer 3, reported) | {V['x']} / {V['n']} (CP UB {pct(V['cp95'], 1)})"
                 + (f", trials {V['ks']}" if V["ks"] else "") + f"; m_b min {pct(V['min'])}, max {pct(V['max'])} |")
    L.append(f"| band necessity | f_pt ≠ f_b on {Bn['n_active']} / {c['n_complete']} trials"
             + (f" (band-declined: {Bn['n_band_declined']})" if Bn["n_band_declined"] else "")
             + (f"; point contracts with m_pt > tau: **{Bn['n_pt_fail']} / {Bn['n_pt']}**" + (f" (CP UB {pct(Bn['cp95_pt_fail'], 1)})" if Bn["n_pt"] else "")
                + (f", trials {Bn['pt_fail_ks']}" if Bn["pt_fail_ks"] else "") if Bn["n_active"] else " (band idle)") + " |")
    L.append(f"| acceptance rate | {A['n']} / {A['N']}" + (f" = {pct(A['rate'], 0)}" if A["rate"] is not None else "")
             + (f"; declined trials {A['declined_ks']}" if A["declined_ks"] else "") + " |")
    if U is None:
        L.append("| utilization cost f_b − f_oracle (grid bins) | " + ("no reference yet" if P1["x"] is None else "n/a") + " |")
    else:
        L.append(f"| utilization cost f_b − f_oracle (grid bins) | median {U['median']:g}, mean {U['mean']:.2f}; distribution "
                 + ", ".join(f"{b:+d}: {n}" for b, n in U["dist"].items()) + " |")
    if C is None:
        L.append("| corr(M_g, m_b) over admitted trials | - |")
    else:
        fmt = lambda x: "undefined (zero variance)" if x is None else f"{x:+.2f}"   # noqa: E731
        L.append(f"| corr(M_g, m_b) over admitted trials | n = {C['n']}, Pearson {fmt(C['pearson'])}, Spearman {fmt(C['spearman'])} |")
    if c["diag"]["probe_solo_ks"]:
        L.append(f"| diagnostic: probe~solo | {len(c['diag']['probe_solo_ks'])} / {c['n_complete']} probes, trials {c['diag']['probe_solo_ks']} |")
    L.append("")
    L.append(f"**Verdict (§6):** {c['verdict']}")
    for w in c["warnings"]:
        L.append(f"- WARN {w}")
    L.append("")
    return "\n".join(L)


# ----------------------------------------------------------------------------- main
def cond_key(name):
    m = re.match(r"([A-Za-z]+)(\d+)", name)
    return (m.group(1), int(m.group(2))) if m else (name, 0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--board", choices=["nano", "nx"], required=True)
    ap.add_argument("--cond", nargs="+", help="condition ids (default: every directory under results/e3_<board>)")
    ap.add_argument("--tau", type=float, default=0.02)
    ap.add_argument("--D", type=float, help="override D in ms (default: condition.txt D_ms, else pick.json D_us)")
    ap.add_argument("--B", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=20260915)
    ap.add_argument("--results", help="results root (default results/e3_<board>)")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    root = a.results or os.path.join(LAB, "results", f"e3_{a.board}")
    if not os.path.isdir(root):
        sys.exit(f"no such directory {root}")
    conds = a.cond or sorted((d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))), key=cond_key)
    out = []
    for cid in conds:
        d = os.path.join(root, cid)
        if not os.path.isdir(d):
            out.append(dict(cond=cid, error=f"directory {d} not found", warnings=[]))
            continue
        out.append(analyse(d, cid, a.tau, a.B, a.seed, a.D))
    if a.json:
        print(json.dumps(dict(board=a.board, results=os.path.relpath(root, LAB), tau=a.tau, B=a.B, seed=a.seed, conditions=out), indent=1))
        return
    for c in out:
        print(md_block(c))


if __name__ == "__main__":
    main()

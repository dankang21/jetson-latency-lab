#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Table III (deployed blind-vs-aware governor, one representative cell per (workload, SKU)).

Vision rows: six (workload, SKU) entries -- MobileNetV2 on the NX and the Nano, ViT-Small on the NX
and the Nano, ResNet-50 and YOLOv8s on the NX -- each scored over its pre-stated deadline grid
(docs/protocols/E1_E2_protocol.md and its E1b / Nano ViT-Small addenda, frozen 2026-09-14/15) on
its interleaved sessions under results/govrep_rev, with the rule, fits, cell scoring and energy
convention of Table II (campaign/govdeploy_common.py, campaign/govdeploy_stats.py).  The
representative row is the TIGHTEST grid deadline at which the aware arm admits a clock (does not
decline).  Headline fits are the rate-matched endpoint cells (results/refit33/<dir>, endpoint p50 =
median of three repeat p50s) when their release rate is the deployment rate; otherwise the original
profiling cells, which for MobileNetV2 on the Nano are the 50 Hz deployment-rate cells (the
results/refit33/nano cells are a 33 Hz re-profile and are printed as an extra evaluation).  The
original-cell evaluation of every grid cell is printed as the second evaluation of the same runs.
Where a representative cell's dE interval spans zero, the per-pair energies and the two-state
VDD_IN baseline (run powers split at their largest gap) behind it are printed.
Qwen2.5 rows: the Qwen-1.5B 2,000-token entry (D = 29 ms/tok, aware 1020 MHz: miss % [run-level
95 % UB]) is computed as Table V's 2,000-token column (campaign/qwengov_2k.py); the other Qwen cells
are printed by campaign/analyze_qwen_deploy.py (1.5B) and campaign/analyze_breadth.py (3B).

Read-only on results/ (python3 + numpy; no scipy); prints to stdout, writes nothing.
Usage: solo_stats.py [--brief]
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LAB = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from govdeploy_common import original_fits, refit_fits, that, load_session, evaluate   # noqa: E402
import qwengov_2k                                                                      # noqa: E402

NX_GRID = [510, 612, 714, 816, 918, 1020, 1122, 1173]
NANO_GRID = [306, 408, 510, 612, 714, 816, 918, 1020]

# Table order.  D grids are the pre-stated ones (E1_E2_protocol.md and its E1b / Nano ViT
# addenda, frozen 2026-09-14/15).
ENTRIES = [
    dict(key="mnv2_nx", label="MobileNetV2 (NX)", sku="NX", hz=33.0, lo=306, fmax=1173, grid=NX_GRID,
         D=[4.4, 4.6, 4.8, 5.0, 5.2, 5.4, 5.6, 5.8, 6.0],
         sessions=["nx", "nx_mobilenet", "nx_mobilenet_s2", "nx_mobilenet_s3"],
         orig=lambda emc, g: f"{LAB}/results/nx_validation/nx_est_emc{emc}_gpu{g}_mobilenet.json",
         refit=f"{LAB}/results/refit33/nx"),
    dict(key="mnv2_nano", label="MobileNetV2 (Nano)", sku="Nano", hz=50.0, lo=306, fmax=1020, grid=NANO_GRID,
         D=[8.5, 9.0, 9.5, 10.0],
         sessions=["nano", "nano_mobilenet", "nano_mobilenet_s2", "nano_mobilenet_s3"],
         orig=lambda emc, g: f"{LAB}/results/nano_grid/mobilenet/est_emc{emc}_gpu{g}.json",
         refit=f"{LAB}/results/refit33/nano"),
    dict(key="vit_nx", label="ViT-Small (NX)", sku="NX", hz=20.0, lo=306, fmax=1173, grid=NX_GRID,
         D=[14.0, 14.5, 15.0, 15.5, 16.0, 17.0, 17.5, 18.5, 19.0],
         sessions=["nx_vit"],
         orig=lambda emc, g: f"{LAB}/results/nx_validation/nx_est_emc{emc}_gpu{g}_vit.json",
         refit=f"{LAB}/results/refit33/nx_vit"),
    dict(key="vit_nano", label="ViT-Small (Nano)", sku="Nano", hz=20.0, lo=306, fmax=1020, grid=NANO_GRID,
         D=[14.0, 14.5, 15.0, 15.5, 16.5, 17.0, 18.0, 18.5, 20.0, 20.5],
         sessions=["nano_vit"],
         orig=lambda emc, g: f"{LAB}/results/nano_grid/vit/est_emc{emc}_gpu{g}.json",
         refit=f"{LAB}/results/refit33/nano_vit"),
    dict(key="resnet50_nx", label="ResNet-50 (NX)", sku="NX", hz=20.0, lo=306, fmax=1173, grid=NX_GRID,
         D=[14.0, 14.5, 15.0, 15.5, 16.0, 17.0, 17.5, 18.5, 19.0],
         sessions=["nx_resnet50"],
         orig=lambda emc, g: f"{LAB}/results/nx_validation/nx_est_emc{emc}_gpu{g}_resnet50.json",
         refit=f"{LAB}/results/refit33/nx_resnet50"),
    dict(key="yolov8s_nx", label="YOLOv8s (NX)", sku="NX", hz=10.0, lo=306, fmax=1173, grid=NX_GRID,
         D=[29.5, 30.0, 31.0, 31.5, 33.0, 33.5, 35.5, 36.0, 38.5, 39.0],
         sessions=["nx_yolov8s"],
         orig=lambda emc, g: f"{LAB}/results/nx_validation/nx_est_emc{emc}_gpu{g}_yolov8s.json",
         refit=f"{LAB}/results/refit33/nx_yolov8s"),
]


# ------------------------------------------------------------- evaluate ----
def representative(rows):
    """Tightest pre-stated deadline at which the aware arm admits a clock."""
    for r in rows:
        if r["oka"]:
            return r
    raise AssertionError("aware arm declines at every grid deadline")


def viol0(cell):
    """Total violating runs over the pooled sessions (None if no data)."""
    if not cell["has_data"]:
        return None
    return sum(k for k in cell["viol"] if k is not None)


def show_cell(c, ok):
    if not c["has_data"]:
        return f"{c['f']}{'' if ok else '(fallback)'} -> n/a"
    return (f"{c['f']}{'' if ok else '(fallback)'} -> {c['med']:.1f} % med "
            f"({c['mn']:.1f}-{c['mx']:.1f}) viol {c['viol']}")


def show_dE(dE):
    if dE == "same":
        return "same clock"
    if dE is None:
        return "n/a"
    return (f"{dE['mean']:+.2f} mJ [t {dE['t_lo']:+.2f}, {dE['t_hi']:+.2f}] "
            f"[boot {dE['b_lo']:+.2f}, {dE['b_hi']:+.2f}] {dE['pos']}/{dE['n']} (p={dE['sign_p']:.3f}) "
            f"= {dE['rel']:+.2f} % of E_blind {dE['Eb']:.1f}")


def print_rows(tag, B, fits, info, rows, log=print):
    hzc = info["aware"]["hz"]
    log(f"  [{tag}] fit cells @{hzc:.0f} Hz: " + "; ".join(
        f"{arm} {info[arm]['cells'][B['lo']]:.0f}/{info[arm]['cells'][B['fmax']]:.0f} us "
        f"k={fits[arm][0]:.0f} b={fits[arm][1]:.0f}" for arm in ("blind", "aware")))
    for arm in ("blind", "aware"):
        log(f"        {arm:5s} T(f)*r us: " + " ".join(f"{f}:{that(fits[arm], f):.0f}" for f in B["grid"]))
    for r in rows:
        log(f"    D={r['D']:5.1f} | blind {show_cell(r['blind'], r['okb']):46s} | "
            f"aware {show_cell(r['aware'], r['oka']):46s} | dE {show_dE(r['dE'])}")


def ci_spans_zero(dE):
    return isinstance(dE, dict) and dE["t_lo"] < 0 < dE["t_hi"]


def run_baseline_states(sessions, clocks):
    """Per-run trimmed-mean VDD_IN [mW] at the given clocks, pooled over sessions, split at
    the largest gap: returns (low-state mean, high-state mean, n_low, n_high).  Used only to
    document a two-state supply baseline when a representative cell's dE CI spans zero."""
    P = []
    for s in sessions:
        for f in clocks:
            for p in s["runs"].get(f, {}):
                P.append(s["E"][f][p] * s["runs"][f][p]["hz"])   # E/inf * rate = mean VDD_IN mW
    P = np.sort(np.asarray(P))
    gaps = np.diff(P)
    i = int(np.argmax(gaps))
    lo, hi = P[:i + 1], P[i + 1:]
    return float(lo.mean()), float(hi.mean()), len(lo), len(hi)


def nano_vit_facts(B, rows):
    """Grid summary of the Nano ViT-Small session (Table III note, main text Sec. IV, Supp. Nano
    ViT-Small table), with its structural expectations asserted: deadlines with a violating run
    per arm, the blind deadlines violated in every run and in some runs, and the aware knife-edge
    at the SKU's top clock."""
    n = len(rows)
    assert all(r["blind"]["has_data"] and r["aware"]["has_data"] and r["okb"] and r["oka"] for r in rows)
    bfull = [r for r in rows if viol0(r["blind"]) == 10]
    bpart = [r for r in rows if 1 <= viol0(r["blind"]) < 10]
    bviol = [r for r in rows if viol0(r["blind"]) >= 1]
    aviol = [r for r in rows if viol0(r["aware"]) >= 1]
    knife = rows[0]
    assert knife["fa"] == B["fmax"] and viol0(knife["aware"]) >= 1, "expected aware knife-edge at the top clock"
    assert all(viol0(r["aware"]) < 10 for r in aviol), "aware arm fails a cell outright"
    others = [r for r in aviol if r is not knife]
    return dict(n=n, bfull=bfull, bpart=bpart, bviol=bviol, aviol=aviol, knife=knife, others=others)


def mnv2_nano_facts(rows):
    """Blind cells with a violating run on the Nano MobileNetV2 grid as (row, runs over tau per
    session); asserts that the aware arm never violates."""
    bv = [(r, r["blind"]["viol"]) for r in rows if viol0(r["blind"]) >= 1]
    av = [r for r in rows if viol0(r["aware"]) >= 1]
    assert not av, "Nano MobileNetV2 aware arm violates"
    return bv


# ----------------------------------------------------------------- build ----
def build(log=print, rate_check=True):
    """Score every entry.  rate_check=True takes the refit cells as the headline only when they were
    profiled at the deployment rate (otherwise the original cells, the refit printed as an extra
    evaluation); rate_check=False takes them whenever they are complete.  log() receives the
    diagnostic lines.  Returns dict(entries=[...], nsess, rate_matched, orig_only) with, per entry:
    B, sessions, head_tag, head_rows, rep, ci (two-state baseline or None), nano_vit (facts or
    None), mnv2_nano (blind violations or None)."""
    out, nsess, rate_matched, orig_only = [], {}, [], []
    for B in ENTRIES:
        sessions = [s for s in (load_session(n, B["hz"]) for n in B["sessions"]) if s]
        assert sessions, f"no session for {B['label']} under results/govrep_rev"
        nsess[B["key"]] = len(sessions)
        log(f"\n##### {B['label']} {B['hz']:.0f} Hz: {len(sessions)} session(s)")
        for s in sessions:
            log(f"  session {s['name']}: clocks/pairs {s['pairs']}, {s['t0']} .. {s['t1']} UTC")
        ofits, oinfo = original_fits(B)
        rfits, rinfo = refit_fits(B)
        off_rate = rfits is not None and abs(rinfo["aware"]["hz"] - B["hz"]) > 1e-6
        if rfits is not None and not (rate_check and off_rate):
            head_tag, head_fits, head_info = "rate-matched", rfits, rinfo
            rate_matched.append(B["label"])
        else:
            head_tag, head_fits, head_info = "original", ofits, oinfo
            orig_only.append(B["label"])
        head_rows = evaluate(B, head_fits, sessions)
        print_rows(head_tag + " (headline)", B, head_fits, head_info, head_rows, log)
        if head_tag == "rate-matched":
            print_rows("original (second evaluation of the same runs)", B, ofits, oinfo,
                       evaluate(B, ofits, sessions), log)
        elif rfits is not None:
            print_rows(f"{rinfo['aware']['hz']:.0f} Hz re-profile cells, not at the {B['hz']:.0f} Hz deployment "
                       "rate (extra evaluation of the same runs)", B, rfits, rinfo, evaluate(B, rfits, sessions), log)
        rep = representative(head_rows)
        assert rep["blind"]["has_data"] and rep["aware"]["has_data"], f"{B['label']}: representative cell lacks data"
        log(f"  => representative row: D={rep['D']} blind {rep['fb'] if rep['okb'] else 'declines'} / aware {rep['fa']}")
        ci = None
        if ci_spans_zero(rep["dE"]):
            # Document, do not subset: print the per-pair energies and the two-state split.
            fb, fa = rep["fb"], rep["fa"]
            for f in (fb, fa):
                for s in sessions:
                    if f in s["E"]:
                        log(f"     E/inf {f} MHz ({s['name']}): " + " ".join(f"{s['E'][f][p]:6.1f}" for p in sorted(s["E"][f])))
            lo, hi, nlo, nhi = run_baseline_states(sessions, (fb, fa))
            log(f"     VDD_IN baseline states over the {nlo + nhi} runs at {fb}/{fa} MHz: "
                f"{lo:.0f} mW (n={nlo}) vs {hi:.0f} mW (n={nhi}), gap {hi - lo:.0f} mW")
            assert B["key"] == "vit_nano", f"{B['label']}: dE CI spans zero -- caption note is written for Nano ViT-Small only"
            assert nlo >= 3 and nhi >= 3 and hi - lo > 500, "two-state baseline expectation not met; rewrite the caption note"
            ci = dict(lo=lo, hi=hi, nlo=nlo, nhi=nhi)
        out.append(dict(B=B, sessions=sessions, head_tag=head_tag, head_rows=head_rows, rep=rep, ci=ci,
                        nano_vit=nano_vit_facts(B, head_rows) if B["key"] == "vit_nano" else None,
                        mnv2_nano=mnv2_nano_facts(head_rows) if B["key"] == "mnv2_nano" else None))
    return dict(entries=out, nsess=nsess, rate_matched=rate_matched, orig_only=orig_only)


# ------------------------------------------------------------ plain text ----
def pm(x):
    return "100" if x >= 99.95 else f"{x:.1f}"


def miss_txt(c):
    if not c["has_data"]:
        return "n/a"
    ks = ", ".join("--" if k is None else f"{k}/10" for k in c["viol"])
    rng = "" if c["mn"] == c["mx"] else f" [{pm(c['mn'])}-{pm(c['mx'])}]"
    return f"{pm(c['med'])}{rng} ({ks})"


def dE_txt(dE):
    if dE == "same":
        return "0 (same f)"
    if dE is None:
        return "n/a"
    return f"{dE['mean']:+.1f} [{dE['t_lo']:.1f}, {dE['t_hi']:.1f}]"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--brief", action="store_true", help="skip the per-entry grid diagnostics; print the table and notes only")
    a = ap.parse_args()
    res = build(log=(lambda *_: None) if a.brief else print, rate_check=True)
    print("\n## Table III, vision rows: the tightest pre-stated deadline at which the aware arm admits a clock\n")
    print("| workload (SKU) | D (ms) | blind f* | blind miss % med [min-max] (runs > tau) | aware f* | "
          "aware miss % med [min-max] (runs > tau) | dE mJ/inf [paired-t 95 % CI] | fits | sessions |")
    print("|---|---:|---|---|---|---|---|---|---:|")
    for e in res["entries"]:
        r, B = e["rep"], e["B"]
        blind = f"{r['fb']}" if r["okb"] else "declines"
        bmiss = miss_txt(r["blind"]) if r["okb"] else f"({r['fb']}: {miss_txt(r['blind'])})"
        print(f"| {B['label']} | {r['D']:.1f} | {blind} | {bmiss} | {r['fa']} | {miss_txt(r['aware'])} | "
              f"{dE_txt(r['dE'])} | {e['head_tag']} | {len(e['sessions'])} |")
    print(f"\nheadline fits: rate-matched for {', '.join(res['rate_matched']) or 'none'}; "
          f"original (deployment-rate) cells for {', '.join(res['orig_only']) or 'none'}")
    for e in res["entries"]:
        r, B = e["rep"], e["B"]
        if isinstance(r["dE"], dict):
            d = r["dE"]
            print(f"{B['label']} D={r['D']:.1f}: dE {d['mean']:+.2f} mJ [t {d['t_lo']:+.2f}, {d['t_hi']:+.2f}], "
                  f"{d['pos']}/{d['n']} pairs positive, {d['rel']:+.2f} % of E_blind {d['Eb']:.1f} mJ")
        if e["ci"]:
            c = e["ci"]
            print(f"  {B['label']}: the dE interval spans zero; run powers at {r['fb']}/{r['fa']} MHz split into "
                  f"{c['lo']:.0f} mW (n={c['nlo']}) and {c['hi']:.0f} mW (n={c['nhi']}), {c['hi'] - c['lo']:.0f} mW apart")
        if e["mnv2_nano"] is not None:
            bv = e["mnv2_nano"]
            print(f"  {B['label']}: blind cells with a violating run: "
                  + ("; ".join(f"D={x['D']:.1f} ms runs > tau per session {v}" for x, v in bv) or "none")
                  + "; the aware arm never violates")
        if e["nano_vit"]:
            F = e["nano_vit"]
            k = F["knife"]
            full_d = ", ".join("%.1f" % x["D"] for x in F["bfull"])
            full_med = ", ".join(pm(x["blind"]["med"]) for x in F["bfull"])
            part_d = ", ".join("%.1f" % x["D"] for x in F["bpart"])
            part_k = (min(viol0(x["blind"]) for x in F["bpart"]), max(viol0(x["blind"]) for x in F["bpart"]))
            others = "".join("; %.1f ms %d/10 (max %s %%)" % (x["D"], viol0(x["aware"]), pm(x["aware"]["mx"]))
                             for x in F["others"])
            print(f"  {B['label']}, all {F['n']} deadlines {B['D'][0]:.1f}-{B['D'][-1]:.1f} ms: blind violates at "
                  f"{len(F['bviol'])} of {F['n']} -- every run at {full_d} ms (median {full_med} %), "
                  f"{part_k[0]}-{part_k[1]} of 10 runs at {part_d} ms; aware violates at {len(F['aviol'])}: "
                  f"{k['D']:.1f} ms at the top clock {k['fa']} MHz (median {pm(k['aware']['med'])} %, max "
                  f"{pm(k['aware']['mx'])} %, {viol0(k['aware'])}/10 runs > tau){others}; "
                  f"the other {F['n'] - len(F['aviol'])} aware cells are clean")
    q = qwengov_2k.cell_2k(29, qwengov_2k.picks(29)[1])
    print(f"\nQwen-1.5B (NX), D = 29 ms/tok, aware {q['f']} MHz: 2,000-token miss {q['mean_pct']:.1f} "
          f"[{q['ub_pct']:.1f}] % (exact {q['mean_pct']:.3f} [{q['ub_pct']:.3f}], {q['n']} runs; as Table V, "
          "campaign/qwengov_2k.py).  Other Qwen cells: campaign/analyze_qwen_deploy.py (1.5B), "
          "campaign/analyze_breadth.py (3B).")


if __name__ == "__main__":
    main()

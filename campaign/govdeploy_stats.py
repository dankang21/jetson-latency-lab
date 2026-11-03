#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Table II (deployed MobileNetV2 governor at EMC 2133 MHz): every panel, cell by cell.

Both arms are scored under the deployed two-cell rule over the pre-stated deadline grids
(docs/protocols/E1_E2_protocol.md, frozen 2026-09-14) on the interleaved E1 sessions under
results/govrep_rev; rule, fits, loader and cell scoring are campaign/govdeploy_common.py:
  panels, in table order
    NX (33 Hz), rate-matched 33 Hz-cell fits (results/refit33/nx; headline)
    Nano (50 Hz), 50 Hz-cell fits (results/nano_grid/mobilenet; profiled at the deployment rate)
    NX, original 8 Hz-cell fits (results/nx_validation; second evaluation of the same runs)
  (--nano-refit33 adds the Nano under the 33 Hz re-profile cells of results/refit33/nano, which
  are not at the 50 Hz deployment rate, as an extra panel before the Nano one).
Per panel: the endpoint cells and fits, T(f)*r over the grid, and per deadline D the blind and
aware picks (a declined arm falls back to the top clock, shown in parentheses), the per-run miss
% (pairs 1..10, per session), its median [min-max] over all pooled runs and the runs above
tau = 2 % per session ('--' where a session did not run the clock), and dE = aware - blind
calibrated VDD_IN energy per inference, paired by pair index with the pairs of all sessions
pooled: mean, paired-t 95 % CI, paired bootstrap 95 % CI, positive pairs and sign-test p, and
mean dE / mean E_blind (t quantiles from a fixed table, so no interval depends on scipy).  The
table blocks print the cells as the article rounds them (miss % to one decimal, '100' from 99.95
on; dE to 0.1 mJ).  The per-session blocks give each session's median miss % and runs over tau
per cell with, per session and arm, the deadlines at which a run violates (main text Sec. IV and
Fig. 1 count them), and for every contrasting cell each session's mean dE and positive pairs, the
spread of the session means and the t interval over the session means (sessions - 1 df), which
the Table II note quotes.  Last, the energy per inference by clock (all runs, and per session;
main text Sec. III quotes the NX range) and the response-time p50 / p99 per clock and session
(pooled over the session's runs; main text Sec. IV quotes the 816 MHz p99 of the third session).

Read-only on results/ (python3 + numpy; no scipy); prints to stdout, writes nothing.
Usage: govdeploy_stats.py [--nano-refit33] [--brief]
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from govdeploy_common import (BOARDS, original_fits, refit_fits, that, load_session,   # noqa: E402
                              evaluate, session_dE)


# ----------------------------------------------------------------- build ----
def build(nano_refit33=False, log=print):
    """Score every Table II panel.  log() receives the per-panel diagnostic lines (fits, picks,
    cells, per-run miss %).  Returns dict(panels=[...] in table order, nsess={board: n},
    nx_variants=[tag, ...], na=[(clock, SKU name, clocks run), ...]) where each panel is
    dict(board, tag ('rate' | 'orig'), hz_cells, refit (a refit was scored for the board),
    rows, sessions, fits, info)."""
    created, nsess, nx_variants, na, footmarks = [], {}, [], [], []
    for board in ("nx", "nano"):
        B = BOARDS[board]
        sessions = [s for s in (load_session(n, B["hz"]) for n in B["sessions"]) if s]
        assert sessions, f"no {board} session under results/govrep_rev"
        nsess[board] = len(sessions)
        log(f"\n##### {B['name']} MobileNetV2 {B['hz']:.0f} Hz: {len(sessions)} session(s)")
        for s in sessions:
            log(f"  session {s['name']}: clocks/pairs {s['pairs']}, {s['t0']} .. {s['t1']} UTC")
        ofits, oinfo = original_fits(B)
        # the Nano refit dir holds 33 Hz cells (50 Hz deployment): scored only on request
        rfits, rinfo = refit_fits(B) if (board == "nx" or nano_refit33) else (None, None)
        variants = []
        if rfits is not None:
            variants.append(("rate", rfits, rinfo))
        variants.append(("orig", ofits, oinfo))
        for tag, fits, info in variants:
            hzc = info["aware"]["hz"]
            log(f"  [{tag}] fit cells @{hzc:.0f} Hz: " + "; ".join(
                f"{arm} {info[arm]['cells'][B['lo']]:.0f}/{info[arm]['cells'][B['fmax']]:.0f} us "
                f"k={fits[arm][0]:.0f} b={fits[arm][1]:.0f}" for arm in ("blind", "aware")))
            for arm in ("blind", "aware"):
                log(f"        {arm:5s} T(f)*r us: " + " ".join(f"{f}:{that(fits[arm], f):.0f}" for f in B["grid"]))
            rows = evaluate(B, fits, sessions)
            for r in rows:
                def show(c, ok):
                    if not c["has_data"]:
                        return f"{c['f']}{'' if ok else '(fallback)'} -> n/a"
                    return (f"{c['f']}{'' if ok else '(fallback)'} -> {c['med']:.1f} % med "
                            f"({c['mn']:.1f}-{c['mx']:.1f}) viol {c['viol']}")
                dE = r["dE"]
                dEs = ("same clock" if dE == "same" else "n/a" if dE is None else
                       f"{dE['mean']:+.2f} mJ [t {dE['t_lo']:+.2f}, {dE['t_hi']:+.2f}] "
                       f"[boot {dE['b_lo']:+.2f}, {dE['b_hi']:+.2f}] {dE['pos']}/{dE['n']} (p={dE['sign_p']:.3f}) "
                       f"= {dE['rel']:+.2f} % of E_blind {dE['Eb']:.1f}")
                log(f"    D={r['D']:5.1f} | blind {show(r['blind'], r['okb']):48s} | "
                    f"aware {show(r['aware'], r['oka']):48s} | dE {dEs}")
            log("    per-run miss % (pairs 1..10, per session):")
            for r in rows:
                for arm in ("blind", "aware"):
                    c = r[arm]
                    for s, l in zip(sessions, c["per"]):
                        if l is not None:
                            log(f"      D={r['D']:5.1f} {arm:5s} {c['f']:4d} {s['name']:16s}: "
                                + " ".join(f"{x:5.1f}" for x in l))
            created.append(dict(board=board, tag=tag, hz_cells=hzc, refit=rfits is not None, rows=rows,
                                sessions=sessions, fits=fits, info=info))
            # picked clocks no session ran (first occurrence over the panels)
            for r in rows:
                for arm in ("blind", "aware"):
                    c = r[arm]
                    if not c["has_data"] and c["f"] not in footmarks:
                        footmarks.append(c["f"])
                        na.append((c["f"], B["name"], sorted({f for s in sessions for f in s["runs"]})))
            if board == "nx":
                nx_variants.append(tag)
    # table order: NX headline, Nano panel(s), then the NX second evaluation
    order = list(range(len(created)))
    if nsess and len(created) >= 3 and created[0]["board"] == "nx" and created[0]["tag"] == "rate":
        nano = [i for i, p in enumerate(created) if p["board"] == "nano"]
        nx_second = [i for i, p in enumerate(created) if p["board"] == "nx" and p["tag"] != "rate"]
        order = [0] + nano + nx_second
    return dict(panels=[created[i] for i in order], nsess=nsess, nx_variants=nx_variants, na=na)


# ------------------------------------------------------------ plain text ----
def panel_title(p):
    """Plain-text panel heading (the Table II panel titles without markup)."""
    hzc, B = p["hz_cells"], BOARDS[p["board"]]
    if p["board"] == "nx":
        if p["tag"] == "rate":
            return f"NX (33 Hz), rate-matched {hzc:.0f} Hz-cell fits (headline)"
        if not p["refit"]:
            return f"NX (33 Hz), {hzc:.0f} Hz-cell fits"
        return f"NX, original {hzc:.0f} Hz-cell fits (second evaluation of the same runs)"
    if p["tag"] == "rate" and abs(hzc - B["hz"]) > 1e-6:
        return f"Nano (50 Hz), {hzc:.0f} Hz re-profile cells (not rate-matched)"
    if p["tag"] == "rate":
        return f"Nano (50 Hz), rate-matched {hzc:.0f} Hz-cell fits (headline)"
    if not p["refit"]:
        return f"Nano (50 Hz), {hzc:.0f} Hz-cell fits (profiled at the deployment rate)"
    return f"Nano, original {hzc:.0f} Hz-cell fits"


def pm(x):
    """Miss % to one decimal; '100' when every cycle missed (99.95+)."""
    return "100" if x >= 99.95 else f"{x:.1f}"


def miss_txt(c):
    """'med [min-max] (k/10 per session)'; the bracket collapses when min = max; '--' for a session
    that did not run the clock; 'n/a' when no session ran it."""
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


def row_txt(r):
    """One table row: blind / aware pick and miss, dE; a declined arm in parentheses."""
    blind = f"{r['fb']}" if r["okb"] else "declines"
    bmiss = miss_txt(r["blind"]) if r["okb"] else f"({r['fb']}: {miss_txt(r['blind'])})"
    if r["oka"]:
        aware, amiss, dE = f"{r['fa']}", miss_txt(r["aware"]), dE_txt(r["dE"])
    else:
        aware, amiss = "declines", f"({r['fa']}: {miss_txt(r['aware'])})"
        dE = f"({dE_txt(r['dE'])})" if r["dE"] not in ("same", None) else dE_txt(r["dE"])
    return [f"{r['D']:.1f}", blind, bmiss, aware, amiss, dE]


def print_table(title, rows_txt, head):
    print(f"\n### {title}\n")
    print("| " + " | ".join(head) + " |")
    print("|" + "|".join("---:" if i == 0 else "---" for i in range(len(head))) + "|")
    for cells in rows_txt:
        print("| " + " | ".join(cells) + " |")


HEAD = ["D (ms)", "blind f*", "blind miss % med [min-max] (runs > tau)", "aware f*",
        "aware miss % med [min-max] (runs > tau)", "dE mJ/inf [paired-t 95 % CI]"]


def print_session_misses(p):
    """Per-session view of one panel's miss cells: each session's median miss % and runs over tau per
    cell, and per session and arm the deadlines with a violating run among those whose pick the session
    ran (a declined arm is listed with its fallback, not counted as a violation)."""
    names = [s["name"] for s in p["sessions"]]
    print(f"\n### {panel_title(p)}: per session\n")
    print("| D (ms) | blind f* | blind: median % (runs > tau) per session | aware f* | aware: median % (runs > tau) per session |")
    print("|---:|---|---|---|---|")

    def per(c):
        return "; ".join(f"{n} --" if l is None else f"{n} {pm(float(np.median(l)))} ({k}/{len(l)})"
                         for n, l, k in zip(names, c["per"], c["viol"] if c["has_data"] else [None] * len(names)))
    for r in p["rows"]:
        b = f"{r['fb']}" if r["okb"] else f"declines ({r['fb']})"
        a = f"{r['fa']}" if r["oka"] else f"declines ({r['fa']})"
        print(f"| {r['D']:.1f} | {b} | {per(r['blind'])} | {a} | {per(r['aware'])} |")
    for arm, ok in (("blind", "okb"), ("aware", "oka")):
        pooled = set()
        for j, n in enumerate(names):
            ev = [r for r in p["rows"] if r[ok] and r[arm]["per"][j] is not None]
            vio = [r["D"] for r in ev if r[arm]["viol"][j] >= 1]
            full = [r["D"] for r in ev if r[arm]["viol"][j] == len(r[arm]["per"][j])]
            dec = [r for r in p["rows"] if not r[ok]]
            pooled |= set(vio)
            print(f"{arm} arm, session {n}: a violating run at {len(vio)} of the {len(ev)} admitted deadlines whose pick it ran"
                  + (f" ({', '.join(f'{d:.1f}' for d in vio)} ms; every run over tau at {len(full)}"
                     + (f": {', '.join(f'{d:.1f}' for d in full)} ms" if full else "") + ")" if vio else "")
                  + "".join(f"; declines {r['D']:.1f} ms (fallback {r[arm]['f']}: "
                            + ("not run" if r[arm]["per"][j] is None else f"{r[arm]['viol'][j]}/{len(r[arm]['per'][j])} runs over tau")
                            + ")" for r in dec))
        adm = [r for r in p["rows"] if r[ok] and r[arm]["has_data"]]
        top = max((r[arm]["mx"] for r in adm), default=None)
        print(f"{arm} arm, pooled over the sessions: a violating run at {len(pooled)} of the {len(adm)} admitted deadlines"
              + (f" ({', '.join(f'{d:.1f}' for d in sorted(pooled))} ms)" if pooled else "")
              + ("" if top is None else f"; largest run at an admitted pick {top:.1f} %"))


def print_sessions(p):
    """Per-session paired energy differences of every contrasting cell of one panel."""
    print(f"\n### {panel_title(p)}: dE per session (aware - blind, mJ/inf)\n")
    names = [s["name"] for s in p["sessions"]]
    print("| D (ms) | fb -> fa | pooled n | pooled mean [paired-t 95 % CI] | pairs > 0 | "
          + " | ".join(f"{n}: n, mean, >0" for n in names)
          + " | spread of session means | t interval over session means (df) |")
    print("|---:|---|---:|---|---:|" + "---|" * len(names) + "---:|---|")
    sums = []
    for r in p["rows"]:
        if not isinstance(r["dE"], dict):
            continue
        sd = session_dE(p["sessions"], r["fb"], r["fa"])
        by = {x["name"]: x for x in sd["sessions"]}
        cols = [f"{by[n]['n']}, {by[n]['mean']:+.2f}, {by[n]['pos']}/{by[n]['n']}" if n in by else "--" for n in names]
        iv = f"[{sd['lo']:+.2f}, {sd['hi']:+.2f}] ({sd['df']})" if "lo" in sd else "-- (one session)"
        dE = r["dE"]
        print(f"| {r['D']:.1f} | {r['fb']} -> {r['fa']}{'' if r['oka'] else ' (aware declines)'} | {dE['n']} | "
              f"{dE['mean']:+.2f} [{dE['t_lo']:+.2f}, {dE['t_hi']:+.2f}] | {dE['pos']}/{dE['n']} | "
              + " | ".join(cols) + f" | {sd['spread']:.2f} | {iv} |")
        sums.append((r, sd))
    if not sums:
        print("(no contrasting cell)")
        return
    every_pos = all(sd["all_pos"] for _, sd in sums)
    worst = max(sums, key=lambda t: t[1]["spread"])
    excl = [sd["lo"] > 0 or sd["hi"] < 0 for _, sd in sums if "lo" in sd]
    print(f"\nover the {len(sums)} contrasting cells: every pair of every session positive: "
          f"{'yes' if every_pos else 'no'}; largest spread of the session means {worst[1]['spread']:.2f} mJ "
          f"(D = {worst[0]['D']:.1f} ms); t interval over the session means excludes zero in "
          f"{sum(excl)} of {len(excl)} cells with two or more sessions")


# ------------------------------------------------------------------ main ----
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nano-refit33", action="store_true",
                    help="also score the Nano under the results/refit33/nano cells (33 Hz re-profile, "
                         "not the 50 Hz deployment rate) as an extra panel")
    ap.add_argument("--brief", action="store_true",
                    help="skip the per-panel diagnostics (fits, raw cells, per-run miss %%); print the tables only")
    a = ap.parse_args()
    res = build(nano_refit33=a.nano_refit33, log=(lambda *_: None) if a.brief else print)
    print("\n## Table II, panel by panel (cells rounded as in the article)")
    seen = {}
    for p in res["panels"]:
        seen.setdefault(p["board"], [s["name"] for s in p["sessions"]])
    print("sessions: " + "; ".join(f"{BOARDS[b]['name']} {len(v)} ({', '.join(v)})" for b, v in seen.items()))
    for f, name, ran in res["na"]:
        print(f"n/a: {f} MHz is not in the {name} run set ({', '.join(map(str, ran))} MHz)")
    for p in res["panels"]:
        B = BOARDS[p["board"]]
        fb, fa = p["fits"]["blind"], p["fits"]["aware"]
        title = panel_title(p)
        print_table(title, [row_txt(r) for r in p["rows"]], HEAD)
        print(f"\nfits ({p['hz_cells']:.0f} Hz endpoint cells): blind k={fb[0]:.0f} b={fb[1]:.0f}, "
              f"aware k={fa[0]:.0f} b={fa[1]:.0f}; grid {B['grid'][0]}-{B['grid'][-1]} MHz, top clock {B['fmax']} MHz")
    print("\n## Miss cells by session")
    for p in res["panels"]:
        print_session_misses(p)
    print("\n## Paired energy differences by session")
    for p in res["panels"]:
        print_sessions(p)
    print("\n## Energy per inference by clock (trimmed-mean VDD_IN / release rate, mJ/inf; every run of every session)")
    done = set()
    for p in res["panels"]:
        if p["board"] in done:
            continue
        done.add(p["board"])
        ss = p["sessions"]
        print(f"\n{BOARDS[p['board']]['name']} ({BOARDS[p['board']]['hz']:.0f} Hz):")
        for f in sorted({f for s in ss for f in s["E"]}):
            E = [s["E"][f][q] for s in ss if f in s["E"] for q in s["E"][f]]
            per = ", ".join(f"{s['name']} {sum(s['E'][f].values()) / len(s['E'][f]):.1f}" for s in ss if f in s["E"])
            print(f"  {f:4d} MHz: {sum(E) / len(E):6.1f} (runs {len(E)}, {min(E):.1f}-{max(E):.1f}; {per})")
    print("\n## Response time by clock and session (response_us of the session's runs pooled: p50 / p99, ms)")
    done = set()
    for p in res["panels"]:
        if p["board"] in done:
            continue
        done.add(p["board"])
        ss = p["sessions"]
        print(f"\n{BOARDS[p['board']]['name']} ({BOARDS[p['board']]['hz']:.0f} Hz):")
        for f in sorted({f for s in ss for f in s["runs"]}):
            cells = []
            for s in ss:
                if f in s["runs"]:
                    v = np.concatenate([s["runs"][f][q]["v"] for q in sorted(s["runs"][f])])
                    cells.append(f"{s['name']} {np.percentile(v, 50) / 1e3:.3f} / {np.percentile(v, 99) / 1e3:.3f}")
            print(f"  {f:4d} MHz: " + "; ".join(cells))


if __name__ == "__main__":
    main()

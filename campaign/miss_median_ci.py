#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Per-run miss rates of the solo vision deployments: for every arm, deadline and session of main-text
Table II and Supp. Tables S3 and S4, the median of the session's ten per-run miss % and a distribution-free
interval for the median of the session's run distribution (Supp. Tables S30 and S31).

Each observation is one 1,000-cycle run; its miss % is the share of the run's cycles whose response time
exceeds the deadline (response_us > D), computed by the code that prints those tables
(campaign/govdeploy_common.py, campaign/govdeploy_stats.py, campaign/e1b_stats.py,
campaign/nano_vit_grid.py), with the same picks.  For the n runs of one arm at one deadline in one
session (n = 10 throughout), the interval is [x_(k), x_(n+1-k)] of the sorted values, with the largest k
for which 1 - 2 P(Bin(n, 1/2) <= k - 1) >= 0.95; for n = 10, k = 2 and the coverage is
1 - 2 (1 + 10) / 1024 = 97.85 %.  If the runs of a session are independent draws from one distribution,
the interval contains the median of that distribution with at least that probability; discrete values
and ties are allowed.  The intervals are pointwise: they are not simultaneous over deadlines or arms,
sessions are not pooled, and cells with the same clock in a session share runs, whichever arm or panel picks it.  They describe the per-run miss
rate of one session condition; they are neither an interval for the mean miss rate nor a bound on the
chance that a run exceeds the 2 % budget, and the budget counts of the tables (runs over tau) are not
derived from them.  A declined arm is scored at its top-clock fallback, marked 'fallback'.

Read-only on results/ (python3 + numpy; no scipy); prints to stdout, writes nothing.
Usage: miss_median_ci.py [--tex]      (--tex: the LaTeX rows of Supp. Tables S30 and S31, to stdout)
"""
import argparse
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import govdeploy_stats as GS   # noqa: E402  Table II panels (NX rate-matched, Nano, NX original 8 Hz cells)
import nano_vit_grid as NV     # noqa: E402  Supp. Table S4
import e1b_stats as EB         # noqa: E402  Supp. Table S3


def order_k(n, level=0.95):
    """Largest k >= 1 with 1 - 2 P(Bin(n, 1/2) <= k - 1) >= level, and that coverage; None if n is too small."""
    best = None
    for k in range(1, n // 2 + 1):
        tail = sum(math.comb(n, i) for i in range(k)) / 2.0 ** n
        cov = 1.0 - 2.0 * tail
        if cov >= level:
            best = (k, cov)
    return best


def summarise(vals):
    """dict(n, med, lo, hi, cov) for one session's per-run miss %, or None without data."""
    if not vals:
        return None
    x = sorted(float(v) for v in vals)
    n = len(x)
    kc = order_k(n)
    out = dict(n=n, med=float(np.median(x)))
    if kc:
        k, cov = kc
        out.update(lo=x[k - 1], hi=x[n - k], cov=cov, k=k)
    return out


def fmt_pct(v):
    """miss % as the tables round it (one decimal, '100' from 99.95 on, '0' for no miss); a median of ten
    runs of 1,000 cycles may end in .x5 and keeps its second decimal."""
    if v >= 99.95:
        return "100"
    if v == 0:
        return "0"
    s = f"{v:.2f}"
    return s[:-1] if s.endswith("0") else s


def cell_text(s, tex=False):
    if s is None:
        return "--" if tex else "not run"
    if "lo" not in s:
        return fmt_pct(s["med"])
    return f"{fmt_pct(s['med'])} [{fmt_pct(s['lo'])}, {fmt_pct(s['hi'])}]"


def mark(f, ok, tex=False):
    if ok:
        return str(f)
    return f"{f}$^\\dagger$" if tex else f"{f} (fallback)"


# ------------------------------------------------------------------ panels ----
def table2_panels():
    """Main-text Table II: NX rate-matched 33 Hz-cell fits (headline), Nano 50 Hz-cell fits, NX original
    8 Hz-cell fits (second evaluation of the same runs), each with its sessions in session order."""
    res = GS.build(log=lambda *a, **k: None)
    out = []
    for p in res["panels"]:
        B = GS.BOARDS[p["board"]]
        title = (f"{B['name']} MobileNetV2, "
                 + ("rate-matched 33 Hz-cell fits (headline)" if p["tag"] == "rate" and p["board"] == "nx"
                    else ("original 8 Hz-cell fits (second evaluation of the same runs)" if p["board"] == "nx"
                          else f"{p['hz_cells']:.0f} Hz-cell fits")))
        rows = []
        for r in p["rows"]:
            row = dict(D=r["D"])
            for arm, ok in (("blind", r["okb"]), ("aware", r["oka"])):
                c = r[arm]
                row[arm] = dict(f=c["f"], ok=ok, sess=[summarise(l) if l is not None else None for l in c["per"]])
            rows.append(row)
        out.append(dict(title=title, sessions=[s["name"] for s in p["sessions"]], rows=rows))
    return out


def breadth_panels():
    """Supp. Table S3: NX ViT-Small, YOLOv8s, ResNet-50, rate-matched endpoint refits (one session each)."""
    out = []
    for model in ("vit", "yolov8s", "resnet50"):
        M = EB.MODELS[model]
        runs = EB.load_runs(model)
        fits, info = EB.refit33_fits(model)
        assert fits is not None, f"rate-matched refit cells missing for {model}"
        rows = []
        for D in M["grid"]:
            D_us = D * 1000.0
            row = dict(D=D)
            for arm in ("blind", "aware"):
                f, ok = EB.pick(fits[arm], D_us)
                c = EB.arm_cell(runs, f, ok, D_us)
                row[arm] = dict(f=f, ok=ok, sess=[summarise(c["miss"]) if c.get("has_data") else None])
            rows.append(row)
        out.append(dict(title=f"NX {M['name']}", sessions=[f"nx_{model}"], rows=rows))
    return out


def nanovit_panel():
    """Supp. Table S4: Nano ViT-Small, rate-matched 20 Hz-cell fits (one session)."""
    b = NV.build()
    rows = []
    for r in b["rows"]:
        row = dict(D=r["D"])
        for arm, ok in (("blind", r["okb"]), ("aware", r["oka"])):
            c = r[arm]
            row[arm] = dict(f=c["f"], ok=ok, sess=[summarise(l) if l is not None else None for l in c["per"]])
        rows.append(row)
    return dict(title="Nano ViT-Small", sessions=[b["s"]["name"]], rows=rows)


# ------------------------------------------------------------------ output ----
def print_md(panel):
    ses = panel["sessions"]
    print(f"\n### {panel['title']}  (sessions: {', '.join(ses)})\n")
    head = ["D (ms)", "blind f*"] + [f"blind {s}" for s in ses] + ["aware f*"] + [f"aware {s}" for s in ses]
    print("| " + " | ".join(head) + " |")
    print("|" + "---|" * len(head))
    for r in panel["rows"]:
        cells = [f"{r['D']:.1f}", mark(r["blind"]["f"], r["blind"]["ok"])] + [cell_text(s) for s in r["blind"]["sess"]]
        cells += [mark(r["aware"]["f"], r["aware"]["ok"])] + [cell_text(s) for s in r["aware"]["sess"]]
        print("| " + " | ".join(cells) + " |")


def tex_tabular(panels):
    """The body rows of one supplementary table (Supp. Table S30: the Table II panels; S31: Supp. Tables S3
    and S4): a panel heading row, then per deadline D, the blind pick and its sessions, the aware pick and its
    sessions; a panel with fewer sessions than the widest one leaves the remaining columns empty."""
    NS = max(len(p["sessions"]) for p in panels)
    L = []
    for i, p in enumerate(panels):
        if i:
            L.append("    \\addlinespace")
        L.append(f"    \\multicolumn{{{3 + 2 * NS}}}{{@{{}}l}}{{\\textit{{{p['title']}}}}}\\\\")
        pad = [""] * (NS - len(p["sessions"]))
        for r in p["rows"]:
            cells = [f"{r['D']:.1f}", mark(r["blind"]["f"], r["blind"]["ok"], tex=True)]
            cells += [cell_text(s, tex=True) for s in r["blind"]["sess"]] + pad
            cells += [mark(r["aware"]["f"], r["aware"]["ok"], tex=True)]
            cells += [cell_text(s, tex=True) for s in r["aware"]["sess"]] + pad
            L.append("    " + " & ".join(cells) + " \\\\")
    return "\n".join(L)


def check(panels):
    """Every interval holds its median, every session has ten runs, and the coverage is the stated one.
    Returns (session cells with data, run scores, distinct runs = distinct (session, clock) x 10)."""
    n_cells = n_runs = 0
    distinct = set()
    for p in panels:
        for r in p["rows"]:
            for arm in ("blind", "aware"):
                for name, s in zip(p["sessions"], r[arm]["sess"]):
                    if s is None:
                        continue
                    n_cells += 1
                    n_runs += s["n"]
                    distinct.add((name, r[arm]["f"]))
                    assert s["n"] == 10, (p["title"], r["D"], arm, s["n"])
                    assert s["lo"] <= s["med"] <= s["hi"], (p["title"], r["D"], arm, s)
                    assert abs(s["cov"] - (1 - 2 * 11 / 1024)) < 1e-12
    return n_cells, n_runs, 10 * len(distinct)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tex", action="store_true", help="print the LaTeX tabular rows of Supp. Tables S30 and S31")
    a = ap.parse_args()
    t2 = table2_panels()
    br = breadth_panels()
    nv = nanovit_panel()
    allp = t2 + br + [nv]
    n_cells, n_runs, n_distinct = check(allp)
    k, cov = order_k(10)
    if a.tex:
        print("% Supp. Table S30 (main-text Table II panels)")
        print(tex_tabular(t2))
        print("% Supp. Table S31 (Supp. Tables S3 and S4)")
        print(tex_tabular(br + [nv]))
        return
    print("# Per-run miss %: session median [interval for the median] by arm, deadline and session")
    print(f"interval: [x_({k}), x_({10 + 1 - k})] of the ten sorted per-run values of one session; pointwise coverage "
          f"of the median >= {100 * cov:.2f} % if the runs are independent draws from one distribution; "
          f"not simultaneous; sessions not pooled; cells with the same clock in a session share runs.")
    print(f"{n_cells} session cells with data ({n_runs:,} run scores of {n_distinct} runs); 'not run' = the session did not run that clock; "
          "'fallback' = declined arm scored at the top clock.")
    print("\n## Main-text Table II (Supp. Table S30)")
    for p in t2:
        print_md(p)
    print("\n## Supp. Tables S3 and S4 (Supp. Table S31)")
    for p in br + [nv]:
        print_md(p)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""EMC-axis numbers of the article and its supplement that the other analysis
scripts do not print: the cross-SKU 2133 -> 3199 MHz penalties and the low-clock
collapse (main text Sec. IV, Supp. Sec. S3), the consolidated EMC table (Supp. Table
S23), the TensorRT-vs-ONNX Runtime penalties (Supp. Table S27), the session repeats and
the batch axis (Supp. Sec. S3), the LLM decode axes (Supp. Table S13) and the 28
two-cell refit pairs (main text Sec. IV, Supp. Sec. S3).

Read-only over results/ (python3 + numpy; nothing is written).  Run from anywhere:
    python3 campaign/emc_axis_stats.py [--section NAME ...] [--results DIR]
Sections: xsku, s23, s27, reps, batch, llm, pairs (default: all).

Statistics.  A vision or proxy cell is one harness run JSON (harness/infer_bench.py):
p50 of response_us (from the scheduled release) and of compute_us over the timed loop.
Both are printed; 'resp' and 'comp' below.  A decode cell (tools/slm_decode_bench) has
only the per-token compute_us, printed in both columns.  'n' is the timed iterations
(tokens for decode).  A run whose compute p50 exceeds its release period cannot keep up
with its releases: its response times accumulate a queue over the run, so its response
p50 is not a per-inference latency.  Such cells are flagged BACKLOG.

Data (results/):
  xsku, s23   Nano ONNX Runtime: campaign_g2/partA/emc{204,665,2133,3199}_<w>.json
              (w = mobilenet, vit, proxy = GEMV 12 x 4096^2, cproxyv1 = GEMM 4 x 2048^3,
              cproxyv2 = L2-resident GEMM 64 x 1024^3, slm = Qwen2.5-1.5B Q4_K_M);
              Nano TensorRT fp16: e5_tensorrt/e5_<w>_emc<E>.json;
              NX ONNX Runtime: nx_validation/nx_emc<E>_<w>.json, ViT-Small only at the
              deployment pair, nx_validation/nx_est_emc{2133,3199}_gpu1173_vit.json.
              (665 in a file name is the 665.6 MHz point.)
  reps        parta_reps/ (Nano) and parta_reps_nx/ (NX): s<k>_e{2133,3199}_<w>.json,
              three sessions (campaign/c6_reps.sh).
  batch       batch_axis/mn_e{204,2133,3199}_b{1,4,8}.json (Nano MobileNetV2,
              campaign/c9_batch.sh); per-image and per-batch ratios are the same.
  llm         llm_axes/<tag>_e{2133,3199}_r{1,2,3}.{json,power.csv} (NX, GPU 1173 MHz,
              300 tokens, campaign/c10_llm_axes.sh).  ms/tok = median of the three runs'
              p50; penalty from those medians; energy per token = mean VDD_IN power over
              the run's power log less its first and last 10 % of samples (the slice
              p[n//10:-n//10], as campaign/e1b_stats.py) times the run's median token
              time, averaged over the runs.
  pairs       estimator_break/emc2133_gpu<g>_<w>.json (300 iterations, compute_us p50 as in
              campaign/analyze_estimator.py): T(f) = k/f + b through every pair of the eight
              GPU cells, worst |relative error| on the other six.
"""
import argparse
import csv
import glob
import itertools
import json
import os
import re
import sys

import numpy as np

LAB = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EMCS = (204, 665, 2133, 3199)
NAME = {"mobilenet": "MobileNetV2", "vit": "ViT-Small", "proxy": "GEMV 12x4096^2",
        "cproxyv1": "GEMM 4x2048^3", "cproxyv2": "GEMM 64x1024^3 (L2)",
        "slm": "Qwen2.5-1.5B (ms/tok)"}
WL = ("mobilenet", "vit", "proxy", "cproxyv1", "cproxyv2", "slm")


def emc_label(e):
    return "665.6" if e == 665 else str(e)


class Cell:
    """p50 of response and compute (ms), n, period (ms) of one run JSON."""

    def __init__(self, path):
        d = json.load(open(path))
        self.path = path
        m = d.get("meta", {})
        self.comp = d["compute_us"]["p50"] / 1e3
        self.resp = d["response_us"]["p50"] / 1e3 if "response_us" in d else self.comp
        self.decode = "response_us" not in d
        self.n = m.get("iters", m.get("tokens"))
        self.period = (m["period_us"] / 1e3) if "period_us" in m else None
        self.backlog = self.period is not None and self.comp > self.period

    def get(self, key):
        return self.resp if key == "resp" else self.comp


def load(path):
    return Cell(path) if os.path.isfile(path) else None


def pen(a, b, key):
    """Penalty of cell a against cell b in %, (p50_a / p50_b - 1) * 100."""
    return (a.get(key) / b.get(key) - 1.0) * 100.0


def grid(R):
    """{(sku_runtime, w): {emc: Cell}} for the three blocks of Supp. Table S23."""
    G = {}
    for w in WL:
        G[("Nano ORT", w)] = {e: load(f"{R}/campaign_g2/partA/emc{e}_{w}.json") for e in EMCS}
        if w in ("mobilenet", "vit", "proxy", "cproxyv2"):
            G[("Nano TRT", w)] = {e: load(f"{R}/e5_tensorrt/e5_{w}_emc{e}.json") for e in EMCS}
        if w == "vit":
            G[("NX ORT", w)] = {e: load(f"{R}/nx_validation/nx_est_emc{e}_gpu1173_vit.json")
                                for e in EMCS}
        elif w != "cproxyv1":
            G[("NX ORT", w)] = {e: load(f"{R}/nx_validation/nx_emc{e}_{w}.json") for e in EMCS}
    for k in list(G):
        G[k] = {e: c for e, c in G[k].items() if c is not None}
        if not G[k]:
            del G[k]
    return G


def sec_xsku(R):
    G = grid(R)
    print("== [main Sec. IV; Supp. Sec. S3] 2133 -> 3199 MHz median penalty, (p50 at 2133 / "
          "p50 at 3199 - 1); GPU pinned at the top clock (Nano 1020, NX 1173 MHz) ==")
    print(f"  {'board':5s} {'workload':22s} {'resp':>8s} {'comp':>8s}   n(2133/3199)")
    for board in ("NX", "Nano"):
        for w in WL:
            c = G.get((f"{board} ORT", w))
            if not c or 2133 not in c or 3199 not in c:
                continue
            a, b = c[2133], c[3199]
            print(f"  {board:5s} {NAME[w]:22s} {pen(a, b, 'resp'):+8.2f} {pen(a, b, 'comp'):+8.2f}"
                  f"   {a.n}/{b.n}")
    print("  (decode rows: per-token compute p50 in both columns)")
    print()
    print("== [Supp. Sec. S3] low-clock collapse: p50(204)/p50(665.6) and p50(204)/p50(3199) ==")
    print(f"  {'board':5s} {'workload':22s} {'204/665.6 resp':>15s} {'comp':>6s} "
          f"{'204/3199 resp':>14s} {'comp':>6s}")
    span = {}
    for board in ("Nano", "NX"):
        for w in WL:
            c = G.get((f"{board} ORT", w))
            if not c or not all(e in c for e in EMCS):
                continue
            r1 = [c[204].get(k) / c[665].get(k) for k in ("resp", "comp")]
            r2 = [c[204].get(k) / c[3199].get(k) for k in ("resp", "comp")]
            span.setdefault(board, []).append((r1, r2))
            print(f"  {board:5s} {NAME[w]:22s} {r1[0]:15.2f} {r1[1]:6.2f} {r2[0]:14.2f} {r2[1]:6.2f}")
    for board, rows in span.items():
        for i, k in enumerate(("resp", "comp")):
            lo1, hi1 = min(r[0][i] for r in rows), max(r[0][i] for r in rows)
            lo2, hi2 = min(r[1][i] for r in rows), max(r[1][i] for r in rows)
            print(f"  {board} ({k}): 665.6->204 x{lo1:.2f}-{hi1:.2f}; 204-vs-3199 span "
                  f"x{lo2:.2f}-{hi2:.2f} over {len(rows)} workloads")
    print()


def sec_s23(R):
    G = grid(R)
    print("== [Supp. Table S23] median latency (ms) at each lockable EMC point, ratio to the "
          "3199 MHz cell in parentheses ==")
    for block in ("Nano ORT", "Nano TRT", "NX ORT"):
        print(f"  -- {block} --")
        for w in WL:
            c = G.get((block, w))
            if not c:
                continue
            for key in ("resp", "comp"):
                if key == "comp" and all(x.decode for x in c.values()):
                    continue
                base = c.get(3199)
                cells = []
                for e in EMCS:
                    if e not in c:
                        cells.append(f"{emc_label(e)}: --")
                        continue
                    x = c[e]
                    r = f" ({x.get(key) / base.get(key):.2f})" if base is not None and e != 3199 else ""
                    cells.append(f"{emc_label(e)}: {x.get(key):.3f}{r}")
                print(f"  {NAME[w]:22s} {key:4s} " + "  ".join(cells))
            ns = sorted({x.n for x in c.values()})
            per = sorted({round(x.period, 1) for x in c.values() if x.period})
            flags = [f"{emc_label(e)} MHz (comp {x.comp:.1f} > period {x.period:.1f} ms)"
                     for e, x in c.items() if x.backlog]
            print(f"  {'':22s} n = {'/'.join(map(str, ns))}"
                  + (f", release period {'/'.join(map(str, per))} ms" if per else "")
                  + (f"; BACKLOG at {', '.join(flags)}" if flags else ""))
    print()


def sec_s27(R):
    G = grid(R)
    print("== [Supp. Table S27; Supp. Sec. S8] 2133 -> 3199 MHz penalty (%) under TensorRT fp16 "
          "and ONNX Runtime CUDA fp32, Nano ==")
    print(f"  {'workload':22s} {'TRT resp':>9s} {'TRT comp':>9s} {'ORT resp':>9s} {'ORT comp':>9s}")
    for w in ("mobilenet", "vit", "proxy", "cproxyv2"):
        t, o = G.get(("Nano TRT", w)), G.get(("Nano ORT", w))
        if not t or not o:
            continue
        print(f"  {NAME[w]:22s} {pen(t[2133], t[3199], 'resp'):+9.2f} {pen(t[2133], t[3199], 'comp'):+9.2f}"
              f" {pen(o[2133], o[3199], 'resp'):+9.2f} {pen(o[2133], o[3199], 'comp'):+9.2f}")
    t = G.get(("Nano TRT", "cproxyv2"))
    if t:
        print(f"  L2-resident GEMM under TensorRT: p50 3199 {t[3199].resp:.2f} ms vs 2133 {t[2133].resp:.2f} ms"
              f" (resp), {t[3199].comp:.2f} vs {t[2133].comp:.2f} ms (comp)")
    for w in ("proxy", "cproxyv2"):
        t, o = G.get(("Nano TRT", w)), G.get(("Nano ORT", w))
        if t and 204 in t and o and 204 in o:
            x = t[204]
            print(f"  {NAME[w]} at 204 MHz: TRT resp {x.resp:.1f} ms, comp {x.comp:.1f} ms "
                  f"(x{x.comp / t[3199].comp:.1f} of its 3199 comp; release period {x.period:.1f} ms"
                  f"{', BACKLOG' if x.backlog else ''}); ORT comp {o[204].comp:.1f} ms "
                  f"(x{o[204].comp / o[3199].comp:.1f})")
    print()


def sec_reps(R):
    print("== [Supp. Sec. S3] session repeats: 2133 -> 3199 MHz penalty (%) in each of three "
          "same-day sessions, and its span (max - min, pp) ==")
    for board, sub in (("Nano", "parta_reps"), ("NX", "parta_reps_nx")):
        cells = {}
        for f in glob.glob(f"{R}/{sub}/s*_e*_*.json"):
            m = re.match(r"s(\d+)_e(\d+)_(\w+)\.json$", os.path.basename(f))
            if m:
                cells[(m.group(3), int(m.group(1)), int(m.group(2)))] = Cell(f)
        if not cells:
            print(f"  {board}: no data under results/{sub}")
            continue
        wls = [w for w in WL if any(k[0] == w for k in cells)]
        sess = sorted({k[1] for k in cells})
        summ = {"resp": [], "comp": []}
        for w in wls:
            parts = []
            for key in ("resp", "comp"):
                p = [pen(cells[(w, s, 2133)], cells[(w, s, 3199)], key) for s in sess
                     if (w, s, 2133) in cells and (w, s, 3199) in cells]
                if not p:
                    continue
                summ[key].append((w, min(p), max(p), max(p) - min(p)))
                parts.append(f"{key} " + " ".join(f"{x:+.2f}" for x in p) + f" span {max(p) - min(p):.2f}")
            n = cells[(w, sess[0], 2133)].n
            print(f"  {board:4s} {NAME[w]:22s} n={n:<5} " + " | ".join(parts))
        for key in ("resp", "comp"):
            rows = summ[key]
            for lim in (1.0, 2.0):
                ok = [NAME[w] for w, _, _, s in rows if round(s, 1) <= lim]
                bad = [f"{NAME[w]} {s:.2f}" for w, _, _, s in rows if round(s, 1) > lim]
                print(f"  {board} {key}: span <= {lim:.0f} pp for {len(ok)} of {len(rows)} workloads"
                      + (f"; above: {', '.join(bad)}" if bad else ""))
        for w, lo, hi, s in summ["resp"]:
            if w in ("mobilenet", "cproxyv2"):
                print(f"  {board} {NAME[w]} (resp): {lo:+.2f} to {hi:+.2f} % across the sessions")
    print()


def sec_batch(R):
    print("== [Supp. Sec. S3] batch axis, Nano MobileNetV2: per-image 2133 -> 3199 MHz penalty "
          "and the 204 MHz collapse p50(204)/p50(3199) ==")
    for b in (1, 4, 8):
        c = {e: load(f"{R}/batch_axis/mn_e{e}_b{b}.json") for e in (204, 2133, 3199)}
        if any(v is None for v in c.values()):
            print(f"  batch {b}: missing cells")
            continue
        print(f"  batch {b}: penalty comp {pen(c[2133], c[3199], 'comp'):+.3f} % (resp "
              f"{pen(c[2133], c[3199], 'resp'):+.3f}); collapse comp x{c[204].comp / c[3199].comp:.3f}"
              f" (resp x{c[204].resp / c[3199].resp:.3f}); per-batch comp p50 3199 {c[3199].comp:.2f} ms,"
              f" n={c[3199].n}")
    print()


def trimmed_mean_power(path):
    v = np.array([float(r["vdd_in_mw"]) for r in csv.DictReader(open(path))])
    n = len(v)
    return float(v[n // 10:-n // 10].mean()) if n >= 20 else float("nan")


def sec_llm(R):
    print("== [Supp. Table S13; main Sec. V] EMC exposure of LLM decode (NX, GPU 1173 MHz, "
          "300 tokens, three runs per EMC point) ==")
    rows = [("size", "Qwen2.5-0.5B Q4", "q05"), ("size", "Qwen2.5-1.5B Q4", "q15"),
            ("size", "Qwen2.5-3B Q4", "q3"), ("quant", "1.5B Q8_0", "q15q8"),
            ("context", "1.5B Q4, 64-token prompt", "q15p64"),
            ("context", "1.5B Q4, 4k prompt", "q15p4096")]
    print(f"  {'axis':8s} {'config':26s} {'ms/tok@2133':>11s} {'penalty':>8s} {'(pooled)':>9s} "
          f"{'mJ/tok@2133':>11s} {'mJ/tok@3199':>11s}")
    pens = []
    for axis, label, tag in rows:
        med, pooled, ej = {}, {}, {}
        for e in (2133, 3199):
            js = sorted(glob.glob(f"{R}/llm_axes/{tag}_e{e}_r*.json"))
            if not js:
                break
            p50 = []
            toks = []
            en = []
            for j in js:
                p = json.load(open(j))["compute_us"]["p50"] / 1e3
                p50.append(p)
                c = j[:-5] + ".csv"
                if os.path.isfile(c):
                    toks.append(np.array([float(r["dur_ns"]) for r in csv.DictReader(open(c))]) / 1e6)
                pw = j[:-5] + ".power.csv"
                if os.path.isfile(pw):
                    en.append(trimmed_mean_power(pw) * p / 1e3)
            med[e] = float(np.median(p50))
            pooled[e] = float(np.median(np.concatenate(toks))) if toks else float("nan")
            ej[e] = float(np.mean(en)) if en else float("nan")
        if len(med) < 2:
            print(f"  {axis:8s} {label:26s} missing")
            continue
        p = (med[2133] / med[3199] - 1) * 100
        pens.append((label, p))
        pp = (pooled[2133] / pooled[3199] - 1) * 100
        print(f"  {axis:8s} {label:26s} {med[2133]:11.3f} {p:+8.2f} {pp:+9.2f} {ej[2133]:11.1f} {ej[3199]:11.1f}")
    if pens:
        main = [p for lab, p in pens if "64-token" not in lab]
        print(f"  penalty range over the table rows: {min(main):+.2f} to {max(main):+.2f} %")
    print()


def sec_pairs(R):
    gpus = (306, 408, 510, 612, 714, 816, 918, 1020)
    print("== [main Sec. IV; Supp. Sec. S3] two-cell refit at EMC 2133 MHz from every pair of the "
          "eight GPU cells: worst |error| on the other six (compute p50) ==")
    for w in ("mobilenet", "vit", "proxy", "cproxyv2"):
        c = {}
        for g in gpus:
            x = load(f"{R}/estimator_break/emc2133_gpu{g}_{w}.json")
            if x is not None:
                c[g] = x.comp
        if len(c) != len(gpus):
            print(f"  {NAME[w]}: missing cells")
            continue
        worst = {}
        for a, b in itertools.combinations(gpus, 2):
            k = (c[a] - c[b]) / (1.0 / a - 1.0 / b)
            bb = c[b] - k / b
            worst[(a, b)] = max(abs(k / g + bb - c[g]) / c[g] for g in gpus if g not in (a, b)) * 100
        ok = [p for p, v in worst.items() if v <= 10.0]
        bad = sorted(p for p in worst if p not in ok)
        adj = [p for p in bad if gpus.index(p[1]) - gpus.index(p[0]) == 1]
        best = min(worst.values())
        print(f"  {NAME[w]:22s} pairs with worst <= 10 %: {len(ok)}/28; endpoint pair (306, 1020) worst "
              f"{worst[(306, 1020)]:.2f} %, best pair {best:.2f} %, gap {worst[(306, 1020)] - best:.2f} pp")
        if bad:
            print(f"  {'':22s} failing pairs: {len(bad)} ({len(adj)} of them adjacent cells): "
                  + " ".join(f"{a}/{b}" for a, b in bad))
    print()


SECTIONS = {"xsku": sec_xsku, "s23": sec_s23, "s27": sec_s27, "reps": sec_reps,
            "batch": sec_batch, "llm": sec_llm, "pairs": sec_pairs}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--section", action="append", choices=sorted(SECTIONS),
                    help="print only this section (repeatable)")
    ap.add_argument("--results", default=os.path.join(LAB, "results"),
                    help="results directory (default: the checkout's results/)")
    a = ap.parse_args()
    R = os.path.abspath(a.results)
    if not os.path.isdir(R):
        sys.exit(f"no results directory at {R}")
    for s in a.section or list(SECTIONS):
        SECTIONS[s](R)


if __name__ == "__main__":
    main()

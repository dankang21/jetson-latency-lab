#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Solo locked-clock dispersion p99/p50 of the profiling cells behind the safety multiplier
r = 1.035 (main text Sec. IV, Supp. Sec. S2 'Margin provenance and sensitivity'): the June
Nano EMC sweep and the endpoint cells of every deployed two-cell fit.

Read-only over results/ (python3 standard library; nothing is written).  Run from anywhere:
    python3 campaign/profile_dispersion.py [--results DIR]

A cell is one run JSON of harness/infer_bench.py (tools/slm_decode_bench for the Qwen2.5 decode
cell): its p99 / p50 of compute_us and of response_us, as the harness records them over the
timed loop (a decode cell has compute_us only).  'mode' is root for the root characterization
harness (empty meta.rt_warnings) and user for the user-mode runs; Hz and n are the release rate
and the timed iterations.  Per group the maximum of each ratio is printed with its cell, for
every EMC point and over the whole group.

Groups (results/):
  June Nano EMC sweep   campaign_g2/partA/emc{204,665,2133,3199}_<w>.json (one run per workload
                        and EMC point, 2026-06-11; campaign/analyze_g2.py prints the compute
                        p99/p50 at 3199 MHz of the same cells)
  endpoint cells        the GPU 306 MHz and top-clock cells at EMC 3199 and 2133 MHz through
                        which the deployed two-cell fits T(f) = k/f + b run:
                        nx_validation/nx_est_emc{3199,2133}_gpu{306,1173}_<m>.json (NX original
                        cells; m = mobilenet, vit, yolov8s, resnet50),
                        refit33/{nx,nx_vit,nx_yolov8s,nx_resnet50}/emc{3199,2133}_g{306,1173}_r{1,2,3}.json
                        (NX rate-matched cells, three repeats),
                        nano_grid/{mobilenet,vit}/est_emc{3199,2133}_gpu{306,1020}.json (Nano
                        original cells),
                        refit33/nano/emc{3199,2133}_g{306,1020}_r{1,2,3}.json (Nano MobileNetV2
                        33 Hz re-profile, three repeats) and
                        refit33/nano_vit/emc{3199,2133}_g{306,1020}_r{1,2,3}.json (Nano ViT-Small
                        rate-matched cells, three repeats).
"""
import argparse
import json
import os
import sys

LAB = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EMC_SWEEP = (204, 665, 2133, 3199)
SWEEP_WL = ("mobilenet", "vit", "proxy", "cproxyv1", "cproxyv2", "slm")
ENDPOINT_EMC = (3199, 2133)
NX_MODELS = ("mobilenet", "vit", "yolov8s", "resnet50")


def groups():
    """(title, [(EMC MHz, path relative to results/), ...]) in print order."""
    g = [("June Nano EMC sweep, campaign_g2/partA",
          [(e, f"campaign_g2/partA/emc{e}_{w}.json") for e in EMC_SWEEP for w in SWEEP_WL])]
    g.append(("NX original endpoint cells, nx_validation",
              [(e, f"nx_validation/nx_est_emc{e}_gpu{f}_{m}.json")
               for m in NX_MODELS for e in ENDPOINT_EMC for f in (306, 1173)]))
    for sub in ("nx", "nx_vit", "nx_yolov8s", "nx_resnet50"):
        g.append((f"NX rate-matched endpoint cells, refit33/{sub}",
                  [(e, f"refit33/{sub}/emc{e}_g{f}_r{k}.json")
                   for e in ENDPOINT_EMC for f in (306, 1173) for k in (1, 2, 3)]))
    for m in ("mobilenet", "vit"):
        g.append((f"Nano original endpoint cells, nano_grid/{m}",
                  [(e, f"nano_grid/{m}/est_emc{e}_gpu{f}.json") for e in ENDPOINT_EMC for f in (306, 1020)]))
    for sub, what in (("nano", "Nano 33 Hz re-profile"), ("nano_vit", "Nano rate-matched endpoint cells")):
        g.append((f"{what}, refit33/{sub}",
                  [(e, f"refit33/{sub}/emc{e}_g{f}_r{k}.json")
                   for e in ENDPOINT_EMC for f in (306, 1020) for k in (1, 2, 3)]))
    return g


def cell(path):
    """(compute p99/p50, response p99/p50 or None, mode, hz, n) of one run JSON."""
    d = json.load(open(path))
    m = d.get("meta", {})
    c = d["compute_us"]["p99"] / d["compute_us"]["p50"]
    r = d["response_us"]["p99"] / d["response_us"]["p50"] if "response_us" in d else None
    rt = m.get("rt_warnings")
    mode = "-" if rt is None else ("root" if rt == [] else "user")
    return c, r, mode, m.get("hz"), m.get("iters", m.get("tokens"))


def fmt_max(rows, key):
    """'1.0636 (name)' for the largest value of rows[*][key], '-' if there is none."""
    vals = [(r[key], r["name"]) for r in rows if r[key] is not None]
    if not vals:
        return "-"
    v, name = max(vals)
    return f"{v:.4f} ({name})"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default=os.path.join(LAB, "results"),
                    help="results directory (default: the checkout's results/)")
    a = ap.parse_args()
    R = os.path.abspath(a.results)
    if not os.path.isdir(R):
        sys.exit(f"no results directory at {R}")
    summary = []
    for title, cells in groups():
        print(f"== {title}")
        print(f"   {'cell (p99/p50)':<36} {'EMC':>5} {'compute':>8} {'response':>8}  mode  Hz/n")
        rows = []
        for emc, rel in cells:
            p = os.path.join(R, rel)
            if not os.path.isfile(p):
                print(f"   {rel:<36} missing")
                continue
            c, r, mode, hz, n = cell(p)
            name = os.path.basename(rel)[:-5]
            rows.append(dict(emc=emc, name=name, c=c, r=r))
            rs = "-" if r is None else f"{r:.4f}"
            print(f"   {name:<36} {emc:>5} {c:>8.4f} {rs:>8}  {mode:<4}  {hz}/{n}")
        for emc in sorted({x["emc"] for x in rows}, key=lambda e: -e):
            sub = [x for x in rows if x["emc"] == emc]
            print(f"   max at EMC {emc}: compute {fmt_max(sub, 'c')}, response {fmt_max(sub, 'r')}")
            summary.append((title, str(emc), len(sub), fmt_max(sub, "c"), fmt_max(sub, "r")))
        print(f"   max over the group: compute {fmt_max(rows, 'c')}, response {fmt_max(rows, 'r')}")
        summary.append((title, "all", len(rows), fmt_max(rows, "c"), fmt_max(rows, "r")))
        print()
    print("== maxima of p99/p50 per group and EMC point")
    print(f"{'group':<52} {'EMC':>5} {'n':>3}  {'compute (cell)':<42} response (cell)")
    for title, emc, n, c, r in summary:
        print(f"{title:<52} {emc:>5} {n:>3}  {c:<42} {r}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Band-width sensitivity of the dispersion-banded admission rule: the banded
decision replayed over all nineteen probes collected on the calibration
condition (NX, MobileNetV2 at 33 Hz against the 10 Hz ViT-Small co-runner,
EMC 2133, D = 12 ms) at candidate band widths h (supplement, band-width
subsection, label tab:bandwidth).  No board runs: every pick is recomputed
from the committed probe CSVs.

Probes (200 cycles at f_probe = 1173 MHz under the co-runner, response_us):
  reference  results/joint_gov/probe_co_g1173.csv            (1)
  round-1    results/fresh_loop/vprobe_t<1..8>.csv           (8)
  held-out   results/banded_loop/probe_t<1..10>.csv          (10)
Rule: admit_pick.evaluate, unchanged (the transcription of banded_pick.py and
fresh_pick.py that admit_pick.py --selftest checks against the deployed picks):
  M_g = max - p50 of the probe, T(f) = k/f + b through the GPU 306/1173 MHz
  endpoint cells at EMC 2133, m*(f) = (D - T(f) r) f / f_probe, r = 1.035;
  banded pick = lowest f in {510, ..., 1173} with M_g + h <= m*(f)
  (h = 0 is the point policy).  h is in probe-clock units, as deployed.
Widths are multiples of sigma = h/2 = 200 us, the adopted h = 400 us having
been set as twice the s.d. of M_g over the nine round-1 probes (the reference
probe and round-1 t1-t8; that s.d. is printed).  The widths that keep all
nineteen probes at 918 MHz form the interval
  ( max_i [m*(816) - M_g,i],  min_i [m*(918) - M_g,i] ].
Cells: the original 8 Hz cells (results/nx_validation/nx_est_emc2133_gpu
{306,1173}_mobilenet.json, the cells of every recorded pick; Table tab:bandwidth)
and the rate-matched 33 Hz cells (results/refit33/nx/cells_mobilenet/, the
median of three repeat p50s per endpoint; post-hoc sensitivity).
Per probe: M_g, the point and banded picks at the adopted width and the
probe-clock slack m*(f) - M_g at the point pick, at 816 and at 918 MHz.

Read-only on results/ (python3 + numpy); prints to stdout, writes nothing.
Usage: replay_bandwidth.py [--cells 8hz|33hz|both] [--widths 0,1,1.5,2,2.5]
                           [--sigma-us 200] [--json]
"""
import argparse, glob, json, os, re, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LAB = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from admit_pick import load_probe, load_cells, fit_2cell, evaluate, GRID, FPROBE, CELLS, H_US, R_MULT   # noqa: E402

D_US = 12000.0
INF = "INFEASIBLE"
CELL_SETS = {
    "8hz": ("original 8 Hz endpoint cells", CELLS[("nx", "mobilenet")]),
    "33hz": ("rate-matched 33 Hz endpoint cells",
             [f"{LAB}/results/refit33/nx/cells_mobilenet/nx_est33_emc2133_gpu306_mobilenet.json",
              f"{LAB}/results/refit33/nx/cells_mobilenet/nx_est33_emc2133_gpu1173_mobilenet.json"]),
}


def trial(path, rx):
    return int(re.search(rx, os.path.basename(path)).group(1))


def probes19():
    """[(label, csv)] in the order reference, round-1 t1-t8, held-out t1-t10."""
    fresh = sorted(glob.glob(f"{LAB}/results/fresh_loop/vprobe_t*.csv"), key=lambda p: trial(p, r"vprobe_t(\d+)"))
    held = sorted(glob.glob(f"{LAB}/results/banded_loop/probe_t*.csv"), key=lambda p: trial(p, r"probe_t(\d+)"))
    return ([("ref", f"{LAB}/results/joint_gov/probe_co_g1173.csv")]
            + [("r1_t%d" % trial(p, r"vprobe_t(\d+)"), p) for p in fresh]
            + [("ho_t%d" % trial(p, r"probe_t(\d+)"), p) for p in held])


def replay(cells, probes, widths_sigma, sigma):
    grid, fp = GRID["nx"], FPROBE["nx"]
    v = {lab: load_probe(p) for lab, p in probes}
    table = []
    for w in widths_sigma:
        picks = {lab: evaluate(v[lab], cells, D_US, grid, fp, w * sigma, R_MULT)["f_b"] for lab in v}
        dist = {}
        for f in picks.values():
            dist[str(f)] = dist.get(str(f), 0) + 1
        table.append(dict(h_sigma=w, h_us=w * sigma, dist=dist, picks=picks))
    per_probe = []
    for lab in v:
        r = evaluate(v[lab], cells, D_US, grid, fp, H_US, R_MULT)
        pf = {x["f"]: x for x in r["per_f"]}
        per_probe.append(dict(probe=lab, Mg_us=r["Mg_us"], f_pt=r["f_pt"], f_b=r["f_b"],
                              slack_pt_us=None if r["f_pt"] == INF else pf[r["f_pt"]]["margin_mg_us"],
                              slack816_us=pf[816]["margin_mg_us"], slack918_us=pf[918]["margin_mg_us"]))
    lo = max(x["slack816_us"] for x in per_probe)      # h must exceed this, or some probe admits 816
    hi = min(x["slack918_us"] for x in per_probe)      # h may not exceed this, or some probe skips 918
    k, b, _, _ = fit_2cell(cells)
    return dict(k_us_GHz=k, b_us=b, table=table, per_probe=per_probe,
                all_918=dict(h_gt_us=lo, h_le_us=hi, h_gt_sigma=lo / sigma, h_le_sigma=hi / sigma, nonempty=lo < hi))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cells", choices=["8hz", "33hz", "both"], default="both")
    ap.add_argument("--widths", default="0,1,1.5,2,2.5", help="band widths in units of sigma")
    ap.add_argument("--sigma-us", type=float, default=H_US / 2, help="sigma in us (adopted h = 2 sigma = 400 us)")
    ap.add_argument("--json", action="store_true", help="print the numbers as JSON instead of tables")
    a = ap.parse_args()

    widths = [float(x) for x in a.widths.split(",")]
    probes = probes19()
    assert len(probes) == 19, f"expected 19 probes, found {len(probes)}"
    mg9 = [float(load_probe(p).max() - np.median(load_probe(p))) for _, p in probes[:9]]
    out = dict(D_us=D_US, R=R_MULT, fprobe=FPROBE["nx"], grid=GRID["nx"], sigma_us=a.sigma_us,
               h_eff_918_us=H_US * FPROBE["nx"] / 918, probes=[lab for lab, _ in probes],
               round1_mg_sd_us=dict(ddof0=float(np.std(mg9)), ddof1=float(np.std(mg9, ddof=1)),
                                    range=float(max(mg9) - min(mg9))),
               cells={})
    sets = ["8hz", "33hz"] if a.cells == "both" else [a.cells]
    for tag in sets:
        name, files = CELL_SETS[tag]
        cells = load_cells(files)
        out["cells"][tag] = dict(name=name, files=[os.path.relpath(f, LAB) for f in files],
                                 cells_us={str(g): cells[g] for g in sorted(cells)},
                                 **replay(cells, probes, widths, a.sigma_us))
    if a.json:
        print(json.dumps(out, indent=1))
        return

    sd = out["round1_mg_sd_us"]
    print(f"=== band-width sensitivity: {len(probes)} probes (reference 1, round-1 8, held-out 10), "
          f"D={D_US/1000:g} ms, r={R_MULT}, f_probe={FPROBE['nx']} MHz ===")
    print(f"  s.d. of M_g over the nine round-1 probes (reference + round-1 t1-t8): {sd['ddof0']:.1f} us (ddof=0), "
          f"{sd['ddof1']:.1f} us (ddof=1), range {sd['range']:.1f} us")
    print(f"  sigma = {a.sigma_us:g} us; adopted h = {H_US:g} us = {H_US/a.sigma_us:g} sigma, "
          f"applied at 918 MHz as h*f_probe/918 = {out['h_eff_918_us']:.1f} us in deadline units")
    for tag in sets:
        c = out["cells"][tag]
        print(f"\n--- {c['name']} (" + ", ".join(f"{g} MHz {p:.1f} us" for g, p in c["cells_us"].items())
              + f"): k={c['k_us_GHz']:.2f} us*GHz, b={c['b_us']:.2f} us")
        cols = [str(f) for f in GRID["nx"] if any(str(f) in t["dist"] for t in c["table"])]
        cols += [INF] if any(INF in t["dist"] for t in c["table"]) else []
        print(f"  {'h':<20}" + "".join(f"{(x + ' MHz') if x != INF else 'declined':>10}" for x in cols)
              + "   picks off 918 MHz")
        for t in c["table"]:
            lab = ("0" if t["h_sigma"] == 0 else f"{t['h_sigma']:g} sigma") + f" ({t['h_us']:g} us)"
            off = [f"{p}:{f}" for p, f in t["picks"].items() if f != 918]
            print(f"  {lab:<20}" + "".join(f"{t['dist'].get(x, 0):>10}" for x in cols)
                  + f"   {' '.join(off) if off else '-'}")
        iv = c["all_918"]
        if iv["nonempty"]:
            print(f"  widths keeping all {len(probes)} probes at 918 MHz: h in ({iv['h_gt_us']:.1f}, {iv['h_le_us']:.1f}] us"
                  f" = ({iv['h_gt_sigma']:.2f} sigma, {iv['h_le_sigma']:.2f} sigma]")
        else:
            print(f"  no width keeps all {len(probes)} probes at 918 MHz: max m*(816)-M_g = {iv['h_gt_us']:.1f} us"
                  f" >= min m*(918)-M_g = {iv['h_le_us']:.1f} us")
        print(f"  {'probe':<8}{'M_g us':>9}{'f_pt':>7}{'f_b':>7}{'m*(f_pt)-M_g':>14}{'m*(816)-M_g':>13}{'m*(918)-M_g':>13}"
              f"   (f_b at h = {H_US:g} us)")
        for x in c["per_probe"]:
            s = "-" if x["slack_pt_us"] is None else f"{x['slack_pt_us']:.1f}"
            print(f"  {x['probe']:<8}{x['Mg_us']:>9.1f}{str(x['f_pt']):>7}{str(x['f_b']):>7}{s:>14}"
                  f"{x['slack816_us']:>13.1f}{x['slack918_us']:>13.1f}")


if __name__ == "__main__":
    main()

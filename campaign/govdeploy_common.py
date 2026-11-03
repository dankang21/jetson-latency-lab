# SPDX-License-Identifier: Apache-2.0
"""Deployed MobileNetV2 governor (Table II): per-board configuration, the two-cell
endpoint fits of both arms, the deployed pick rule and the loader of the interleaved
E1 sessions.  Shared by campaign/e5_han_baseline.py (and through it e5b_emcfree.py
and e5b_contrast.py), campaign/make_fig_govdeploy_rev.py and the Table II generator.

Read-only on results/ (python3 + numpy); nothing here writes a file.

Rule (both arms): T(f) = k/f + b through the GPU 306 and top-clock endpoint cells
(response_us p50), r = 1.035, pick = lowest f in the SKU's grid with T(f)*r <= D,
else the policy DECLINES (fallback to the top clock).
  blind : endpoints at EMC 3199 (profiling point);  aware : endpoints at EMC 2133.
  NX  original fit  : results/nx_validation/nx_est_emc{3199,2133}_gpu{306,1173}_mobilenet.json (8 Hz cells)
      rate-matched  : results/refit33/nx/emc{3199,2133}_g{306,1173}_r{1,2,3}.json (33 Hz, endpoint
                      p50 = median of the three repeat p50s)
  Nano original fit : results/nano_grid/mobilenet/est_emc{3199,2133}_gpu{306,1020}.json (50 Hz =
                      deployment rate, so rate-matched by construction)
      refit dir     : results/refit33/nano holds the 33 Hz re-profile of 2026-10-28 (not the
                      50 Hz deployment rate); refit_fits() reads it only when a caller asks.
Deployment data: results/govrep_rev/<session>/g<f>_p<1..10>.{csv,json,power.csv}, 1000 cycles per
run, EMC locked 2133; load_session() returns None for a session directory that does not exist.
"""
import csv
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LAB = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from e1b_stats import fit_2cell, energy, R   # noqa: E402

BOARDS = {
    "nx": dict(name="NX", hz=33.0, lo=306, fmax=1173,
               grid=[510, 612, 714, 816, 918, 1020, 1122, 1173],
               D=[4.4, 4.6, 4.8, 5.0, 5.2, 5.4, 5.6, 5.8, 6.0],
               sessions=["nx", "nx_mobilenet", "nx_mobilenet_s2", "nx_mobilenet_s3"],
               orig=lambda emc, g: f"{LAB}/results/nx_validation/nx_est_emc{emc}_gpu{g}_mobilenet.json",
               refit=f"{LAB}/results/refit33/nx"),
    "nano": dict(name="Nano", hz=50.0, lo=306, fmax=1020,
                 grid=[306, 408, 510, 612, 714, 816, 918, 1020],
                 D=[8.5, 9.0, 9.5, 10.0],
                 sessions=["nano", "nano_mobilenet", "nano_mobilenet_s2", "nano_mobilenet_s3"],
                 orig=lambda emc, g: f"{LAB}/results/nano_grid/mobilenet/est_emc{emc}_gpu{g}.json",
                 refit=f"{LAB}/results/refit33/nano"),
}


# ------------------------------------------------------------------ fits ----
def p50_json(path):
    d = json.load(open(path))
    return float(d["response_us"]["p50"]), float(d["meta"]["hz"])


def original_fits(B):
    out, info = {}, {}
    for arm, emc in (("blind", 3199), ("aware", 2133)):
        cells, hz = {}, None
        for g in (B["lo"], B["fmax"]):
            cells[g], hz = p50_json(B["orig"](emc, g))
        out[arm] = fit_2cell(cells[B["lo"]], cells[B["fmax"]], B["lo"], B["fmax"])
        info[arm] = dict(cells=cells, hz=hz)
    return out, info


def refit_fits(B):
    d = B["refit"]
    out, info = {}, {}
    for arm, emc in (("blind", 3199), ("aware", 2133)):
        cells, hz = {}, None
        for g in (B["lo"], B["fmax"]):
            files = [f"{d}/emc{emc}_g{g}_r{r}.json" for r in (1, 2, 3)]
            if not all(os.path.isfile(f) for f in files):
                return None, None
            ps = [p50_json(f) for f in files]
            hz = ps[0][1]
            cells[g] = float(np.median([p for p, _ in ps]))
        out[arm] = fit_2cell(cells[B["lo"]], cells[B["fmax"]], B["lo"], B["fmax"])
        info[arm] = dict(cells=cells, hz=hz)
    return out, info


def that(kb, f):
    k, b = kb
    return (k / f + b) * R


def pick(kb, D_us, B):
    """(clock, feasible): lowest f in the grid with T(f)*r <= D, else (fmax, False) = declined."""
    for f in B["grid"]:
        if that(kb, f) <= D_us:
            return f, True
    return B["fmax"], False


# ------------------------------------------------------------------ data ----
def load_session(name, hz_expected):
    root = f"{LAB}/results/govrep_rev/{name}"
    files = sorted(glob.glob(f"{root}/g*_p*.csv"))
    files = [f for f in files if not f.endswith(".power.csv")]
    if not os.path.isdir(root) or not files:
        return None
    runs = {}
    for csvp in files:
        g, p = os.path.basename(csvp)[:-4].split("_p")
        f, pair = int(g[1:]), int(p)
        v = np.array([float(r["response_us"]) for r in csv.DictReader(open(csvp))])
        meta = json.load(open(csvp[:-4] + ".json"))["meta"]
        assert abs(float(meta["hz"]) - hz_expected) < 1e-6, (csvp, meta["hz"])
        assert len(v) == 1000, (csvp, len(v))
        runs.setdefault(f, {})[pair] = dict(v=v, stem=csvp[:-4], hz=float(meta["hz"]),
                                            ts=meta.get("timestamp", ""))
    E = {f: {p: energy(runs[f][p]) for p in runs[f]} for f in runs}
    ts = [runs[f][p]["ts"] for f in runs for p in runs[f]]
    return dict(name=name, runs=runs, E=E, t0=min(ts)[:16], t1=max(ts)[:16],
                pairs={f: len(runs[f]) for f in sorted(runs)})

#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
E5: the Han et al. monotone power-law latency model, fitted to the committed
GPU x EMC estimator sweeps, and exposed as a drop-in estimator for the
admission / governor pick code.

Model (arXiv 2509.17970, Eq. 4):

    t = lambda * f_mem^-beta + mu * f_com^-gamma            (Han et al.)

which this paper writes, with an added constant offset C (supplement: "we add
a constant offset C when fitting"), as

    T(f_gpu, f_emc) = A * f_gpu^-alpha + B * f_emc^-beta + C          [us]

Internally the frequencies are normalised by F0 = 1000 MHz so that A and B
are in microseconds at 1 GHz; `predict()` takes plain MHz.

Data sources (all offline, nothing is run on the clocks):

  Nano (host jetson-orin)
    estimator_break  results/estimator_break/emc{3199,2133,665}_gpu{306..1020}_{mobilenet,vit,proxy,cproxyv2}.json
                     3 EMC x 8 GPU, 300 iters, root, 8 Hz (mobilenet, proxy) / 3 Hz (vit, cproxyv2)
    nano_grid        results/nano_grid/{mobilenet,vit}/est_emc{3199,2133}_gpu{306..1020}.json
                     2 EMC x 8 GPU, 800 iters, user, 50 Hz / 10 Hz   (deployment protocol)
    campaign_g2      results/campaign_g2/partA/emc{204,665,2133,3199}_{wl}.json      GPU 1020 only, 1000 iters
    pilot_emc        results/pilot_emc/emc{204,2133,3199}_{cnn,vit,proxy,cproxy,cproxyv1,slm}.json  GPU 1020, 200 iters
    mech_dose        results/mech_dose/e{204,665,2133,3199}_g{612..1020}[_repN]_bench.json (cproxyv2), 170 iters
  NX (host Pandora)
    nx_validation    results/nx_validation/nx_est_emc{3199,2133}_gpu{306..1173}_{mobilenet,vit,resnet50,yolov8s,proxy}.json
                     plus nx_est_emc665_gpu*_mobilenet.json (800 iters @ 33 Hz; the 3199/2133 sweeps are
                     300 iters @ 8 Hz for mobilenet/proxy and 800 iters @ 20/10/5 Hz for vit/resnet50/yolov8s)
    nx_validation_r33 results/nx_validation_r33/nx_est_emc{3199,2133,665}_gpu{306..1173}_{mobilenet,vit,resnet50,yolov8s}.json
                     3 EMC x 10 GPU per model, 800 iters at the DEPLOYMENT release rate (33/20/10/20 Hz),
                     one session 2026-09-15 (the rate-matched E5 grid; `--source nx_validation_r33`)
    nx_catalogue     results/nx_validation/nx_emc{204,665,2133,3199}_{mobilenet,proxy,cproxyv2,slm}.json  GPU 1173, 500 iters
    nx_pairs         results/nx_pairs/e{3199,665}_g{612..1173}_r{1,2,3}.json (mobilenet), 1000 iters @ 33 Hz
  Any directory     --sweep-dir DIR (or load_sweep_dir(DIR, model)): every [nx_][est_]emc<tok>_gpu<f>_<model>.json in DIR

Workload aliases: mobilenet|mnv2|cnn, vit, proxy|gemv|decode (GEMV 12x4096^2
decode proxy), cproxyv2|gemm|l2|cproxy (L2-resident GEMM 64x1024^3),
cproxyv1 (GEMM 4x2048^3), resnet50, yolov8s, slm (Qwen2.5-1.5B, compute_us only).

Fitting.  Non-linear least squares in (A, alpha, B, beta, C).  The venv python
that runs on the boards has numpy but NO scipy, so the primary fitter is a
scipy-free grid search over (alpha, beta) with a linear least-squares solve of
(A, B, C) at every grid point (A, B constrained non-negative so the sum stays
monotone, as the source model is), refined twice around the minimum.  When
scipy is importable (system python3 on the Nano has scipy 1.8.0) the grid
optimum is polished with scipy.optimize.least_squares and both are reported.

Identifiability.  With the constant C free, beta is NOT identifiable from two
EMC levels (B*e1^-beta + C and B*e2^-beta + C are two numbers for three
unknowns); the fit needs >= 3 distinct EMC levels, and alpha needs >= 3 GPU
levels.  `--inventory` prints which board/model combinations satisfy this.

Usage
-----
    python3 campaign/han_fit.py --inventory                    # EMC levels per board/model
    python3 campaign/han_fit.py --reproduce                    # supplementary numbers, reported vs reproduced
    python3 campaign/han_fit.py --board nano --model mobilenet # fit on the Nano estimator_break grid
    python3 campaign/han_fit.py --loco-table --key compute_us  # Supp. S3 per-workload in-sample + leave-one-cell-out
    python3 campaign/han_fit.py --board nano --model gemm --loco --no-scipy   # per-cell LOCO listing
    python3 campaign/han_fit.py --board nx --model mobilenet --holdout-emc 2133 --key response_us
    python3 campaign/han_fit.py --board nx --model vit --source nx_validation   # refuses: 2 EMC levels
    python3 campaign/han_fit.py --board nx --model vit         # auto -> nx_validation_r33 (3 levels, rate-matched)
    python3 campaign/han_fit.py --board nx --model yolov8s --sweep-dir results/nx_validation_r33 --holdout-emc 2133
    python3 campaign/han_fit.py --board nano --model gemm --train-emc 3199,665.6 --holdout-emc 2133
    python3 campaign/han_fit.py --board nx --model mobilenet --save results/han_fit/nx_mobilenet.json

Governor integration (E5)
-------------------------
Every pick script today evaluates a two-cell endpoint refit
    k = (T_lo - T_hi) / (1e3/f_lo - 1e3/f_hi);  b = T_hi - k*1e3/f_hi;  T_hat(f) = k*1e3/f + b
from `results/nx_validation/nx_est_emc2133_gpu*_<model>.json` (banded_pick.py
lines 14-17, fresh_pick.py lines 12-15, admit_pick.py::fit_2cell,
joint_gov_policy.py::fit_2cell).  To deploy the Han model as the
estimator instead, replace ONLY the T_hat line and keep the feasibility rule:

    import sys; sys.path.insert(0, "campaign")     # run from the repository root
    from han_fit import load_cells, fit_han, predict, pick_han

    cells  = load_cells("nx", "mobilenet", key="response_us")   # list of Cell records
    params = fit_han(cells)                                       # raises if < 3 EMC levels
    # ---- admit_pick.py / banded_pick.py (vision, maximum guard, optional band) ----
    #   for g in grid:
    #       mstar = (D_us - predict(params, g, f_emc=2133) * R) / (fprobe / g)
    #       if Mg <= mstar - band_us: pick = g; break
    # which is exactly what pick_han() does:
    pick, slack = pick_han(params, f_emc=2133, D_us=12000, Mg=Mg,
                           grid=(510, 612, 714, 816, 918, 1020, 1122, 1173),
                           fprobe=1173, R=1.035, band_us=400)
    # ---- joint_gov_policy.py::pick(k, b, D_us, m, gamma, H) ----
    #   t = predict(params, f, f_emc) * (1 + gamma*H);  feasible iff t*R + m <= D_us
    # ---- nx_live_gov.py::pick(off) (in-loop EWMA-calibrated governor) ----
    #   (predict(params, f, a.emc) + off) * 1.035 <= D_us      (add --emc, --han-params)
    #   the EWMA residual then calibrates the Han model exactly as it calibrates k/f+b.

`params` is a plain dict (A, alpha, B, beta, C, f0_mhz, unit, ...) so it can be
passed on the command line as --han-params A,alpha,B,beta,C or saved with --save
and reloaded with `load_params(path)`.  `predict` is pure numpy and accepts
scalars or arrays.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import sys
from collections import defaultdict, namedtuple

import numpy as np

try:  # optional polish; the on-device venv has no scipy
    from scipy.optimize import least_squares as _scipy_least_squares  # type: ignore
    HAVE_SCIPY = True
except Exception:  # pragma: no cover
    _scipy_least_squares = None
    HAVE_SCIPY = False

LAB = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(LAB, "results")
F0_MHZ = 1000.0            # frequency normalisation for A, B
EMC_TOKEN = {"204": 204.0, "665": 665.6, "2133": 2133.0, "3199": 3199.0}   # file token -> MHz
NX_CATALOGUE_GPU = 1173    # GPU pin of nx_emc*_<wl>.json (supp. Table 'catalogue' caption)
NANO_CATALOGUE_GPU = 1020  # GPU pin of campaign_g2/partA and pilot_emc cells

ALIASES = {
    "mobilenet": "mobilenet", "mnv2": "mobilenet", "mobilenetv2": "mobilenet", "cnn": "mobilenet",
    "vit": "vit", "vit-small": "vit",
    "proxy": "proxy", "gemv": "proxy", "decode": "proxy", "decode_proxy": "proxy",
    "cproxyv2": "cproxyv2", "gemm": "cproxyv2", "l2": "cproxyv2", "cproxy": "cproxyv2",
    "gemm_l2": "cproxyv2", "compute_proxy": "cproxyv2",
    "cproxyv1": "cproxyv1", "gemm2048": "cproxyv1",
    "resnet50": "resnet50", "resnet": "resnet50",
    "yolov8s": "yolov8s", "yolo": "yolov8s",
    "slm": "slm", "qwen": "slm",
}
PRETTY = {"mobilenet": "MobileNetV2", "vit": "ViT-Small", "proxy": "GEMV decode proxy",
          "cproxyv2": "GEMM L2-resident", "cproxyv1": "GEMM 4x2048^3", "resnet50": "ResNet-50",
          "yolov8s": "YOLOv8s", "slm": "Qwen2.5-1.5B (ms/tok)"}
MODELS = ["mobilenet", "vit", "proxy", "cproxyv2", "cproxyv1", "resnet50", "yolov8s", "slm"]

Cell = namedtuple("Cell", "f_emc f_gpu p50_us source key hz iters path")

# ----------------------------------------------------------------------------
# data loading
# ----------------------------------------------------------------------------


def canon(model: str) -> str:
    m = ALIASES.get(model.lower())
    if m is None:
        raise SystemExit(f"unknown model '{model}'; known: {sorted(set(ALIASES))}")
    return m


def _p50(path: str, key: str):
    d = json.load(open(path))
    blk = d.get(key)
    if not blk or "p50" not in blk:
        return None, d.get("meta", {})
    return float(blk["p50"]), d.get("meta", {})


def _cell(path, f_emc, f_gpu, source, key):
    v, meta = _p50(path, key)
    if v is None:
        return None
    return Cell(float(f_emc), int(f_gpu), v, source, key, meta.get("hz"), meta.get("iters"), path)


# each loader: (board, model, key) -> list[Cell]
def _load_estimator_break(model, key):
    out = []
    for f in glob.glob(os.path.join(RES, "estimator_break", f"emc*_gpu*_{model}.json")):
        m = re.match(r"emc(\d+)_gpu(\d+)_(\w+)\.json$", os.path.basename(f))
        if m and m.group(3) == model and m.group(1) in EMC_TOKEN:
            c = _cell(f, EMC_TOKEN[m.group(1)], m.group(2), "estimator_break", key)
            if c:
                out.append(c)
    return out


def _load_nano_grid(model, key):
    out = []
    for f in glob.glob(os.path.join(RES, "nano_grid", model, "est_emc*_gpu*.json")):
        m = re.match(r"est_emc(\d+)_gpu(\d+)\.json$", os.path.basename(f))
        if m and m.group(1) in EMC_TOKEN:
            c = _cell(f, EMC_TOKEN[m.group(1)], m.group(2), "nano_grid", key)
            if c:
                out.append(c)
    return out


def _load_campaign_g2(model, key):
    out = []
    for f in glob.glob(os.path.join(RES, "campaign_g2", "partA", f"emc*_{model}.json")):
        m = re.match(r"emc(\d+)_(\w+)\.json$", os.path.basename(f))
        if m and m.group(2) == model and m.group(1) in EMC_TOKEN:
            c = _cell(f, EMC_TOKEN[m.group(1)], NANO_CATALOGUE_GPU, "campaign_g2", key)
            if c:
                out.append(c)
    return out


def _load_pilot_emc(model, key):
    tok = {"mobilenet": "cnn", "cproxyv2": "cproxy"}.get(model, model)
    out = []
    for f in glob.glob(os.path.join(RES, "pilot_emc", f"emc*_{tok}.json")):
        m = re.match(r"emc(\d+)_(\w+)\.json$", os.path.basename(f))
        # emc1701_* cells did not hold the lock (tegrastats shows 2133) -> excluded
        if m and m.group(2) == tok and m.group(1) in EMC_TOKEN:
            c = _cell(f, EMC_TOKEN[m.group(1)], NANO_CATALOGUE_GPU, "pilot_emc", key)
            if c:
                out.append(c)
    return out


def _load_mech_dose(model, key):
    if model != "cproxyv2":
        return []
    out = []
    for f in glob.glob(os.path.join(RES, "mech_dose", "e*_g*_bench.json")):
        m = re.match(r"e(\d+)_g(\d+)(?:_rep\d+)?_bench\.json$", os.path.basename(f))
        if m and m.group(1) in EMC_TOKEN:
            c = _cell(f, EMC_TOKEN[m.group(1)], m.group(2), "mech_dose", key)
            if c:
                out.append(c)
    return out


def _load_nx_validation(model, key):
    out = []
    for f in glob.glob(os.path.join(RES, "nx_validation", f"nx_est_emc*_gpu*_{model}.json")):
        m = re.match(r"nx_est_emc(\d+)_gpu(\d+)_(\w+)\.json$", os.path.basename(f))
        if m and m.group(3) == model and m.group(1) in EMC_TOKEN:
            c = _cell(f, EMC_TOKEN[m.group(1)], m.group(2), "nx_validation", key)
            if c:
                out.append(c)
    return out


def _load_sweep_dir(dirname, model, key, source=None):
    """Every [nx_][est_]emc<tok>_gpu<f>_<model>.json under `dirname` (one sweep directory)."""
    src = source or os.path.basename(os.path.normpath(dirname))
    out = []
    for f in glob.glob(os.path.join(dirname, f"*emc*_gpu*_{model}.json")):
        m = re.match(r"(?:nx_)?(?:est_)?emc(\d+)_gpu(\d+)_(\w+)\.json$", os.path.basename(f))
        if m and m.group(3) == model and m.group(1) in EMC_TOKEN:
            c = _cell(f, EMC_TOKEN[m.group(1)], m.group(2), src, key)
            if c:
                out.append(c)
    return out


def load_sweep_dir(dirname, model, key="response_us"):
    """Cells of one sweep directory (e.g. results/nx_validation_r33) for `model`, sorted like load_cells."""
    cells = _load_sweep_dir(dirname, canon(model), key)
    return sorted(cells, key=lambda c: (-c.f_emc, c.f_gpu, c.source))


def _load_nx_validation_r33(model, key):
    return _load_sweep_dir(os.path.join(RES, "nx_validation_r33"), model, key, "nx_validation_r33")


def _load_nx_catalogue(model, key):
    out = []
    for f in glob.glob(os.path.join(RES, "nx_validation", f"nx_emc*_{model}.json")):
        m = re.match(r"nx_emc(\d+)_(\w+)\.json$", os.path.basename(f))
        if m and m.group(2) == model and m.group(1) in EMC_TOKEN:
            c = _cell(f, EMC_TOKEN[m.group(1)], NX_CATALOGUE_GPU, "nx_catalogue", key)
            if c:
                out.append(c)
    return out


def _load_nx_pairs(model, key):
    if model != "mobilenet":
        return []
    out = []
    for f in glob.glob(os.path.join(RES, "nx_pairs", "e*_g*_r*.json")):
        m = re.match(r"e(\d+)_g(\d+)_r(\d+)\.json$", os.path.basename(f))
        if m and m.group(1) in EMC_TOKEN:
            c = _cell(f, EMC_TOKEN[m.group(1)], m.group(2), "nx_pairs", key)
            if c:
                out.append(c)
    return out


# board -> ordered {source: (loader, kind)}; kind 'sweep' = multi-GPU sweep, 'point' = single GPU clock
SOURCES = {
    "nano": {
        "estimator_break": (_load_estimator_break, "sweep"),
        "nano_grid": (_load_nano_grid, "sweep"),
        "mech_dose": (_load_mech_dose, "sweep"),
        "campaign_g2": (_load_campaign_g2, "point"),
        "pilot_emc": (_load_pilot_emc, "point"),
    },
    "nx": {
        "nx_validation": (_load_nx_validation, "sweep"),
        "nx_validation_r33": (_load_nx_validation_r33, "sweep"),   # listed after nx_validation: ties under "auto" keep the old source
        "nx_pairs": (_load_nx_pairs, "sweep"),
        "nx_catalogue": (_load_nx_catalogue, "point"),
    },
}


def load_cells(board: str, model: str, key: str = "response_us", sources="auto"):
    """Return the committed p50 cells for (board, model) as a list of Cell.

    sources: "auto"  -> the multi-GPU sweep source with the most EMC levels
             "all"   -> union of every source for that board (protocols mixed!)
             name / comma-separated names -> exactly those sources
    """
    board = board.lower()
    model = canon(model)
    if board not in SOURCES:
        raise SystemExit(f"board must be one of {list(SOURCES)}")
    table = SOURCES[board]
    if sources == "all":
        names = list(table)
    elif sources == "auto":
        # prefer the most complete grid: (min #GPU points over its EMC levels, #EMC levels, #cells)
        best, best_n = None, (-1, -1, -1)
        for name, (loader, kind) in table.items():
            if kind != "sweep":
                continue
            cs = loader(model, key)
            if not cs:
                continue
            per = defaultdict(set)
            for c in cs:
                per[c.f_emc].add(c.f_gpu)
            n = (min(len(v) for v in per.values()), len(per), len(cs))
            if n > best_n:
                best, best_n = name, n
        names = [best] if best else []
    else:
        names = [s.strip() for s in sources.split(",")]
        for n in names:
            if n not in table:
                raise SystemExit(f"unknown source '{n}' for board {board}; known: {list(table)}")
    cells = []
    for n in names:
        cells.extend(table[n][0](model, key))
    return sorted(cells, key=lambda c: (-c.f_emc, c.f_gpu, c.source))


def cells_dict(cells):
    """{(f_emc, f_gpu): median p50_us} over duplicates (reps / sources)."""
    acc = defaultdict(list)
    for c in cells:
        acc[(c.f_emc, c.f_gpu)].append(c.p50_us)
    return {k: float(np.median(v)) for k, v in acc.items()}


# ----------------------------------------------------------------------------
# model
# ----------------------------------------------------------------------------


def predict(params, f_gpu, f_emc):
    """T_hat [us] = A*(f_gpu/f0)^-alpha + B*(f_emc/f0)^-beta + C.

    params: dict with keys A, alpha, B, beta, C (and optionally f0_mhz, default
    1000).  f_gpu, f_emc in MHz; scalars or numpy arrays (broadcast).
    """
    f0 = float(params.get("f0_mhz", F0_MHZ))
    fg = np.asarray(f_gpu, dtype=float) / f0
    fe = np.asarray(f_emc, dtype=float) / f0
    t = (params["A"] * fg ** (-params["alpha"])
         + params["B"] * fe ** (-params["beta"])
         + params.get("C", 0.0))
    return float(t) if np.ndim(t) == 0 else t


def _design(fg, fe, alpha, beta, with_c):
    cols = [fg ** (-alpha), fe ** (-beta)]
    if with_c:
        cols.append(np.ones_like(fg))
    return np.column_stack(cols)


def _lin_solve(X, y, w, nonneg):
    """Weighted LSQ for (A, B[, C]); A, B >= 0 by exhaustive active set."""
    Xw, yw = X * w[:, None], y * w
    ncol = X.shape[1]
    best = None
    subsets = [()] if not nonneg else [(), (0,), (1,), (0, 1)]
    for zero in subsets:
        keep = [j for j in range(ncol) if j not in zero]
        if not keep:
            continue
        sol, *_ = np.linalg.lstsq(Xw[:, keep], yw, rcond=None)
        coef = np.zeros(ncol)
        coef[keep] = sol
        if nonneg and (coef[0] < 0 or coef[1] < 0):
            continue
        sse = float(np.sum((Xw @ coef - yw) ** 2))
        if best is None or sse < best[0]:
            best = (sse, coef)
    return best


def fit_han(cells, train=None, with_c=True, weight="abs", nonneg=True,
            alpha_range=(0.05, 3.0), beta_range=(0.05, 3.0), fix_beta=None,
            polish="auto"):
    """Fit T = A f_gpu^-alpha + B f_emc^-beta + C to the given cells.

    cells : list[Cell] or {(f_emc, f_gpu): p50_us}
    train : optional iterable of (f_emc, f_gpu) keys to fit on (default: all)
    weight: "abs" (residuals in us) or "rel" (residuals divided by measured p50)
    nonneg: constrain A, B >= 0 (monotone sum, as in the source model)
    fix_beta: pin beta (only way to fit with 2 EMC levels; flagged in result)
    polish: "auto" = scipy if importable, True/False to force

    Returns a params dict usable by predict(), with fit diagnostics.
    """
    d = cells if isinstance(cells, dict) else cells_dict(cells)
    keys = sorted(d) if train is None else [k for k in train if k in d]
    if len(keys) < (4 if with_c else 3):
        raise ValueError(f"need at least {4 if with_c else 3} cells, have {len(keys)}")
    n_emc = len({k[0] for k in keys})
    n_gpu = len({k[1] for k in keys})
    if fix_beta is None and with_c and n_emc < 3:
        raise ValueError(f"beta not identifiable: {n_emc} EMC level(s) in training set "
                         f"(need >= 3 with the free offset C; pass fix_beta=... to pin it)")
    if n_gpu < 3:
        raise ValueError(f"alpha not identifiable: {n_gpu} GPU level(s) in training set")
    fe = np.array([k[0] for k in keys]) / F0_MHZ
    fg = np.array([k[1] for k in keys]) / F0_MHZ
    y = np.array([d[k] for k in keys], dtype=float)
    w = 1.0 / y if weight == "rel" else np.ones_like(y)

    def solve(al, be):
        return _lin_solve(_design(fg, fe, al, be, with_c), y, w, nonneg)

    # coarse grid -> two refinements
    def grid_search(a_lo, a_hi, b_lo, b_hi, step):
        best = None
        for al in np.arange(a_lo, a_hi + 1e-12, step):
            for be in ([fix_beta] if fix_beta is not None else np.arange(b_lo, b_hi + 1e-12, step)):
                r = solve(al, be)
                if r and (best is None or r[0] < best[0]):
                    best = (r[0], al, be, r[1])
        return best
    best = grid_search(alpha_range[0], alpha_range[1], beta_range[0], beta_range[1], 0.05)
    if best is None:
        raise RuntimeError("grid search found no admissible solution")
    for step, halfw in ((0.005, 0.06), (0.0005, 0.006)):
        _, al, be, _ = best
        b2 = grid_search(max(al - halfw, 1e-3), al + halfw,
                         max(be - halfw, 1e-3), be + halfw, step)
        if b2 and b2[0] <= best[0]:
            best = b2
    sse, al, be, coef = best
    params = {"A": float(coef[0]), "alpha": float(al), "B": float(coef[1]), "beta": float(be),
              "C": float(coef[2]) if with_c else 0.0, "f0_mhz": F0_MHZ, "unit": "us",
              "with_c": bool(with_c), "weight": weight, "nonneg": bool(nonneg),
              "fix_beta": fix_beta, "n_train": len(keys), "n_emc_levels": n_emc,
              "n_gpu_levels": n_gpu, "train_keys": [list(k) for k in keys],
              "sse_grid": float(sse), "fitter": "numpy-grid"}
    use_scipy = HAVE_SCIPY if polish == "auto" else bool(polish)
    if use_scipy and HAVE_SCIPY:
        p0 = [params["A"], al, params["B"], be] + ([params["C"]] if with_c else [])

        def resid(p):
            t = p[0] * fg ** (-p[1]) + p[2] * fe ** (-(fix_beta if fix_beta is not None else p[3]))
            if with_c:
                t = t + p[4]
            return (t - y) * w
        lo = [0.0 if nonneg else -np.inf, 1e-3, 0.0 if nonneg else -np.inf, 1e-3] + ([-np.inf] if with_c else [])
        hi = [np.inf, 10.0, np.inf, 10.0] + ([np.inf] if with_c else [])
        try:
            r = _scipy_least_squares(resid, p0, bounds=(lo, hi), xtol=1e-12, ftol=1e-12, gtol=1e-12,
                                     max_nfev=20000)
            sse_s = float(np.sum(r.fun ** 2))
            params["sse_scipy"] = sse_s
            if sse_s <= sse * (1 + 1e-9):
                p = r.x
                params.update({"A": float(p[0]), "alpha": float(p[1]), "B": float(p[2]),
                               "beta": float(fix_beta if fix_beta is not None else p[3]),
                               "C": float(p[4]) if with_c else 0.0, "fitter": "numpy-grid+scipy"})
        except Exception as ex:  # pragma: no cover
            params["scipy_error"] = str(ex)
    return params


def score(params, cells, keys=None):
    """Per-cell (key, measured, predicted, rel_err) for the given cells/keys."""
    d = cells if isinstance(cells, dict) else cells_dict(cells)
    keys = sorted(d) if keys is None else [k for k in keys if k in d]
    rows = []
    for k in keys:
        t = predict(params, k[1], k[0])
        rows.append((k, d[k], t, (t - d[k]) / d[k]))
    return rows


def err_stats(rows):
    e = np.array([abs(r[3]) for r in rows]) * 100.0 if rows else np.array([np.nan])
    return float(np.median(e)), float(np.max(e)), float(np.mean(e))


def loco(cells, keys=None, **fit_kw):
    """Leave-one-cell-out: for every cell, refit on all the other cells and score that cell.

    Returns rows (key, measured, predicted, rel_err, params) in the order of `keys`
    (default: every cell).  `fit_kw` is passed to fit_han; polish defaults to
    False so the figures come from the scipy-free numpy grid fitter and are
    reproducible on the on-device venv.
    """
    d = cells if isinstance(cells, dict) else cells_dict(cells)
    keys = sorted(d) if keys is None else [k for k in keys if k in d]
    fit_kw = dict({"polish": False}, **fit_kw)
    rows = []
    for k in keys:
        p = fit_han(d, train=[q for q in sorted(d) if q != k], **fit_kw)
        t = predict(p, k[1], k[0])
        rows.append((k, d[k], t, (t - d[k]) / d[k], p))
    return rows


def loco_table(key="compute_us", weight="abs", models=("mobilenet", "vit", "proxy", "cproxyv2"),
               nonneg=True, with_c=True, beta_max=6.0):
    """Supp. S3 per-workload figures under ONE protocol on the Nano estimator_break grid
    (24 cells = EMC {3199, 2133, 665.6} x GPU {306..1020}): full-grid in-sample median / max
    |rel err| and leave-one-cell-out median / max, numpy grid fitter only (no scipy polish).

    beta_max: the grid searches beta in [0.05, beta_max].  The GEMM's least-squares
    optimum sits at beta ~ 3.65 (its 665.6 MHz row is nearly a step), above the
    fitter's default cap of 3.0; the SSE is flat to < 0.1 % for beta >= 3, which
    leaves every median/max figure unchanged (< 0.3 pp) but moves the GEMM's
    in-sample T(2133)-T(3199) from +0.38 ms (beta capped at 3) to +0.21 ms (the
    optimum, also what the scipy polish reaches).  Default 6.0 so the numpy-only
    run reproduces the optimum.  Returns {model: {...}}.
    """
    out = {}
    fk = dict(with_c=with_c, weight=weight, nonneg=nonneg, beta_range=(0.05, float(beta_max)))
    print(f"Nano estimator_break grid, p50 {key}, weight {weight}, A,B>=0={nonneg}, C free={with_c}, "
          f"fitter numpy-grid (polish off), alpha in [0.05, 3], beta in [0.05, {beta_max:g}]")
    hdr = (f"{'workload':<18} {'n':>3} | {'in-sample med/max %':>20} {'(max cell)':<16} | "
           f"{'LOCO med/max %':>15} {'(max cell)':<16} | {'dT(2133-3199)@1020 ms: meas / in / loco':>40}")
    print(hdr)
    print("-" * len(hdr))
    for m in models:
        d = cells_dict(_load_estimator_break(m, key))
        p = fit_han(d, polish=False, **fk)
        ins = score(p, d)
        lo = loco(d, **fk)
        i_med, i_max, _ = err_stats(ins)
        l_med, l_max, _ = err_stats(lo)
        i_cell = max(ins, key=lambda r: abs(r[3]))[0]
        l_cell = max(lo, key=lambda r: abs(r[3]))[0]
        top = max(k[1] for k in d)
        dm = (d[(2133.0, top)] - d[(3199.0, top)]) / 1e3
        dp_in = (predict(p, top, 2133.0) - predict(p, top, 3199.0)) / 1e3
        lo_by = {r[0]: r for r in lo}
        # leave-(2133, top)-out prediction of the inversion step, as in protocol P2
        dp_lo = (lo_by[(2133.0, top)][2] - predict(lo_by[(2133.0, top)][4], top, 3199.0)) / 1e3
        out[m] = {"n": len(d), "in_med": i_med, "in_max": i_max, "in_max_cell": i_cell,
                  "loco_med": l_med, "loco_max": l_max, "loco_max_cell": l_cell,
                  "d_meas_ms": dm, "d_in_ms": dp_in, "d_loco_ms": dp_lo, "params": p,
                  "loco_rows": [(r[0], r[1], r[2], r[3]) for r in lo]}
        print(f"{PRETTY[m]:<18} {len(d):>3} | {i_med:8.1f} / {i_max:<8.1f} {str(i_cell):<16} | "
              f"{l_med:5.1f} / {l_max:<7.1f} {str(l_cell):<16} | {dm:+6.2f} / {dp_in:+6.2f} / {dp_lo:+6.2f}")
    return out


def two_cell_kb(cells, f_emc):
    """Endpoint two-cell refit T = k*1e3/f_gpu + b at one EMC level (banded_pick.py lines 14-17)."""
    d = cells if isinstance(cells, dict) else cells_dict(cells)
    pts = {g: v for (e, g), v in d.items() if abs(e - f_emc) < 1e-6}
    lo, hi = min(pts), max(pts)
    k = (pts[lo] - pts[hi]) / (1e3 / lo - 1e3 / hi)
    return k, pts[hi] - k * (1e3 / hi)


def pick_han(params, f_emc, D_us, Mg, grid=(510, 612, 714, 816, 918, 1020, 1122, 1173),
             fprobe=1173, R=1.035, band_us=0.0, offset_us=0.0):
    """banded_pick.py / admit_pick.py admission rule with the Han T_hat.

    Returns (pick_mhz, slack_us); pick 0 means no feasible clock (INFEASIBLE).
    Mg is the probe guard (max - p50, or p98 - p50) measured at fprobe.
    """
    for g in grid:
        t_hat = predict(params, g, f_emc) + offset_us
        mstar = (D_us - t_hat * R) / (fprobe / g)
        if Mg <= mstar - band_us:
            return g, float(mstar - Mg)
    return 0, float("nan")


def params_from_string(s):
    A, alpha, B, beta, C = [float(x) for x in s.split(",")]
    return {"A": A, "alpha": alpha, "B": B, "beta": beta, "C": C, "f0_mhz": F0_MHZ, "unit": "us"}


def load_params(path):
    return json.load(open(path))


# ----------------------------------------------------------------------------
# inventory
# ----------------------------------------------------------------------------


def inventory(key="response_us", boards=("nano", "nx"), models=MODELS, verbose=True):
    """Print EMC levels x GPU points per board/model/source; return summary dict."""
    summary = {}
    for board in boards:
        if verbose:
            print(f"\n== {board.upper()}: EMC levels available per model (p50 key {key}) ==")
            print(f"{'model':<18} {'source':<16} {'kind':<6} {'EMC MHz: #GPU points (iters@Hz)'}")
        for model in models:
            fittable_src = None
            for name, (loader, kind) in SOURCES[board].items():
                cs = loader(model, key)
                if not cs and key == "response_us":      # slm has compute_us only
                    cs = loader(model, "compute_us")
                    if cs:
                        cs = [c._replace(key="compute_us") for c in cs]
                if not cs:
                    continue
                per = defaultdict(list)
                for c in cs:
                    per[c.f_emc].append(c)
                desc = []
                for e in sorted(per, reverse=True):
                    gs = sorted({c.f_gpu for c in per[e]})
                    prot = sorted({f"{c.iters}@{c.hz}" for c in per[e]})
                    desc.append(f"{e:g}: {len(gs)}g [{gs[0]}..{gs[-1]}] ({','.join(prot)})")
                n_emc, n_gpu = len(per), len({c.f_gpu for c in cs})
                min_g = min(len({c.f_gpu for c in v}) for v in per.values())
                ok = kind == "sweep" and n_emc >= 3 and n_gpu >= 3
                full = ok and min_g >= 3          # every EMC level is itself a GPU sweep
                if full and fittable_src is None:
                    fittable_src = name
                keynote = "" if cs[0].key == key else f" [{cs[0].key} only]"
                if verbose:
                    tag = "   <- identifiable" if full else ("   <- identifiable (sparse: single-GPU levels)" if ok else "")
                    print(f"{PRETTY.get(model, model):<18} {name:<16} {kind:<6} {'; '.join(desc)}{keynote}{tag}")
                summary.setdefault(board, {}).setdefault(model, {})[name] = {
                    "kind": kind, "emc_levels": sorted(per, reverse=True), "n_gpu": n_gpu, "n_cells": len(cs)}
            if board in summary and model in summary[board]:
                summary[board][model]["_fittable_source"] = fittable_src
    if verbose:
        print("\n== Identifiability verdict (needs a multi-GPU sweep at >= 3 EMC levels) ==")
        for board in boards:
            for model in models:
                info = summary.get(board, {}).get(model)
                if not info:
                    continue
                src = info.get("_fittable_source")
                sweeps = {n: v for n, v in info.items() if not n.startswith("_") and v["kind"] == "sweep"}
                if src:
                    print(f"  {board:<5} {PRETTY.get(model, model):<18} OK   via {src} "
                          f"(EMC {', '.join(f'{e:g}' for e in info[src]['emc_levels'])})")
                elif sweeps:
                    have = sorted({e for v in sweeps.values() for e in v["emc_levels"]}, reverse=True)
                    missing = [e for e in (3199.0, 2133.0, 665.6) if e not in have]
                    print(f"  {board:<5} {PRETTY.get(model, model):<18} NO   sweeps at EMC "
                          f"{', '.join(f'{e:g}' for e in have)} -> needs a "
                          f"{', '.join(f'{e:g}' for e in missing) or 'third-level'} MHz GPU sweep first")
                else:
                    print(f"  {board:<5} {PRETTY.get(model, model):<18} NO   single-GPU-clock cells only "
                          f"(no GPU sweep at any EMC level)")
    return summary


# ----------------------------------------------------------------------------
# reproduction of the supplementary numbers
# ----------------------------------------------------------------------------

REPORTED = [
    # (id, text as printed in the supplement / main text, value)
    ("grid", "Nano GPUxEMC grid size (cells)", 20),
    ("mn_in", "MobileNetV2 in-sample median |err| %", 1.2),
    ("mn_out", "MobileNetV2 held-out 2133 MHz cell(s) median |err| %", 2.7),
    ("vit_in", "ViT-Small in-sample median |err| %", 2.7),
    ("vit_out", "ViT-Small held-out 2133 MHz median |err| %", 6.4),
    ("gemv_med", "GEMV proxy held-out median |err| %", 16.7),
    ("gemv_max", "GEMV proxy held-out max |err| %", 31.0),
    ("gemm_med", "GEMM L2-resident held-out median |err| %", 14.8),
    ("gemm_max", "GEMM L2-resident held-out max |err| %", 22.0),
    ("gemm_dpred", "GEMM L2 predicted T(2133)-T(3199) at GPU 1020, ms", 2.67),
    ("gemm_dmeas", "GEMM L2 measured  T(2133)-T(3199) at GPU 1020, ms", -1.70),
]

NANO_GPUS = [306, 408, 510, 612, 714, 816, 918, 1020]
NANO_EMCS = [3199.0, 2133.0, 665.6]


def _protocols():
    """Candidate training/held-out splits of the 24-cell estimator_break grid.

    Each entry: name -> dict(train, hold, with_c, fix_beta, extra)
      fix_beta  None    : free (requires >= 3 EMC levels in `train`)
                "solve" : P1 only -- beta is unidentified, pick the member of the
                          equal-SSE family whose GEMM prediction equals the reported +2.67 ms
                float   : pinned
      extra     None or "campaign_g2": merge the 4 catalogue points at GPU 1020 into the training set
    """
    allc = [(e, g) for e in NANO_EMCS for g in NANO_GPUS]
    row2133 = [c for c in allc if c[0] == 2133.0]
    no2133 = [c for c in allc if c[0] != 2133.0]
    P = {}
    P["P1 train{3199,665.6}x8 (16) / hold 2133x8  [beta UNIDENTIFIED; beta solved for +2.67]"] = \
        dict(train=no2133, hold=row2133, with_c=True, fix_beta="solve", extra=None)
    P["P1b same split, beta pinned 1.0"] = dict(train=no2133, hold=row2133, with_c=True, fix_beta=1.0, extra=None)
    P["P1c same split, beta pinned 2.0"] = dict(train=no2133, hold=row2133, with_c=True, fix_beta=2.0, extra=None)
    P["P1n same split, pure Han form (no C; identifiable from 2 levels)"] = \
        dict(train=no2133, hold=row2133, with_c=False, fix_beta=None, extra=None)
    P["P2 train all but (2133,1020) (23) / hold (2133,1020)"] = \
        dict(train=[c for c in allc if c != (2133.0, 1020)], hold=[(2133.0, 1020)], with_c=True, fix_beta=None, extra=None)
    P["P3 train all 24 / 'held-out' = 2133 row (in-sample)"] = \
        dict(train=allc, hold=row2133, with_c=True, fix_beta=None, extra=None)
    P["P4 train 20 = all but 2133@{714,816,918,1020} / hold those 4"] = \
        dict(train=[c for c in allc if not (c[0] == 2133.0 and c[1] >= 714)],
             hold=[c for c in allc if c[0] == 2133.0 and c[1] >= 714], with_c=True, fix_beta=None, extra=None)
    P["P5 train 20 = all but 2133@{408,612,816,1020} / hold those 4"] = \
        dict(train=[c for c in allc if not (c[0] == 2133.0 and c[1] in (408, 612, 816, 1020))],
             hold=[c for c in allc if c[0] == 2133.0 and c[1] in (408, 612, 816, 1020)], with_c=True,
             fix_beta=None, extra=None)
    P["P6 train 16 + campaign_g2 4 pts@1020 (20 records, 18 cells) / hold 2133x8"] = \
        dict(train=no2133, hold=row2133, with_c=True, fix_beta=None, extra="campaign_g2")
    return P


def _solve_beta_for_dpred(d, train, target_ms, weight, with_c=True, lo=0.05, hi=6.0):
    """P1 family: dpred(beta) = B(beta)*(2.133^-b - 3.199^-b) is monotone in beta; bisect."""
    def dpred(be):
        p = fit_han(d, train=train, with_c=with_c, weight=weight, fix_beta=be, polish=False)
        return (predict(p, 1020, 2133.0) - predict(p, 1020, 3199.0)) / 1e3, p
    f_lo, f_hi = dpred(lo)[0], dpred(hi)[0]
    if not (min(f_lo, f_hi) <= target_ms <= max(f_lo, f_hi)):
        return None
    for _ in range(28):   # 6 / 2^28 -> beta to ~2e-8
        mid = 0.5 * (lo + hi)
        f_mid, _ = dpred(mid)
        if (f_mid - target_ms) * (f_lo - target_ms) <= 0:
            hi, f_hi = mid, f_mid
        else:
            lo, f_lo = mid, f_mid
    return 0.5 * (lo + hi)


def reproduce(keys=("compute_us", "response_us"), weights=("abs", "rel")):
    print("Reported numbers (supplement; main text):")
    for _, txt, v in REPORTED:
        print(f"  {txt:<58} {v:+.2f}" if isinstance(v, float) else f"  {txt:<58} {v}")
    g2c = cells_dict(_load_campaign_g2("cproxyv2", "compute_us"))
    g2r = cells_dict(_load_campaign_g2("cproxyv2", "response_us"))
    dm_g2 = (g2c[(2133.0, 1020)] - g2c[(3199.0, 1020)]) / 1e3
    dm_g2r = (g2r[(2133.0, 1020)] - g2r[(3199.0, 1020)]) / 1e3
    print(f"\nMeasured inversion at GPU 1020, T(2133)-T(3199):")
    print(f"  campaign_g2/partA (1000 iters, the 'full campaign'): compute_us {dm_g2:+.2f} ms "
          f"({g2c[(2133.0, 1020)] / 1e3:.2f} vs {g2c[(3199.0, 1020)] / 1e3:.2f}), response_us {dm_g2r:+.2f} ms"
          f"   <- the reported -1.70 (16.93 vs 18.63)")
    dm_eb = {}
    for key in keys:
        eb = cells_dict(_load_estimator_break("cproxyv2", key))
        dm_eb[key] = (eb[(2133.0, 1020)] - eb[(3199.0, 1020)]) / 1e3
        print(f"  estimator_break (300 iters, the fit grid), {key}: {dm_eb[key]:+.2f} ms "
              f"({eb[(2133.0, 1020)] / 1e3:.2f} vs {eb[(3199.0, 1020)] / 1e3:.2f})")
    print(f"\nFitter: numpy grid{' + scipy polish' if HAVE_SCIPY else ' (scipy not importable)'}; A,B >= 0.")
    print("Columns: in-sample / held-out median |rel err| % (mn, vit); held-out median/max % (gemv, gemm); "
          "gemm dpred = Han T(2133)-T(3199) @ GPU 1020 [ms]")
    hdr = (f"{'protocol':<88} {'key':<11} {'w':<3} | {'mn in/out':>11} | {'vit in/out':>11} | "
           f"{'gemv med/max':>13} | {'gemm med/max':>13} | {'gemm dpred':>10} | {'gemm a,b':>9}")
    print("\n" + hdr)
    print("-" * len(hdr))
    results = []
    for key in keys:
        eb = {m: cells_dict(_load_estimator_break(m, key)) for m in ("mobilenet", "vit", "proxy", "cproxyv2")}
        g2 = {m: cells_dict(_load_campaign_g2(m, key)) for m in eb}
        for w in weights:
            for pname, pr in _protocols().items():
                row = {"protocol": pname, "key": key, "weight": w,
                       "n_train": len(pr["train"]) + (4 if pr["extra"] else 0)}
                for m in eb:
                    d = dict(eb[m])
                    train = list(pr["train"])
                    if pr["extra"] == "campaign_g2":
                        # merge the 4 catalogue points; duplicated (2133/3199, 1020) keys take the median
                        merged = defaultdict(list)
                        for k in train:
                            merged[k].append(d[k])
                        for k, v in g2[m].items():
                            merged[k].append(v)
                        d = {k: float(np.median(v)) for k, v in merged.items()}
                        d.update({k: eb[m][k] for k in pr["hold"]})
                        train = sorted(merged)
                    fb = pr["fix_beta"]
                    if fb == "solve":
                        fb = _solve_beta_for_dpred(d, train, 2.67, w) if m == "cproxyv2" else None
                        if m != "cproxyv2":
                            # for the other workloads there is no reported dpred to pin beta with;
                            # report the family's range instead (beta 0.3 .. 3.0)
                            fam = []
                            for be in (0.3, 0.5, 1.0, 1.5, 2.0, 3.0):
                                pp = fit_han(d, train=train, with_c=True, weight=w, fix_beta=be, polish=False)
                                so = err_stats(score(pp, d, pr["hold"]))
                                fam.append((err_stats(score(pp, d, train))[0], so[0], so[1]))
                            row[m] = {"in_med": fam[0][0], "in_max": float("nan"),
                                      "out_med": float("nan"), "out_max": float("nan"),
                                      "out_med_range": (min(f[1] for f in fam), max(f[1] for f in fam)),
                                      "out_max_range": (min(f[2] for f in fam), max(f[2] for f in fam)),
                                      "alpha": float("nan"), "beta": float("nan"), "C": float("nan")}
                            continue
                        if fb is None:
                            row[m] = {"in_med": float("nan"), "in_max": float("nan"), "out_med": float("nan"),
                                      "out_max": float("nan"), "alpha": float("nan"), "beta": float("nan"), "C": float("nan")}
                            row["gemm_dpred"] = float("nan")
                            continue
                    p = fit_han(d, train=train, with_c=pr["with_c"], weight=w, fix_beta=fb)
                    ins = err_stats(score(p, d, train))
                    out = err_stats(score(p, d, pr["hold"]))
                    row[m] = {"in_med": ins[0], "in_max": ins[1], "out_med": out[0], "out_max": out[1],
                              "alpha": p["alpha"], "beta": p["beta"], "C": p["C"], "sse": p["sse_grid"]}
                    if m == "cproxyv2":
                        row["gemm_dpred"] = (predict(p, 1020, 2133.0) - predict(p, 1020, 3199.0)) / 1e3
                results.append(row)
                mn, vt, px = row["mobilenet"], row["vit"], row["proxy"]

                def rng(r, k):
                    return (f"{r[k + '_range'][0]:.1f}-{r[k + '_range'][1]:.1f}" if k + "_range" in r
                            else f"{r[k]:.1f}")
                print(f"{pname:<88} {key:<11} {w:<3} | "
                      f"{mn['in_med']:4.1f}/{rng(mn, 'out_med'):<8} | {vt['in_med']:4.1f}/{rng(vt, 'out_med'):<8} | "
                      f"{rng(px, 'out_med'):>9}/{rng(px, 'out_max'):<9} | "
                      f"{row['cproxyv2']['out_med']:5.1f}/{row['cproxyv2']['out_max']:<6.1f} | "
                      f"{row['gemm_dpred']:+9.2f} | "
                      f"{row['cproxyv2']['alpha']:.2f},{row['cproxyv2']['beta']:.2f}")
    # P1 non-identifiability demonstration
    print("\n== P1 non-identifiability: train {3199, 665.6} only, C free -> training SSE is flat in beta; "
          "the 2133 prediction is not determined by the data ==")
    for key in keys:
        for m in ("cproxyv2", "mobilenet"):
            d = cells_dict(_load_estimator_break(m, key))
            train = [k for k in d if k[0] != 2133.0]
            hold = [k for k in d if k[0] == 2133.0]
            line = []
            for be in (0.3, 0.5, 1.0, 1.5, 2.0, 3.0):
                p = fit_han(d, train=train, fix_beta=be, polish=False)
                dp = (predict(p, 1020, 2133.0) - predict(p, 1020, 3199.0)) / 1e3
                line.append(f"b={be:g}: SSE {p['sse_grid']:.4g} dpred {dp:+.2f} hold-med {err_stats(score(p, d, hold))[0]:.1f}%")
            print(f"  {PRETTY[m]:<18} {key:<11} " + " | ".join(line))
    # closest protocol per reported number
    print("\n== reported vs reproduced (closest protocol per number; distance in the number's own unit) ==")
    getters = {
        "grid": lambda r: r["n_train"],
        "mn_in": lambda r: r["mobilenet"]["in_med"], "mn_out": lambda r: r["mobilenet"]["out_med"],
        "vit_in": lambda r: r["vit"]["in_med"], "vit_out": lambda r: r["vit"]["out_med"],
        "gemv_med": lambda r: r["proxy"]["out_med"], "gemv_max": lambda r: r["proxy"]["out_max"],
        "gemm_med": lambda r: r["cproxyv2"]["out_med"], "gemm_max": lambda r: r["cproxyv2"]["out_max"],
        "gemm_dpred": lambda r: r["gemm_dpred"],
    }
    print(f"{'number':<58} {'reported':>9} {'closest':>9}  protocol / key / weight")
    for rid, txt, v in REPORTED:
        if rid == "gemm_dmeas":
            print(f"{txt:<58} {v:>9.2f} {dm_g2:>9.2f}  campaign_g2/partA compute_us (fit grid estimator_break gives "
                  f"{dm_eb['compute_us']:+.2f})")
            continue
        g = getters[rid]
        cands = [r for r in results if not math.isnan(g(r))]
        best = min(cands, key=lambda r: abs(g(r) - v))
        print(f"{txt:<58} {v:>9.2f} {g(best):>9.2f}  {best['protocol'].split(' ')[0]} / {best['key']} / {best['weight']}")
    return results


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------


def _parse_emc_list(s):
    return [float(x) for x in s.split(",")] if s else []


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--board", choices=["nano", "nx"])
    ap.add_argument("--model", help="mobilenet|vit|proxy(gemv)|cproxyv2(gemm)|resnet50|yolov8s|...")
    ap.add_argument("--key", default="response_us", choices=["response_us", "compute_us"],
                    help="p50 source (pick scripts use response_us; analyze_estimator.py uses compute_us)")
    ap.add_argument("--source", default="auto", help="auto | all | comma-separated source names")
    ap.add_argument("--sweep-dir", default=None,
                    help="read the cells from this directory instead of the named sources "
                         "([nx_][est_]emc<tok>_gpu<f>_<model>.json; e.g. results/nx_validation_r33)")
    ap.add_argument("--train-emc", default="", help="EMC levels (MHz) to train on, e.g. 3199,665.6")
    ap.add_argument("--holdout-emc", default="", help="EMC levels (MHz) to hold out and score")
    ap.add_argument("--weight", default="abs", choices=["abs", "rel"])
    ap.add_argument("--no-c", action="store_true", help="fit the pure Han form without the offset C")
    ap.add_argument("--allow-negative", action="store_true", help="do not constrain A, B >= 0")
    ap.add_argument("--fix-beta", type=float, default=None, help="pin beta (2-level fallback; not identifiable)")
    ap.add_argument("--deploy-emc", type=float, default=2133.0, help="EMC level for the governor comparison")
    ap.add_argument("--D-us", type=float, default=12000.0)
    ap.add_argument("--Mg", type=float, default=0.0, help="probe guard (us) for the pick_han demo")
    ap.add_argument("--band-us", type=float, default=0.0)
    ap.add_argument("--save", help="write params JSON here")
    ap.add_argument("--inventory", action="store_true")
    ap.add_argument("--reproduce", action="store_true")
    ap.add_argument("--loco", action="store_true",
                    help="also score leave-one-cell-out: refit on all other cells, predict each cell in turn "
                         "(numpy grid fitter, no scipy polish)")
    ap.add_argument("--loco-table", action="store_true",
                    help="Supp. S3 per-workload table for the four Nano estimator_break workloads: full-grid "
                         "in-sample and leave-one-cell-out median/max |rel err| under one protocol (numpy only)")
    ap.add_argument("--no-scipy", action="store_true", help="never polish with scipy (numpy grid fitter only)")
    ap.add_argument("--beta-max", type=float, default=None,
                    help="upper end of the numpy grid's beta search (fitter default 3.0; --loco-table uses 6.0 "
                         "so the GEMM optimum beta ~ 3.65 is reachable without scipy)")
    a = ap.parse_args(argv)

    if a.loco_table:
        loco_table(a.key, a.weight, nonneg=not a.allow_negative, with_c=not a.no_c,
                   **({"beta_max": a.beta_max} if a.beta_max is not None else {}))
        if not (a.board and a.model):
            return 0
    if a.inventory or (not a.reproduce and not a.loco_table and not (a.board and a.model)):
        inventory(a.key)
        if not a.reproduce and not (a.board and a.model):
            return 0
    if a.reproduce:
        reproduce()
        if not (a.board and a.model):
            return 0

    model = canon(a.model)
    if a.sweep_dir:
        cells = load_sweep_dir(a.sweep_dir, model, a.key)
        if not cells:
            sys.exit(f"no *emc*_gpu*_{model}.json cells under {a.sweep_dir}")
    else:
        cells = load_cells(a.board, model, a.key, a.source)
    if not cells:
        sys.exit(f"no cells for board={a.board} model={model} source={a.source}"
                 + (" (no multi-GPU sweep exists for this model on this board; --source all shows the "
                    "single-GPU catalogue cells, which cannot identify alpha)" if a.source == "auto" else ""))
    d = cells_dict(cells)
    srcs = sorted({c.source for c in cells})
    per = defaultdict(set)
    for c in cells:
        per[c.f_emc].add(c.f_gpu)
    print(f"== {a.board.upper()} {PRETTY.get(model, model)}: {len(cells)} records / {len(d)} distinct cells "
          f"from {', '.join(srcs)} (p50 {a.key}) ==")
    for e in sorted(per, reverse=True):
        print(f"  EMC {e:g} MHz: {len(per[e])} GPU points {sorted(per[e])}")
    train_emc = _parse_emc_list(a.train_emc)
    hold_emc = _parse_emc_list(a.holdout_emc)
    if train_emc:
        train = [k for k in d if any(abs(k[0] - e) < 1e-6 for e in train_emc)]
    else:
        train = [k for k in d if not any(abs(k[0] - e) < 1e-6 for e in hold_emc)]
    hold = [k for k in d if any(abs(k[0] - e) < 1e-6 for e in hold_emc)]
    n_emc = len({k[0] for k in train})
    if n_emc < 3 and a.fix_beta is None and not a.no_c:
        have = sorted({k[0] for k in train}, reverse=True)
        missing = [e for e in (3199.0, 2133.0, 665.6) if e not in have]
        print(f"\nNOT IDENTIFIABLE: training set has {n_emc} EMC level(s) {[f'{e:g}' for e in have]}; "
              f"beta needs >= 3.  Sweep {', '.join(f'{e:g}' for e in missing)} MHz on this board/model first "
              f"(or pass --fix-beta / --no-c / --source all to see a provisional fit).")
        return 2
    brange = {"beta_range": (0.05, a.beta_max)} if a.beta_max is not None else {}
    p = fit_han(d, train=train, with_c=not a.no_c, weight=a.weight, nonneg=not a.allow_negative,
                fix_beta=a.fix_beta, polish=False if a.no_scipy else "auto", **brange)
    print(f"\nfit ({p['fitter']}; {p['n_train']} cells, {p['n_emc_levels']} EMC x {p['n_gpu_levels']} GPU levels; "
          f"weight={a.weight}; A,B>=0={not a.allow_negative})")
    print(f"  T_hat(f_gpu,f_emc) = {p['A']:.1f}*(f_gpu/1000)^-{p['alpha']:.3f} + {p['B']:.1f}*(f_emc/1000)^-{p['beta']:.3f}"
          f" + {p['C']:.1f}   [us]")
    if p.get("fix_beta") is not None:
        print(f"  (beta PINNED at {p['fix_beta']}; not identifiable from the data)")
    ins = err_stats(score(p, d, train))
    print(f"  in-sample |rel err|: median {ins[0]:.1f}%  max {ins[1]:.1f}%  mean {ins[2]:.1f}%")
    if hold:
        out = score(p, d, hold)
        st = err_stats(out)
        print(f"  held-out EMC {a.holdout_emc}: median {st[0]:.1f}%  max {st[1]:.1f}%  mean {st[2]:.1f}%")
        for k, meas, pred_, e in out:
            print(f"    EMC {k[0]:g} GPU {k[1]:>4}: measured {meas / 1e3:7.2f} ms  Han {pred_ / 1e3:7.2f} ms  ({e * 100:+.1f}%)")
    if a.loco:
        lo = loco(d, keys=train, with_c=not a.no_c, weight=a.weight, nonneg=not a.allow_negative,
                  fix_beta=a.fix_beta, **brange)
        st = err_stats(lo)
        worst = max(lo, key=lambda r: abs(r[3]))
        print(f"  leave-one-cell-out over the {len(lo)} training cells (numpy grid, no polish): "
              f"median {st[0]:.1f}%  max {st[1]:.1f}% at EMC {worst[0][0]:g} GPU {worst[0][1]}  mean {st[2]:.1f}%")
        for k, meas, pred_, e, _ in lo:
            print(f"    EMC {k[0]:g} GPU {k[1]:>4}: measured {meas / 1e3:7.2f} ms  LOCO {pred_ / 1e3:7.2f} ms  ({e * 100:+.1f}%)")
    top = max(k[1] for k in d)
    if (2133.0, top) in d and (3199.0, top) in d:
        dm = (d[(2133.0, top)] - d[(3199.0, top)]) / 1e3
        dp = (predict(p, top, 2133.0) - predict(p, top, 3199.0)) / 1e3
        print(f"  T(2133)-T(3199) at GPU {top}: measured {dm:+.2f} ms, Han {dp:+.2f} ms "
              f"({'SIGN OK' if np.sign(dm) == np.sign(dp) else 'WRONG SIGN'})")
    # governor view: Han vs two-cell at the deployment EMC
    de = a.deploy_emc
    if any(abs(k[0] - de) < 1e-6 for k in d):
        k2, b2 = two_cell_kb(d, de)
        gpus = sorted({k[1] for k in d if abs(k[0] - de) < 1e-6})
        print(f"\n  deployment EMC {de:g} MHz: T_hat per clock (measured / Han / two-cell endpoint refit) [ms]")
        for g in gpus:
            print(f"    {g:>5} MHz: {d[(de, g)] / 1e3:7.2f} / {predict(p, g, de) / 1e3:7.2f} / {(k2 * 1e3 / g + b2) / 1e3:7.2f}")
        grid = tuple(gpus)
        fprobe = max(gpus)
        pk, sl = pick_han(p, de, a.D_us, a.Mg, grid=grid, fprobe=fprobe, band_us=a.band_us)
        pk2 = next((g for g in grid if a.Mg <= (a.D_us - (k2 * 1e3 / g + b2) * 1.035) / (fprobe / g) - a.band_us), 0)
        print(f"  pick at D={a.D_us / 1e3:g} ms, Mg={a.Mg:g} us, band={a.band_us:g} us: Han -> {pk or 'INFEASIBLE'} "
              f"(slack {sl:.0f} us); two-cell -> {pk2 or 'INFEASIBLE'}")
    if a.save:
        os.makedirs(os.path.dirname(os.path.abspath(a.save)), exist_ok=True)
        p_out = dict(p, board=a.board, model=model, key=a.key, sources=srcs)
        json.dump(p_out, open(a.save, "w"), indent=1)
        print(f"\nparams -> {a.save}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""E3 admission decision from one fresh probe (protocol E3_protocol.md §1, §4 steps 3-4).

Pure function + CLI.  The maths is a verbatim transcription of the frozen
pipeline and is checked against it by --selftest:

  M_g      = max - p50 of the probe response_us            banded_pick.py:13, fresh_pick.py:12
  T_hat(f) = k/f + b, two-cell endpoint refit at EMC 2133   banded_pick.py:14-17 (min/max gpu cell)
  m*(f)    = (D - T_hat(f)*R) / (f_probe/f)                 banded_pick.py:21  (M_g boundary at f)
  point    f_pt = lowest f in G with T_hat(f)*R + M_g*f_probe/f <= D     fresh_pick.py:17-18  (h = 0)
  banded   f_b  = lowest f in G with M_g <= m*(f) - h,  h = 400 us        banded_pick.py:20-23
  R = 1.035  (banded_pick.py:18)      gamma = 0 for vision (protocol §1)

NOTE on the band: banded_pick.py thresholds the *probe-scale* margin
m*(f) - M_g against h.  In the protocol's slack notation
s(f) = D - [T_hat(f)*R + M_g*f_probe/f] this is  s(f) >= h * f_probe/f,
i.e. the band is worth h at f_probe and h*f_probe/f at lower clocks.  The
JSON reports both numbers per f (s_us, margin_mg_us) so the paper can state
whichever it names; the pick itself follows the code.

Usage
  admit_pick.py PROBE.csv --D-ms 12 [--board nx|nano] [--primary mobilenet|vit]
                [--cells C1.json C2.json ...] [--grid 510,...] [--fprobe N]
                [--h-us 400] [--R 1.035] [--out pick.json] [--brief]
      --brief prints one line "Mg_us f_pt f_b" (0 = INFEASIBLE) for the shell loop;
      otherwise the full JSON goes to stdout.  --out also writes the JSON.
  admit_pick.py --score CONTRACT.csv --D-ms 12 [--brief]
      miss rate re-scored offline as response_us > D (the Table-II convention).
  admit_pick.py --selftest
      re-runs the pick on results/banded_loop/probe_t*.csv and
      results/fresh_loop/vprobe_t*.csv, compares with the committed picks
      (dep_t<t>_g<pick> file names, fresh_loop/decisions.csv) and with the
      stdout of banded_pick.py / fresh_pick.py themselves.
  admit_pick.py --mg-us 4400 --D-ms 24 --board nx --primary vit
      planning aid (no probe): evaluate the rule for a hypothetical M_g, e.g.
      to place the knife-edge D of protocol §3.  Not used by admit_loop.sh.
"""
import argparse, csv, glob, json, os, re, subprocess, sys
import numpy as np

LAB = "/home/dk/mobile/jetson-latency-lab"
R_MULT = 1.035            # fresh_pick.py:9, banded_pick.py:18
H_US = 400.0              # banded_pick.py:10 (fixed, not re-tuned per condition)
GRID = {"nx":   [510, 612, 714, 816, 918, 1020, 1122, 1173],   # fresh_pick.py:17
        "nano": [306, 408, 510, 612, 714, 816, 918, 1020]}     # protocol §1 (devfreq table)
FPROBE = {"nx": 1173, "nano": 1020}                            # protocol §1
CELLS = {  # two-cell solo model sources, protocol §1 row "latency model"
    ("nx", "mobilenet"): [f"{LAB}/results/nx_validation/nx_est_emc2133_gpu306_mobilenet.json",
                          f"{LAB}/results/nx_validation/nx_est_emc2133_gpu1173_mobilenet.json"],
    ("nx", "vit"):       [f"{LAB}/results/nx_validation/nx_est_emc2133_gpu306_vit.json",
                          f"{LAB}/results/nx_validation/nx_est_emc2133_gpu1173_vit.json"],
    ("nano", "mobilenet"): [f"{LAB}/results/nano_grid/mobilenet/est_emc2133_gpu306.json",
                            f"{LAB}/results/nano_grid/mobilenet/est_emc2133_gpu1020.json"],
    ("nano", "vit"):       [f"{LAB}/results/nano_grid/vit/est_emc2133_gpu306.json",
                            f"{LAB}/results/nano_grid/vit/est_emc2133_gpu1020.json"],
}
DEVFREQ = "/sys/devices/platform/17000000.gpu/devfreq/17000000.gpu/available_frequencies"


def load_probe(path):
    """response_us column of a harness.infer_bench CSV (banded_pick.py:12)."""
    return np.array([float(x["response_us"]) for x in csv.DictReader(open(path)) if x.get("response_us")])


def load_cells(files):
    """{gpu_MHz: response p50 us} from estimator-sweep JSONs (banded_pick.py:14-15)."""
    return {int(re.search(r"gpu(\d+)", os.path.basename(f)).group(1)): json.load(open(f))["response_us"]["p50"]
            for f in files}


def fit_2cell(cells):
    """Endpoint refit k/f + b through the min and max gpu cells (banded_pick.py:16-17)."""
    lo, hi = min(cells), max(cells)
    k = (cells[lo] - cells[hi]) / (1e3 / lo - 1e3 / hi)
    b = cells[hi] - k * (1e3 / hi)
    return k, b, lo, hi


def evaluate(v, cells, D_us, grid, fprobe, h_us=H_US, R=R_MULT, Mg_override=None):
    """Point (h=0) and banded (h=h_us) picks plus per-f slack. Pure function."""
    if Mg_override is None:
        p50, vmax = float(np.median(v)), float(v.max())          # banded_pick.py:13
        Mg = vmax - p50
    else:
        p50, vmax, Mg = None, None, float(Mg_override)
    k, b, lo, hi = fit_2cell(cells)
    per_f, f_pt, f_b = [], None, None
    for g in grid:
        That = k * 1e3 / g + b
        mstar = (D_us - That * R) / (fprobe / g)                 # banded_pick.py:21
        s = D_us - (That * R + Mg * fprobe / g)                  # protocol §1 slack s(f)
        ok0 = That * R + Mg * fprobe / g <= D_us                 # fresh_pick.py:18   (h = 0)
        okb = Mg <= mstar - h_us                                 # banded_pick.py:22  (h = h_us)
        if ok0 and f_pt is None:
            f_pt = g
        if okb and f_b is None:
            f_b = g
        per_f.append({"f": g, "T_hat_us": That, "s_us": s, "mstar_us": mstar,
                      "margin_mg_us": mstar - Mg, "h_in_s_us": h_us * fprobe / g,
                      "ok_h0": bool(ok0), "ok_hb": bool(okb)})
    inf = "INFEASIBLE"
    return {
        "n_cycles": None if v is None else int(len(v)),
        "p50_us": p50, "max_us": vmax, "Mg_us": Mg,
        "D_us": D_us, "fprobe": fprobe, "grid": list(grid), "R": R, "h_us": h_us,
        "cells": {str(g): cells[g] for g in sorted(cells)}, "cell_lo": lo, "cell_hi": hi,
        "k_us_GHz": k, "b_us": b,
        "per_f": per_f,
        "f_pt": f_pt if f_pt is not None else inf,
        "f_b": f_b if f_b is not None else inf,
        "band_active": f_pt != f_b,
        "s_at_f_pt_us": next((r["s_us"] for r in per_f if r["f"] == f_pt), None),
        "s_at_f_b_us": next((r["s_us"] for r in per_f if r["f"] == f_b), None),
    }


def detect_board():
    try:
        top = max(int(x) for x in open(DEVFREQ).read().split()) // 1_000_000
        return "nx" if top >= 1100 else "nano"
    except Exception:
        return None


def score(path, D_us):
    v = load_probe(path)
    m = int(np.count_nonzero(v > D_us))
    return {"csv": path, "n": int(len(v)), "D_us": D_us, "misses": m, "miss_rate": m / len(v)}


def selftest():
    """Reproduce the committed E3 picks. Exit 1 on any mismatch."""
    py = sys.executable
    cells = load_cells(glob.glob(f"{LAB}/results/nx_validation/nx_est_emc2133_gpu*_mobilenet.json"))
    grid, fp, D = GRID["nx"], FPROBE["nx"], 12000.0
    bad = 0
    rows = []
    # (A) banded held-out round: committed pick = g<f> in dep_t<t>_g<f>.json; also banded_pick.py stdout
    for pf in sorted(glob.glob(f"{LAB}/results/banded_loop/probe_t*.csv"),
                     key=lambda p: int(re.search(r"probe_t(\d+)", p).group(1))):
        t = re.search(r"probe_t(\d+)", pf).group(1)
        r = evaluate(load_probe(pf), cells, D, grid, fp)
        deps = glob.glob(f"{LAB}/results/banded_loop/dep_t{t}_g*.json")
        committed = int(re.search(r"_g(\d+)\.json", deps[0]).group(1)) if deps else None
        ref = subprocess.run([py, f"{LAB}/campaign/banded_pick.py", pf], capture_output=True, text=True).stdout.split()
        ok = (str(r["f_b"]) == str(committed)) and ref == [f"{r['Mg_us']:.0f}", str(r["f_b"])]
        bad += not ok
        rows.append(f"banded_loop t{t:>2}: Mg={r['Mg_us']:7.0f} f_pt={r['f_pt']:>4} f_b={r['f_b']:>4} "
                    f"committed={committed} banded_pick.py='{' '.join(ref)}' {'OK' if ok else 'MISMATCH'}")
    # (B) round-1 fresh probes: committed Mg/pick in decisions.csv (h = 0); also fresh_pick.py stdout
    dec = {int(x["trial"]): x for x in csv.DictReader(open(f"{LAB}/results/fresh_loop/decisions.csv")) if x["arm"] == "vision"}
    for pf in sorted(glob.glob(f"{LAB}/results/fresh_loop/vprobe_t*.csv"),
                     key=lambda p: int(re.search(r"vprobe_t(\d+)", p).group(1))):
        t = int(re.search(r"vprobe_t(\d+)", pf).group(1))
        r = evaluate(load_probe(pf), cells, D, grid, fp)
        ref = subprocess.run([py, f"{LAB}/campaign/fresh_pick.py", "vision", pf], capture_output=True, text=True).stdout.split()
        d = dec.get(t)
        ok = d is not None and f"{r['Mg_us']:.0f}" == d["Mg"] and str(r["f_pt"]) == d["pick"] \
            and ref == [f"{r['Mg_us']:.0f}", str(r["f_pt"])]
        bad += not ok
        rows.append(f"fresh_loop  t{t:>2}: Mg={r['Mg_us']:7.0f} f_pt={r['f_pt']:>4} f_b={r['f_b']:>4} "
                    f"decisions.csv=({d['Mg'] if d else '-'},{d['pick'] if d else '-'}) "
                    f"fresh_pick.py='{' '.join(ref)}' {'OK' if ok else 'MISMATCH'}")
    k, b, lo, hi = fit_2cell(cells)
    print(f"selftest: NX mobilenet two-cell refit cells {lo}/{hi}: k={k:.4f} us*GHz b={b:.4f} us  (python {py})")
    print("\n".join(rows))
    print(f"selftest: {len(rows)} probes, {bad} mismatches -> {'PASS' if bad == 0 else 'FAIL'}")
    return 0 if bad == 0 else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("probe", nargs="?", help="probe CSV from harness.infer_bench (response_us column)")
    ap.add_argument("--D-ms", type=float, help="contract deadline D in ms")
    ap.add_argument("--board", choices=["nx", "nano"], help="default: autodetect from devfreq available_frequencies")
    ap.add_argument("--primary", choices=["mobilenet", "vit"], help="selects the default solo cells")
    ap.add_argument("--cells", nargs="+", help="solo-sweep JSONs; min/max gpu of the set are the two cells")
    ap.add_argument("--grid", help="comma list of candidate MHz (default per board)")
    ap.add_argument("--fprobe", type=int, help="probe clock MHz (default per board)")
    ap.add_argument("--h-us", type=float, default=H_US)
    ap.add_argument("--R", type=float, default=R_MULT)
    ap.add_argument("--mg-us", type=float, help="planning aid: use this M_g instead of a probe")
    ap.add_argument("--out", help="write the JSON here as well")
    ap.add_argument("--brief", action="store_true", help="print 'Mg_us f_pt f_b' (0 = INFEASIBLE) / miss_rate only")
    ap.add_argument("--score", metavar="CONTRACT_CSV", help="re-score a contract CSV against --D-ms")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()

    if a.selftest:
        sys.exit(selftest())
    if a.D_ms is None:
        ap.error("--D-ms is required")
    D_us = a.D_ms * 1000.0
    if a.score:
        r = score(a.score, D_us)
        print(f"{r['miss_rate']:.4f}" if a.brief else json.dumps(r, indent=1))
        return
    board = a.board or detect_board()
    if board is None and (a.grid is None or a.fprobe is None or a.cells is None):
        ap.error("--board could not be autodetected; pass --board (or --grid/--fprobe/--cells)")
    if a.cells:
        cell_files = a.cells
    elif a.primary:
        cell_files = CELLS[(board, a.primary)]
    else:
        ap.error("--cells or --primary is required")
    grid = [int(x) for x in a.grid.split(",")] if a.grid else GRID[board]
    fprobe = a.fprobe or FPROBE[board]
    if a.mg_us is not None:
        v = None
    elif a.probe:
        v = load_probe(a.probe)
    else:
        ap.error("PROBE.csv or --mg-us is required")
    res = evaluate(v, load_cells(cell_files), D_us, grid, fprobe, a.h_us, a.R, a.mg_us)
    res.update({"probe_csv": a.probe, "board": board, "primary": a.primary, "cell_files": cell_files,
                "rule": "banded_pick.py:20-23 / fresh_pick.py:17-18 (see admit_pick.py docstring)"})
    if a.out:
        with open(a.out, "w") as f:
            json.dump(res, f, indent=1)
    if a.brief:
        z = lambda x: 0 if x == "INFEASIBLE" else x
        print(f"{res['Mg_us']:.0f} {z(res['f_pt'])} {z(res['f_b'])}")
    else:
        print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()

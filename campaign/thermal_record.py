#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Thermal-gate record of the sessions behind the paper (2026-10-29).

Gate (identical in govrep_rev*.sh, sweep_rate.sh, govrep_emcfree.sh, admit_loop.sh, admit_refsweep.sh, launch_ext.sh,
horizon_ext.sh; every session from September 2026 on): before each gated unit the harness takes the maximum over all
readable thermal zones (cat thermal_zone*/temp | sort -rn | head -1) and waits in 20 s steps, at most 60 (20 min), until
it reads <= 55 C, then proceeds regardless.  Gated units: an interleaved pair (govrep_rev*), an endpoint-refit repeat
(refit mode), a rate-matched sweep cell (sweep_rate.sh), an EMC-free re-run round, an admission trial, one whole
reference sweep (admit_refsweep.sh: one gate, eight cells), a launch (launch_ext.sh, horizon_ext.sh).  Cells inside a
unit run back to back.  Earlier sessions: the June 11 Nano characterization campaign (run_g2.sh) waited before each cell
for the gpu-thermal zone to read < 55 C (10 s steps, at most 5 min, reading not logged); the July 3 session repeats of
Supp. S3 (c6_reps.sh) ran this same gate once per session (three sessions); every other June/July session behind the
paper ran ungated.  tegrastats traces, where
kept, hold run temperatures and are summarized for disclosure.

What the logs hold:
  * characterization sessions (results/govrep_rev, refit33, nx_validation_r33, e5b_live): session.log holds the reading
    at the END of every cell ('t=<millideg>'); the json meta.timestamp is written when the run ends.  The idle time
    before run k is start_k - end_{k-1} with start_k = ts_k - dur_k, dur = (warmup + iters)/hz; the session's per-cell
    overhead is the smallest idle time; a gated start whose idle time exceeds it by >= 15 s waited (reported on the
    gate's 20 s grid), by >= 1190 s timed out (the gate checks up to 60 times with a 20 s sleep after each failed check:
    a pass at the 60th check waits 1180 s, a timeout 1200 s).  The first unit's wait, and the first unit after a session
    restart (a second 'start=' line), is timed from that start line (absent in the sweep log: not reconstructable).
    Gated starts are identified from the file names (new pair / new (EMC, repeat) / every sweep cell / new round).
  * launches, admission trials, reference sweeps: 'tgate=<millideg>' is re-read right after the gate (the unit's start
    reading), 'tend='/'t=' at the end, and a 'WARN thermal gate timed out' line precedes each timed-out unit.
Read-only over results/ (needs the session.log files; the pre-September tegrastats summary scans every
results/**/*.tegra, including results/e4_replication and results/partb_ext from the Zenodo deposit, so extract those
in place first; a WARN line names either folder when it is absent).  Prints the markdown record; --md also writes it
to a file and --tex writes the supplementary float (label tab:thermal):
    python3 campaign/thermal_record.py [--tex OUT.tex] [--md OUT.md]
"""
import argparse
import glob
import json
import os
import re
from collections import defaultdict
from datetime import datetime

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LAB = os.path.dirname(HERE)
RES = os.path.join(LAB, "results")
TO_S = 1190.0
WAIT_S = 15.0
# Zenodo folders (10.5281/zenodo.21236659) whose tegrastats traces the pre-September summary includes
ZENODO_TEGRA = (("e4_replication", "e4_replication_nano.tar.gz"), ("partb_ext", "partb_ext_nano.tar.gz"))


def _pair(n):
    return int(n.split("_p")[1])
def _repeat(n):
    m = re.match(r"emc(\d+)_g\d+_r(\d+)", n); return (m.group(1), m.group(2))
def _round(n):
    return int(re.match(r"e[\d.]+_g\d+_r(\d+)", n).group(1))
def _cell(n):
    return n

CHAR = [
    ("govrep_rev/nx", "NX MobileNetV2, session 1 (Table II)", "pair", _pair),
    ("govrep_rev/nx_mobilenet", "NX MobileNetV2, session 2 (Table II)", "pair", _pair),
    ("govrep_rev/nx_mobilenet_s3", "NX MobileNetV2, session 3 (Table II)", "pair", _pair),
    ("govrep_rev/nx_vit", "NX ViT-Small (Table~\\ref{tab:breadthgov})", "pair", _pair),
    ("govrep_rev/nx_yolov8s", "NX YOLOv8s (Table~\\ref{tab:breadthgov})", "pair", _pair),
    ("govrep_rev/nx_resnet50", "NX ResNet-50 (Table~\\ref{tab:breadthgov})", "pair", _pair),
    ("govrep_rev/nano", "Nano MobileNetV2, session 1 (Table II)", "pair", _pair),
    ("govrep_rev/nano_mobilenet_s2", "Nano MobileNetV2, session 2 (Table II)", "pair", _pair),
    ("govrep_rev/nano_vit", "Nano ViT-Small (Table~\\ref{tab:nanovit})", "pair", _pair),
    ("refit33/nx", "NX MobileNetV2 endpoint refit", "repeat", _repeat),
    ("refit33/nx_vit", "NX ViT-Small endpoint refit", "repeat", _repeat),
    ("refit33/nx_yolov8s", "NX YOLOv8s endpoint refit", "repeat", _repeat),
    ("refit33/nx_resnet50", "NX ResNet-50 endpoint refit", "repeat", _repeat),
    ("refit33/nano", "Nano MobileNetV2 33\\,Hz re-profile (\\S S2, Nano rows)", "repeat", _repeat),
    ("refit33/nano_vit", "Nano ViT-Small endpoint refit", "repeat", _repeat),
    ("nx_validation_r33", "NX rate-matched sweeps (Table~\\ref{tab:e5han})", "cell", _cell),
    ("e5b_live/nx", "EMC-free re-run (Table~\\ref{tab:e5blive})", "round", _round),
]
LAUNCH = [
    ("launch_ext/nx/mobilenet_vit10_g1020_D12", "NX fresh launches, Sept.\\ (main text \\S VII)"),
    ("horizon_ext/nx/H1000", "NX decode launches, $H{=}1{,}000$ (Table~\\ref{tab:e8horizon})"),
    ("horizon_ext/nx/H3000", "NX decode launches, $H{=}3{,}000$ (Table~\\ref{tab:e8horizon})"),
]
E3 = [("e3_nano", "Nano admission trials"), ("e3_nx", "NX admission trials"), ("e3_nx_r33", "NX rate-matched re-runs (Table~\\ref{tab:r33})")]


def runs_in(dirpath):
    out = []
    for jp in sorted(glob.glob(os.path.join(dirpath, "*.json"))):
        try:
            m = json.load(open(jp))["meta"]
        except Exception:
            continue
        if "timestamp" not in m or "iters" not in m:
            continue
        ts = datetime.fromisoformat(m["timestamp"].replace("Z", "+00:00"))
        out.append(dict(name=os.path.basename(jp)[:-5], ts=ts,
                        dur=(int(m["iters"]) + int(m.get("warmup", 0))) / float(m["hz"])))
    out.sort(key=lambda r: r["ts"])
    return out


def end_temps(session_log):
    """run name -> end-of-cell reading (C); handles 'g816_p1: OK t=', 'p1 g816: OK t=' and 'vit emc665 gpu306: OK t='."""
    T = {}
    if not os.path.isfile(session_log):
        return T
    for line in open(session_log):
        m = re.match(r"^\s*(.+?): (?:OK|FAIL) t=(\d+)\s*$", line)
        if not m:
            continue
        key = m.group(1).strip()
        mm = re.match(r"p(\d+) g(\d+)$", key)
        if mm:
            key = f"g{mm.group(2)}_p{mm.group(1)}"
        mm = re.match(r"(\S+) emc(\d+) gpu(\d+)$", key)
        if mm:
            key = f"nx_est_emc{mm.group(2)}_gpu{mm.group(3)}_{mm.group(1)}"
        T[key] = int(m.group(2)) / 1000.0
    return T


def session_starts(session_log):
    """every top-level 'start=' line (a session restarted in the same directory logs a second one)"""
    out = []
    if not os.path.isfile(session_log):
        return out
    for line in open(session_log):
        if line.startswith(" "):
            continue
        m = re.search(r"\bstart=(\S+)", line)
        if m:
            out.append(datetime.fromisoformat(m.group(1)))
    return out


def char_row(rel, unit_key):
    d = os.path.join(RES, rel)
    R = runs_in(d)
    slog = os.path.join(d, "session.log")
    T = end_temps(slog)
    idle = [None] + [(R[k]["ts"] - R[k - 1]["ts"]).total_seconds() - R[k]["dur"] for k in range(1, len(R))]
    ovh = min(x for x in idle[1:] if x is not None)
    waited, timeouts, waits, anomalies = 0, 0, [], 0
    def classify(w):
        nonlocal waited, timeouts
        if w >= TO_S:
            timeouts += 1; waits.append(1200.0)
        elif w >= WAIT_S:
            waited += 1; waits.append(round(w / 20.0) * 20.0)
    H = session_starts(slog)
    h0 = H[0] if H else None
    first = None
    if h0 is not None:
        first = (R[0]["ts"] - h0).total_seconds() - R[0]["dur"] - ovh
        classify(first)
    gated = 1
    for k in range(1, len(R)):
        is_gated = unit_key(R[k]["name"]) != unit_key(R[k - 1]["name"])
        w = idle[k] - ovh
        start_k = R[k]["ts"].timestamp() - R[k]["dur"]
        rs = [h for h in H[1:] if R[k - 1]["ts"].timestamp() < h.timestamp() <= start_k]
        if rs:   # first unit after a restart: timed from the restart's start line, like the session's first unit
            w = start_k - rs[-1].timestamp() - ovh
        if is_gated:
            gated += 1
            classify(w)
        elif w >= WAIT_S:
            anomalies += 1
    tv = list(T.values())
    return dict(runs=len(R), gated=gated, waited=waited, timeouts=timeouts, waits=waits, anomalies=anomalies,
                first=first, tend=(min(tv), max(tv)) if tv else None, over=sum(1 for v in tv if v > 55.0) if tv else None,
                matched=sum(1 for r in R if r["name"] in T))


def launch_row(rel):
    starts, ends, to, pending = [], [], 0, False
    for line in open(os.path.join(RES, rel, "session.log")):
        if "WARN thermal gate timed out" in line:
            pending = True; continue
        m = re.match(r"\s*t\d+:.*tgate=(\d+) tend=(\d+)", line)
        if m:
            starts.append(int(m.group(1)) / 1000.0); ends.append(int(m.group(2)) / 1000.0); to += pending; pending = False
    return dict(n=len(starts), timeouts=to, starts=starts, tstart=(min(starts), max(starts)), tend=(min(ends), max(ends)),
                over=sum(1 for v in starts if v > 55.0))


def e3_rows(group):
    rows = []
    for d in sorted(glob.glob(os.path.join(RES, group, "*"))):
        slog = os.path.join(d, "session.log")
        if not os.path.isdir(d) or not os.path.isfile(slog):
            continue
        tstart, tend, rend, to_t, to_r, pending, nref, to_starts, pass_over = [], [], [], 0, 0, False, 0, [], []
        last_ref_gate = None
        for line in open(slog):
            if "WARN thermal gate timed out" in line:
                pending = True; continue
            m = re.match(r"\s*t\d+:.*tgate=(\d+) tend=(\d+)", line)
            if m:
                v = int(m.group(1)) / 1000.0
                tstart.append(v); tend.append(int(m.group(2)) / 1000.0)
                if pending:
                    to_t += 1; to_starts.append(v)
                elif v > 55.0:
                    pass_over.append(v)
                pending = False
                continue
            m = re.match(r"\s*ref g\d+:.*tgate=(\d+) t=(\d+)", line)
            if m:
                g = int(m.group(1)) / 1000.0
                if g != last_ref_gate:
                    nref += 1; to_r += pending; pending = False; last_ref_gate = g
                rend.append(int(m.group(2)) / 1000.0)
        if not tstart and not rend:
            continue
        rows.append(dict(cond=os.path.basename(d), trials=len(tstart), refs=nref, ref_cells=len(rend), to_t=to_t, to_r=to_r,
                         to_starts=to_starts, pass_over=pass_over,
                         tstart=(min(tstart), max(tstart)) if tstart else None, tend=(min(tend), max(tend)) if tend else None,
                         rend=(min(rend), max(rend)) if rend else None, over=sum(1 for v in tstart if v > 55.0)))
    return rows


def early_traces():
    """tegrastats traces of the sessions before September 2026: start/end/max of the hottest zone per trace."""
    st, en, mx, dirs = [], [], [], set()
    for f in glob.glob(os.path.join(RES, "**", "*.tegra"), recursive=True):
        vals = []
        for line in open(f, errors="ignore"):
            v = [float(x) for x in re.findall(r"@(-?\d+(?:\.\d+)?)C", line)]
            v = [x for x in v if -20 < x < 150]
            if v:
                vals.append(max(v))
        if vals:
            st.append(vals[0]); en.append(vals[-1]); mx.append(max(vals)); dirs.add(os.path.relpath(f, RES).split("/")[0])
    return dict(n=len(st), start=(min(st), max(st)), over=sum(1 for x in st if x > 55.0), max=max(mx), dirs=sorted(dirs))


def g2_traces():
    st = []
    for f in glob.glob(os.path.join(RES, "campaign_g2", "**", "*.tegra"), recursive=True):
        for line in open(f, errors="ignore"):
            v = [float(x) for x in re.findall(r"@(-?\d+(?:\.\d+)?)C", line) if -20 < float(x) < 150]
            if v:
                st.append(max(v)); break
    return dict(n=len(st), start=(min(st), max(st)), over=sum(1 for x in st if x > 55.0))


def rng(t):
    return "--" if t is None else (f"{t[0]:.1f}" if abs(t[0] - t[1]) < 0.05 else f"{t[0]:.1f}--{t[1]:.1f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tex", default=None, help="write the LaTeX table (tab:thermal) here")
    ap.add_argument("--md", default=None, help="write the markdown record here as well")
    a = ap.parse_args()
    absent = [(d, t) for d, t in ZENODO_TEGRA if not os.path.isdir(os.path.join(RES, d))]
    if absent:
        print("WARN " + " and ".join(f"results/{d}" for d, _ in absent)
              + (" is" if len(absent) == 1 else " are") + " absent, so the pre-September tegrastats summary"
              " leaves out " + ("its" if len(absent) == 1 else "their") + " traces: extract "
              + " and ".join(t for _, t in absent) + " of the Zenodo deposit 10.5281/zenodo.21236659 into results/")
    md = ["## Thermal-gate record (campaign/thermal_record.py; do not edit by hand)", "",
          "### Characterization sessions (end-of-cell readings; waits reconstructed)", "",
          "| session | runs | gated unit | gated starts | waited / timed out | wait s | end-of-cell T °C | cells > 55 °C | first-unit wait s |",
          "|---|---:|---|---:|---|---|---|---|---|"]
    A, S = [], {}
    for rel, label, unit, key in CHAR:
        r = char_row(rel, key); S[rel] = r
        assert r["anomalies"] <= 1, (rel, r["anomalies"])
        wt = "--" if not r["waits"] else (f"{min(r['waits']):.0f}" if min(r["waits"]) == max(r["waits"]) else f"{min(r['waits']):.0f}--{max(r['waits']):.0f}")
        te = rng(r["tend"]); ov = f"{r['over']}/{r['runs']}"
        fw = "n/r" if r["first"] is None else f"{max(r['first'], 0):.0f}"
        md.append(f"| {label} | {r['runs']} | {unit} | {r['gated']} | {r['waited']} / {r['timeouts']} | {wt.replace('--','–')} | {te.replace('--','–')} | {ov} | {fw} |")
        A.append(f"    {label} & {r['runs']} & {unit} & {r['gated']} & {r['waited']}\\,/\\,{r['timeouts']} & {wt} & {te} & {ov} \\\\")
    md += ["", "### Launches, admission trials and reference sweeps (start reading logged after the gate)", "",
           "| session | units | timed out | start T °C | end T °C | ref.-cell end T °C | starts > 55 °C |", "|---|---|---:|---|---|---|---|"]
    B = []
    LG = {}
    for rel, label in LAUNCH:
        r = launch_row(rel); LG[rel] = r
        md.append(f"| {label} | {r['n']} launches | {r['timeouts']} | {rng(r['tstart'])} | {rng(r['tend'])} | – | {r['over']}/{r['n']} |")
        B.append(f"    {label} & {r['n']} launches & {r['timeouts']} & {rng(r['tstart'])} & {rng(r['tend'])} & -- & {r['over']}/{r['n']} \\\\")
    E = {}
    for group, label in E3:
        rows = e3_rows(group); E[group] = rows
        B.append(f"    \\multicolumn{{7}}{{@{{}}l}}{{\\textit{{{label}}}}} \\\\")
        for r in rows:
            units = f"{r['trials']} trials" + (f", {r['refs']} sweep" if r["refs"] else "")
            md.append(f"| {label} {r['cond']} | {units} | {r['to_t'] + r['to_r']} | {rng(r['tstart'])} | {rng(r['tend'])} | {rng(r['rend'])} | {r['over']}/{r['trials']} |")
            B.append(f"    \\quad {r['cond']} & {units} & {r['to_t'] + r['to_r']} & {rng(r['tstart'])} & {rng(r['tend'])} & {rng(r['rend'])} & {r['over']}/{r['trials']} \\\\")
    # headline numbers
    nx_dep = [k for k in S if k.startswith("govrep_rev/nx")]; nano_dep = [k for k in S if k.startswith("govrep_rev/nano")]
    dep_to = sum(S[k]["timeouts"] for k in nx_dep + nano_dep); dep_gs = sum(S[k]["gated"] for k in nx_dep + nano_dep)
    nx_lo, nx_hi = min(S[k]["tend"][0] for k in nx_dep), max(S[k]["tend"][1] for k in nx_dep)
    refit_b = ["refit33/nx_vit", "refit33/nx_yolov8s", "refit33/nx_resnet50"]
    refit_to = sum(S[k]["timeouts"] for k in refit_b)
    sw = S["nx_validation_r33"]
    e3nx = E["e3_nx"]; nx_trials = sum(r["trials"] for r in e3nx); nx_to = sum(r["to_t"] for r in e3nx); nx_to_r = sum(r["to_r"] for r in e3nx)
    to_st = [v for r in e3nx for v in r["to_starts"]]; po = [v for r in e3nx for v in r["pass_over"]]
    nano_po = [v for r in E["e3_nano"] for v in r["pass_over"]]; nano_trials = sum(r["trials"] for r in E["e3_nano"])
    r33_po = [v for r in E["e3_nx_r33"] for v in r["pass_over"]]; r33_trials = sum(r["trials"] for r in E["e3_nx_r33"])
    r33_to = sum(r["to_t"] + r["to_r"] for r in E["e3_nx_r33"])
    assert r33_to == 0 and r33_trials == 30, (r33_to, r33_trials)
    nano_to = sum(r["to_t"] + r["to_r"] for r in E["e3_nano"])
    n2 = next(r for r in e3nx if r["cond"] == "N2")
    l_over = [(rel, v) for rel in LG for v in LG[rel]["starts"] if v > 55.0]
    mx_logs = max([S[k]["tend"][1] for k in S] + [LG[k]["tend"][1] for k in LG] +
                  [v[1] for g in E for r in E[g] for v in (r["tend"], r["rend"]) if v])
    mx_vision = max([S[k]["tend"][1] for k in S] + [LG[LAUNCH[0][0]]["tend"][1]] +
                    [v[1] for g in E for r in E[g] for v in (r["tend"], r["rend"]) if v])
    early = early_traces(); g2 = g2_traces()
    first_waits = {k: S[k]["first"] for k in S if S[k]["first"] is not None and S[k]["first"] >= WAIT_S}
    head = (f"deployment sessions: {dep_gs} gated pair starts, {dep_to} timeouts; NX end-of-cell {nx_lo:.1f}--{nx_hi:.1f} C; "
            f"breadth refits: {refit_to} timeouts (every 3199 MHz repeat); sweeps: {sw['timeouts']} of {sw['gated']} timed out, {sw['waited']} waited; "
            f"re-run: {S['e5b_live/nx']['timeouts']} timeouts; NX admission: {nx_to} of {nx_trials} trials timed out (N2 {n2['to_t']}; logged starts "
            f"{min(to_st):.2f}--{max(to_st):.2f} C), {len(po)} more passed but logged {min(po):.2f}--{max(po):.2f} C, {nx_to_r} reference sweeps timed out; "
            f"Nano admission: {nano_to} timeouts, {len(nano_po)} of {nano_trials} passed but logged {min(nano_po):.2f}--{max(nano_po):.2f} C; "
            f"launches: {sum(LG[k]['timeouts'] for k in LG)} timeouts, {len(l_over)} starts > 55 C ({', '.join(f'{v:.2f}' for _, v in l_over)}); "
            f"max in these logs {mx_logs:.1f} C (vision {mx_vision:.1f}); first-unit waits >= 15 s: "
            + ", ".join(f"{k} {v:.0f} s" for k, v in first_waits.items())
            + f"; pre-September tegrastats traces: {early['n']} (dirs {', '.join(early['dirs'])}), starts {early['start'][0]:.1f}--{early['start'][1]:.1f} C, "
            f"{early['over']} above 55, max {early['max']:.1f} C; June 11 Nano cells: {g2['n']} traces, starts {g2['start'][0]:.1f}--{g2['start'][1]:.1f} C, {g2['over']} above 55")
    md += ["", "Headline: " + head]
    cap = (
        "Thermal-gate record of the sessions behind the main text and supplement from September 2026 on. Before every "
        "gated unit (an interleaved pair, an endpoint-refit repeat, a rate-matched sweep cell, an EMC-free re-run round, an "
        "admission trial, a whole eight-cell reference sweep, a launch) the harness takes the hottest reading over the "
        "board's thermal zones and waits in 20\\,s steps, for at most 20\\,min, until it reads ${\\le}55\\,^\\circ$C, then "
        "proceeds regardless; the cells inside a unit run back to back. \\textbf{Top:} characterization sessions, whose "
        "harness logs only the end-of-cell reading: the wait before each gated start is reconstructed from the run "
        "timestamps and the session's start line (idle time minus the session's per-cell overhead; ${\\ge}15$\\,s counts "
        "as waited, ${\\ge}1190$\\,s as timed out, since a pass at the gate's last check waits 1180\\,s and a timeout "
        "1200\\,s; a restarted session's first unit is timed from its own start line; the sweep log has no start line, so "
        "its first cell is counted but not classified), the waits are given on "
        "the gate's 20\\,s grid (--: none) and the last column counts cells ending above $55\\,^\\circ$C. "
        "\\textbf{Bottom:} launches, admission trials and reference sweeps, whose harness logs the reading right after the "
        "gate (the unit's start), the end reading and every timed-out gate. "
        f"The gate held before all {dep_gs} deployment pairs on both SKUs, but the NX runs at ${nx_lo:.1f}$--${nx_hi:.1f}\\,^\\circ$C "
        "within a pair, and an idle NX with the EMC locked at 3199\\,MHz often stays above $55\\,^\\circ$C for the whole wait: "
        f"the gate timed out before every 3199\\,MHz repeat of the three NX breadth refits (the MobileNetV2 and Nano refits "
        f"passed), before {sw['timeouts']} of the 40 sweep cells at 3199\\,MHz (none at 665.6 or 2133\\,MHz) and before rounds "
        f"2--10 of the EMC-free re-run. At the deployment point it timed out before {nx_to} of the {nx_trials} registered NX "
        f"admission trials ({n2['to_t']} of them in N2; logged starts ${min(to_st):.1f}$--${max(to_st):.1f}\\,^\\circ$C), "
        f"none of their reference sweeps and none of the {r33_trials} rate-matched re-runs; another {len(po)} of those {nx_trials} "
        f"trials, {len(r33_po)} of the {r33_trials} re-runs and {len(nano_po)} of the {nano_trials} Nano trials "
        f"passed it but read ${min(po + r33_po + nano_po):.1f}$--${max(po + r33_po + nano_po):.1f}\\,^\\circ$C a "
        "moment later (the last column counts all starts above $55\\,^\\circ$C). The highest reading in these logs is "
        f"${mx_logs:.1f}\\,^\\circ$C, at the end of a $3{{,}}000$-token decode launch (${mx_vision:.1f}\\,^\\circ$C among the "
        "vision sessions); the gate reads the CPU, GPU, SOC and \\texttt{tj} zones on both boards and the CV zones on the "
        "NX, and no reading reaches the $70\\,^\\circ$C passive trip of the zones other than \\texttt{tj} or the "
        "$74\\,^\\circ$C active trip of \\texttt{tj}, whose lowest active trip ($35\\,^\\circ$C) lies below every reading "
        "(trip points read from each board's sysfs, the NX's on 2026-09-15). Sessions before September 2026 ran without "
        "this per-unit gate: the June 11 Nano characterization cells waited before each cell for the GPU zone to read "
        "below $55\\,^\\circ$C (at most 5\\,min, reading not logged), the July 3 session repeats of "
        "\\S\\ref{sec:rq1} ran this gate once per session, and the other June and July sessions ran ungated. "
        f"Where they kept a \\texttt{{tegrastats}} trace ({early['n']} cells and runs), the hottest zone read "
        f"${early['start'][0]:.1f}$--${early['start'][1]:.1f}\\,^\\circ$C at the start ({early['over']} starts above "
        f"$55\\,^\\circ$C) and at most ${early['max']:.1f}\\,^\\circ$C.")
    L = [r"\begin{table*}[!t]", r"  \centering", "  \\caption{" + cap + "}", r"  \label{tab:thermal}",
         r"  \footnotesize", r"  \setlength{\tabcolsep}{4pt}",
         r"  \begin{tabular}{@{}>{\raggedright\arraybackslash}p{150pt}rlrlll@{}}", r"    \toprule",
         r"    characterization session & runs & gated unit (starts) & & waited\,/\,timed out & wait (s) & end-of-cell $T$ ($^\circ$C), ${>}55$ \\",
         r"    \midrule"]
    # fold the 'gated starts' count into the unit column to save a column
    for row in A:
        cells = [c.strip() for c in row.strip().rstrip("\\").split("&")]
        label, runs, unit, gs, wt_to, wait, te, ov = cells
        L.append(f"    {label} & {runs} & {unit} ({gs}) & & {wt_to} & {wait} & {te}, {ov} \\\\")
    L += [r"    \bottomrule", r"  \end{tabular}", r"  \par\medskip",
          r"  \begin{tabular}{@{}>{\raggedright\arraybackslash}p{150pt}lrlll l@{}}", r"    \toprule",
          r"    launch / trial session & gated units & timed out & start $T$ ($^\circ$C) & end $T$ ($^\circ$C) & ref.\ cells, end $T$ & starts ${>}55$ \\",
          r"    \midrule"] + B + [r"    \bottomrule", r"  \end{tabular}", r"\end{table*}"]
    if a.tex:
        open(a.tex, "w").write("\n".join(L) + "\n")
    if a.md:
        open(a.md, "w").write("\n".join(md) + "\n")
    print("\n".join(md))
    written = [p for p in (a.tex, a.md) if p]
    if written:
        print(f"\nwrote {' and '.join(written)}")


if __name__ == "__main__":
    main()

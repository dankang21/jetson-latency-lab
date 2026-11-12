#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""E3 transfer study of the frozen dispersion band: one row per pre-stated condition (Supp. transfer
table) and the verdict counts by setting (main-text Table VII, transfer-campaign rows).

One row per condition of docs/protocols/E3_protocol.md section 3, in protocol order (C0, N1-N9 on the
NX; A1-A4, A5r, A6r, A7 on the Nano), from campaign/e3_stats.py --json for both boards: every endpoint
is e3_stats' recomputation from the trial files under results/e3_<board>/<cond>/ (protocol sections
5-6: tau = 2 %, run-level bootstrap B = 10 000, seed 20260915).  A5 and A6 as first run were
classified at the time as protocol-execution failures (section 8, 2026-09-15: 7 of 10 A6 probes and one
A5 probe were indistinguishable from the solo cell; the cause was not established); they are listed
apart and their re-runs A5r/A6r are the rows.  Columns:
  n          completed launches (k/N while short of the protocol's N)
  P1         admitted launches with f_b < f_oracle (one-sided 95 % Clopper-Pearson UB, %)
  P2         pooled banded miss % [run-level bootstrap one-sided 95 % UB]; '!' when above tau
  > tau      contracts with per-launch miss > tau (min-max per-launch miss, %)
  band       launches with f_pt != f_b out of n (removed point admissions that missed tau / re-run)
  accept.    admitted / completed
  util.      median f_b - f_oracle in grid bins (f_oracle, MHz)
  corr       Pearson / Spearman of the probe margin M_g with the banded miss m_b over admitted launches
  verdict    section 6 as restated on 2026-09-15: supported iff P1 = 0 and P2 holds (not supported if
             P1 > 0 or P2 fails; band idle when f_pt = f_b on every launch), asserted against e3_stats'
             own verdict; and under the original wording frozen on 2026-09-14, which in addition
             required every point-policy admission the band removed to have failed (a band-idle
             condition meets it vacuously)
Then the diagnostic probe~solo launches, the superseded A5/A6, the per-board summary and the counts by
setting: the 14 transfer conditions (all but the C0 control and the N9/A7 mechanism tests) and the
groups of Table VII (phase-sweeping MobileNetV2: N1-N4, A1-A4; integer-period-multiple ViT-S: N5-N8,
A5r, A6r; 7 Hz mechanism tests: N9, A7), each with launches, admissions, declines, pooled-miss range
(conditions within tau), launches over tau, P1 (launches and conditions with P1 = 0) and the verdict
counts under both wordings.

Read-only on results/ (python3 + numpy, through e3_stats.py in a subprocess); prints to stdout,
writes nothing.
Usage: e3_table.py
"""
import argparse
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
LAB = os.path.dirname(HERE)
E3_STATS = os.path.join(HERE, "e3_stats.py")

SHORT = {"mobilenet": "MNv2", "vit": "ViT-S", "yolov8s": "YOLOv8s", "resnet50": "ResNet-50"}
BOARD = {"nx": "NX", "nano": "Nano"}
# protocol section 3 (+ section 8 amendments), in protocol order:
# id, board, primary, primary Hz, tenant, tenant Hz, D ms, planned launches
PROTOCOL = [
    ("C0", "nx", "mobilenet", 33, "vit", 10, 12.0, 10),
    ("N1", "nx", "mobilenet", 33, "yolov8s", 5, 12.0, 20),
    ("N2", "nx", "mobilenet", 33, "yolov8s", 10, 12.0, 20),
    ("N3", "nx", "mobilenet", 33, "resnet50", 5, 12.0, 20),
    ("N4", "nx", "mobilenet", 33, "resnet50", 10, 12.0, 20),
    ("N5", "nx", "vit", 20, "yolov8s", 5, 32.5, 20),
    ("N6", "nx", "vit", 20, "yolov8s", 10, 32.5, 20),
    ("N7", "nx", "vit", 20, "resnet50", 5, 32.5, 20),
    ("N8", "nx", "vit", 20, "resnet50", 10, 32.5, 20),
    ("N9", "nx", "vit", 20, "yolov8s", 7, 32.5, 10),
    ("A1", "nano", "mobilenet", 33, "yolov8s", 5, 13.0, 10),
    ("A2", "nano", "mobilenet", 33, "yolov8s", 10, 13.0, 10),
    ("A3", "nano", "mobilenet", 33, "resnet50", 5, 13.0, 10),
    ("A4", "nano", "mobilenet", 33, "resnet50", 10, 13.0, 10),
    ("A5r", "nano", "vit", 20, "yolov8s", 5, 37.5, 10),
    ("A6r", "nano", "vit", 20, "yolov8s", 10, 37.5, 10),
    ("A7", "nano", "vit", 20, "yolov8s", 7, 37.5, 10),
]
SUPERSEDED = {"A5r": "A5", "A6r": "A6"}     # shown re-run -> superseded first run (section 8, 2026-09-15)
MECHANISM = ("A7", "N9")                     # pre-stated incommensurate-rate mechanism tests
CONTROL = ("C0",)                            # calibration condition re-run as a positive control
GROUPS = [
    ("phase-sweeping MobileNetV2 (N1-N4, A1-A4)", ("N1", "N2", "N3", "N4", "A1", "A2", "A3", "A4")),
    ("integer-period-multiple ViT-S (N5-N8, A5r, A6r)", ("N5", "N6", "N7", "N8", "A5r", "A6r")),
    ("7 Hz mechanism tests (N9, A7)", ("N9", "A7")),
    ("control C0", ("C0",)),
]


# ----------------------------------------------------------------- inputs ----
def run_stats(board):
    """(cond -> block, header) from e3_stats --json; (None, reason) when the board has no results directory."""
    r = subprocess.run([sys.executable, "-B", E3_STATS, "--board", board, "--json"], capture_output=True, text=True)
    if r.returncode != 0:
        msg = (r.stderr or r.stdout).strip()
        if "no such directory" in msg:
            return None, msg
        sys.exit(f"e3_stats --board {board} failed:\n{msg}")
    d = json.loads(r.stdout)
    return {c["cond"]: c for c in d["conditions"]}, d


def collect(log=print):
    """e3_stats blocks of both boards: (stats, hdr, tau, B); log() receives one header line per board."""
    stats, hdr = {}, {}
    for board in ("nano", "nx"):
        stats[board], hdr[board] = run_stats(board)
        if stats[board] is None:
            log(f"# {BOARD[board]}: no results directory ({hdr[board]}) - every condition pending")
        else:
            h = hdr[board]
            log(f"# {BOARD[board]}: {h['results']} tau={h['tau']} B={h['B']} seed={h['seed']}; "
                f"conditions found {sorted(stats[board])}")
    taus = {h["tau"] for h in hdr.values() if isinstance(h, dict)}
    Bs = {h["B"] for h in hdr.values() if isinstance(h, dict)}
    assert len(taus) <= 1 and len(Bs) <= 1, (taus, Bs)
    tau, B = (taus.pop() if taus else 0.02), (Bs.pop() if Bs else 10000)
    return stats, hdr, tau, B


def check_block(p, c):
    """The condition directory must describe the pre-stated condition (a wrong directory is a build error)."""
    cond, board, primary, phz, co, chz, D, _ = p
    got = (c["board"], c["primary"], str(c["pri_hz"]), c["co"], str(c["co_hz"]), float(c["D_ms"]))
    want = (board, primary, str(phz), co, str(chz), D)
    assert got == want, f"{cond}: results/{c['dir']} describes {got}, protocol says {want}"


def state(p, c):
    """'pending' (no completed launch / absent), 'in progress' (short of N) or 'complete'."""
    if c is None or c.get("error") or c["n_complete"] == 0:
        return "pending"
    return "in progress" if c["n_complete"] < p[7] else "complete"


def iter_conditions(stats, log=print):
    """(protocol row, e3_stats block or None, state) in protocol order; an e3_stats error makes the
    condition pending (logged as it is reached)."""
    for p in PROTOCOL:
        cond, board = p[0], p[1]
        c = (stats[board] or {}).get(cond)
        if c is not None and c.get("error"):
            log(f"  {cond}: e3_stats error: {c['error']} -> pending")
            c = None
        if c is not None:
            check_block(p, c)
        yield p, c, state(p, c)


# ---------------------------------------------------------------- verdicts ----
def pc(x, nd=1):
    """Fraction -> percent string; '100' when every cycle missed (99.95+)."""
    v = 100.0 * x
    return "100" if v >= 99.95 else f"{v:.{nd}f}"


def verdict_plain(c):
    """Section 6 verdict from the endpoint fields; asserted against e3_stats' own verdict string."""
    P1, P2, Bn = c["P1"], c["P2"], c["band"]
    if P2 is None:
        s = "no admissions"
    elif not P2["passed"]:
        s = "not supported (P1, P2)" if P1["x"] else "not supported (P2)"
    elif P1["x"] is None:
        s = "pending reference"
    elif P1["x"] > 0:
        s = "not supported (P1)"
    elif Bn["n_active"] == 0:
        s = "band idle"
    else:
        s = "supported"
    head = {"supported": "SUPPORTED", "band idle": "BAND IDLE", "pending reference": "pending reference",
            "no admissions": "no admissions"}.get(s, "NOT SUPPORTED")
    assert c["verdict"].startswith(head), (c["cond"], s, c["verdict"])
    return s


def verdict_original(c):
    """The verdict under the protocol's original 'supported' wording (frozen 2026-09-14): as restated, and in
    addition every point-policy admission the band removed (each re-run as its own contract) must have failed;
    a band-idle condition removed none and meets it vacuously."""
    v = verdict_plain(c)
    if v in ("supported", "band idle"):
        Bn = c["band"]
        return "supported" if Bn["n_pt_fail"] == Bn["n_pt"] else "not supported (band removed passing admissions)"
    return v


def superseded_facts(stats_nano):
    """Per superseded original (A5, A6): P2, probes indistinguishable from the solo cell, lowest banded pick."""
    out = []
    for shown, orig in SUPERSEDED.items():
        c = stats_nano.get(orig) if stats_nano else None
        if c and not c.get("error") and c["P2"] is not None:
            fmin = min(t["f_b"] for t in c["trials"] if t.get("complete") and isinstance(t.get("f_b"), int))
            out.append(dict(shown=shown, orig=orig, available=True, P2=c["P2"]["pooled"],
                            n_solo=len(c["diag"]["probe_solo_ks"]), n_complete=c["n_complete"], fmin=fmin))
        else:
            out.append(dict(shown=shown, orig=orig, available=False))
    return out


# ---------------------------------------------------------------- summary ----
def ranges(ids):
    """'N1, N2, N3' -> 'N1-N3' for consecutive numbered ids of one prefix; other ids are listed as they are."""
    out, run = [], []

    def flush():
        if run:
            out.append(run[0][0] if len(run) == 1 else f"{run[0][0]}-{run[-1][0]}")
        run.clear()
    for cid in ids:
        m = re.fullmatch(r"([A-Za-z]+)(\d+)", cid)
        if m and run and run[-1][1] == m.group(1) and run[-1][2] + 1 == int(m.group(2)):
            run.append((cid, m.group(1), int(m.group(2))))
        else:
            flush()
            if m:
                run.append((cid, m.group(1), int(m.group(2))))
            else:
                out.append(cid)
    flush()
    return ", ".join(out)


def summary(entries):
    """One main-text-ready line: per board, supported / not supported / band idle / in progress / pending."""
    out = []
    for board in ("nano", "nx"):
        es = [e for e in entries if e[0][1] == board]
        sup, nots, idle, prog, pend = [], [], [], [], []
        for p, c, st in es:
            cond, N = p[0], p[7]
            if st == "pending":
                pend.append(cond)
                continue
            if st == "in progress":
                prog.append(f"{cond} ({c['n_complete']}/{N} launches)")
                continue
            P1, P2 = c["P1"], c["P2"]
            p1 = "no ref." if P1["x"] is None else f"P1 {P1['x']}/{P1['n']}"
            p2 = "P2 undefined" if P2 is None else f"P2 {pc(P2['pooled'], 2)} %"
            v = verdict_plain(c)
            if v == "supported":
                sup.append(f"{cond} {p1}, {p2}")
            elif v == "band idle":
                idle.append(f"{cond}{' mechanism test' if cond in MECHANISM else ''} ({p1}, {p2})")
            else:
                nots.append(f"{cond} ({p1}, {p2})")
        parts = [f"{len(sup)} of {len(es)} supported" + (f" ({'; '.join(sup)})" if sup else "")]
        if nots:
            parts.append("not supported: " + ", ".join(nots))
        if idle:
            parts.append("band idle: " + ", ".join(idle))
        if prog:
            parts.append("in progress: " + ", ".join(prog))
        if pend:
            parts.append("pending: " + ranges(pend))
        out.append(f"{BOARD[board]}: " + "; ".join(parts))
    return " | ".join(out)


def group_counts(entries, conds):
    """Sums over complete conditions of one setting (launches, admissions, declines, pooled-miss range,
    conditions within tau, launches over tau, P1, conditions with P1 = 0, verdict counts)."""
    es = [(p, c) for p, c, st in entries if p[0] in conds and st == "complete"]
    if not es:
        return None
    cs = [c for _, c in es]
    p2 = [c["P2"]["pooled"] for c in cs if c["P2"] is not None]
    return dict(n_cond=len(cs), launches=sum(c["n_complete"] for c in cs), admitted=sum(c["acceptance"]["n"] for c in cs),
                declined=sum(c["n_complete"] - c["acceptance"]["n"] for c in cs),
                p2_min=min(p2) if p2 else None, p2_max=max(p2) if p2 else None,
                within=sum(1 for c in cs if c["P2"] is not None and c["P2"]["passed"]),
                over_tau=sum(c["violations"]["x"] for c in cs if c["violations"] is not None),
                p1=sum(c["P1"]["x"] for c in cs if c["P1"]["x"] is not None),
                p1_zero=sum(1 for c in cs if c["P1"]["x"] == 0),
                restated=sum(1 for c in cs if verdict_plain(c) == "supported"),
                restated_idle=sum(1 for c in cs if verdict_plain(c) == "band idle"),
                original=sum(1 for c in cs if verdict_original(c) == "supported"))


# ------------------------------------------------------------------- main ----
def row_txt(p, c, st):
    cond, board, primary, phz, co, chz, D, N = p
    head = [cond, BOARD[board], f"{SHORT[primary]} {phz} Hz", f"{SHORT[co]} ({chz} Hz)", f"{D:.1f}"]
    if st == "pending":
        return head + [f"0/{N}"] + ["--"] * 7 + ["pending", "pending"]
    n = c["n_complete"]
    P1, P2, V, Bn, A, U, C = (c[k] for k in ("P1", "P2", "violations", "band", "acceptance", "utilization", "corr"))
    p1 = "no ref." if P1["x"] is None else f"{P1['x']}/{P1['n']} ({pc(P1['cp95'])})"
    p2 = "--" if P2 is None else f"{pc(P2['pooled'], 2)} [{pc(P2['boot_ub95'], 2)}]" + ("" if P2["passed"] else " !")
    if V is None:
        lv = "--"
    else:
        rng = pc(V["min"]) if V["min"] == V["max"] else f"{pc(V['min'])}-{pc(V['max'])}"
        lv = f"{V['x']}/{V['n']} ({rng})"
    bd = "0 (idle)" if Bn["n_active"] == 0 else f"{Bn['n_active']}/{n} ({Bn['n_pt_fail']}/{Bn['n_pt']})"
    util = "--" if U is None else f"{U['median']:+g} ({c['reference']['f_oracle']})"
    corr = "--" if C is None or C.get("pearson") is None or C["pearson"] != C["pearson"] else \
        f"{C['pearson']:+.2f} / {C['spearman']:+.2f}"
    vr = "in progress" if st == "in progress" else verdict_plain(c)
    vo = "in progress" if st == "in progress" else verdict_original(c)
    return head + [str(n) if n == N else f"{n}/{N}", p1, p2, lv, bd, f"{A['n']}/{A['N']}", util, corr, vr, vo]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.parse_args()
    stats, hdr, tau, B = collect(log=print)
    entries = list(iter_conditions(stats, log=print))
    cols = ["cond", "board", "primary", "tenant (rate)", "D (ms)", "n", "P1: f_b < f_o (UB %)", "P2: miss % [UB]",
            "launches > tau (min-max %)", "band fired (failed/removed)", "accept.", "util. bins (f_o)",
            "corr M_g, m_b (P / S)", "verdict (restated 2026-09-15)", "verdict (original wording)"]
    print(f"\ntau = {pc(tau, 0)} %, run-level bootstrap B = {B}; '!' = P2 above tau\n")
    print("| " + " | ".join(cols) + " |")
    print("|" + "|".join("---" for _ in cols) + "|")
    for p, c, st in entries:
        print("| " + " | ".join(row_txt(p, c, st)) + " |")
    solo = [(p[0], c) for p, c, st in entries if st != "pending" and c["diag"]["probe_solo_ks"]]
    if solo:
        print("\nprobe indistinguishable from the solo cell (M_g <= 2x the solo cell's own max-p50 at f_probe; diagnostic): "
              + "; ".join(f"{cid} {len(c['diag']['probe_solo_ks'])}/{c['n_complete']} "
                          f"(t{', t'.join(map(str, c['diag']['probe_solo_ks']))})" for cid, c in solo))
    for x in superseded_facts(stats["nano"]):
        if x["available"]:
            print(f"superseded {x['orig']} (re-run as {x['shown']}): P2 {pc(x['P2'], 2)} %, {x['n_solo']}/{x['n_complete']} "
                  f"probes indistinguishable from the solo cell, picks down to {x['fmin']} MHz")
        else:
            print(f"superseded {x['orig']} (re-run as {x['shown']}): not available")
    print("\nsummary: " + summary(entries))
    transfer = tuple(p[0] for p in PROTOCOL if p[0] not in MECHANISM + CONTROL)
    print("\n| setting | conditions | launches | admitted | declined | pooled miss % range (within tau) | "
          "launches > tau | P1 launches | P1 = 0 | supported, restated wording | supported, original wording |")
    print("|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|")
    rows = GROUPS + [(f"{len(transfer)} transfer conditions (all but C0, N9, A7)", transfer),
                     ("all scored conditions, incl. C0", tuple(p[0] for p in PROTOCOL))]
    for name, conds in rows:
        g = group_counts(entries, conds)
        if g is None:
            print(f"| {name} | 0 | - | - | - | - | - | - | - | - | - |")
            continue
        rng = "-" if g["p2_min"] is None else f"{pc(g['p2_min'], 2)}-{pc(g['p2_max'], 2)} ({g['within']}/{g['n_cond']})"
        idle = f" (+{g['restated_idle']} band idle)" if g["restated_idle"] else ""
        print(f"| {name} | {g['n_cond']} | {g['launches']} | {g['admitted']} | {g['declined']} | {rng} | "
              f"{g['over_tau']}/{g['admitted']} | {g['p1']}/{g['admitted']} | {g['p1_zero']}/{g['n_cond']} | "
              f"{g['restated']}/{g['n_cond']}{idle} | {g['original']}/{g['n_cond']} |")


if __name__ == "__main__":
    main()

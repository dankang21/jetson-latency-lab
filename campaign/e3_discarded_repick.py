#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""E3 on the NX: picks re-derived from the probe summaries of the first NX series (2026-09-16).

Post hoc.  The first NX series produced a probe for every trial but no contract, because the pick
script had not been deployed to the board; it was set aside unscored and the admission loops were
re-run on 2026-10-13 (docs/protocols/E3_protocol.md, amendment log).  The probe files of that series
are not in the release tree, but the JSON summaries of 178 of its 180 probes are in the history of
this repository (commit 9383875).  The banded pick of the frozen rule depends on a probe only through
its maximum and its median, which the summaries hold, so the picks that series would have made can
be re-derived with campaign/admit_pick.py and compared with the same reference frontiers as the
registered series (the reference sweeps of 2026-09-16, kept in results/e3_nx).

No contract ran in that series: the aggregate criterion (P2) and the per-launch outcomes cannot be
computed from it, no condition can be shown to be supported in it (under the rule a pick below the
frontier withholds support by itself), and nothing here replaces a registered result.

The script first checks the method on the registered series: it re-derives every registered pick
from the registered probe's JSON summary and compares it with the pick recorded in session.log.

    /usr/bin/python3 -B campaign/e3_discarded_repick.py [--commit 9383875]
"""
import argparse
import json
import os
import re
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import admit_pick as AP          # noqa: E402
import e3_stats as ES            # noqa: E402

LAB = ES.LAB
TAU = 0.02          # miss budget as a fraction, as campaign/e3_stats.py takes it
CONDS = ["C0", "N1", "N2", "N3", "N4", "N5", "N6", "N7", "N8", "N9"]


def git(*a):
    r = subprocess.run(["git", "-C", LAB, *a], capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"git {' '.join(a)} failed: {r.stderr.strip()[:200]}")
    return r.stdout


def pick_from_summary(js, cond, cells):
    mg = float(js["response_us"]["max"]) - float(js["response_us"]["p50"])
    r = AP.evaluate(np.array([0.0]), cells, cond["D_ms"] * 1000.0, cond["grid"], cond["fprobe"],
                    h_us=cond["h_us"], R=cond["R"], Mg_override=mg)
    return mg, r["f_pt"], r["f_b"]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--commit", default="9383875")
    a = ap.parse_args()
    names = git("ls-tree", "-r", "--name-only", a.commit, "--", "results/e3_nx").split()
    print(f"# E3 NX: picks re-derived from the probe summaries at commit {a.commit} (post hoc; no contract ran in that series)")
    print()
    print("| cond | registered series: picks reproduced from the JSON summaries | reference frontier (MHz) | registered: picks below the frontier | "
          "first series: probes | dates (UTC) | M_g range (us) | banded picks | picks below the frontier |")
    print("|---|---|---:|---|---:|---|---|---|---|")
    tot_reg = tot_ok = tot_first = 0
    for c in CONDS:
        cdir = os.path.join(LAB, "results", "e3_nx", c)
        warn = []
        cond = ES.read_condition(cdir, warn)
        cells = AP.load_cells(cond["cell_files"])
        res = ES.analyse(cdir, c, TAU, 2000, 20260914)      # registered analysis (bootstrap size and seed do not enter P1)
        fo = res["reference"]["f_oracle"]                   # lowest grid clock whose reference contract meets tau
        assert res["reference"]["status"] == "complete" and isinstance(fo, int), (c, res["reference"])
        # method check on the registered series
        log_trials, _, _, _ = ES.parse_session_log(os.path.join(cdir, "session.log"))
        ok = n = 0
        for k, t in sorted(log_trials.items()):
            p = os.path.join(cdir, f"t{k}.probe.json")
            if not os.path.isfile(p):
                continue
            _, f_pt, f_b = pick_from_summary(json.load(open(p)), cond, cells)
            n += 1
            ok += int(f_b == t["f_b"] and f_pt == t["f_pt"])
        reg_below = f"{res['P1']['x']}/{res['P1']['n']}"
        # the first series, from the history
        rows = []
        for nm in sorted((x for x in names if x.startswith(f"results/e3_nx/{c}/") and x.endswith(".probe.json")),
                         key=lambda s: int(re.search(r"t(\d+)\.probe", s).group(1))):
            js = json.loads(git("show", f"{a.commit}:{nm}"))
            ts = js["meta"]["timestamp"]
            if not ts.startswith("2026-09-16"):
                continue            # the commit also holds the first probe of the 2026-10-13 re-run of C0
            mg, f_pt, f_b = pick_from_summary(js, cond, cells)
            rows.append((int(re.search(r"t(\d+)\.probe", nm).group(1)), ts, mg, f_pt, f_b))
        if not rows:
            raise SystemExit(f"no probe summary dated 2026-09-16 for {c} at commit {a.commit} "
                             "(the summaries of the first series are held at commit 9383875)")
        adm = [r for r in rows if isinstance(r[4], int)]
        below = sum(1 for r in adm if isinstance(fo, int) and r[4] < fo)
        dist = {}
        for r in rows:
            dist[str(r[4])] = dist.get(str(r[4]), 0) + 1
        days = sorted({r[1][:10] for r in rows})
        print(f"| {c} | {ok}/{n} | {fo} | {reg_below} | {len(rows)} | {', '.join(days)} | {min(r[2] for r in rows):.0f}-{max(r[2] for r in rows):.0f} | "
              + ", ".join(f"{k}: {v}" for k, v in sorted(dist.items(), key=lambda kv: (kv[0] == 'INFEASIBLE', int(kv[0]) if kv[0].isdigit() else 0)))
              + f" | {below}/{len(adm)} |")
        tot_reg += n; tot_ok += ok; tot_first += len(rows)
    print()
    print(f"Registered picks reproduced from the JSON summaries: {tot_ok}/{tot_reg}. Probe summaries of the first series used: {tot_first}.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Every nvpmodel power mode of a board's active nvpmodel configuration file, as per-mode knob rows
(Supp. tab:nvpmodel): cores online, CPU / GPU / EMC caps, TPC power-gating mask, DLA / PVA caps,
the PM_CONFIG DEFAULT mode, and every parameter group a mode sets.

Read-only (python3 only).  Run from anywhere:
    python3 campaign/nvpmodel_modes.py [CONF[:LABEL] ...]
Default: repro/nvpmodel.conf (Orin Nano Super, the active conf of the board, L4T R36.5.0) and
repro/nx/nvpmodel.conf (Orin NX 16 GB) when present.  The NX conf is the stock L4T R36.4.3 file
/etc/nvpmodel/nvpmodel_p3767_0000_super.conf of that board, sha256
7d7e83807e95e68a934a7a33f61cce80b8a02b170de37be286c617e43f826461; any copy can be passed as an argument
(the sha256 of every conf parsed is printed).
Conf values: MAX_FREQ -1 = INT_MAX = uncapped (the hardware maximum then applies, which the conf does
not hold); CPU caps are in kHz, GPU / EMC / DLA / PVA caps in Hz (printed in MHz).  DLA caps are
printed core/falcon and PVA caps VPS/AXI.
"""
import hashlib
import os
import re
import sys
from collections import OrderedDict

LAB = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CONFS = [(os.path.join(LAB, "repro", "nvpmodel.conf"), "Orin Nano"),
                 (os.path.join(LAB, "repro", "nx", "nvpmodel.conf"), "Orin NX")]
CLOCK_GROUPS = ("CPU_ONLINE", "CPU_A78", "GPU", "EMC", "TPC_POWER_GATING", "FBP_POWER_GATING", "DLA", "PVA")


def parse(path):
    modes = OrderedDict(); cur = None; default = None
    for ln_no, ln in enumerate(open(path), 1):
        s = ln.strip()
        if not s or s.startswith("#"): continue
        m = re.match(r"<\s*POWER_MODEL\s+ID=(\d+)\s+NAME=(\S+)\s*>", s)
        if m:
            cur = int(m.group(1)); modes[cur] = dict(name=m.group(2), line=ln_no, kv=OrderedDict()); continue
        m = re.match(r"<\s*PM_CONFIG\s+DEFAULT=(\d+)\s*>", s)
        if m: default = (int(m.group(1)), ln_no); cur = None; continue
        if s.startswith("<"): cur = None; continue
        if cur is not None:
            p = s.split()
            if len(p) == 3: modes[cur]["kv"][(p[0], p[1])] = (p[2], ln_no)
    return modes, default


def summarize(modes):
    rows = []
    for mid, m in modes.items():
        kv = m["kv"]
        online = [k for k in kv if k[0] == "CPU_ONLINE"]
        n_on = sum(1 for k in online if kv[k][0] == "1"); n_all = len(online)
        off = [k[1] for k in online if kv[k][0] == "0"]
        cpu_max = sorted({kv[k][0] for k in kv if k[0].startswith("CPU_A78") and k[1] == "MAX_FREQ"})
        gpu_max = kv.get(("GPU", "MAX_FREQ"), ("?", 0)); emc = kv.get(("EMC", "MAX_FREQ"), ("?", 0))
        tpc = kv.get(("TPC_POWER_GATING", "TPC_PG_MASK"), ("?", 0)); fbp = kv.get(("FBP_POWER_GATING", "FBP_PG_MASK"), ("?", 0))
        other = {f"{k[0]}.{k[1]}": v[0] for k, v in kv.items() if k[0].startswith(("DLA", "PVA"))}
        params = sorted({k[0] for k in kv})
        rows.append(dict(id=mid, name=m["name"], line=m["line"], on=n_on, all=n_all, off=off, cpu_max=cpu_max,
                         gpu_max=gpu_max, emc=emc, tpc=tpc, fbp=fbp, other=other, params=params))
    return rows


def mhz(v):
    v = int(v)
    if v == -1: return "uncapped"
    return f"{v/1e6:g}" if v > 1e7 else f"{v/1e3:g}"


def caps(other):
    """DLA core/falcon and PVA VPS/AXI caps, as the table prints them"""
    if not other:
        return "n/a (no DLA/PVA parameters)"
    out = []
    for unit, a, b in (("DLA0", "DLA0_CORE", "DLA0_FALCON"), ("DLA1", "DLA1_CORE", "DLA1_FALCON"), ("PVA0", "PVA0_VPS", "PVA0_AXI")):
        x, y = other.get(f"{a}.MAX_FREQ"), other.get(f"{b}.MAX_FREQ")
        if x is not None or y is not None:
            out.append(f"{unit} {mhz(x) if x is not None else '?'}/{mhz(y) if y is not None else '?'}")
    return "; ".join(out)


def report(path, label):
    modes, default = parse(path)
    sha = hashlib.sha256(open(path, "rb").read()).hexdigest()
    print(f"## {label}: {os.path.relpath(path, LAB) if path.startswith(LAB) else path} (sha256 {sha})\n"
          f"PM_CONFIG DEFAULT={default[0]} (line {default[1]})")
    rows = summarize(modes)
    for r in rows:
        print(f"id {r['id']} {r['name']:<11} (line {r['line']}): cores {r['on']}/{r['all']} off={r['off']} "
              f"CPU max={[mhz(x) for x in r['cpu_max']]} GPU max={mhz(r['gpu_max'][0])} (l{r['gpu_max'][1]}) "
              f"EMC max={mhz(r['emc'][0])} (l{r['emc'][1]}) TPC_PG={r['tpc'][0]} (l{r['tpc'][1]}) FBP_PG={r['fbp'][0]} "
              f"other={ {k: mhz(v) for k, v in r['other'].items()} }")
        print(f"      params set: {r['params']}")
    print("\n| id: mode | CPU online | CPU max (MHz) | GPU max (MHz) | TPC_PG_MASK | EMC max (MHz) | DLA / PVA caps (MHz) | groups beyond clocks/cores/gating |")
    print("|---|---|---|---|---:|---|---|---|")
    for r in rows:
        cores = f"{r['on']} of {r['all']}" + (f" ({', '.join(r['off'])} off)" if r["off"] else "")
        extra = [p for p in r["params"] if not p.startswith(CLOCK_GROUPS)]
        print(f"| {r['id']}: {r['name']}{' (DEFAULT)' if r['id'] == default[0] else ''} | {cores} | "
              f"{', '.join(mhz(x) for x in r['cpu_max'])} | {mhz(r['gpu_max'][0])} | {r['tpc'][0]} | {mhz(r['emc'][0])} | "
              f"{caps(r['other'])} | {', '.join(extra) or 'none'} |")
    print()


def main():
    args = sys.argv[1:]
    if args and args[0] in ("-h", "--help"):
        print(__doc__)
        return
    confs = [(a.split(":", 1)[0], a.split(":", 1)[1] if ":" in a else os.path.basename(a)) for a in args] or DEFAULT_CONFS
    for path, label in confs:
        if not os.path.isfile(path):
            print(f"## {label}: {path} not present (skipped)\n")
            continue
        report(path, label)


if __name__ == "__main__":
    main()

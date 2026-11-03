#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""EMC / CPU / GPU state of the Orin Nano stock power-mode observation traces (Supp. tab:powermode,
main text Sec. III), parsed from tegrastats: per trace and per nvpmodel mode, the EMC clock of every
sample, the number of EMC transitions, the EMC utilisation, the CPU and GPU clocks and the trace span.

Read-only over results/ (python3 only).  Run from anywhere:
    python3 campaign/powermode_trace.py [--json]

Traces (Orin Nano, 2026-06-12, EMC unlocked, CPU and GPU pinned at the mode's caps, tegrastats at
100 ms):  MAXN_SUPER results/e1_governor/{idle,mobilenet,proxy,slm}.tegra;
25W results/replication_25w/{governor,gov_idle,gov_mobilenet,gov_proxy,gov_slm}.tegra;
15W results/replication_15w/{idle,mobilenet,proxy,slm}.tegra.
A transition is a change of the EMC clock between consecutive samples.  The span of a trace is its
last minus its first tegrastats timestamp.
"""
import argparse
import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path

R = Path(__file__).resolve().parent.parent / "results"
EMC = re.compile(r"EMC_FREQ (\d+)%@(\d+)")
CPU = re.compile(r"CPU \[([^\]]*)\]")
GR3D = re.compile(r"GR3D_FREQ (\d+)%@\[(\d+)\]")
TS = re.compile(r"^(\d\d-\d\d-\d{4} \d\d:\d\d:\d\d)")
CASES = [
    ("MAXN_SUPER", R / "e1_governor", ["idle", "mobilenet", "proxy", "slm"]),
    ("25W", R / "replication_25w", ["governor", "gov_idle", "gov_mobilenet", "gov_proxy", "gov_slm"]),
    ("15W", R / "replication_15w", ["idle", "mobilenet", "proxy", "slm"]),
]


def span_s(lines):
    ts = [TS.match(ln) for ln in (lines[0], lines[-1])]
    if not all(ts):
        return None
    t0, t1 = (datetime.strptime(m.group(1), "%m-%d-%Y %H:%M:%S") for m in ts)
    return (t1 - t0).total_seconds()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", action="store_true", help="also print the per-trace record as JSON at the end")
    a = ap.parse_args()
    out = {}
    all_n = all_tr = 0; spans = []
    for mode, base, names in CASES:
        tot = Counter(); tot_n = 0; tot_tr = 0; cpu_all = Counter(); gpu_all = Counter(); util_all = []
        online_all = Counter()
        for name in names:
            f = base / f"{name}.tegra"
            lines = f.read_text().splitlines()
            rates, utils, cpus, gpus, onl = [], [], Counter(), Counter(), Counter()
            for ln in lines:
                m = EMC.search(ln)
                if not m: continue
                utils.append(int(m.group(1))); rates.append(int(m.group(2)))
                c = CPU.search(ln)
                if c:
                    cores = c.group(1).split(",")
                    n_on = 0
                    for x in cores:
                        if "@" in x:
                            n_on += 1; cpus[int(x.split("@")[1])] += 1
                    onl[(len(cores), n_on)] += 1
                g = GR3D.search(ln)
                if g: gpus[int(g.group(2))] += 1
            trans = sum(1 for x, y in zip(rates, rates[1:]) if x != y)
            utils_s = sorted(utils)
            u_med = utils_s[len(utils_s)//2] if utils_s else None
            u_mean = sum(utils) / len(utils) if utils else None
            sp = span_s(lines)
            print(f"{mode:<10} {str(f.relative_to(R)):<38} n={len(rates):>5} EMC={dict(Counter(rates))} trans={trans} "
                  f"EMCutil med={u_med}% max={max(utils) if utils else None}% mean={u_mean:.1f}% CPU={dict(cpus.most_common(4))} "
                  f"cores(listed,on)={dict(onl)} GPU={dict(gpus.most_common(4))} t0={lines[0][:19]} t1={lines[-1][:19]} span={sp:.0f} s")
            tot.update(rates); tot_n += len(rates); tot_tr += trans; cpu_all.update(cpus); gpu_all.update(gpus)
            util_all += utils; online_all.update(onl); spans.append(sp)
            out[f"{mode}/{name}"] = dict(n=len(rates), emc=dict(Counter(rates)), transitions=trans,
                                         emc_util_max=max(utils) if utils else None, emc_util_mean=u_mean,
                                         cpu=dict(cpus), gpu=dict(gpus), span_s=sp)
        print(f"  => {mode}: traces {len(names)} n={tot_n} EMC={dict(tot)} transitions={tot_tr} EMCutil max={max(util_all)}% "
              f"CPU={dict(cpu_all.most_common(5))} cores={dict(online_all)} GPU={dict(gpu_all.most_common(5))}\n")
        all_n += tot_n; all_tr += tot_tr
    print(f"all modes: {len(out)} traces, {all_n} samples, {all_tr} EMC transitions, spans {min(spans):.0f}-{max(spans):.0f} s")
    if a.json:
        print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()

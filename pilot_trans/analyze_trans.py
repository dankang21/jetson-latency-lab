#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
RQ3 pilot verdict: how much does a frequency transition cost, per domain?

Joins each transition record (t_write_ns) with the probe trace (end_ns,dur_ns,
same CLOCK_MONOTONIC) and reports, per (domain, pair, direction):

  stall_us     workload-observed: for every chunk in the transient window,
               excess = dur - nearest(pre_level, post_level); stall is the max
               excess. Level-classified so the legitimate slow-level duration
               is not double-counted as stall (a max-vs-max baseline had a
               masking dead zone of Q_slow - Q_fast, up to ~245us).
  noise_us     the SAME statistic computed over the steady window of the same
               transition (no switch in it) — the noise floor. The verdict
               uses net = stall_p95 - noise_p95, because SCHED_FIFO probes on
               an unshielded core still take timer ticks.
  settle_ms    first run of 3 consecutive in-band (15% of post level) samples.
               Meaningless when the pair is level-degenerate (levels within
               20%) — flagged per pair, settle suppressed.
  readback_us  diagnostic only: cached driver / firmware bookkeeping values,
               NOT hardware state (see run_trans_pilot docstring).

Kill criterion (pre-registered): if net stall p95 < 100us on every domain,
RQ3 is demoted from a headline contribution to an appendix.

Data-quality gates: pairs with readback timeouts, dropped transitions
(empty windows), a dead probe at loop end, or a nonzero probe exit are
flagged loudly and excluded from the verdict.
"""

import bisect
import csv
import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "results" / "pilot_trans"
TRANSIENT_MS = 100      # window after t_write searched for the stall
STEADY_MS = (100, 200)  # window after t_write treated as new steady state
PRE_MS = 120            # before t_write; short enough not to touch the previous
                        # transition's transient (transitions are >=250ms apart)
KILL_US = 100.0
DEGENERATE_LEVEL = 0.20
SETTLE_BAND = 0.15
SETTLE_CONSEC = 3


def load_probe(path: Path):
    ends, durs = [], []
    with open(path) as f:
        next(f)
        for line in f:
            e, d = line.split(",")
            ends.append(int(e))
            durs.append(int(d))
    return ends, durs


def window(ends, durs, lo_ns, hi_ns):
    """durs of samples whose end falls in [lo, hi)."""
    i = bisect.bisect_left(ends, lo_ns)
    j = bisect.bisect_left(ends, hi_ns)
    return durs[i:j]


def max_excess(samples, levels):
    """Max of (dur - nearest level) — the level-classified stall statistic."""
    return max(d - min(levels, key=lambda lv: abs(d - lv)) for d in samples)


def analyze_pair(label: str):
    rows = list(csv.DictReader(open(OUT / f"{label}.csv")))
    ends, durs = load_probe(OUT / f"{label}.probe.csv")
    meta = {}
    meta_path = OUT / f"{label}.meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())

    per_dir = defaultdict(lambda: defaultdict(list))
    quality = {
        "timeouts": sum(1 for r in rows if r["readback_timeout"] == "1"),
        "dropped": 0,
        "probe_alive": meta.get("probe_alive_at_loop_end", True),
        "probe_rc": meta.get("probe_returncode", 0),
        "degenerate": False,
    }

    for r in rows:
        t = int(r["t_write_ns"])
        pre = window(ends, durs, t - PRE_MS * 1_000_000, t)
        steady = window(ends, durs, t + STEADY_MS[0] * 1_000_000,
                        t + STEADY_MS[1] * 1_000_000)
        trans = window(ends, durs, t, t + TRANSIENT_MS * 1_000_000)
        if not pre or not steady or not trans:
            quality["dropped"] += 1
            continue
        pre_med, post_med = st.median(pre), st.median(steady)
        levels = (pre_med, post_med)
        if abs(pre_med - post_med) < DEGENERATE_LEVEL * max(levels):
            quality["degenerate"] = True

        stall_us = max_excess(trans, levels) / 1000.0
        noise_us = max_excess(steady, (post_med,)) / 1000.0

        settle_ms = None
        if not quality["degenerate"]:
            i0 = bisect.bisect_left(ends, t)
            i1 = bisect.bisect_left(ends, t + STEADY_MS[1] * 1_000_000)
            consec = 0
            for k in range(i0, i1):
                if abs(durs[k] - post_med) <= SETTLE_BAND * post_med:
                    consec += 1
                    if consec >= SETTLE_CONSEC:
                        settle_ms = (ends[k] - t) / 1e6
                        break
                else:
                    consec = 0

        d = per_dir[r["direction"]]
        d["stall_us"].append(stall_us)
        d["noise_us"].append(noise_us)
        if settle_ms is not None:
            d["settle_ms"].append(settle_ms)
        if r["readback_us"]:
            d["readback_us"].append(float(r["readback_us"]))
        d["write_us"].append(float(r["write_us"]))
        d["quantum_us"].append(max(levels) / 1000.0)
    return per_dir, quality


def p95(xs):
    return sorted(xs)[int(0.95 * (len(xs) - 1))] if xs else float("nan")


def main():
    labels = sorted(p.stem for p in OUT.glob("*_*.csv")
                    if not p.name.endswith(".probe.csv"))
    if not labels:
        sys.exit(f"no results under {OUT} — run pilot_trans/run_trans_pilot.py first")

    print(f"{'pair':<24} {'dir':<5} {'n':>3} {'stall p50/p95 us':>18} "
          f"{'noise p95':>10} {'net p95':>9} {'settle p50 ms':>14} "
          f"{'readback p50':>13} {'quantum':>8}")
    domain_worst = defaultdict(float)
    excluded = []
    for label in labels:
        per_dir, q = analyze_pair(label)
        domain = label.split("_")[0]
        bad = (q["timeouts"] > 0 or q["dropped"] > 0
               or not q["probe_alive"] or q["probe_rc"] not in (0, None))
        if bad:
            excluded.append((label, q))
        for direction in ("down", "up"):
            d = per_dir.get(direction)
            if not d or not d["stall_us"]:
                continue
            s50 = st.median(d["stall_us"])
            s95, n95 = p95(d["stall_us"]), p95(d["noise_us"])
            net = s95 - n95
            if not bad:
                domain_worst[domain] = max(domain_worst[domain], net)
            settle = (st.median(d["settle_ms"])
                      if d["settle_ms"] else float("nan"))
            rb = (st.median(d["readback_us"])
                  if d["readback_us"] else float("nan"))
            flag = " DEGEN" if q["degenerate"] else ""
            print(f"{label:<24} {direction:<5} {len(d['stall_us']):>3} "
                  f"{s50:>8.1f} /{s95:>8.1f} {n95:>10.1f} {net:>9.1f} "
                  f"{settle:>14.2f} {rb:>13.1f} "
                  f"{st.median(d['quantum_us']):>8.1f}{flag}")

    if excluded:
        print("\n!! EXCLUDED FROM VERDICT (data-quality gate):")
        for label, q in excluded:
            print(f"   {label}: timeouts={q['timeouts']} dropped={q['dropped']} "
                  f"probe_alive={q['probe_alive']} probe_rc={q['probe_rc']}")

    print("\n== verdict (kill: net stall p95 < 100us on every domain) ==")
    if not domain_worst:
        sys.exit("no pair passed the data-quality gate — fix the runs first")
    alive = {k: v for k, v in domain_worst.items() if v >= KILL_US}
    for dom, worst in sorted(domain_worst.items(), key=lambda kv: -kv[1]):
        print(f"  {dom}: worst net stall p95 = {worst:.1f} us "
              f"{'(above kill line)' if worst >= KILL_US else '(below)'}")
    if alive:
        print(f"RQ3 ALIVE on {sorted(alive)} — transition cost is "
              "deadline-relevant at the 1-10ms regime.")
    else:
        print("RQ3 DEAD as a headline: all domains under 100us net. "
              "Demote to appendix per the pre-registered criterion.")


if __name__ == "__main__":
    main()

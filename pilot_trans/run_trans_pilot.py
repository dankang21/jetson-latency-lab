#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
RQ3 pilot: per-domain frequency-transition cost on an integrated SoC.

For each clock domain (cpu / gpu / emc), runs the matching probe on a pinned
core (SCHED_FIFO 50 — SCHED_OTHER probes showed 100-800us preemption noise,
above the kill line) while toggling the domain's frequency between two table
rates N times. Records, per transition:

  t_write_ns   CLOCK_MONOTONIC just before the sysfs/debugfs write
  write_us     duration of the write() syscall itself
  readback_us  time until the domain's cur-rate file reports the target

readback_us is DIAGNOSTIC ONLY — none of these files report hardware state:
  cpu  scaling_cur_freq is the cpufreq driver's cached value, updated
       synchronously by the setspeed write path (says nothing about NAFLL
       relock); cpuinfo_cur_freq does counter-measure but cross-calls the
       measured core, so we never poll it here.
  gpu  devfreq cur_freq is clk-framework software state.
  emc  debugfs rate is BPMP table bookkeeping (reads 3199000000 while the
       hardware pto_counter reads ~3191887872); pto_counter is sampled once
       before/after each pair as ground truth, never polled mid-transition
       (each debugfs read is a 0.5-0.9ms BPMP MRQ round trip).
The workload-truth number for the paper is the probe trace, not readback.

The orchestrator pins itself to cores 0-3 (away from the probe on core 5)
and sleeps between readback polls so the poll loop cannot perturb the
transient window it is measuring.

Run as root: sudo python3 pilot_trans/run_trans_pilot.py
Results: results/pilot_trans/<domain>_<rateA>_<rateB>.{csv,probe.csv}
"""

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "pilot_trans"
PROBE_CPU = 5          # probe runs here; cpu domain also toggles this core's policy
ORCH_CPUS = {0, 1, 2, 3}
SETTLE_S = 0.25        # between transitions; > any expected transient
POLL_SLEEP_S = 0.0005
POLL_TIMEOUT_S = 0.5
EMC_POLL_GRACE_S = 0.002   # let the BPMP run the switch before MRQ-polling it
PROBE_PAD_S = 3.0

GPU_DIR = Path("/sys/devices/platform/17000000.gpu/devfreq/17000000.gpu")
EMC_DIR = Path("/sys/kernel/debug/bpmp/debug/clk/emc")

CPU_TABLE_MIN, CPU_TABLE_MAX = 115200, 1728000

# (domain, rate_a, rate_b) — a is the "high" end, units are the domain's own
# (kHz for cpufreq, Hz for devfreq/BPMP). Pairs chosen as max<->mid and
# max<->min of each domain's table.
# (domain, rate_a, rate_b, transitions) — transitions sized per probe rate
# so the sample buffer holds the whole run (cpu probe ~5.4us/chunk).
MATRIX = [
    ("cpu", 1728000, 729600, 300),
    ("cpu", 1728000, 115200, 300),
    ("gpu", 1020000000, 612000000, 400),
    ("gpu", 1020000000, 306000000, 400),
    ("emc", 3199000000, 2133000000, 400),
    ("emc", 3199000000, 665600000, 400),
]


def read_str(p: Path) -> str:
    return p.read_text().strip()


def write_str(p: Path, v) -> float:
    """Write and return the write() duration in us."""
    t0 = time.monotonic_ns()
    with open(p, "w") as f:
        f.write(str(v))
    return (time.monotonic_ns() - t0) / 1000.0


def cpu_policy_dir(cpu: int) -> Path:
    for pol in sorted(Path("/sys/devices/system/cpu/cpufreq").glob("policy*")):
        if str(cpu) in read_str(pol / "affected_cpus").split():
            return pol
    sys.exit(f"no cpufreq policy contains cpu{cpu}")


class Domain:
    """set(rate): initiate the switch, return write_us. cur(): current rate."""

    def __init__(self, name: str):
        self.name = name
        if name == "cpu":
            self.pol = cpu_policy_dir(PROBE_CPU)
            self.saved_gov = read_str(self.pol / "scaling_governor")
            self.saved_min = read_str(self.pol / "scaling_min_freq")
            self.saved_max = read_str(self.pol / "scaling_max_freq")
            # widen the policy first or setspeed gets silently clamped by
            # leftover min/max pinning (e.g. a crashed set_domain.sh 'all')
            write_str(self.pol / "scaling_min_freq", CPU_TABLE_MIN)
            write_str(self.pol / "scaling_max_freq", CPU_TABLE_MAX)
            write_str(self.pol / "scaling_governor", "userspace")
            self.set_path = self.pol / "scaling_setspeed"
            self.cur_path = self.pol / "scaling_cur_freq"
        elif name == "gpu":
            self.saved = (read_str(GPU_DIR / "min_freq"),
                          read_str(GPU_DIR / "max_freq"))
            self.cur_path = GPU_DIR / "cur_freq"
        elif name == "emc":
            write_str(EMC_DIR / "mrq_rate_locked", 1)
            self.cur_path = EMC_DIR / "rate"
        else:
            raise ValueError(name)

    def set(self, rate) -> float:
        if self.name == "cpu":
            return write_str(self.set_path, rate)
        if self.name == "gpu":
            # min <= max must hold at every step: raising writes max first,
            # lowering writes min first. (On this 5.15 kernel devfreq QoS
            # would tolerate min>max via clamping, with the transition firing
            # during the SECOND write — don't rely on it.)
            last = getattr(self, "_last_rate", None)
            if last is None or rate > last:
                us = write_str(GPU_DIR / "max_freq", rate)
                us += write_str(GPU_DIR / "min_freq", rate)
            else:
                us = write_str(GPU_DIR / "min_freq", rate)
                us += write_str(GPU_DIR / "max_freq", rate)
            self._last_rate = rate
            return us
        return write_str(EMC_DIR / "rate", rate)

    def cur(self) -> int:
        return int(read_str(self.cur_path))

    def hw_counter(self):
        """Hardware-truth snapshot where one exists (EMC pto_counter)."""
        if self.name == "emc":
            try:
                return int(read_str(EMC_DIR / "pto_counter"))
            except (OSError, ValueError):
                return None
        return None

    def restore(self):
        if self.name == "cpu":
            write_str(self.pol / "scaling_governor", self.saved_gov)
            write_str(self.pol / "scaling_min_freq", self.saved_min)
            write_str(self.pol / "scaling_max_freq", self.saved_max)
        elif self.name == "gpu":
            write_str(GPU_DIR / "min_freq", self.saved[0])
            write_str(GPU_DIR / "max_freq", self.saved[1])
        elif self.name == "emc":
            write_str(EMC_DIR / "mrq_rate_locked", 0)


def start_probe(domain: str, out_csv: Path, ms: int) -> subprocess.Popen:
    here = Path(__file__).parent
    rt = ["chrt", "-f", "50"]
    if domain == "gpu":
        cmd = rt + ["taskset", "-c", str(PROBE_CPU),
                    str(here / "gpu_probe"), "--ms", str(ms),
                    "--out", str(out_csv)]
    else:
        chunk = "cpu" if domain == "cpu" else "mem"
        cmd = rt + [str(here / "freq_probe"), "--chunk", chunk,
                    "--cpu", str(PROBE_CPU), "--ms", str(ms),
                    "--out", str(out_csv)]
    return subprocess.Popen(cmd, stdout=subprocess.DEVNULL)


def poll_until(dom: Domain, target, t_write_ns: int):
    """Poll cur() with sleeps until it reports target. Returns (us, timed_out)."""
    if dom.name == "emc":
        time.sleep(EMC_POLL_GRACE_S)
    deadline = t_write_ns + int(POLL_TIMEOUT_S * 1e9)
    while time.monotonic_ns() < deadline:
        if dom.cur() == target:
            return (time.monotonic_ns() - t_write_ns) / 1000.0, False
        time.sleep(POLL_SLEEP_S)
    return None, True


def run_pair(domain: str, rate_a, rate_b, transitions):
    label = f"{domain}_{rate_a}_{rate_b}"
    print(f"================ {label} ================")
    dom = Domain(domain)
    rows = []
    probe = None
    pair_meta = {}
    try:
        dom.set(rate_a)
        time.sleep(0.5)
        if dom.cur() != rate_a:
            print(f"  ABORT: {domain} did not reach {rate_a} "
                  f"(cur={dom.cur()}) — pair skipped", file=sys.stderr)
            return
        pair_meta["hw_counter_before"] = dom.hw_counter()

        # realistic budget + slack (readbacks are ms-scale; timeouts rare
        # and flagged by the analyzer's quality gate if they truncate)
        probe_ms = int((transitions * (SETTLE_S + 0.06)
                        + 2 * PROBE_PAD_S + 30) * 1000)
        probe = start_probe(domain, OUT / f"{label}.probe.csv", probe_ms)
        time.sleep(PROBE_PAD_S)

        cur = rate_a
        for i in range(transitions):
            target = rate_b if cur == rate_a else rate_a
            direction = "down" if target < cur else "up"
            t_write = time.monotonic_ns()
            write_us = dom.set(target)
            readback_us, timed_out = poll_until(dom, target, t_write)
            rows.append({
                "i": i, "from": cur, "to": target, "direction": direction,
                "t_write_ns": t_write, "write_us": round(write_us, 2),
                "readback_us": round(readback_us, 2) if readback_us else None,
                "readback_timeout": timed_out,
            })
            cur = target
            time.sleep(SETTLE_S)

        pair_meta["loop_end_ns"] = time.monotonic_ns()
        pair_meta["probe_alive_at_loop_end"] = probe.poll() is None
        pair_meta["hw_counter_after"] = dom.hw_counter()
        time.sleep(PROBE_PAD_S)
        probe.terminate()   # probes exit their loop and write the CSV on SIGTERM
        try:
            probe.wait(timeout=probe_ms / 1000 + 30)
        except subprocess.TimeoutExpired:
            probe.kill()
        pair_meta["probe_returncode"] = probe.returncode
    finally:
        if probe and probe.poll() is None:
            probe.kill()
        dom.restore()

    with open(OUT / f"{label}.csv", "w") as f:
        f.write("i,from,to,direction,t_write_ns,write_us,readback_us,readback_timeout\n")
        for r in rows:
            f.write(f"{r['i']},{r['from']},{r['to']},{r['direction']},"
                    f"{r['t_write_ns']},{r['write_us']},"
                    f"{r['readback_us'] if r['readback_us'] is not None else ''},"
                    f"{int(r['readback_timeout'])}\n")
    (OUT / f"{label}.meta.json").write_text(json.dumps(pair_meta, indent=2))
    n_to = sum(1 for r in rows if r["readback_timeout"])
    print(f"  {len(rows)} transitions, {n_to} readback timeouts, "
          f"probe alive at loop end: {pair_meta.get('probe_alive_at_loop_end')}")


RT_RUNTIME = Path("/proc/sys/kernel/sched_rt_runtime_us")


def main():
    if os.geteuid() != 0:
        sys.exit("run as root (cpufreq/devfreq/debugfs writes)")
    signal.signal(signal.SIGTERM, lambda *a: sys.exit(1))  # run finally blocks
    os.sched_setaffinity(0, ORCH_CPUS)                     # stay off the probe core
    OUT.mkdir(parents=True, exist_ok=True)

    # The probes busy-spin at FIFO 50, so default RT throttling (950ms/1s)
    # inserts a 50ms gap every second — observed as 46-51ms fake stalls that
    # contaminated both transient and noise windows in the first run.
    saved_rt_runtime = read_str(RT_RUNTIME)
    write_str(RT_RUNTIME, -1)

    meta = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "probe_cpu": PROBE_CPU,
        "settle_s": SETTLE_S, "matrix": MATRIX,
        "readback_semantics": {
            "cpu": "scaling_cur_freq = cpufreq driver cached value (not hardware)",
            "gpu": "devfreq cur_freq = clk framework software state (not hardware)",
            "emc": "debugfs rate = BPMP bookkeeping; pto_counter snapshots in "
                   "<pair>.meta.json are the hardware truth",
        },
        "note": "probes run SCHED_FIFO 50 on the probe core; orchestrator pinned "
                "to cores 0-3. CPU+GPU otherwise dynamic; EMC locked only during "
                "emc pairs. Run with the board otherwise idle.",
    }
    (OUT / "meta.json").write_text(json.dumps(meta, indent=2))

    try:
        for domain, a, b, n in MATRIX:
            run_pair(domain, a, b, n)
    finally:
        write_str(RT_RUNTIME, saved_rt_runtime)
    print("done. analyze: python3 pilot_trans/analyze_trans.py")


if __name__ == "__main__":
    main()

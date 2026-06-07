#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Part 2 analysis: join latency (harness JSON) with the clock/thermal trace
(tegrastats) for each domain profile.

The point: show not just THAT latency changed per domain, but WHAT CLOCK the GPU
actually held during each run — turning "the clock dropped" from inference into
evidence.

  python3 part2/analyze_part2.py
"""
import glob
import json
import re
import statistics as st

PROFILES = ["free", "gpu_only", "cpu_only", "all"]

GR3D = re.compile(r"GR3D_FREQ\s+\d+%@\[?(\d+)\]?")
EMC = re.compile(r"EMC_FREQ\s+\d+%@\[?(\d+)\]?")
CPU = re.compile(r"CPU \[([^\]]+)\]")
GPUT = re.compile(r"gpu@([\d.]+)C")
PWR = re.compile(r"VDD_CPU_GPU_CV (\d+)mW")


def parse_tegra(path):
    gpu, cpu, emc, temp, pwr = [], [], [], [], []
    try:
        lines = open(path).read().splitlines()
    except FileNotFoundError:
        return None
    for ln in lines:
        m = GR3D.search(ln)
        if m:
            gpu.append(int(m.group(1)))
        m = EMC.search(ln)
        if m:
            emc.append(int(m.group(1)))
        m = CPU.search(ln)
        if m:
            freqs = [int(x.split("@")[1]) for x in m.group(1).split(",")]
            cpu.append(max(freqs))
        m = GPUT.search(ln)
        if m:
            temp.append(float(m.group(1)))
        m = PWR.search(ln)
        if m:
            pwr.append(int(m.group(1)))
    if not gpu:
        return None
    gpu_max_seen = max(gpu)
    return {
        "n": len(gpu),
        "gpu_med": st.median(gpu),
        "gpu_max": gpu_max_seen,
        "gpu_pct_at_top": 100.0 * sum(1 for g in gpu if g >= 1016) / len(gpu),
        "cpu_med": st.median(cpu) if cpu else None,
        "emc_med": st.median(emc) if emc else None,
        "temp_max": max(temp) if temp else None,
        "pwr_med": st.median(pwr) if pwr else None,
    }


def load_json(prof):
    p = f"results/p2_{prof}.json"
    try:
        return json.load(open(p))
    except FileNotFoundError:
        return None


def main():
    lat_rows, clk_rows = [], []
    for prof in PROFILES:
        j = load_json(prof)
        t = parse_tegra(f"results/p2_{prof}.tegra")
        if j:
            r, c, jt = j["response_us"], j["compute_us"], j["release_jitter_us"]
            lat_rows.append((prof, r["p50"], r["p99.99"], c["p50"],
                             jt["p99.99"], j["deadline_miss_count"]))
        if t:
            clk_rows.append((prof, t["gpu_med"], t["gpu_pct_at_top"],
                             t["cpu_med"], t["emc_med"], t["temp_max"],
                             t["pwr_med"]))

    print("\n=== Latency by clock domain (100 Hz, 100k cycles) ===")
    print("| profile | resp p50 (ms) | resp p99.99 (ms) | compute p50 (ms) | "
          "jitter p99.99 (us) | misses |")
    print("|---|---|---|---|---|---|")
    for prof, p50, p9999, cp50, jit, miss in lat_rows:
        print(f"| {prof} | {p50/1000:.3f} | {p9999/1000:.3f} | "
              f"{cp50/1000:.3f} | {jit:.1f} | {miss} |")

    print("\n=== Clock / thermal during each run (tegrastats trace) ===")
    print("| profile | GPU med (MHz) | % at top clk | CPU med (MHz) | "
          "EMC med | gpu max temp | VDD_CPU_GPU_CV med (mW) |")
    print("|---|---|---|---|---|---|---|")
    for prof, gmed, gtop, cmed, emed, tmax, pmed in clk_rows:
        print(f"| {prof} | {gmed:.0f} | {gtop:.0f}% | {cmed or '-'} | "
              f"{emed or '-'} | {tmax or '-'}C | {pmed or '-'} |")

    print("\nRead it together: the domain whose fix drops compute back to ~3.9 ms")
    print("is the culprit; the trace shows that domain's clock pinned at top while")
    print("the others stayed dynamic.")


if __name__ == "__main__":
    main()

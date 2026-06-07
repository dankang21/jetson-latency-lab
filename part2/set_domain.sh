#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Per-domain clock control for the DVFS decomposition (Part 2).
#
# jetson_clocks pins CPU+GPU+EMC together, so it cannot tell you WHICH domain
# the periodic-inference penalty comes from. This script fixes one domain at a
# time (min=max) while leaving the others dynamic.
#
# Paths are specific to this board (Orin Nano Super, confirmed via sysfs):
#   GPU: /sys/devices/platform/17000000.gpu/devfreq/17000000.gpu/{min,max}_freq
#   CPU: /sys/devices/system/cpu/cpu*/cpufreq/scaling_{min,max}_freq + governor
# EMC is left dynamic in every profile (we only OBSERVE it via tegrastats);
# forcing EMC via bpmp debug is risky and unnecessary for the decomposition.
#
# Usage: sudo ./set_domain.sh {free|gpu_only|cpu_only|all}
set -euo pipefail

GPU=/sys/devices/platform/17000000.gpu/devfreq/17000000.gpu
GPU_MIN=306000000
GPU_MAX=1020000000
CPU_MIN=115200
CPU_MAX=1728000

gpu_fix()  { echo $GPU_MAX | tee $GPU/max_freq >/dev/null; echo $GPU_MAX | tee $GPU/min_freq >/dev/null; }
gpu_free() { echo $GPU_MIN | tee $GPU/min_freq >/dev/null; echo $GPU_MAX | tee $GPU/max_freq >/dev/null; }

cpu_fix() {
  for c in /sys/devices/system/cpu/cpu*/cpufreq; do
    echo performance | tee $c/scaling_governor >/dev/null
    echo $CPU_MAX    | tee $c/scaling_min_freq  >/dev/null
  done
}
cpu_free() {
  for c in /sys/devices/system/cpu/cpu*/cpufreq; do
    echo schedutil | tee $c/scaling_governor >/dev/null
    echo $CPU_MIN  | tee $c/scaling_min_freq  >/dev/null
  done
}

case "${1:-}" in
  free)     gpu_free; cpu_free ;;
  gpu_only) gpu_fix;  cpu_free ;;
  cpu_only) gpu_free; cpu_fix  ;;
  all)      gpu_fix;  cpu_fix  ;;   # equivalent to jetson_clocks for CPU+GPU
  *) echo "usage: $0 {free|gpu_only|cpu_only|all}"; exit 1 ;;
esac

sleep 2
echo "[set_domain: $1] GPU min/cur = $(cat $GPU/min_freq)/$(cat $GPU/cur_freq), \
CPU0 gov/min = $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor)/\
$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_min_freq)"

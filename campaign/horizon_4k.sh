#!/bin/bash
# HORIZON-4K: where the horizon layer bites (NX).
# At H=2000 the measured KV drift (+2.3-2.6%) sits inside the static 3.5%
# margin, so horizon-blind acceptance survives by accident. At H=4000,
# gamma*H (~5-7%) exceeds the margin: horizon-blind accepts (D=28ms) and the
# second half of the response violates; the joint policy's quoted horizon
# cap H* separates the two. Qwen2.5-1.5B Q4, 4000-token decode, EMC 2133,
# GPU {1122, 1173} x 3 reps (~110 s per run).
# Env: GGUF_DIR (GGUF models) overrides the default path below.
set -u
LAB="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BENCH=$LAB/tools/slm_decode_bench
MD="${GGUF_DIR:-/home/dk/mobile/models}"
G=/sys/devices/platform/17000000.gpu/devfreq/17000000.gpu
cd "$LAB" || exit 1; mkdir -p results/horizon_4k
restore(){ sudo bash -c "$LAB/pilot_emc/emc_ctl.sh unlock >/dev/null 2>&1
  echo 306000000 > $G/min_freq; echo 1173000000 > $G/max_freq
  echo 950000 > /proc/sys/kernel/sched_rt_runtime_us
  for c in /sys/devices/system/cpu/cpu*/cpufreq; do echo schedutil > \$c/scaling_governor; done
  chown -R dk:dk results/horizon_4k 2>/dev/null"; }
trap restore EXIT
sudo bash -c "echo -1 > /proc/sys/kernel/sched_rt_runtime_us
for c in /sys/devices/system/cpu/cpu*/cpufreq; do echo performance > \$c/scaling_governor; done"
sudo bash -c "$LAB/pilot_emc/emc_ctl.sh lock 2133000000 >/dev/null"

for f in 1122 1173; do
  sudo bash -c "echo ${f}000000 > $G/min_freq; echo ${f}000000 > $G/max_freq"; sleep 0.5
  for r in 1 2 3; do
    "$BENCH" --model "$MD/qwen2.5-1.5b-instruct-q4_k_m.gguf" \
      --tokens 4000 --warmup 40 --cpu 7 --prio 80 \
      --out "results/horizon_4k/q15h4k_g${f}_r${r}" >/dev/null 2>&1
    echo "  g${f} r${r}: $([ -s results/horizon_4k/q15h4k_g${f}_r${r}.json ] && echo OK || echo FAIL)"
  done
done
echo "HORIZON_4K DONE"

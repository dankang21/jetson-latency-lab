#!/bin/bash
# C1-live + C2-live launcher (NX). Whole experiment as root (clock writes are
# in the control loop); e4_replication precedent shows root+ORT is fine here.
# Env: PY (python3 with onnxruntime) and ONNX_DIR (ONNX models) override the default paths below.
set -u
PY="${PY:-/home/dk/mobile/rt-infer-bench/.venv/bin/python3}"
ONNX_DIR="${ONNX_DIR:-/home/dk/mobile/rt-infer-bench/models}"
LAB="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MN=$ONNX_DIR/mobilenetv2-12.onnx
G=/sys/devices/platform/17000000.gpu/devfreq/17000000.gpu
cd "$LAB" || exit 1; mkdir -p results/live_gov
restore(){ sudo bash -c "$LAB/pilot_emc/emc_ctl.sh unlock >/dev/null 2>&1
  echo 306000000 > $G/min_freq; echo 1173000000 > $G/max_freq
  echo 950000 > /proc/sys/kernel/sched_rt_runtime_us
  for c in /sys/devices/system/cpu/cpu*/cpufreq; do echo schedutil > \$c/scaling_governor; done
  chown -R dk:dk results/live_gov 2>/dev/null"; }
trap restore EXIT
sudo bash -c "echo -1 > /proc/sys/kernel/sched_rt_runtime_us
for c in /sys/devices/system/cpu/cpu*/cpufreq; do echo performance > \$c/scaling_governor; done
$LAB/pilot_emc/emc_ctl.sh lock 2133000000 >/dev/null"   # sets mrq_rate_locked + bwmgr_halt

echo "=== C1-live: online EWMA calibrator (alpha=0.1), 3 reps x 3 deadlines ==="
for D in 4.6 4.8 5.0; do
  for r in 1 2 3; do
    sudo bash -c "echo 2133000000 > /sys/kernel/debug/bpmp/debug/clk/emc/rate"; sleep 0.3
    sudo taskset -c 7 "$PY" campaign/nx_live_gov.py --mode calib --model "$MN" \
      --hz 33 --cycles 2000 --deadline-ms "$D" --alpha 0.1 \
      --out results/live_gov/calib_D${D}_r${r}.csv
  done
done

echo "=== C2-live: first-miss EMC upshift 2133->3199 (GPU 918, D=5.0) ==="
for arm in trigger static2133 static3199; do
  for r in 1 2 3; do
    sudo bash -c "echo 2133000000 > /sys/kernel/debug/bpmp/debug/clk/emc/rate"; sleep 0.3
    sudo taskset -c 7 "$PY" campaign/nx_live_gov.py --mode emcswitch --model "$MN" \
      --hz 33 --cycles 1000 --deadline-ms 5.0 --gpu 918 --arm "$arm" \
      --out results/live_gov/emcsw_${arm}_r${r}.csv
  done
done
echo "NX_LIVE DONE"

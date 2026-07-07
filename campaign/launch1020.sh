#!/bin/bash
# Launch-realization characterization at 1020 MHz (one bin above the
# banded pick): 8 fresh co-runner launches x 1,000-cycle contracts, D=12.
set -u
PY=/home/dk/mobile/rt-infer-bench/.venv/bin/python3
LAB=/home/dk/mobile/jetson-latency-lab
MN=/home/dk/mobile/rt-infer-bench/models/mobilenetv2-12.onnx
VIT=/home/dk/mobile/rt-infer-bench/models/vit_small_patch16_224.onnx
G=/sys/devices/platform/17000000.gpu/devfreq/17000000.gpu
cd "$LAB" || exit 1; mkdir -p results/launch1020
CO_PID=""
restore(){ [ -n "$CO_PID" ] && kill "$CO_PID" 2>/dev/null
  sudo bash -c "$LAB/pilot_emc/emc_ctl.sh unlock >/dev/null 2>&1
  echo 306000000 > $G/min_freq; echo 1173000000 > $G/max_freq
  echo 950000 > /proc/sys/kernel/sched_rt_runtime_us
  for c in /sys/devices/system/cpu/cpu*/cpufreq; do echo schedutil > \$c/scaling_governor; done
  chown -R dk:dk results/launch1020 2>/dev/null"; }
trap restore EXIT
sudo bash -c "echo -1 > /proc/sys/kernel/sched_rt_runtime_us
for c in /sys/devices/system/cpu/cpu*/cpufreq; do echo performance > \$c/scaling_governor; done
echo 1020000000 > $G/min_freq; echo 1020000000 > $G/max_freq"
sudo bash -c "$LAB/pilot_emc/emc_ctl.sh lock 2133000000 >/dev/null"
for t in 1 2 3 4 5 6 7 8; do
  "$PY" -m harness.infer_bench --backend onnxruntime --model "$VIT" \
    --hz 10 --iters 100000 --warmup 30 --cpu 5 --label co_vit \
    --out /tmp/co_l1020 >/dev/null 2>&1 & CO_PID=$!
  sleep 20
  "$PY" -m harness.infer_bench --backend onnxruntime --model "$MN" \
    --hz 33 --iters 1000 --warmup 60 --cpu 7 --label l1020_t${t} \
    --out results/launch1020/dep_t${t} >/dev/null 2>&1
  kill "$CO_PID" 2>/dev/null; wait "$CO_PID" 2>/dev/null; CO_PID=""; sleep 2
  echo "  t${t}: $([ -s results/launch1020/dep_t${t}.json ] && echo OK || echo FAIL)"
done
echo "LAUNCH1020 DONE"

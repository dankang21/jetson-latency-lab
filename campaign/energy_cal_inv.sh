#!/bin/bash
# Calibrated inversion + EMC energy on the Nano (guava) via 1 ms-sampled onboard INA.
# Saturated (continuously busy) so VDD_IN reads pure active power;
# energy/inf = mean active power x compute latency. GPU pinned at the Nano max.
set -u
PY=/home/dk/mobile/rt-infer-bench/.venv/bin/python3
LAB=/home/dk/mobile/jetson-latency-lab
CPROXY=$LAB/pilot_emc/models/compute_proxy_fp16.onnx
MOBILE=/home/dk/mobile/rt-infer-bench/models/mobilenetv2-12.onnx
G=/sys/devices/platform/17000000.gpu/devfreq/17000000.gpu
cd "$LAB" || exit 1; mkdir -p results/energy_cal
restore(){ sudo bash -c "$LAB/pilot_emc/emc_ctl.sh unlock >/dev/null 2>&1
  echo 306000000 > $G/min_freq; echo 1020000000 > $G/max_freq
  echo 950000 > /proc/sys/kernel/sched_rt_runtime_us
  for c in /sys/devices/system/cpu/cpu*/cpufreq; do echo schedutil > \$c/scaling_governor; done
  chown -R dk:dk results/energy_cal 2>/dev/null"; }
trap restore EXIT
sudo bash -c "echo -1 > /proc/sys/kernel/sched_rt_runtime_us
for c in /sys/devices/system/cpu/cpu*/cpufreq; do echo performance > \$c/scaling_governor; done
echo 1020000000 > $G/min_freq; echo 1020000000 > $G/max_freq"
cell(){ # wl model emcHz emclabel rep
  local wl=$1 model=$2 emc=$3 lab=$4 rep=$5
  local tag=inv_${wl}_emc${lab}_r${rep}
  sudo bash -c "$LAB/pilot_emc/emc_ctl.sh lock ${emc} >/dev/null"
  sleep 0.5
  taskset -c 0 "$PY" campaign/ina_log.py results/energy_cal/${tag}.power.csv 35 &
  local LP=$!
  "$PY" -m harness.infer_bench --backend onnxruntime --model "$model" \
    --hz 120 --iters 600 --warmup 60 --cpu 7 --label "$tag" \
    --out results/energy_cal/${tag} >/dev/null 2>&1
  kill -TERM $LP 2>/dev/null; wait $LP 2>/dev/null
  echo "  ${tag}: $([ -s results/energy_cal/${tag}.json ] && echo OK || echo FAIL)"
}
for rep in 1 2 3; do
  cell cproxyv2 "$CPROXY" 2133000000 2133 "$rep"
  cell cproxyv2 "$CPROXY" 3199000000 3199 "$rep"
done
for rep in 1 2; do
  cell mobilenet "$MOBILE" 2133000000 2133 "$rep"
  cell mobilenet "$MOBILE" 3199000000 3199 "$rep"
done
echo "ENERGY_CAL_INV DONE"

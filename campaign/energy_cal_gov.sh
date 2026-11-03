#!/bin/bash
# Calibrated deployed-governor energy on the NX (hebesu) via 1 ms-sampled onboard INA.
# Periodic deployment duty; energy/inf = mean VDD_IN power / release rate.
# infer_bench + ina_log as USER; clocks/EMC via sudo. Self-contained.
# Env: PY (python3 with onnxruntime) and ONNX_DIR (ONNX models) override the default paths below.
set -u
PY="${PY:-/home/dk/mobile/rt-infer-bench/.venv/bin/python3}"
ONNX_DIR="${ONNX_DIR:-/home/dk/mobile/rt-infer-bench/models}"
LAB="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MOBILE=$ONNX_DIR/mobilenetv2-12.onnx
VIT=$ONNX_DIR/vit_small_patch16_224.onnx
G=/sys/devices/platform/17000000.gpu/devfreq/17000000.gpu
cd "$LAB" || exit 1; mkdir -p results/energy_cal
restore(){ sudo bash -c "$LAB/pilot_emc/emc_ctl.sh unlock >/dev/null 2>&1
  echo 306000000 > $G/min_freq; echo 1173000000 > $G/max_freq
  echo 950000 > /proc/sys/kernel/sched_rt_runtime_us
  for c in /sys/devices/system/cpu/cpu*/cpufreq; do echo schedutil > \$c/scaling_governor; done
  chown -R dk:dk results/energy_cal 2>/dev/null"; }
trap restore EXIT
sudo bash -c "echo -1 > /proc/sys/kernel/sched_rt_runtime_us
for c in /sys/devices/system/cpu/cpu*/cpufreq; do echo performance > \$c/scaling_governor; done
$LAB/pilot_emc/emc_ctl.sh lock 2133000000 >/dev/null"
cell(){ # wl model freq hz iters rep
  local wl=$1 model=$2 f=$3 hz=$4 iters=$5 rep=$6
  local tag=gov_${wl}_g${f}_r${rep}
  sudo bash -c "echo ${f}000000 > $G/min_freq; echo ${f}000000 > $G/max_freq"
  sleep 0.4
  taskset -c 0 "$PY" campaign/ina_log.py results/energy_cal/${tag}.power.csv 35 &
  local LP=$!
  "$PY" -m harness.infer_bench --backend onnxruntime --model "$model" \
    --hz "$hz" --iters "$iters" --warmup 80 --cpu 7 --label "$tag" \
    --out results/energy_cal/${tag} >/dev/null 2>&1
  kill -TERM $LP 2>/dev/null; wait $LP 2>/dev/null
  echo "  ${tag}: $([ -s results/energy_cal/${tag}.json ] && echo OK || echo FAIL)"
}
for rep in 1 2; do
  for f in 816 918 1020 1122 1173; do cell mobilenet "$MOBILE" "$f" 33 600 "$rep"; done
  for f in 816 918 1020 1122 1173; do cell vit "$VIT" "$f" 20 350 "$rep"; done
done
echo "ENERGY_CAL_GOV DONE"

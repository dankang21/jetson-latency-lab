#!/bin/bash
# Rate-matched estimator sweeps for the Han (3-EMC) fit and two-cell refits:
# EMC {665.6, 2133, 3199} x GPU 306..1173 at each model's DEPLOYMENT rate,
# 800 iters, user-mode loop -> results/nx_validation_r33/nx_est_emc<E>_gpu<f>_<model>.json
# (new dir; the submitted 8/5/10 Hz sweeps in results/nx_validation are untouched).
# Env: PY (python3 with onnxruntime) and ONNX_DIR (ONNX models) override the default paths below.
set -u
PY="${PY:-/home/dk/mobile/rt-infer-bench/.venv/bin/python3}"
LAB="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; M="${ONNX_DIR:-/home/dk/mobile/rt-infer-bench/models}"
G=/sys/devices/platform/17000000.gpu/devfreq/17000000.gpu
GMAX=$(( $(tr ' ' '\n' < $G/available_frequencies | sort -n | tail -1) / 1000000 ))
FREQS=${FREQS:-"306 408 510 612 714 816 918 1020 1122 1173"}
OUT=results/nx_validation_r33; cd "$LAB" || exit 1; mkdir -p $OUT
restore(){ sudo -n bash -c "$LAB/pilot_emc/emc_ctl.sh unlock >/dev/null 2>&1
  echo 306000000 > $G/min_freq; echo ${GMAX}000000 > $G/max_freq
  echo 950000 > /proc/sys/kernel/sched_rt_runtime_us
  for c in /sys/devices/system/cpu/cpu*/cpufreq; do echo schedutil > \$c/scaling_governor; done"; }
trap restore EXIT
tmax(){ cat /sys/devices/virtual/thermal/thermal_zone*/temp 2>/dev/null | sort -rn | head -1; }
tgate(){ for i in $(seq 1 60); do t=$(tmax); [ "${t:-0}" -le 55000 ] && return 0; sleep 20; done; }
sudo -n bash -c "echo -1 > /proc/sys/kernel/sched_rt_runtime_us
for c in /sys/devices/system/cpu/cpu*/cpufreq; do echo performance > \$c/scaling_governor; done"
for spec in "vit $M/vit_small_patch16_224.onnx 20 60" "yolov8s $M/yolov8s.onnx 10 40" "resnet50 $M/resnet50-v1-12.onnx 20 60" "mobilenet $M/mobilenetv2-12.onnx 33 120"; do
  set -- $spec; NAME=$1; MP=$2; HZ=$3; WARM=$4
  for E in 665 2133 3199; do
    RATE=$([ $E = 665 ] && echo 665600000 || echo ${E}000000)
    sudo -n bash -c "$LAB/pilot_emc/emc_ctl.sh lock $RATE" | tee -a $OUT/emc_lock.txt | grep -q "actual=$RATE" || { echo "EMC lock $E failed"; exit 1; }
    for f in $FREQS; do
      TAG=$OUT/nx_est_emc${E}_gpu${f}_${NAME}; [ -s $TAG.json ] && continue
      tgate; sudo -n bash -c "echo ${f}000000 > $G/min_freq; echo ${f}000000 > $G/max_freq"; sleep 0.4
      "$PY" -m harness.infer_bench --backend onnxruntime --model "$MP" --hz $HZ --iters 800 --warmup $WARM --cpu 7 \
        --label r33_${NAME}_e${E}_g${f} --out $TAG >/dev/null 2>&1
      echo "  $NAME emc$E gpu$f: $([ -s $TAG.json ] && echo OK || echo FAIL) t=$(tmax)" | tee -a $OUT/session.log
    done
  done
done
echo "SWEEP_RATE DONE $(date -Is)" | tee -a $OUT/session.log

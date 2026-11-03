#!/bin/bash
# Workload-breadth deployed-governor generalization on the NX (hebesu):
# ResNet-50 (CNN, heavier than MobileNetV2) + YOLOv8s (detection, 640x640).
# Mirrors vit_deploy.sh exactly:
#   PHASE 1: estimator sweep @ EMC {3199,2133} x GPU freqs (1 run/cell) -> policy fit
#   PHASE 2: 3-repeat deployment @ EMC 2133 across the GPU-responsive range,
#            with tegrastats (clock ground truth) + 1 ms-sampled INA (energy).
# infer_bench runs as the invoking USER; clocks/tegrastats via sudo. Self-contained.
# Env: PY (python3 with onnxruntime) and ONNX_DIR (ONNX models) override the default paths below.
set -u
PY="${PY:-/home/dk/mobile/rt-infer-bench/.venv/bin/python3}"
ONNX_DIR="${ONNX_DIR:-/home/dk/mobile/rt-infer-bench/models}"
G=/sys/devices/platform/17000000.gpu/devfreq/17000000.gpu
LAB="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$LAB" || exit 1
FREQS="306 408 510 612 714 816 918 1020 1122 1173"
DEPLOY_FREQS="612 714 816 918 1020 1122 1173"

restore(){ sudo bash -c "$LAB/pilot_emc/emc_ctl.sh unlock >/dev/null 2>&1
  echo 306000000 > $G/min_freq; echo 1173000000 > $G/max_freq
  echo 950000 > /proc/sys/kernel/sched_rt_runtime_us
  for c in /sys/devices/system/cpu/cpu*/cpufreq; do echo schedutil > \$c/scaling_governor; done
  chown -R dk:dk results/nx_validation results/govrep_resnet50 results/govrep_yolov8s 2>/dev/null"; }
trap restore EXIT
setclk(){ sudo bash -c "echo ${1}000000 > $G/min_freq; echo ${1}000000 > $G/max_freq"; }

sudo bash -c "echo -1 > /proc/sys/kernel/sched_rt_runtime_us
for c in /sys/devices/system/cpu/cpu*/cpufreq; do echo performance > \$c/scaling_governor; done"

run_model(){  # run_model <name> <model_path> <hz> <sweep_warm> <dep_warm> <ina_dur>
  local NAME=$1 MODEL=$2 HZ=$3 SW=$4 DW=$5 INA=$6
  mkdir -p results/nx_validation "results/govrep_${NAME}"
  echo "=== [$NAME] PHASE 1: estimator sweep (hz $HZ) ==="
  for EMC in 3199 2133; do
    sudo bash -c "$LAB/pilot_emc/emc_ctl.sh lock ${EMC}000000 >/dev/null"
    for f in $FREQS; do
      setclk "$f"; sleep 0.4
      "$PY" -m harness.infer_bench --backend onnxruntime --model "$MODEL" \
        --hz "$HZ" --iters 800 --warmup "$SW" --cpu 7 --label ${NAME}_e${EMC}_g${f} \
        --out results/nx_validation/nx_est_emc${EMC}_gpu${f}_${NAME} >/dev/null 2>&1
      echo "  sweep emc${EMC} gpu${f}: $([ -s results/nx_validation/nx_est_emc${EMC}_gpu${f}_${NAME}.json ] && echo OK || echo FAIL)"
    done
  done
  echo "=== [$NAME] PHASE 2: 3-repeat deployment @ EMC 2133 + INA ==="
  sudo bash -c "$LAB/pilot_emc/emc_ctl.sh lock 2133000000 >/dev/null"
  for f in $DEPLOY_FREQS; do
    for r in 1 2 3; do
      setclk "$f"; sleep 0.4
      local TAG="results/govrep_${NAME}/g${f}_r${r}"
      sudo timeout "$INA" tegrastats --interval 100 > "${TAG}.tegra" 2>/dev/null &
      taskset -c 0 "$PY" campaign/ina_log.py "${TAG}.power.csv" "$INA" &
      local LP=$!
      "$PY" -m harness.infer_bench --backend onnxruntime --model "$MODEL" \
        --hz "$HZ" --iters 1000 --warmup "$DW" --cpu 7 --label ${NAME}_g${f}_r${r} \
        --out "$TAG" >/dev/null 2>&1
      kill -TERM $LP 2>/dev/null; wait $LP 2>/dev/null
      sudo pkill -x tegrastats 2>/dev/null; sleep 0.3
      echo "  deploy g${f}_r${r}: $([ -s ${TAG}.json ] && echo OK || echo FAIL)"
    done
  done
}

# ResNet-50 at hz10 (not 20): its 306MHz@2133 sweep cell can reach ~50ms, and a
# 50ms period would backlog the queue (period must exceed worst-cell compute).
#           name      model                                                  hz  sw  dw  ina
run_model resnet50 $ONNX_DIR/resnet50-v1-12.onnx 10  50  60 130
run_model yolov8s  $ONNX_DIR/yolov8s.onnx         5  30  40 260
echo "BREADTH_DEPLOY DONE"

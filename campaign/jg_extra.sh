#!/bin/bash
# joint_gov supplementary cells: 714/816/918 under co x3 (clock-scaled margin picks)
# Env: PY (python3 with onnxruntime) and ONNX_DIR (ONNX models) override the default paths below.
set -u
PY="${PY:-/home/dk/mobile/rt-infer-bench/.venv/bin/python3}"
ONNX_DIR="${ONNX_DIR:-/home/dk/mobile/rt-infer-bench/models}"
LAB="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MN=$ONNX_DIR/mobilenetv2-12.onnx
VIT=$ONNX_DIR/vit_small_patch16_224.onnx
G=/sys/devices/platform/17000000.gpu/devfreq/17000000.gpu
cd "$LAB" || exit 1
CO_PID=""
restore(){ [ -n "$CO_PID" ] && kill "$CO_PID" 2>/dev/null
  sudo bash -c "$LAB/pilot_emc/emc_ctl.sh unlock >/dev/null 2>&1
  echo 306000000 > $G/min_freq; echo 1173000000 > $G/max_freq
  echo 950000 > /proc/sys/kernel/sched_rt_runtime_us
  for c in /sys/devices/system/cpu/cpu*/cpufreq; do echo schedutil > \$c/scaling_governor; done
  chown -R dk:dk results/joint_gov 2>/dev/null"; }
trap restore EXIT
sudo bash -c "echo -1 > /proc/sys/kernel/sched_rt_runtime_us
for c in /sys/devices/system/cpu/cpu*/cpufreq; do echo performance > \$c/scaling_governor; done"
sudo bash -c "$LAB/pilot_emc/emc_ctl.sh lock 2133000000 >/dev/null"
"$PY" -m harness.infer_bench --backend onnxruntime --model "$VIT" \
  --hz 10 --iters 100000 --warmup 30 --cpu 5 --label co_vit \
  --out /tmp/co_vit_jg2 >/dev/null 2>&1 & CO_PID=$!
sleep 20
for f in 714 816 918; do
  for r in 1 2 3; do
    sudo bash -c "echo ${f}000000 > $G/min_freq; echo ${f}000000 > $G/max_freq"; sleep 0.4
    "$PY" -m harness.infer_bench --backend onnxruntime --model "$MN" \
      --hz 33 --iters 1000 --warmup 60 --cpu 7 --label jg_dep_g${f}_r${r} \
      --out results/joint_gov/dep_co_g${f}_r${r} >/dev/null 2>&1
  done
  echo "  dep gpu${f} x3: $([ -s results/joint_gov/dep_co_g${f}_r3.json ] && echo OK || echo FAIL)"
done
echo "JG_EXTRA DONE"

#!/bin/bash
# Held-out validation of the dispersion-banded guard: 10 fresh end-to-end
# trials (fresh co-runner -> fresh probe -> banded decision -> contract).
# Env: PY (python3 with onnxruntime) and ONNX_DIR (ONNX models) override the default paths below.
set -u
PY="${PY:-/home/dk/mobile/rt-infer-bench/.venv/bin/python3}"
ONNX_DIR="${ONNX_DIR:-/home/dk/mobile/rt-infer-bench/models}"
LAB="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MN=$ONNX_DIR/mobilenetv2-12.onnx
VIT=$ONNX_DIR/vit_small_patch16_224.onnx
G=/sys/devices/platform/17000000.gpu/devfreq/17000000.gpu
cd "$LAB" || exit 1; mkdir -p results/banded_loop
CO_PID=""
restore(){ [ -n "$CO_PID" ] && kill "$CO_PID" 2>/dev/null
  sudo bash -c "$LAB/pilot_emc/emc_ctl.sh unlock >/dev/null 2>&1
  echo 306000000 > $G/min_freq; echo 1173000000 > $G/max_freq
  echo 950000 > /proc/sys/kernel/sched_rt_runtime_us
  for c in /sys/devices/system/cpu/cpu*/cpufreq; do echo schedutil > \$c/scaling_governor; done
  chown -R dk:dk results/banded_loop 2>/dev/null"; }
trap restore EXIT
setclk(){ sudo bash -c "echo ${1}000000 > $G/min_freq; echo ${1}000000 > $G/max_freq"; }
sudo bash -c "echo -1 > /proc/sys/kernel/sched_rt_runtime_us
for c in /sys/devices/system/cpu/cpu*/cpufreq; do echo performance > \$c/scaling_governor; done"
sudo bash -c "$LAB/pilot_emc/emc_ctl.sh lock 2133000000 >/dev/null"
for t in 1 2 3 4 5 6 7 8 9 10; do
  "$PY" -m harness.infer_bench --backend onnxruntime --model "$VIT" \
    --hz 10 --iters 100000 --warmup 30 --cpu 5 --label co_vit \
    --out /tmp/co_bl >/dev/null 2>&1 & CO_PID=$!
  sleep 20
  setclk 1173; sleep 0.4
  "$PY" -m harness.infer_bench --backend onnxruntime --model "$MN" \
    --hz 33 --iters 200 --warmup 60 --cpu 7 --label bl_probe_t${t} \
    --out results/banded_loop/probe_t${t} >/dev/null 2>&1
  read MG PICK <<< "$("$PY" campaign/banded_pick.py results/banded_loop/probe_t${t}.csv)"
  setclk "$PICK"; sleep 0.4
  "$PY" -m harness.infer_bench --backend onnxruntime --model "$MN" \
    --hz 33 --iters 1000 --warmup 60 --cpu 7 --label bl_dep_t${t} \
    --out results/banded_loop/dep_t${t}_g${PICK} >/dev/null 2>&1
  kill "$CO_PID" 2>/dev/null; wait "$CO_PID" 2>/dev/null; CO_PID=""; sleep 2
  echo "  t${t}: Mg=${MG}us pick=${PICK} $([ -s results/banded_loop/dep_t${t}_g${PICK}.json ] && echo OK || echo FAIL)"
done
echo "BANDED_LOOP DONE"

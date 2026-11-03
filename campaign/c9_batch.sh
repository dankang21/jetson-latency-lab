#!/bin/bash
# C9: batch-size axis — does the per-image EMC penalty survive batching?
# MobileNetV2 (symbolic batch dim, BENCH_BATCH knob) x batch {1,4,8} x
# EMC {204, 2133, 3199}, GPU pinned max. Per-cell hz auto-probed (20-iter
# probe -> period >= 3.5x p50) to respect period > worst compute (no backlog).
# Env: PY (python3 with onnxruntime) and ONNX_DIR (ONNX models) override the default paths below.
set -u
PY="${PY:-/home/dk/mobile/rt-infer-bench/.venv/bin/python3}"
ONNX_DIR="${ONNX_DIR:-/home/dk/mobile/rt-infer-bench/models}"
LAB="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MN=$ONNX_DIR/mobilenetv2-12.onnx
G=/sys/devices/platform/17000000.gpu/devfreq/17000000.gpu
cd "$LAB" || exit 1; mkdir -p results/batch_axis
restore(){ sudo bash -c "$LAB/pilot_emc/emc_ctl.sh unlock >/dev/null 2>&1
  echo 306000000 > $G/min_freq; echo 1020000000 > $G/max_freq
  echo 950000 > /proc/sys/kernel/sched_rt_runtime_us
  for c in /sys/devices/system/cpu/cpu*/cpufreq; do echo schedutil > \$c/scaling_governor; done
  chown -R dk:dk results/batch_axis 2>/dev/null"; }
trap restore EXIT
sudo bash -c "echo -1 > /proc/sys/kernel/sched_rt_runtime_us
for c in /sys/devices/system/cpu/cpu*/cpufreq; do echo performance > \$c/scaling_governor; done
echo 1020000000 > $G/min_freq; echo 1020000000 > $G/max_freq"

for EMC in 3199 2133 204; do
  [ "$EMC" = 204 ] && EHZ=204000000 || EHZ=${EMC}000000
  sudo bash -c "$LAB/pilot_emc/emc_ctl.sh lock $EHZ >/dev/null"; sleep 1
  for B in 1 4 8; do
    # probe p50 to size the period
    BENCH_BATCH=$B "$PY" -m harness.infer_bench --backend onnxruntime --model "$MN" \
      --hz 2 --iters 16 --warmup 8 --cpu 5 --label probe --out /tmp/c9probe >/dev/null 2>&1
    P50=$( "$PY" -c "import json;print(json.load(open('/tmp/c9probe.json'))['compute_us']['p50']/1000)" )
    HZ=$( "$PY" -c "import math;print(max(1,min(50,int(1000/(3.5*$P50)))))" )
    BENCH_BATCH=$B "$PY" -m harness.infer_bench --backend onnxruntime --model "$MN" \
      --hz "$HZ" --iters 400 --warmup 40 --cpu 5 --label c9_e${EMC}_b${B} \
      --out results/batch_axis/mn_e${EMC}_b${B} >/dev/null 2>&1
    echo "  emc${EMC} b${B}: probe_p50=${P50}ms hz=${HZ} -> $([ -s results/batch_axis/mn_e${EMC}_b${B}.json ] && echo OK || echo FAIL)"
  done
done
echo "C9_BATCH DONE"

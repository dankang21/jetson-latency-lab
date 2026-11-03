#!/bin/bash
# JOINT: EMC + horizon + tenancy composed into ONE live policy (NX).
# Layers: EMC -> 2-cell refit @2133 (reuses committed nx_validation cells);
#         horizon -> drift coefficient gamma (vision: stationary, gamma=0);
#         tenancy -> 200-cycle max-clock probe under the co-runner => M_tail.
# Live sequence: (1) probe under co, (2) policy picks for
#   {blind, emc-solo, joint} x D in {9,12} ms, (3) deploy every picked clock
#   under co x3 (union of candidate picks so the analysis maps pick->cells).
# Victim: MobileNetV2 @ hz33, EMC locked 2133. Co-runner: ViT-Small @10Hz.
# Env: PY (python3 with onnxruntime) and ONNX_DIR (ONNX models) override the default paths below.
set -u
PY="${PY:-/home/dk/mobile/rt-infer-bench/.venv/bin/python3}"
ONNX_DIR="${ONNX_DIR:-/home/dk/mobile/rt-infer-bench/models}"
LAB="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MN=$ONNX_DIR/mobilenetv2-12.onnx
VIT=$ONNX_DIR/vit_small_patch16_224.onnx
G=/sys/devices/platform/17000000.gpu/devfreq/17000000.gpu
cd "$LAB" || exit 1; mkdir -p results/joint_gov
CO_PID=""
restore(){ [ -n "$CO_PID" ] && kill "$CO_PID" 2>/dev/null
  sudo bash -c "$LAB/pilot_emc/emc_ctl.sh unlock >/dev/null 2>&1
  echo 306000000 > $G/min_freq; echo 1173000000 > $G/max_freq
  echo 950000 > /proc/sys/kernel/sched_rt_runtime_us
  for c in /sys/devices/system/cpu/cpu*/cpufreq; do echo schedutil > \$c/scaling_governor; done
  chown -R dk:dk results/joint_gov 2>/dev/null"; }
trap restore EXIT
setclk(){ sudo bash -c "echo ${1}000000 > $G/min_freq; echo ${1}000000 > $G/max_freq"; }
sudo bash -c "echo -1 > /proc/sys/kernel/sched_rt_runtime_us
for c in /sys/devices/system/cpu/cpu*/cpufreq; do echo performance > \$c/scaling_governor; done"
sudo bash -c "$LAB/pilot_emc/emc_ctl.sh lock 2133000000 >/dev/null"

co_start(){
  "$PY" -m harness.infer_bench --backend onnxruntime --model "$VIT" \
    --hz 10 --iters 100000 --warmup 30 --cpu 5 --label co_vit \
    --out /tmp/co_vit_jg >/dev/null 2>&1 & CO_PID=$!
  sleep 20
}
co_stop(){ [ -n "$CO_PID" ] && kill "$CO_PID" 2>/dev/null; wait "$CO_PID" 2>/dev/null; CO_PID=""; sleep 1; }

echo "=== PHASE 1: tenancy probe (200 cycles, max clock, under co) ==="
co_start
setclk 1173; sleep 0.4
"$PY" -m harness.infer_bench --backend onnxruntime --model "$MN" \
  --hz 33 --iters 200 --warmup 60 --cpu 7 --label jg_probe \
  --out results/joint_gov/probe_co_g1173 >/dev/null 2>&1
echo "  probe: $([ -s results/joint_gov/probe_co_g1173.json ] && echo OK || echo FAIL)"

echo "=== PHASE 2: policy picks (logged before deployment) ==="
"$PY" campaign/joint_gov_policy.py | tee results/joint_gov/picks.txt

echo "=== PHASE 3: deploy candidate picks under co (x3, 1000 cycles) ==="
for f in 510 612 1020 1122; do
  for r in 1 2 3; do
    setclk "$f"; sleep 0.4
    "$PY" -m harness.infer_bench --backend onnxruntime --model "$MN" \
      --hz 33 --iters 1000 --warmup 60 --cpu 7 --label jg_dep_g${f}_r${r} \
      --out results/joint_gov/dep_co_g${f}_r${r} >/dev/null 2>&1
  done
  echo "  dep gpu${f} x3: $([ -s results/joint_gov/dep_co_g${f}_r3.json ] && echo OK || echo FAIL)"
done
co_stop
echo "JOINT_GOV DONE"

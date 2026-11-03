#!/bin/bash
# TMC calibration round: enlarge independent-run counts on the three
# true-admission cells.
#   A: vision co-tenancy guarded pick (MNv2 @918, D=12) r4..r10  (3->10)
#   B: joint Qwen+ViT co-tenant accept (@1173, 2k tokens) r4..r9 (3->9)
#   C: horizon-only solo decode (@1173, 2k tokens) r6..r8        (5->8)
# Env: PY (python3 with onnxruntime), ONNX_DIR (ONNX models) and GGUF_DIR (GGUF models) override the default paths below.
set -u
PY="${PY:-/home/dk/mobile/rt-infer-bench/.venv/bin/python3}"
ONNX_DIR="${ONNX_DIR:-/home/dk/mobile/rt-infer-bench/models}"
LAB="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MN=$ONNX_DIR/mobilenetv2-12.onnx
VIT=$ONNX_DIR/vit_small_patch16_224.onnx
BENCH=$LAB/tools/slm_decode_bench
MD="${GGUF_DIR:-/home/dk/mobile/models}"
G=/sys/devices/platform/17000000.gpu/devfreq/17000000.gpu
cd "$LAB" || exit 1
CO_PID=""
restore(){ [ -n "$CO_PID" ] && kill "$CO_PID" 2>/dev/null
  sudo bash -c "$LAB/pilot_emc/emc_ctl.sh unlock >/dev/null 2>&1
  echo 306000000 > $G/min_freq; echo 1173000000 > $G/max_freq
  echo 950000 > /proc/sys/kernel/sched_rt_runtime_us
  for c in /sys/devices/system/cpu/cpu*/cpufreq; do echo schedutil > \$c/scaling_governor; done
  chown -R dk:dk results/joint_gov results/joint_hxt results/boundary_stats 2>/dev/null"; }
trap restore EXIT
setclk(){ sudo bash -c "echo ${1}000000 > $G/min_freq; echo ${1}000000 > $G/max_freq"; }
sudo bash -c "echo -1 > /proc/sys/kernel/sched_rt_runtime_us
for c in /sys/devices/system/cpu/cpu*/cpufreq; do echo performance > \$c/scaling_governor; done"
sudo bash -c "$LAB/pilot_emc/emc_ctl.sh lock 2133000000 >/dev/null"

"$PY" -m harness.infer_bench --backend onnxruntime --model "$VIT" \
  --hz 10 --iters 100000 --warmup 30 --cpu 5 --label co_vit \
  --out /tmp/co_vit_reps >/dev/null 2>&1 & CO_PID=$!
sleep 20

echo "=== A: vision co @918, r4..r10 ==="
setclk 918; sleep 0.4
for r in 4 5 6 7 8 9 10; do
  "$PY" -m harness.infer_bench --backend onnxruntime --model "$MN" \
    --hz 33 --iters 1000 --warmup 60 --cpu 7 --label rep_g918_r${r} \
    --out results/joint_gov/dep_co_g918_r${r} >/dev/null 2>&1
  echo "  r${r}: $([ -s results/joint_gov/dep_co_g918_r${r}.json ] && echo OK || echo FAIL)"
done

echo "=== B: joint decode co @1173 2k, r4..r9 ==="
setclk 1173; sleep 0.5
for r in 4 5 6 7 8 9; do
  "$BENCH" --model "$MD/qwen2.5-1.5b-instruct-q4_k_m.gguf" \
    --tokens 2000 --warmup 40 --cpu 7 --prio 80 \
    --out results/joint_hxt/q15co_g1173_r${r} >/dev/null 2>&1
  echo "  r${r}: $([ -s results/joint_hxt/q15co_g1173_r${r}.json ] && echo OK || echo FAIL)"
done
kill "$CO_PID" 2>/dev/null; wait "$CO_PID" 2>/dev/null; CO_PID=""; sleep 2

echo "=== C: horizon solo @1173 2k, r6..r8 ==="
for r in 6 7 8; do
  "$BENCH" --model "$MD/qwen2.5-1.5b-instruct-q4_k_m.gguf" \
    --tokens 2000 --warmup 40 --cpu 7 --prio 80 \
    --out results/boundary_stats/q15_g1173_r${r} >/dev/null 2>&1
  echo "  r${r}: $([ -s results/boundary_stats/q15_g1173_r${r}.json ] && echo OK || echo FAIL)"
done
echo "TMC_REPS DONE"

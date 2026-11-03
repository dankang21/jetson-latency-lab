#!/bin/bash
# JOINT-HxT: horizon x tenancy in ONE workload (NX) — the composed policy's
# missing joint-stress cell. Qwen2.5-1.5B Q4 2,000-token decode against the
# 10 Hz ViT-Small co-runner, EMC locked 2133.
#   probe : 200-token decode at top clock UNDER co (level + tail, bounded)
#   deploy: 2,000-token decode x3 at {1122, 1173} under co
# Policy scoring is offline (analyze side): T_co(f)*(1+gamma*H)*r + M <= D
# with T_co level and M = p98-p50 both from the probe only.
# Env: PY (python3 with onnxruntime), ONNX_DIR (ONNX models) and GGUF_DIR (GGUF models) override the default paths below.
set -u
PY="${PY:-/home/dk/mobile/rt-infer-bench/.venv/bin/python3}"
ONNX_DIR="${ONNX_DIR:-/home/dk/mobile/rt-infer-bench/models}"
LAB="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BENCH=$LAB/tools/slm_decode_bench
MD="${GGUF_DIR:-/home/dk/mobile/models}"
VIT=$ONNX_DIR/vit_small_patch16_224.onnx
G=/sys/devices/platform/17000000.gpu/devfreq/17000000.gpu
cd "$LAB" || exit 1; mkdir -p results/joint_hxt
CO_PID=""
restore(){ [ -n "$CO_PID" ] && kill "$CO_PID" 2>/dev/null
  sudo bash -c "$LAB/pilot_emc/emc_ctl.sh unlock >/dev/null 2>&1
  echo 306000000 > $G/min_freq; echo 1173000000 > $G/max_freq
  echo 950000 > /proc/sys/kernel/sched_rt_runtime_us
  for c in /sys/devices/system/cpu/cpu*/cpufreq; do echo schedutil > \$c/scaling_governor; done
  chown -R dk:dk results/joint_hxt 2>/dev/null"; }
trap restore EXIT
setclk(){ sudo bash -c "echo ${1}000000 > $G/min_freq; echo ${1}000000 > $G/max_freq"; }
sudo bash -c "echo -1 > /proc/sys/kernel/sched_rt_runtime_us
for c in /sys/devices/system/cpu/cpu*/cpufreq; do echo performance > \$c/scaling_governor; done"
sudo bash -c "$LAB/pilot_emc/emc_ctl.sh lock 2133000000 >/dev/null"

"$PY" -m harness.infer_bench --backend onnxruntime --model "$VIT" \
  --hz 10 --iters 100000 --warmup 30 --cpu 5 --label co_vit \
  --out /tmp/co_vit_hxt >/dev/null 2>&1 & CO_PID=$!
sleep 20

echo "=== probe: 200-token decode @1173 under co ==="
setclk 1173; sleep 0.5
"$BENCH" --model "$MD/qwen2.5-1.5b-instruct-q4_k_m.gguf" \
  --tokens 200 --warmup 40 --cpu 7 --prio 80 \
  --out results/joint_hxt/probe_co_g1173 >/dev/null 2>&1
echo "  probe: $([ -s results/joint_hxt/probe_co_g1173.json ] && echo OK || echo FAIL)"

echo "=== deploy: 2000-token decode x3 under co ==="
for f in 1173 1122; do
  setclk "$f"; sleep 0.5
  for r in 1 2 3; do
    "$BENCH" --model "$MD/qwen2.5-1.5b-instruct-q4_k_m.gguf" \
      --tokens 2000 --warmup 40 --cpu 7 --prio 80 \
      --out "results/joint_hxt/q15co_g${f}_r${r}" >/dev/null 2>&1
    echo "  g${f} r${r}: $([ -s results/joint_hxt/q15co_g${f}_r${r}.json ] && echo OK || echo FAIL)"
  done
done
echo "JOINT_HXT DONE"

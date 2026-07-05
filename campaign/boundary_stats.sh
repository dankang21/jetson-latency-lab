#!/bin/bash
# B1 (Blocker #2): boundary-cell statistical hardening on the NX.
# 10 independent 1000-cycle MNv2 deployments at the aware clocks (1173, 1122)
# + 5x2000-token Qwen 1.5B/3B runs at the aware clocks. Miss-rate run-level CIs.
set -u
PY=/home/dk/mobile/rt-infer-bench/.venv/bin/python3
LAB=/home/dk/mobile/jetson-latency-lab
BENCH=$LAB/tools/slm_decode_bench
MN=/home/dk/mobile/rt-infer-bench/models/mobilenetv2-12.onnx
Q15=/home/dk/mobile/models/qwen2.5-1.5b-instruct-q4_k_m.gguf
Q3=/home/dk/mobile/models/qwen2.5-3b-instruct-q4_k_m.gguf
G=/sys/devices/platform/17000000.gpu/devfreq/17000000.gpu
cd "$LAB" || exit 1; mkdir -p results/boundary_stats
restore(){ sudo bash -c "$LAB/pilot_emc/emc_ctl.sh unlock >/dev/null 2>&1
  echo 306000000 > $G/min_freq; echo 1173000000 > $G/max_freq
  echo 950000 > /proc/sys/kernel/sched_rt_runtime_us
  for c in /sys/devices/system/cpu/cpu*/cpufreq; do echo schedutil > \$c/scaling_governor; done
  chown -R dk:dk results/boundary_stats 2>/dev/null"; }
trap restore EXIT
sudo bash -c "echo -1 > /proc/sys/kernel/sched_rt_runtime_us
for c in /sys/devices/system/cpu/cpu*/cpufreq; do echo performance > \$c/scaling_governor; done"
setclk(){ sudo bash -c "echo ${1}000000 > $G/min_freq; echo ${1}000000 > $G/max_freq"; }
sudo bash -c "$LAB/pilot_emc/emc_ctl.sh lock 2133000000 >/dev/null"

echo "=== MNv2 10x1000 @ aware clocks ==="
for f in 1173 1122; do
  for r in $(seq 1 10); do
    setclk $f; sleep 0.4
    "$PY" -m harness.infer_bench --backend onnxruntime --model "$MN" \
      --hz 33 --iters 1000 --warmup 120 --cpu 7 --label bs_mn_g${f}_r${r} \
      --out results/boundary_stats/mn_g${f}_r${r} >/dev/null 2>&1
    echo "  mn g${f} r${r}: $([ -s results/boundary_stats/mn_g${f}_r${r}.json ] && echo OK || echo FAIL)"
  done
done
echo "=== Qwen1.5B 5x2000tok @ aware clocks ==="
for f in 918 1020 1122 1173; do
  for r in 1 2 3 4 5; do
    setclk $f; sleep 0.4
    "$BENCH" --model "$Q15" --tokens 2000 --warmup 40 --cpu 7 --prio 80 \
      --out results/boundary_stats/q15_g${f}_r${r} >/dev/null 2>&1
    echo "  q15 g${f} r${r}: $([ -s results/boundary_stats/q15_g${f}_r${r}.json ] && echo OK || echo FAIL)"
  done
done
echo "=== Qwen3B 5x2000tok @ aware clocks ==="
for f in 1020 1122 1173; do
  for r in 1 2 3 4 5; do
    setclk $f; sleep 0.4
    "$BENCH" --model "$Q3" --tokens 2000 --warmup 40 --cpu 7 --prio 80 \
      --out results/boundary_stats/q3_g${f}_r${r} >/dev/null 2>&1
    echo "  q3 g${f} r${r}: $([ -s results/boundary_stats/q3_g${f}_r${r}.json ] && echo OK || echo FAIL)"
  done
done
echo "BOUNDARY_STATS DONE"

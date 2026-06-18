#!/bin/bash
# ViT-Small deployed-governor generalization on the NX (hebesu).
# Phase 1: estimator sweep @ EMC {3199,2133} x GPU freqs (1 run/cell) -> policy fit.
# Phase 2: 3-repeat deployment @ EMC 2133 across the GPU-responsive range + energy.
# infer_bench runs as the invoking USER; clocks/tegrastats via sudo. Self-contained.
set -u
PY=/home/dk/mobile/rt-infer-bench/.venv/bin/python3
MODEL=/home/dk/mobile/rt-infer-bench/models/vit_small_patch16_224.onnx
G=/sys/devices/platform/17000000.gpu/devfreq/17000000.gpu
LAB=/home/dk/mobile/jetson-latency-lab
cd "$LAB" || exit 1
mkdir -p results/nx_validation results/govrep_vit
FREQS="306 408 510 612 714 816 918 1020 1122 1173"
DEPLOY_FREQS="612 714 816 918 1020 1122 1173"

setclk(){ sudo bash -c "echo ${1}000000 > $G/min_freq; echo ${1}000000 > $G/max_freq"; }
restore(){ sudo bash -c "$LAB/pilot_emc/emc_ctl.sh unlock >/dev/null 2>&1
  echo 306000000 > $G/min_freq; echo 1173000000 > $G/max_freq
  echo 950000 > /proc/sys/kernel/sched_rt_runtime_us
  for c in /sys/devices/system/cpu/cpu*/cpufreq; do echo schedutil > \$c/scaling_governor; done
  chown -R dk:dk results/nx_validation results/govrep_vit 2>/dev/null"; }
trap restore EXIT

sudo bash -c "echo -1 > /proc/sys/kernel/sched_rt_runtime_us
for c in /sys/devices/system/cpu/cpu*/cpufreq; do echo performance > \$c/scaling_governor; done"

echo "=== PHASE 1: estimator sweep ==="
for EMC in 3199 2133; do
  sudo bash -c "$LAB/pilot_emc/emc_ctl.sh lock ${EMC}000000 >/dev/null"
  for f in $FREQS; do
    setclk $f; sleep 0.4
    "$PY" -m harness.infer_bench --backend onnxruntime --model "$MODEL" \
      --hz 20 --iters 800 --warmup 100 --cpu 7 --label vest_e${EMC}_g${f} \
      --out results/nx_validation/nx_est_emc${EMC}_gpu${f}_vit >/dev/null 2>&1
    echo "  sweep emc${EMC} gpu${f}: $([ -s results/nx_validation/nx_est_emc${EMC}_gpu${f}_vit.json ] && echo OK || echo FAIL)"
  done
done

echo "=== PHASE 2: 3-repeat deployment @ EMC 2133 ==="
sudo bash -c "$LAB/pilot_emc/emc_ctl.sh lock 2133000000 >/dev/null"
for f in $DEPLOY_FREQS; do
  for r in 1 2 3; do
    setclk $f; sleep 0.4
    sudo timeout 70 tegrastats --interval 100 > results/govrep_vit/g${f}_r${r}.tegra 2>/dev/null &
    "$PY" -m harness.infer_bench --backend onnxruntime --model "$MODEL" \
      --hz 20 --iters 1000 --warmup 120 --cpu 7 --label vgr_g${f}_r${r} \
      --out results/govrep_vit/g${f}_r${r} >/dev/null 2>&1
    sudo pkill -x tegrastats 2>/dev/null; sleep 0.3
    echo "  deploy g${f}_r${r}: $([ -s results/govrep_vit/g${f}_r${r}.json ] && echo OK || echo FAIL)"
  done
done
echo "VIT_DEPLOY DONE"

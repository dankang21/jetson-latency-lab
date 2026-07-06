#!/bin/bash
# C3: tenancy — does the blind-vs-aware separation survive a GPU co-runner,
# and does a solo-profiled EMC table remain sufficient?
# Victim: MobileNetV2 @ hz33, EMC 2133 (deployment point), NX.
# Co-runner: ViT-Small at a fixed 10 Hz (rate-limited, ~30-40% GPU duty),
#            second infer_bench process on other cores.
# Arms: co in {solo(reuse existing), vit10} x policy {blind, aware-solo,
#       aware-reprofiled-under-co}; PLUS EMC-pin check under two clients.
set -u
PY=/home/dk/mobile/rt-infer-bench/.venv/bin/python3
LAB=/home/dk/mobile/jetson-latency-lab
MN=/home/dk/mobile/rt-infer-bench/models/mobilenetv2-12.onnx
VIT=/home/dk/mobile/rt-infer-bench/models/vit_small_patch16_224.onnx
G=/sys/devices/platform/17000000.gpu/devfreq/17000000.gpu
cd "$LAB" || exit 1; mkdir -p results/tenancy
CO_PID=""
restore(){ [ -n "$CO_PID" ] && kill "$CO_PID" 2>/dev/null
  sudo bash -c "$LAB/pilot_emc/emc_ctl.sh unlock >/dev/null 2>&1
  echo 306000000 > $G/min_freq; echo 1173000000 > $G/max_freq
  echo 950000 > /proc/sys/kernel/sched_rt_runtime_us
  for c in /sys/devices/system/cpu/cpu*/cpufreq; do echo schedutil > \$c/scaling_governor; done
  chown -R dk:dk results/tenancy 2>/dev/null"; }
trap restore EXIT
setclk(){ sudo bash -c "echo ${1}000000 > $G/min_freq; echo ${1}000000 > $G/max_freq"; }
sudo bash -c "echo -1 > /proc/sys/kernel/sched_rt_runtime_us
for c in /sys/devices/system/cpu/cpu*/cpufreq; do echo performance > \$c/scaling_governor; done"
sudo bash -c "$LAB/pilot_emc/emc_ctl.sh lock 2133000000 >/dev/null"

co_start(){ # long-running rate-limited ViT co-runner on cpu 5
  "$PY" -m harness.infer_bench --backend onnxruntime --model "$VIT" \
    --hz 10 --iters 100000 --warmup 30 --cpu 5 --label co_vit \
    --out /tmp/co_vit >/dev/null 2>&1 & CO_PID=$!
  sleep 20   # co-runner warm + steady
}
co_stop(){ [ -n "$CO_PID" ] && kill "$CO_PID" 2>/dev/null; wait "$CO_PID" 2>/dev/null; CO_PID=""; sleep 1; }

echo "=== PHASE 1: victim profile UNDER co-runner (2-cell refit basis @2133) ==="
co_start
for f in 306 612 918 1020 1122 1173; do
  setclk "$f"; sleep 0.4
  "$PY" -m harness.infer_bench --backend onnxruntime --model "$MN" \
    --hz 33 --iters 600 --warmup 60 --cpu 7 --label ten_prof_g${f} \
    --out results/tenancy/prof_co_g${f} >/dev/null 2>&1
  echo "  prof-under-co gpu${f}: $([ -s results/tenancy/prof_co_g${f}.json ] && echo OK || echo FAIL)"
done
co_stop

echo "=== PHASE 2: deployments under co-runner (all candidate clocks x 3) ==="
co_start
for f in 918 1020 1122 1173; do
  for r in 1 2 3; do
    setclk "$f"; sleep 0.4
    taskset -c 0 "$PY" campaign/ina_log.py results/tenancy/dep_co_g${f}_r${r}.power.csv 45 & LP=$!
    "$PY" -m harness.infer_bench --backend onnxruntime --model "$MN" \
      --hz 33 --iters 1000 --warmup 60 --cpu 7 --label ten_dep_g${f}_r${r} \
      --out results/tenancy/dep_co_g${f}_r${r} >/dev/null 2>&1
    kill -TERM $LP 2>/dev/null; wait $LP 2>/dev/null
  done
  echo "  dep-under-co gpu${f} x3: $([ -s results/tenancy/dep_co_g${f}_r3.json ] && echo OK || echo FAIL)"
done
co_stop

echo "=== PHASE 3: EMC-pin robustness under TWO GPU clients (stock EMC, 90s) ==="
sudo bash -c "$LAB/pilot_emc/emc_ctl.sh unlock >/dev/null"   # EMC ungoverned
setclk 1173
co_start
"$PY" -m harness.infer_bench --backend onnxruntime --model "$MN" \
  --hz 33 --iters 2500 --warmup 60 --cpu 7 --label ten_pin \
  --out results/tenancy/pin_check >/dev/null 2>&1 &
VP=$!
sudo timeout 85 tegrastats --interval 500 > results/tenancy/pin_check.tegra 2>/dev/null
wait $VP 2>/dev/null
co_stop
grep -oE "EMC_FREQ [0-9]+%@[0-9]+" results/tenancy/pin_check.tegra | awk -F@ '{print $2}' | sort | uniq -c | sed 's/^/  EMC dist: /'
echo "C3_TENANCY DONE"

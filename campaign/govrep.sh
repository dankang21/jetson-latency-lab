#!/bin/bash
# E3 governor repeats: mobilenet @ EMC 2133, 5 GPU freqs x 3 repeats, with energy.
# Self-contained (defines its own vars); infer_bench runs as the invoking USER.
set -u
PY=/home/dk/mobile/rt-infer-bench/.venv/bin/python3
MODEL=/home/dk/mobile/rt-infer-bench/models/mobilenetv2-12.onnx
G=/sys/devices/platform/17000000.gpu/devfreq/17000000.gpu
cd /home/dk/mobile/jetson-latency-lab
mkdir -p results/govrep
sudo bash -c "echo -1 > /proc/sys/kernel/sched_rt_runtime_us
for c in /sys/devices/system/cpu/cpu*/cpufreq; do echo performance > \$c/scaling_governor; done
./pilot_emc/emc_ctl.sh lock 2133000000 >/dev/null"
for f in 510 612 714 816 918; do
  for r in 1 2 3; do
    sudo bash -c "echo ${f}000000 > $G/min_freq; echo ${f}000000 > $G/max_freq"
    sleep 0.4
    sudo timeout 45 tegrastats --interval 100 > results/govrep/g${f}_r${r}.tegra 2>/dev/null &
    "$PY" -m harness.infer_bench --backend onnxruntime --model "$MODEL" \
      --hz 33 --iters 1000 --warmup 120 --cpu 7 --label gr_g${f}_r${r} \
      --out results/govrep/g${f}_r${r} >/dev/null 2>&1
    sudo pkill -x tegrastats 2>/dev/null
    sleep 0.3
    echo "g${f}_r${r}: $([ -s results/govrep/g${f}_r${r}.json ] && echo OK || echo FAIL)"
  done
done
sudo bash -c "./pilot_emc/emc_ctl.sh unlock >/dev/null
echo 306000000 > $G/min_freq; echo 1173000000 > $G/max_freq
echo 950000 > /proc/sys/kernel/sched_rt_runtime_us
for c in /sys/devices/system/cpu/cpu*/cpufreq; do echo schedutil > \$c/scaling_governor; done
chown -R dk:dk results/govrep"
echo "GOVREP DONE"

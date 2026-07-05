#!/bin/bash
# NX Wave-2 chain: B3 interleaved dE pairs -> B5 kernel identity (nsys) ->
# C5 directed-pair coverage -> C4a sustained Qwen decode soak (1h, segmented).
set -u
PY=/home/dk/mobile/rt-infer-bench/.venv/bin/python3
LAB=/home/dk/mobile/jetson-latency-lab
BENCH=$LAB/tools/slm_decode_bench
MN=/home/dk/mobile/rt-infer-bench/models/mobilenetv2-12.onnx
VIT=/home/dk/mobile/rt-infer-bench/models/vit_small_patch16_224.onnx
CPX=$LAB/pilot_emc/models/compute_proxy_fp16.onnx
Q15=/home/dk/mobile/models/qwen2.5-1.5b-instruct-q4_k_m.gguf
G=/sys/devices/platform/17000000.gpu/devfreq/17000000.gpu
cd "$LAB" || exit 1
mkdir -p results/ab_energy results/mech_nx results/nx_pairs results/soak_qwen
restore(){ sudo bash -c "$LAB/pilot_emc/emc_ctl.sh unlock >/dev/null 2>&1
  echo 306000000 > $G/min_freq; echo 1173000000 > $G/max_freq
  echo 950000 > /proc/sys/kernel/sched_rt_runtime_us
  for c in /sys/devices/system/cpu/cpu*/cpufreq; do echo schedutil > \$c/scaling_governor; done
  chown -R dk:dk results/ab_energy results/mech_nx results/nx_pairs results/soak_qwen 2>/dev/null"; }
trap restore EXIT
setclk(){ sudo bash -c "echo ${1}000000 > $G/min_freq; echo ${1}000000 > $G/max_freq"; }
sudo bash -c "echo -1 > /proc/sys/kernel/sched_rt_runtime_us
for c in /sys/devices/system/cpu/cpu*/cpufreq; do echo performance > \$c/scaling_governor; done"

echo "=== B3: interleaved A/B dE pairs @ EMC 2133 ==="
sudo bash -c "$LAB/pilot_emc/emc_ctl.sh lock 2133000000 >/dev/null"
run_ab(){ # run_ab <model> <hz> <ina> <fa> <fb> <tag>
  local M=$1 HZ=$2 INA=$3 FA=$4 FB=$5 TAG=$6
  for p in $(seq 1 10); do
    for f in $FA $FB; do
      setclk "$f"; sleep 0.4
      taskset -c 0 "$PY" campaign/ina_log.py "results/ab_energy/${TAG}_p${p}_g${f}.power.csv" "$INA" & local LP=$!
      "$PY" -m harness.infer_bench --backend onnxruntime --model "$M" \
        --hz "$HZ" --iters 1000 --warmup 60 --cpu 7 --label ab_${TAG}_p${p}_g${f} \
        --out "results/ab_energy/${TAG}_p${p}_g${f}" >/dev/null 2>&1
      kill -TERM $LP 2>/dev/null; wait $LP 2>/dev/null
    done
    echo "  ${TAG} pair $p done"
  done
}
run_ab "$MN" 33 45 1122 918 mn
run_ab "$VIT" 20 65 918 816 vit

echo "=== B5: kernel-identity nsys @ GPU 1173 ==="
NSYS=$(command -v nsys || ls /usr/local/cuda*/bin/nsys /opt/nvidia/nsight-systems*/bin/nsys 2>/dev/null | head -1)
if [ -n "$NSYS" ]; then
  for EMC in 2133 3199; do
    sudo bash -c "$LAB/pilot_emc/emc_ctl.sh lock ${EMC}000000 >/dev/null"
    setclk 1173
    for r in 1 2 3; do
      "$NSYS" profile -o "results/mech_nx/e${EMC}_r${r}" --force-overwrite true \
        "$PY" -m harness.infer_bench --backend onnxruntime --model "$CPX" \
        --hz 4 --iters 170 --warmup 20 --cpu 7 --label nxmech_e${EMC}_r${r} \
        --out "results/mech_nx/e${EMC}_r${r}_bench" >/dev/null 2>&1
      "$NSYS" export --type sqlite --force-overwrite true -o "results/mech_nx/e${EMC}_r${r}.sqlite" "results/mech_nx/e${EMC}_r${r}.nsys-rep" >/dev/null 2>&1
      echo -n "  nx e${EMC} r${r}: "
      "$PY" campaign/mech_kdur_extract.py "results/mech_nx/e${EMC}_r${r}.sqlite" "results/mech_nx/kdur_e${EMC}_r${r}.txt" || echo "EXTRACT FAIL"
      rm -f "results/mech_nx/e${EMC}_r${r}.nsys-rep" "results/mech_nx/e${EMC}_r${r}.sqlite"
    done
  done
else
  echo "  nsys NOT FOUND on NX — B5 skipped (report in analysis)"
fi

echo "=== C5: directed-pair coverage (MNv2) ==="
# fit-source sweep at 665.6 MHz
sudo bash -c "$LAB/pilot_emc/emc_ctl.sh lock 665600000 >/dev/null"
for f in 306 408 510 612 714 816 918 1020 1122 1173; do
  setclk "$f"; sleep 0.4
  "$PY" -m harness.infer_bench --backend onnxruntime --model "$MN" \
    --hz 33 --iters 800 --warmup 60 --cpu 7 --label est665_g${f} \
    --out "results/nx_validation/nx_est_emc665_gpu${f}_mobilenet" >/dev/null 2>&1
  echo "  sweep emc665 gpu${f}: $([ -s results/nx_validation/nx_est_emc665_gpu${f}_mobilenet.json ] && echo OK || echo FAIL)"
done
# deployments at target EMC 3199 and 665.6 (2133 deploys already exist in govrep/)
for EMC in 3199 665; do
  [ "$EMC" = 665 ] && HZ_E=665600000 || HZ_E=${EMC}000000
  sudo bash -c "$LAB/pilot_emc/emc_ctl.sh lock $HZ_E >/dev/null"
  for f in 612 714 816 918 1020 1122 1173; do
    for r in 1 2 3; do
      setclk "$f"; sleep 0.4
      taskset -c 0 "$PY" campaign/ina_log.py "results/nx_pairs/e${EMC}_g${f}_r${r}.power.csv" 45 & LP=$!
      "$PY" -m harness.infer_bench --backend onnxruntime --model "$MN" \
        --hz 33 --iters 1000 --warmup 60 --cpu 7 --label pr_e${EMC}_g${f}_r${r} \
        --out "results/nx_pairs/e${EMC}_g${f}_r${r}" >/dev/null 2>&1
      kill -TERM $LP 2>/dev/null; wait $LP 2>/dev/null
    done
    echo "  deploy e${EMC} g${f} x3: $([ -s results/nx_pairs/e${EMC}_g${f}_r3.json ] && echo OK || echo FAIL)"
  done
done

echo "=== C4a: sustained Qwen-1.5B decode soak (60 x 2000 tokens @ aware 1020, EMC 2133) ==="
sudo bash -c "$LAB/pilot_emc/emc_ctl.sh lock 2133000000 >/dev/null"
setclk 1020
sudo tegrastats --interval 1000 > results/soak_qwen/soak.tegra 2>/dev/null &
taskset -c 0 "$PY" campaign/ina_log.py results/soak_qwen/soak.power.csv 4200 & LP=$!
for seg in $(seq 1 60); do
  "$BENCH" --model "$Q15" --tokens 2000 --warmup 10 --cpu 7 --prio 80 \
    --out "results/soak_qwen/seg${seg}" >/dev/null 2>&1
  [ $((seg % 10)) = 0 ] && echo "  soak segment $seg/60"
done
kill -TERM $LP 2>/dev/null; wait $LP 2>/dev/null
sudo pkill -x tegrastats 2>/dev/null
echo "NX_WAVE2 DONE"

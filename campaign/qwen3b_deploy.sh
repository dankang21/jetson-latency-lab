#!/bin/bash
# Qwen2.5-3B-Instruct (Q4_K_M) deployed-governor: the LLM-scale second point
# (mirrors qwen_deploy.sh, which measured the 1.5B). Saturated token decode;
# miss@D = P(per-token decode > D); energy/token = mean VDD_IN power x p50 decode.
# Env: PY (python3 with onnxruntime) and GGUF_DIR (GGUF models) override the default paths below.
set -u
PY="${PY:-/home/dk/mobile/rt-infer-bench/.venv/bin/python3}"
GGUF_DIR="${GGUF_DIR:-/home/dk/mobile/models}"
LAB="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BENCH=$LAB/tools/slm_decode_bench
MODEL=$GGUF_DIR/qwen2.5-3b-instruct-q4_k_m.gguf
G=/sys/devices/platform/17000000.gpu/devfreq/17000000.gpu
cd "$LAB" || exit 1; mkdir -p results/qwen3b_deploy
restore(){ sudo bash -c "$LAB/pilot_emc/emc_ctl.sh unlock >/dev/null 2>&1
  echo 306000000 > $G/min_freq; echo 1173000000 > $G/max_freq
  echo 950000 > /proc/sys/kernel/sched_rt_runtime_us
  for c in /sys/devices/system/cpu/cpu*/cpufreq; do echo schedutil > \$c/scaling_governor; done
  chown -R dk:dk results/qwen3b_deploy 2>/dev/null"; }
trap restore EXIT
sudo bash -c "echo -1 > /proc/sys/kernel/sched_rt_runtime_us
for c in /sys/devices/system/cpu/cpu*/cpufreq; do echo performance > \$c/scaling_governor; done"
setclk(){ sudo bash -c "echo ${1}000000 > $G/min_freq; echo ${1}000000 > $G/max_freq"; }

echo "=== PHASE 1: estimator sweep ==="
for EMC in 3199 2133; do
  sudo bash -c "$LAB/pilot_emc/emc_ctl.sh lock ${EMC}000000 >/dev/null"
  for f in 510 612 714 816 918 1020 1122 1173; do
    setclk "$f"; sleep 0.4
    "$BENCH" --model "$MODEL" --tokens 150 --warmup 40 --cpu 7 --prio 80 \
      --out results/qwen3b_deploy/qsweep_emc${EMC}_gpu${f} >/dev/null 2>&1
    echo "  sweep emc${EMC} gpu${f}: $([ -s results/qwen3b_deploy/qsweep_emc${EMC}_gpu${f}.json ] && echo OK || echo FAIL)"
  done
done

echo "=== PHASE 2: 3-repeat deployment @ EMC 2133 + INA3221 VDD_IN log (max rate, ~3.6 ms per sample) ==="
sudo bash -c "$LAB/pilot_emc/emc_ctl.sh lock 2133000000 >/dev/null"
for f in 714 816 918 1020 1122 1173; do
  for r in 1 2 3; do
    setclk "$f"; sleep 0.4
    local_tag=q3dep_g${f}_r${r}
    taskset -c 0 "$PY" campaign/ina_log.py results/qwen3b_deploy/${local_tag}.power.csv 120 &
    LP=$!
    "$BENCH" --model "$MODEL" --tokens 300 --warmup 40 --cpu 7 --prio 80 \
      --out results/qwen3b_deploy/${local_tag} >/dev/null 2>&1
    kill -TERM $LP 2>/dev/null; wait $LP 2>/dev/null
    echo "  dep g${f}_r${r}: $([ -s results/qwen3b_deploy/${local_tag}.json ] && echo OK || echo FAIL)"
  done
done
echo "QWEN3B_DEPLOY DONE"

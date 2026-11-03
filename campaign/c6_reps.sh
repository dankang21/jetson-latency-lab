#!/bin/bash
# C6: Part-A upper-pair {2133,3199} MHz session repeats — error bands for every
# quoted 2133->3199 penalty. 6 workloads x 2 EMC x 3 sessions, GPU pinned max.
# Board-agnostic: detects Nano (1020) vs NX (1173) by devfreq max. 5-min
# cooldown + 55C thermal gate between sessions.
# Env: PY (python3 with onnxruntime), ONNX_DIR (ONNX models) and GGUF_DIR (GGUF models) override the default paths below.
set -u
PY="${PY:-/home/dk/mobile/rt-infer-bench/.venv/bin/python3}"
GGUF_DIR="${GGUF_DIR:-/home/dk/mobile/models}"
LAB="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BENCH=$LAB/tools/slm_decode_bench
G=/sys/devices/platform/17000000.gpu/devfreq/17000000.gpu
MDIR="${ONNX_DIR:-/home/dk/mobile/rt-infer-bench/models}"
Q15=$GGUF_DIR/qwen2.5-1.5b-instruct-q4_k_m.gguf
GMAX=$(( $(cat $G/max_freq) / 1000000 ))
CPU=$([ "$GMAX" -ge 1100 ] && echo 7 || echo 5)
cd "$LAB" || exit 1; mkdir -p results/parta_reps
restore(){ sudo bash -c "$LAB/pilot_emc/emc_ctl.sh unlock >/dev/null 2>&1
  echo 306000000 > $G/min_freq
  echo 950000 > /proc/sys/kernel/sched_rt_runtime_us
  for c in /sys/devices/system/cpu/cpu*/cpufreq; do echo schedutil > \$c/scaling_governor; done
  chown -R dk:dk results/parta_reps 2>/dev/null"; }
trap restore EXIT
sudo bash -c "echo -1 > /proc/sys/kernel/sched_rt_runtime_us
for c in /sys/devices/system/cpu/cpu*/cpufreq; do echo performance > \$c/scaling_governor; done
echo ${GMAX}000000 > $G/min_freq; echo ${GMAX}000000 > $G/max_freq"

tgate(){ # wait until GPU <= 55C
  for i in $(seq 1 60); do
    t=$(cat /sys/devices/virtual/thermal/thermal_zone*/temp 2>/dev/null | sort -rn | head -1)
    [ "${t:-0}" -le 55000 ] && return 0; sleep 20
  done
}

run_cell(){ # run_cell <sess> <emc> <name> <model> <hz> <iters>
  local S=$1 EMC=$2 NAME=$3 M=$4 HZ=$5 IT=$6
  local OUT="results/parta_reps/s${S}_e${EMC}_${NAME}"
  "$PY" -m harness.infer_bench --backend onnxruntime --model "$M" \
    --hz "$HZ" --iters "$IT" --warmup 60 --cpu "$CPU" --label "c6_${S}_${EMC}_${NAME}" \
    --out "$OUT" >/dev/null 2>&1
  echo "  s${S} e${EMC} ${NAME}: $([ -s $OUT.json ] && echo OK || echo FAIL)"
}

for S in 1 2 3; do
  tgate
  for EMC in 3199 2133; do
    sudo bash -c "$LAB/pilot_emc/emc_ctl.sh lock ${EMC}000000 >/dev/null"; sleep 1
    run_cell $S $EMC mobilenet "$MDIR/mobilenetv2-12.onnx" 33 1000
    run_cell $S $EMC vit "$MDIR/vit_small_patch16_224.onnx" 10 800
    run_cell $S $EMC cproxyv2 "$LAB/pilot_emc/models/compute_proxy_fp16.onnx" 4 600
    run_cell $S $EMC cproxyv1 "$LAB/pilot_emc/models/compute_proxy_v1_gemm2048.onnx" 4 600
    run_cell $S $EMC proxy "$LAB/pilot_emc/models/decode_proxy_fp16.onnx" 4 600
    if [ -x "$BENCH" ] && [ -f "$Q15" ]; then
      "$BENCH" --model "$Q15" --tokens 150 --warmup 40 --cpu "$CPU" --prio 80 \
        --out "results/parta_reps/s${S}_e${EMC}_slm" >/dev/null 2>&1
      echo "  s${S} e${EMC} slm: $([ -s results/parta_reps/s${S}_e${EMC}_slm.json ] && echo OK || echo FAIL)"
    fi
  done
  [ "$S" -lt 3 ] && { echo "  cooldown 300s"; sleep 300; }
done
echo "C6_REPS DONE"

#!/bin/bash
# E1b / refit runner (rev3: per-model refit dir) (NX-first; board-agnostic). Same interleaved, thermally
# gated, EMC-2133-locked, user-mode protocol as govrep_rev.sh, generalized:
#   MODEL=vit|yolov8s|resnet50|mobilenet  (rates 20/10/20/33 Hz)
#   CLOCKS="612 714 ..."  PAIRS=10          -> results/govrep_rev/<board>_<model>/
#   MODE=refit            -> two-cell endpoints (306,<max>) at EMC 2133 and 3199,
#                            33 Hz MobileNetV2, 3 repeats -> results/refit33/<board>/
# Board detection uses available_frequencies (max_freq may be pinned).
set -u
PY=/home/dk/mobile/rt-infer-bench/.venv/bin/python3
LAB=/home/dk/mobile/jetson-latency-lab
M=/home/dk/mobile/rt-infer-bench/models
MODEL=${MODEL:-mobilenet}; MODE=${MODE:-pairs}; PAIRS=${PAIRS:-10}
case $MODEL in
  mobilenet) MP=$M/mobilenetv2-12.onnx;        HZ=33; WARM=120;;
  vit)       MP=$M/vit_small_patch16_224.onnx; HZ=20; WARM=100;;
  yolov8s)   MP=$M/yolov8s.onnx;               HZ=10; WARM=60;;
  resnet50)  MP=$M/resnet50-v1-12.onnx;        HZ=20; WARM=100;;
  *) echo "unknown MODEL $MODEL"; exit 1;;
esac
G=/sys/devices/platform/17000000.gpu/devfreq/17000000.gpu
GMAX=$(( $(tr ' ' '\n' < $G/available_frequencies | sort -n | tail -1) / 1000000 ))
if [ "$GMAX" -ge 1100 ]; then BOARD=nx; CPU=7; INA=40; else BOARD=nano; CPU=5; INA=30; fi
cd "$LAB" || exit 1
restore(){ sudo -n bash -c "$LAB/pilot_emc/emc_ctl.sh unlock >/dev/null 2>&1
  echo 306000000 > $G/min_freq; echo ${GMAX}000000 > $G/max_freq
  echo 950000 > /proc/sys/kernel/sched_rt_runtime_us
  for c in /sys/devices/system/cpu/cpu*/cpufreq; do echo schedutil > \$c/scaling_governor; done"; }
trap restore EXIT
setclk(){ sudo -n bash -c "echo ${1}000000 > $G/min_freq; echo ${1}000000 > $G/max_freq"; }
tmax(){ cat /sys/devices/virtual/thermal/thermal_zone*/temp 2>/dev/null | sort -rn | head -1; }
tgate(){ for i in $(seq 1 60); do t=$(tmax); [ "${t:-0}" -le 55000 ] && return 0; sleep 20; done; }
lock_emc(){ sudo -n bash -c "$LAB/pilot_emc/emc_ctl.sh lock ${1}000000" | tee -a "$OUT/emc_lock.txt"
  grep -q "actual=${1}000000" "$OUT/emc_lock.txt" || { echo "EMC lock $1 failed"; exit 1; }; }
cell(){ # cell <f> <tag> : one 1000-cycle run at GPU f with INA log
  setclk "$1"; sleep 0.4
  taskset -c 0 "$PY" campaign/ina_log.py "$2.power.csv" "$INA" & LP=$!
  "$PY" -m harness.infer_bench --backend onnxruntime --model "$MP" --hz "$HZ" --iters 1000 \
    --warmup "$WARM" --cpu "$CPU" --label "$(basename "$2")" --out "$2" >/dev/null 2>&1
  kill -TERM $LP 2>/dev/null; wait $LP 2>/dev/null
  echo "  $(basename "$2"): $([ -s $2.json ] && echo OK || echo FAIL) t=$(tmax)" | tee -a "$OUT/session.log"; }
sudo -n bash -c "echo -1 > /proc/sys/kernel/sched_rt_runtime_us
for c in /sys/devices/system/cpu/cpu*/cpufreq; do echo performance > \$c/scaling_governor; done"
if [ "$MODE" = refit ]; then
  OUT=results/refit33/${BOARD}_${MODEL}; mkdir -p "$OUT"
  echo "refit board=$BOARD host=$(hostname) commit=$(git rev-parse --short HEAD) hz=$HZ model=$MODEL start=$(date -Is)" >> "$OUT/session.log"
  for emc in 2133 3199; do
    lock_emc $emc
    for rep in 1 2 3; do tgate
      for f in 306 $GMAX; do TAG="$OUT/emc${emc}_g${f}_r${rep}"; [ -s "$TAG.json" ] && continue; cell "$f" "$TAG"; done
    done
  done
  echo "REFIT DONE $(date -Is)" | tee -a "$OUT/session.log"; exit 0
fi
CLOCKS=${CLOCKS:?set CLOCKS}
OUT=results/govrep_rev/${BOARD}_${MODEL}; mkdir -p "$OUT"
lock_emc 2133
echo "board=$BOARD host=$(hostname) commit=$(git rev-parse --short HEAD) model=$MODEL pairs=$PAIRS clocks=$CLOCKS hz=$HZ start=$(date -Is)" >> "$OUT/session.log"
for p in $(seq 1 "$PAIRS"); do tgate
  for f in $CLOCKS; do TAG="$OUT/g${f}_p${p}"; [ -s "$TAG.json" ] && { echo "  skip $TAG"; continue; }; cell "$f" "$TAG"; done
done
echo "GOVREP_REV DONE $(date -Is)" | tee -a "$OUT/session.log"

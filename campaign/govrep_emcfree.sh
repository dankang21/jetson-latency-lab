#!/bin/bash
# E5-B-live runner (NX): interleaved re-run of the EMC-free picks of the Han joint-policy replay
# (docs/protocols/E1_E2_protocol.md §E5-B-live, pre-stated 2026-10-28 before the run).
# Same thermally gated, user-mode, INA-logged protocol as govrep_rev2.sh (MobileNetV2, 33 Hz,
# 1000 cycles, warm-up 120), but the EMC is re-locked with BPMP readback before EVERY cell and
# the readback line is logged per cell.  Cells are emc:gpu pairs run round-robin, ROUNDS times.
#   CELLS="2133:1173 3199:1173 2133:1122 3199:1122 2133:1020 3199:1020 3199:918" ROUNDS=10
#   -> results/e5b_live/<board>/e<emc>_g<gpu>_r<round>.{json,csv,power.csv}, emc_lock.txt, session.log
# Env: PY (python3 with onnxruntime) and ONNX_DIR (ONNX models) override the default paths below.
set -u
PY="${PY:-/home/dk/mobile/rt-infer-bench/.venv/bin/python3}"
ONNX_DIR="${ONNX_DIR:-/home/dk/mobile/rt-infer-bench/models}"
LAB="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MP=$ONNX_DIR/mobilenetv2-12.onnx; HZ=33; WARM=120
CELLS=${CELLS:-"2133:1173 3199:1173 2133:1122 3199:1122 2133:1020 3199:1020 3199:918"}
ROUNDS=${ROUNDS:-10}
G=/sys/devices/platform/17000000.gpu/devfreq/17000000.gpu
GMAX=$(( $(tr ' ' '\n' < $G/available_frequencies | sort -n | tail -1) / 1000000 ))
if [ "$GMAX" -ge 1100 ]; then BOARD=nx; CPU=7; INA=40; else BOARD=nano; CPU=5; INA=30; fi
cd "$LAB" || exit 1
OUT=results/e5b_live/$BOARD; mkdir -p "$OUT"
restore(){ sudo -n bash -c "$LAB/pilot_emc/emc_ctl.sh unlock >/dev/null 2>&1
  echo 306000000 > $G/min_freq; echo ${GMAX}000000 > $G/max_freq
  echo 950000 > /proc/sys/kernel/sched_rt_runtime_us
  for c in /sys/devices/system/cpu/cpu*/cpufreq; do echo schedutil > \$c/scaling_governor; done"; }
trap restore EXIT
setclk(){ sudo -n bash -c "echo ${1}000000 > $G/min_freq; echo ${1}000000 > $G/max_freq"; }
tmax(){ cat /sys/devices/virtual/thermal/thermal_zone*/temp 2>/dev/null | sort -rn | head -1; }
tgate(){ for i in $(seq 1 60); do t=$(tmax); [ "${t:-0}" -le 55000 ] && return 0; sleep 20; done; }
lock_emc(){ # lock <emc_mhz> <tag>: BPMP lock + readback, one logged line per cell; abort on mismatch
  local line; line=$(sudo -n bash -c "$LAB/pilot_emc/emc_ctl.sh lock ${1}000000")
  echo "$2 $line" >> "$OUT/emc_lock.txt"
  echo "$line" | grep -q "actual=${1}000000" || { echo "EMC lock $1 failed for $2: $line" | tee -a "$OUT/session.log"; exit 1; }; }
cell(){ # cell <gpu_mhz> <tag>: one 1000-cycle run at GPU f with INA log
  setclk "$1"; sleep 0.4
  taskset -c 0 "$PY" campaign/ina_log.py "$2.power.csv" "$INA" & LP=$!
  "$PY" -m harness.infer_bench --backend onnxruntime --model "$MP" --hz "$HZ" --iters 1000 \
    --warmup "$WARM" --cpu "$CPU" --label "$(basename "$2")" --out "$2" >/dev/null 2>&1
  kill -TERM $LP 2>/dev/null; wait $LP 2>/dev/null
  echo "  $(basename "$2"): $([ -s $2.json ] && echo OK || echo FAIL) t=$(tmax)" | tee -a "$OUT/session.log"; }
sudo -n bash -c "echo -1 > /proc/sys/kernel/sched_rt_runtime_us
for c in /sys/devices/system/cpu/cpu*/cpufreq; do echo performance > \$c/scaling_governor; done"
echo "e5b_live board=$BOARD host=$(hostname) commit=$(git rev-parse --short HEAD) cells=\"$CELLS\" rounds=$ROUNDS hz=$HZ start=$(date -Is)" >> "$OUT/session.log"
for r in $(seq 1 "$ROUNDS"); do tgate
  for c in $CELLS; do emc=${c%%:*}; f=${c##*:}; TAG="$OUT/e${emc}_g${f}_r${r}"
    [ -s "$TAG.json" ] && { echo "  skip $TAG"; continue; }
    lock_emc "$emc" "$(basename "$TAG")"; cell "$f" "$TAG"
  done
done
echo "E5B_LIVE DONE $(date -Is)" | tee -a "$OUT/session.log"

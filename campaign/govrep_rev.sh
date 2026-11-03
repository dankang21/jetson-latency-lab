#!/bin/bash
# E1: balanced blind/aware replication for Table II,
# interleaved within one thermally gated session so both arms share the
# same session conditions. Every clock that the two-cell (deployed) rule
# can pick across the pre-stated deadline grid is run once per pair index.
#   NX  : clocks 816 918 1020 1122 1173, hz 33, warmup 120, victim cpu 7
#   Nano: clocks 408 510 612,            hz 50, warmup 100, victim cpu 5
# Protocol is identical to the committed Table II runs (user-mode periodic
# loop, EMC locked 2133 via BPMP, GPU min=max pinned, CPU performance).
# Writes results/govrep_rev/$BOARD/g${f}_p${p}.{json,csv,power.csv};
# skips cells whose JSON exists, so it can be resumed.
# Env: PY (python3 with onnxruntime) and ONNX_DIR (ONNX models) override the default paths below.
# Usage: PAIRS=10 ./campaign/govrep_rev.sh
set -u
PY="${PY:-/home/dk/mobile/rt-infer-bench/.venv/bin/python3}"
ONNX_DIR="${ONNX_DIR:-/home/dk/mobile/rt-infer-bench/models}"
LAB="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL=${MODEL:-mobilenet}
CLK_OVERRIDE="${CLOCKS:-}"
case $MODEL in
  mobilenet) MN=$ONNX_DIR/mobilenetv2-12.onnx; MHZ=;;
  vit)       MN=$ONNX_DIR/vit_small_patch16_224.onnx; MHZ=20;;
  *) echo "unknown MODEL"; exit 1;;
esac
G=/sys/devices/platform/17000000.gpu/devfreq/17000000.gpu
GMAX=$(( $(cat $G/max_freq) / 1000000 ))
if [ "$GMAX" -ge 1100 ]; then
  BOARD=nx;   CPU=7; HZ=33; WARM=120; CLOCKS="816 918 1020 1122 1173"; INA=40
else
  BOARD=nano; CPU=5; HZ=50; WARM=100; CLOCKS="408 510 612";           INA=30
fi
PAIRS=${PAIRS:-10}
[ -n "$CLK_OVERRIDE" ] && CLOCKS="$CLK_OVERRIDE"; HZ=${MHZ:-$HZ}; HZ=${HZ_OVERRIDE:-$HZ}
OUT=results/govrep_rev/$BOARD$( [ "$MODEL" != mobilenet ] && echo _$MODEL )
cd "$LAB" || exit 1; mkdir -p "$OUT"
restore(){ sudo -n bash -c "$LAB/pilot_emc/emc_ctl.sh unlock >/dev/null 2>&1
  echo 306000000 > $G/min_freq; echo ${GMAX}000000 > $G/max_freq
  echo 950000 > /proc/sys/kernel/sched_rt_runtime_us
  for c in /sys/devices/system/cpu/cpu*/cpufreq; do echo schedutil > \$c/scaling_governor; done"; }
trap restore EXIT
setclk(){ sudo -n bash -c "echo ${1}000000 > $G/min_freq; echo ${1}000000 > $G/max_freq"; }
tmax(){ cat /sys/devices/virtual/thermal/thermal_zone*/temp 2>/dev/null | sort -rn | head -1; }
tgate(){ for i in $(seq 1 60); do t=$(tmax); [ "${t:-0}" -le 55000 ] && return 0; sleep 20; done; }
sudo -n bash -c "echo -1 > /proc/sys/kernel/sched_rt_runtime_us
for c in /sys/devices/system/cpu/cpu*/cpufreq; do echo performance > \$c/scaling_governor; done"
sudo -n bash -c "$LAB/pilot_emc/emc_ctl.sh lock 2133000000" | tee "$OUT/emc_lock.txt"
grep -q "actual=2133000000" "$OUT/emc_lock.txt" || { echo "EMC lock failed"; exit 1; }
echo "board=$BOARD host=$(hostname) commit=$(git rev-parse --short HEAD) pairs=$PAIRS clocks=$CLOCKS hz=$HZ start=$(date -Is)" >> "$OUT/session.log"
for p in $(seq 1 "$PAIRS"); do
  tgate
  for f in $CLOCKS; do
    TAG="$OUT/g${f}_p${p}"
    [ -s "$TAG.json" ] && { echo "  skip $TAG"; continue; }
    setclk "$f"; sleep 0.4
    taskset -c 0 "$PY" campaign/ina_log.py "$TAG.power.csv" "$INA" & LP=$!
    "$PY" -m harness.infer_bench --backend onnxruntime --model "$MN" \
      --hz "$HZ" --iters 1000 --warmup "$WARM" --cpu "$CPU" --label rev_${BOARD}_${MODEL}_g${f}_p${p} \
      --out "$TAG" >/dev/null 2>&1
    kill -TERM $LP 2>/dev/null; wait $LP 2>/dev/null
    echo "  p${p} g${f}: $([ -s $TAG.json ] && echo OK || echo FAIL) t=$(tmax)" | tee -a "$OUT/session.log"
  done
done
echo "GOVREP_REV DONE $(date -Is)" | tee -a "$OUT/session.log"

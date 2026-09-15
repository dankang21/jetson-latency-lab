#!/bin/bash
# E3 reference sweep (protocol E3_protocol.md §5 "utilization cost"): for ONE
# condition, one 1000-cycle contract at EVERY f in G with the same co-runner
# running, so f_oracle = lowest f whose measured miss <= tau (2 %) is known.
# Same pinning, EMC, rates and contract arguments as admit_loop.sh; one
# co-runner launch for the whole sweep (§5: "same co-runner"), thermal gate
# once before the launch. Ascending f (c3_tenancy.sh / joint_gov.sh order).
# Output: results/e3_<board>/<cond>/ref_g<f>.{json,csv}[,power.csv], session.log
# Resumable: existing ref_g<f>.json cells are skipped (a resumed sweep gets
# a fresh co-runner launch for the remaining cells; logged).
# Usage: ./campaign/admit_refsweep.sh --primary {mobilenet,vit} --co {yolov8s,resnet50,vit}
#          --co-hz N --D ms --cond ID [--ina] [--force] [--dry-run]
set -u
PY=/home/dk/mobile/rt-infer-bench/.venv/bin/python3
LAB=/home/dk/mobile/jetson-latency-lab
MDIR=/home/dk/mobile/rt-infer-bench/models
G=/sys/devices/platform/17000000.gpu/devfreq/17000000.gpu
EMC_RATE=/sys/kernel/debug/bpmp/debug/clk/emc/rate
INA_S=120; TAU=0.02

PRIMARY=""; CO=""; CO_HZ=""; D_MS=""; COND=""; INA=false; FORCE=false; DRY=false
while [ $# -gt 0 ]; do case "$1" in
  --primary) PRIMARY=$2; shift 2;;  --co) CO=$2; shift 2;;  --co-hz) CO_HZ=$2; shift 2;;
  --D) D_MS=$2; shift 2;;  --cond) COND=$2; shift 2;;
  --ina) INA=true; shift;;  --force) FORCE=true; shift;;  --dry-run) DRY=true; shift;;
  *) echo "unknown arg $1"; exit 2;; esac; done
[ -n "$PRIMARY" ] && [ -n "$CO" ] && [ -n "$CO_HZ" ] && [ -n "$D_MS" ] && [ -n "$COND" ] || \
  { sed -n '/^# Usage/,/^set -u/p' "$0" | head -2; exit 2; }

GMAX=$(( $(tr ' ' '\n' < $G/available_frequencies | sort -n | tail -1) / 1000000 ))
if [ "$GMAX" -ge 1100 ]; then
  BOARD=nx;   VCPU=7; CCPU=5; FPROBE=1173; GRID="510 612 714 816 918 1020 1122 1173"
else
  BOARD=nano; VCPU=5; CCPU=4; FPROBE=1020; GRID="306 408 510 612 714 816 918 1020"
fi
[ "$FPROBE" -eq "$GMAX" ] || { echo "devfreq max ${GMAX} != f_probe ${FPROBE}"; exit 1; }
case $PRIMARY in
  mobilenet) PRI_MODEL=$MDIR/mobilenetv2-12.onnx;        PRI_HZ=33;;
  vit)       PRI_MODEL=$MDIR/vit_small_patch16_224.onnx; PRI_HZ=20;;
  *) echo "--primary must be mobilenet|vit"; exit 2;; esac
case $CO in
  yolov8s)  CO_MODEL=$MDIR/yolov8s.onnx;;
  resnet50) CO_MODEL=$MDIR/resnet50-v1-12.onnx;;
  vit)      CO_MODEL=$MDIR/vit_small_patch16_224.onnx;;
  *) echo "--co must be yolov8s|resnet50|vit"; exit 2;; esac
OUT=results/e3_$BOARD/$COND
cd "$LAB" || exit 1; $DRY || mkdir -p "$OUT"
for f in "$PRI_MODEL" "$CO_MODEL"; do [ -s "$f" ] || { echo "missing $f"; exit 1; }; done
SIG="board=$BOARD primary=$PRIMARY pri_hz=$PRI_HZ co=$CO co_hz=$CO_HZ D_ms=$D_MS fprobe=$FPROBE grid=${GRID// /,}"
if [ -s "$OUT/condition.txt" ] && [ "$(cut -d' ' -f1-7 "$OUT/condition.txt")" != "$(echo "$SIG" | cut -d' ' -f1-7)" ]; then
  echo "condition mismatch in $OUT:"; cat "$OUT/condition.txt"; echo "vs: $SIG"; exit 1; fi
if ! $DRY && ! $FORCE && pgrep -f harness.infer_bench >/dev/null; then
  echo "another infer_bench is running (E1?); refuse to start (use --force)"; exit 1; fi

exec 3>&1
run(){ if $DRY; then echo "+ $*" >&3; return 0; fi; "$@"; }
nap(){ $DRY || sleep "$1"; }
log(){ if $DRY; then echo "$*"; else echo "$*" | tee -a "$OUT/session.log"; fi; }
CO_PID=""; LP=""
restore(){ [ -n "$CO_PID" ] && kill "$CO_PID" 2>/dev/null; pkill -f "[c]o_${CO}" 2>/dev/null
  [ -n "$LP" ] && kill -TERM "$LP" 2>/dev/null
  $DRY && return 0
  sudo -n bash -c "$LAB/pilot_emc/emc_ctl.sh unlock >/dev/null 2>&1
  echo 306000000 > $G/min_freq; echo ${GMAX}000000 > $G/max_freq
  echo 950000 > /proc/sys/kernel/sched_rt_runtime_us
  for c in /sys/devices/system/cpu/cpu*/cpufreq; do echo schedutil > \$c/scaling_governor; done"; }
trap restore EXIT
setclk(){ run sudo -n bash -c "echo ${1}000000 > $G/min_freq; echo ${1}000000 > $G/max_freq"; }
tmax(){ cat /sys/devices/virtual/thermal/thermal_zone*/temp 2>/dev/null | sort -rn | head -1; }
tgate(){ $DRY && return 0; TG_WAIT=0
  for i in $(seq 1 60); do t=$(tmax); [ "${t:-0}" -le 55000 ] && return 0; sleep 20; TG_WAIT=$((TG_WAIT+20)); done
  log "  WARN thermal gate timed out after ${TG_WAIT}s (t=$(tmax)); proceeding"; return 1; }
emc_check(){ $DRY && return 0; local r; r=$(sudo -n cat $EMC_RATE 2>/dev/null)
  [ "$r" = 2133000000 ] && return 0
  log "  WARN EMC rate=$r != 2133000000 -> re-lock"
  sudo -n bash -c "$LAB/pilot_emc/emc_ctl.sh lock 2133000000" | tee -a "$OUT/session.log"; }
co_start(){ RF=/tmp/co_e3_${COND}_${k:-ref}.ready; rm -f "$RF"; local CL="$OUT/co_${k:-ref}.log"; $DRY && CL=/dev/null
  run "$PY" -m harness.infer_bench --backend onnxruntime --model "$CO_MODEL" \
    --hz "$CO_HZ" --iters 100000 --warmup 30 --cpu "$CCPU" --label co_${CO} \
    --out /tmp/co_e3_${COND}_ref --ready-file "$RF" >"$CL" 2>&1 & CO_PID=$!
  $DRY && return 0; local w=0; until [ -f "$RF" ] || [ $w -ge 120 ]; do sleep 1; w=$((w+1)); done
  if [ -f "$RF" ]; then log "  co-runner ready after ${w}s"; else log "  WARN co-runner NOT ready after ${w}s"; fi; nap 5; }
co_stop(){ [ -n "$CO_PID" ] && kill "$CO_PID" 2>/dev/null; pkill -f "[c]o_${CO}" 2>/dev/null; wait "$CO_PID" 2>/dev/null; CO_PID=""; nap 2; }
score(){ $DRY && { echo "-"; return; }; [ -s "$1" ] && "$PY" campaign/admit_pick.py --score "$1" --D-ms "$D_MS" --brief || echo FAIL; }

run sudo -n bash -c "echo -1 > /proc/sys/kernel/sched_rt_runtime_us
for c in /sys/devices/system/cpu/cpu*/cpufreq; do echo performance > \$c/scaling_governor; done"
if $DRY; then echo "+ sudo -n $LAB/pilot_emc/emc_ctl.sh lock 2133000000"; else
  sudo -n bash -c "$LAB/pilot_emc/emc_ctl.sh lock 2133000000" | tee "$OUT/emc_lock_ref.txt"
  grep -q "actual=2133000000" "$OUT/emc_lock_ref.txt" || { echo "EMC lock failed"; exit 1; }; fi
TODO=""; for f in $GRID; do [ -s "$OUT/ref_g${f}.json" ] && echo "  skip ref g${f}" || TODO="$TODO $f"; done
log "refsweep cond=$COND host=$(hostname) commit=$(git rev-parse --short HEAD 2>/dev/null) $SIG cells=${TODO# } ina=$INA dry=$DRY start=$(date -Is)"
if [ -n "$TODO" ]; then
  tgate; T_GATE=$(tmax); emc_check
  setclk "$FPROBE"; nap 0.4      # co-runner warms at f_probe, as in admit_loop.sh
  co_start
  for f in $TODO; do
    TAG="$OUT/ref_g${f}"
    setclk "$f"; nap 0.4
    if $INA; then run taskset -c 0 "$PY" campaign/ina_log.py "$TAG.power.csv" "$INA_S" & LP=$!; fi
    run "$PY" -m harness.infer_bench --backend onnxruntime --model "$PRI_MODEL" \
      --hz "$PRI_HZ" --iters 1000 --warmup 60 --cpu "$VCPU" --deadline-ms "$D_MS" \
      --label e3_${BOARD}_${COND}_ref_g${f} --out "$TAG" >/dev/null 2>&1
    [ -n "$LP" ] && { kill -TERM $LP 2>/dev/null; wait $LP 2>/dev/null; LP=""; }
    log "  ref g${f}: m=$(score "$TAG.csv") tgate=${T_GATE} t=$(tmax) $(date -Is)"
  done
  co_stop
fi
# f_oracle = lowest f in G with measured miss <= tau (protocol §5)
if ! $DRY; then
  ORACLE=INFEASIBLE
  for f in $GRID; do m=$(score "$OUT/ref_g${f}.csv"); [ "$m" != FAIL ] && awk -v m="$m" -v t="$TAU" 'BEGIN{exit !(m<=t)}' && { ORACLE=$f; break; }; done
  log "REFSWEEP DONE cond=$COND f_oracle=$ORACLE tau=$TAU $(date -Is)"
else
  echo "REFSWEEP DONE (dry-run) cond=$COND"
fi

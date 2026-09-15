#!/bin/bash
# E3: pre-registered prospective validation of the
# dispersion-banded admission guard -- ONE condition, K independent trials.
# Protocol: docs/protocols/E3_protocol.md §1 (frozen constants) and
# §4 (per-trial procedure); mapping in docs/protocols/E3_runner_notes.md.
# Per trial (§4 steps 1-7):
#   1 thermal gate <=55C, EMC 2133 (BPMP lock, verified), GPU pinned f_probe
#   2 co-runner launched on its own core at --co-hz, sleep 20
#   3 200-cycle probe at f_probe (warmup 60) with the primary
#   4 admit_pick.py -> f_pt (h=0) and f_b (h=400us), per-f slack -> t<k>.pick.json
#   5 banded contract: 1000 cycles at f_b (skipped if INFEASIBLE)
#   6 point contract: 1000 cycles at f_pt, only if f_pt != f_b
#   7 kill co-runner; session.log line; t<k>.done marker
# Board autodetected: NX  f_probe 1173, victim cpu 7, co cpu 5, G 510..1173
#                     Nano f_probe 1020, victim cpu 5, co cpu 4, G 306..1020
# Primary rates: mobilenet 33 Hz, vit 20 Hz (protocol §3). User-mode
# infer_bench (same rt_warnings as every committed row), clocks via sudo -n.
# Output: results/e3_<board>/<cond>/t<k>.{probe,contract_b,contract_pt}.{json,csv},
#         t<k>.pick.json, t<k>.done, condition.txt, emc_lock.txt, session.log
# Resumable: trials with contract_b.json or a .done marker are skipped; an
# interrupted trial is re-run from scratch (fresh probe), nothing reused.
# Usage: ./campaign/admit_loop.sh --primary {mobilenet,vit} --co {yolov8s,resnet50,vit}
#          --co-hz N --D ms --trials K --cond ID [--t0 1] [--ina] [--force] [--dry-run]
set -u
PY=/home/dk/mobile/rt-infer-bench/.venv/bin/python3
LAB=/home/dk/mobile/jetson-latency-lab
MDIR=/home/dk/mobile/rt-infer-bench/models
G=/sys/devices/platform/17000000.gpu/devfreq/17000000.gpu
EMC_RATE=/sys/kernel/debug/bpmp/debug/clk/emc/rate
H_US=400; INA_S=120

PRIMARY=""; CO=""; CO_HZ=""; D_MS=""; TRIALS=""; COND=""; T0=1; INA=false; FORCE=false; DRY=false
while [ $# -gt 0 ]; do case "$1" in
  --primary) PRIMARY=$2; shift 2;;  --co) CO=$2; shift 2;;  --co-hz) CO_HZ=$2; shift 2;;
  --D) D_MS=$2; shift 2;;  --trials) TRIALS=$2; shift 2;;  --cond) COND=$2; shift 2;;
  --t0) T0=$2; shift 2;;  --ina) INA=true; shift;;  --force) FORCE=true; shift;;  --dry-run) DRY=true; shift;;
  *) echo "unknown arg $1"; exit 2;; esac; done
[ -n "$PRIMARY" ] && [ -n "$CO" ] && [ -n "$CO_HZ" ] && [ -n "$D_MS" ] && [ -n "$TRIALS" ] && [ -n "$COND" ] || \
  { sed -n '/^# Usage/,/^set -u/p' "$0" | head -2; exit 2; }

# board autodetect from the devfreq table (not max_freq: that reads the
# currently pinned clock if another campaign is running or crashed pinned)
GMAX=$(( $(tr ' ' '\n' < $G/available_frequencies | sort -n | tail -1) / 1000000 ))
if [ "$GMAX" -ge 1100 ]; then
  BOARD=nx;   VCPU=7; CCPU=5; FPROBE=1173; GRID="510 612 714 816 918 1020 1122 1173"
  CELLS="results/nx_validation/nx_est_emc2133_gpu306_${PRIMARY}.json results/nx_validation/nx_est_emc2133_gpu1173_${PRIMARY}.json"
else
  BOARD=nano; VCPU=5; CCPU=4; FPROBE=1020; GRID="306 408 510 612 714 816 918 1020"
  CELLS="results/nano_grid/${PRIMARY}/est_emc2133_gpu306.json results/nano_grid/${PRIMARY}/est_emc2133_gpu1020.json"
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
for f in "$PRI_MODEL" "$CO_MODEL" $CELLS; do [ -s "$f" ] || { echo "missing $f"; exit 1; }; done
SIG="board=$BOARD primary=$PRIMARY pri_hz=$PRI_HZ co=$CO co_hz=$CO_HZ D_ms=$D_MS fprobe=$FPROBE grid=${GRID// /,} h_us=$H_US R=1.035 cells=${CELLS// /,}"
if [ -s "$OUT/condition.txt" ] && [ "$(cat "$OUT/condition.txt")" != "$SIG" ]; then
  echo "condition mismatch in $OUT:"; cat "$OUT/condition.txt"; echo "vs: $SIG"; exit 1; fi
$DRY || echo "$SIG" > "$OUT/condition.txt"
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
co_start(){ RF=/tmp/co_e3_${COND}_${k:-ref}.ready; rm -f "$RF"; local CL="$OUT/co_${k:-ref}.log"; $DRY && CL=/dev/null # fresh co-runner process on its own core, trace discarded (fresh_loop.sh:26-28)
  run "$PY" -m harness.infer_bench --backend onnxruntime --model "$CO_MODEL" \
    --hz "$CO_HZ" --iters 100000 --warmup 30 --cpu "$CCPU" --label co_${CO} \
    --out /tmp/co_e3_${COND}_t${k} --ready-file "$RF" >"$CL" 2>&1 & CO_PID=$!
  $DRY && return 0; local w=0; until [ -f "$RF" ] || [ $w -ge 120 ]; do sleep 1; w=$((w+1)); done
  if [ -f "$RF" ]; then log "  co-runner ready after ${w}s"; else log "  WARN co-runner NOT ready after ${w}s"; fi; nap 5; }
co_stop(){ [ -n "$CO_PID" ] && kill "$CO_PID" 2>/dev/null; pkill -f "[c]o_${CO}" 2>/dev/null; wait "$CO_PID" 2>/dev/null; CO_PID=""; nap 2; }
contract(){ # contract <f> <out_prefix> <kind>: 1000 periodic cycles at f under the co-runner
  setclk "$1"; nap 0.4
  if $INA; then run taskset -c 0 "$PY" campaign/ina_log.py "$2.power.csv" "$INA_S" & LP=$!; fi
  run "$PY" -m harness.infer_bench --backend onnxruntime --model "$PRI_MODEL" \
    --hz "$PRI_HZ" --iters 1000 --warmup 60 --cpu "$VCPU" --deadline-ms "$D_MS" \
    --label e3_${BOARD}_${COND}_${3}_t${k}_g${1} --out "$2" >/dev/null 2>&1
  [ -n "$LP" ] && { kill -TERM $LP 2>/dev/null; wait $LP 2>/dev/null; LP=""; }; }
score(){ $DRY && { echo "-"; return; }; [ -s "$1" ] && "$PY" campaign/admit_pick.py --score "$1" --D-ms "$D_MS" --brief || echo FAIL; }

run sudo -n bash -c "echo -1 > /proc/sys/kernel/sched_rt_runtime_us
for c in /sys/devices/system/cpu/cpu*/cpufreq; do echo performance > \$c/scaling_governor; done"
if $DRY; then echo "+ sudo -n $LAB/pilot_emc/emc_ctl.sh lock 2133000000"; else
  sudo -n bash -c "$LAB/pilot_emc/emc_ctl.sh lock 2133000000" | tee "$OUT/emc_lock.txt"
  grep -q "actual=2133000000" "$OUT/emc_lock.txt" || { echo "EMC lock failed"; exit 1; }; fi
log "session cond=$COND host=$(hostname) commit=$(git rev-parse --short HEAD 2>/dev/null) $SIG trials=$T0..$((T0+TRIALS-1)) ina=$INA dry=$DRY start=$(date -Is)"

for k in $(seq "$T0" $((T0+TRIALS-1))); do
  TAG="$OUT/t${k}"
  { [ -s "$TAG.contract_b.json" ] || [ -s "$TAG.done" ]; } && { echo "  skip t${k}"; continue; }
  $DRY || rm -f "$TAG".probe.* "$TAG".contract_b.* "$TAG".contract_pt.* "$TAG".pick.json   # §4: nothing reused
  # 1. thermal gate, EMC, f_probe
  tgate; T_GATE=$(tmax); emc_check
  setclk "$FPROBE"; nap 0.4
  # 2. co-runner, 20 s ahead
  co_start
  # 3. probe
  run "$PY" -m harness.infer_bench --backend onnxruntime --model "$PRI_MODEL" \
    --hz "$PRI_HZ" --iters 200 --warmup 60 --cpu "$VCPU" --label e3_${BOARD}_${COND}_probe_t${k} \
    --out "$TAG.probe" >/dev/null 2>&1
  if ! $DRY && [ ! -s "$TAG.probe.csv" ]; then co_stop; log "  t${k}: PROBE FAIL t=$(tmax) $(date -Is)"; continue; fi
  # 4. picks (both recorded; per-f slack in the JSON)
  if $DRY; then echo "+ $PY campaign/admit_pick.py $TAG.probe.csv --D-ms $D_MS --board $BOARD --primary $PRIMARY --cells $CELLS --grid ${GRID// /,} --fprobe $FPROBE --h-us $H_US --out $TAG.pick.json --brief"
    MG='<Mg>'; FPT='<f_pt>'; FB='<f_b>'
  else
    read MG FPT FB <<< "$("$PY" campaign/admit_pick.py "$TAG.probe.csv" --D-ms "$D_MS" --board "$BOARD" \
      --primary "$PRIMARY" --cells $CELLS --grid "${GRID// /,}" --fprobe "$FPROBE" --h-us "$H_US" \
      --out "$TAG.pick.json" --brief)"
    [ -n "${FB:-}" ] || { co_stop; log "  t${k}: PICK FAIL $(date -Is)"; continue; }
  fi
  # 5. banded contract at f_b (declined trial: no contract, recorded as INFEASIBLE)
  MB=-; MPT=-
  if [ "$FB" != 0 ]; then contract "$FB" "$TAG.contract_b" contract_b; MB=$(score "$TAG.contract_b.csv"); fi
  # 6. point contract at f_pt only if the band changed the decision
  if [ "$FPT" != "$FB" ] && [ "$FPT" != 0 ]; then contract "$FPT" "$TAG.contract_pt" contract_pt; MPT=$(score "$TAG.contract_pt.csv"); fi
  # 7. stop co-runner, log
  co_stop
  LINE="  t${k}: Mg=${MG}us f_pt=${FPT} f_b=${FB} m_b=${MB} m_pt=${MPT} tgate=${T_GATE} tend=$(tmax) $(date -Is)"
  $DRY || echo "$LINE" > "$TAG.done"
  log "$LINE"
done
log "ADMIT_LOOP DONE cond=$COND $(date -Is)"

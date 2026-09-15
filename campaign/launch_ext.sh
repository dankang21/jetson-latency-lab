#!/bin/bash
# E4: launch-level uncertainty (how often a fixed admitted design point
# violates its budget across fresh launches): extend the fresh-launch
# series at ONE fixed design point (f, D, primary, co-runner, co-rate) by K
# more launches. No probe and no pick: a "launch" = a fresh co-runner process
# (fresh CUDA context, fresh phase) against a fresh victim process at a clock
# fixed in advance (launch1020.sh lines 23-33).
# Per launch k:
#   1 thermal gate <=55C, EMC 2133 verified (re-locked if drifted), GPU pinned f
#   2 fresh co-runner on its own core at --co-hz (--iters 100000 --warmup 30), sleep 20
#   3 fresh victim: 1000 periodic cycles, --warmup 60, --deadline-ms D  [+ INA log]
#   4 co-runner killed, sleep 2; miss re-scored from the CSV (admit_pick.py
#     --score, response_us > D); session.log line
# Board autodetected from devfreq available_frequencies:
#   NX   victim cpu 7, co cpu 5, steps 306..1173     Nano victim cpu 5, co cpu 4, steps 306..1020
# Primary rates: mobilenet 33 Hz, vit 20 Hz (E3_protocol.md section 3; 33 Hz = launch1020).
# DEFAULTS == the submitted design point == launch1020.sh verbatim:
#   --f 1020 --D 12 --primary mobilenet --co vit --co-hz 10  ->  tag mobilenet_vit10_g1020_D12
#   MobileNetV2 --hz 33 --iters 1000 --warmup 60 --cpu 7; ViT-Small co-runner
#   --hz 10 --iters 100000 --warmup 30 --cpu 5; EMC 2133 (BPMP lock); CPU
#   governors performance; sched_rt_runtime_us -1; user-mode infer_bench
#   (same rt_warnings as every committed row).
# The committed eight launches of that tag are results/launch1020/dep_t<k>.{json,csv},
# k = 1..8 (meta.host Pandora = NX, 2026-07-07, 53.6 s apart). They ARE t1..t8
# of the series, so on the NX the default --t0 for that tag is 9 and the new
# launches are t9..t(8+K); launch_ext_stats.py merges both locations. Writing
# t1..t8 of that tag on the NX is refused (would shadow the committed ones).
# Differences from launch1020.sh, none of which touch the measurement: thermal
# gate + EMC re-verify before every launch (launch1020 locked once, no gate);
# --deadline-ms D is passed so the JSON/CSV miss column is already at D
# (launch1020 scored D offline; harness/infer_bench.py lines 60-81: the
# deadline only changes the miss column, never the schedule); the GPU pin is
# re-asserted per launch (same value; a no-op); label lext_<board>_<tag>_t<k>.
# Output: results/launch_ext/<board>/<tag>/t<k>.{json,csv}[,power.csv],
#         condition.txt, emc_lock.txt, session.log
# Resumable: a launch whose t<k>.json exists is skipped; a partial launch
# (csv without json) is deleted and re-run from scratch.
# Usage: ./campaign/launch_ext.sh --launches K [--f MHz] [--D ms] [--primary {mobilenet,vit}]
#          [--co {vit,yolov8s,resnet50}] [--co-hz N] [--t0 START] [--ina] [--force] [--dry-run]
#        (24-launch submitted series on hebesu:  ./campaign/launch_ext.sh --launches 16)
set -u
PY=/home/dk/mobile/rt-infer-bench/.venv/bin/python3
LAB=/home/dk/mobile/jetson-latency-lab
MDIR=/home/dk/mobile/rt-infer-bench/models
G=/sys/devices/platform/17000000.gpu/devfreq/17000000.gpu
EMC_RATE=/sys/kernel/debug/bpmp/debug/clk/emc/rate
EMC_HZ=2133000000; INA_S=120
COMMITTED_TAG=mobilenet_vit10_g1020_D12; COMMITTED_DIR=results/launch1020; COMMITTED_N=8

F=1020; D_MS=12; PRIMARY=mobilenet; CO=vit; CO_HZ=10; K=""; T0=""; INA=false; FORCE=false; DRY=false
usage(){ sed -n '/^# Usage/,/^set -u/p' "$0" | head -3; }
while [ $# -gt 0 ]; do case "$1" in
  --f) F=$2; shift 2;;  --D) D_MS=$2; shift 2;;  --primary) PRIMARY=$2; shift 2;;
  --co) CO=$2; shift 2;;  --co-hz) CO_HZ=$2; shift 2;;  --launches) K=$2; shift 2;;
  --t0) T0=$2; shift 2;;  --ina) INA=true; shift;;  --force) FORCE=true; shift;;  --dry-run) DRY=true; shift;;
  -h|--help) usage; exit 0;;  *) echo "unknown arg $1"; usage; exit 2;; esac; done
[ -n "$K" ] && [ "$K" -ge 1 ] 2>/dev/null || { echo "--launches K (>=1) is required"; usage; exit 2; }
[ "$F" -ge 1 ] 2>/dev/null || { echo "--f must be an integer MHz"; exit 2; }
awk -v d="$D_MS" 'BEGIN{exit !(d+0>0)}' || { echo "--D must be > 0 (ms)"; exit 2; }
awk -v h="$CO_HZ" 'BEGIN{exit !(h+0>0)}' || { echo "--co-hz must be > 0"; exit 2; }
[ -z "$T0" ] || [ "$T0" -ge 1 ] 2>/dev/null || { echo "--t0 must be an integer >= 1"; exit 2; }

# board autodetect from the devfreq table (not max_freq: that reads the
# currently pinned clock if another campaign is running or crashed pinned)
STEPS=$(tr ' ' '\n' < $G/available_frequencies | sed '/^$/d' | sort -n | awk '{printf "%d ", $1/1000000}')
GMAX=${STEPS% }; GMAX=${GMAX##* }                    # last (highest) step, MHz
if [ "$GMAX" -ge 1100 ]; then BOARD=nx; VCPU=7; CCPU=5; else BOARD=nano; VCPU=5; CCPU=4; fi
case " $STEPS" in *" $F "*) ;; *) echo "--f $F is not a devfreq step on $BOARD (steps: $STEPS)"; exit 2;; esac
case $PRIMARY in
  mobilenet) PRI_MODEL=$MDIR/mobilenetv2-12.onnx;        PRI_HZ=33;;
  vit)       PRI_MODEL=$MDIR/vit_small_patch16_224.onnx; PRI_HZ=20;;
  *) echo "--primary must be mobilenet|vit"; exit 2;; esac
case $CO in
  yolov8s)  CO_MODEL=$MDIR/yolov8s.onnx;;
  resnet50) CO_MODEL=$MDIR/resnet50-v1-12.onnx;;
  vit)      CO_MODEL=$MDIR/vit_small_patch16_224.onnx;;
  *) echo "--co must be yolov8s|resnet50|vit"; exit 2;; esac
TAG=${PRIMARY}_${CO}${CO_HZ}_g${F}_D${D_MS}
OUT=results/launch_ext/$BOARD/$TAG
COMMITTED=false; [ "$BOARD" = nx ] && [ "$TAG" = "$COMMITTED_TAG" ] && COMMITTED=true
if [ -z "$T0" ]; then $COMMITTED && T0=$((COMMITTED_N+1)) || T0=1; fi
if $COMMITTED && [ "$T0" -le "$COMMITTED_N" ] && ! $FORCE; then
  echo "t1..t${COMMITTED_N} of $TAG on the NX are the committed $COMMITTED_DIR/dep_t1..${COMMITTED_N}; use --t0 $((COMMITTED_N+1)) (default) or --force"; exit 2; fi
T1=$((T0+K-1))

cd "$LAB" || exit 1; $DRY || mkdir -p "$OUT"
for f in "$PRI_MODEL" "$CO_MODEL"; do [ -s "$f" ] || { echo "missing $f"; $DRY && echo "  (dry-run: continuing)" || exit 1; }; done
SIG="board=$BOARD tag=$TAG primary=$PRIMARY pri_model=$(basename "$PRI_MODEL") pri_hz=$PRI_HZ iters=1000 warmup=60 vcpu=$VCPU co=$CO co_model=$(basename "$CO_MODEL") co_hz=$CO_HZ co_iters=100000 co_warmup=30 ccpu=$CCPU f=$F D_ms=$D_MS emc=$EMC_HZ"
if [ -s "$OUT/condition.txt" ] && [ "$(cat "$OUT/condition.txt")" != "$SIG" ]; then
  echo "condition mismatch in $OUT:"; cat "$OUT/condition.txt"; echo "vs: $SIG"; exit 1; fi
$DRY || echo "$SIG" > "$OUT/condition.txt"
if ! $DRY && ! $FORCE && pgrep -f harness.infer_bench >/dev/null; then
  echo "another infer_bench is running (E1/E3?); refuse to start (use --force)"; exit 1; fi

exec 3>&1
run(){ if $DRY; then echo "+ $*" >&3; return 0; fi; "$@"; }
nap(){ $DRY || sleep "$1"; }
log(){ if $DRY; then echo "$*"; else echo "$*" | tee -a "$OUT/session.log"; fi; }
CO_PID=""; LP=""
restore(){ [ -n "$CO_PID" ] && kill "$CO_PID" 2>/dev/null
  [ -n "$LP" ] && kill -TERM "$LP" 2>/dev/null
  $DRY && return 0
  sudo -n bash -c "$LAB/pilot_emc/emc_ctl.sh unlock >/dev/null 2>&1
  echo 306000000 > $G/min_freq; echo ${GMAX}000000 > $G/max_freq
  echo 950000 > /proc/sys/kernel/sched_rt_runtime_us
  for c in /sys/devices/system/cpu/cpu*/cpufreq; do echo schedutil > \$c/scaling_governor; done"; }
trap restore EXIT
trap 'exit 130' INT TERM
setclk(){ run sudo -n bash -c "echo ${1}000000 > $G/min_freq; echo ${1}000000 > $G/max_freq"; }
tmax(){ cat /sys/devices/virtual/thermal/thermal_zone*/temp 2>/dev/null | sort -rn | head -1; }
tgate(){ $DRY && return 0; TG_WAIT=0
  for i in $(seq 1 60); do t=$(tmax); [ "${t:-0}" -le 55000 ] && return 0; sleep 20; TG_WAIT=$((TG_WAIT+20)); done
  log "  WARN thermal gate timed out after ${TG_WAIT}s (t=$(tmax)); proceeding"; return 1; }
emc_check(){ $DRY && return 0; local r; r=$(sudo -n cat $EMC_RATE 2>/dev/null)
  [ "$r" = "$EMC_HZ" ] && return 0
  log "  WARN EMC rate=$r != $EMC_HZ -> re-lock"
  sudo -n bash -c "$LAB/pilot_emc/emc_ctl.sh lock $EMC_HZ" | tee -a "$OUT/session.log"; }
BG_PID=""
bg(){ # bg <cmd...>: background a command, pid in BG_PID; in dry-run print it in place (transcript in real order)
  if $DRY; then echo "+ $* &" >&3; BG_PID=""; return 0; fi; "$@" >/dev/null 2>&1 & BG_PID=$!; }
co_start(){ # fresh co-runner process on its own core, trace discarded (launch1020.sh:24-27)
  bg "$PY" -m harness.infer_bench --backend onnxruntime --model "$CO_MODEL" \
    --hz "$CO_HZ" --iters 100000 --warmup 30 --cpu "$CCPU" --label co_${CO} \
    --out /tmp/co_lext_${TAG}_t${k}; CO_PID=$BG_PID
  nap 20; }
co_stop(){ [ -n "$CO_PID" ] && kill "$CO_PID" 2>/dev/null; wait "$CO_PID" 2>/dev/null; CO_PID=""; nap 2; }
score(){ # score <prefix>: CSV re-scored at D (Table-II convention) and the harness's own count at D
  local c j; c=$("$PY" campaign/admit_pick.py --score "$1.csv" --D-ms "$D_MS" --brief 2>/dev/null) || c=NA
  j=$("$PY" -c "import json,sys; print('%.4f' % json.load(open(sys.argv[1]))['deadline_miss_rate'])" "$1.json" 2>/dev/null) || j=NA
  echo "${c} (json ${j})"; }

run sudo -n bash -c "echo -1 > /proc/sys/kernel/sched_rt_runtime_us
for c in /sys/devices/system/cpu/cpu*/cpufreq; do echo performance > \$c/scaling_governor; done"
if $DRY; then echo "+ sudo -n $LAB/pilot_emc/emc_ctl.sh lock $EMC_HZ"; else
  sudo -n bash -c "$LAB/pilot_emc/emc_ctl.sh lock $EMC_HZ" | tee "$OUT/emc_lock.txt"
  grep -q "actual=$EMC_HZ" "$OUT/emc_lock.txt" || { echo "EMC lock failed"; exit 1; }; fi
setclk "$F"; nap 0.4
log "session tag=$TAG host=$(hostname) commit=$(git rev-parse --short HEAD 2>/dev/null) $SIG launches=t$T0..t$T1 committed=$COMMITTED ina=$INA dry=$DRY start=$(date -Is)"

for k in $(seq "$T0" "$T1"); do
  P="$OUT/t${k}"
  [ -s "$P.json" ] && { echo "  skip t${k}"; continue; }
  $DRY || rm -f "$P.csv" "$P.power.csv"          # partial launch: nothing reused
  # 1. thermal gate, EMC verify, GPU pinned f
  tgate; T_GATE=$(tmax); emc_check
  setclk "$F"; nap 0.4
  # 2. co-runner, 20 s ahead
  co_start
  # 3. fresh victim process: the 1000-cycle contract at D (INA log alongside if asked)
  if $INA; then bg taskset -c 0 "$PY" campaign/ina_log.py "$P.power.csv" "$INA_S"; LP=$BG_PID; fi
  run "$PY" -m harness.infer_bench --backend onnxruntime --model "$PRI_MODEL" \
    --hz "$PRI_HZ" --iters 1000 --warmup 60 --cpu "$VCPU" --deadline-ms "$D_MS" \
    --label lext_${BOARD}_${TAG}_t${k} --out "$P" >/dev/null 2>&1
  [ -n "$LP" ] && { kill -TERM $LP 2>/dev/null; wait $LP 2>/dev/null; LP=""; }
  # 4. stop co-runner, score, log
  co_stop
  if $DRY; then M="-"; elif [ -s "$P.json" ]; then M=$(score "$P"); else M=FAIL; fi
  log "  t${k}: miss=${M} f=${F} D=${D_MS} tgate=${T_GATE} tend=$(tmax) $(date -Is)"
done
log "LAUNCH_EXT DONE tag=$TAG board=$BOARD launches=t$T0..t$T1 $(date -Is)"

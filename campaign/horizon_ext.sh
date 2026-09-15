#!/bin/bash
# E8: whether the decode-horizon term, shown under co-tenancy only at the
# 2,000-token horizon, holds at other horizons -- decode-horizon coverage under
# co-tenancy at H tokens, K independent launches.
# Protocol: docs/protocols/E8_protocol.md (frozen 2026-09-15, including
# the amendment: D is a post-hoc parameter of the per-token trace, so one
# K-launch series per H serves D = 28 and D = 44 ms alike).
# One launch k = one fresh slm_decode_bench process decoding H tokens against a
# fresh ViT-Small co-runner, i.e. joint_hxt.sh lines 30-33 (co-runner) and
# 46-48 (victim) with --tokens H, plus the admit_loop.sh hygiene:
#   1 thermal gate <=55C, EMC 2133 verified (re-locked if drifted), GPU pinned --gpu
#   2 fresh co-runner on its own core (--hz 10 --iters 100000 --warmup 30), sleep 20
#   3 fresh victim: slm_decode_bench --model qwen2.5-1.5b-instruct-q4_k_m.gguf
#     --tokens H --warmup 40 --cpu <victim> --prio 80 --out t<k>
#     (joint_hxt.sh:46-48 verbatim except --tokens/--out); stdout+stderr -> t<k>.log
#   4 co-runner killed, sleep 2; token miss % at D = 28/44 re-scored from the
#     CSV (dur_ns > D, analyze_joint.py convention); session.log line
# Board autodetected from devfreq available_frequencies (admit_loop.sh:43-50):
#   NX   victim cpu 7, co cpu 5, top 1173      Nano victim cpu 5, co cpu 4, top 1020
# sudo / priority: exactly joint_hxt.sh -- clocks, EMC and governors via sudo,
# the bench itself is NOT run under sudo (joint_hxt.sh:37-39, 46-48 call
# "$BENCH" directly; no campaign script ever ran it under sudo). --prio 80 is
# passed as there; without CAP_SYS_NICE / an rtprio ulimit the bench prints
# "SCHED_FIFO denied (need root)" and continues user-mode (the protocol's
# "user-mode loop as in results/joint_hxt"). joint_hxt.sh discarded that
# warning; t<k>.log keeps it, so the effective mode is on record per launch.
# The JSON's meta.prio records the argument, not the effective policy.
# Output: results/horizon_ext/<board>/H<H>/t<k>.{csv,json,log}, condition.txt,
#         emc_lock.txt, session.log
# Resumable: a launch whose t<k>.csv exists is skipped (the bench writes the
# CSV only after the full token loop, so a CSV is a completed measurement; a
# launch killed mid-loop leaves nothing and is re-run from scratch).
# Statistics: campaign/horizon_ext_stats.py
# Usage: ./campaign/horizon_ext.sh --H TOKENS --launches K [--t0 1] [--gpu 1173]
#          [--co-hz 10] [--board nx|nano  (dry-run preview aid only)] [--dry-run] [--force]
#        E8 on hebesu:  ./campaign/horizon_ext.sh --H 1000 --launches 5
#                       ./campaign/horizon_ext.sh --H 3000 --launches 5
set -u
PY=/home/dk/mobile/rt-infer-bench/.venv/bin/python3
LAB=/home/dk/mobile/jetson-latency-lab
BENCH=$LAB/tools/slm_decode_bench
MODEL=/home/dk/mobile/models/qwen2.5-1.5b-instruct-q4_k_m.gguf
VIT=/home/dk/mobile/rt-infer-bench/models/vit_small_patch16_224.onnx
G=/sys/devices/platform/17000000.gpu/devfreq/17000000.gpu
EMC_RATE=/sys/kernel/debug/bpmp/debug/clk/emc/rate
EMC_HZ=2133000000
WARMUP=40; PRIO=80            # joint_hxt.sh:47
D_SCORE="28 44"               # E8_protocol.md: post-hoc deadlines logged per launch (ms)

H=""; K=""; T0=1; F=1173; CO_HZ=10; BOARD_OVR=""; FORCE=false; DRY=false
usage(){ sed -n '/^# Usage/,/^set -u/p' "$0" | head -4; }
while [ $# -gt 0 ]; do case "$1" in
  --H) H=$2; shift 2;;  --launches) K=$2; shift 2;;  --t0) T0=$2; shift 2;;
  --gpu) F=$2; shift 2;;  --co-hz) CO_HZ=$2; shift 2;;  --board) BOARD_OVR=$2; shift 2;;
  --force) FORCE=true; shift;;  --dry-run) DRY=true; shift;;
  -h|--help) usage; exit 0;;  *) echo "unknown arg $1"; usage; exit 2;; esac; done
[ -n "$H" ] && [ "$H" -ge 1 ] 2>/dev/null || { echo "--H TOKENS (integer >= 1) is required"; usage; exit 2; }
[ "$H" -ge 1000 ] || $FORCE || { echo "--H $H < 1000: the head-500 / tail-500 windows of horizon_ext_stats.py would overlap; --force for a smoke test"; exit 2; }
[ -n "$K" ] && [ "$K" -ge 1 ] 2>/dev/null || { echo "--launches K (>=1) is required"; usage; exit 2; }
[ "$T0" -ge 1 ] 2>/dev/null || { echo "--t0 must be an integer >= 1"; exit 2; }
[ "$F" -ge 1 ] 2>/dev/null || { echo "--gpu must be an integer MHz"; exit 2; }
awk -v h="$CO_HZ" 'BEGIN{exit !(h+0>0)}' || { echo "--co-hz must be > 0"; exit 2; }
[ -z "$BOARD_OVR" ] || $DRY || { echo "--board only previews another board's command lines under --dry-run; real runs autodetect"; exit 2; }

# board autodetect from the devfreq table (not max_freq: that reads the
# currently pinned clock if another campaign is running or crashed pinned)
if [ -n "$BOARD_OVR" ]; then
  case $BOARD_OVR in
    nx)   STEPS="306 408 510 612 714 816 918 1020 1122 1173";;
    nano) STEPS="306 408 510 612 714 816 918 1020";;
    *) echo "--board must be nx|nano"; exit 2;; esac
else
  STEPS=$(tr ' ' '\n' < $G/available_frequencies | sed '/^$/d' | sort -n | awk '{printf "%d ", $1/1000000}')
fi
GMAX=${STEPS% }; GMAX=${GMAX##* }                    # last (highest) step, MHz
if [ "$GMAX" -ge 1100 ]; then BOARD=nx; VCPU=7; CCPU=5; else BOARD=nano; VCPU=5; CCPU=4; fi
case " $STEPS " in *" $F "*) ;; *) echo "--gpu $F is not a devfreq step on $BOARD (steps: $STEPS)"; exit 2;; esac
OUT=results/horizon_ext/$BOARD/H$H
T1=$((T0+K-1))

cd "$LAB" || exit 1; $DRY || mkdir -p "$OUT"
for f in "$BENCH" "$MODEL" "$VIT"; do [ -s "$f" ] || { echo "missing $f"; $DRY && echo "  (dry-run: continuing)" || exit 1; }; done
SIG="board=$BOARD H=$H model=$(basename "$MODEL") warmup=$WARMUP vcpu=$VCPU prio=$PRIO co=vit co_model=$(basename "$VIT") co_hz=$CO_HZ co_iters=100000 co_warmup=30 ccpu=$CCPU gpu=$F emc=$EMC_HZ"
if [ -s "$OUT/condition.txt" ] && [ "$(cat "$OUT/condition.txt")" != "$SIG" ]; then
  echo "condition mismatch in $OUT:"; cat "$OUT/condition.txt"; echo "vs: $SIG"; exit 1; fi
$DRY || echo "$SIG" > "$OUT/condition.txt"
if ! $DRY && ! $FORCE && { pgrep -f harness.infer_bench >/dev/null || pgrep -f tools/slm_decode_bench >/dev/null; }; then
  echo "another infer_bench / slm_decode_bench is running (E1/E3/E4?); refuse to start (use --force)"; exit 1; fi

exec 3>&1
run(){ if $DRY; then echo "+ $*" >&3; return 0; fi; "$@"; }
nap(){ $DRY || sleep "$1"; }
log(){ if $DRY; then echo "$*"; else echo "$*" | tee -a "$OUT/session.log"; fi; }
CO=vit; CO_PID=""
restore(){ [ -n "$CO_PID" ] && kill "$CO_PID" 2>/dev/null; pkill -f "[c]o_${CO}" 2>/dev/null
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
co_start(){ # fresh ViT-Small co-runner on its own core, trace discarded (joint_hxt.sh:30-33)
  bg "$PY" -m harness.infer_bench --backend onnxruntime --model "$VIT" \
    --hz "$CO_HZ" --iters 100000 --warmup 30 --cpu "$CCPU" --label co_vit \
    --out /tmp/co_hext_H${H}_t${k}; CO_PID=$BG_PID
  nap 20; }
co_stop(){ [ -n "$CO_PID" ] && kill "$CO_PID" 2>/dev/null; pkill -f "[c]o_${CO}" 2>/dev/null; wait "$CO_PID" 2>/dev/null; CO_PID=""; nap 2; }
victim(){ # victim <prefix>: the H-token decode, joint_hxt.sh:46-48 arguments except --tokens/--out; not under sudo
  if $DRY; then echo "+ $BENCH --model $MODEL --tokens $H --warmup $WARMUP --cpu $VCPU --prio $PRIO --out $1 > $1.log 2>&1" >&3; return 0; fi
  "$BENCH" --model "$MODEL" --tokens "$H" --warmup "$WARMUP" --cpu "$VCPU" --prio "$PRIO" --out "$1" > "$1.log" 2>&1; }
score(){ # score <csv>: token miss % at each D in D_SCORE (dur_ns > D ms, strict) and the token count
  awk -F, -v ds="$D_SCORE" 'BEGIN{n=split(ds,d," ")}
    NR>1{t++; for(i=1;i<=n;i++) if($2>d[i]*1e6) m[i]++}
    END{for(i=1;i<=n;i++) printf "D%s=%.2f%% ", d[i], (t?100*m[i]/t:0); printf "n=%d", t}' "$1"; }

run sudo -n bash -c "echo -1 > /proc/sys/kernel/sched_rt_runtime_us
for c in /sys/devices/system/cpu/cpu*/cpufreq; do echo performance > \$c/scaling_governor; done"
if $DRY; then echo "+ sudo -n $LAB/pilot_emc/emc_ctl.sh lock $EMC_HZ"; else
  sudo -n bash -c "$LAB/pilot_emc/emc_ctl.sh lock $EMC_HZ" | tee "$OUT/emc_lock.txt"
  grep -q "actual=$EMC_HZ" "$OUT/emc_lock.txt" || { echo "EMC lock failed"; exit 1; }; fi
log "session H=$H host=$(hostname) commit=$(git rev-parse --short HEAD 2>/dev/null) $SIG launches=t$T0..t$T1 dry=$DRY start=$(date -Is)"

for k in $(seq "$T0" "$T1"); do
  P="$OUT/t${k}"
  if [ -s "$P.csv" ]; then echo "  skip t${k} (csv exists$([ -s "$P.json" ] || echo ', json missing'))"; continue; fi
  $DRY || rm -f "$P.json" "$P.log"                 # partial launch: nothing reused
  # 1. thermal gate, EMC verify, GPU pinned
  tgate; T_GATE=$(tmax); emc_check
  setclk "$F"; nap 0.5                             # joint_hxt.sh:36,44 (sleep 0.5 after the pin)
  # 2. fresh co-runner, 20 s ahead
  co_start
  # 3. fresh victim process: H-token decode under the co-runner
  victim "$P"
  # 4. stop co-runner, score, log
  co_stop
  if $DRY; then M="-"; elif [ -s "$P.csv" ]; then M=$(score "$P.csv"); else M="FAIL (see $P.log)"; fi
  log "  t${k}: H=${H} gpu=${F} miss ${M} tgate=${T_GATE} tend=$(tmax) $(date -Is)"
done
log "HORIZON_EXT DONE H=$H board=$BOARD launches=t$T0..t$T1 $(date -Is)"

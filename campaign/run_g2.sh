#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# G2 main campaign (overnight, ~8.5h):
#   Part A (RQ1): EMC 4-point full curve (the BPMP-lockable set) x 6 workloads x 1k iters
#                 — maps the full curve shape incl. the cproxyv2 inversion.
#   Part B (RQ2): contention x tail, 100k runs
#                 — mobilenetv2 @50Hz x EMC{2133,3199} x adv{0,2,4}
#                 — decode_proxy @25Hz x EMC{2133} x adv{0,4}
#                 Deadline is a POST-HOC analysis parameter: full response
#                 distributions are recorded, so miss curves can be computed
#                 for any deadline (the Part 3 loose-deadline mistake cannot
#                 recur by construction).
#
# Assembled from reviewed, pilot-validated components: pilot_emc/emc_ctl.sh,
# part2/set_domain.sh, tools/membw, tools/slm_decode_bench, harness.infer_bench.
# CPU+GPU pinned throughout; EMC locked per cell; thermal gate between cells;
# RT throttling disabled for the whole campaign (SLM cells busy-spin FIFO 80
# on core 5 for up to ~100s — default throttling would inject 50ms gaps).
# Every cell is failure-isolated (one bad cell doesn't kill the night).
#
# Usage: sudo ./campaign/run_g2.sh [A|B|AB]   (default AB)
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
[ "$(id -u)" = 0 ] || { echo "run as root: sudo $0" >&2; exit 1; }

PARTS="${1:-AB}"
case "$PARTS" in A|B|AB) ;; *) echo "usage: $0 [A|B|AB]" >&2; exit 1 ;; esac
PY="${PY:-/home/dk/mobile/rt-infer-bench/.venv/bin/python3}"
EMC_CTL="$ROOT/pilot_emc/emc_ctl.sh"
MEMBW="$ROOT/tools/membw"
SLM_BENCH="$ROOT/tools/slm_decode_bench"
SLM_MODEL=/home/dk/mobile/models/qwen2.5-1.5b-instruct-q4_k_m.gguf
OUT=results/campaign_g2
mkdir -p "$OUT/partA" "$OUT/partB"

# model path + hz per workload (hz sized for the slowest EMC point)
declare -A MODEL HZ
MODEL[mobilenet]=/home/dk/mobile/rt-infer-bench/models/mobilenetv2-12.onnx;  HZ[mobilenet]=25
MODEL[vit]=/home/dk/mobile/rt-infer-bench/models/vit_small_patch16_224.onnx; HZ[vit]=5
MODEL[proxy]="$ROOT/pilot_emc/models/decode_proxy_fp16.onnx";                 HZ[proxy]=5
MODEL[cproxyv1]="$ROOT/pilot_emc/models/compute_proxy_v1_gemm2048.onnx";      HZ[cproxyv1]=4
MODEL[cproxyv2]="$ROOT/pilot_emc/models/compute_proxy_fp16.onnx";             HZ[cproxyv2]=4

GPU_TEMP_ZONE=""
for z in /sys/class/thermal/thermal_zone*; do
  [ "$(cat "$z/type" 2>/dev/null)" = "gpu-thermal" ] && GPU_TEMP_ZONE="$z/temp" && break
done

ADV_PID=""; TEGRA_PID=""; WL_PID=""
SAVED_RT_RUNTIME=$(cat /proc/sys/kernel/sched_rt_runtime_us)
cleanup() {
  # workload first: it must not keep measuring under unlocked/free clocks
  [ -n "$WL_PID" ] && kill "$WL_PID" 2>/dev/null || true
  [ -n "$ADV_PID" ] && kill "$ADV_PID" 2>/dev/null || true
  [ -n "$TEGRA_PID" ] && kill "$TEGRA_PID" 2>/dev/null || true
  echo "$SAVED_RT_RUNTIME" > /proc/sys/kernel/sched_rt_runtime_us || true
  "$EMC_CTL" unlock >/dev/null || true
  ./part2/set_domain.sh free >/dev/null || true
  chown -R dk:dk "$OUT" 2>/dev/null || true
  echo "[g2] cleanup done $(date -Is)"
}
trap cleanup EXIT
trap 'exit 143' TERM INT   # uniform path: EXIT trap runs, orphan-free

log() { echo "[g2 $(date +%H:%M:%S)] $*"; }

thermal_gate() {  # wait (max 5 min) for GPU < 55C so cells start comparable
  [ -n "$GPU_TEMP_ZONE" ] || return 0
  local waited=0 t
  while t="$(cat "$GPU_TEMP_ZONE" 2>/dev/null)" && [ "${t:-0}" -ge 55000 ] \
        && [ "$waited" -lt 300 ]; do
    sleep 10; waited=$((waited + 10))
  done
  log "thermal gate: gpu=${t:-?}mC after ${waited}s"
}

start_trace() {  # $1 = output file
  : > "$1"
  tegrastats --interval 500 >> "$1" &
  TEGRA_PID=$!
}
stop_trace() {
  [ -n "$TEGRA_PID" ] && kill "$TEGRA_PID" 2>/dev/null || true
  wait "$TEGRA_PID" 2>/dev/null || true
  TEGRA_PID=""
}

run_workload() {  # $1=out_prefix $2=workload $3=iters $4=hz_override(optional)
  local out="$1" wl="$2" iters="$3" hz="${4:-}" budget rc
  # background + tracked pid: cleanup can kill it (an orphaned foreground
  # child would keep "measuring" under unlocked clocks after SIGTERM and
  # write a json that resume then trusts forever). timeout caps a hung cell
  # so one CUDA hang can't eat the night.
  if [ "$wl" = slm ]; then
    budget=$((iters * 1 + 900))   # ≥181ms/token at EMC 204 + load + margin
    timeout -k 60 "$budget" \
      "$SLM_BENCH" --model "$SLM_MODEL" --tokens "$iters" --warmup 50 \
      --cpu 5 --prio 80 --out "$out" &
  else
    [ -n "$hz" ] || hz="${HZ[$wl]}"
    budget=$(( (iters + 100) / hz + 900 ))
    timeout -k 60 "$budget" \
      "$PY" -m harness.infer_bench \
      --backend onnxruntime --model "${MODEL[$wl]}" \
      --hz "$hz" --iters "$iters" --warmup 100 \
      --cpu 5 --prio 80 \
      --label "$(basename "$out")" --out "$out" &
  fi
  WL_PID=$!
  wait "$WL_PID"; rc=$?
  WL_PID=""
  return "$rc"
}

cell() {  # $1=dir $2=label $3=emc_rate $4=adv_threads $5=workload $6=iters $7=hz_override
  local dir="$1" label="$2" rate="$3" adv="$4" wl="$5" iters="$6" hz="${7:-}"
  local out="$OUT/$dir/$label"
  # resume key: json AND (for harness cells) the raw csv — infer_bench writes
  # json before csv, and Part B's post-hoc deadline analysis needs the csv
  if [ -s "$out.json" ] && { [ "$wl" = slm ] || [ -s "$out.csv" ]; }; then
    log "skip $label (exists)"; return 0
  fi
  log "cell $label start"
  thermal_gate
  "$EMC_CTL" lock "$rate" > "$out.emc_lock" || { log "!! $label EMC lock failed"; return 1; }
  local actual
  actual=$(grep -o 'actual=[0-9]*' "$out.emc_lock" | cut -d= -f2)
  if [ "$actual" != "$rate" ]; then
    log "!! $label EMC rounded: requested $rate got $actual — cell invalid"
    return 1
  fi
  sleep 2
  if [ "$adv" -gt 0 ]; then
    "$MEMBW" -t "$adv" -c 0,1,2,3 -m 256 -M write -d 5400 \
      > "$out.membw.json" 2> "$out.membw.log" &
    ADV_PID=$!
    sleep 3
  fi
  start_trace "$out.tegra"
  run_workload "$out" "$wl" "$iters" "$hz" || log "!! $label workload failed"
  stop_trace
  if [ -n "$ADV_PID" ]; then
    kill -TERM "$ADV_PID" 2>/dev/null || true
    wait "$ADV_PID" 2>/dev/null || true
    ADV_PID=""
  fi
  "$EMC_CTL" status > "$out.emc_after"
  log "cell $label done"
}

log "campaign start: parts=$PARTS"
echo -1 > /proc/sys/kernel/sched_rt_runtime_us
./part2/set_domain.sh all >/dev/null || { log "set_domain all FAILED — abort"; exit 1; }

# Pre-flight: every Part A rate must lock EXACTLY (the BPMP rounds off-table
# requests — 1701.5MHz silently became 2133MHz in the pilot). Abort now, at
# launch time, rather than discover mislabeled cells in the morning.
for rate in 204000000 665600000 2133000000 3199000000; do
  got=$("$EMC_CTL" lock "$rate" | grep -o 'actual=[0-9]*' | cut -d= -f2)
  [ "$got" = "$rate" ] || { log "pre-flight FAIL: $rate locks to $got — fix MATRIX"; exit 1; }
done
"$EMC_CTL" unlock >/dev/null
log "pre-flight: all 4 EMC rates lock exactly"

if [[ "$PARTS" == *A* ]]; then
  log "=== Part A: RQ1 EMC 4-point curve ==="
  # Lockable set probed 2026-06-11: requests round UP to {204, 665.6, 2133, 3199}MHz.
  # dvfs_table entries 1600/2750/3200 are voltage operating points, not lockable rates.
  for rate in 204000000 665600000 2133000000 3199000000; do
    mhz=$((rate / 1000000))
    for wl in mobilenet vit proxy cproxyv1 cproxyv2 slm; do
      if [ "$wl" = slm ]; then iters=500; else iters=1000; fi
      cell partA "emc${mhz}_${wl}" "$rate" 0 "$wl" "$iters" || true
    done
  done
fi

if [[ "$PARTS" == *B* ]]; then
  log "=== Part B: RQ2 contention x tail (100k) ==="
  for rate in 2133000000 3199000000; do
    mhz=$((rate / 1000000))
    for adv in 0 2 4; do
      cell partB "emc${mhz}_adv${adv}_mobilenet" "$rate" "$adv" mobilenet 100000 50 || true
    done
  done
  for adv in 0 4; do
    cell partB "emc2133_adv${adv}_proxy" 2133000000 "$adv" proxy 100000 25 || true
  done
fi

log "campaign complete"

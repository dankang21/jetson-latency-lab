#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# E1+E2: the reviewer-demanded estimator experiments.
#
# E1 (governor observation): with the EMC UNLOCKED and CPU/GPU pinned, does
# the stock actmon governor actually move the EMC during inference? If yes,
# "b was constant because EMC was fixed in their setup" is not a defense —
# deployment does not fix it. Traces tegrastats at 100ms per workload.
#
# E2 (estimator break): fit set = GPU frequency sweep (8 devfreq points) at
# EMC 3199; test sets = same sweep at EMC 2133 and 665.6. A FLAME-style
# T(f_gpu) = k/f_gpu + b fitted on the 3199 cells is then evaluated on the
# other EMC points (analyze_estimator.py): in-scope vs out-of-scope error,
# and the error reduction from an EMC-aware term (E3).
#
# ~2.2h total. Usage: sudo ./campaign/run_estimator_break.sh
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
[ "$(id -u)" = 0 ] || { echo "run as root" >&2; exit 1; }

PY="${PY:-/home/dk/mobile/rt-infer-bench/.venv/bin/python3}"
EMC_CTL="$ROOT/pilot_emc/emc_ctl.sh"
SLM_BENCH="$ROOT/tools/slm_decode_bench"
SLM_MODEL=/home/dk/mobile/models/qwen2.5-1.5b-instruct-q4_k_m.gguf
GPU=/sys/devices/platform/17000000.gpu/devfreq/17000000.gpu
OUT_E1=results/e1_governor
OUT_E2=results/estimator_break
mkdir -p "$OUT_E1" "$OUT_E2"

declare -A MODEL HZ
MODEL[mobilenet]=/home/dk/mobile/rt-infer-bench/models/mobilenetv2-12.onnx;  HZ[mobilenet]=8
MODEL[vit]=/home/dk/mobile/rt-infer-bench/models/vit_small_patch16_224.onnx; HZ[vit]=3
MODEL[proxy]="$ROOT/pilot_emc/models/decode_proxy_fp16.onnx";                 HZ[proxy]=8
MODEL[cproxyv2]="$ROOT/pilot_emc/models/compute_proxy_fp16.onnx";             HZ[cproxyv2]=3

TEGRA_PID=""; WL_PID=""
cleanup() {
  [ -n "$WL_PID" ] && kill "$WL_PID" 2>/dev/null || true
  [ -n "$TEGRA_PID" ] && kill "$TEGRA_PID" 2>/dev/null || true
  "$EMC_CTL" unlock >/dev/null || true
  ./part2/set_domain.sh free >/dev/null || true
  chown -R dk:dk "$OUT_E1" "$OUT_E2" 2>/dev/null || true
  echo "[eb] cleanup done $(date -Is)"
}
trap cleanup EXIT
trap 'exit 143' TERM INT

log() { echo "[eb $(date +%H:%M:%S)] $*"; }

set_gpu() {  # min<=max invariant: drop min to floor, set max, raise min
  echo 306000000 > $GPU/min_freq
  echo "$1" > $GPU/max_freq
  echo "$1" > $GPU/min_freq
  sleep 1
}

run_wl() {  # $1=out $2=wl $3=iters $4(optional)=hz
  local out="$1" wl="$2" iters="$3" hz="${4:-}"
  if [ "$wl" = slm ]; then
    timeout -k 60 900 "$SLM_BENCH" --model "$SLM_MODEL" --tokens "$iters" \
      --warmup 50 --cpu 5 --prio 80 --out "$out" &
  else
    [ -n "$hz" ] || hz="${HZ[$wl]}"
    timeout -k 60 $(( (iters+50)/hz + 600 )) \
      "$PY" -m harness.infer_bench --backend onnxruntime \
      --model "${MODEL[$wl]}" --hz "$hz" --iters "$iters" --warmup 50 \
      --cpu 5 --prio 80 --label "$(basename "$out")" --out "$out" &
  fi
  WL_PID=$!; wait "$WL_PID"; local rc=$?; WL_PID=""; return $rc
}

log "=== E1: stock governor, EMC unlocked, CPU/GPU pinned ==="
./part2/set_domain.sh all >/dev/null || { log "set_domain failed"; exit 1; }
"$EMC_CTL" unlock >/dev/null
sleep 3

e1_trace() {  # $1=label $2=wl $3=iters
  log "E1 $1"
  : > "$OUT_E1/$1.tegra"
  tegrastats --interval 100 >> "$OUT_E1/$1.tegra" &
  TEGRA_PID=$!
  if [ "$2" = idle ]; then sleep 30; else run_wl "$OUT_E1/$1" "$2" "$3" || true; fi
  kill "$TEGRA_PID" 2>/dev/null || true; wait "$TEGRA_PID" 2>/dev/null || true
  TEGRA_PID=""
}
e1_trace idle idle 0
e1_trace mobilenet mobilenet 1500
e1_trace proxy proxy 600
e1_trace slm slm 500

log "=== E2: GPU sweep x EMC {3199 fit, 2133/665.6 test} ==="
for rate in 3199000000 2133000000 665600000; do
  mhz=$((rate / 1000000))
  "$EMC_CTL" lock "$rate" > "$OUT_E2/emc${mhz}.emc_lock"
  grep -q "actual=$rate" "$OUT_E2/emc${mhz}.emc_lock" || { log "EMC $rate rounded — abort"; exit 1; }
  sleep 2
  for gpu in 306000000 408000000 510000000 612000000 714000000 816000000 918000000 1020000000; do
    gmhz=$((gpu / 1000000))
    set_gpu "$gpu"
    for wl in mobilenet vit proxy cproxyv2; do
      label="emc${mhz}_gpu${gmhz}_${wl}"
      if [ -s "$OUT_E2/$label.json" ] && [ -s "$OUT_E2/$label.csv" ]; then
        log "skip $label"; continue
      fi
      run_wl "$OUT_E2/$label" "$wl" 300 || log "!! $label failed"
    done
  done
  set_gpu 1020000000
done

log "E1+E2 complete"

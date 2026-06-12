#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Adversary effectiveness check (G1 pre-work for the RQ2 campaign).
#
# Part 3's stress-ng adversaries moved the victim ~0.3% — page-fault/cache
# pressure, not DRAM bandwidth pressure. Before designing the RQ2 contention
# matrix around tools/membw, verify it actually hurts:
#
#   EMC {3199, 2133} x adversary {0, 2, 4 write-threads on cores 0-3}
#                    x workload {cnn, proxy}, 300 iters each
#
# Pass criterion: proxy p50 slowdown >20% at any adversary level (the
# bandwidth-bound victim must feel a bandwidth adversary; if it does not,
# the adversary design is wrong, not the hypothesis).
#
# Victim on core 5 (FIFO 80), adversary on 0-3, core 4 left for the OS.
# CPU+GPU pinned throughout; EMC locked per cell (pilot_emc semantics).
#
# Usage: sudo ./pilot_contention/run_adv_check.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
[ "$(id -u)" = 0 ] || { echo "run as root: sudo $0" >&2; exit 1; }

PY="${PY:-/home/dk/mobile/rt-infer-bench/.venv/bin/python3}"
CNN_MODEL="${CNN_MODEL:-/home/dk/mobile/rt-infer-bench/models/mobilenetv2-12.onnx}"
PROXY_MODEL="${PROXY_MODEL:-$ROOT/pilot_emc/models/decode_proxy_fp16.onnx}"
MEMBW="$ROOT/tools/membw"
EMC_CTL="$ROOT/pilot_emc/emc_ctl.sh"
ITERS="${ITERS:-300}"; WARMUP="${WARMUP:-50}"
HZ_CNN="${HZ_CNN:-25}"; HZ_PROXY="${HZ_PROXY:-3}"
ADV_CPUS="0,1,2,3"
OUT=results/pilot_contention
mkdir -p "$OUT"

[ -x "$MEMBW" ] || { echo "build it first: gcc -O2 -pthread -o tools/membw tools/membw.c" >&2; exit 1; }

ADV_PID=""
cleanup() {
  [ -n "$ADV_PID" ] && kill "$ADV_PID" 2>/dev/null || true
  "$EMC_CTL" unlock || true
  ./part2/set_domain.sh free || true
}
trap cleanup EXIT

./part2/set_domain.sh all

run_cell() {
  local emc_rate="$1" adv="$2" workload="$3" model="$4" hz="$5"
  local mhz=$(( emc_rate / 1000000 ))
  local label="emc${mhz}_adv${adv}_${workload}"
  echo "================ $label ================"

  "$EMC_CTL" lock "$emc_rate" > "$OUT/${label}.emc_lock"
  sleep 2
  # thermal snapshot per cell: locked clocks + memset load can still throttle
  # via soctherm, which would confound cross-cell comparisons
  paste <(cat /sys/class/thermal/thermal_zone*/type) \
        <(cat /sys/class/thermal/thermal_zone*/temp) > "$OUT/${label}.thermal" 2>/dev/null || true

  ADV_PID=""
  if [ "$adv" -gt 0 ]; then
    # adversary runs SCHED_OTHER on cores 0-3; victim is FIFO 80 on core 5
    "$MEMBW" -t "$adv" -c "$ADV_CPUS" -m 256 -M write \
      > "$OUT/${label}.membw.json" 2> "$OUT/${label}.membw.log" &
    ADV_PID=$!
    sleep 3   # let its pages fault in and throughput stabilize
  fi

  "$PY" -m harness.infer_bench \
    --backend onnxruntime --model "$model" \
    --hz "$hz" --iters "$ITERS" --warmup "$WARMUP" \
    --cpu 5 --prio 80 \
    --label "$label" --out "$OUT/$label" || true

  if [ -n "$ADV_PID" ]; then
    kill -TERM "$ADV_PID" 2>/dev/null || true
    wait "$ADV_PID" 2>/dev/null || true
    ADV_PID=""
  fi
  "$EMC_CTL" status > "$OUT/${label}.emc_after"
}

for emc in 3199000000 2133000000; do
  for adv in 0 2 4; do
    run_cell "$emc" "$adv" cnn   "$CNN_MODEL"   "$HZ_CNN"
    run_cell "$emc" "$adv" proxy "$PROXY_MODEL" "$HZ_PROXY"
  done
done

echo "done. analyze: python3 pilot_contention/analyze_adv.py"

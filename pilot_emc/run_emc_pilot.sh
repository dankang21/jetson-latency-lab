#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# G0-② EMC sensitivity pilot (RQ1 kill-criteria check before the main campaign).
#
# Matrix: EMC {min, mid, max} x workload {cnn (compute-heavy control),
# proxy (memory-bound decode proxy)}. CPU+GPU are pinned the whole time
# (part2/set_domain.sh all), so EMC is the ONLY moving clock — any latency
# shift between cells is attributable to the memory domain.
#
# Kill check (analyze_emc_pilot.py prints the verdict):
#   RQ1 dies  if p50 shift max<->min EMC is <5% for BOTH workloads.
#   RQ1 lives best if cnn barely moves and proxy moves a lot — that is the
#   "frequency-independent constant b is actually b(f_emc)" shape.
#
# Usage:
#   sudo ./pilot_emc/run_emc_pilot.sh
# Env overrides:
#   PY, CNN_MODEL, PROXY_MODEL, ITERS, WARMUP, CPU, PRIO,
#   HZ_CNN (default 25), HZ_PROXY (default 3 — proxy can hit ~100ms at EMC min)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

[ "$(id -u)" = 0 ] || { echo "run as root: sudo $0" >&2; exit 1; }

PY="${PY:-$HOME/mobile/rt-infer-bench/.venv/bin/python3}"
[ -x "$PY" ] || PY=/home/dk/mobile/rt-infer-bench/.venv/bin/python3
CNN_MODEL="${CNN_MODEL:-/home/dk/mobile/rt-infer-bench/models/mobilenetv2-12.onnx}"
PROXY_MODEL="${PROXY_MODEL:-$ROOT/pilot_emc/models/decode_proxy_fp16.onnx}"
ITERS="${ITERS:-1000}"; WARMUP="${WARMUP:-100}"
CPU="${CPU:-5}"; PRIO="${PRIO:-80}"
HZ_CNN="${HZ_CNN:-25}"; HZ_PROXY="${HZ_PROXY:-3}"
INTERVAL_MS="${INTERVAL_MS:-200}"
EMC_CTL="$ROOT/pilot_emc/emc_ctl.sh"
OUT=results/pilot_emc
mkdir -p "$OUT"

[ -f "$PROXY_MODEL" ] || { echo "proxy model missing — generate it first (no root needed):
  /home/dk/mobile/rt-infer-bench/vit_export/bin/python3 pilot_emc/make_decode_proxy.py" >&2; exit 1; }

# Whatever happens, give the board its clocks back.
cleanup() {
  "$EMC_CTL" unlock || true
  ./part2/set_domain.sh free || true
}
trap cleanup EXIT

# CPU+GPU pinned for the entire sweep; only EMC moves between cells.
./part2/set_domain.sh all
"$EMC_CTL" status | tee "$OUT/emc_before.txt"

read -r EMC_MIN EMC_MID EMC_MAX <<< "$("$EMC_CTL" points)"
echo "EMC sweep points (Hz): $EMC_MIN $EMC_MID $EMC_MAX"

run_cell() {
  local rate="$1" workload="$2" model="$3" hz="$4"
  local mhz=$(( rate / 1000000 ))
  local label="emc${mhz}_${workload}"
  echo "================ $label ================"

  "$EMC_CTL" lock "$rate" | tee "$OUT/${label}.emc_lock"
  sleep 3   # let the rate + thermals settle

  local tegra="$OUT/${label}.tegra"
  : > "$tegra"
  tegrastats --interval "$INTERVAL_MS" >> "$tegra" &
  local TPID=$!

  "$PY" -m harness.infer_bench \
    --backend onnxruntime --model "$model" \
    --hz "$hz" --iters "$ITERS" --warmup "$WARMUP" \
    --cpu "$CPU" --prio "$PRIO" \
    --label "$label" --out "$OUT/$label" || true

  kill "$TPID" 2>/dev/null || true; wait "$TPID" 2>/dev/null || true
  # Read the rate back AFTER the run — proves the lock held under load.
  "$EMC_CTL" status | tee "$OUT/${label}.emc_after"
}

for rate in "$EMC_MIN" "$EMC_MID" "$EMC_MAX"; do
  run_cell "$rate" cnn   "$CNN_MODEL"   "$HZ_CNN"
  run_cell "$rate" proxy "$PROXY_MODEL" "$HZ_PROXY"
done

echo "done. verdict: python3 pilot_emc/analyze_emc_pilot.py"

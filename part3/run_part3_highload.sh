#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Part 3 high-load sweep: push contention to a realistic worst case and find the
# breaking point (if any). Companion to run_part3.sh (normal load).
#
# Clocks pinned (jetson_clocks). Inference is SCHED_FIFO prio 80 on a dedicated
# core. Outputs results/p3hi_<profile>.{json,csv,tegra}.
#
# Memory safety: 8 GB board. OS + inference use ~2-3 GB, so stressor RAM is
# capped well under the rest and --oom-avoid is set, so the OOM killer never
# touches the inference process. vm-bytes are absolute (not %) for predictability.
#
# Usage:
#   sudo PY=$(which python3) MODEL=~/mobile/rt-infer-bench/models/mobilenetv2-12.onnx \
#        CPU=5 ITERS=100000 INTERVAL_MS=200 ./part3/run_part3_highload.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
MODEL="${MODEL:-$HOME/mobile/rt-infer-bench/models/mobilenetv2-12.onnx}"
CPU="${CPU:-5}"; HZ="${HZ:-100}"; ITERS="${ITERS:-100000}"
WARMUP="${WARMUP:-2000}"; INTERVAL_MS="${INTERVAL_MS:-200}"; RAMP="${RAMP:-3}"
PY="${PY:-$(which python3)}"
mkdir -p results

PROFILE_SECS=$(( (ITERS + WARMUP) / HZ + RAMP + 60 ))

stop_stress () {
  timeout 10 pkill -TERM -f "stress-ng" 2>/dev/null || true
  sleep 1
  timeout 10 pkill -KILL -f "stress-ng" 2>/dev/null || true
}
cleanup () { [[ -n "${TEGRA_PID:-}" ]] && kill "$TEGRA_PID" 2>/dev/null || true; stop_stress; }
trap cleanup EXIT INT TERM

echo "[part3-hi] locking clocks (jetson_clocks)"
jetson_clocks || echo "  (jetson_clocks unavailable / already set)"

run_profile () {
  local name="$1" sargs="$2"
  echo "================ profile: $name ($(date +%H:%M:%S)) ================"
  if [[ -n "$sargs" ]]; then
    # shellcheck disable=SC2086
    stress-ng $sargs --oom-avoid --timeout "${PROFILE_SECS}s" >/dev/null 2>&1 &
    echo "  stress-ng args: $sargs  (ramp ${RAMP}s, self-timeout ${PROFILE_SECS}s)"
    sleep "$RAMP"
  fi
  local tegra="results/p3hi_${name}.tegra"; : > "$tegra"
  tegrastats --interval "$INTERVAL_MS" >> "$tegra" &
  TEGRA_PID=$!
  "$PY" -m harness.infer_bench \
    --backend onnxruntime --model "$MODEL" \
    --hz "$HZ" --iters "$ITERS" --warmup "$WARMUP" \
    --cpu "$CPU" --prio 80 \
    --label "p3hi_${name}" --out "results/p3hi_${name}" || true
  kill "$TEGRA_PID" 2>/dev/null || true; TEGRA_PID=""
  [[ -n "$sargs" ]] && stop_stress
  echo "  -> results/p3hi_${name}.{json,csv} + .tegra  ($(date +%H:%M:%S))"
}

# --- memory bandwidth sweep: STREAM-style workers, escalating ---
# --stream hammers memory bandwidth directly (closer to what GPU inference fights
# for than --vm). Worker count is the knob.
run_profile mem_2   "--stream 2"
run_profile mem_4   "--stream 4"
run_profile mem_6   "--stream 6"
run_profile mem_8   "--stream 8"

# --- vm pressure sweep: absolute bytes, OOM-safe (total stays < ~4.8 GB) ---
run_profile vm_light  "--vm 4 --vm-bytes 600m"
run_profile vm_heavy  "--vm 6 --vm-bytes 800m"

# --- other axes at maximum, one shot each ---
run_profile cache_max "--cache 8 --cache-level 3"
run_profile irq_storm "--timer 16 --timerfd 8 --clock 8"

# --- everything at once, OOM-safe ---
run_profile combined_max "--cpu 6 --stream 4 --vm 4 --vm-bytes 500m --cache 4 --io 4"

echo "done ($(date +%H:%M:%S)). analyze: python3 part3/analyze_part3_highload.py"

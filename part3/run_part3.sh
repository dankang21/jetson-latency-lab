#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Part 3 driver (hang-proof). Does contention break the 100 Hz deadline, and does
# it break COMPUTE (GPU/memory/cache) or JITTER (scheduler/IRQ)?
#
# Inference runs SCHED_FIFO prio 80 on a dedicated core; stressors cannot steal
# its CPU time, so what leaks in is shared-resource contention. Clocks are pinned
# (jetson_clocks) so DVFS variation does not masquerade as a stress effect.
#
# Lifecycle (why this won't hang like the previous version):
#   * each stressor has a HARD --timeout = the measurement window, so it
#     self-terminates even if every kill below is a no-op.
#   * teardown is `timeout`-wrapped pkill; no unbounded wait/sleep loop, never
#     waits on a pid.
#
# Usage:
#   sudo PY=$(which python3) MODEL=~/mobile/rt-infer-bench/models/mobilenetv2-12.onnx \
#        CPU=5 ITERS=100000 INTERVAL_MS=200 ./part3/run_part3.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
MODEL="${MODEL:-$HOME/mobile/rt-infer-bench/models/mobilenetv2-12.onnx}"
CPU="${CPU:-5}"; HZ="${HZ:-100}"; ITERS="${ITERS:-100000}"
WARMUP="${WARMUP:-2000}"; INTERVAL_MS="${INTERVAL_MS:-200}"; RAMP="${RAMP:-3}"
PY="${PY:-$(which python3)}"
mkdir -p results

# Per-profile run length (s) + margin. stress-ng self-kills after this no matter
# what, so a stuck teardown can never leave it running (the 12-hour bug).
PROFILE_SECS=$(( (ITERS + WARMUP) / HZ + RAMP + 60 ))

# Kill any stressor, bounded. `timeout` guarantees return even if pkill wedges;
# we never wait on a pid.
stop_stress () {
  timeout 10 pkill -TERM -f "stress-ng" 2>/dev/null || true
  sleep 1
  timeout 10 pkill -KILL -f "stress-ng" 2>/dev/null || true
}

cleanup () {
  [[ -n "${TEGRA_PID:-}" ]] && kill "$TEGRA_PID" 2>/dev/null || true
  stop_stress
}
trap cleanup EXIT INT TERM

echo "[part3] locking clocks (jetson_clocks) for a stable baseline"
jetson_clocks || echo "  (jetson_clocks unavailable / already set)"

run_profile () {
  local name="$1" sargs="$2"
  echo "================ profile: $name ($(date +%H:%M:%S)) ================"

  if [[ -n "$sargs" ]]; then
    # shellcheck disable=SC2086
    stress-ng $sargs --timeout "${PROFILE_SECS}s" >/dev/null 2>&1 &
    echo "  stress-ng args: $sargs  (ramp ${RAMP}s, self-timeout ${PROFILE_SECS}s)"
    sleep "$RAMP"
  fi

  local tegra="results/p3_${name}.tegra"; : > "$tegra"
  tegrastats --interval "$INTERVAL_MS" >> "$tegra" &
  TEGRA_PID=$!

  "$PY" -m harness.infer_bench \
    --backend onnxruntime --model "$MODEL" \
    --hz "$HZ" --iters "$ITERS" --warmup "$WARMUP" \
    --cpu "$CPU" --prio 80 \
    --label "p3_${name}" --out "results/p3_${name}" || true

  kill "$TEGRA_PID" 2>/dev/null || true; TEGRA_PID=""
  [[ -n "$sargs" ]] && stop_stress
  echo "  -> results/p3_${name}.{json,csv} + .tegra  ($(date +%H:%M:%S))"
}

run_profile baseline ""
run_profile cpu      "--cpu 6"
run_profile memory   "--vm 4 --vm-bytes 25%"
run_profile cache    "--cache 4"
run_profile io       "--io 4"
run_profile irq      "--timer 8 --timerfd 4 --clock 4"
run_profile combined "--cpu 4 --vm 2 --vm-bytes 25% --timer 4 --io 2"

echo "done ($(date +%H:%M:%S)). analyze: python3 part3/analyze_part3.py"

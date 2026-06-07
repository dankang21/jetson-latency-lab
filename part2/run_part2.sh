#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Part 2 driver: for each clock-domain profile, log tegrastats in parallel while
# running the 100 Hz inference loop. Produces, per profile:
#   results/p2_<profile>.json / .csv      (latency, from the harness)
#   results/p2_<profile>.tegra            (clock + temp + power trace, 200 ms)
#
# The tegrastats trace is the evidence that closes "the clock actually dropped"
# (vs. inferring it). Cycle latency and clock trace share a wall-clock axis.
#
# Usage:
#   sudo ./part2/run_part2.sh
# Env:
#   MODEL=~/mobile/rt-infer-bench/models/mobilenetv2-12.onnx
#   CPU=5  HZ=100  ITERS=100000  WARMUP=2000  INTERVAL_MS=200
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
MODEL="${MODEL:-$HOME/mobile/rt-infer-bench/models/mobilenetv2-12.onnx}"
CPU="${CPU:-5}"; HZ="${HZ:-100}"; ITERS="${ITERS:-100000}"
WARMUP="${WARMUP:-2000}"; INTERVAL_MS="${INTERVAL_MS:-200}"
PY="${PY:-$(which python3)}"
mkdir -p results

run_profile () {
  local prof="$1"
  echo "================ profile: $prof ================"
  ./part2/set_domain.sh "$prof"
  sleep 3   # let clocks settle at the new floor

  # start tegrastats trace in background
  local tegra="results/p2_${prof}.tegra"
  : > "$tegra"
  tegrastats --interval "$INTERVAL_MS" >> "$tegra" &
  local TPID=$!

  # run the periodic loop (same harness as Part 1)
  "$PY" -m harness.infer_bench \
    --backend onnxruntime --model "$MODEL" \
    --hz "$HZ" --iters "$ITERS" --warmup "$WARMUP" \
    --cpu "$CPU" --prio 80 \
    --label "p2_${prof}" --out "results/p2_${prof}" || true

  kill "$TPID" 2>/dev/null || true; wait "$TPID" 2>/dev/null || true
  echo "  -> results/p2_${prof}.{json,csv} + .tegra"
}

# free first (natural DVFS), then isolate each domain, then all (= Part 1 clocked)
run_profile free
run_profile gpu_only
run_profile cpu_only
run_profile all

# leave the board in a clean dynamic state
./part2/set_domain.sh free
echo "done. analyze: python3 part2/analyze_part2.py"

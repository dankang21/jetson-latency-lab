#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Run the full stress matrix. Each profile: start stress-ng, let it ramp, run
# the pinned RT inference loop, then tear the stressor down.
#
# Usage:
#   sudo ./experiments/run_matrix.sh
#
# Env overrides:
#   MODEL=models/mobilenetv2.onnx   path to the ONNX model
#   BACKEND=onnxruntime|baseline    inference backend (default onnxruntime)
#   CPU=5                           isolated core to pin the loop to
#   PRIO=80                         SCHED_FIFO priority
#   HZ=100  ITERS=100000  WARMUP=2000
#   MITIGATIONS=1                   apply jetson_clocks + IRQ steering first
#
# Prereqs:
#   sudo apt-get install stress-ng
#   For a real RT story, isolate CPU $CPU at boot: add to kernel cmdline
#     isolcpus=$CPU nohz_full=$CPU rcu_nocbs=$CPU
#   and confirm with: cat /sys/devices/system/cpu/isolated
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MODEL="${MODEL:-models/mobilenetv2.onnx}"
BACKEND="${BACKEND:-onnxruntime}"
CPU="${CPU:-5}"
PRIO="${PRIO:-80}"
HZ="${HZ:-100}"
ITERS="${ITERS:-100000}"
WARMUP="${WARMUP:-2000}"
MITIGATIONS="${MITIGATIONS:-0}"
OUTDIR="results"
mkdir -p "$OUTDIR"

if [[ "$MITIGATIONS" == "1" ]]; then
  echo "[mitigations] locking clocks via jetson_clocks"
  jetson_clocks || echo "  (jetson_clocks not available / already set)"
  # Steer most IRQs away from the isolated core (best effort).
  for irq in /proc/irq/[0-9]*; do
    echo "$(printf '%x' $((0xffffffff & ~(1<<CPU))))" > "$irq/smp_affinity" 2>/dev/null || true
  done
fi

run_one () {
  local name="$1" sargs="$2" warmup_s="${3:-3}"
  local tag="$name"; [[ "$MITIGATIONS" == "1" ]] && tag="${name}_tuned"
  echo "=== profile: $tag ==="
  local pid=""
  if [[ "$sargs" != "null" && -n "$sargs" ]]; then
    # shellcheck disable=SC2086
    stress-ng $sargs --timeout 0 &
    pid=$!
    echo "  stress-ng pid=$pid args: $sargs  (ramp ${warmup_s}s)"
    sleep "$warmup_s"
  fi
  python3 -m harness.infer_bench \
    --backend "$BACKEND" --model "$MODEL" \
    --hz "$HZ" --iters "$ITERS" --warmup "$WARMUP" \
    --cpu "$CPU" --prio "$PRIO" \
    --label "$tag" --out "$OUTDIR/$tag" || true
  if [[ -n "$pid" ]]; then
    kill "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
  fi
}

# --- matrix (keep in sync with experiments/profiles.yaml) ---
run_one baseline       "null"
run_one cpu_partial    "--cpu 4"
run_one cpu_full       "--cpu 8"
run_one mem_bandwidth  "--vm 4 --vm-bytes 80% --vm-method all"
run_one cache_thrash   "--cache 6 --cache-level 3"
run_one io_stress      "--io 4 --hdd 2 --hdd-bytes 256m"
run_one irq_load       "--timer 8 --timerfd 8 --sched 4"
run_one combined       "--cpu 4 --vm 2 --vm-bytes 50% --cache 4 --io 2 --timer 4"

# --- thermal: long ramp to provoke throttling ---
run_one thermal        "--cpu 6 --matrix 6" 120

echo "done. summaries in $OUTDIR/*.json"
echo "next: python3 analysis/analyze.py && python3 analysis/plot.py"

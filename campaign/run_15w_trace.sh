#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Fill the 15W row of tab:powermode: stock-governor EMC behavior under the
# 15W nvpmodel profile (idle/CNN/GEMV/SLM traces). Restores MAXN_SUPER.
# Usage: sudo ./campaign/run_15w_trace.sh   (~10 min)
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$ROOT"
[ "$(id -u)" = 0 ] || { echo "run as root" >&2; exit 1; }

PY=/home/dk/mobile/rt-infer-bench/.venv/bin/python3
SLM_BENCH="$ROOT/tools/slm_decode_bench"
SLM_MODEL=/home/dk/mobile/models/qwen2.5-1.5b-instruct-q4_k_m.gguf
EMC_CTL="$ROOT/pilot_emc/emc_ctl.sh"
GPU=/sys/devices/platform/17000000.gpu/devfreq/17000000.gpu
OUT=results/replication_15w; mkdir -p "$OUT"

TEGRA_PID=""; WL_PID=""
cleanup() {
  [ -n "$WL_PID" ] && kill "$WL_PID" 2>/dev/null || true
  [ -n "$TEGRA_PID" ] && kill "$TEGRA_PID" 2>/dev/null || true
  "$EMC_CTL" unlock >/dev/null || true
  ./part2/set_domain.sh free >/dev/null || true
  nvpmodel -m 2 < /dev/null >/dev/null 2>&1 || true
  chown -R dk:dk "$OUT" 2>/dev/null || true
  echo "[r15] restored: $(nvpmodel -q 2>/dev/null | head -1)"
}
trap cleanup EXIT
trap 'exit 143' TERM INT
log() { echo "[r15 $(date +%H:%M:%S)] $*"; }

log "switching to 15W (mode 0)"
nvpmodel -m 0 < /dev/null || { log "switch FAILED"; exit 1; }
sleep 3
# pin CPU/GPU at the 15W caps so only the EMC governor is observed
CPU_CAP=$(cat /sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq)
for c in /sys/devices/system/cpu/cpu*/cpufreq; do
  echo performance > $c/scaling_governor || true
done
echo 306000000 > $GPU/min_freq; GMAX=$(cat $GPU/max_freq)
echo "$GMAX" > $GPU/min_freq
"$EMC_CTL" unlock >/dev/null
sleep 3
log "15W pinned: cpu_cap=$CPU_CAP gpu=$(cat $GPU/cur_freq)"

trace() { # $1=label $2=cmd...
  local label="$1"; shift
  log "trace $label"
  : > "$OUT/$label.tegra"
  tegrastats --interval 100 >> "$OUT/$label.tegra" &
  TEGRA_PID=$!
  "$@" || true
  kill "$TEGRA_PID" 2>/dev/null || true; wait "$TEGRA_PID" 2>/dev/null || true
  TEGRA_PID=""
}
run_bg() { "$@" & WL_PID=$!; wait "$WL_PID"; local rc=$?; WL_PID=""; return $rc; }

trace idle sleep 30
trace mobilenet run_bg timeout -k 30 300 "$PY" -m harness.infer_bench \
  --backend onnxruntime --model /home/dk/mobile/rt-infer-bench/models/mobilenetv2-12.onnx \
  --hz 8 --iters 480 --warmup 50 --cpu 5 --prio 80 \
  --label r15_mobilenet --out "$OUT/r15_mobilenet"
trace proxy run_bg timeout -k 30 300 "$PY" -m harness.infer_bench \
  --backend onnxruntime --model "$ROOT/pilot_emc/models/decode_proxy_fp16.onnx" \
  --hz 8 --iters 480 --warmup 50 --cpu 5 --prio 80 \
  --label r15_proxy --out "$OUT/r15_proxy"
trace slm run_bg timeout -k 60 600 "$SLM_BENCH" --model "$SLM_MODEL" \
  --tokens 300 --warmup 30 --cpu 5 --prio 80 --out "$OUT/r15_slm"
log "15W traces complete"

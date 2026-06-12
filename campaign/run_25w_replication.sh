#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Single-board replication under a different firmware/clock configuration:
# the 25W nvpmodel profile (CPU 1344MHz, GPU 918MHz, EMC<=3199MHz) vs the
# MAXN_SUPER profile every other measurement used. Re-checks (a) the RQ1
# EMC curve including the cproxyv2 inversion, (b) the stock governor's EMC
# behavior in this mode (fills the 25W row of tab:powermode).
#
# Restores MAXN_SUPER (mode 2) and dynamic clocks on exit, success or not.
# Usage: sudo ./campaign/run_25w_replication.sh   (~20 min)
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
[ "$(id -u)" = 0 ] || { echo "run as root" >&2; exit 1; }

PY="${PY:-/home/dk/mobile/rt-infer-bench/.venv/bin/python3}"
EMC_CTL="$ROOT/pilot_emc/emc_ctl.sh"
GPU=/sys/devices/platform/17000000.gpu/devfreq/17000000.gpu
OUT=results/replication_25w
mkdir -p "$OUT"

CPU_25W=1344000
GPU_25W=918000000

declare -A MODEL HZ
MODEL[mobilenet]=/home/dk/mobile/rt-infer-bench/models/mobilenetv2-12.onnx;  HZ[mobilenet]=8
MODEL[proxy]="$ROOT/pilot_emc/models/decode_proxy_fp16.onnx";                 HZ[proxy]=8
MODEL[cproxyv2]="$ROOT/pilot_emc/models/compute_proxy_fp16.onnx";             HZ[cproxyv2]=3

TEGRA_PID=""; WL_PID=""
cleanup() {
  [ -n "$WL_PID" ] && kill "$WL_PID" 2>/dev/null || true
  [ -n "$TEGRA_PID" ] && kill "$TEGRA_PID" 2>/dev/null || true
  "$EMC_CTL" unlock >/dev/null || true
  ./part2/set_domain.sh free >/dev/null || true
  nvpmodel -m 2 < /dev/null >/dev/null 2>&1 || true
  chown -R dk:dk "$OUT" 2>/dev/null || true
  echo "[r25] restored MAXN_SUPER: $(nvpmodel -q 2>/dev/null | head -1)"
}
trap cleanup EXIT
trap 'exit 143' TERM INT

log() { echo "[r25 $(date +%H:%M:%S)] $*"; }

log "switching to 25W (mode 1)"
nvpmodel -m 1 < /dev/null || { log "nvpmodel switch FAILED"; exit 1; }
sleep 3
nvpmodel -q | head -2

pin_25w() {
  for c in /sys/devices/system/cpu/cpu*/cpufreq; do
    echo performance > $c/scaling_governor
    echo $CPU_25W > $c/scaling_max_freq
    echo $CPU_25W > $c/scaling_min_freq
  done
  echo 306000000 > $GPU/min_freq
  echo $GPU_25W > $GPU/max_freq
  echo $GPU_25W > $GPU/min_freq
  sleep 1
  log "pinned: cpu0=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq) gpu=$(cat $GPU/cur_freq)"
}

# 1) stock-governor observation in this mode (EMC unlocked)
pin_25w
"$EMC_CTL" unlock >/dev/null
sleep 3
log "governor trace (idle 30s + mobilenet 90s)"
: > "$OUT/governor.tegra"
tegrastats --interval 100 >> "$OUT/governor.tegra" &
TEGRA_PID=$!
sleep 30
timeout -k 30 300 "$PY" -m harness.infer_bench --backend onnxruntime \
  --model "${MODEL[mobilenet]}" --hz 8 --iters 600 --warmup 50 \
  --cpu 5 --prio 80 --label r25_gov_mobilenet --out "$OUT/r25_gov_mobilenet" &
WL_PID=$!; wait "$WL_PID" || true; WL_PID=""
kill "$TEGRA_PID" 2>/dev/null || true; wait "$TEGRA_PID" 2>/dev/null || true
TEGRA_PID=""

# 2) RQ1 replication: 4 EMC points x 3 workloads x 300 iters
for rate in 204000000 665600000 2133000000 3199000000; do
  mhz=$((rate / 1000000))
  "$EMC_CTL" lock "$rate" > "$OUT/emc${mhz}.emc_lock"
  grep -q "actual=$rate" "$OUT/emc${mhz}.emc_lock" || { log "EMC $rate rounded — skip point"; continue; }
  sleep 2
  for wl in mobilenet proxy cproxyv2; do
    label="r25_emc${mhz}_${wl}"
    [ -s "$OUT/$label.json" ] && { log "skip $label"; continue; }
    timeout -k 30 $(( 300 / HZ[$wl] + 600 )) \
      "$PY" -m harness.infer_bench --backend onnxruntime \
      --model "${MODEL[$wl]}" --hz "${HZ[$wl]}" --iters 300 --warmup 50 \
      --cpu 5 --prio 80 --label "$label" --out "$OUT/$label" &
    WL_PID=$!; wait "$WL_PID" || log "!! $label failed"; WL_PID=""
  done
done

log "replication complete"

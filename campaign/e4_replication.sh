#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# E4: RQ2 replication on both SKUs. Addresses the single-run-per-cell
# weakness by repeating the headline burst cell and filling out the NX Part-B
# matrix, so clustering / GPD margins carry run-to-run spread, not point estimates.
# Modeled on the proven cell() in campaign/run_nx_validation.sh (infer_bench runs
# as root here, like the v1 campaign; the harness does mlockall AFTER CUDA init).
#
# Usage: sudo PY=... MOBILENET=... OUT=results/e4_replication ./e4_replication.sh \
#          "label1:emc_hz:adv" "label2:emc:adv" ...
#   spec = label:emc_rate_hz:adv_threads     (100k cycles @ 50 Hz each)
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$ROOT"
[ "$(id -u)" = 0 ] || { echo "run as root" >&2; exit 1; }
PY="${PY:?set PY}"; MOBILENET="${MOBILENET:?set MOBILENET}"
EMC_CTL="$ROOT/pilot_emc/emc_ctl.sh"; MEMBW="$ROOT/tools/membw"
OUT="${OUT:-results/e4_replication}"; mkdir -p "$OUT"
ITERS="${ITERS:-100000}"; HZ="${HZ:-50}"
NCPU=$(nproc); PROBE_CPU=$((NCPU-1))
log(){ echo "[e4 $(date +%H:%M:%S)] $*"; }

# pin CPU + GPU at max, disable RT throttle, drop swap (same hygiene as v1)
for c in /sys/devices/system/cpu/cpu*/cpufreq; do
  echo performance > "$c/scaling_governor"; echo "$(cat "$c/cpuinfo_max_freq")" > "$c/scaling_min_freq"
done
GPU_DIR=$(dirname "$(ls /sys/devices/platform/*.gpu/devfreq/*/available_frequencies 2>/dev/null | head -1)")
GMAX=$(awk '{print $NF}' "$GPU_DIR/available_frequencies"); GMIN=$(awk '{print $1}' "$GPU_DIR/available_frequencies")
echo "$GMAX" > "$GPU_DIR/max_freq"; echo "$GMAX" > "$GPU_DIR/min_freq"
SAVED_RT=$(cat /proc/sys/kernel/sched_rt_runtime_us); echo -1 > /proc/sys/kernel/sched_rt_runtime_us
SWAP=$(swapon --show=NAME --noheadings | head -1); swapoff -a 2>/dev/null && log "swap off"

cleanup(){
  "$EMC_CTL" unlock >/dev/null 2>&1 || true
  echo "$SAVED_RT" > /proc/sys/kernel/sched_rt_runtime_us 2>/dev/null || true
  echo "$GMIN" > "$GPU_DIR/min_freq" 2>/dev/null || true
  for c in /sys/devices/system/cpu/cpu*/cpufreq; do echo schedutil > "$c/scaling_governor" 2>/dev/null || true; done
  [ -n "$SWAP" ] && swapon -a 2>/dev/null || true
  chown -R "$(stat -c %U "$ROOT")" "$OUT" 2>/dev/null || true
  log "cleanup done"
}
trap cleanup EXIT
trap 'exit 143' TERM INT

cell(){ # $1=label $2=emc_hz $3=adv
  local label="$1" emc="$2" adv="$3" o="$OUT/$1"
  if [ -s "$o.json" ]; then log "skip $label (exists)"; return 0; fi
  log "cell $label  emc=$emc adv=$adv iters=$ITERS"
  "$EMC_CTL" lock "$emc" > "$o.emc_lock"
  grep -q "actual=$emc" "$o.emc_lock" || { log "!! $label EMC lock mismatch ($(cat $o.emc_lock))"; return 1; }
  sleep 2
  local AP=""
  if [ "$adv" -gt 0 ]; then
    "$MEMBW" -t "$adv" -c "0,1,2,3" -m 256 -M write -d $((ITERS/HZ+300)) > "$o.membw.json" 2>"$o.membw.log" & AP=$!; sleep 3
  fi
  : > "$o.tegra"; tegrastats --interval 500 >> "$o.tegra" & local TP=$!
  "$PY" -m harness.infer_bench --backend onnxruntime --model "$MOBILENET" \
    --hz "$HZ" --iters "$ITERS" --warmup 200 --cpu "$PROBE_CPU" --prio 80 \
    --label "$label" --out "$o" >/dev/null 2>&1 || log "!! $label workload rc=$?"
  kill "$TP" 2>/dev/null || true; wait "$TP" 2>/dev/null || true
  [ -n "$AP" ] && { kill -TERM "$AP" 2>/dev/null || true; wait "$AP" 2>/dev/null || true; }
  "$EMC_CTL" status > "$o.emc_after"
  [ -s "$o.json" ] && log "OK $label ($(${PY} -c "import json;print('miss=%.4f%%'%(json.load(open('$o.json'))['deadline_miss_rate']*100))" 2>/dev/null))" || log "!! $label NO JSON"
}

for spec in "$@"; do
  IFS=: read -r lbl emc adv <<< "$spec"
  cell "$lbl" "$emc" "$adv" || true
done
log "E4 replication complete"

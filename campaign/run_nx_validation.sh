#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Orin NX 16GB targeted validation for the two-SKU paper revision (v2).
# *** WRITTEN BEFORE THE DEVICE ARRIVED — UNTESTED ON NX. ***
# Everything board-specific is DISCOVERED at runtime and gated:
#   - CPU max freq / core count from cpufreq
#   - GPU devfreq node + available_frequencies (pin at max)
#   - EMC lockable set probed via lock-and-readback over the dvfs_table
# Scope (agreed; do NOT expand):
#   P1  RQ1 curve: lockable EMC points x {mobilenet, proxy, cproxyv2, slm}
#   P2  estimator break: GPU sweep x EMC {max, mid} x {mobilenet, proxy}
#   P3  burst worst-cell: mobilenet @50Hz, EMC mid, 2 adversary threads, 100k
#   P4  actuation lag: EMC + GPU pairs via pilot_trans (TRANS_MATRIX env)
# Prereqs on the NX: same L4T major (R36) as the Nano; onnxruntime-gpu venv;
# proxy models regenerated (make_decode_proxy.py / make_compute_proxy.py);
# Qwen GGUF + llama.cpp built; membw + probes compiled (see tool headers).
#
# Usage: sudo PY=/path/to/ort-python ./campaign/run_nx_validation.sh [P1|P2|P3|P4|ALL]
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
[ "$(id -u)" = 0 ] || { echo "run as root" >&2; exit 1; }
PHASE="${1:-ALL}"

PY="${PY:?set PY to a python with onnxruntime-gpu}"
EMC_CTL="$ROOT/pilot_emc/emc_ctl.sh"
MEMBW="$ROOT/tools/membw"
SLM_BENCH="$ROOT/tools/slm_decode_bench"
SLM_MODEL="${SLM_MODEL:-$HOME/models/qwen2.5-1.5b-instruct-q4_k_m.gguf}"
OUT=results/nx_validation
mkdir -p "$OUT"

# ---------- discovery ----------
MODEL_NAME=$(tr -d '\0' < /proc/device-tree/model)
echo "$MODEL_NAME" | grep -qi "orin nx" || {
  echo "WARNING: device reports '$MODEL_NAME', not Orin NX — continuing anyway" >&2; }
NCPU=$(nproc)
PROBE_CPU=$((NCPU - 1))
CPU_MAX=$(cat /sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq)
CPU_MIN=$(cat /sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_min_freq)
GPU_DIR=$(dirname "$(ls /sys/devices/platform/*.gpu/devfreq/*/available_frequencies 2>/dev/null | head -1)")
[ -n "$GPU_DIR" ] || { echo "no GPU devfreq node found" >&2; exit 1; }
read -ra GPU_FREQS <<< "$(cat "$GPU_DIR/available_frequencies")"
GPU_MAX=${GPU_FREQS[-1]}; GPU_MIN=${GPU_FREQS[0]}

# EMC lockable set: probe every dvfs_table rate, keep exact lock-backs
mapfile -t CAND < <(sudo awk '/vdd_core/ {print $2}' /sys/kernel/debug/bpmp/debug/clk/emc/dvfs_table | sort -un)
LOCKABLE=()
for r in "${CAND[@]}"; do
  got=$("$EMC_CTL" lock "$r" | grep -o 'actual=[0-9]*' | cut -d= -f2)
  [ "$got" = "$r" ] && LOCKABLE+=("$r")
done
"$EMC_CTL" unlock >/dev/null
EMC_MAX=${LOCKABLE[-1]}
EMC_MID=${LOCKABLE[$(( ${#LOCKABLE[@]} / 2 ))]}
{
  echo "model: $MODEL_NAME"; echo "ncpu: $NCPU  cpu_max_khz: $CPU_MAX"
  echo "gpu_dir: $GPU_DIR"; echo "gpu_freqs: ${GPU_FREQS[*]}"
  echo "emc_lockable: ${LOCKABLE[*]}"; echo "emc_max: $EMC_MAX emc_mid: $EMC_MID"
} | tee "$OUT/discovery.txt"

# ---------- clock pinning ----------
TEGRA_PID=""; ADV_PID=""; WL_PID=""
cleanup() {
  for v in WL_PID ADV_PID TEGRA_PID; do
    [ -n "${!v}" ] && kill "${!v}" 2>/dev/null || true
  done
  "$EMC_CTL" unlock >/dev/null 2>&1 || true
  for c in /sys/devices/system/cpu/cpu*/cpufreq; do
    echo schedutil > $c/scaling_governor 2>/dev/null || true
    echo "$CPU_MIN" > $c/scaling_min_freq 2>/dev/null || true
  done
  echo "$GPU_MIN" > "$GPU_DIR/min_freq" 2>/dev/null || true
  echo "$GPU_MAX" > "$GPU_DIR/max_freq" 2>/dev/null || true
  chown -R "$(stat -c %U "$ROOT")" "$OUT" 2>/dev/null || true
  echo "[nx] cleanup done"
}
trap cleanup EXIT
trap 'exit 143' TERM INT
log() { echo "[nx $(date +%H:%M:%S)] $*"; }

pin_all() {
  for c in /sys/devices/system/cpu/cpu*/cpufreq; do
    echo performance > $c/scaling_governor
    echo "$CPU_MAX" > $c/scaling_max_freq
    echo "$CPU_MAX" > $c/scaling_min_freq
  done
  set_gpu "$GPU_MAX"
}
set_gpu() {
  echo "$GPU_MIN" > "$GPU_DIR/min_freq"
  echo "$1" > "$GPU_DIR/max_freq"
  echo "$1" > "$GPU_DIR/min_freq"
  sleep 1
}

declare -A MODEL HZ
MODEL[mobilenet]="${MOBILENET:-$HOME/models/mobilenetv2-12.onnx}"; HZ[mobilenet]=8
MODEL[proxy]="$ROOT/pilot_emc/models/decode_proxy_fp16.onnx";       HZ[proxy]=5
MODEL[cproxyv2]="$ROOT/pilot_emc/models/compute_proxy_fp16.onnx";   HZ[cproxyv2]=3

run_wl() {  # $1=out $2=wl $3=iters $4=hz(optional)
  local out="$1" wl="$2" iters="$3" hz="${4:-}"
  if [ "$wl" = slm ]; then
    timeout -k 60 $((iters + 900)) "$SLM_BENCH" --model "$SLM_MODEL" \
      --tokens "$iters" --warmup 50 --cpu "$PROBE_CPU" --prio 80 --out "$out" &
  else
    [ -n "$hz" ] || hz="${HZ[$wl]}"
    timeout -k 60 $(( (iters+100)/hz + 900 )) \
      "$PY" -m harness.infer_bench --backend onnxruntime \
      --model "${MODEL[$wl]}" --hz "$hz" --iters "$iters" --warmup 100 \
      --cpu "$PROBE_CPU" --prio 80 --label "$(basename "$out")" --out "$out" &
  fi
  WL_PID=$!; wait "$WL_PID"; local rc=$?; WL_PID=""; return $rc
}

cell() {  # $1=label $2=emc $3=adv $4=wl $5=iters $6=hz
  local label="$1" rate="$2" adv="$3" wl="$4" iters="$5" hz="${6:-}"
  local out="$OUT/$label"
  if [ -s "$out.json" ] && { [ "$wl" = slm ] || [ -s "$out.csv" ]; }; then
    log "skip $label"; return 0; fi
  log "cell $label"
  "$EMC_CTL" lock "$rate" > "$out.emc_lock"
  grep -q "actual=$rate" "$out.emc_lock" || { log "!! $label EMC rounded"; return 1; }
  sleep 2
  if [ "$adv" -gt 0 ]; then
    "$MEMBW" -t "$adv" -c "0,1,2,3" -m 256 -M write -d 5400 \
      > "$out.membw.json" 2> "$out.membw.log" &
    ADV_PID=$!; sleep 3
  fi
  : > "$out.tegra"; tegrastats --interval 500 >> "$out.tegra" & TEGRA_PID=$!
  run_wl "$out" "$wl" "$iters" "$hz" || log "!! $label workload failed"
  kill "$TEGRA_PID" 2>/dev/null || true; wait "$TEGRA_PID" 2>/dev/null || true; TEGRA_PID=""
  if [ -n "$ADV_PID" ]; then kill -TERM "$ADV_PID" 2>/dev/null || true; wait "$ADV_PID" 2>/dev/null || true; ADV_PID=""; fi
  "$EMC_CTL" status > "$out.emc_after"
}

pin_all

if [[ "$PHASE" == "ALL" || "$PHASE" == "P1" ]]; then
  log "=== P1: RQ1 curve over ${#LOCKABLE[@]} lockable EMC points ==="
  for rate in "${LOCKABLE[@]}"; do
    mhz=$((rate / 1000000))
    for wl in mobilenet proxy cproxyv2; do
      cell "nx_emc${mhz}_${wl}" "$rate" 0 "$wl" 500 || true
    done
    [ -f "$SLM_MODEL" ] && cell "nx_emc${mhz}_slm" "$rate" 0 slm 300 || true
  done
fi

if [[ "$PHASE" == "ALL" || "$PHASE" == "P2" ]]; then
  log "=== P2: estimator break, GPU sweep x EMC {max,mid} ==="
  for rate in "$EMC_MAX" "$EMC_MID"; do
    mhz=$((rate / 1000000))
    for g in "${GPU_FREQS[@]}"; do
      set_gpu "$g"
      for wl in mobilenet proxy; do
        cell "nx_est_emc${mhz}_gpu$((g/1000000))_${wl}" "$rate" 0 "$wl" 300 || true
      done
    done
    set_gpu "$GPU_MAX"
  done
fi

if [[ "$PHASE" == "ALL" || "$PHASE" == "P3" ]]; then
  log "=== P3: burst worst-cell (100k, EMC mid, 2 adversary) ==="
  echo -1 > /proc/sys/kernel/sched_rt_runtime_us
  cell "nx_burst_emc$((EMC_MID/1000000))_adv2_mobilenet" "$EMC_MID" 2 mobilenet 100000 50 || true
  echo 950000 > /proc/sys/kernel/sched_rt_runtime_us
fi

if [[ "$PHASE" == "ALL" || "$PHASE" == "P4" ]]; then
  log "=== P4: actuation lag (EMC + GPU pairs) ==="
  GPU_MIDF=${GPU_FREQS[$(( ${#GPU_FREQS[@]} / 2 ))]}
  TRANS_MATRIX="emc:$EMC_MAX:$EMC_MID:400,gpu:$GPU_MAX:$GPU_MIDF:400" \
  PROBE_CPU="$PROBE_CPU" CPU_TABLE_MIN="$CPU_MIN" CPU_TABLE_MAX="$CPU_MAX" \
    python3 pilot_trans/run_trans_pilot.py || log "!! P4 failed"
fi

log "NX validation complete — analyze with the standard analyzers + discovery.txt"

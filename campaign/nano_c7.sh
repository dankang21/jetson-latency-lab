#!/bin/bash
# C7: Part-B extension on the Nano — sensitivity-vs-clustering scatter for ViT
# and the L2-resident GEMM (cproxyv2). Run AS ROOT, mirroring e4_replication.sh
# exactly (same FIFO/prio/warmup protocol as the existing Part-B MobileNet cells,
# so clustering statistics are protocol-comparable).
# 8 cells x 100k cycles @ P=40ms (hz25): {vit,cproxyv2} x EMC{2133,3199} x adv{0,2}.
# Env: PY (python3 with onnxruntime) and ONNX_DIR (ONNX models) override the default paths below.
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-/home/dk/mobile/rt-infer-bench/.venv/bin/python3}"
ONNX_DIR="${ONNX_DIR:-/home/dk/mobile/rt-infer-bench/models}"
EMC_CTL="$ROOT/pilot_emc/emc_ctl.sh"; MEMBW="$ROOT/tools/membw"
VIT=$ONNX_DIR/vit_small_patch16_224.onnx
CPX=$ROOT/pilot_emc/models/compute_proxy_fp16.onnx
G=/sys/devices/platform/17000000.gpu/devfreq/17000000.gpu
OUT=$ROOT/results/partb_ext
HZ=25; ITERS=100000; PROBE_CPU=5
cd "$ROOT" || exit 1; mkdir -p "$OUT"
log(){ echo "[$(date +%H:%M:%S)] $*"; }
cleanup(){ "$EMC_CTL" unlock >/dev/null 2>&1
  echo 306000000 > $G/min_freq; echo 1020000000 > $G/max_freq
  echo 950000 > /proc/sys/kernel/sched_rt_runtime_us
  pkill -x tegrastats 2>/dev/null; pkill -x membw 2>/dev/null
  chown -R dk:dk "$OUT" 2>/dev/null; }
trap cleanup EXIT
trap 'exit 143' TERM INT
echo -1 > /proc/sys/kernel/sched_rt_runtime_us
for c in /sys/devices/system/cpu/cpu*/cpufreq; do echo performance > $c/scaling_governor; done
echo 1020000000 > $G/min_freq; echo 1020000000 > $G/max_freq

cell(){ # $1=model_path $2=label $3=emc_hz $4=adv
  local M="$1" label="$2" emc="$3" adv="$4" o="$OUT/$2"
  if [ -s "$o.json" ]; then log "skip $label (exists)"; return 0; fi
  log "cell $label emc=$emc adv=$adv"
  "$EMC_CTL" lock "$emc" > "$o.emc_lock"
  grep -q "actual=$emc" "$o.emc_lock" || { log "!! $label EMC mismatch"; return 1; }
  sleep 2
  local AP=""
  if [ "$adv" -gt 0 ]; then
    "$MEMBW" -t "$adv" -c "0,1,2,3" -m 256 -M write -d $((ITERS/HZ+300)) > "$o.membw.json" 2>"$o.membw.log" & AP=$!; sleep 3
  fi
  : > "$o.tegra"; tegrastats --interval 500 >> "$o.tegra" & local TP=$!
  "$PY" -m harness.infer_bench --backend onnxruntime --model "$M" \
    --hz "$HZ" --iters "$ITERS" --warmup 200 --cpu "$PROBE_CPU" --prio 80 \
    --label "$label" --out "$o" >/dev/null 2>&1 || log "!! $label rc=$?"
  kill "$TP" 2>/dev/null || true; wait "$TP" 2>/dev/null || true
  [ -n "$AP" ] && { kill -TERM "$AP" 2>/dev/null || true; wait "$AP" 2>/dev/null || true; }
  "$EMC_CTL" status > "$o.emc_after"
  [ -s "$o.json" ] && log "OK $label" || log "!! $label NO JSON"
}

for emc in 2133000000 3199000000; do
  for adv in 0 2; do
    tag="emc$(echo $emc | cut -c1-4)_adv${adv}"
    cell "$VIT" "vit_${tag}"  "$emc" "$adv"
    cell "$CPX" "cpx_${tag}"  "$emc" "$adv"
  done
done
echo "NANO_C7 DONE"

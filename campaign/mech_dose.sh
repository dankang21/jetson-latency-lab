#!/bin/bash
# E-D mechanism dose-response (Orin Nano — the inversion platform).
# Question 1 (dose-response): per-kernel GEMM stall-tail vs ALL FOUR lockable EMC
#   points at fixed GPU 1020 (the inversion configuration). Paper currently has
#   only 2133 vs 3199; a monotone curve upgrades "consistent with" materially.
# Question 2 (interaction): does the stall need the top GPU clock? GPU {612,816}
#   at EMC 3199 (1020@3199 comes from Q1).
# Per cell: nsys-profiled cproxyv2 run (ORT, hz4) -> sqlite -> per-kernel durations.
# Env: PY (python3 with onnxruntime) overrides the default path below.
set -u
PY="${PY:-/home/dk/mobile/rt-infer-bench/.venv/bin/python3}"
LAB="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
G=/sys/devices/platform/17000000.gpu/devfreq/17000000.gpu
MODEL=$LAB/pilot_emc/models/compute_proxy_fp16.onnx
OUT=$LAB/results/mech_dose
cd "$LAB" || exit 1; mkdir -p "$OUT"
restore(){ sudo bash -c "$LAB/pilot_emc/emc_ctl.sh unlock >/dev/null 2>&1
  echo 306000000 > $G/min_freq; echo 1020000000 > $G/max_freq
  chown -R dk:dk $OUT 2>/dev/null"; }
trap restore EXIT
setgpu(){ sudo bash -c "echo ${1}000000 > $G/min_freq; echo ${1}000000 > $G/max_freq"; }

cell(){  # cell <emc_hz> <gpu_mhz> <label>
  local EMC=$1 GPU=$2 L=$3
  sudo bash -c "$LAB/pilot_emc/emc_ctl.sh lock $EMC >/dev/null"
  setgpu "$GPU"; sleep 0.5
  # tegrastats ground truth for the cell (clock really held?)
  sudo timeout 55 tegrastats --interval 500 > "$OUT/$L.tegra" 2>/dev/null &
  nsys profile -o "$OUT/$L" --force-overwrite true \
    "$PY" -m harness.infer_bench --backend onnxruntime --model "$MODEL" \
    --hz 4 --iters 170 --warmup 20 --cpu 5 --label "$L" \
    --out "$OUT/${L}_bench" >/dev/null 2>&1
  sudo pkill -x tegrastats 2>/dev/null
  nsys export --type sqlite --force-overwrite true -o "$OUT/$L.sqlite" "$OUT/$L.nsys-rep" >/dev/null 2>&1
  echo -n "  $L: "
  "$PY" campaign/mech_kdur_extract.py "$OUT/$L.sqlite" "$OUT/kdur_$L.txt" || echo "EXTRACT FAIL"
  rm -f "$OUT/$L.nsys-rep" "$OUT/$L.sqlite"   # keep only kdur + bench + tegra (reports are large)
}

echo "=== Q1: EMC dose-response @ GPU 1020 ==="
cell  204000000 1020 e204_g1020
cell  665600000 1020 e665_g1020
cell 2133000000 1020 e2133_g1020
cell 3199000000 1020 e3199_g1020

echo "=== Q2: GPU-clock interaction @ EMC 3199 ==="
cell 3199000000  816 e3199_g816
cell 3199000000  612 e3199_g612

echo "MECH_DOSE DONE"

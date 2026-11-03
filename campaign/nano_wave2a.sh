#!/bin/bash
# Nano Wave-2a (user mode; sudo only for clocks):
#   A5: timestamped kernel events at the inversion corner (event statistics)
#   B4: 3-rep nsys at GPU {918,1020} @ EMC 3199 (onset boundary + CIs)
#   B2-full: Nano companion governor grid (MNv2 + ViT), sweeps + 3-rep deploys + INA
# Env: PY (python3 with onnxruntime) and ONNX_DIR (ONNX models) override the default paths below.
set -u
PY="${PY:-/home/dk/mobile/rt-infer-bench/.venv/bin/python3}"
ONNX_DIR="${ONNX_DIR:-/home/dk/mobile/rt-infer-bench/models}"
LAB="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MN=$ONNX_DIR/mobilenetv2-12.onnx
VIT=$ONNX_DIR/vit_small_patch16_224.onnx
CPX=$LAB/pilot_emc/models/compute_proxy_fp16.onnx
G=/sys/devices/platform/17000000.gpu/devfreq/17000000.gpu
cd "$LAB" || exit 1
mkdir -p results/mech_dose results/nano_grid
restore(){ sudo bash -c "$LAB/pilot_emc/emc_ctl.sh unlock >/dev/null 2>&1
  echo 306000000 > $G/min_freq; echo 1020000000 > $G/max_freq
  echo 950000 > /proc/sys/kernel/sched_rt_runtime_us
  chown -R dk:dk results/mech_dose results/nano_grid 2>/dev/null"; }
trap restore EXIT
setclk(){ sudo bash -c "echo ${1}000000 > $G/min_freq; echo ${1}000000 > $G/max_freq"; }
sudo bash -c "echo -1 > /proc/sys/kernel/sched_rt_runtime_us"

nsys_cell(){ # nsys_cell <emc_hz> <gpu> <label> <extractor>
  local EMC=$1 GPU=$2 L=$3 EX=$4
  sudo bash -c "$LAB/pilot_emc/emc_ctl.sh lock $EMC >/dev/null"
  setclk "$GPU"; sleep 0.5
  nsys profile -o "results/mech_dose/$L" --force-overwrite true \
    "$PY" -m harness.infer_bench --backend onnxruntime --model "$CPX" \
    --hz 4 --iters 170 --warmup 20 --cpu 5 --label "$L" \
    --out "results/mech_dose/${L}_bench" >/dev/null 2>&1
  nsys export --type sqlite --force-overwrite true -o "results/mech_dose/$L.sqlite" "results/mech_dose/$L.nsys-rep" >/dev/null 2>&1
  echo -n "  $L: "
  "$PY" "campaign/$EX" "results/mech_dose/$L.sqlite" \
    "results/mech_dose/$([ "$EX" = mech_kdur_ts.py ] && echo ts_$L.csv || echo kdur_$L.txt)" || echo "EXTRACT FAIL"
  rm -f "results/mech_dose/$L.nsys-rep" "results/mech_dose/$L.sqlite"
}

echo "=== A5: timestamped events (inversion corner + control) ==="
nsys_cell 3199000000 1020 ts_e3199_g1020 mech_kdur_ts.py
nsys_cell 2133000000 1020 ts_e2133_g1020 mech_kdur_ts.py

echo "=== B4: onset boundary 3-rep (GPU 918, 1020 @ EMC 3199) ==="
for r in 1 2 3; do
  nsys_cell 3199000000  918 e3199_g918_rep${r} mech_kdur_extract.py
  nsys_cell 3199000000 1020 e3199_g1020_rep${r} mech_kdur_extract.py
done

echo "=== B2-full: Nano companion grid (MNv2 hz50, ViT hz10) ==="
sudo bash -c "for c in /sys/devices/system/cpu/cpu*/cpufreq; do echo performance > \$c/scaling_governor; done"
grid(){ # grid <name> <model> <hz> <ina_dur>
  local NAME=$1 M=$2 HZ=$3 INA=$4
  mkdir -p "results/nano_grid/${NAME}"
  for EMC in 3199 2133; do
    sudo bash -c "$LAB/pilot_emc/emc_ctl.sh lock ${EMC}000000 >/dev/null"
    for f in 306 408 510 612 714 816 918 1020; do
      setclk "$f"; sleep 0.4
      "$PY" -m harness.infer_bench --backend onnxruntime --model "$M" \
        --hz "$HZ" --iters 800 --warmup 60 --cpu 5 --label ng_${NAME}_e${EMC}_g${f} \
        --out "results/nano_grid/${NAME}/est_emc${EMC}_gpu${f}" >/dev/null 2>&1
      echo "  [$NAME] sweep emc${EMC} gpu${f}: $([ -s results/nano_grid/${NAME}/est_emc${EMC}_gpu${f}.json ] && echo OK || echo FAIL)"
    done
  done
  sudo bash -c "$LAB/pilot_emc/emc_ctl.sh lock 2133000000 >/dev/null"
  for f in 510 612 714 816 918 1020; do
    for r in 1 2 3; do
      setclk "$f"; sleep 0.4
      taskset -c 0 "$PY" campaign/ina_log.py "results/nano_grid/${NAME}/g${f}_r${r}.power.csv" "$INA" & local LP=$!
      "$PY" -m harness.infer_bench --backend onnxruntime --model "$M" \
        --hz "$HZ" --iters 1000 --warmup 60 --cpu 5 --label ngd_${NAME}_g${f}_r${r} \
        --out "results/nano_grid/${NAME}/g${f}_r${r}" >/dev/null 2>&1
      kill -TERM $LP 2>/dev/null; wait $LP 2>/dev/null
    done
    echo "  [$NAME] deploy g${f} x3: $([ -s results/nano_grid/${NAME}/g${f}_r3.json ] && echo OK || echo FAIL)"
  done
}
grid mobilenet "$MN" 50 40
grid vit "$VIT" 10 130
echo "NANO_WAVE2A DONE"

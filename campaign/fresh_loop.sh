#!/bin/bash
# Fresh-probe end-to-end admission-loop validation (NX):
# 8 trials x (fresh co-runner launch -> fresh probe -> guarded pick ->
# disjoint production contract) for the two co-tenancy acceptances.
set -u
PY=/home/dk/mobile/rt-infer-bench/.venv/bin/python3
LAB=/home/dk/mobile/jetson-latency-lab
MN=/home/dk/mobile/rt-infer-bench/models/mobilenetv2-12.onnx
VIT=/home/dk/mobile/rt-infer-bench/models/vit_small_patch16_224.onnx
BENCH=$LAB/tools/slm_decode_bench
MD=/home/dk/mobile/models
G=/sys/devices/platform/17000000.gpu/devfreq/17000000.gpu
cd "$LAB" || exit 1; mkdir -p results/fresh_loop
CO_PID=""
restore(){ [ -n "$CO_PID" ] && kill "$CO_PID" 2>/dev/null
  sudo bash -c "$LAB/pilot_emc/emc_ctl.sh unlock >/dev/null 2>&1
  echo 306000000 > $G/min_freq; echo 1173000000 > $G/max_freq
  echo 950000 > /proc/sys/kernel/sched_rt_runtime_us
  for c in /sys/devices/system/cpu/cpu*/cpufreq; do echo schedutil > \$c/scaling_governor; done
  chown -R dk:dk results/fresh_loop 2>/dev/null"; }
trap restore EXIT
setclk(){ sudo bash -c "echo ${1}000000 > $G/min_freq; echo ${1}000000 > $G/max_freq"; }
sudo bash -c "echo -1 > /proc/sys/kernel/sched_rt_runtime_us
for c in /sys/devices/system/cpu/cpu*/cpufreq; do echo performance > \$c/scaling_governor; done"
sudo bash -c "$LAB/pilot_emc/emc_ctl.sh lock 2133000000 >/dev/null"
co_start(){ "$PY" -m harness.infer_bench --backend onnxruntime --model "$VIT" \
  --hz 10 --iters 100000 --warmup 30 --cpu 5 --label co_vit \
  --out /tmp/co_fl >/dev/null 2>&1 & CO_PID=$!; sleep 20; }
co_stop(){ [ -n "$CO_PID" ] && kill "$CO_PID" 2>/dev/null; wait "$CO_PID" 2>/dev/null; CO_PID=""; sleep 2; }

echo "trial,arm,Mg,pick,extra" > results/fresh_loop/decisions.csv
echo "=== ARM 1: vision D=12 guarded, 8 end-to-end trials ==="
for t in 1 2 3 4 5 6 7 8; do
  co_start
  setclk 1173; sleep 0.4
  "$PY" -m harness.infer_bench --backend onnxruntime --model "$MN" \
    --hz 33 --iters 200 --warmup 60 --cpu 7 --label flv_probe_t${t} \
    --out results/fresh_loop/vprobe_t${t} >/dev/null 2>&1
  read MG PICK <<< "$("$PY" campaign/fresh_pick.py vision results/fresh_loop/vprobe_t${t}.csv)"
  setclk "$PICK"; sleep 0.4
  "$PY" -m harness.infer_bench --backend onnxruntime --model "$MN" \
    --hz 33 --iters 1000 --warmup 60 --cpu 7 --label flv_dep_t${t} \
    --out results/fresh_loop/vdep_t${t}_g${PICK} >/dev/null 2>&1
  co_stop
  echo "${t},vision,${MG},${PICK}," >> results/fresh_loop/decisions.csv
  echo "  t${t}: Mg=${MG}us pick=${PICK} $([ -s results/fresh_loop/vdep_t${t}_g${PICK}.json ] && echo OK || echo FAIL)"
done

echo "=== ARM 2: joint decode, 8 end-to-end trials ==="
for t in 1 2 3 4 5 6 7 8; do
  co_start
  setclk 1173; sleep 0.5
  "$BENCH" --model "$MD/qwen2.5-1.5b-instruct-q4_k_m.gguf" \
    --tokens 200 --warmup 40 --cpu 7 --prio 80 \
    --out results/fresh_loop/dprobe_t${t} >/dev/null 2>&1
  read MG DACC DSTAR <<< "$("$PY" campaign/fresh_pick.py decode results/fresh_loop/dprobe_t${t}.csv)"
  "$BENCH" --model "$MD/qwen2.5-1.5b-instruct-q4_k_m.gguf" \
    --tokens 2000 --warmup 40 --cpu 7 --prio 80 \
    --out results/fresh_loop/ddep_t${t}_D${DACC} >/dev/null 2>&1
  co_stop
  echo "${t},decode,${MG},${DACC},${DSTAR}" >> results/fresh_loop/decisions.csv
  echo "  t${t}: Mg=${MG}ms D_accept=${DACC} (D*=${DSTAR}) $([ -s results/fresh_loop/ddep_t${t}_D${DACC}.json ] && echo OK || echo FAIL)"
done
echo "FRESH_LOOP DONE"

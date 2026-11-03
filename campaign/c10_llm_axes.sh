#!/bin/bash
# C10: LLM axes on the NX — the edge-LLM citation payload.
#   SIZE : Qwen2.5 {0.5B, 1.5B, 3B} Q4_K_M x EMC {2133, 3199} x 3 reps
#   QUANT: 1.5B Q8_0 x EMC {2133, 3199} x 3 reps (pre-registered prediction:
#          penalty moves from Q4's +18.7% toward the GEMV anchor's +44.5%)
#   CTX  : 1.5B Q4, --prompt-tokens {64, 4096} x EMC {2133,3199} x 3 reps
#   PREFILL: llama-bench -p 512 -n 128 at both EMC points (pp/tg split)
# GPU pinned 1173; per-token csv + INA3221 VDD_IN log (median 3.6 ms per sample) on every slm run.
# Env: PY (python3 with onnxruntime), GGUF_DIR (GGUF models) and LLAMA_CPP (llama.cpp tree) override the default paths below.
set -u
PY="${PY:-/home/dk/mobile/rt-infer-bench/.venv/bin/python3}"
LLAMA_CPP="${LLAMA_CPP:-/home/dk/mobile/llama.cpp}"
LAB="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BENCH=$LAB/tools/slm_decode_bench
LB=$LLAMA_CPP/build/tools/llama-bench/llama-bench
[ -x "$LB" ] || LB=$(find "$LLAMA_CPP" -name llama-bench -type f -executable 2>/dev/null | head -1)
MD="${GGUF_DIR:-/home/dk/mobile/models}"
G=/sys/devices/platform/17000000.gpu/devfreq/17000000.gpu
cd "$LAB" || exit 1; mkdir -p results/llm_axes
restore(){ sudo bash -c "$LAB/pilot_emc/emc_ctl.sh unlock >/dev/null 2>&1
  echo 306000000 > $G/min_freq; echo 1173000000 > $G/max_freq
  echo 950000 > /proc/sys/kernel/sched_rt_runtime_us
  for c in /sys/devices/system/cpu/cpu*/cpufreq; do echo schedutil > \$c/scaling_governor; done
  chown -R dk:dk results/llm_axes 2>/dev/null"; }
trap restore EXIT
sudo bash -c "echo -1 > /proc/sys/kernel/sched_rt_runtime_us
for c in /sys/devices/system/cpu/cpu*/cpufreq; do echo performance > \$c/scaling_governor; done
echo 1173000000 > $G/min_freq; echo 1173000000 > $G/max_freq"

run(){ # run <tag> <model> <emc> <rep> <extra args...>
  local TAG=$1 M=$2 EMC=$3 R=$4; shift 4
  sudo bash -c "$LAB/pilot_emc/emc_ctl.sh lock ${EMC}000000 >/dev/null"; sleep 0.5
  taskset -c 0 "$PY" campaign/ina_log.py "results/llm_axes/${TAG}_e${EMC}_r${R}.power.csv" 150 & local LP=$!
  "$BENCH" --model "$M" --tokens 300 --warmup 40 --cpu 7 --prio 80 "$@" \
    --out "results/llm_axes/${TAG}_e${EMC}_r${R}" >/dev/null 2>&1
  kill -TERM $LP 2>/dev/null; wait $LP 2>/dev/null
  echo "  ${TAG} e${EMC} r${R}: $([ -s results/llm_axes/${TAG}_e${EMC}_r${R}.json ] && echo OK || echo FAIL)"
}

echo "=== SIZE axis ==="
for EMC in 3199 2133; do
  for R in 1 2 3; do
    run q05  "$MD/qwen2.5-0.5b-instruct-q4_k_m.gguf" $EMC $R
    run q15  "$MD/qwen2.5-1.5b-instruct-q4_k_m.gguf" $EMC $R
    run q3   "$MD/qwen2.5-3b-instruct-q4_k_m.gguf"  $EMC $R
  done
done
echo "=== QUANT axis (1.5B Q8_0) ==="
for EMC in 3199 2133; do for R in 1 2 3; do
  run q15q8 "$MD/qwen2.5-1.5b-instruct-q8_0.gguf" $EMC $R
done; done
echo "=== CONTEXT axis (1.5B Q4, prompt 64 vs 4096) ==="
for EMC in 3199 2133; do for R in 1 2 3; do
  run q15p64   "$MD/qwen2.5-1.5b-instruct-q4_k_m.gguf" $EMC $R --prompt-tokens 64
  run q15p4096 "$MD/qwen2.5-1.5b-instruct-q4_k_m.gguf" $EMC $R --prompt-tokens 4096
done; done
echo "=== PREFILL (llama-bench pp512/tg128) ==="
if [ -x "$LB" ]; then
  for EMC in 3199 2133; do
    sudo bash -c "$LAB/pilot_emc/emc_ctl.sh lock ${EMC}000000 >/dev/null"; sleep 0.5
    "$LB" -m "$MD/qwen2.5-1.5b-instruct-q4_k_m.gguf" -p 512 -n 128 -ngl 99 -o csv \
      > "results/llm_axes/prefill_e${EMC}.csv" 2>/dev/null
    echo "  prefill e${EMC}: $([ -s results/llm_axes/prefill_e${EMC}.csv ] && echo OK || echo FAIL)"
  done
else
  echo "  llama-bench not found: prefill skipped"
fi
echo "C10_LLM_AXES DONE"

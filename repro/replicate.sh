#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Community replication kit (~20 min on a Jetson Orin Nano Super).
#
# Reproduces the paper's two most scrutinized RQ1 results on YOUR board:
#   (1) the EMC-frequency latency shift (MobileNetV2, 2133 vs 3199 MHz)
#   (2) the L2-resident GEMM inversion at the top GPU clock
#       (faster at 2133 than 3199 MHz on our unit -- does YOUR unit agree?)
#
# Requirements: JetPack 6.x (L4T R36), root, a python with onnxruntime-gpu
# (https://pypi.jetson-ai-lab.io wheels or NVIDIA's), and a python with the
# `onnx` package to generate the 2 MiB proxy model.
#
# Usage:
#   sudo PY=/path/to/python-with-ort GENPY=/path/to/python-with-onnx \
#        ./repro/replicate.sh
# Output: repro/replication_result.json  -- please open an issue with it!
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
[ "$(id -u)" = 0 ] || { echo "run as root (clock control)" >&2; exit 1; }
PY="${PY:?set PY to a python with onnxruntime-gpu}"
GENPY="${GENPY:-$PY}"
OUT=repro/replication_out
mkdir -p "$OUT"

MOBILENET="$OUT/mobilenetv2-12.onnx"
[ -f "$MOBILENET" ] || wget -q -O "$MOBILENET" \
  "https://github.com/onnx/models/raw/main/validated/vision/classification/mobilenet/model/mobilenetv2-12.onnx" \
  || { echo "mobilenet download failed"; exit 1; }
CPROXY="$ROOT/pilot_emc/models/compute_proxy_fp16.onnx"
[ -f "$CPROXY" ] || "$GENPY" pilot_emc/make_compute_proxy.py

cleanup() {
  ./pilot_emc/emc_ctl.sh unlock >/dev/null 2>&1 || true
  ./part2/set_domain.sh free >/dev/null 2>&1 || true
}
trap cleanup EXIT
trap 'exit 143' TERM INT

./part2/set_domain.sh all >/dev/null || { echo "clock pinning failed (paths are Orin Nano specific; see part2/set_domain.sh)"; exit 1; }

run_cell() {  # $1=emc_rate $2=label $3=model $4=hz
  ./pilot_emc/emc_ctl.sh lock "$1" > "$OUT/$2.emc_lock"
  grep -q "actual=$1" "$OUT/$2.emc_lock" || { echo "EMC $1 not lockable on this board"; return 1; }
  sleep 2
  "$PY" -m harness.infer_bench --backend onnxruntime --model "$3" \
    --hz "$4" --iters 300 --warmup 50 --cpu 5 --prio 80 \
    --label "$2" --out "$OUT/$2"
}

for emc in 2133000000 3199000000; do
  mhz=$((emc / 1000000))
  run_cell "$emc" "rep_emc${mhz}_mobilenet" "$MOBILENET" 8 || true
  run_cell "$emc" "rep_emc${mhz}_cproxyv2" "$CPROXY" 3 || true
done

python3 - <<'EOF'
import json
from pathlib import Path
out = Path("repro/replication_out")
r = {}
for f in out.glob("rep_*.json"):
    r[f.stem] = json.loads(f.read_text())["compute_us"]["p50"]
res = {"board": Path("/proc/device-tree/model").read_text().strip("\x00"),
       "l4t": Path("/etc/nv_tegra_release").read_text().split(",")[0],
       "p50_us": r}
def shift(wl):
    a, b = r.get(f"rep_emc2133_{wl}"), r.get(f"rep_emc3199_{wl}")
    return round((a - b) / b * 100, 1) if a and b else None
res["mobilenet_shift_2133_vs_3199_pct"] = shift("mobilenet")
res["cproxyv2_shift_2133_vs_3199_pct"] = shift("cproxyv2")
res["inversion_reproduced"] = (shift("cproxyv2") or 0) < 0
Path("repro/replication_result.json").write_text(json.dumps(res, indent=2))
print(json.dumps(res, indent=2))
print("\nPaper's unit: mobilenet +11.3%, cproxyv2 -9.1% (inversion).")
print("Please share repro/replication_result.json via a GitHub issue!")
EOF

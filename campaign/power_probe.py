#!/usr/bin/env python3
# Feasibility probe: can the onboard INA3221 resolve per-inference power?
# Logs VDD_IN (+CPU_GPU_CV, SOC) at max rate while a mobilenet burst runs.
import time, subprocess, os, sys
H = "/sys/devices/platform/bus@0/c240000.i2c/i2c-1/1-0040/hwmon/hwmon1"
def rail(i):  # power mW = V(mV)*I(mA)/1000
    return int(open(f"{H}/in{i}_input").read()) * int(open(f"{H}/curr{i}_input").read()) / 1000.0
PY = "/home/dk/mobile/rt-infer-bench/.venv/bin/python3"
MODEL = "/home/dk/mobile/rt-infer-bench/models/mobilenetv2-12.onnx"
log = []
t0 = time.time()
proc = None
launched = False
while time.time() - t0 < 8.0:
    t = time.time() - t0
    log.append((t, rail(1)))
    if t > 2.0 and not launched:  # start a 4s back-to-back inference burst
        proc = subprocess.Popen([PY, "-m", "harness.infer_bench", "--backend", "onnxruntime",
            "--model", MODEL, "--hz", "1000", "--iters", "2000", "--warmup", "50",
            "--cpu", "7", "--label", "pprobe", "--out", "/tmp/pprobe"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, cwd="/home/dk/mobile/jetson-latency-lab")
        launched = True
if proc: proc.wait()
import statistics as st
ts = [r[0] for r in log]; ps = [r[1] for r in log]
dts = [ts[i+1]-ts[i] for i in range(len(ts)-1)]
idle = [p for t, p in log if t < 1.8]
active = [p for t, p in log if 3.0 < t < 5.5]
print(f"samples={len(log)} over {ts[-1]:.1f}s -> {len(log)/ts[-1]:.0f} Hz; median dt={st.median(dts)*1000:.2f}ms")
print(f"distinct power values: {len(set(ps))} (of {len(ps)} samples)")
print(f"idle power (t<1.8s):   mean {st.mean(idle):.0f} mW, sd {st.pstdev(idle):.0f}, n={len(idle)}")
print(f"active power (3-5.5s): mean {st.mean(active):.0f} mW, sd {st.pstdev(active):.0f}, n={len(active)}")
print(f"active-idle delta: {st.mean(active)-st.mean(idle):.0f} mW  (clear separation = INA tracks load)")
# how fast does it ramp 10->90% on the load edge near t=2.0?
edge = [(t, p) for t, p in log if 1.9 < t < 2.6]
lo, hi = st.mean(idle), st.mean(active)
cross = [t for t, p in edge if p > lo + 0.5*(hi-lo)]
if cross: print(f"load-edge crossing ~{cross[0]-2.0:.3f}s after burst start (rise time)")

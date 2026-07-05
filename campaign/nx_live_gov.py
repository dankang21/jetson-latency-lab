#!/usr/bin/env python3
# Live closed-loop governor experiments on the NX. RUN AS ROOT (clock writes).
#  --mode calib     C1 step 2: blind k/f+b fit + EWMA offset calibrator; repick
#                   every cycle; GPU clock switched live via devfreq. Logs the
#                   full clock trajectory (convergence, transient misses,
#                   oscillation) — the live counterpart of the trace replay.
#  --mode emcswitch C2: fixed GPU clock at an infeasible 2133 cell; on the first
#                   deadline miss, raise EMC 2133->3199 by direct BPMP rate
#                   write (mrq_rate_locked+bwmgr_halt must already be set by the
#                   launcher). Logs per-cycle EMC readback -> recovery latency.
import time, csv, argparse
import numpy as np
import onnxruntime as ort

G = "/sys/devices/platform/17000000.gpu/devfreq/17000000.gpu"
EMC = "/sys/kernel/debug/bpmp/debug/clk/emc/rate"
def wr(p, v): open(p, "w").write(str(v))
def set_gpu(mhz):
    wr(f"{G}/min_freq", mhz*1000000); wr(f"{G}/max_freq", mhz*1000000)
def rd_emc():
    try: return int(open(EMC).read())
    except Exception: return -1

ap = argparse.ArgumentParser()
ap.add_argument("--mode", required=True, choices=["calib", "emcswitch"])
ap.add_argument("--model", required=True)
ap.add_argument("--hz", type=float, default=33)
ap.add_argument("--cycles", type=int, default=2000)
ap.add_argument("--deadline-ms", type=float, required=True)
ap.add_argument("--k", type=float, default=3207403.0)  # blind fit @3199 (us, MHz)
ap.add_argument("--b", type=float, default=1325.0)
ap.add_argument("--alpha", type=float, default=0.1)
ap.add_argument("--gpu", type=int, default=918)        # emcswitch fixed clock
ap.add_argument("--arm", default="trigger", choices=["trigger", "static2133", "static3199"])
ap.add_argument("--out", required=True)
a = ap.parse_args()

sess = ort.InferenceSession(a.model, providers=["CUDAExecutionProvider"])
inp = sess.get_inputs()[0]
shape = [1 if (d is None or isinstance(d, str)) else d for d in inp.shape]
x = np.random.default_rng(42).standard_normal(shape).astype(np.float32)
feed = {inp.name: x}
for _ in range(80): sess.run(None, feed)

P = 1e9/a.hz; D_us = a.deadline_ms*1000
G_LIST = [612, 714, 816, 918, 1020, 1122, 1173]
def pick(off):
    for f in G_LIST:
        if (a.k/f + a.b + off)*1.035 <= D_us: return f
    return G_LIST[-1]

def wait_until(t_ns):
    rem = t_ns - time.monotonic_ns()
    if rem > 3_000_000: time.sleep((rem - 2_000_000)/1e9)
    while time.monotonic_ns() < t_ns: pass

rows = []
if a.mode == "calib":
    off = 0.0; cur = pick(0.0); set_gpu(cur); time.sleep(0.1)
    t0 = time.monotonic_ns() + int(P)
    for i in range(a.cycles):
        sched = t0 + int(i*P)
        wait_until(sched)
        sess.run(None, feed)
        resp = (time.monotonic_ns() - sched)/1000.0
        miss = 1 if resp > D_us else 0
        off = (1 - a.alpha)*off + a.alpha*(resp - (a.k/cur + a.b))
        new = pick(off); sw = 0
        if new != cur:
            set_gpu(new); sw = 1; cur = new   # actuates in the post-completion slack
        rows.append((i, cur, f"{resp:.1f}", miss, sw, f"{off:.1f}"))
    hdr = "cycle,gpu_mhz,response_us,miss,switched,offset_us"
else:
    set_gpu(a.gpu); time.sleep(0.1)
    if a.arm == "static3199": wr(EMC, 3199000000); time.sleep(0.1)
    switched = (a.arm != "trigger")
    t0 = time.monotonic_ns() + int(P)
    for i in range(a.cycles):
        sched = t0 + int(i*P)
        wait_until(sched)
        sess.run(None, feed)
        resp = (time.monotonic_ns() - sched)/1000.0
        miss = 1 if resp > D_us else 0
        sw = 0
        if a.arm == "trigger" and miss and not switched:
            wr(EMC, 3199000000)               # next-release actuation, in slack
            switched = True; sw = 1
        rows.append((i, a.gpu, f"{resp:.1f}", miss, sw, rd_emc()))
    hdr = "cycle,gpu_mhz,response_us,miss,switched,emc_hz"

with open(a.out, "w") as f:
    f.write(hdr + "\n")
    for r in rows: f.write(",".join(map(str, r)) + "\n")
n_miss = sum(int(r[3]) for r in rows)
print(f"{a.mode} arm={a.arm} D={a.deadline_ms} cycles={a.cycles} misses={n_miss} "
      f"switches={sum(int(r[4]) for r in rows)} -> {a.out}")

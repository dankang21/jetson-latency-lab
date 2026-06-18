#!/usr/bin/env python3
# Max-rate onboard INA3221 logger: VDD_IN (+CPU_GPU_CV, SOC) power in mW with
# wall-clock timestamps. Board-agnostic (locates the ina3221 hwmon). Stops on
# SIGTERM or after the optional duration. Usage: ina_log.py <out.csv> [dur_s]
import sys, time, signal, glob
H = None
for h in glob.glob('/sys/class/hwmon/hwmon*'):
    try:
        if open(h + '/name').read().strip() == 'ina3221':
            H = h; break
    except Exception:
        pass
if H is None:
    sys.stderr.write('no ina3221 hwmon found\n'); sys.exit(1)
out = sys.argv[1]
dur = float(sys.argv[2]) if len(sys.argv) > 2 else 3600.0
run = [True]
signal.signal(signal.SIGTERM, lambda *a: run.__setitem__(0, False))
signal.signal(signal.SIGINT, lambda *a: run.__setitem__(0, False))
f = open(out, 'w'); f.write('t,vdd_in_mw,cpugpu_mw,soc_mw\n')
rd = lambda p: int(open(H + '/' + p).read())
t0 = time.time()
while run[0] and time.time() - t0 < dur:
    try:
        p1 = rd('in1_input') * rd('curr1_input') / 1000.0
        p2 = rd('in2_input') * rd('curr2_input') / 1000.0
        p3 = rd('in3_input') * rd('curr3_input') / 1000.0
        f.write(f'{time.time()-t0:.4f},{p1:.1f},{p2:.1f},{p3:.1f}\n')
    except Exception:
        pass
f.close()

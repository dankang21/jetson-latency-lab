#!/usr/bin/env python3
# Calibrated per-inference energy from onboard INA3221 VDD_IN logs (median 3.6-3.9 ms per sample).
#  - governor cells (periodic deployment duty): E/inf = mean VDD_IN power / release rate
#  - inversion cells (saturated, continuously busy): E/inf = mean active power x compute latency
# Run with /usr/bin/python3 after the energy_cal campaigns + rsync.
import csv, json, os, statistics as st, glob, re
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # results/... paths are repo-relative
R = "results/energy_cal"

def meanpow(tag, rail="vdd_in_mw"):
    f = f"{R}/{tag}.power.csv"
    if not os.path.exists(f): return None
    p = [float(r[rail]) for r in csv.DictReader(open(f)) if r.get(rail)]
    if len(p) < 20: return None
    n = len(p); p = p[n//10:-n//10]          # trim warmup/edge
    return st.mean(p)

def compute_ms(tag):
    f = f"{R}/{tag}.json"
    if not os.path.exists(f): return None
    d = json.load(open(f)); return d["compute_us"]["p50"] / 1000.0

def reps(prefix):
    return sorted(set(re.search(r'_r(\d+)$', os.path.basename(x)[:-10]).group(1)
                      for x in glob.glob(f"{R}/{prefix}_r*.power.csv")))

# ---------- GOVERNOR (periodic) : E/inf = mean_power / rate ----------
def gov_energy(wl, freq, rate):
    es = []
    for rp in reps(f"gov_{wl}_g{freq}"):
        mp = meanpow(f"gov_{wl}_g{freq}_r{rp}")
        if mp: es.append(mp / rate)          # mW / (1/s) = mJ
    return (st.mean(es), st.pstdev(es), len(es)) if es else None

print("=== GOVERNOR calibrated energy (E/inf = mean VDD_IN power / rate) ===")
GOV = {
  "mobilenet": dict(rate=33, deadlines=[(4.6,1122,1173),(4.8,1020,1173),(5.0,918,1122)]),
  "vit":       dict(rate=20, deadlines=[(13.6,918,1122),(14.2,816,1020),(15.0,816,918)]),
}
for wl, cfg in GOV.items():
    print(f"-- {wl} (rate {cfg['rate']}Hz) --")
    for D, fb, fa in cfg["deadlines"]:
        eb, ea = gov_energy(wl, fb, cfg["rate"]), gov_energy(wl, fa, cfg["rate"])
        if eb and ea:
            print(f"   D={D}ms blind {fb}MHz {eb[0]:.1f}mJ | aware {fa}MHz {ea[0]:.1f}mJ "
                  f"| dE={ea[0]-eb[0]:+.1f}mJ (aware {'lower' if ea[0]<eb[0] else 'higher'})")
        else:
            print(f"   D={D}ms: missing cells (blind {fb}:{bool(eb)} aware {fa}:{bool(ea)})")

# ---------- INVERSION (saturated) : E/inf = active power x compute latency ----------
print("\n=== INVERSION / EMC calibrated energy (E/inf = active power x compute latency) ===")
def inv_energy(wl, emc):
    es, ps, ls = [], [], []
    for rp in reps(f"inv_{wl}_emc{emc}"):
        mp, lat = meanpow(f"inv_{wl}_emc{emc}_r{rp}"), compute_ms(f"inv_{wl}_emc{emc}_r{rp}")
        if mp and lat: es.append(mp/1000.0*lat); ps.append(mp); ls.append(lat)  # mW*ms/1000=mJ
    return (st.mean(es), st.pstdev(es), st.mean(ps), st.mean(ls), len(es)) if es else None
for wl in ("cproxyv2", "mobilenet"):
    a, b = inv_energy(wl, 2133), inv_energy(wl, 3199)
    if a and b:
        print(f"-- {wl} --")
        print(f"   EMC 2133: {a[2]:.0f}mW x {a[3]:.2f}ms = {a[0]:.1f}±{a[1]:.1f}mJ (n={a[4]})")
        print(f"   EMC 3199: {b[2]:.0f}mW x {b[3]:.2f}ms = {b[0]:.1f}±{b[1]:.1f}mJ (n={b[4]})")
        print(f"   -> 3199 energy {100*(b[0]-a[0])/a[0]:+.1f}% ; latency {100*(b[3]-a[3])/a[3]:+.1f}% ; power {100*(b[2]-a[2])/a[2]:+.1f}%")

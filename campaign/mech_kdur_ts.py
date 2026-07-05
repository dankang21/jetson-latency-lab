#!/usr/bin/env python3
# Extract (start_ns, dur_ns) for the dominant GEMM kernel from an nsys sqlite
# export — time-domain event statistics (inter-stall intervals, dwell runs).
# Usage: mech_kdur_ts.py <report.sqlite> <out.csv>
import sqlite3, sys

db, out = sys.argv[1], sys.argv[2]
con = sqlite3.connect(db)
rows = con.execute("""
  SELECT s.value, k.start, k.end - k.start
  FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON k.demangledName = s.id
""").fetchall()
con.close()
by = {}
for name, start, dur in rows:
    if 'gemm' in name.lower():
        by.setdefault(name, []).append((start, dur))
if not by:
    sys.stderr.write("no gemm kernels\n"); sys.exit(1)
name = max(by, key=lambda n: len(by[n]))
ev = sorted(by[name])
with open(out, 'w') as f:
    f.write("start_ns,dur_ns\n")
    for s, d in ev:
        f.write(f"{s},{d}\n")
print(f"kernel={name[:60]} n={len(ev)} -> {out}")

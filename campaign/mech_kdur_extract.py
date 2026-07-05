#!/usr/bin/env python3
# Extract per-kernel GEMM durations (ns) from an nsys sqlite export.
# Usage: mech_kdur_extract.py <report.sqlite> <out.txt>
# Picks the highest-count kernel whose name matches 'gemm' (cuBLAS GEMM);
# writes one duration-in-ns per line plus a summary to stdout.
import sqlite3, sys, statistics as st

db, out = sys.argv[1], sys.argv[2]
con = sqlite3.connect(db)
cur = con.cursor()
rows = cur.execute("""
  SELECT s.value AS name, k.end - k.start AS dur
  FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON k.demangledName = s.id
""").fetchall()
con.close()
by = {}
for name, dur in rows:
    by.setdefault(name, []).append(dur)
gemm = {n: d for n, d in by.items() if 'gemm' in n.lower()}
if not gemm:
    sys.stderr.write("no gemm kernels; kernels seen: %s\n" % list(by)[:5]); sys.exit(1)
name = max(gemm, key=lambda n: len(gemm[n]))
d = sorted(gemm[name])
open(out, 'w').write('\n'.join(map(str, d)) + '\n')
n = len(d)
p = lambda q: d[min(n - 1, int(q * n))]
med = p(0.5)
tail = sum(1 for x in d if x > 1.1 * med) / n
print(f"kernel={name[:60]} n={n} med_us={med/1e3:.1f} sd_us={st.pstdev(d)/1e3:.1f} "
      f"p99_us={p(0.99)/1e3:.1f} max_us={d[-1]/1e3:.1f} tail>1.1xmed={tail*100:.2f}%")

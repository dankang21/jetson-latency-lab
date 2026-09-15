#!/usr/bin/env python3
"""E4 launch-level statistics for one fixed design point (one launch_ext tag).

Reads every launch of a tag -- the committed eight (results/launch1020/dep_t1..8,
NX, tag mobilenet_vit10_g1020_D12, launch1020.sh) plus everything written by
campaign/launch_ext.sh under results/launch_ext/<board>/<tag>/t<k>.csv -- and
re-scores each 1000-cycle contract offline as response_us > D (the Table-II
convention, identical to admit_pick.py --score; the committed JSONs carry the
period as deadline, so the CSV is the only common ground). Prints
  * per-launch miss % (n, misses, source file),
  * the number x of launches with miss > THR % (default 2 %) out of N,
  * the Clopper-Pearson 95 % upper bound on the per-launch exceedance
    probability P(miss > THR %): one-sided 95 % (the bound proper) and the
    upper end of the two-sided 95 % CI (97.5 % one-sided),
  * the aggregate (pooled) miss % with a run-level bootstrap 95 % CI: whole
    launches resampled with replacement, B resamples, fixed seed.
numpy only (the on-device venv has no scipy): the CP bound inverts the exact
binomial CDF by bisection. --json emits the same numbers machine-readably.
Usage: launch_ext_stats.py [--tag mobilenet_vit10_g1020_D12] [--board nx|nano]
         [--D-ms D] [--thr 2] [--B 10000] [--seed 20260915] [--no-committed] [--json]
"""
import argparse, glob, json, math, os, re, sys
import numpy as np

LAB = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMMITTED_TAG, COMMITTED_BOARD, COMMITTED_N = "mobilenet_vit10_g1020_D12", "nx", 8
COMMITTED_PREFIX = os.path.join(LAB, "results", "launch1020", "dep_t")   # + k
CO_NAMES = ("vit", "yolov8s", "resnet50")


def parse_tag(tag):
    m = re.match(r"^(mobilenet|vit)_([a-z0-9]+)_g(\d+)_D([0-9.]+)$", tag)
    if not m:
        sys.exit(f"bad tag {tag!r} (expected <primary>_<co><co-hz>_g<f>_D<D>)")
    primary, cohz, f, D = m.groups()
    co = next((c for c in CO_NAMES if cohz.startswith(c)), None)
    if co is None or not cohz[len(co):]:
        sys.exit(f"bad co-runner field {cohz!r} in tag (co in {CO_NAMES} followed by its Hz)")
    return dict(primary=primary, co=co, co_hz=float(cohz[len(co):]), f=int(f), D_ms=float(D))


def load_launch(prefix, D_us):
    """prefix without extension -> dict or None (incomplete launch: csv without json)."""
    csv, js = prefix + ".csv", prefix + ".json"
    if not (os.path.isfile(csv) and os.path.isfile(js) and os.path.getsize(js) > 0):
        return None
    v = np.loadtxt(csv, delimiter=",", skiprows=1, usecols=3)      # response_us
    meta = json.load(open(js))
    misses = int(np.count_nonzero(v > D_us))
    r = dict(csv=csv, n=int(v.size), misses=misses, miss=misses / v.size,
             p99_us=float(np.percentile(v, 99)), max_us=float(v.max()),
             host=meta["meta"].get("host"), label=meta["meta"].get("label"))
    # cross-check against the harness's own count when it was run at the same D
    if abs(float(meta["meta"]["deadline_us"]) - D_us) < 1e-6 and meta["deadline_miss_count"] != misses:
        r["warn"] = f"json miss count {meta['deadline_miss_count']} != csv re-score {misses}"
    return r


def binom_cdf(x, n, p):
    """P(X <= x), X ~ Binomial(n, p); exact, log-space."""
    if p <= 0.0:
        return 1.0
    if p >= 1.0:
        return 1.0 if x >= n else 0.0
    lp, lq = math.log(p), math.log1p(-p)
    return sum(math.exp(math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1)
                        + i * lp + (n - i) * lq) for i in range(0, x + 1))


def cp_upper(x, n, alpha):
    """Clopper-Pearson one-sided upper bound at level 1-alpha: p_U with P(X <= x | p_U) = alpha."""
    if x >= n:
        return 1.0
    lo, hi = 0.0, 1.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if binom_cdf(x, n, mid) > alpha:
            lo = mid
        else:
            hi = mid
    return hi


def bootstrap_pooled(misses, n, B, seed):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(misses), size=(B, len(misses)))
    pooled = misses[idx].sum(axis=1) / n[idx].sum(axis=1)
    return float(np.percentile(pooled, 2.5)), float(np.percentile(pooled, 97.5))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default=COMMITTED_TAG)
    ap.add_argument("--board", choices=["nx", "nano"], default=COMMITTED_BOARD)
    ap.add_argument("--D-ms", type=float, help="override the D encoded in the tag")
    ap.add_argument("--thr", type=float, default=2.0, help="per-launch exceedance threshold, %% (default 2)")
    ap.add_argument("--B", type=int, default=10000, help="bootstrap resamples")
    ap.add_argument("--seed", type=int, default=20260915)
    ap.add_argument("--no-committed", action="store_true", help="ignore results/launch1020 even for the committed tag")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()

    t = parse_tag(a.tag)
    D_ms = a.D_ms if a.D_ms is not None else t["D_ms"]
    D_us = D_ms * 1000.0
    launches, skipped = [], []
    if a.tag == COMMITTED_TAG and a.board == COMMITTED_BOARD and not a.no_committed:
        for k in range(1, COMMITTED_N + 1):
            r = load_launch(f"{COMMITTED_PREFIX}{k}", D_us)
            (launches if r else skipped).append(dict(k=k, src="launch1020", **(r or {"csv": f"{COMMITTED_PREFIX}{k}.csv"})))
    ext_dir = os.path.join(LAB, "results", "launch_ext", a.board, a.tag)
    for csv in sorted(glob.glob(os.path.join(ext_dir, "t*.csv"))):
        m = re.search(r"/t(\d+)\.csv$", csv)
        if not m:
            continue
        k = int(m.group(1))
        r = load_launch(csv[:-4], D_us)
        (launches if r else skipped).append(dict(k=k, src="launch_ext", **(r or {"csv": csv})))
    launches.sort(key=lambda r: (r["k"], r["src"]))
    if not launches:
        sys.exit(f"no complete launches for tag {a.tag} on {a.board} (looked in {COMMITTED_PREFIX}* and {ext_dir})")
    ks = [r["k"] for r in launches]
    dup = sorted({k for k in ks if ks.count(k) > 1})

    N = len(launches)
    misses = np.array([r["misses"] for r in launches])
    n = np.array([r["n"] for r in launches])
    miss_pct = 100.0 * misses / n
    x = int(np.count_nonzero(miss_pct > a.thr))
    cp95, cp975 = cp_upper(x, N, 0.05), cp_upper(x, N, 0.025)
    pooled = float(misses.sum() / n.sum())
    lo, hi = bootstrap_pooled(misses, n, a.B, a.seed)
    out = dict(tag=a.tag, board=a.board, D_ms=D_ms, thr_pct=a.thr, N=N, x_exceed=x,
               cp_upper_95_one_sided=cp95, cp_upper_two_sided_95ci=cp975,
               pooled_miss_pct=100 * pooled, pooled_boot95_pct=[100 * lo, 100 * hi],
               mean_launch_miss_pct=float(miss_pct.mean()), max_launch_miss_pct=float(miss_pct.max()),
               total_cycles=int(n.sum()), total_misses=int(misses.sum()),
               B=a.B, seed=a.seed, duplicate_k=dup, incomplete=[s["csv"] for s in skipped],
               launches=[dict(k=r["k"], src=r["src"], n=r["n"], misses=r["misses"], miss_pct=100 * r["miss"],
                              p99_us=r["p99_us"], max_us=r["max_us"], host=r["host"], csv=os.path.relpath(r["csv"], LAB))
                         for r in launches])
    if a.json:
        print(json.dumps(out, indent=1))
        return

    print(f"tag {a.tag}  board {a.board}  D = {D_ms:g} ms  ({t['primary']} victim, {t['co']} co-runner @ {t['co_hz']:g} Hz, GPU {t['f']} MHz)")
    print(f"{'k':>3} {'src':<10} {'n':>5} {'miss':>5} {'miss%':>7} {'p99 us':>9} {'max us':>9}  host     file")
    for r in launches:
        flag = "  <-- > %g %%" % a.thr if 100 * r["miss"] > a.thr else ""
        warn = "  WARN " + r["warn"] if "warn" in r else ""
        print(f"{r['k']:>3} {r['src']:<10} {r['n']:>5} {r['misses']:>5} {100*r['miss']:>7.2f} {r['p99_us']:>9.1f} {r['max_us']:>9.1f}  {str(r['host']):<8} {os.path.relpath(r['csv'], LAB)}{flag}{warn}")
    for s in skipped:
        print(f"  skipped incomplete launch (no json): {os.path.relpath(s['csv'], LAB)}")
    if dup:
        print(f"  WARN duplicate launch indices across sources: {dup}")
    print()
    print(f"launches N = {N}   ({int(n.sum())} cycles, {int(misses.sum())} misses at D = {D_ms:g} ms)")
    print(f"launches with miss > {a.thr:g} %:  x = {x} / {N}")
    print(f"Clopper-Pearson upper bound on P(launch miss > {a.thr:g} %):  {100*cp95:.1f} % (one-sided 95 %),  "
          f"{100*cp975:.1f} % (upper end of two-sided 95 % CI)")
    print(f"aggregate miss (pooled over launches): {100*pooled:.3f} %   run-level bootstrap 95 % CI "
          f"[{100*lo:.3f}, {100*hi:.3f}] %  (B = {a.B}, seed {a.seed})")
    print(f"per-launch miss: mean {miss_pct.mean():.3f} %, max {miss_pct.max():.3f} %")


if __name__ == "__main__":
    main()

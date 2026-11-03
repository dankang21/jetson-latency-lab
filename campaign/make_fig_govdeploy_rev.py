#!/usr/bin/env python3
"""Fig. 2 (fig:govdeploy): deployed EMC-blind vs EMC-aware MobileNetV2 governor on
the NX, per-run deadline miss against the deadline.

Replaces the July three-run figure (results/govrep).  Data and rule are exactly
those of Table II (campaign/govdeploy_common.py): the three interleaved E1 sessions
results/govrep_rev/{nx,nx_mobilenet,nx_mobilenet_s3} (auto-detected; EMC locked 2133,
33 Hz, 10 pairs x 1000 cycles per clock), the rate-matched two-cell fits from results/refit33/nx (median p50 of three
33 Hz repeats at GPU 306/1173, blind @EMC 3199, aware @EMC 2133), r = 1.035, pick =
lowest grid clock with T(f) r <= D, else the aware arm DECLINES (top-clock fallback,
drawn hollow/dotted).

Curve: over a fine deadline grid 4.4--6.0 ms (0.02 ms plus the exact pick thresholds,
so the cliffs are vertical) each arm's pick is evaluated and the per-run miss % of that
clock's runs, pooled over all detected sessions (30 runs with the three above; 20 for
714 MHz, sessions 2 and 3 only), is
summarised as the median (line) and the min--max band.  The nine pre-stated deadlines
are marked at the median.  The thin dashed line is the blind arm under its original
8 Hz-cell picks (Table II, last panel).  Miss is on a symlog axis (linear below 0.1 %,
the per-run resolution) so zero-miss runs are drawn at 0.

Read-only on results/; writes the PDF to --out (default results/govrep_rev/govdeploy.pdf).
Run pinned with the system matplotlib:
    taskset -c 1 nice -n 19 /usr/bin/python3 campaign/make_fig_govdeploy_rev.py [--out FILE] [--png DIR]
Figure size (3.45 x 3.6 in, tight bbox) is that of the July figure so the float layout
in the main text is unchanged.
"""
import argparse
import io
import itertools
import os
import re
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__))
LAB = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from govdeploy_common import BOARDS, load_session, refit_fits, original_fits, pick, that  # noqa: E402
from e1b_stats import miss_pct, TAU, R  # noqa: E402

OUT = f"{LAB}/results/govrep_rev/govdeploy.pdf"
C_BLIND, C_AWARE = "#c0392b", "#2471a3"        # same arm colours as Fig. 1
LINTHRESH = 0.1                                # per-run resolution: 1 miss in 1000 cycles
TARGET_HW = 272.06 / 276.628                   # tight-bbox h/w of the July govdeploy.pdf (pdfinfo)

plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.titlesize": 8,
                     "axes.labelsize": 8, "legend.fontsize": 6.5, "xtick.labelsize": 7,
                     "ytick.labelsize": 7, "axes.spines.top": False, "axes.spines.right": False})


def saved_hw(fig, pad):
    """(h/w, w_pt, h_pt) of the PDF that savefig(bbox_inches='tight') would write."""
    buf = io.BytesIO()
    fig.savefig(buf, format="pdf", bbox_inches="tight", pad_inches=pad)
    m = re.search(rb"/MediaBox\s*\[\s*0\s+0\s+([\d.]+)\s+([\d.]+)\s*\]", buf.getvalue())
    w, h = float(m.group(1)), float(m.group(2))
    return h / w, w, h


def pooled_runs(sessions, f):
    """response_us arrays of every run at clock f, both sessions (session order, pair order)."""
    return [s["runs"][f][p]["v"] for s in sessions if f in s["runs"] for p in sorted(s["runs"][f])]


def miss_dist(sessions, f, D_ms):
    return np.array([miss_pct(v, D_ms * 1000.0) for v in pooled_runs(sessions, f)])


def deadline_grid(B, fits_list, lo=4.4, hi=6.0, step=0.02):
    """Fine grid plus every pick threshold T(f) r (just below and at it) inside [lo, hi]."""
    Ds = set(np.round(np.arange(lo, hi + 1e-9, step), 4).tolist())
    for kb in fits_list:
        for f in B["grid"]:
            t = that(kb, f) / 1000.0
            if lo < t < hi:
                Ds.add(round(t - 1e-6, 7))
                Ds.add(round(t, 7))
    return np.array(sorted(Ds))


def segments(B, sessions, kb, Ds):
    """[(f, feasible, D array, med, mn, mx)] over maximal runs of D with the same pick."""
    picks = [pick(kb, D * 1000.0, B) for D in Ds]
    out = []
    for key, grp in itertools.groupby(range(len(Ds)), key=lambda i: picks[i]):
        idx = list(grp)
        f, ok = key
        if not pooled_runs(sessions, f):
            continue
        D = Ds[idx]
        dist = np.array([miss_dist(sessions, f, d) for d in D])
        out.append((f, ok, D, np.median(dist, axis=1), dist.min(axis=1), dist.max(axis=1)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--png", default=None, help="also write a PNG preview into this directory")
    a = ap.parse_args()

    B = BOARDS["nx"]
    sessions = [s for s in (load_session(n, B["hz"]) for n in B["sessions"]) if s]
    assert len(sessions) >= 2, [s["name"] for s in sessions]
    rfits, rinfo = refit_fits(B)
    assert rfits is not None, "rate-matched refit (results/refit33/nx) incomplete"
    ofits, _ = original_fits(B)
    print("sessions:", ", ".join(f"{s['name']} {s['pairs']}" for s in sessions))
    for arm in ("blind", "aware"):
        print(f"  rate-matched {arm}: k={rfits[arm][0]:.0f} b={rfits[arm][1]:.0f}  "
              + " ".join(f"{f}:{that(rfits[arm], f) / 1000:.3f}" for f in B["grid"]))
    print(f"  original-8Hz blind: k={ofits['blind'][0]:.0f} b={ofits['blind'][1]:.0f}")

    Ds = deadline_grid(B, [rfits["blind"], rfits["aware"], ofits["blind"]])
    seg_b = segments(B, sessions, rfits["blind"], Ds)
    seg_a = segments(B, sessions, rfits["aware"], Ds)
    seg_o = segments(B, sessions, ofits["blind"], Ds)

    fig, ax = plt.subplots(figsize=(3.45, 3.6))

    def draw_arm(segs, col, lw, band=True, ls_ok="-", ls_decl=":"):
        """Per-pick pieces (band + median) joined by vertical connectors at the cliffs."""
        prev = None
        for f, ok, D, med, mn, mx in segs:
            if band:
                if ok:
                    ax.fill_between(D, mn, mx, color=col, alpha=0.18, lw=0)
                else:
                    ax.fill_between(D, mn, mx, facecolor="none", edgecolor=col, hatch="////",
                                    lw=0, alpha=0.35)
            ax.plot(D, med, color=col, lw=lw, ls=ls_ok if ok else ls_decl)
            if prev is not None:
                ax.plot([prev[0], D[0]], [prev[1], med[0]], color=col, lw=lw, ls=ls_ok)
            prev = (D[-1], med[-1])

    draw_arm(seg_b, C_BLIND, 1.4)
    draw_arm(seg_a, C_AWARE, 1.4)
    # blind under the original 8 Hz-cell picks (thin dashed, median only)
    for f, ok, D, med, mn, mx in seg_o:
        ax.plot(D, med, color=C_BLIND, lw=0.7, ls="--", alpha=0.85)
    for (f0, _, D0, m0, _, _), (f1, _, D1, m1, _, _) in zip(seg_o, seg_o[1:]):
        ax.plot([D0[-1], D1[0]], [m0[-1], m1[0]], color=C_BLIND, lw=0.7, ls="--", alpha=0.85)
    # tau budget
    ax.axhline(TAU, color="k", ls="--", lw=0.8)
    ax.text(6.0, TAU * 1.15, r"$\tau$ = 2%", fontsize=6.5, ha="right", va="bottom")
    # pre-stated grid points at the median of the pick's runs; picks listed under the axis
    rows = {"blind": [], "aware": []}
    nmax = max(len(pooled_runs(sessions, f)) for f in B["grid"] if pooled_runs(sessions, f))
    short = {}   # clocks with fewer runs than nmax (not run in every session) -> '*' note
    for D in B["D"]:
        fb, okb = pick(rfits["blind"], D * 1000.0, B)
        fa, oka = pick(rfits["aware"], D * 1000.0, B)
        db, da = miss_dist(sessions, fb, D), miss_dist(sessions, fa, D)
        mb, ma = float(np.median(db)), float(np.median(da))
        for f, n in ((fb, len(db)), (fa, len(da))):
            if n < nmax:
                short[f] = n
        ax.plot(D, mb, "s", ms=3.6, color=C_BLIND, mec="white", mew=0.4, zorder=5)
        rows["blind"].append(f"{fb}" + ("" if okb else "d") + ("*" if len(db) < nmax else ""))
        if oka:
            ax.plot(D, ma, "o", ms=3.6, color=C_AWARE, mec="white", mew=0.4, zorder=5)
            rows["aware"].append(f"{fa}" + ("*" if len(da) < nmax else ""))
        else:
            ax.plot(D, ma, "o", ms=4.2, mfc="white", mec=C_AWARE, mew=1.0, zorder=5)
            ax.annotate("declines\n(1173 fallback)", xy=(D, ma), xytext=(D + 0.045, 28),
                        fontsize=6, color="k", ha="left", va="center",
                        arrowprops=dict(arrowstyle="-", lw=0.6, color=C_AWARE))
            rows["aware"].append("decl.")
        print(f"  D={D:.1f}: blind {fb}{'' if okb else '(fallback)'} n={len(db)} med {mb:5.1f} | "
              f"aware {fa}{'' if oka else '(fallback)'} n={len(da)} med {ma:5.2f}")

    ax.set_yscale("symlog", linthresh=LINTHRESH, linscale=0.35)
    ax.set_ylim(-0.03, 400)
    ax.set_yticks([0, 0.1, 1, 10, 100])
    ax.set_yticklabels(["0", "0.1", "1", "10", "100"])
    ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_xlim(4.36, 6.04)
    ax.set_xticks(B["D"])
    ax.set_xticklabels([f"{d:.1f}" for d in B["D"]])
    ax.set_ylabel("per-run deadline miss (%)")
    nsess = {1: "one", 2: "two", 3: "three", 4: "four"}.get(len(sessions), str(len(sessions)))
    ax.set_title(f"NX MobileNetV2 @ EMC 2133 (33 Hz), {nsess} sessions")
    ax.grid(True, ls=":", alpha=0.3)
    # picked clock rows (Table II f* columns) under the tick labels, then the x label
    import matplotlib.transforms as mtrans
    tr = mtrans.blended_transform_factory(ax.transData, ax.transAxes)
    for arm, col, y in (("blind", C_BLIND, -0.115), ("aware", C_AWARE, -0.175)):
        ax.text(4.29, y, f"{arm} $f^\\star$", color="k", fontsize=5.6, ha="right", va="center",
                transform=tr)
        for D, s in zip(B["D"], rows[arm]):
            ax.text(D, y, s, color="k", fontsize=5.6, ha="center", va="center", transform=tr)
    ax.set_xlabel("deadline $D$ (ms)", labelpad=17)
    handles = [
        Line2D([0], [0], color=C_BLIND, lw=1.4, marker="s", ms=3.6, label="EMC-blind (fit @3199)"),
        Line2D([0], [0], color=C_AWARE, lw=1.4, marker="o", ms=3.6, label="EMC-aware (2-cell @2133)"),
        Patch(facecolor="#888", alpha=0.25, label="min–max over runs (median line)"),
        Line2D([0], [0], color=C_AWARE, lw=1.2, ls=":", marker="o", ms=4.2, mfc="white",
               mec=C_AWARE, label="aware declines: 1173 fallback (d)"),
        Line2D([0], [0], color=C_BLIND, lw=0.7, ls="--", label="blind, original 8 Hz-cell picks"),
        Line2D([0], [0], color="k", lw=0.8, ls="--", label=r"$\tau$ = 2 % budget"),
    ]
    for f, n in sorted(short.items()):
        ran = [i + 1 for i, s in enumerate(sessions) if f in s["runs"]]
        which = ("session " + ", ".join(map(str, ran)) + " only") if len(ran) == 1 else \
                ("sessions " + ", ".join(map(str, ran)) + " only")
        handles.append(Line2D([0], [0], lw=0, label=f"* {n} runs ({f} MHz: {which})"))
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.115), ncol=2,
               frameon=False, fontsize=6, handlelength=1.8, columnspacing=1.2, labelspacing=0.35)
    fig.subplots_adjust(bottom=0.30, top=0.93, left=0.15, right=0.98)
    # Match the July figure's tight-bbox aspect (272.06 x 276.628 pt) so the float
    # layout in the main text is unchanged: adjust the figure height until the saved PDF's
    # MediaBox has the same h/w (the width is fixed by the legend/axes, the height follows).
    PAD = 0.1
    for _ in range(6):
        hw, wpt, hpt = saved_hw(fig, PAD)
        if abs(hw - TARGET_HW) < 5e-4:
            break
        w, h = fig.get_size_inches()
        fig.set_size_inches(w, h + (TARGET_HW - hw) * wpt / 72.0)
    print(f"  figure {fig.get_size_inches()[0]:.2f} x {fig.get_size_inches()[1]:.2f} in, "
          f"saved bbox {wpt:.1f} x {hpt:.1f} pt (h/w {hw:.4f}, target {TARGET_HW:.4f})")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    fig.savefig(a.out, bbox_inches="tight", pad_inches=PAD)
    print("wrote", a.out)
    if a.png:
        os.makedirs(a.png, exist_ok=True)
        fig.savefig(os.path.join(a.png, "govdeploy.png"), dpi=220, bbox_inches="tight", pad_inches=PAD)
    plt.close(fig)


if __name__ == "__main__":
    main()

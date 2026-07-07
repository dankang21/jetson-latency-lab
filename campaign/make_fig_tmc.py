#!/usr/bin/env python3
"""TMC 12-page figures: F1 overview (solo failure / admission pipeline /
joint-stress outcome) and F3 layers (horizon erosion / co-tenant bimodal /
admission at the cliff). Reads committed results; writes paper/tmc12/figs/."""
import csv, glob, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

LAB = "/home/dk/mobile/jetson-latency-lab"
OUT = f"{LAB}/paper/tmc12/figs"
plt.rcParams.update({"font.size": 8, "font.family": "serif",
                     "axes.spines.top": False, "axes.spines.right": False})
BUDGET = 2.0

def dur_cells(pat):
    out = {}
    for f in sorted(glob.glob(pat)):
        g = int(re.search(r"g(\d+)", f).group(1))
        out.setdefault(g, []).append(
            np.array([float(x["dur_ns"]) for x in csv.DictReader(open(f))]) / 1e6)  # ms
    return out

# ═══ F1 ═══
fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.15),
                       gridspec_kw={"width_ratios": [1, 1.75, 0.9]})
# (a) solo: blind vs aware
D = ["4.6", "4.8", "5.0"]
blind = [28.4, 27.5, 24.8]; berr = [3.4, 4.2, 5.0]
aware = [0.81, 0.16, 0.08]; aub = [1.04, 0.25, 0.13]
x = np.arange(3); w = 0.38
a = ax[0]
a.bar(x - w/2, blind, w, yerr=berr, capsize=2, color="#c0392b", label="EMC-blind")
a.bar(x + w/2, aware, w, color="#2471a3", label="EMC-aware")
for i, (v, u) in enumerate(zip(aware, aub)):
    a.plot([x[i]+w/2]*2, [v, u], color="#1a5276", lw=1)
a.axhline(BUDGET, ls="--", lw=0.8, color="k")
a.text(-0.42, BUDGET*1.18, r"$\tau$=2%", fontsize=6.5, ha="left")
a.set_yscale("log"); a.set_ylim(0.05, 60)
a.set_xticks(x); a.set_xticklabels([f"D={d}" for d in D], fontsize=7)
a.set_ylabel("deadline miss (%)")
a.set_title("(a) solo: blind vs aware (MNv2, NX)", fontsize=8)
a.legend(fontsize=6, frameon=False, loc="upper right", handlelength=1.2)
# (b) pipeline schematic
b = ax[1]; b.axis("off"); b.set_xlim(0, 11.2); b.set_ylim(0, 10.5)
def box(x0, y0, wd, ht, text, fc="#eaf2f8"):
    b.add_patch(FancyBboxPatch((x0, y0), wd, ht, boxstyle="round,pad=0.15",
                fc=fc, ec="#5d6d7e", lw=0.8))
    b.text(x0 + wd/2, y0 + ht/2, text, ha="center", va="center", fontsize=6.3)
box(0.1, 7.4, 3.3, 2.2, "EMC state\n2-cell refit\n$\\to \\hat{T}(f)$")
box(0.1, 4.2, 3.3, 2.2, "horizon state\nmeasured drift\n$\\to \\gamma$")
box(0.1, 0.6, 3.3, 2.6, "occupancy state\n200-cycle probe $\\to M$\nguard: $\\max{-}p50$")
box(4.6, 3.2, 3.5, 3.8,
    "admit lowest $f$:\n$\\hat{T}(f)(1{+}\\gamma H)\\,r$\n$+\\;M_g\\,f_{probe}/f + h \\leq D$\n($h{=}2\\sigma_{probe}$)",
    fc="#fdf2e9")
box(9.0, 5.8, 1.9, 1.8, "admit", fc="#d5f5e3")
box(9.0, 2.2, 1.9, 1.8, "decline", fc="#fadbd8")
for y in (8.5, 5.3, 1.9):
    b.add_patch(FancyArrowPatch((3.55, y), (4.6, 5.1), arrowstyle="->",
                mutation_scale=7, lw=0.7, color="#5d6d7e"))
b.add_patch(FancyArrowPatch((8.25, 5.6), (9.0, 6.7), arrowstyle="->", mutation_scale=7, lw=0.7, color="#5d6d7e"))
b.add_patch(FancyArrowPatch((8.25, 4.6), (9.0, 3.1), arrowstyle="->", mutation_scale=7, lw=0.7, color="#5d6d7e"))
b.set_title("(b) bounded-probe contract admission", fontsize=8)
# (c) joint stress
c = ax[2]
vals = [32.3, 0.11]; ubs = [32.4, 0.14]
xc = [0, 1]
c.bar(xc, vals, 0.55, color=["#c0392b", "#1e8449"])
for i, (v, u) in enumerate(zip(vals, ubs)):
    c.plot([xc[i]]*2, [v, u], color="k", lw=1)
c.axhline(BUDGET, ls="--", lw=0.8, color="k")
c.set_yscale("log"); c.set_ylim(0.05, 60)
c.set_xticks(xc)
c.set_xticklabels(["tenancy-blind\n$D$=28", "guarded accept\n$D$=44"], fontsize=6.5)
c.set_title("(c) joint stress: Qwen + ViT co-tenant", fontsize=8)
c.set_ylabel("token miss (%)")
fig.tight_layout(w_pad=1.2)
fig.savefig(f"{OUT}/f1_overview.pdf", bbox_inches="tight")
print("f1_overview.pdf")

# ═══ F3 ═══
solo = dur_cells(f"{LAB}/results/boundary_stats/q15_g1173_r*.csv")[1173]
h4 = dur_cells(f"{LAB}/results/horizon_4k/q15h4k_g1173_r*.csv")[1173]
co = dur_cells(f"{LAB}/results/joint_hxt/q15co_g1173_r*.csv")[1173]

fig, ax = plt.subplots(1, 3, figsize=(7.0, 1.95))
# (a) horizon erosion: rolling median per token
a = ax[0]
m2 = np.median(np.vstack([r[:2000] for r in solo]), axis=0)
m4 = np.median(np.vstack([r[:4000] for r in h4]), axis=0)
k = 100
roll2 = np.convolve(m2, np.ones(k)/k, mode="valid")
roll4 = np.convolve(m4, np.ones(k)/k, mode="valid")
a.plot(np.arange(len(roll4)) + k/2, roll4, color="#8e44ad", lw=1.0, label="H = 4,000")
a.plot(np.arange(len(roll2)) + k/2, roll2, color="#2471a3", lw=1.2, label="H = 2,000")
a.axhline(27, ls="--", lw=0.8, color="#c0392b")
a.text(3900, 27.06, "D=27", fontsize=6.5, ha="right", color="#c0392b")
a.axhline(28, ls=":", lw=0.8, color="#7d6608")
a.text(3900, 28.06, "D=28", fontsize=6.5, ha="right", color="#7d6608")
a.set_xlabel("token index", fontsize=7); a.set_ylabel("median ms/token")
a.set_title("(a) decode-length erosion (1173 MHz)", fontsize=8)
a.legend(fontsize=6.5, frameon=False, loc="upper left")
a.annotate("0.83% miss\n(tokens 1–500)", xy=(520, 26.55), fontsize=6, ha="center")
a.annotate("9.9% miss\n(1501–2000)", xy=(1650, 27.12), fontsize=6, ha="center")
# (b) co-tenant bimodal histogram
b = ax[1]
allv = np.concatenate(co)
b.hist(allv, bins=120, color="#2471a3", log=True)
b.axvline(np.median(allv), color="k", lw=0.8, ls="--")
b.text(np.median(allv)+0.8, 40, f"p50 {np.median(allv):.1f} ms", fontsize=6.2)
b.annotate(f"{np.mean(allv>32)*100:.0f}% of tokens\ncollide (42–43 ms)",
           xy=(41.8, 200), xytext=(31.5, 250), fontsize=6.2,
           arrowprops=dict(arrowstyle="->", lw=0.7))
b.set_xlabel("ms/token", fontsize=7); b.set_ylabel("tokens")
b.set_title("(b) decode under ViT co-tenant", fontsize=8)
# (c) admission at the cliff
c = ax[2]
labels = ["blind\nD=28", "declined\n1122@43", "declined\n1173@43", "accepted\n1173@44"]
vals = [32.3, 9.03, 0.41, 0.11]
ubs = [32.4, 9.50, 0.49, 0.14]
cols = ["#c0392b", "#e67e22", "#f4d03f", "#1e8449"]
xc = np.arange(4)
c.bar(xc, vals, 0.6, color=cols)
for i, (v, u) in enumerate(zip(vals, ubs)):
    c.plot([xc[i]]*2, [v, u], color="k", lw=1)
c.axhline(BUDGET, ls="--", lw=0.8, color="k")
c.text(3.4, 2.3, r"$\tau$", fontsize=7)
c.set_yscale("log"); c.set_ylim(0.05, 60)
c.set_xticks(xc); c.set_xticklabels(labels, fontsize=5.6)
c.set_ylabel("token miss (%)")
c.set_title("(c) admission at the collision cliff", fontsize=8)
fig.tight_layout(w_pad=1.2)
fig.savefig(f"{OUT}/f3_layers.pdf", bbox_inches="tight")
print("f3_layers.pdf")

#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Generate the paper's figures from results/campaign_g2 (+ pilot data).

Run with any python that has matplotlib + numpy.
Outputs PDF (for LaTeX) + PNG (for quick viewing) into paper/figs/
(created if absent; the directory is gitignored).
"""

import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
G2 = ROOT / "results" / "campaign_g2"
FIGS = ROOT / "paper" / "figs"
FIGS.mkdir(parents=True, exist_ok=True)

WL_LABEL = {
    "mobilenet": "MobileNetV2",
    "vit": "ViT-Small",
    "proxy": "decode proxy (GEMV)",
    "cproxyv1": "GEMM 2048$^3$",
    "cproxyv2": "GEMM L2-resident",
    "slm": "Qwen2.5-1.5B decode",
}
WL_ORDER = ["proxy", "vit", "cproxyv1", "slm", "mobilenet", "cproxyv2"]
EMC = [204, 665, 2133, 3199]

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
    "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "figure.dpi": 150, "savefig.bbox": "tight",
})
FIGSIZE = (3.45, 2.3)   # single column IEEE


def save(fig, name):
    fig.savefig(FIGS / f"{name}.pdf")
    fig.savefig(FIGS / f"{name}.png")
    plt.close(fig)
    print(f"wrote {name}.pdf/.png")


def load_part_a():
    cells = defaultdict(dict)
    for f in sorted((G2 / "partA").glob("emc*.json")):
        m = re.match(r"emc(\d+)_(\w+)\.json", f.name)
        cells[m.group(2)][int(m.group(1))] = json.loads(f.read_text())["compute_us"]
    return cells


def fig_rq1_curves(cells):
    fig, ax = plt.subplots(figsize=(3.45, 2.7))
    for wl in WL_ORDER:
        p50 = np.array([cells[wl][m]["p50"] for m in EMC])
        norm = p50 / p50[-1]
        ax.plot(EMC, norm, "o-", ms=3, lw=1, label=WL_LABEL[wl])
    ax.axhline(1.0, color="gray", lw=0.5, ls=":")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(EMC)
    ax.set_xticklabels(["204", "665.6", "2133", "3199"])
    ax.minorticks_off()
    ax.set_xlabel("EMC frequency (MHz, locked; CPU/GPU pinned)")
    ax.set_ylabel("p50 latency / p50 at 3199 MHz")
    # legend below the plot box so it never overlaps the curves
    ax.legend(ncol=3, frameon=False, fontsize=6.5,
              loc="upper center", bbox_to_anchor=(0.5, -0.24),
              handlelength=1.4, columnspacing=1.0)
    save(fig, "rq1_curves")


def fig_rq1_spectrum(cells):
    fig, ax = plt.subplots(figsize=FIGSIZE)
    shifts = []
    for wl in WL_ORDER:
        p2133 = cells[wl][2133]["p50"]
        p3199 = cells[wl][3199]["p50"]
        shifts.append((p2133 - p3199) / p3199 * 100)
    colors = ["#c44" if s < 0 else "#369" for s in shifts]
    y = np.arange(len(WL_ORDER))
    ax.barh(y, shifts, color=colors, height=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels([WL_LABEL[w] for w in WL_ORDER])
    ax.axvline(0, color="black", lw=0.7)
    ax.set_xlabel("p50 shift, EMC 3199$\\to$2133 MHz (%)")
    for yi, s in zip(y, shifts):
        ax.text(s + (1.5 if s >= 0 else -1.5), yi, f"{s:+.1f}%",
                va="center", ha="left" if s >= 0 else "right", fontsize=7)
    ax.set_xlim(-18, 58)
    save(fig, "rq1_spectrum")


def load_cell(name):
    a = np.loadtxt(G2 / "partB" / f"{name}.csv", delimiter=",", skiprows=1)
    return a[:, 3]   # response_us


def fig_rq2_cliffs():
    fig, ax = plt.subplots(figsize=FIGSIZE)
    dls = np.linspace(4.0, 7.0, 121)
    styles = {(2133, 0): ("#369", ":"), (2133, 2): ("#369", "--"),
              (2133, 4): ("#369", "-"), (3199, 0): ("#c44", ":"),
              (3199, 2): ("#c44", "--"), (3199, 4): ("#c44", "-")}
    for (mhz, adv), (c, ls) in styles.items():
        resp = load_cell(f"emc{mhz}_adv{adv}_mobilenet")
        miss = [(resp > d * 1000).mean() * 100 for d in dls]
        miss = [max(m, 1e-4) for m in miss]
        ax.plot(dls, miss, color=c, ls=ls, lw=1,
                label=f"EMC {mhz}, {adv} adv")
    ax.set_yscale("log")
    ax.set_ylim(8e-4, 120)
    ax.set_xlabel("deadline (ms)")
    ax.set_ylabel("miss rate (%), 100k cycles")
    ax.legend(frameon=False, ncol=2)
    ax.set_title("MobileNetV2 @ 50 Hz: the miss cliff", fontsize=8)
    save(fig, "rq2_cliffs")


def bursts_of(miss):
    out, i, n = [], 0, len(miss)
    while i < n:
        if miss[i]:
            j = i
            while j < n and miss[j]:
                j += 1
            out.append(j - i)
            i = j
        else:
            i += 1
    return np.array(out)


def fig_rq2_bursts():
    fig, ax = plt.subplots(figsize=FIGSIZE)
    resp = load_cell("emc2133_adv2_mobilenet")
    dl = np.percentile(resp, 99.9)
    lens = bursts_of(resp > dl)
    p = 0.001
    kmax = lens.max()
    ks = np.arange(1, kmax + 1)
    emp = np.array([(lens == k).sum() for k in ks], dtype=float)
    # geometric run lengths expected under independence, same #misses
    n_miss = lens.sum()
    geo = np.array([n_miss * (1 - p) ** 2 * p ** (k - 1) for k in ks])
    geo = geo / geo.sum() * len(lens)
    ax.bar(ks - 0.2, emp, width=0.4, label="observed", color="#369")
    ax.bar(ks + 0.2, np.maximum(geo, 1e-3), width=0.4,
           label="independent misses (geometric)", color="#bbb")
    ax.set_yscale("log")
    ax.set_ylim(5e-3, 60)
    ax.set_xlabel("burst length (consecutive misses)")
    ax.set_ylabel("number of bursts")
    ax.set_title("MobileNetV2, EMC 2133 + adversary, deadline = p99.9",
                 fontsize=8)
    ax.legend(frameon=False)
    save(fig, "rq2_bursts")


def gpd_pwm(excess):
    x = np.sort(excess)
    n = len(x)
    b0 = x.mean()
    pw = 1.0 - (np.arange(1, n + 1) - 0.35) / n
    b1 = (x * pw).mean()
    xi = 2.0 - b0 / (b0 - 2 * b1)
    sigma = 2.0 * b0 * b1 / (b0 - 2 * b1)
    return xi, sigma


def fig_rq2_tailfit():
    fig, ax = plt.subplots(figsize=FIGSIZE)
    resp = load_cell("emc2133_adv2_mobilenet") / 1000.0
    n = len(resp)
    srt = np.sort(resp)
    surv = 1.0 - np.arange(1, n + 1) / (n + 1)
    sel = srt >= np.percentile(resp, 95)
    ax.semilogy(srt[sel], surv[sel], ".", ms=2, color="#369",
                label="empirical survival")
    # Gaussian fit
    mu, sd = resp.mean(), resp.std()
    xs = np.linspace(srt[sel][0], srt[-1] * 1.02, 200)
    from math import erf
    gs = [0.5 * (1 - erf((x - mu) / (sd * np.sqrt(2)))) for x in xs]
    ax.semilogy(xs, np.maximum(gs, 1e-9), "--", color="#999",
                label=f"Gaussian ($\\mu$+3$\\sigma$={mu + 3 * sd:.2f} ms)")
    # GPD fit above p99
    u = np.percentile(resp, 99.0)
    xi, sigma = gpd_pwm(resp[resp > u] - u)
    zeta = 0.01
    xs2 = xs[xs > u]
    gpd = zeta * np.maximum(1 + xi * (xs2 - u) / sigma, 1e-12) ** (-1 / xi)
    ax.semilogy(xs2, gpd, "-", color="#c44",
                label=f"GPD above p99 ($\\xi$={xi:.2f})")
    ax.set_ylim(1e-6, 0.06)
    ax.set_xlabel("response time (ms)")
    ax.set_ylabel("P(response > x)")
    ax.set_title("MobileNetV2, EMC 2133 + adversary: tail fits", fontsize=8)
    ax.legend(frameon=False, loc="lower left", fontsize=6.5)
    save(fig, "rq2_tailfit")


def fig_actuation():
    fig, ax = plt.subplots(figsize=(3.45, 1.9))
    domains = ["CPU", "GPU", "EMC"]
    settle = [1.2, 5.0, 8.0]       # workload-observed settle p50 (ms), n=150-200/dir
    stall = [0.0491, 0.0916, 0.0235]  # worst RAW stall p95 (ms)
    y = np.arange(len(domains))
    ax.barh(y + 0.18, settle, height=0.34, color="#369",
            label="actuation lag (settle p50)")
    ax.barh(y - 0.18, stall, height=0.34, color="#c44",
            label="workload stall (raw p95)")
    ax.set_yticks(y)
    ax.set_yticklabels(domains)
    ax.set_xscale("log")
    ax.set_xlim(1e-3, 30)
    ax.axvline(0.1, color="gray", lw=0.6, ls=":")
    ax.text(0.1, 2.45, "100 µs kill line", fontsize=6, ha="center")
    ax.set_xlabel("ms (log)")
    ax.legend(frameon=False, loc="lower right", fontsize=6.5)
    save(fig, "rq3_actuation")


def main():
    cells = load_part_a()
    fig_rq1_curves(cells)
    fig_rq1_spectrum(cells)
    fig_rq2_cliffs()
    fig_rq2_bursts()
    fig_rq2_tailfit()
    fig_actuation()


if __name__ == "__main__":
    main()


def fig_govsim():
    import json
    data = json.loads((ROOT / "results" / "estimator_break" /
                       "govsim_curves.json").read_text())["mobilenet"]
    fig, ax = plt.subplots(figsize=FIGSIZE)
    styles = {"A+G": ("#c44", "-"), "A+P": ("#c44", "--"),
              "C+G": ("#369", "-"), "C+P": ("#369", "--")}
    names = {"A+G": "blind + Gauss", "A+P": "blind + GPD",
             "C+G": "aware + Gauss", "C+P": "aware + GPD"}
    for pol, (c, ls) in styles.items():
        cur = data[pol]["curve"]
        miss = [max(m, 1e-2) for m in cur["miss_pct"]]
        ax.plot(cur["D_ms"], miss, color=c, ls=ls, lw=1, label=names[pol])
    ax.axhline(100 / 300, color="gray", lw=0.6, ls=":")
    ax.text(ax.get_xlim()[1] * 0.99, 100 / 300 * 1.2, "1/300 floor",
            fontsize=6, ha="right")
    ax.set_yscale("log")
    ax.set_ylim(8e-3, 130)
    ax.set_xlabel("deadline (ms)")
    ax.set_ylabel("achieved miss rate (%) at EMC 2133")
    ax.set_title("MobileNetV2: governor outcomes, profiled @3199",
                 fontsize=8)
    ax.legend(frameon=False, fontsize=6.5, ncol=2)
    save(fig, "govsim")


def fig_evtval():
    """Out-of-sample EVT validation: first-50k GPD prediction vs last-50k
    empirical quantiles, per Part B cell. Replaces the in-sample survival
    plot as the main Fig 5 (review: figure must show the validation the
    text claims)."""
    import math
    def gpd_q(u, xi, sigma, zeta, p):
        r = p / zeta
        if abs(xi) < 1e-6:
            return u - sigma * math.log(r)
        return u + sigma / xi * (r ** (-xi) - 1.0)
    import re as _re
    cells, preds, emps, marks = [], [], [], []
    for f in sorted((G2 / "partB").glob("emc*_adv*_*.csv")):
        if not _re.match(r"emc\d+_adv\d+_\w+\.csv$", f.name):
            continue
        resp = np.loadtxt(f, delimiter=",", skiprows=1)[:, 3]
        a, b = resp[:50000], resp[50000:]
        u = np.percentile(a, 99.0)
        xi, sigma = gpd_pwm(a[a > u] - u)
        zeta = (a > u).mean()
        for q, p_t, mk in ((99.9, 1e-3, "o"), (99.99, 1e-4, "s")):
            preds.append(gpd_q(u, xi, sigma, zeta, p_t) / 1000)
            emps.append(np.percentile(b, q) / 1000)
            marks.append(mk)
        cells.append(f.stem)
    fig, ax = plt.subplots(figsize=FIGSIZE)
    pr, em, mk = np.array(preds), np.array(emps), np.array(marks)
    for m, lab in (("o", "p99.9"), ("s", "p99.99")):
        sel = mk == m
        ax.plot(em[sel], pr[sel], m, ms=4, mfc="none",
                color="#369" if m == "o" else "#c44", label=lab)
    lo, hi = min(em.min(), pr.min()) * 0.98, max(em.max(), pr.max()) * 1.02
    xs = np.linspace(lo, hi, 10)
    ax.plot(xs, xs, "-", color="gray", lw=0.7)
    ax.fill_between(xs, xs * 0.94, xs * 1.06, color="gray", alpha=0.15,
                    label=r"$\pm$6% band")
    ax.set_xlabel("observed quantile, last 50k cycles (ms)")
    ax.set_ylabel("GPD prediction from first 50k (ms)")
    ax.set_title("Out-of-sample tail validation, all 8 cells", fontsize=8)
    ax.legend(frameon=False, fontsize=6.5)
    save(fig, "rq2_evtval")


def fig_govsim_all():
    """Appendix: governor outcome curves for all four workloads."""
    import json
    data = json.loads((ROOT / "results" / "estimator_break" /
                       "govsim_curves.json").read_text())
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.4))
    styles = {"A+G": ("#c44", "-"), "A+P": ("#c44", "--"),
              "C+G": ("#369", "-"), "C+P": ("#369", "--")}
    names = {"A+G": "blind+G", "A+P": "blind+P",
             "C+G": "aware+G", "C+P": "aware+P"}
    for ax, wl in zip(axes.flat, ["mobilenet", "vit", "proxy", "cproxyv2"]):
        for pol, (c, ls) in styles.items():
            cur = data[wl][pol]["curve"]
            miss = [max(m, 1e-2) for m in cur["miss_pct"]]
            ax.plot(cur["D_ms"], miss, color=c, ls=ls, lw=0.9)
        ax.axhline(100 / 300, color="gray", lw=0.5, ls=":")
        ax.set_yscale("log")
        ax.set_ylim(8e-3, 130)
        ax.set_title(WL_LABEL.get(wl, wl), fontsize=8)
        ax.set_xlabel("deadline (ms)", fontsize=7)
        ax.set_ylabel("miss %", fontsize=7)
    handles = [plt.Line2D([0], [0], color=c, ls=ls, label=names[p])
               for p, (c, ls) in styles.items()]
    fig.legend(handles=handles, ncol=4, frameon=False, fontsize=7,
               loc="lower center", bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout()
    save(fig, "govsim_all")

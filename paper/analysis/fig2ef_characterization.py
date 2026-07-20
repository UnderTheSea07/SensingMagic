#!/usr/bin/env python3
"""Own-figure redraws for Fig. 2e (geometry-sensitivity) and Fig. 2f
(dynamics + durability) from the reliability-project dataset.

Deliberately different visual form from the reliability paper's figures
(3D response surface there -> 2D annotated heatmaps here) to avoid duplicate
publication. Sources (all spot-checkable CSV):
  - Fig2exp_experimental_points.csv  (46 rows, per combo/direction/depth)
  - fatigue_sensitivity_vs_cycles.csv (+ raw hall row counts for bad rate)
  - Fig8_all_events_stats.csv        (81 tap events, response/recovery)
Outputs: paper/figures/fig2e_geometry_sensitivity.{pdf,png},
         paper/figures/fig2f_dynamics_durability.{pdf,png}

The fig2f sub-panels are labelled f4/f5 (continuing f1-f3 of
fig2f_response_event) so they cannot clash with the top-level a-f
letters of the Fig. 2 composite.
"""
import glob
import os
import re

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA = os.path.expanduser("~/Desktop/磁毛传感器可靠性分析原始数据")
PROC = os.path.join(DATA, "处理结果与图表_20260717")
FIGS = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(FIGS, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 7.5, "axes.labelsize": 8, "axes.titlesize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.8,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8, "ytick.major.width": 0.8,
    "xtick.major.size": 2.8, "ytick.major.size": 2.8,
    "xtick.direction": "out", "ytick.direction": "out",
    "axes.spines.top": False, "axes.spines.right": False,
    "hatch.linewidth": 0.5,
    "pdf.fonttype": 42, "figure.dpi": 300,
})

# Okabe-Ito palette (colour-blind safe)
OI = {"blue": "#0072B2", "verm": "#D55E00", "green": "#009E73",
      "orange": "#E69F00", "sky": "#56B4E9", "mauve": "#CC79A7",
      "black": "#000000", "grey": "#888888"}

# single-hue sequential ramp built on the Okabe-Ito blues
OI_BLUES = matplotlib.colors.LinearSegmentedColormap.from_list(
    "oi_blues", ["#E7F1F9", "#A8D2EC", "#56B4E9", "#0072B2", "#014A73"])


def _text_on(rgba):
    """Adaptive annotation colour: white on dark cells, near-black on light."""
    def lin(c):
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
    r, g, b = rgba[:3]
    lum = 0.2126 * lin(r) + 0.7152 * lin(g) + 0.0722 * lin(b)
    return "white" if lum < 0.38 else "#1A1A1A"

LS, DS = [3.0, 5.0, 8.0], [0.5, 0.8, 1.0]


def fig2e():
    from matplotlib.colors import Normalize
    from matplotlib.patches import Rectangle
    from matplotlib.ticker import MaxNLocator
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    pts = pd.read_csv(os.path.join(PROC, "figures_paper", "data",
                                   "Fig2exp_experimental_points.csv"))
    fig, axes = plt.subplots(1, 2, figsize=(5.6, 2.55))
    for ax, cn, en in zip(axes, ["切向", "法向"], ["Tangential", "Normal"]):
        sub = pts[(pts.direction == cn) & (~pts.outlier)]
        med = sub.groupby(["L", "d"]).S_mT_per_N.median()
        grid = np.full((3, 3), np.nan)
        for (L, d), v in med.items():
            grid[DS.index(d), LS.index(L)] = v
        norm = Normalize(np.nanmin(grid), np.nanmax(grid))
        mesh = ax.pcolormesh(np.arange(4), np.arange(4), grid,
                             cmap=OI_BLUES, norm=norm,
                             edgecolors="white", linewidth=1.5)
        ax.set_aspect("equal")
        for i, d in enumerate(DS):
            for j, L in enumerate(LS):
                v = grid[i, j]
                if np.isnan(v):
                    ax.add_patch(Rectangle((j, i), 1, 1, facecolor="#F7F7F7",
                                           edgecolor="#C9C9C9", hatch="///",
                                           lw=0, zorder=2))
                    ax.text(j + 0.5, i + 0.5, "n/a", ha="center", va="center",
                            fontsize=6.8, color="#8A8A8A", zorder=3,
                            bbox=dict(boxstyle="round,pad=0.22", fc="white",
                                      ec="none", alpha=0.9))
                else:
                    ax.text(j + 0.5, i + 0.5, f"{v:.1f}", ha="center",
                            va="center", fontsize=7,
                            color=_text_on(OI_BLUES(norm(v))))
        ax.set_xticks(np.arange(3) + 0.5, [f"{v:g}" for v in LS])
        ax.set_yticks(np.arange(3) + 0.5, [f"{v:g}" for v in DS])
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.tick_params(length=0, pad=3)
        ax.set_xlabel("Cilium length $L$ (mm)")
        if ax is axes[0]:
            ax.set_ylabel("Cilium diameter $d$ (mm)")
        n_pts = len(sub)
        ax.set_title(f"{en} ($n$ = {n_pts})", loc="left", pad=5)
        cax = make_axes_locatable(ax).append_axes("right", size="5.5%",
                                                  pad=0.07)
        cb = fig.colorbar(mesh, cax=cax)
        cb.outline.set_linewidth(0.6)
        cb.locator = MaxNLocator(nbins=5)
        cb.update_ticks()
        cb.ax.tick_params(labelsize=7, width=0.8, length=2.5)
        cb.set_label("$S$ (mT N$^{-1}$)", fontsize=7.5, labelpad=2)
    fig.tight_layout(pad=0.6, w_pad=1.4)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIGS, f"fig2e_geometry_sensitivity.{ext}"))
    print("fig2e written; tangential max",
          pts[(pts.direction == "切向") & (~pts.outlier)].S_mT_per_N.max())


def fatigue_bad_fraction():
    """bad-row fraction per checkpoint from raw hall file row counts."""
    fr = {}
    for s in ["S01", "S02", "S03"]:
        for hp in glob.glob(os.path.join(DATA, "磁数据", f"M-{s}",
                                         f"{s}_Z_*_hall_*.csv")):
            cyc = int(re.search(r"_Z_(\d+)_hall", hp).group(1))
            with open(hp) as fh:
                nrows = sum(1 for _ in fh) - 1
            fr[(s, cyc)] = nrows
    return fr


def fig2f():
    fat = pd.read_csv(os.path.join(PROC, "processed",
                                   "fatigue_sensitivity_vs_cycles.csv"))
    totals = fatigue_bad_fraction()
    fat["total_rows"] = fat.apply(lambda r: totals.get((r.sensor, r.cycles),
                                                       np.nan), axis=1)
    fat["bad_frac"] = fat.n_bad_rows / fat.total_rows
    resp = pd.read_csv(os.path.join(PROC, "figures", "data",
                                    "Fig8_all_events_stats.csv"))

    fig, axes = plt.subplots(1, 2, figsize=(5.6, 2.4),
                             gridspec_kw={"width_ratios": [1.5, 1]})
    axa = axes[0]
    cols = {"S01": OI["black"], "S02": OI["verm"], "S03": OI["blue"]}
    for s, g in fat.groupby("sensor"):
        g = g.sort_values("cycles")
        ok = g.bad_frac <= 0.09
        axa.plot(g.cycles / 1000, g.S_uT_per_N, "-", lw=1.0, alpha=0.55,
                 color=cols[s], zorder=2)
        axa.plot(g.cycles[ok] / 1000, g.S_uT_per_N[ok], "o", ms=4.2,
                 color=cols[s], mec="white", mew=0.5, ls="none",
                 label=f"{s}", zorder=3)
        axa.plot(g.cycles[~ok] / 1000, g.S_uT_per_N[~ok], "o", ms=4.2,
                 mfc="white", mec=cols[s], mew=0.9, ls="none", zorder=3)
    axa.set_xlabel("Compression cycles ($\\times 10^3$)")
    axa.set_ylabel("Sensitivity $\\Delta B_z/\\Delta F_z$ (µT N$^{-1}$)")
    axa.margins(x=0.03)
    lo, hi = axa.get_ylim()
    axa.set_ylim(lo, hi + 0.14 * (hi - lo))   # headroom band for legend/note
    axa.legend(frameon=False, loc="upper left", ncols=3,
               handlelength=1.0, handletextpad=0.15, columnspacing=0.9,
               borderaxespad=0.15, borderpad=0)
    axa.text(0.985, 0.975, "open = >9% bad packets",
             transform=axa.transAxes, ha="right", va="top",
             fontsize=6.8, color="#555555")
    axa.set_title("f4  Durability over 150k cycles", loc="left", pad=4)

    axb = axes[1]
    counts, bins, _ = axb.hist(resp.resp * 1000, bins=15, color=OI["blue"],
                               edgecolor="white", lw=0.5)
    med = resp.resp.median() * 1000
    fast = resp.resp.min() * 1000
    ymax = axb.get_ylim()[1]
    axb.axvline(med, color=OI["verm"], lw=1.2, zorder=3)
    axb.annotate(f"median {med:.0f} ms", (med, ymax * 0.97),
                 fontsize=6.8, color=OI["verm"], xytext=(4, 0),
                 textcoords="offset points", ha="left", va="top")
    axb.annotate(f"fastest\n{fast:.0f} ms",
                 xy=(bins[0] + 0.4 * (bins[1] - bins[0]), counts[0] + 0.3),
                 xytext=(fast + 0.045 * (bins[-1] - bins[0]), ymax * 0.66),
                 fontsize=6.8, color="#555555", ha="left", va="bottom",
                 linespacing=1.15,
                 arrowprops=dict(arrowstyle="-", lw=0.6, color="#999999",
                                 shrinkA=2, shrinkB=0))
    axb.set_xlabel("10–90% response time (ms)")
    axb.set_ylabel("Events")
    axb.set_title(f"f5  Tap response ($n$ = {len(resp)})", loc="left", pad=4)

    fig.tight_layout(pad=0.6)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIGS, f"fig2f_dynamics_durability.{ext}"))
    print("fig2f written; reliable-checkpoint range:",
          fat[fat.bad_frac <= 0.09].groupby("sensor").S_uT_per_N
             .agg(["min", "max"]).round(0).to_dict())


if __name__ == "__main__":
    fig2e()
    fig2f()

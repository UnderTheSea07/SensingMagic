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
    "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7,
    "xtick.labelsize": 6, "ytick.labelsize": 6, "legend.fontsize": 6,
    "axes.linewidth": 0.6, "pdf.fonttype": 42, "figure.dpi": 300,
})

LS, DS = [3.0, 5.0, 8.0], [0.5, 0.8, 1.0]


def fig2e():
    pts = pd.read_csv(os.path.join(PROC, "figures_paper", "data",
                                   "Fig2exp_experimental_points.csv"))
    fig, axes = plt.subplots(1, 2, figsize=(5.6, 2.6))
    for ax, cn, en in zip(axes, ["切向", "法向"], ["Tangential", "Normal"]):
        sub = pts[(pts.direction == cn) & (~pts.outlier)]
        med = sub.groupby(["L", "d"]).S_mT_per_N.median()
        grid = np.full((3, 3), np.nan)
        for (L, d), v in med.items():
            grid[DS.index(d), LS.index(L)] = v
        im = ax.imshow(grid, origin="lower", cmap="viridis", aspect="auto")
        for i, d in enumerate(DS):
            for j, L in enumerate(LS):
                v = grid[i, j]
                if np.isnan(v):
                    ax.text(j, i, "excl.", ha="center", va="center",
                            fontsize=6, color="0.4")
                else:
                    ax.text(j, i, f"{v:.1f}", ha="center", va="center",
                            fontsize=6.5,
                            color="white" if v < np.nanmax(grid) * 0.6 else "black")
        ax.set_xticks(range(3), [f"{v:g}" for v in LS])
        ax.set_yticks(range(3), [f"{v:g}" for v in DS])
        ax.set_xlabel("Cilium length $L$ (mm)")
        ax.set_ylabel("Cilium diameter $d$ (mm)")
        n_pts = len(sub)
        ax.set_title(f"{en} loading (median $S$, {n_pts} points)", loc="left")
        cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label("$S$ (mT N$^{-1}$)")
    fig.tight_layout(pad=0.5)
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
    cols = {"S01": "#000000", "S02": "#D55E00", "S03": "#0072B2"}
    for s, g in fat.groupby("sensor"):
        g = g.sort_values("cycles")
        ok = g.bad_frac <= 0.09
        axa.plot(g.cycles / 1000, g.S_uT_per_N, "-", lw=0.7, color=cols[s])
        axa.plot(g.cycles[ok] / 1000, g.S_uT_per_N[ok], "o", ms=4,
                 color=cols[s], label=f"{s}")
        axa.plot(g.cycles[~ok] / 1000, g.S_uT_per_N[~ok], "o", ms=4,
                 mfc="none", color=cols[s])
    axa.set_xlabel("Compression cycles ($\\times 10^3$)")
    axa.set_ylabel("Sensitivity $\\Delta B_z/\\Delta F_z$ (µT N$^{-1}$)")
    axa.legend(frameon=False, title="sample", loc="center left")
    axa.set_title("a  150k-cycle durability (open = >9% bad packets)",
                  loc="left")

    axb = axes[1]
    axb.hist(resp.resp * 1000, bins=15, color="#0072B2", alpha=0.85)
    med = resp.resp.median() * 1000
    fast = resp.resp.min() * 1000
    axb.axvline(med, color="#D55E00", lw=1)
    axb.annotate(f"median {med:.0f} ms", (med, axb.get_ylim()[1] * 0.9),
                 fontsize=6, color="#D55E00", xytext=(4, 0),
                 textcoords="offset points")
    axb.annotate(f"fastest {fast:.0f} ms", (fast, axb.get_ylim()[1] * 0.55),
                 fontsize=6, xytext=(4, 0), textcoords="offset points")
    axb.set_xlabel("10–90% response time (ms)")
    axb.set_ylabel("Events")
    axb.set_title(f"b  Tap response ($n$={len(resp)})", loc="left")

    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(pad=0.5)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIGS, f"fig2f_dynamics_durability.{ext}"))
    print("fig2f written; reliable-checkpoint range:",
          fat[fat.bad_frac <= 0.09].groupby("sensor").S_uT_per_N
             .agg(["min", "max"]).round(0).to_dict())


if __name__ == "__main__":
    fig2e()
    fig2f()

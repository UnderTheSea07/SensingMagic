#!/usr/bin/env python
"""figB_BF_calibration.py

Figure fig3c_BF_calibration: deltaB-F calibration of the magnetic-cilia patch
under MANUAL loading, with the ATI Nano17 force/torque sensor as reference.

Panel a: loading-branch trajectory of one manual press event
         (|dBz| in uT vs |Fz| in mN), with small arrowheads along the
         trajectory indicating the loading direction.
Panel b: plateau scatter across manual-loading trials (3 samples x 2 target
         depths x 20 trials), with ONE pooled linear fit. Slope k (uT/N) and
         R^2 are computed from the data in this script (nothing hardcoded).

Outputs:
  /Users/arielzhang/Desktop/SensingMagic/paper/figures/fig3c_BF_calibration.pdf
  /Users/arielzhang/Desktop/SensingMagic/paper/figures/fig3c_BF_calibration.png
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 7.5,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 6.8,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.major.size": 2.8,
        "ytick.major.size": 2.8,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "pdf.fonttype": 42,
        "figure.dpi": 300,
    }
)

# ---------------------------------------------------------------- data paths
LOADING_CSV = (
    "/Users/arielzhang/Desktop/磁毛传感器可靠性分析原始数据/"
    "处理结果与图表_20260717/figures/data/Fig5a_loading_branch.csv"
)
PLATEAU_CSV = (
    "/Users/arielzhang/Desktop/磁毛传感器可靠性分析原始数据/"
    "处理结果与图表_20260717/figures/data/Fig5b_plateau_scatter.csv"
)
OUT_DIR = "/Users/arielzhang/Desktop/SensingMagic/paper/figures"

BLUE = "#0072B2"    # depth -1 mm
VERMILLION = "#D55E00"  # depth -2 mm
BLACK = "#000000"
GREY = "#888888"
ANNOT = "#555555"

# ---------------------------------------------------------------- load data
lb = pd.read_csv(LOADING_CSV)
F_load_mN = np.abs(lb["Fz_N_loading"].to_numpy()) * 1e3  # N -> mN, magnitude
dB_load = np.abs(lb["dBz_uT_loading"].to_numpy())        # uT, magnitude

pl = pd.read_csv(PLATEAU_CSV)
pl["F_mN"] = np.abs(pl["F_plateau"]) * 1e3   # N -> mN, magnitude
pl["dB_uT"] = np.abs(pl["dB_plateau"])       # uT, magnitude

# ------------------------------------------------- pooled linear fit (panel b)
# Fit |dB| (uT) against |F| (N) so the slope is directly in uT/N.
F_all_N = pl["F_mN"].to_numpy() * 1e-3
dB_all = pl["dB_uT"].to_numpy()
fit = stats.linregress(F_all_N, dB_all)
k_uT_per_N = fit.slope
intercept_uT = fit.intercept
r2 = fit.rvalue ** 2
n_pts = len(pl)

# ---------------------------------------------------------------- figure
# width_ratios compensate for the legend placed to the right of panel b,
# so the two drawn axes end up visually balanced
fig, (ax_a, ax_b) = plt.subplots(
    1, 2, figsize=(7.0, 2.4), constrained_layout=True,
    gridspec_kw={"width_ratios": [1.0, 1.45]},
)

# ---- panel a: loading branch of one manual press event ----------------------
ax_a.plot(
    F_load_mN,
    dB_load,
    color=BLUE,
    lw=1.2,
    marker="o",
    ms=2.4,
    markevery=10,
    markerfacecolor=BLUE,
    markeredgecolor="white",
    markeredgewidth=0.5,
    zorder=3,
)

# small arrowheads along the trajectory indicating loading direction,
# placed at fixed fractions of the (normalised) arc length
xn = (F_load_mN - F_load_mN.min()) / np.ptp(F_load_mN)
yn = (dB_load - dB_load.min()) / np.ptp(dB_load)
arc = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(xn), np.diff(yn)))])
for frac in (0.28, 0.58, 0.86):
    i = int(np.searchsorted(arc, frac * arc[-1]))
    i = min(i, len(F_load_mN) - 4)
    ax_a.annotate(
        "",
        xy=(F_load_mN[i + 3], dB_load[i + 3]),
        xytext=(F_load_mN[i], dB_load[i]),
        arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=0.9,
                        mutation_scale=7, shrinkA=0, shrinkB=0),
        zorder=4,
    )

ax_a.set_xlabel(r"$|F_z|$ (mN)")
ax_a.set_ylabel(r"$|\Delta B_z|$ ($\mu$T)")
ax_a.set_title("a  Loading branch, single manual press", loc="left")
ax_a.spines[["top", "right"]].set_visible(False)
# note in the empty lower-right corner, clear of the trajectory
ax_a.text(0.97, 0.04, "manual loading\nforce from ATI reference",
          transform=ax_a.transAxes, ha="right", va="bottom",
          fontsize=6.8, color=ANNOT, linespacing=1.35)

# ---- panel b: plateau scatter + pooled fit ----------------------------------
depth_colors = {"-1mm": BLUE, "-2mm": VERMILLION}
depth_labels = {"-1mm": "depth 1 mm", "-2mm": "depth 2 mm"}
sample_markers = {1: "o", 2: "s", 3: "^"}

for depth, cdep in depth_colors.items():
    for sample, mk in sample_markers.items():
        sub = pl[(pl["depth"] == depth) & (pl["sample"] == sample)]
        ax_b.scatter(
            sub["F_mN"],
            sub["dB_uT"],
            s=13,
            marker=mk,
            facecolors=cdep,
            edgecolors="white",
            linewidths=0.5,
            alpha=0.9,
            zorder=3,
        )

# pooled fit line across the observed force range
Fline_N = np.linspace(F_all_N.min(), F_all_N.max(), 50)
ax_b.plot(
    Fline_N * 1e3,
    intercept_uT + k_uT_per_N * Fline_N,
    color=BLACK,
    lw=1.0,
    ls="--",
    zorder=2,
)

ax_b.set_xlabel(r"$|F_z|$ plateau (mN)")
ax_b.set_ylabel(r"$|\Delta B_z|$ plateau ($\mu$T)")
ax_b.set_title("b  Plateau amplitudes, manual-loading trials", loc="left")
ax_b.spines[["top", "right"]].set_visible(False)

# legend: depth colors + sample markers (grey) + fit line, in the empty
# margin to the right of the panel (the interior is fully occupied by data)
legend_handles = [
    plt.Line2D([], [], ls="none", marker="o", ms=3.2, color=BLUE,
               markeredgecolor="white", markeredgewidth=0.4,
               label=depth_labels["-1mm"]),
    plt.Line2D([], [], ls="none", marker="o", ms=3.2, color=VERMILLION,
               markeredgecolor="white", markeredgewidth=0.4,
               label=depth_labels["-2mm"]),
    plt.Line2D([], [], ls="none", marker="o", ms=3, color=GREY, label="sample 1"),
    plt.Line2D([], [], ls="none", marker="s", ms=3, color=GREY, label="sample 2"),
    plt.Line2D([], [], ls="none", marker="^", ms=3, color=GREY, label="sample 3"),
    plt.Line2D([], [], ls="--", lw=1.0, color=BLACK, label="pooled linear fit"),
]
ax_b.legend(handles=legend_handles, frameon=False, loc="center left",
            bbox_to_anchor=(1.01, 0.5), handletextpad=0.4,
            borderaxespad=0.0, labelspacing=0.45)

# y-headroom so the topmost scatter points clear the stats annotation:
# place the data maximum no higher than 82% of the axes height.
y0, _ = ax_b.get_ylim()
ax_b.set_ylim(y0, y0 + (dB_all.max() - y0) / 0.82)

# annotation: computed slope and R^2, honest wording
annot = (
    "pooled fit across manual-loading trials\n"
    rf"$k$ = {k_uT_per_N/1e3:.1f}$\times$10$^3$ $\mu$T/N,  "
    rf"$R^2$ = {r2:.2f}  ($n$ = {n_pts})"
)
ax_b.text(0.02, 0.98, annot, transform=ax_b.transAxes,
          ha="left", va="top", fontsize=6.8, color=ANNOT, zorder=5,
          linespacing=1.35,
          bbox=dict(facecolor="white", edgecolor="none", alpha=0.85,
                    boxstyle="square,pad=0.15"))

# ---------------------------------------------------------------- save
for ext in ("pdf", "png"):
    fig.savefig(f"{OUT_DIR}/fig3c_BF_calibration.{ext}", facecolor="white")

print(f"pooled fit: k = {k_uT_per_N:.1f} uT/N "
      f"(= {k_uT_per_N/1e3:.2f} x10^3 uT/N), intercept = {intercept_uT:.1f} uT, "
      f"R^2 = {r2:.3f}, n = {n_pts}")

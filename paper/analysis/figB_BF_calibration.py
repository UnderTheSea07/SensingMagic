#!/usr/bin/env python
"""figB_BF_calibration.py

Figure fig3c_BF_calibration: deltaB-F calibration of the magnetic-cilia patch
under MANUAL loading, with the ATI Nano17 force/torque sensor as reference.

Panel a: loading-branch trajectory of one manual press event
         (|dBz| in uT vs |Fz| in mN).
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
        "font.size": 7,
        "axes.labelsize": 7,
        "axes.titlesize": 7,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "legend.fontsize": 6,
        "axes.linewidth": 0.6,
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

BLUE = "#0072B2"   # depth -1 mm
ORANGE = "#D55E00" # depth -2 mm
BLACK = "#000000"
GREY = "#666666"

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
fig, (ax_a, ax_b) = plt.subplots(
    1, 2, figsize=(7.0, 2.4), constrained_layout=True
)

# ---- panel a: loading branch of one manual press event ----------------------
ax_a.plot(
    F_load_mN,
    dB_load,
    color=BLUE,
    lw=0.8,
    marker="o",
    ms=1.6,
    mew=0,
    zorder=3,
)
ax_a.set_xlabel(r"$|F_z|$ (mN)")
ax_a.set_ylabel(r"$|\Delta B_z|$ ($\mu$T)")
ax_a.set_title("a  Loading branch, single manual press", loc="left")
ax_a.spines[["top", "right"]].set_visible(False)
ax_a.text(0.03, 0.97, "manual loading\nforce from ATI reference",
          transform=ax_a.transAxes, ha="left", va="top",
          fontsize=6, color=GREY)

# ---- panel b: plateau scatter + pooled fit ----------------------------------
depth_colors = {"-1mm": BLUE, "-2mm": ORANGE}
depth_labels = {"-1mm": "target depth 1 mm", "-2mm": "target depth 2 mm"}
sample_markers = {1: "o", 2: "s", 3: "^"}

for depth, cdep in depth_colors.items():
    for sample, mk in sample_markers.items():
        sub = pl[(pl["depth"] == depth) & (pl["sample"] == sample)]
        ax_b.scatter(
            sub["F_mN"],
            sub["dB_uT"],
            s=7,
            marker=mk,
            facecolors=cdep,
            edgecolors="none",
            alpha=0.75,
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
    zorder=4,
)

ax_b.set_xlabel(r"$|F_z|$ plateau (mN)")
ax_b.set_ylabel(r"$|\Delta B_z|$ plateau ($\mu$T)")
ax_b.set_title("b  Plateau amplitudes, manual-loading trials", loc="left")
ax_b.spines[["top", "right"]].set_visible(False)

# legend: depth colors + sample markers (grey) + fit line
legend_handles = [
    plt.Line2D([], [], ls="none", marker="o", ms=3, color=BLUE,
               label=depth_labels["-1mm"]),
    plt.Line2D([], [], ls="none", marker="o", ms=3, color=ORANGE,
               label=depth_labels["-2mm"]),
    plt.Line2D([], [], ls="none", marker="o", ms=3, color=GREY, label="sample 1"),
    plt.Line2D([], [], ls="none", marker="s", ms=3, color=GREY, label="sample 2"),
    plt.Line2D([], [], ls="none", marker="^", ms=3, color=GREY, label="sample 3"),
    plt.Line2D([], [], ls="--", lw=1.0, color=BLACK, label="pooled linear fit"),
]
ax_b.legend(handles=legend_handles, frameon=False, loc="center left",
            bbox_to_anchor=(1.01, 0.5), handletextpad=0.4,
            borderaxespad=0.0, labelspacing=0.4)

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
          ha="left", va="top", fontsize=6, color=BLACK, zorder=5,
          bbox=dict(facecolor="white", edgecolor="none", alpha=0.85,
                    boxstyle="square,pad=0.15"))

# ---------------------------------------------------------------- save
for ext in ("pdf", "png"):
    fig.savefig(f"{OUT_DIR}/fig3c_BF_calibration.{ext}", facecolor="white")

print(f"pooled fit: k = {k_uT_per_N:.1f} uT/N "
      f"(= {k_uT_per_N/1e3:.2f} x10^3 uT/N), intercept = {intercept_uT:.1f} uT, "
      f"R^2 = {r2:.3f}, n = {n_pts}")

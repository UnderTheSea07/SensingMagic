#!/usr/bin/env python
"""Figure 3 (main text): quad-sensor (2x2 MLX90393) shear/press decoding.

One journal-grade composite (Nature double column, 7.2 in wide) that
consolidates the quad-sensor dataset results:

  a  protocol + array schematic -- pure vector art, explicitly labeled
     "schematic"; the cilia-patch footprint is drawn offset toward s2 and
     labeled "inferred" (position deduced from per-sensor baselines /
     z-sensitivities: nearest s2, farthest s1). Trial counts are computed
     at runtime from the feature tables.
  b  representative x-shear trial (the trial with the best force-magnetic
     alignment |r|): 12-channel baseline-subtracted time series as a
     4-row small multiple, magnetic-anchored left/right stage shading,
     press marker; window trimmed to press + ~6 cycles; shared y-scale.
     Raw data re-aligned by importing quadA_process_x (same pipeline).
  c  fingerprint matrix: 5 conditions x 12 channels; cell = per-trial
     stage mean averaged across trials (equal trial weighting), exactly
     as in quadB_decode.py.
  d  LDA 2-D projection of per-cycle samples (fit on ALL samples --
     visualization only, labeled as such) + LOTO confusion mini-panel.
     The reported accuracy comes exclusively from the leave-one-TRIAL-out
     loop run here (StandardScaler fit inside training folds only).
  e  z-press sensitivity: single-sensor control vs quad sensors
     (equal logger gain assumed; magnetic units are logger counts,
     nominally uT -- flagged on the axes).

Every number on the figure is computed at runtime from
  paper/analysis/out/quad_features_{x,y,z}.csv
  paper/analysis/out/quad_z_control_comparison.csv
and the raw representative-trial CSV. Nothing is hardcoded.

Output: paper/figures/fig3_quad_main.{pdf,png}
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrow, Patch, Rectangle
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------- paths
ROOT = "/Users/arielzhang/Desktop/SensingMagic"
OUT = f"{ROOT}/paper/analysis/out"
FIG = f"{ROOT}/paper/figures"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import quadA_process_x as qx  # noqa: E402  (alignment pipeline, reused as-is)

REP = "20260717_201742"  # representative x-shear trial (best alignment |r|)

# ---------------------------------------------------------------------- style
BLUE, GREEN, VERM = "#0072B2", "#009E73", "#D55E00"   # Okabe-Ito
SKY, AMBER, GREY = "#56B4E9", "#E69F00", "#888888"
INK, FAINT = "#1a1a1a", "#666666"
mpl.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 7.5,
    "axes.titlesize": 8, "axes.labelsize": 7.5,
    "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 6.8, "axes.linewidth": 0.8,
    "xtick.direction": "out", "ytick.direction": "out",
    "xtick.major.width": 0.8, "ytick.major.width": 0.8,
    "xtick.major.size": 2.8, "ytick.major.size": 2.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 300,
    "legend.frameon": False,
})
DIVCMAP = LinearSegmentedColormap.from_list("oi_div", [BLUE, "#FFFFFF", VERM])
SEQCMAP = LinearSegmentedColormap.from_list("oi_seq", ["#FFFFFF", BLUE])
AX_COLOR = {"x": BLUE, "y": GREEN, "z": VERM}            # panel b line colors
CLS_COLOR = {"x-shear": BLUE, "y-shear": VERM, "z-press": GREEN}

CH = [f"dB{a}{s}" for s in range(4) for a in "xyz"]

# ======================================================================= data
fx = pd.read_csv(f"{OUT}/quad_features_x.csv")
fy = pd.read_csv(f"{OUT}/quad_features_y.csv")
fz = pd.read_csv(f"{OUT}/quad_features_z.csv")
ctrl = pd.read_csv(f"{OUT}/quad_z_control_comparison.csv")
n_tr = {n: df.trial.nunique() for n, df in [("x", fx), ("y", fy), ("z", fz)]}
print(f"trials: x {n_tr['x']}, y {n_tr['y']}, z {n_tr['z']}; "
      f"rows {len(fx)}+{len(fy)}+{len(fz)}")

# ---- fingerprint matrix: per-trial stage mean, then grand mean --------------
def cond_mean(df, stage):
    per_trial = df[df.stage == stage].groupby("trial")[CH].mean()
    return per_trial.mean().values

conditions = [("x-shear L", fx, "left"), ("x-shear R", fx, "right"),
              ("y-shear L", fy, "left"), ("y-shear R", fy, "right"),
              ("z-press", fz, "down")]
M = np.vstack([cond_mean(df, st) for _, df, st in conditions])
cond_names = [c[0] for c in conditions]

# ---- decoding samples (identical construction to quadB_decode.py) -----------
def shear_diff(df):
    l = df[df.stage == "left"].set_index(["trial", "cycle"])[CH]
    r = df[df.stage == "right"].set_index(["trial", "cycle"])[CH]
    return (r - l).dropna()

sam = []
for lab, d in [("x-shear", shear_diff(fx)), ("y-shear", shear_diff(fy))]:
    t = d.reset_index()
    t["label"] = lab
    t["trial_id"] = lab[0] + ":" + t["trial"].astype(str)
    sam.append(t[["trial_id", "label"] + CH])
tz = fz[fz.stage == "down"].copy()
tz["label"] = "z-press"
tz["trial_id"] = "z:" + tz["trial"].astype(str)
sam.append(tz[["trial_id", "label"] + CH])
S = pd.concat(sam, ignore_index=True)
X, y_lab, groups = S[CH].values, S["label"].values, S["trial_id"].values
classes = ["x-shear", "y-shear", "z-press"]
n_cyc = {c: int(np.sum(y_lab == c)) for c in classes}
print(f"samples: {len(S)} cycles / {S.trial_id.nunique()} trials "
      f"({', '.join(f'{c} {n_cyc[c]}' for c in classes)})")

# ---- leave-one-TRIAL-out LDA (scaler fit inside training folds only) --------
yt, yp = [], []
for g in np.unique(groups):
    tr, te = groups != g, groups == g
    sc = StandardScaler().fit(X[tr])
    m = LinearDiscriminantAnalysis().fit(sc.transform(X[tr]), y_lab[tr])
    yt.extend(y_lab[te])
    yp.extend(m.predict(sc.transform(X[te])))
yt, yp = np.array(yt), np.array(yp)
conf = np.zeros((3, 3), int)
for a, b in zip(yt, yp):
    conf[classes.index(a), classes.index(b)] += 1
acc = np.trace(conf) / conf.sum()
print(f"LOTO LDA accuracy: {acc*100:.2f}% ({np.trace(conf)}/{conf.sum()})")

# ---- LDA projection for VISUALIZATION only ----------------------------------
sc_all = StandardScaler().fit(X)
lda_vis = LinearDiscriminantAnalysis(n_components=2).fit(
    sc_all.transform(X), y_lab)
P = lda_vis.transform(sc_all.transform(X))
print(f"LDA(viz) discriminant variance ratio: "
      f"{lda_vis.explained_variance_ratio_[0]:.3f}, "
      f"{lda_vis.explained_variance_ratio_[1]:.3f}")

# ---- z control --------------------------------------------------------------
s_ctrl = float(ctrl.loc[ctrl.row == "ctrl", "S_uT_per_N"].iloc[0])
s_quad = [float(ctrl.loc[ctrl.row == f"quad_s{s}", "S_uT_per_N"].iloc[0])
          for s in range(4)]
print(f"S_z ctrl {s_ctrl:.1f}, quad {['%.1f' % v for v in s_quad]}")

# ---- representative trial: re-align raw data via the quadA pipeline ---------
pairs, _, _ = qx.pair_files()
mp, fp = next((mp, fp) for mp, fp, _ in pairs if REP in mp)
rows_rep, trace, al, notes = qx.process_pair(mp, fp, REP)
assert trace is not None, f"representative trial {REP} failed alignment"
print(f"representative {REP}: r_align={al['r_align']:.3f}, "
      f"cycles={len(al['ext_left'])}; notes={notes if notes else 'none'}")
tm, dB = trace["tm"], trace["dB"]
t_press = al["t_press"]
extL, extR = al["ext_left"], al["ext_right"]
T_half = float(np.median(np.diff(np.sort(np.concatenate([extL, extR])))))

# =============================================================== figure layout
W, H = 7.2, 6.56
fig = plt.figure(figsize=(W, H))

def ax_in(x, y_top, w, h, **kw):
    """Axes from top-left corner, in inches."""
    return fig.add_axes([x / W, (H - y_top - h) / H, w / W, h / H], **kw)

def letter(x, y_top, s):
    fig.text(x / W, (H - y_top) / H, s, fontsize=10, fontweight="bold",
             family="sans-serif", ha="left", va="top", color="black")

# ---------------------------------------------------------------- panel a ---
letter(0.06, 0.06, "a")
axA = ax_in(0.06, 0.30, 2.82, 1.72)
axA.set_xlim(0, 2.82)
axA.set_ylim(0, 1.72)
axA.set_axis_off()
axA.text(2.80, 1.68, "schematic", ha="right", va="top", fontsize=6,
         style="italic", color=FAINT)

# array top view -----------------------------------------------------------
bc = (0.62, 0.78)                       # board center
axA.add_patch(Rectangle((bc[0] - 0.52, bc[1] - 0.52), 1.04, 1.04,
                        fc="#f4f4f4", ec=GREY, lw=0.8, zorder=1))
axA.add_patch(Circle((bc[0] - 0.12, bc[1] - 0.12), 0.38, fc=BLUE, alpha=0.07,
                     ec=BLUE, lw=0.8, ls=(0, (3, 2)), zorder=2))
chip_pos = {"s0": (-0.25, 0.25), "s1": (0.25, 0.25),
            "s2": (-0.25, -0.25), "s3": (0.25, -0.25)}
for name, (dx_, dy_) in chip_pos.items():
    cx, cy = bc[0] + dx_, bc[1] + dy_
    axA.add_patch(Rectangle((cx - 0.085, cy - 0.085), 0.17, 0.17,
                            fc="#dddddd", ec="#555555", lw=0.7, zorder=3))
    axA.text(cx, cy, name, ha="center", va="center", fontsize=6.5,
             color=INK, zorder=4)
axA.text(bc[0], 1.42, "2×2 MLX90393 array\n(top view)", ha="center",
         va="bottom", fontsize=6.8, color=INK, linespacing=1.15,
         multialignment="center")
axA.annotate("cilia patch footprint\n(inferred from baselines)",
             xy=(bc[0] - 0.34, bc[1] - 0.38), xytext=(0.10, 0.03),
             fontsize=6, color=BLUE, ha="left", va="bottom",
             arrowprops=dict(arrowstyle="-", color=BLUE, lw=0.6,
                             shrinkA=1, shrinkB=1))
# x/y axes arrows (bottom-right of the board)
ox, oy = bc[0] + 0.62, bc[1] - 0.52
axA.add_patch(FancyArrow(ox, oy, 0.18, 0, width=0.001, head_width=0.03,
                         head_length=0.04, color=INK, lw=0.6))
axA.add_patch(FancyArrow(ox, oy, 0, 0.18, width=0.001, head_width=0.03,
                         head_length=0.04, color=INK, lw=0.6))
axA.text(ox + 0.26, oy, "x", fontsize=6.5, ha="left", va="center", color=INK)
axA.text(ox, oy + 0.26, "y", fontsize=6.5, ha="center", va="bottom", color=INK)

# protocol timelines -------------------------------------------------------
tx0, tx1 = 1.62, 2.78
axA.text(tx0, 1.44, "shear trials (x or y)", fontsize=6.5, color=INK,
         va="bottom")
# z displacement: press 1 mm, hold, reset
zy, zstep = 1.30, 0.09
t = np.array([tx0, tx0 + 0.06, tx0 + 0.11, tx1 - 0.14, tx1 - 0.09, tx1])
z = np.array([zy, zy, zy - zstep, zy - zstep, zy, zy])
axA.plot(t, z, color=GREY, lw=0.9, solid_capstyle="round")
axA.text(tx0 + 0.14, zy + 0.015, "press 1 mm", fontsize=6, color=FAINT,
         va="bottom")
axA.text(tx1, zy + 0.015, "reset", fontsize=6, color=FAINT, va="bottom",
         ha="right")
# lateral displacement: +-2 mm, ~20 cycles
ly, lamp = 1.05, 0.085
tt = np.linspace(tx0 + 0.14, tx1 - 0.16, 300)
axA.plot(tt, ly + lamp * np.sin(2 * np.pi * 5.0 *
                                (tt - tt[0]) / (tt[-1] - tt[0])),
         color=GREY, lw=0.9)
axA.plot([tx0, tx0 + 0.14], [ly, ly], color=GREY, lw=0.9)
axA.plot([tx1 - 0.16, tx1], [ly, ly], color=GREY, lw=0.9)
axA.text((tx0 + tx1) / 2, ly - lamp - 0.035, "shear ±2 mm × ~20",
         fontsize=6, color=FAINT, ha="center", va="top")
# z trials: 2 mm press cycles
axA.text(tx0, 0.68, "z trials", fontsize=6.5, color=INK, va="bottom")
py, pamp = 0.56, 0.085
seg = np.linspace(tx0, tx1 - 0.10, 200)
sq = py - pamp * 0.5 * (1 - np.cos(2 * np.pi * 4.0 *
                                   (seg - seg[0]) / (seg[-1] - seg[0])))
axA.plot(seg, sq, color=GREY, lw=0.9)
axA.text((tx0 + tx1) / 2, py - pamp - 0.035, "press 2 mm × 20",
         fontsize=6, color=FAINT, ha="center", va="top")
axA.text(tx0, 0.05, f"machine-driven\ntrials: x {n_tr['x']} · "
         f"y {n_tr['y']} · z {n_tr['z']}", fontsize=6, color=FAINT,
         va="bottom", linespacing=1.3)

# ---------------------------------------------------------------- panel b ---
letter(3.02, 0.06, "b")
xb, wb = 3.46, 3.66
hb, gb = 0.40, 0.035
mask_pre = tm < t_press + 8.5
t0, t1 = -1.0, float(extR[min(5, len(extR) - 1)] - t_press + 0.55 * T_half)
win = (tm - t_press >= t0 - 0.05) & (tm - t_press <= t1 + 0.05)
ylo = float(dB[win].min())
yhi = float(dB[win].max())
pad = 0.06 * (yhi - ylo)
axs_b = []
for s in range(4):
    ax = ax_in(xb, 0.30 + s * (hb + gb), wb, hb)
    axs_b.append(ax)
    for k in range(7):          # one extra span so shading reaches the xlim
        if k < len(extL):
            ax.axvspan(extL[k] - T_half - t_press, extL[k] - t_press,
                       color=SKY, alpha=0.13, lw=0, zorder=0)
        if k < len(extR):
            ax.axvspan(extR[k] - T_half - t_press, extR[k] - t_press,
                       color=AMBER, alpha=0.13, lw=0, zorder=0)
    ax.axvline(0, color=GREY, lw=0.7, ls=(0, (2, 2)), zorder=1)
    for j, a in enumerate("xyz"):
        ax.plot(tm[win] - t_press, dB[win, 3 * s + j], color=AX_COLOR[a],
                lw=0.55, zorder=2)
    ax.set_xlim(t0, t1)
    ax.set_ylim(ylo - pad, yhi + pad)
    ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(3))
    ax.text(0.008, 0.80, f"s{s}", transform=ax.transAxes, fontsize=7,
            fontweight="bold", color=INK)
    if s < 3:
        ax.tick_params(labelbottom=False)
        ax.spines["bottom"].set_visible(False)
        ax.tick_params(axis="x", length=0)
axs_b[3].set_xlabel("Time from press (s)", labelpad=1.5)
fig.text((xb - 0.48) / W, (H - (0.30 + (4 * hb + 3 * gb) / 2)) / H,
         "ΔB (µT, nominal)", rotation=90, va="center", ha="center",
         fontsize=7.5)
# press marker + L/R tags in the top row
axs_b[0].text(-0.08, 1.06, "press", transform=axs_b[0].get_xaxis_transform(),
              ha="right", va="bottom", fontsize=6, color=FAINT)
axs_b[0].text(extL[0] - T_half / 2 - t_press, 1.06, "L",
              transform=axs_b[0].get_xaxis_transform(), ha="center",
              va="bottom", fontsize=6, color=FAINT)
axs_b[0].text(extR[0] - T_half / 2 - t_press, 1.06, "R",
              transform=axs_b[0].get_xaxis_transform(), ha="center",
              va="bottom", fontsize=6, color=FAINT)
axs_b[0].text(0.995, 0.80, f"x-shear trial (1 of {n_tr['x']})",
              transform=axs_b[0].transAxes, ha="right", fontsize=6,
              color=FAINT)
hnd = ([Line2D([], [], color=AX_COLOR[a], lw=1.1, label=f"$B_{a}$")
        for a in "xyz"]
       + [Patch(fc=SKY, alpha=0.30, label="left"),
          Patch(fc=AMBER, alpha=0.30, label="right")])
axs_b[0].legend(handles=hnd, ncol=5, loc="lower right",
                bbox_to_anchor=(1.0, 1.16), handlelength=1.0,
                columnspacing=0.7, handletextpad=0.4, borderaxespad=0)

# ---------------------------------------------------------------- panel c ---
letter(0.06, 2.50, "c")
axC = ax_in(0.84, 2.74, 5.68, 1.32)
vmax = float(np.abs(M).max())
axC.imshow(M, cmap=DIVCMAP, norm=TwoSlopeNorm(vmin=-vmax, vcenter=0,
                                              vmax=vmax), aspect="auto")
for j in range(1, 12):
    axC.axvline(j - 0.5, color="white", lw=2.2 if j % 3 == 0 else 0.9)
for i in range(1, 5):
    axC.axhline(i - 0.5, color="white", lw=2.2 if i == 4 else 0.9)
for i in range(5):
    for j in range(12):
        v = M[i, j]
        axC.text(j, i, f"{v:+.0f}", ha="center", va="center", fontsize=6,
                 color="white" if abs(v) > 0.55 * vmax else INK)
axC.set_xticks(range(12), [f"$B_{a}$" for _ in range(4) for a in "xyz"])
axC.set_yticks(range(5), cond_names)
axC.tick_params(length=0, pad=2.5)
for sp in axC.spines.values():
    sp.set_visible(False)
for s in range(4):
    axC.text(3 * s + 1, -0.62, f"s{s}", ha="center", va="bottom",
             fontsize=7, fontweight="bold", color=INK)
# colorbar axes derived from the heatmap axes so heights match exactly
posC = axC.get_position()
cax = fig.add_axes([6.56 / W, posC.y0, 0.09 / W, posC.height])
cb = fig.colorbar(axC.images[0], cax=cax)
cb.set_label("ΔB (µT, nominal)", fontsize=6.8)
cb.ax.tick_params(labelsize=6.2, width=0.8, length=2.4)
cb.outline.set_linewidth(0.6)

# ---------------------------------------------------------------- panel d ---
letter(0.06, 4.42, "d")
axD = ax_in(0.46, 4.66, 2.10, 1.50)
for c in classes:
    m_ = y_lab == c
    axD.scatter(P[m_, 0], P[m_, 1], s=8, c=CLS_COLOR[c], label=c,
                edgecolors="white", linewidths=0.35, alpha=0.9, zorder=3)
axD.set_xlabel("LD 1", labelpad=1.5)
axD.set_ylabel("LD 2", labelpad=1.5)
axD.margins(0.10)
axD.legend(loc="upper left", handletextpad=0.15, borderaxespad=0.15,
           labelspacing=0.25, markerscale=1.1)
axD.text(0.42, 0.02, "visualization projection;\naccuracy from "
         "leave-one-trial-out", transform=axD.transAxes, ha="center",
         va="bottom", fontsize=6, color=FAINT, multialignment="center")

# LOTO confusion mini-panel
axE = ax_in(3.12, 4.92, 1.06, 1.06)
confn = conf / conf.sum(axis=1, keepdims=True)
axE.imshow(confn, cmap=SEQCMAP, vmin=0, vmax=1)
# full 3x3 cell grid: light grey border on every cell so zero cells read
# as part of the matrix rather than floating numbers
for i in range(3):
    for j in range(3):
        axE.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fc="none",
                                ec="#c9c9c9", lw=0.6, zorder=3))
for i in range(3):
    for j in range(3):
        axE.text(j, i, f"{conf[i, j]}", ha="center", va="center",
                 fontsize=6.5, color="white" if confn[i, j] > 0.6 else INK)
short = ["x", "y", "z"]
axE.set_xticks(range(3), short)
axE.set_yticks(range(3), short)
axE.set_xlabel("predicted", fontsize=6.5, labelpad=1.5)
axE.set_ylabel("true", fontsize=6.5, labelpad=1.5)
axE.tick_params(length=0, labelsize=6.5, pad=2)
for sp in axE.spines.values():
    sp.set_visible(False)
axE.set_title(f"LOTO {acc*100:.1f}%", fontsize=7.5, pad=3)
axE.text(0.5, -0.34, f"{np.trace(conf)}/{conf.sum()} cycles, "
         f"{S.trial_id.nunique()} trials", transform=axE.transAxes,
         ha="center", va="top", fontsize=6, color=FAINT)

# ---------------------------------------------------------------- panel e ---
letter(4.42, 4.42, "e")
axF = ax_in(4.94, 4.66, 2.16, 1.50)
vals = [s_ctrl] + s_quad
names = ["ctrl", "s0", "s1", "s2", "s3"]
axF.bar(range(5), vals, color=["#000000", BLUE, BLUE, BLUE, BLUE],
        width=0.62, edgecolor="white", linewidth=0.5)
for i, v in enumerate(vals):
    axF.text(i, v + 0.015 * max(vals), f"{v:.0f}", ha="center", va="bottom",
             fontsize=6.2, color=INK)
axF.set_xticks(range(5), names, fontsize=6.8)
axF.set_ylabel("$S_z$ (µT N$^{-1}$, nominal)", labelpad=1.5)
axF.set_ylim(0, max(vals) * 1.16)
axF.text(2.5, -0.155, "array nodes s0–s3",
         transform=axF.get_xaxis_transform(), ha="center", va="top",
         fontsize=6, color=FAINT)
axF.text(0.98, 0.97, "equal logger gain assumed", transform=axF.transAxes,
         ha="right", va="top", fontsize=5.8, color=FAINT)

# ---------------------------------------------------------------------- save
for ext in ("pdf", "png"):
    out = f"{FIG}/fig3_quad_main.{ext}"
    fig.savefig(out, facecolor="white")
    print("wrote", out)
plt.close(fig)

#!/usr/bin/env python
"""Quad-sensor decoding synthesis (Fig. 3 companion analysis).

Inputs (produced by the per-direction alignment/feature pipelines):
  paper/analysis/out/quad_features_{x,y,z}.csv
      per-cycle plateau features: trial, cycle, stage, 12 baseline-subtracted
      magnetic channels dB{x,y,z}{0..3} (logger counts, nominally uT), stage-mean
      Fx/Fy/Fz (N) and per-cycle peak |F|.
  paper/analysis/out/quad_z_control_comparison.csv
      single-sensor control vs quad per-sensor z-press sensitivity (uT/N).

Outputs:
  paper/figures/fig3_quad_fingerprints.{pdf,png}  (fingerprint matrix + LDA scatter)
  paper/figures/fig3_quad_decoding.{pdf,png}      (LOTO confusion + force-dB corr + z control)
  console report: fingerprint-claim tests, leave-one-TRIAL-out decoding accuracy,
      per-direction force-field Pearson r, control sensitivity ratios.

Honesty notes baked in:
  - Every number is computed from the feature tables; nothing is assumed.
  - Trials already excluded upstream (3 x-trials, 1 y-pair + 1 y-trial) are simply
    absent from the CSVs; counts are reported.
  - Decoding uses leave-one-TRIAL-out: no cycle of a held-out trial ever appears
    in training; standardization is fit on the training fold only.
  - The LDA scatter is fit on ALL samples for visualization only and is labeled
    as such; the reported accuracy comes exclusively from the LOTO loop.
"""

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from scipy import stats
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# ----------------------------------------------------------------------------- paths
ROOT = "/Users/arielzhang/Desktop/SensingMagic"
OUT = f"{ROOT}/paper/analysis/out"
FIG = f"{ROOT}/paper/figures"

# ----------------------------------------------------------------------------- style
OI = ["#0072B2", "#D55E00", "#009E73", "#E69F00", "#56B4E9", "#CC79A7", "#000000", "#888888"]
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
})
DIVCMAP = LinearSegmentedColormap.from_list("oi_div", ["#0072B2", "#FFFFFF", "#D55E00"])
SEQCMAP = LinearSegmentedColormap.from_list("oi_seq", ["#FFFFFF", "#0072B2"])

CH = [f"dB{a}{s}" for s in range(4) for a in "xyz"]          # dBx0,dBy0,dBz0,...  (sensor-major)
CH_LABEL = [f"{a.upper()}{s}" for s in range(4) for a in "xyz"]

# ----------------------------------------------------------------------------- load
fx = pd.read_csv(f"{OUT}/quad_features_x.csv")
fy = pd.read_csv(f"{OUT}/quad_features_y.csv")
fz = pd.read_csv(f"{OUT}/quad_features_z.csv")
ctrl = pd.read_csv(f"{OUT}/quad_z_control_comparison.csv")

print("=" * 78)
print("QUAD-SENSOR DECODING SYNTHESIS")
print("=" * 78)
for name, df in [("x", fx), ("y", fy), ("z", fz)]:
    print(f"  {name}-trials: {df.trial.nunique()} trials, {len(df)} feature rows, "
          f"stages {sorted(df.stage.unique())}")

# =============================================================================
# 1. FINGERPRINT MATRIX  (signed mean dB per channel per condition)
# =============================================================================
# Per-trial mean first, then grand mean across trials (equal trial weighting).
def cond_mean(df, stage):
    per_trial = df[df.stage == stage].groupby("trial")[CH].mean()
    return per_trial.mean(), per_trial.std(), len(per_trial)

conditions = [
    ("x-shear L", fx, "left"), ("x-shear R", fx, "right"),
    ("y-shear L", fy, "left"), ("y-shear R", fy, "right"),
    ("z-press",   fz, "down"),
]
M = np.vstack([cond_mean(df, st)[0].values for _, df, st in conditions])
cond_names = [c[0] for c in conditions]

# Shear differentials (right - left) per cycle, per trial — used for claim tests
def shear_diff(df):
    l = df[df.stage == "left"].set_index(["trial", "cycle"])[CH]
    r = df[df.stage == "right"].set_index(["trial", "cycle"])[CH]
    return (r - l).dropna()

dx = shear_diff(fx)   # x-shear differential, 17 trials x 20 cycles
dy = shear_diff(fy)   # y-shear differential, 19 trials x 20 cycles

def diff_stats(d):
    per_trial = d.groupby("trial").mean()
    mean, sd = per_trial.mean(), per_trial.std()
    sem = sd / np.sqrt(len(per_trial))
    return mean, sd, sem, len(per_trial)

mx, sdx_, semx, ntx = diff_stats(dx)
my, sdy_, semy, nty = diff_stats(dy)

print("\n--- Shear differentials (right - left), mean +/- sd across trials ---")
print("x-shear (n=%d trials):" % ntx)
print("  " + "  ".join(f"{c}={mx[c]:+.0f}+-{sdx_[c]:.0f}" for c in CH))
print("y-shear (n=%d trials):" % nty)
print("  " + "  ".join(f"{c}={my[c]:+.0f}+-{sdy_[c]:.0f}" for c in CH))

# ---- quantitative test of the team's qualitative claims -----------------------
# 'flat'      : |mean| < 0.20 * max |same-axis differential|  AND |mean| < small
# 'together'  : sign(dX)==sign(dY), both significant (|mean| > 2*sem)
# 'opposite'  : sign(dX)!=sign(dY), both significant
FLAT_FRAC = 0.20

def test_flat(mean, sem, ch, axis_max):
    frac = abs(mean[ch]) / axis_max
    ok = frac < FLAT_FRAC
    return ok, f"{ch}={mean[ch]:+.0f} ({frac:.2f} of max same-axis |dB|) -> {'FLAT' if ok else 'NOT flat'}"

def test_pair(mean, sem, chx, chy, want_same):
    sig = (abs(mean[chx]) > 2 * sem[chx]) and (abs(mean[chy]) > 2 * sem[chy])
    same = np.sign(mean[chx]) == np.sign(mean[chy])
    ok = sig and (same == want_same)
    rel = "together" if same else "opposite"
    return ok, (f"{chx}={mean[chx]:+.0f}, {chy}={mean[chy]:+.0f} -> move {rel}"
                + ("" if sig else " (NOT significant)"))

claims = []
axmax_x = max(abs(mx[f"dBx{s}"]) for s in range(4))
axmax_y_ofx = max(abs(mx[f"dBy{s}"]) for s in range(4))
# x-shear claims
ok, msg = test_flat(mx, semx, "dBx0", axmax_x)
claims.append(("x-shear: sensor0 X ~unchanged", ok, msg))
ok, msg = test_pair(mx, semx, "dBx2", "dBy2", want_same=True)
claims.append(("x-shear: sensor2 X,Y together", ok, msg))
ok, msg = test_pair(mx, semx, "dBx1", "dBy1", want_same=False)
claims.append(("x-shear: sensor1 X,Y opposite", ok, msg))
ok, msg = test_pair(mx, semx, "dBx3", "dBy3", want_same=False)
claims.append(("x-shear: sensor3 X,Y opposite", ok, msg))
# y-shear claims
axmax_x_ofy = max(abs(my[f"dBx{s}"]) for s in range(4))
ok, msg = test_flat(my, semy, "dBx1", axmax_x_ofy)
claims.append(("y-shear: sensor1 X ~unchanged", ok, msg))
ok, msg = test_pair(my, semy, "dBx2", "dBy2", want_same=True)
claims.append(("y-shear: sensor2 X,Y together", ok, msg))
ok, msg = test_pair(my, semy, "dBx0", "dBy0", want_same=False)
claims.append(("y-shear: sensor0 X,Y opposite", ok, msg))
ok, msg = test_pair(my, semy, "dBx3", "dBy3", want_same=False)
claims.append(("y-shear: sensor3 X,Y opposite", ok, msg))

print("\n--- Team qualitative claims, quantified (on right-left differentials) ---")
n_ok = 0
for name, ok, msg in claims:
    n_ok += ok
    print(f"  [{'CONFIRMED' if ok else 'NOT SUPPORTED'}] {name}: {msg}")
print(f"  => {n_ok}/{len(claims)} claims confirmed")
# where the actual near-null channel sits for the two failed claims
frac_y_s1_ofy = abs(my["dBy1"]) / max(abs(my[f"dBy{s}"]) for s in range(4))
print(f"  note: y-shear near-null channel on sensor1 is Y (dBy1={my['dBy1']:+.0f}, "
      f"{frac_y_s1_ofy:.2f} of max |dY|), not X (dBx1={my['dBx1']:+.0f} = largest X)")
print(f"  note: x-shear sensor1 X is small but SAME sign as Y "
      f"(dBx1={mx['dBx1']:+.0f} vs dBy1={mx['dBy1']:+.0f})")

# =============================================================================
# 2. DIRECTION DECODING  (leave-one-TRIAL-out)
# =============================================================================
# Sample = one 12-channel per-cycle feature vector:
#   x/y trials: shear differential (right - left) of that cycle  -> 'x-shear'/'y-shear'
#   z trials  : down-plateau dB of that cycle                    -> 'z-press'
sam = []
for lab, d in [("x-shear", dx), ("y-shear", dy)]:
    t = d.reset_index()
    t["label"] = lab
    t["trial_id"] = lab[0] + ":" + t["trial"].astype(str)
    sam.append(t[["trial_id", "label"] + CH])
tz = fz[fz.stage == "down"].copy()
tz["label"] = "z-press"
tz["trial_id"] = "z:" + tz["trial"].astype(str)
sam.append(tz[["trial_id", "label"] + CH])
S = pd.concat(sam, ignore_index=True)
X = S[CH].values
y_lab = S["label"].values
groups = S["trial_id"].values
classes = ["x-shear", "y-shear", "z-press"]
print(f"\n--- Direction decoding: {len(S)} samples, {S.trial_id.nunique()} trials "
      f"({', '.join(f'{c}: {np.sum(y_lab==c)} cycles / {S[S.label==c].trial_id.nunique()} trials' for c in classes)}) ---")

def loto(model_factory):
    y_true, y_pred = [], []
    for g in np.unique(groups):
        tr, te = groups != g, groups == g
        sc = StandardScaler().fit(X[tr])
        m = model_factory().fit(sc.transform(X[tr]), y_lab[tr])
        y_true.extend(y_lab[te])
        y_pred.extend(m.predict(sc.transform(X[te])))
    return np.array(y_true), np.array(y_pred)

yt, yp = loto(lambda: LinearDiscriminantAnalysis())
conf = np.zeros((3, 3), int)
for a, b in zip(yt, yp):
    conf[classes.index(a), classes.index(b)] += 1
acc = np.trace(conf) / conf.sum()
per_class = {c: conf[i, i] / conf[i].sum() for i, c in enumerate(classes)}
print(f"  LDA  LOTO accuracy: {acc*100:.2f}%  ({np.trace(conf)}/{conf.sum()})")
for c in classes:
    print(f"    {c}: {per_class[c]*100:.2f}%")
print("  confusion (rows true, cols pred):\n", conf)

yt2, yp2 = loto(lambda: LogisticRegression(max_iter=5000, C=1.0))
acc2 = np.mean(yt2 == yp2)
print(f"  multinomial logistic LOTO accuracy (cross-check): {acc2*100:.2f}%")

# LDA projection for visualization ONLY (fit on all samples; accuracy above is LOTO)
sc_all = StandardScaler().fit(X)
lda_vis = LinearDiscriminantAnalysis(n_components=2).fit(sc_all.transform(X), y_lab)
P = lda_vis.transform(sc_all.transform(X))
evr = lda_vis.explained_variance_ratio_
print(f"  LDA(viz) discriminant variance ratio: {evr[0]:.3f}, {evr[1]:.3f}")

# =============================================================================
# 3. FORCE CORRESPONDENCE (shear cycles)
# =============================================================================
def force_diff(df, comp):
    l = df[df.stage == "left"].set_index(["trial", "cycle"])[comp]
    r = df[df.stage == "right"].set_index(["trial", "cycle"])[comp]
    return (r - l).dropna()

def corr_report(d_dB, d_F, direction):
    best_ch = d_dB.mean().abs().idxmax()
    a = d_dB[best_ch].abs()
    f = d_F.abs()
    idx = a.index.intersection(f.index)
    a, f = a.loc[idx], f.loc[idx]
    r_pool, p_pool = stats.pearsonr(f, a)
    # within-trial r (guards against between-trial confounds dominating)
    rw = []
    for t, grp in f.groupby(level="trial"):
        if len(grp) >= 5:
            rr = stats.pearsonr(grp.values, a.loc[t].values)[0]
            rw.append(rr)
    print(f"  {direction}: strongest channel {best_ch}; pooled Pearson r = {r_pool:.3f} "
          f"(p = {p_pool:.2g}, n = {len(f)}); within-trial r median {np.median(rw):.3f} "
          f"[{np.min(rw):.2f}..{np.max(rw):.2f}], n_trials = {len(rw)}")
    return best_ch, f, a, r_pool, p_pool

print("\n--- Force correspondence (per-cycle shear amplitude vs strongest dB channel) ---")
fdx = force_diff(fx, "Fx")
fdy = force_diff(fy, "Fy")
print(f"  x-shear |Fx| diff: {fdx.abs().mean():.3f} +- {fdx.abs().std():.3f} N; "
      f"y-shear |Fy| diff: {fdy.abs().mean():.3f} +- {fdy.abs().std():.3f} N "
      f"(force sensor quantization 0.05-0.16 N)")
bx, fxv, axv, rx_, px_ = corr_report(dx, fdx, "x-shear")
by, fyv, ayv, ry_, py_ = corr_report(dy, fdy, "y-shear")

# =============================================================================
# 4. Z CONTROL comparison
# =============================================================================
print("\n--- Z-press sensitivity, single-sensor control vs quad (uT/N) ---")
print(ctrl.to_string(index=False))
s_ctrl = ctrl.loc[ctrl.row == "ctrl", "S_uT_per_N"].iloc[0]
s_quad = ctrl[ctrl.row != "ctrl"]["S_uT_per_N"]
print(f"  ratio ctrl / best quad = {s_ctrl / s_quad.max():.2f}x; "
      f"/ mean quad = {s_ctrl / s_quad.mean():.2f}x; / worst = {s_ctrl / s_quad.min():.2f}x")

# =============================================================================
# FIGURE 1: fingerprint matrix + LDA scatter
# =============================================================================
fig = plt.figure(figsize=(7.2, 2.75))
gs = fig.add_gridspec(1, 2, width_ratios=[1.75, 1.0], wspace=0.45,
                      left=0.075, right=0.985, top=0.86, bottom=0.15)

# --- (a) fingerprint matrix
axA = fig.add_subplot(gs[0])
vmax = np.abs(M).max()
norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
im = axA.imshow(M, cmap=DIVCMAP, norm=norm, aspect="auto")
axA.set_xticks(range(12), CH_LABEL)
axA.set_yticks(range(len(cond_names)), cond_names)
for s in range(1, 4):                       # sensor group separators
    axA.axvline(3 * s - 0.5, color="#888888", lw=0.6)
axA.axhline(3.5, color="#888888", lw=0.6)   # shear block / press block separator
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        v = M[i, j]
        axA.text(j, i, f"{v:+.0f}", ha="center", va="center", fontsize=5.4,
                 color="white" if abs(v) > 0.55 * vmax else "black")
sec = axA.secondary_xaxis("top")
sec.set_xticks([1, 4, 7, 10], ["sensor 0", "sensor 1", "sensor 2", "sensor 3"])
sec.tick_params(length=0, labelsize=7)
sec.spines["top"].set_visible(False)
axA.tick_params(length=0)
for sp in axA.spines.values():
    sp.set_visible(False)
cb = fig.colorbar(im, ax=axA, fraction=0.035, pad=0.015)
cb.set_label(r"$\Delta B$ ($\mu$T)", fontsize=7)
cb.ax.tick_params(labelsize=6.2, width=0.8, length=2.4)
cb.outline.set_linewidth(0.6)
axA.set_title("Stage-mean field change per channel", fontsize=8, pad=16)
axA.text(-0.075, 1.14, "a", transform=axA.transAxes, fontsize=9, fontweight="bold")

# --- (b) LDA scatter
axB = fig.add_subplot(gs[1])
cls_color = {"x-shear": OI[0], "y-shear": OI[1], "z-press": OI[2]}
for c in classes:
    m_ = y_lab == c
    axB.scatter(P[m_, 0], P[m_, 1], s=9, c=cls_color[c], label=c,
                edgecolors="white", linewidths=0.35, alpha=0.9, zorder=3)
axB.set_xlabel("LD 1")
axB.set_ylabel("LD 2")
axB.set_title("Per-cycle samples, LDA projection", fontsize=8)
axB.margins(0.10)
axB.legend(frameon=False, loc="upper left", handletextpad=0.15,
           borderaxespad=0.2, labelspacing=0.25)
axB.text(0.47, 0.02, f"LOTO accuracy {acc*100:.1f}%\n({np.trace(conf)}/{conf.sum()} cycles,"
         f" {S.trial_id.nunique()} trials)", transform=axB.transAxes, ha="center",
         va="bottom", fontsize=6.8)
axB.text(-0.20, 1.14, "b", transform=axB.transAxes, fontsize=9, fontweight="bold")

fig.savefig(f"{FIG}/fig3_quad_fingerprints.pdf")
fig.savefig(f"{FIG}/fig3_quad_fingerprints.png")
plt.close(fig)

# =============================================================================
# FIGURE 2: confusion + force-dB correlation + z control
# =============================================================================
fig = plt.figure(figsize=(7.2, 2.15))
gs = fig.add_gridspec(1, 4, width_ratios=[0.95, 1.05, 1.05, 1.0], wspace=0.55,
                      left=0.065, right=0.99, top=0.86, bottom=0.21)

# --- (a) confusion matrix
axA = fig.add_subplot(gs[0])
confn = conf / conf.sum(axis=1, keepdims=True)
axA.imshow(confn, cmap=SEQCMAP, vmin=0, vmax=1)
short = ["x", "y", "z"]
axA.set_xticks(range(3), short)
axA.set_yticks(range(3), short)
axA.set_xlabel("predicted")
axA.set_ylabel("true")
for i in range(3):
    for j in range(3):
        axA.text(j, i, f"{conf[i, j]}", ha="center", va="center", fontsize=7,
                 color="white" if confn[i, j] > 0.6 else "black")
axA.set_title(f"LOTO confusion ({acc*100:.1f}%)", fontsize=8)
axA.tick_params(length=0)
for sp in axA.spines.values():
    sp.set_visible(False)
axA.text(-0.32, 1.12, "a", transform=axA.transAxes, fontsize=9, fontweight="bold")

# --- (b,c) force vs dB correlation
for ax, f_, a_, ch_, r_, lab, col, panel in [
        (fig.add_subplot(gs[1]), fxv, axv, bx, rx_, "x-shear", OI[0], "b"),
        (fig.add_subplot(gs[2]), fyv, ayv, by, ry_, "y-shear", OI[1], "c")]:
    ax.scatter(f_, a_, s=8, c=col, edgecolors="white", linewidths=0.3, alpha=0.85, zorder=3)
    k, b0 = np.polyfit(f_, a_, 1)
    xx = np.linspace(f_.min(), f_.max(), 10)
    ax.plot(xx, k * xx + b0, color="#000000", lw=0.9, zorder=4)
    ax.set_xlabel(f"per-cycle $|\\Delta F_{{{lab[0]}}}|$ (N)")
    ax.set_ylabel(f"$|\\Delta B_{{{ch_[2]}{ch_[3]}}}|$, R$-$L ($\\mu$T)")
    ax.set_title(f"{lab}: r = {r_:.2f}", fontsize=8)
    ax.text(-0.34, 1.12, panel, transform=ax.transAxes, fontsize=9, fontweight="bold")

# --- (d) z-sensitivity bars
axD = fig.add_subplot(gs[3])
names = ["ctrl", "s0", "s1", "s2", "s3"]
vals = [s_ctrl] + list(s_quad.values)
cols = ["#000000", OI[0], OI[0], OI[0], OI[0]]
axD.bar(range(5), vals, color=cols, width=0.62, edgecolor="white", linewidth=0.5)
for i, v in enumerate(vals):
    axD.text(i, v + 6, f"{v:.0f}", ha="center", va="bottom", fontsize=6.2)
axD.set_xticks(range(5), names, fontsize=6.8)
axD.text(0.62, -0.21, "quad sensors", transform=axD.transAxes, ha="center",
         va="top", fontsize=6.4, color="#555555")
axD.set_ylabel(r"$S_z$ ($\mu$T N$^{-1}$)")
axD.set_ylim(0, max(vals) * 1.18)
axD.set_title("z-press sensitivity", fontsize=8)
axD.text(-0.34, 1.12, "d", transform=axD.transAxes, fontsize=9, fontweight="bold")

fig.savefig(f"{FIG}/fig3_quad_decoding.pdf")
fig.savefig(f"{FIG}/fig3_quad_decoding.png")
plt.close(fig)

print(f"\nSaved: {FIG}/fig3_quad_fingerprints.(pdf|png), {FIG}/fig3_quad_decoding.(pdf|png)")

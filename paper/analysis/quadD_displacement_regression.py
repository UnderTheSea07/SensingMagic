#!/usr/bin/env python
"""Continuous shear-displacement regression from the quad-MLX90393 dataset.

Task: decode the lateral displacement u(t) of a flat platen (robot end-effector
face, pressed 1 mm into the cilia patch) that translates +/-2 mm laterally in a
machine-driven triangle profile (~20 cycles/trial), from the 12 baseline-
subtracted magnetic channels (4 sensors x 3 axes) alone.

Honesty / disclosed choices
---------------------------
- GROUND TRUTH IS THE COMMANDED TRAJECTORY, NOT A MEASURED ONE: the motion
  machine provides no encoder readback.  u(t) is reconstructed per trial as a
  piecewise-linear triangle wave anchored on the magnetic-derived cycle
  extremes (left/minus extreme = -2 mm, right/plus extreme = +2 mm, linear in
  between), i.e. the platen is ASSUMED to move at constant speed between
  extremes.  Any dwell at the reversals or speed non-uniformity becomes target
  error charged AGAINST the model, not hidden.
- The force files' programmatic stage labels are a planned schedule that the
  machine does not execute exactly (~10 ms/cycle drift); cycle timing is
  therefore taken from the magnetic data itself.  Stage labels are used only
  once per trial, at cycle 1 (where they are still accurate), to anchor the
  left-vs-right identity of the extremes.  This reuses the validated
  pairing/alignment/QC machinery of quadA_process_x.py / quadA_process_y.py
  (imported below) including their trial exclusions (x: 17/20, y: 19/21).
- The y settling trial (coherent ~300 uT first-trial offset, flagged by the
  quadA_process_y outlier screen) is excluded from the LOTO training/metric
  set, but is additionally evaluated as an extra held-out trial and reported
  separately (nothing hidden).
- This experiment is FLAT-PLATEN SHEAR-DISPLACEMENT DECODING, not
  point-contact localization: the whole platen face translates.
- Features are the 12 dB channels only (baseline-subtracted, low-passed 10 Hz,
  framed at ~50 Hz, restricted to the shear window between the first and last
  magnetic-anchored extreme, i.e. after the press and before the reset), then
  z-scored with training-fold statistics.  Magnetic units are logger counts
  (nominally uT); after standardization the features are unitless.
- Models: ridge regression (alpha by grouped CV over training trials only)
  and a small depth-limited random forest.  Validation: leave-one-TRIAL-out;
  metrics pooled over all held-out frames.  Trivial baseline: predicting the
  training-set mean displacement.

Outputs
-------
- paper/figures/fig3_quad_regression.{pdf,png}
- paper/analysis/out/quad_regression_frames.csv (frame-level LOTO predictions)
"""
import os
import re
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy import signal as sg
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import quadA_process_x as qx   # noqa: E402  (reused pairing/alignment/QC)
import quadA_process_y as qy   # noqa: E402

FIG_BASE = ("/Users/arielzhang/Desktop/SensingMagic/paper/figures/"
            "fig3_quad_regression")
OUT_CSV = ("/Users/arielzhang/Desktop/SensingMagic/paper/analysis/out/"
           "quad_regression_frames.csv")
CH_NAMES = qx.CH_NAMES
U_AMP = 2.0          # mm, commanded shear amplitude
FRAME_DT = 0.02      # s  -> 50 Hz frames
LP_FC = 10.0         # Hz feature low-pass
HI_FS = 500.0        # Hz uniform grid used for filtering
OKABE = qx.OKABE     # Okabe-Ito
COL_X, COL_Y, GREY = OKABE[0], OKABE[1], "#999999"

RNG_SEED = 0
ALPHAS = np.logspace(-2, 5, 15)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 7.5,
    "axes.titlesize": 8,
    "axes.labelsize": 7.5,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 6.8,
    "axes.linewidth": 0.8,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.major.size": 2.8,
    "ytick.major.size": 2.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.dpi": 300,
    "legend.frameon": False,
})


# ---------------------------------------------------------------- ground truth
def merge_extremes(t_left, t_right):
    """Merged, sorted extreme times with u values (-2 left, +2 right).
    Truncates at the first alternation violation (disclosed by caller)."""
    t = np.concatenate([t_left, t_right])
    u = np.concatenate([np.full(len(t_left), -U_AMP),
                        np.full(len(t_right), +U_AMP)])
    o = np.argsort(t)
    t, u = t[o], u[o]
    n = len(t)
    for i in range(1, len(t)):
        if u[i] == u[i - 1]:
            n = i
            break
    return t[:n], u[:n], n < len(t)


def detect_extremes_y(tm, dB, shift, segs, notes):
    """Magnetic-anchored cycle extremes for a y trial (same strategy as
    quadA_process_x.align_and_events, adapted to y_minus/y_plus labels):
    band-pass the strongest 0.6-1.5 Hz X/Y channel, take alternating
    derivative-sign flips as the position extremes, anchor minus/plus
    identity once at cycle 1 (end of first labelled y_minus + shift)."""
    fs = 100.0
    g = np.arange(tm[0], tm[-1], 0.01)
    dB_g = np.column_stack([np.interp(g, tm, dB[:, j]) for j in range(12)])
    band = np.column_stack([qx.bandpass(dB_g[:, j], fs) for j in range(12)])
    xy_idx = [j for j in range(12) if j % 3 != 2]
    pw = [np.mean(band[:, j] ** 2) for j in range(12)]
    ref = max(xy_idx, key=lambda j: pw[j])
    refb = band[:, ref]
    env = np.abs(sg.hilbert(refb))

    sh = [s for s in segs if s[0] in ("y_minus", "y_plus")]
    if not sh:
        notes.append("no y_minus/y_plus stages; trial unusable")
        return None
    w_on = sh[0][2] + shift - 0.30
    w_off = sh[-1][3] + shift + 0.30
    env_med = np.median(env[(g > w_on) & (g < w_off)])

    dref = np.gradient(refb)
    flips = np.flatnonzero(np.diff(np.sign(dref)) != 0)
    ext_t, ext_pol = [], []
    for i in flips:
        t = g[i]
        if not (w_on <= t <= w_off):
            continue
        if np.abs(refb[i]) < 0.5 * env[i] or np.abs(refb[i]) < 0.3 * env_med:
            continue
        pol = 1 if refb[i] > 0 else -1
        if ext_t and t - ext_t[-1] < 0.30:
            continue
        if ext_pol and pol == ext_pol[-1]:
            if np.abs(refb[i]) > np.abs(np.interp(ext_t[-1], g, refb)):
                ext_t[-1], ext_pol[-1] = t, pol
            continue
        ext_t.append(t)
        ext_pol.append(pol)
    ext_t, ext_pol = np.asarray(ext_t), np.asarray(ext_pol)
    if len(ext_t) < 6:
        notes.append(f"only {len(ext_t)} extremes found; trial unusable")
        return None

    t_pred = sh[0][3] + shift          # end of first y_minus travel
    j = int(np.argmin(np.abs(ext_t - t_pred)))
    if np.abs(ext_t[j] - t_pred) > 0.35:
        notes.append(f"cycle-1 identity anchor {np.abs(ext_t[j]-t_pred):.2f} s"
                     " from nearest extreme (>0.35 s); trial unusable")
        return None
    minus_pol = ext_pol[j]
    ext_minus = ext_t[ext_pol == minus_pol]
    ext_plus = ext_t[ext_pol == -minus_pol]
    ext_plus = ext_plus[ext_plus > ext_minus[0]]   # protocol: minus first
    n_lab = len([s for s in sh if s[0] == "y_minus"])
    if len(ext_minus) != n_lab or len(ext_plus) != n_lab:
        notes.append(f"extreme count m{len(ext_minus)}/p{len(ext_plus)} vs "
                     f"{n_lab} labelled cycles; truncated to matching prefix")
    n_cyc = min(len(ext_minus), len(ext_plus), n_lab)
    return ext_minus[:n_cyc], ext_plus[:n_cyc]


# ------------------------------------------------------------------- features
def frames_for_trial(tm, dB, ext_t, ext_u):
    """Low-passed (10 Hz) 12-channel features on ~50 Hz frames restricted to
    the shear window [first extreme, last extreme]; target u from the
    commanded triangle."""
    g = np.arange(tm[0], tm[-1], 1.0 / HI_FS)
    Xg = np.column_stack([np.interp(g, tm, dB[:, j]) for j in range(12)])
    b, a = sg.butter(4, LP_FC / (HI_FS / 2))
    Xf = sg.filtfilt(b, a, Xg, axis=0)
    ft = np.arange(ext_t[0], ext_t[-1], FRAME_DT)
    Xfr = np.column_stack([np.interp(ft, g, Xf[:, j]) for j in range(12)])
    u = np.interp(ft, ext_t, ext_u)
    return ft, Xfr, u


# ------------------------------------------------------------------ modelling
def fit_ridge_grouped_cv(Ztr, utr, gtr):
    """Ridge with alpha chosen by grouped CV over TRAINING trials only."""
    groups = np.unique(gtr)
    n_splits = min(5, len(groups))
    gkf = GroupKFold(n_splits=n_splits)
    scores = np.zeros(len(ALPHAS))
    for tr_i, va_i in gkf.split(Ztr, utr, gtr):
        for k, al in enumerate(ALPHAS):
            m = Ridge(alpha=al).fit(Ztr[tr_i], utr[tr_i])
            scores[k] += np.sqrt(np.mean((m.predict(Ztr[va_i])
                                          - utr[va_i]) ** 2))
    best = ALPHAS[int(np.argmin(scores))]
    return Ridge(alpha=best).fit(Ztr, utr), best


def make_rf():
    return RandomForestRegressor(n_estimators=100, max_depth=6,
                                 min_samples_leaf=20, random_state=RNG_SEED,
                                 n_jobs=-1)


def metrics(u, p):
    rmse = float(np.sqrt(np.mean((p - u) ** 2)))
    mae = float(np.mean(np.abs(p - u)))
    r2 = float(1.0 - np.sum((p - u) ** 2)
               / np.sum((u - np.mean(u)) ** 2))
    return rmse, mae, r2


def loto(data, order):
    """Leave-one-trial-out.  data[trial] = (ft, X, u).  Returns dict:
    preds[model][trial] = predicted u, plus per-fold ridge alphas and the
    trivial-baseline predictions."""
    preds = {"ridge": {}, "rf": {}, "base": {}}
    alphas_used = {}
    for held in order:
        tr = [t for t in order if t != held]
        Xtr = np.vstack([data[t][1] for t in tr])
        utr = np.concatenate([data[t][2] for t in tr])
        gtr = np.concatenate([np.full(len(data[t][2]), i)
                              for i, t in enumerate(tr)])
        Xte, ute = data[held][1], data[held][2]
        mu, sd = Xtr.mean(0), Xtr.std(0)
        sd[sd < 1e-9] = 1.0
        Ztr, Zte = (Xtr - mu) / sd, (Xte - mu) / sd
        ridge, al = fit_ridge_grouped_cv(Ztr, utr, gtr)
        alphas_used[held] = al
        preds["ridge"][held] = ridge.predict(Zte)
        preds["rf"][held] = make_rf().fit(Ztr, utr).predict(Zte)
        preds["base"][held] = np.full(len(ute), utr.mean())
    return preds, alphas_used


def pooled(data, preds, order, model):
    u = np.concatenate([data[t][2] for t in order])
    p = np.concatenate([preds[model][t] for t in order])
    return metrics(u, p)


# ------------------------------------------------------------- data ingestion
def load_x():
    print("=" * 72)
    print("x direction: reusing quadA_process_x pairing/alignment/QC")
    print("=" * 72)
    pairs, _, _ = qx.pair_files()
    data, excl, periods = {}, [], []
    for mp, fp, off in pairs:
        trial = re.search(r"(\d{8}_\d{6})", os.path.basename(mp)).group(1)
        rows, trace, al, notes = qx.process_pair(mp, fp, trial)
        for n in notes:
            print(f"  [{trial}] {n}")
        if rows is None:
            excl.append(trial)
            print(f"  [{trial}] EXCLUDED (quadA_process_x criteria)")
            continue
        t_e, u_e, trunc = merge_extremes(al["ext_left"], al["ext_right"])
        if trunc:
            print(f"  [{trial}] extreme alternation violated; truncated to "
                  f"{len(t_e)} extremes")
        ft, X, u = frames_for_trial(trace["tm"], trace["dB"], t_e, u_e)
        data[trial] = (ft, X, u)
        periods.append(2.0 * np.median(np.diff(t_e)))
        print(f"  [{trial}] {len(t_e)} extremes, {len(ft)} frames, "
              f"shear window {t_e[-1]-t_e[0]:.1f} s")
    print(f"x usable: {len(data)} / {len(pairs)} paired "
          f"(excluded: {', '.join(excl) if excl else 'none'})")
    print(f"x cycle period (from magnetic extremes): "
          f"{np.median(periods):.3f} s -> platen speed "
          f"{2*U_AMP/(np.median(periods)/2):.1f} mm/s (assumed constant)")
    return data


def load_y():
    print("=" * 72)
    print("y direction: reusing quadA_process_y pairing/alignment/QC")
    print("=" * 72)
    pairs, _, _ = qy.pair_files()
    kept, excl, feat_rows, traces = [], [], [], {}
    for mp, fp, off in pairs:
        trial = re.search(r"(\d{8}_\d{6})", os.path.basename(mp)).group(1)
        rows, trace, al, anom = qy.process_pair(mp, fp, trial)
        for n in anom:
            print(f"  [{trial}] {n}")
        if rows is None or not np.isfinite(al["r"]) or al["r_pos"] < 0.6:
            excl.append(trial)
            print(f"  [{trial}] EXCLUDED (quadA_process_y criteria)")
            continue
        kept.append(trial)
        feat_rows += rows
        traces[trial] = trace

    # same fault / settling screen as quadA_process_y (thresholds identical)
    feats = pd.DataFrame(feat_rows)
    lm = feats[feats["stage"] == "left"].groupby("trial")[CH_NAMES].mean()
    med = lm.median(axis=0)
    dev = (lm - med).abs()
    sensor_dev = pd.DataFrame(
        {s: dev[[f"dB{a}{s}" for a in "xyz"]].max(axis=1) for s in range(4)})
    fault, settling = [], []
    for t in lm.index:
        sd_t = sensor_dev.loc[t]
        if (sd_t > 300).any() and (sd_t < 30).sum() >= 2:
            fault.append(t)
            print(f"  [{t}] sensor fault (single-sensor deviation "
                  f"{sd_t.max():.0f} uT); EXCLUDED")
        elif dev.loc[t].max() > 100:
            settling.append(t)
            print(f"  [{t}] settling trial (coherent offset, max channel "
                  f"deviation {dev.loc[t].max():.0f} uT vs cross-trial "
                  "median); held out of LOTO set, evaluated separately")
    kept = [t for t in kept if t not in fault]

    data, periods = {}, []
    for trial in kept:
        trace = traces[trial]
        notes = []
        ext = detect_extremes_y(trace["tm"], trace["dB"], trace["shift"],
                                trace["segs"], notes)
        for n in notes:
            print(f"  [{trial}] {n}")
        if ext is None:
            excl.append(trial)
            print(f"  [{trial}] EXCLUDED (no usable magnetic extremes)")
            continue
        t_e, u_e, trunc = merge_extremes(ext[0], ext[1])
        if trunc:
            print(f"  [{trial}] extreme alternation violated; truncated to "
                  f"{len(t_e)} extremes")
        ft, X, u = frames_for_trial(trace["tm"], trace["dB"], t_e, u_e)
        data[trial] = (ft, X, u)
        periods.append(2.0 * np.median(np.diff(t_e)))
        print(f"  [{trial}] {len(t_e)} extremes, {len(ft)} frames, "
              f"shear window {t_e[-1]-t_e[0]:.1f} s")
    print(f"y usable: {len(data)} / {len(pairs)} paired "
          f"(excluded: {', '.join(excl) if excl else 'none'}; "
          f"settling: {', '.join(settling) if settling else 'none'})")
    if periods:
        print(f"y cycle period (from magnetic extremes): "
              f"{np.median(periods):.3f} s -> platen speed "
              f"{2*U_AMP/(np.median(periods)/2):.1f} mm/s (assumed constant)")
    return data, settling


def parity_check(data, csv_path, name, extra_ok=()):
    """Verify this script keeps exactly the trials the quadA pipeline kept."""
    if not os.path.exists(csv_path):
        print(f"[parity] {csv_path} not found; skipping check")
        return
    ref = {str(t) for t in pd.read_csv(csv_path)["trial"].unique()}
    got = set(data.keys()) | set(extra_ok)
    if ref == got:
        print(f"[parity] {name}: trial set matches quadA output "
              f"({len(ref)} trials)")
    else:
        print(f"[parity] {name}: MISMATCH  only-here={sorted(got-ref)}  "
              f"only-quadA={sorted(ref-got)}")


# --------------------------------------------------------------------- figure
def pick_representative(data, preds, order, model):
    """Held-out trial with median RMSE (honest representative)."""
    rmses = [(np.sqrt(np.mean((preds[model][t] - data[t][2]) ** 2)), t)
             for t in order]
    rmses.sort()
    return rmses[len(rmses) // 2][1]


def make_figure(dx, px, ox, dy, py, oy, model, mets, alt_mets, base_rmse,
                model_label):
    fig = plt.figure(figsize=(7.2, 2.55))
    # panel c's slot is widened so its square (box_aspect=1) parity plot can
    # use the full row height while keeping a true 45-degree y=x diagonal
    gs = fig.add_gridspec(1, 4, width_ratios=[1.45, 1.45, 1.14, 0.85],
                          wspace=0.52, left=0.065, right=0.985,
                          top=0.78, bottom=0.345)

    # panels a, b: held-out trial traces (~8 s)
    for k, (dd, pp, order, col, lab) in enumerate(
            [(dx, px, ox, COL_X, "x"), (dy, py, oy, COL_Y, "y")]):
        ax = fig.add_subplot(gs[0, k])
        rep = pick_representative(dd, pp, order, model)
        ft, X, u = dd[rep]
        p = pp[model][rep]
        t0 = ft[0] + 5.0
        w = (ft >= t0) & (ft <= t0 + 8.0)
        ax.plot(ft[w] - t0, u[w], color=GREY, lw=1.5, solid_capstyle="round")
        ax.plot(ft[w] - t0, p[w], color=col, lw=0.9)
        ax.set_ylim(-2.9, 2.9)
        ax.set_yticks([-2, 0, 2])
        ax.set_xlim(0, 8)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("$u_%s$ (mm)" % lab)
        rm = np.sqrt(np.mean((p - u) ** 2))
        ax.set_title(f"{'ab'[k]}   {lab}-shear, held-out trial",
                     loc="left", fontsize=8, pad=3)
        ax.text(0.985, 0.985, f"RMSE {rm:.2f} mm", transform=ax.transAxes,
                fontsize=6.0, color="#444444", ha="right", va="top")
        print(f"panel {'ab'[k]} representative {lab} trial: {rep} "
              f"(held-out {model_label} RMSE {rm:.3f} mm)")

    # shared legend above panels a-b (truth honesty stated in the legend)
    handles = [
        Line2D([], [], color=GREY, lw=1.5,
               label="commanded $u$ (no encoder readback)"),
        Line2D([], [], color=COL_X, lw=1.1,
               label="predicted $\\hat{u}_x$ (LOTO)"),
        Line2D([], [], color=COL_Y, lw=1.1,
               label="predicted $\\hat{u}_y$ (LOTO)"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3,
               bbox_to_anchor=(0.5, 1.005), handlelength=1.3,
               columnspacing=1.2, handletextpad=0.5, borderaxespad=0.0)

    # panel c: predicted vs true
    axc = fig.add_subplot(gs[0, 2])
    lim = 3.0
    axc.plot([-lim, lim], [-lim, lim], color="#cccccc", lw=0.7, ls="--",
             zorder=0)
    for dd, pp, order, col in [(dx, px, ox, COL_X), (dy, py, oy, COL_Y)]:
        u = np.concatenate([dd[t][2] for t in order])
        p = np.concatenate([pp[model][t] for t in order])
        axc.scatter(u[::4], p[::4], s=1.0, c=col, alpha=0.10, lw=0,
                    rasterized=True)
    axc.set_xlim(-lim, lim)
    axc.set_ylim(-lim, lim)
    axc.set_box_aspect(1)          # square panel: equal mm spans -> 45deg y=x
    axc.set_xticks([-2, 0, 2])
    axc.set_yticks([-2, 0, 2])
    axc.set_xlabel("Commanded $u$ (mm)")
    axc.set_ylabel("Predicted $u$ (mm)")
    axc.set_title("c   All held-out frames", loc="left", fontsize=8, pad=3)
    (rx, mx, r2x), (ry, my, r2y) = mets
    # RMSE/MAE block top-left (clear of the diagonal), R^2 block lower right
    # (empty triangle below the diagonal); keeps text off the y=x line, the
    # scatter cloud, and inside the axes box
    axc.text(0.03, 0.985, f"x: {rx:.2f} / {mx:.2f} mm",
             transform=axc.transAxes, fontsize=6.0, color=COL_X, va="top")
    axc.text(0.03, 0.885, f"y: {ry:.2f} / {my:.2f} mm",
             transform=axc.transAxes, fontsize=6.0, color=COL_Y, va="top")
    axc.text(0.03, 0.785, "RMSE / MAE", transform=axc.transAxes,
             fontsize=5.4, color="#666666", va="top")
    axc.text(0.97, 0.125, f"$R^2$ {r2x:.3f}", transform=axc.transAxes,
             fontsize=6.0, color=COL_X, ha="right", va="bottom")
    axc.text(0.97, 0.030, f"$R^2$ {r2y:.3f}", transform=axc.transAxes,
             fontsize=6.0, color=COL_Y, ha="right", va="bottom")

    # panel d: residual histograms
    axd = fig.add_subplot(gs[0, 3])
    bins = np.arange(-1.6, 1.6 + 1e-9, 0.08)
    for dd, pp, order, col, lab in [(dx, px, ox, COL_X, "x"),
                                    (dy, py, oy, COL_Y, "y")]:
        u = np.concatenate([dd[t][2] for t in order])
        p = np.concatenate([pp[model][t] for t in order])
        axd.hist(p - u, bins=bins, histtype="step", color=col, lw=0.9,
                 density=True, label=lab)
    axd.axvline(0, color="#cccccc", lw=0.7, zorder=0)
    axd.set_xlabel("Residual (mm)")
    axd.set_ylabel("Density (mm$^{-1}$)")
    axd.set_xlim(-1.6, 1.6)
    axd.set_title("d   Residuals", loc="left", fontsize=8, pad=3)
    axd.legend(loc="upper right", handlelength=1.0, handletextpad=0.45)

    (arx, _, ar2x), (ary, _, ar2y) = alt_mets
    alt_name = "random forest" if model == "ridge" else "ridge"
    fig.text(0.065, 0.100,
             f"Leave-one-trial-out over {len(ox)} x- and {len(oy)} y-trials; "
             f"{model_label} shown ({alt_name}: x {arx:.2f} mm, y {ary:.2f} mm"
             f" RMSE; mean-prediction baseline {base_rmse:.2f} mm).",
             fontsize=5.8, color="#444444", va="bottom")
    fig.text(0.065, 0.058,
             "Truth is the commanded platen trajectory (no encoder readback);"
             " constant speed between magnetically anchored extremes assumed.",
             fontsize=5.8, color="#444444", va="bottom")
    fig.text(0.065, 0.016,
             "Flat-platen shear-displacement decoding, not point-contact "
             "localization.",
             fontsize=5.8, color="#444444", va="bottom")

    fig.savefig(FIG_BASE + ".png")
    fig.savefig(FIG_BASE + ".pdf")
    plt.close(fig)


# ----------------------------------------------------------------------- main
def main():
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    os.makedirs(os.path.dirname(FIG_BASE), exist_ok=True)

    dx = load_x()
    dy_all, settling = load_y()
    parity_check(dx, qx.OUT_CSV, "x")
    parity_check(dy_all, qy.OUT_CSV, "y")

    ox = sorted(dx.keys())
    oy = sorted(t for t in dy_all if t not in settling)
    dy = {t: dy_all[t] for t in oy}

    print("=" * 72)
    print("leave-one-trial-out regression")
    print("=" * 72)
    px, ax_used = loto(dx, ox)
    py, ay_used = loto(dy, oy)
    print(f"ridge alphas chosen (grouped CV): "
          f"x median {np.median(list(ax_used.values())):.3g}, "
          f"y median {np.median(list(ay_used.values())):.3g}")

    results = {}
    for name, dd, pp, order in [("x", dx, px, ox), ("y", dy, py, oy)]:
        nfr = sum(len(dd[t][2]) for t in order)
        print(f"\n{name}-direction: {len(order)} trials, {nfr} held-out "
              f"frames ({1/FRAME_DT:.0f} Hz)")
        for model in ("ridge", "rf", "base"):
            rmse, mae, r2 = pooled(dd, pp, order, model)
            results[(name, model)] = (rmse, mae, r2)
            tag = {"ridge": "ridge", "rf": "random forest",
                   "base": "mean baseline"}[model]
            print(f"  {tag:14s}: RMSE {rmse:.3f} mm  MAE {mae:.3f} mm  "
                  f"R^2 {r2:.3f}")
        for model in ("ridge", "rf"):
            per = [(t, float(np.sqrt(np.mean((pp[model][t]
                                              - dd[t][2]) ** 2))))
                   for t in order]
            worst = max(per, key=lambda z: z[1])
            best = min(per, key=lambda z: z[1])
            print(f"  {model}: per-trial RMSE {best[1]:.3f} (best {best[0]})"
                  f" .. {worst[1]:.3f} (worst {worst[0]})")

    # settling trial: extra held-out evaluation, disclosed separately
    for t in settling:
        if t not in dy_all:
            continue
        Xtr = np.vstack([dy[q][1] for q in oy])
        utr = np.concatenate([dy[q][2] for q in oy])
        gtr = np.concatenate([np.full(len(dy[q][2]), i)
                              for i, q in enumerate(oy)])
        mu, sd = Xtr.mean(0), Xtr.std(0)
        sd[sd < 1e-9] = 1.0
        Ztr = (Xtr - mu) / sd
        Zte = (dy_all[t][1] - mu) / sd
        ute = dy_all[t][2]
        ridge, _ = fit_ridge_grouped_cv(Ztr, utr, gtr)
        rf = make_rf().fit(Ztr, utr)
        rr = metrics(ute, ridge.predict(Zte))
        rr2 = metrics(ute, rf.predict(Zte))
        print(f"\nsettling trial {t} (excluded from LOTO set, evaluated "
              f"against a model trained on the other {len(oy)} y-trials):")
        print(f"  ridge         : RMSE {rr[0]:.3f} mm  MAE {rr[1]:.3f} mm  "
              f"R^2 {rr[2]:.3f}")
        print(f"  random forest : RMSE {rr2[0]:.3f} mm  MAE {rr2[1]:.3f} mm  "
              f"R^2 {rr2[2]:.3f}")

    # primary model for the figure = better mean pooled RMSE
    mean_rmse = {m: 0.5 * (results[("x", m)][0] + results[("y", m)][0])
                 for m in ("ridge", "rf")}
    model = min(mean_rmse, key=mean_rmse.get)
    alt = "rf" if model == "ridge" else "ridge"
    model_label = {"ridge": "ridge", "rf": "random forest"}[model]
    print(f"\nprimary model for figure: {model_label} "
          f"(mean RMSE {mean_rmse[model]:.3f} vs {mean_rmse[alt]:.3f} mm)")
    base_rmse = 0.5 * (results[("x", "base")][0] + results[("y", "base")][0])

    make_figure(dx, px, ox, dy, py, oy, model,
                (results[("x", model)], results[("y", model)]),
                (results[("x", alt)], results[("y", alt)]),
                base_rmse, model_label)
    print(f"figure -> {FIG_BASE}.png/.pdf")

    # frame-level LOTO predictions for downstream reuse
    rows = []
    for name, dd, pp, order in [("x", dx, px, ox), ("y", dy, py, oy)]:
        for t in order:
            ft, X, u = dd[t]
            rows.append(pd.DataFrame({
                "direction": name, "trial": t, "t_s": ft, "u_true_mm": u,
                "u_ridge_mm": pp["ridge"][t], "u_rf_mm": pp["rf"][t]}))
    pd.concat(rows, ignore_index=True).to_csv(OUT_CSV, index=False)
    print(f"frame predictions -> {OUT_CSV}")


if __name__ == "__main__":
    main()

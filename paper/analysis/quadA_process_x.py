#!/usr/bin/env python
"""Process quad-MLX90393 x-direction shear trials (press 1 mm, shear +/-2 mm in x).

Pipeline
--------
1. Pair magnetic (mlx_v4_quad_*.csv, filename = LOCAL time UTC+8) with force
   (2026*.csv, filename = UTC) files in filename order; verify the constant
   filename-time offset (force_UTC - magnetic_UTC); exclude pairs deviating
   > 10 s from the median offset.
2. Per pair: baseline-subtract the 12 magnetic channels (dB); align force ->
   magnetic time; extract per-cycle left/right extremes; record alignment r.
3. Per cycle/stage (down / left / right): mean dB (12 ch) over the stage
   plateau + mean baseline-subtracted Fx/Fy/Fz + per-cycle peak |F|.
4. Tidy table -> paper/analysis/out/quad_features_x.csv
5. Diagnostic figure -> paper/figures/fig3_quad_x_overview.{pdf,png}

Alignment strategy (all statements below verified on this dataset)
------------------------------------------------------------------
The force 'stage' labels are a software schedule that the motion machine does
not follow exactly: the labelled cycle period is 1.013 s but BOTH physical
streams (force Fx and magnetic) show 1.023-1.024 s, i.e. the labels drift
~10 ms/cycle and are ~0.2-0.25 s late-cycle wrong by trial end.  A 0.25 s
window error on a 1.02 s sinusoid-like swing samples mid-travel instead of
the extreme, so label windows + constant lag are NOT usable for late cycles.
Instead:

- Global lag L0 (mag_time = force_time + L0) is anchored on the press event:
  the labelled 'down' onset (accurate at trial start, before drift builds)
  vs the large sustained departure of low-passed |dB| (the 1 mm press).  A
  small earlier step (~tens of uT) present in most trials is the machine's
  light-touch APPROACH, executed BEFORE force recording starts; the detector
  thresholds relative to the shear amplitude to skip it.
- Cycle timing is taken from the magnetic data itself: the strongest
  0.6-1.5 Hz channel is band-passed and its alternating extremes are the
  left/right stage plateaus (position extremes).  Left-vs-right identity is
  anchored once, at cycle 1 (where labels are still accurate): the extreme
  nearest to (label end of first 'left' stage + L0) defines the 'left'
  polarity; identities then alternate.
- Cross-check: signed cross-correlation of band-passed Fx against the
  magnetic reference channel (100 Hz grid, wide search, half-period alias
  rejected via the press-event lag).  Reported r is |r| of that force-vs-
  magnetic correlation; L_fit - L0 (includes the mechanical phase of Fx,
  friction vs elastic) is reported as QC only and not used for timing.
- 'down' plateau: [t_press + 0.20 s, t_press + 0.40 s], i.e. the pressed
  hold between the press ramp and the first shear.  The machine starts
  shearing ~0.5 s after the press completes, so a slight contamination by
  the first leftward travel is possible (disclosed, not hidden).
- Stage plateau window: +/-0.12 s around each extreme (the dwell/reversal).
- Magnetic baseline: pre-APPROACH window (first sustained departure of |dB|
  above 6 MAD, capped at 3 s).  Force baseline: 'idle' stage median (in
  these x files force recording genuinely starts pre-press, so idle is a
  true zero-force reference).
- Magnetic units are the counts recorded by the logger (nominally uT).
- Force files' embedded unix system_time is 8 h behind the filename time
  (acquisition-PC timezone quirk); pairing uses filename times only.
"""
import glob
import os
import re
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from scipy import signal as sg

# ----------------------------------------------------------------------------
ROOT = ("/private/tmp/claude-502/-Users-arielzhang-Desktop-SensingMagic/"
        "617d7206-542d-4a74-bd36-6e9400bc4ca0/scratchpad/4sensors/4_sensors/"
        "x方向-左右各2mm")
MAG_DIR = os.path.join(ROOT, "磁数据")
FRC_DIR = os.path.join(ROOT, "力数据")
OUT_CSV = "/Users/arielzhang/Desktop/SensingMagic/paper/analysis/out/quad_features_x.csv"
FIG_BASE = "/Users/arielzhang/Desktop/SensingMagic/paper/figures/fig3_quad_x_overview"

CH_NAMES = [f"dB{a}{s}" for s in range(4) for a in "xyz"]  # dBx0..dBz3
RAW_COLS = [f"{A}{s}" for s in range(4) for A in "XYZ"]     # X0..Z3
OKABE = ["#0072B2", "#D55E00", "#009E73", "#E69F00", "#56B4E9",
         "#CC79A7", "#000000", "#888888"]
GRID = 0.01          # 100 Hz working grid
PLATEAU_HALF = 0.12  # s, half-width of the plateau window around an extreme

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


def fname_time(path):
    m = re.search(r"(\d{8})_(\d{6})", os.path.basename(path))
    return datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")


def pair_files():
    """Pair k-th magnetic file with k-th force file; verify constant offset."""
    mags = sorted(glob.glob(os.path.join(MAG_DIR, "mlx_v4_quad_*.csv")))
    frcs = sorted(glob.glob(os.path.join(FRC_DIR, "*.csv")))
    n = min(len(mags), len(frcs))
    if len(mags) != len(frcs):
        print(f"[anomaly] unequal file counts: {len(mags)} magnetic vs "
              f"{len(frcs)} force; pairing first {n} of each")
    offsets = []
    for mp, fp in zip(mags[:n], frcs[:n]):
        off = (fname_time(fp) - fname_time(mp)).total_seconds() + 8 * 3600
        offsets.append(off)
    offsets = np.asarray(offsets)
    med = float(np.median(offsets))
    keep, dropped = [], []
    for k in range(n):
        if abs(offsets[k] - med) > 10.0:
            dropped.append((mags[k], frcs[k], offsets[k]))
        else:
            keep.append((mags[k], frcs[k], offsets[k]))
    print(f"pairs: {n} candidate; filename offset (force_UTC - (mag_local-8h)): "
          f"median {med:.1f} s, range [{offsets.min():.1f}, {offsets.max():.1f}] s,"
          f" spread {offsets.max()-offsets.min():.1f} s")
    for mp, fp, off in dropped:
        print(f"[anomaly] excluded pair (offset {off:.1f} s deviates >10 s "
              f"from median): {os.path.basename(mp)} / {os.path.basename(fp)}")
    return keep, med, offsets


def stage_segments(frc):
    """Contiguous stage runs: list of (stage, cycle, t0, t1)."""
    ch = (frc["stage"] != frc["stage"].shift()).cumsum()
    g = frc.groupby(ch).agg(stage=("stage", "first"), cycle=("cycle", "first"),
                            t0=("elapsed_time_s", "first"),
                            t1=("elapsed_time_s", "last"))
    return [(row.stage, int(row.cycle), float(row.t0), float(row.t1))
            for row in g.itertuples(index=False)]


def bandpass(x, fs, lo=0.6, hi=1.5, order=3):
    b, a = sg.butter(order, [lo / (fs / 2), hi / (fs / 2)], "band")
    return sg.filtfilt(b, a, x)


def lowpass(x, fs, fc=5.0, order=3):
    b, a = sg.butter(order, fc / (fs / 2))
    return sg.filtfilt(b, a, x)


def detect_press(g, absdB_lp, seg_down, notes):
    """Press onset = first sustained crossing of low-passed |dB| above 30 %
    of its 90th percentile (skips the smaller pre-recording approach step).
    Returns t_press or None."""
    big = 0.30 * np.percentile(absdB_lp, 90)
    above = absdB_lp > big
    k = int(0.15 / GRID)
    run = np.convolve(above.astype(int), np.ones(k, int), "valid")
    hit = np.flatnonzero(run == k)
    if len(hit) == 0:
        notes.append("press onset not found (no sustained |dB| departure)")
        return None
    return float(g[hit[0]])


def align_and_events(tm, B, dB, frc, notes):
    """Returns alignment + per-cycle extreme times, or None on failure.

    L0        : mag_time = force_time + L0 (press-event anchored)
    ref       : index of magnetic reference channel (strongest 0.6-1.5 Hz)
    ext_left  : magnetic times of left-stage extremes (cycle 1..n)
    ext_right : magnetic times of right-stage extremes
    r_align   : |r| of band-passed Fx vs reference channel at fitted lag
    L_fit     : that fitted lag (QC only; includes Fx mechanical phase)
    """
    fs = 1.0 / GRID
    g = np.arange(tm[0], tm[-1], GRID)
    dB_g = np.column_stack([np.interp(g, tm, dB[:, j]) for j in range(12)])
    absdB_lp = lowpass(np.linalg.norm(dB_g, axis=1), fs, 5.0)

    segs = stage_segments(frc)
    down = [s for s in segs if s[0] == "down"]
    shear = [s for s in segs if s[0] in ("left", "right")]
    if not down or not shear:
        notes.append("force file missing down/left/right stages")
        return None

    t_press = detect_press(g, absdB_lp, down[0], notes)
    if t_press is None:
        return None
    L0 = t_press - down[0][2]

    # reference channel: strongest fundamental-band X/Y channel
    band = np.column_stack([bandpass(dB_g[:, j], fs) for j in range(12)])
    xy_idx = [j for j in range(12) if j % 3 != 2]  # X,Y channels only
    pw = [np.mean(band[:, j] ** 2) for j in range(12)]
    ref = max(xy_idx, key=lambda j: pw[j])
    refb = band[:, ref]

    # oscillation window from envelope of the reference channel
    env = np.abs(sg.hilbert(refb))
    env_med = np.median(env[(g > t_press + 1) & (g < t_press + 15)])
    inw = env > 0.35 * env_med
    w_on = g[np.argmax(inw)]
    w_off = g[len(inw) - 1 - np.argmax(inw[::-1])]

    # extremes of the reference channel (position extremes = stage plateaus)
    dref = np.gradient(refb)
    sgn = np.sign(dref)
    flips = np.flatnonzero(np.diff(sgn) != 0)
    ext_t, ext_pol = [], []
    for i in flips:
        t = g[i]
        if not (w_on + 0.1 <= t <= w_off - 0.05):
            continue
        if np.abs(refb[i]) < 0.5 * env[i] or np.abs(refb[i]) < 0.3 * env_med:
            continue  # inflection noise, not a real reversal
        pol = 1 if refb[i] > 0 else -1
        if ext_t and t - ext_t[-1] < 0.30:      # merge duplicates
            continue
        if ext_pol and pol == ext_pol[-1]:      # enforce alternation
            if np.abs(refb[i]) > np.abs(np.interp(ext_t[-1], g, refb)):
                ext_t[-1], ext_pol[-1] = t, pol
            continue
        ext_t.append(t)
        ext_pol.append(pol)
    ext_t = np.asarray(ext_t)
    ext_pol = np.asarray(ext_pol)
    if len(ext_t) < 6:
        notes.append(f"only {len(ext_t)} extremes found; trial unusable")
        return None

    # left/right identity anchored at cycle 1 (labels accurate at trial start)
    first_left = [s for s in shear if s[0] == "left"][0]
    t_pred = first_left[3] + L0        # end of first left travel
    j = int(np.argmin(np.abs(ext_t - t_pred)))
    if np.abs(ext_t[j] - t_pred) > 0.30:
        notes.append(f"cycle-1 anchor {np.abs(ext_t[j]-t_pred):.2f} s from "
                     "nearest extreme (>0.30 s); trial excluded")
        return None
    left_pol = ext_pol[j]
    ext_left = ext_t[ext_pol == left_pol]
    ext_right = ext_t[ext_pol == -left_pol]
    # first extreme must be a left (protocol: left travel first)
    ext_right = ext_right[ext_right > ext_left[0]]

    n_lab = len([s for s in shear if s[0] == "left"])
    if len(ext_left) != n_lab or len(ext_right) != n_lab:
        notes.append(f"extreme count L{len(ext_left)}/R{len(ext_right)} vs "
                     f"{n_lab} labelled cycles (reset swing or missed "
                     "reversal); truncated to matching prefix")
    n_cyc = min(len(ext_left), len(ext_right), n_lab)
    ext_left, ext_right = ext_left[:n_cyc], ext_right[:n_cyc]

    # QC cross-correlation: band-passed Fx vs reference channel, wide search,
    # alias rejected by staying within +/-0.30 s of the press-event lag L0.
    tf = frc["elapsed_time_s"].values
    gf = np.arange(tf[0], tf[-1], GRID)
    fxb = bandpass(np.interp(gf, tf, frc["Fx_N"].values.astype(float)), fs)
    best = (-2.0, L0)
    for lag in np.arange(L0 - 0.30, L0 + 0.30 + 1e-9, 0.005):
        gi = np.interp(g, gf + lag, fxb, left=np.nan, right=np.nan)
        ok = ~np.isnan(gi)
        if ok.sum() < 1000:
            continue
        r = np.corrcoef(refb[ok], gi[ok])[0, 1]
        if np.isfinite(r) and abs(r) > best[0]:
            best = (abs(r), lag)
    r_align, L_fit = best

    # label drift check: last labelled right end vs last right extreme
    drift_end = float(ext_right[-1] - (shear[-1][3] + L0))
    return {"L0": L0, "L_fit": L_fit, "r_align": r_align, "ref": ref,
            "t_press": t_press, "ext_left": ext_left, "ext_right": ext_right,
            "drift_end": drift_end, "segs": segs, "w": (w_on, w_off)}


def process_pair(mp, fp, trial):
    mag = pd.read_csv(mp)
    frc = pd.read_csv(fp, encoding="utf-8-sig")
    notes = []

    us = mag["us"].values.astype(np.int64)
    if np.any(np.diff(us) <= 0):
        notes.append(f"{int(np.sum(np.diff(us) <= 0))} non-monotonic us "
                     "samples dropped")
        keep = np.concatenate([[True], np.diff(us) > 0])
        mag = mag[keep].reset_index(drop=True)
        us = mag["us"].values.astype(np.int64)
    gaps = np.diff(us) / 1e6
    if (gaps > 0.05).any():
        notes.append(f"{int((gaps > 0.05).sum())} magnetic gaps >50 ms "
                     f"(max {gaps.max()*1e3:.0f} ms)")
    tm = (us - us[0]) / 1e6
    B = mag[RAW_COLS].values.astype(float)
    if np.isnan(B).any():
        notes.append(f"{int(np.isnan(B).any(axis=1).sum())} NaN magnetic "
                     "rows dropped")
        ok = ~np.isnan(B).any(axis=1)
        tm, B = tm[ok], B[ok]

    # magnetic baseline: pre-approach (first sustained departure above 6 MAD)
    prov = B - np.median(B[tm < 1.5], axis=0)
    n0 = np.linalg.norm(prov, axis=1)
    med = np.median(n0[tm < 1.5])
    mad = 1.4826 * np.median(np.abs(n0[tm < 1.5] - med)) + 1e-9
    above = n0 > med + 6 * mad
    k = int(0.10 * len(tm) / tm[-1])  # 0.10 s of samples
    run = np.convolve(above.astype(int), np.ones(k, int), "valid")
    hit = np.flatnonzero(run == k)
    t_dep = tm[hit[0]] if len(hit) else tm[-1]
    b1 = min(t_dep - 0.25, 3.0)
    if b1 < 0.6:
        notes.append(f"magnetic record starts mid-trial (activity from "
                     f"t={t_dep:.2f} s, duration {tm[-1]:.1f} s): no pre-press "
                     "baseline and no press event -> left/right identity "
                     "unverifiable; trial excluded")
        return None, None, None, notes
    dB = B - np.median(B[tm <= b1], axis=0)

    al = align_and_events(tm, B, dB, frc, notes)
    if al is None:
        return None, None, None, notes
    if al["L0"] < 1.0:
        notes.append(f"press-event lag L0={al['L0']:.2f} s < 1 s (magnetic "
                     "record does not cleanly precede the force record); "
                     "trial excluded")
        return None, None, None, notes

    L0, segs = al["L0"], al["segs"]
    tf = frc["elapsed_time_s"].values
    F = frc[["Fx_N", "Fy_N", "Fz_N"]].values.astype(float)
    idle = frc["stage"].values == "idle"
    F = F - np.median(F[idle] if idle.any() else F[:100], axis=0)
    Fnorm = np.linalg.norm(F, axis=1)

    def mean_win(t_mag_c):
        wm = (tm >= t_mag_c - PLATEAU_HALF) & (tm <= t_mag_c + PLATEAU_HALF)
        t_f = t_mag_c - L0
        wf = (tf >= t_f - PLATEAU_HALF) & (tf <= t_f + PLATEAU_HALF)
        if wm.sum() < 10 or wf.sum() < 5:
            return None, None
        return dB[wm].mean(0), F[wf].mean(0)

    rows = []
    # down (cycle 0): pressed hold right after the press ramp
    mB, mF = mean_win(al["t_press"] + 0.30)
    cyc0 = frc["cycle"].values == 0
    if mB is not None:
        row = {"trial": trial, "cycle": 0, "stage": "down"}
        row.update({n: float(v) for n, v in zip(CH_NAMES, mB)})
        row["Fx"], row["Fy"], row["Fz"] = [float(v) for v in mF]
        row["peakF"] = float(Fnorm[cyc0].max()) if cyc0.any() else np.nan
        rows.append(row)
    else:
        notes.append("down plateau window unusable; down row skipped")

    T_half = np.median(np.diff(np.sort(np.concatenate(
        [al["ext_left"], al["ext_right"]]))))
    for k, (tL, tR) in enumerate(zip(al["ext_left"], al["ext_right"]), 1):
        wc = ((tf >= tL - L0 - T_half) & (tf <= tR - L0 + T_half))
        pk = float(Fnorm[wc].max()) if wc.any() else np.nan
        for st, tc in (("left", tL), ("right", tR)):
            mB, mF = mean_win(tc)
            if mB is None:
                notes.append(f"cycle {k} {st}: window outside records; skipped")
                continue
            row = {"trial": trial, "cycle": k, "stage": st}
            row.update({n: float(v) for n, v in zip(CH_NAMES, mB)})
            row["Fx"], row["Fy"], row["Fz"] = [float(v) for v in mF]
            row["peakF"] = pk
            rows.append(row)

    trace = {"tm": tm, "dB": dB, "al": al, "frc": frc}
    return rows, trace, al, notes


# ----------------------------------------------------------------------------
def make_figure(trace, feats, rep_trial):
    ax_colors = {"x": OKABE[0], "y": OKABE[1], "z": OKABE[2]}
    fill = {"down": "#888888", "left": OKABE[4], "right": OKABE[3]}

    tm, dB, al, frc = (trace["tm"], trace["dB"], trace["al"], trace["frc"])
    L0 = al["L0"]
    tf = frc["elapsed_time_s"].values
    F = frc[["Fx_N", "Fy_N", "Fz_N"]].values.astype(float)
    idle = frc["stage"].values == "idle"
    F = F - np.median(F[idle], axis=0)

    # machine-executed stage bands: travel toward each extreme
    T_half = np.median(np.diff(np.sort(np.concatenate(
        [al["ext_left"], al["ext_right"]]))))
    bands = [("down", al["t_press"] - 0.20, al["t_press"] + 0.20)]
    for st, arr in (("left", al["ext_left"]), ("right", al["ext_right"])):
        for te in arr:
            bands.append((st, te - T_half, te))

    fig = plt.figure(figsize=(7.09, 4.1))
    gs = fig.add_gridspec(5, 2, width_ratios=[1.25, 1.0],
                          height_ratios=[1, 1, 1, 1, 0.65],
                          hspace=0.38, wspace=0.30)
    x0 = L0  # plot in force-record time: t_plot = t_mag - L0
    tlim = (-1.0, tf[-1] + 1.5)
    axs = []
    for s in range(4):
        ax = fig.add_subplot(gs[s, 0], sharex=axs[0] if axs else None)
        axs.append(ax)
        for st, t0, t1 in bands:
            ax.axvspan(t0 - x0, t1 - x0, color=fill[st], alpha=0.13, lw=0)
        for j, a in enumerate("xyz"):
            ax.plot(tm - x0, dB[:, 3 * s + j], color=ax_colors[a], lw=0.6)
        ax.set_xlim(*tlim)
        ax.text(0.006, 0.84, f"S{s}", transform=ax.transAxes, fontsize=7,
                fontweight="bold")
        plt.setp(ax.get_xticklabels(), visible=False)
        ax.set_ylabel("ΔB (µT)")
    axf = fig.add_subplot(gs[4, 0], sharex=axs[0])
    for st, t0, t1 in bands:
        axf.axvspan(t0 - x0, t1 - x0, color=fill[st], alpha=0.13, lw=0)
    axf.plot(tf, F[:, 0], color=OKABE[5], lw=0.6)
    axf.plot(tf, F[:, 2], color=OKABE[7], lw=0.6)
    axf.set_ylabel("F (N)")
    axf.set_xlabel("Time from force-record start (s)")
    hs = [Line2D([], [], color=ax_colors[a], lw=1.1,
                 label=f"$\\Delta B_{a}$") for a in "xyz"]
    hs += [Line2D([], [], color=OKABE[5], lw=1.1, label="$F_x$"),
           Line2D([], [], color=OKABE[7], lw=1.1, label="$F_z$"),
           Patch(fc=fill["down"], alpha=0.3, label="press"),
           Patch(fc=fill["left"], alpha=0.3, label="left (x−)"),
           Patch(fc=fill["right"], alpha=0.3, label="right (x+)")]
    axs[0].legend(handles=hs, ncol=5, loc="lower left",
                  bbox_to_anchor=(-0.02, 1.02), handlelength=1.1,
                  columnspacing=0.8, handletextpad=0.45)
    fig.text(0.015, 0.995, f"a   Representative x-shear trial {rep_trial}",
             va="top", ha="left", fontsize=8)

    # panel b: cycle-averaged stage response per channel
    axb = fig.add_subplot(gs[:, 1])
    scol = {"down": "#000000", "left": OKABE[4], "right": OKABE[3]}
    offs = {"down": -0.24, "left": 0.0, "right": 0.24}
    xpos = np.arange(12)
    for st in ("down", "left", "right"):
        sub = feats[feats["stage"] == st]
        mu = sub[CH_NAMES].mean(0).values
        sd = sub[CH_NAMES].std(0).values
        axb.errorbar(xpos + offs[st], mu, yerr=sd, fmt="o", ms=3.4,
                     mfc=scol[st], mec="white", mew=0.5, color=scol[st],
                     elinewidth=0.8, capsize=1.6, capthick=0.8,
                     label={"down": "press"}.get(st, st))
    axb.axhline(0, color="#888888", lw=0.6, zorder=0)
    for s in range(1, 4):
        axb.axvline(3 * s - 0.5, color="#cccccc", lw=0.5, zorder=0)
    axb.set_xticks(xpos)
    axb.set_xticklabels([f"$B_{a}$" for s in range(4) for a in "xyz"])
    for s in range(4):
        axb.text(3 * s + 1, 1.005, f"S{s}",
                 transform=axb.get_xaxis_transform(), ha="center",
                 fontsize=7, fontweight="bold")
    axb.set_ylabel("Stage-plateau ΔB (µT), mean ± s.d.")
    axb.set_xlabel("Channel")
    axb.legend(loc="upper right", handletextpad=0.4)
    axb.set_title("b   Cycle-averaged response (all trials)", loc="left",
                  fontsize=8, pad=10)

    fig.tight_layout(pad=0.6, rect=(0, 0, 1, 0.94))
    fig.savefig(FIG_BASE + ".png")
    fig.savefig(FIG_BASE + ".pdf")
    plt.close(fig)


def main():
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    os.makedirs(os.path.dirname(FIG_BASE), exist_ok=True)
    pairs, med_off, offsets = pair_files()

    all_rows, aligns, traces, all_anom = [], {}, {}, []
    for mp, fp, off in pairs:
        trial = re.search(r"(\d{8}_\d{6})", os.path.basename(mp)).group(1)
        rows, trace, al, notes = process_pair(mp, fp, trial)
        for nte in notes:
            all_anom.append(f"{trial}: {nte}")
        if rows is None:
            all_anom.append(f"{trial}: trial excluded (see notes above)")
            continue
        aligns[trial] = al
        traces[trial] = trace
        all_rows += rows
        print(f"  {trial}: L0={al['L0']:.2f} s  L_fit-L0={al['L_fit']-al['L0']:+.3f} s"
              f"  r={al['r_align']:.3f}  ref={RAW_COLS[al['ref']]}"
              f"  cycles={len(al['ext_left'])}  label-drift@end="
              f"{al['drift_end']:+.2f} s")

    feats = pd.DataFrame(all_rows)
    cols = ["trial", "cycle", "stage"] + CH_NAMES + ["Fx", "Fy", "Fz", "peakF"]
    feats = feats[cols]
    feats.to_csv(OUT_CSV, index=False)

    rv = np.array([a["r_align"] for a in aligns.values()])
    l0 = np.array([a["L0"] for a in aligns.values()])
    dl = np.array([a["L_fit"] - a["L0"] for a in aligns.values()])
    dr = np.array([a["drift_end"] for a in aligns.values()])
    print(f"\ntrials processed: {len(aligns)} / {len(pairs)} paired")
    print(f"alignment r (|corr|, band-passed Fx vs magnetic ref ch): "
          f"median {np.median(rv):.3f}, min {rv.min():.3f}, max {rv.max():.3f}")
    print(f"press-event lag L0: {l0.min():.2f}..{l0.max():.2f} s "
          f"(median {np.median(l0):.2f}); L_fit-L0 median {np.median(dl):+.3f} s")
    print(f"stage-label drift at trial end (magnetic extremes vs labels+L0): "
          f"median {np.median(dr):+.2f} s -> labels alone are NOT plateau-"
          "accurate late in a trial (handled by magnetic-anchored timing)")
    print(f"feature rows: {len(feats)} -> {OUT_CSV}")

    # key amplitudes
    for st in ("down", "left", "right"):
        sub = feats[feats["stage"] == st]
        mu = sub[CH_NAMES].mean(0)
        c = mu.abs().idxmax()
        print(f"stage {st:5s}: largest |dB| channel {c} = {mu[c]:+.1f} uT "
              f"(mean of {len(sub)} rows); mean F=({sub['Fx'].mean():+.2f},"
              f"{sub['Fy'].mean():+.2f},{sub['Fz'].mean():+.2f}) N, "
              f"peak|F| {sub['peakF'].mean():.2f} N")

    # fingerprint quantification: left - right differential per channel
    piv = feats[feats["stage"].isin(["left", "right"])].pivot_table(
        index=["trial", "cycle"], columns="stage", values=CH_NAMES)
    print("\nshear differential (left - right), mean +- s.d. across "
          "trials x cycles:")
    diffs = {}
    for chn in CH_NAMES:
        d = piv[(chn, "left")] - piv[(chn, "right")]
        diffs[chn] = (d.mean(), d.std())
        print(f"  {chn}: {d.mean():+7.1f} +- {d.std():5.1f} uT")
    xmax = max(abs(v[0]) for k2, v in diffs.items() if k2.startswith("dBx"))
    print("expected x-shear fingerprints, quantified:")
    for s in range(4):
        dx, dy = diffs[f"dBx{s}"][0], diffs[f"dBy{s}"][0]
        rel = "together" if np.sign(dx) == np.sign(dy) else "opposite"
        print(f"  sensor{s}: dX={dx:+.1f}, dY={dy:+.1f} uT -> X,Y move {rel}; "
              f"|dX|/max|dX| = {abs(dx)/xmax:.2f}")

    if all_anom:
        print("\nanomalies / disclosed notes:")
        for a in all_anom:
            print("  " + a)
    else:
        print("\nanomalies: none")

    rep = max(aligns, key=lambda t: aligns[t]["r_align"])
    make_figure(traces[rep], feats, rep)
    print(f"\nfigure ({rep} as representative) -> {FIG_BASE}.png/.pdf")


if __name__ == "__main__":
    main()

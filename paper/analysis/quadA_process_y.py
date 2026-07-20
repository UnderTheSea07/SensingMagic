#!/usr/bin/env python
"""Process quad-MLX90393 y-direction shear trials (press 1 mm, shear +/-2 mm in y).

Pipeline
--------
1. Pair magnetic (mlx_v4_quad_*.csv, filename = LOCAL time UTC+8) with force
   (2026*.csv, filename = UTC) files in filename order; verify the constant
   filename-time offset (force_UTC - magnetic_UTC); exclude pairs deviating
   > 10 s from the median offset.
2. Per pair: baseline-subtract the 12 magnetic channels (dB), align force ->
   magnetic time by shear-band onset + cross-correlation at 100 Hz (see
   notes below; the labeled 'down' onset is NOT usable as an anchor in this
   dataset); record alignment r and r_pos.
3. Per cycle/stage (down, y_minus->'left', y_plus->'right'): mean dB (12 ch)
   over the stage plateau + mean baseline-subtracted Fx/Fy/Fz + per-cycle
   peak |F|.
4. Tidy table -> paper/analysis/out/quad_features_y.csv
5. Diagnostic figure -> paper/figures/fig3_quad_y_overview.{pdf,png}

Notes / disclosed choices (all verified on the data, see repo notes)
--------------------------------------------------------------------
- Force stage labels in these files are y_minus / y_plus; we map
  y_minus -> 'left', y_plus -> 'right' (nominal; machine y-axis sign).
- The actual press occurs BEFORE the force recording starts: the magnetic
  record shows a ~2.5-3 s pressed plateau before the first shear cycle,
  whereas the force stage stream has only 0.4 s between 'down' and the first
  y_minus.  Under signal-based alignment (below) the labeled 'down' window
  falls on the tail of the pressed plateau, so its mean dB is the pressed-
  state response (which is what we want); the press RAMP itself is not
  covered by force stage labels.
- Alignment is therefore anchored on the SHEAR band, which both streams
  share: coarse = magnetic oscillation-envelope onset vs first y_minus
  onset; refined over coarse +/- 0.8 s by maximizing |corr| between the
  commanded-position triangle wave (y_minus/y_plus ramps + the aperiodic
  reset_y tail, which prevents locking onto the wrong 1.02 s cycle) and
  the best of the 12 dB channels on a 100 Hz grid.  r_pos is that
  correlation (primary quality gate, >= 0.6 required); r is |corr| of
  smoothed Fy vs the best dB channel at the chosen lag (honest force-vs-
  magnetic agreement, limited by force-sensor quantization, ~0.05-0.16 N
  steps on a +/-0.3 N signal).
- Within each 0.5 s shear stage the stage commands TRAVEL to that extreme:
  the extreme is reached ~0.35 s in and dwells over the stage end.  Stage
  'plateau' therefore = final 25 % of the stage (not the central 50 %,
  which is mid-travel and averages to ~zero).  Same window for 'down'
  (it sits on the pressed plateau throughout).
- Magnetic baseline: pre-press window of the magnetic record (first
  sustained departure detected; capped at 2 s).  The force 'idle' stage
  maps onto the pressed plateau, so it is NOT usable as magnetic zero.
- Magnetic units are the counts recorded by the logger (nominally uT).
- Forces are baseline-subtracted (idle-stage median removed; note the idle
  reference is the pressed state, so Fz is force change from that state).
"""
import glob
import os
import re
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# ----------------------------------------------------------------------------
ROOT = ("/private/tmp/claude-502/-Users-arielzhang-Desktop-SensingMagic/"
        "617d7206-542d-4a74-bd36-6e9400bc4ca0/scratchpad/4sensors/4_sensors/"
        "y方向-左右各2mm")
MAG_DIR = os.path.join(ROOT, "磁数据")
FRC_DIR = os.path.join(ROOT, "力数据")
OUT_CSV = "/Users/arielzhang/Desktop/SensingMagic/paper/analysis/out/quad_features_y.csv"
FIG_BASE = "/Users/arielzhang/Desktop/SensingMagic/paper/figures/fig3_quad_y_overview"

CH_NAMES = [f"dB{a}{s}" for s in range(4) for a in "xyz"]  # dBx0..dBz3
RAW_COLS = [f"{A}{s}" for s in range(4) for A in "XYZ"]     # X0..Z3
STAGE_MAP = {"down": "down", "y_minus": "left", "y_plus": "right"}
OKABE = ["#0072B2", "#D55E00", "#009E73", "#E69F00", "#56B4E9",
         "#CC79A7", "#000000", "#888888"]

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


def fname_time(path, fmt):
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
        t_mag_utc = fname_time(mp, None)  # local UTC+8 ...
        t_frc_utc = fname_time(fp, None)  # ... vs UTC
        off = (t_frc_utc - t_mag_utc).total_seconds() + 8 * 3600
        offsets.append(off)
    offsets = np.asarray(offsets)
    med = float(np.median(offsets))
    keep, dropped = [], []
    for k in range(n):
        if abs(offsets[k] - med) > 10.0:
            dropped.append((mags[k], frcs[k], offsets[k]))
        else:
            keep.append((mags[k], frcs[k], offsets[k]))
    print(f"pairs: {n} candidate; filename offset (force_UTC - mag_UTC+8h): "
          f"median {med:.1f} s, range [{offsets.min():.1f}, {offsets.max():.1f}] s")
    for mp, fp, off in dropped:
        print(f"[anomaly] excluded pair (offset {off:.1f} s deviates >10 s): "
              f"{os.path.basename(mp)} / {os.path.basename(fp)}")
    return keep, med, offsets


def stage_segments(frc):
    """Contiguous stage runs: list of (stage, cycle, t0, t1)."""
    ch = (frc["stage"] != frc["stage"].shift()).cumsum()
    g = frc.groupby(ch).agg(stage=("stage", "first"), cycle=("cycle", "first"),
                            t0=("elapsed_time_s", "first"),
                            t1=("elapsed_time_s", "last"))
    return [(row.stage, int(row.cycle), float(row.t0), float(row.t1))
            for row in g.itertuples(index=False)]


def rollmed(x, k):
    return pd.Series(x).rolling(k, center=True, min_periods=1).median().values


def rollstd(x, k):
    return pd.Series(x).rolling(k, center=True, min_periods=1).std().values


def smooth(x, k):
    return np.convolve(x, np.ones(k) / k, "same")


def align_pair(tm, B, frc):
    """Signal-anchored alignment.  Returns dict with shift (mag_time =
    force_time + shift), r (Fy vs best dB channel), r_pos, baseline info,
    notes.  Anchors on the shared 20-cycle shear band: coarse via the
    magnetic oscillation-envelope onset vs first y_minus onset, refined by
    cross-correlation over +/- 0.5 s (< half the 1.02 s cycle)."""
    notes = []
    fs = 1.0 / np.median(np.diff(tm))
    dBn = np.linalg.norm(B - np.median(B[tm < tm[0] + 1.0], axis=0), axis=1)

    # oscillation window from rolling std of high-passed norm
    hp = dBn - rollmed(dBn, int(2.0 * fs))
    env = rollstd(hp, int(0.3 * fs))
    thr = 0.25 * np.percentile(env, 98)
    above = (env > thr).astype(int)
    # oscillation REGION: >50 % of a 3 s window above threshold (robust to
    # per-cycle dwells where the envelope dips); then the precise onset =
    # first 0.3 s sustained burst from 1.5 s before the region start (the
    # press step is always >2.5 s earlier when present, so it cannot be
    # picked up here).
    k3 = int(3.0 * fs)
    frac = np.convolve(above, np.ones(k3) / k3, "valid")
    hit = np.flatnonzero(frac > 0.5)
    if len(hit) == 0:
        return None
    k4 = int(0.3 * fs)
    run = np.convolve(above, np.ones(k4, int), "valid")
    start = max(hit[0] - int(1.5 * fs), 0)
    on = np.flatnonzero(run[start:] >= 0.9 * k4)
    if len(on) == 0:
        return None
    i0 = start + on[0]

    segs = stage_segments(frc)
    sh = [s for s in segs if s[0] in ("y_minus", "y_plus")]
    if not sh:
        return None
    sh0, sh1 = sh[0][2], sh[-1][3]
    coarse = tm[i0] - sh0
    # aperiodic anchor: reset_y returns the stage to y=0 right after the
    # last y_plus; including it breaks the 1.02 s periodicity of the shear
    # band so the cross-correlation cannot lock onto the wrong cycle.
    ry = next((s for s in segs if s[0] == "reset_y" and s[2] >= sh1 - 0.1),
              None)

    # contact/press onset -> magnetic baseline window (pre-contact)
    base = dBn[tm < tm[0] + 1.0]
    med = np.median(base)
    mad = 1.4826 * np.median(np.abs(base - med)) + 1e-9
    thr2 = med + max(8 * mad, 0.08 * (np.percentile(dBn, 99) - med))
    k2 = int(0.05 * fs)
    run2 = np.convolve((dBn > thr2).astype(int), np.ones(k2, int), "valid")
    on = np.flatnonzero(run2 == k2)
    t_press = tm[on[0]] if len(on) else np.nan
    if np.isfinite(t_press) and t_press > tm[0] + 0.4:
        b0, b1 = tm[0], min(tm[0] + 2.0, t_press - 0.2)
    else:
        b0, b1 = tm[0], tm[0] + 1.0
        notes.append("no clean pre-contact window; baseline = first 1.0 s "
                     "(may be pressed state)")

    tf = frc["elapsed_time_s"].values
    Fy = frc["Fy_N"].values.astype(float)
    idle = frc["stage"].values == "idle"
    Fy = Fy - np.median(Fy[idle] if idle.any() else Fy[:100])
    g_end = ry[3] if ry is not None else sh1
    g = np.arange(sh0, g_end, 0.01)
    kk = 15  # 0.15 s
    fyg = smooth(np.interp(g, tf, Fy), kk)
    # commanded-position reference (triangle wave): each y_minus/y_plus
    # stage ramps to that extreme; dB tracks position, which is ~90 deg
    # behind the stage square wave, hence this reference.  reset_y ramps
    # back to 0 (aperiodic tail).
    pos = np.zeros_like(g)
    cur = 0.0
    for st, _, t0, t1 in sh + ([ry] if ry is not None else []):
        tgt = 0.0 if st == "reset_y" else (1.0 if st == "y_plus" else -1.0)
        w = (g >= t0) & (g <= t1)
        if w.any():
            pos[w] = cur + (tgt - cur) * (g[w] - t0) / max(t1 - t0, 1e-6)
        pos[g > t1] = tgt
        cur = tgt

    # refine: max over channels of |corr(position wave, dB_j)| per lag
    best = (-2.0, coarse, 0)
    for lag in np.arange(coarse - 0.8, coarse + 0.8 + 1e-9, 0.005):
        tq = g + lag
        ok = (tq >= tm[0]) & (tq <= tm[-1])
        if ok.sum() < len(g) * 0.8:
            continue
        rbest, jbest = -2.0, 0
        for j in range(12):
            mi = np.interp(tq[ok], tm, B[:, j])
            r = abs(np.corrcoef(pos[ok], mi)[0, 1])
            if np.isfinite(r) and r > rbest:
                rbest, jbest = r, j
        if rbest > best[0]:
            best = (rbest, lag, jbest)
    r_pos, shift, jb = best
    if abs(shift - coarse) > 0.78:
        notes.append(f"refined shift at edge of search window "
                     f"(coarse {coarse:.2f}, refined {shift:.2f})")
    mi = smooth(np.interp(g + shift, tm, B[:, jb]), kk)
    r_F = abs(np.corrcoef(fyg, mi)[0, 1])
    return {"shift": shift, "r": r_F, "r_pos": r_pos, "jb": jb,
            "bwin": (b0, b1), "t_press": t_press, "segs": segs,
            "notes": notes}


def process_pair(mp, fp, trial):
    mag = pd.read_csv(mp)
    frc = pd.read_csv(fp, encoding="utf-8-sig")
    anomalies = []

    us = mag["us"].values.astype(np.int64)
    if np.any(np.diff(us) <= 0):
        n_bad = int(np.sum(np.diff(us) <= 0))
        anomalies.append(f"{n_bad} non-monotonic us samples (dropped)")
        keepm = np.concatenate([[True], np.diff(us) > 0])
        mag = mag[keepm].reset_index(drop=True)
        us = mag["us"].values.astype(np.int64)
    gaps = np.diff(us) / 1e6
    if (gaps > 0.05).any():
        anomalies.append(f"{int((gaps > 0.05).sum())} magnetic gaps >50 ms "
                         f"(max {gaps.max()*1e3:.0f} ms)")
    tm = (us - us[0]) / 1e6
    B = mag[RAW_COLS].values.astype(float)
    # leading sensor bring-up dropout: rows where >=6 channels read exactly 0
    dropout = (B == 0).sum(axis=1) >= 6
    if dropout.any():
        idx = np.flatnonzero(dropout)
        if idx[-1] < 0.2 * len(tm):  # confined to the start -> trim
            anomalies.append(f"{len(idx)} rows with >=6 all-zero channels "
                             f"(sensor bring-up dropout, first "
                             f"{tm[idx[-1]]:.2f} s); trimmed")
            keep = np.arange(len(tm)) > idx[-1]
            tm, B = tm[keep], B[keep]
        else:
            anomalies.append(f"{len(idx)} all-zero dropout rows beyond the "
                             "record start; trial excluded")
            return None, None, {"r": np.nan, "r_pos": np.nan}, anomalies
    if np.isnan(B).any():
        nbad = int(np.isnan(B).any(axis=1).sum())
        anomalies.append(f"{nbad} NaN magnetic rows dropped")
        ok = ~np.isnan(B).any(axis=1)
        tm, B = tm[ok], B[ok]

    al = align_pair(tm, B, frc)
    if al is None:
        return None, None, {"r": np.nan, "r_pos": np.nan}, \
            anomalies + ["alignment failed"]
    anomalies += [f"{n}" for n in al["notes"]]
    if any("no clean pre-contact" in n for n in al["notes"]):
        # dB offsets would be referenced to the pressed state -> not
        # comparable with the other trials; exclude rather than mix.
        return None, None, {"r": np.nan, "r_pos": np.nan}, \
            anomalies + ["no pre-contact baseline; trial excluded"]
    shift, segs = al["shift"], al["segs"]

    # baseline-subtracted dB (pre-press magnetic window)
    b0, b1 = al["bwin"]
    dB = B - np.median(B[(tm >= b0) & (tm <= b1)], axis=0)

    # baseline-subtracted forces
    F = frc[["Fx_N", "Fy_N", "Fz_N"]].values.astype(float)
    idle = frc["stage"].values == "idle"
    F = F - np.median(F[idle] if idle.any() else F[:100], axis=0)
    tf = frc["elapsed_time_s"].values
    Fnorm = np.linalg.norm(F, axis=1)

    # per-cycle/stage features: final 25 % of each stage (extreme dwell)
    rows = []
    for stage, cyc, t0, t1 in segs:
        if stage not in STAGE_MAP:
            continue
        dur = t1 - t0
        if dur < 0.05:
            anomalies.append(f"cycle {cyc} stage {stage}: {dur*1e3:.0f} ms "
                             "segment skipped")
            continue
        p0, p1 = t0 + 0.75 * dur, t1
        wm = (tm >= p0 + shift) & (tm <= p1 + shift)
        wf = (tf >= p0) & (tf <= p1)
        if wm.sum() < 10 or wf.sum() < 5:
            anomalies.append(f"cycle {cyc} stage {stage}: window outside "
                             "magnetic record; skipped")
            continue
        wc = frc["cycle"].values == cyc
        row = {"trial": trial, "cycle": cyc, "stage": STAGE_MAP[stage]}
        row.update({n: float(v) for n, v in zip(CH_NAMES, dB[wm].mean(0))})
        row["Fx"], row["Fy"], row["Fz"] = [float(v) for v in F[wf].mean(0)]
        row["peakF"] = float(Fnorm[wc].max())
        rows.append(row)

    trace = {"tm": tm, "dB": dB, "shift": shift, "segs": segs, "frc": frc}
    return rows, trace, al, anomalies


# ----------------------------------------------------------------------------
def make_figure(trace, feats, rep_trial):
    sensors = range(4)
    ax_colors = {"x": OKABE[0], "y": OKABE[1], "z": OKABE[2]}
    stage_fill = {"down": "#888888", "y_minus": OKABE[4], "y_plus": OKABE[3]}

    fig = plt.figure(figsize=(7.09, 3.4))
    outer = fig.add_gridspec(1, 2, width_ratios=[1.25, 1.0])
    inner = outer[0].subgridspec(4, 1, hspace=0.35)

    tm, dB, shift, segs = trace["tm"], trace["dB"], trace["shift"], trace["segs"]
    tf_end = trace["frc"]["elapsed_time_s"].iloc[-1]
    axs = []
    for s in sensors:
        ax = fig.add_subplot(inner[s], sharex=axs[0] if axs else None)
        axs.append(ax)
        for stage, cyc, t0, t1 in segs:
            if stage in stage_fill:
                ax.axvspan(t0, t1, color=stage_fill[stage], alpha=0.14, lw=0)
        for j, a in enumerate("xyz"):
            ax.plot(tm - shift, dB[:, 3 * s + j], color=ax_colors[a],
                    lw=0.7, label=f"$B_{a}$" if s == 0 else None)
        ax.set_xlim(-1.0, tf_end + 1.0)
        ax.text(0.006, 0.86, f"S{s}", transform=ax.transAxes, fontsize=7,
                fontweight="bold")
        if s < 3:
            plt.setp(ax.get_xticklabels(), visible=False)
        ax.set_ylabel("ΔB (µT)")
    axs[-1].set_xlabel("Time from force-record start (s)")
    hs = [Line2D([], [], color=ax_colors[a], lw=1.1,
                 label=f"$\\Delta B_{a}$") for a in "xyz"]
    hs += [Patch(fc=stage_fill["down"], alpha=0.3, label="down"),
           Patch(fc=stage_fill["y_minus"], alpha=0.3, label="left (y−)"),
           Patch(fc=stage_fill["y_plus"], alpha=0.3, label="right (y+)")]
    axs[0].legend(handles=hs, ncol=6, loc="lower left",
                  bbox_to_anchor=(0.0, 1.02), handlelength=1.2,
                  columnspacing=0.9, handletextpad=0.5)
    axs[0].set_title(f"a   Representative trial {rep_trial}", loc="left",
                     fontsize=8, pad=18)

    # panel b: cycle-averaged stage response per channel
    axb = fig.add_subplot(outer[1])
    stages = ["down", "left", "right"]
    scol = {"down": "#000000", "left": OKABE[4], "right": OKABE[3]}
    offs = {"down": -0.22, "left": 0.0, "right": 0.22}
    xpos = np.arange(12)
    for st in stages:
        sub = feats[feats["stage"] == st]
        mu = sub[CH_NAMES].mean(0).values
        sd = sub[CH_NAMES].std(0).values
        axb.errorbar(xpos + offs[st], mu, yerr=sd, fmt="o", ms=3.4,
                     mfc=scol[st], mec="white", mew=0.5, color=scol[st],
                     elinewidth=0.8, capsize=1.6, capthick=0.8, label=st)
    axb.axhline(0, color="#888888", lw=0.6, zorder=0)
    for s in range(1, 4):
        axb.axvline(3 * s - 0.5, color="#cccccc", lw=0.5, zorder=0)
    axb.set_xticks(xpos)
    axb.set_xticklabels([f"$B_{a}$" for s in range(4) for a in "xyz"])
    for s in range(4):
        axb.text(3 * s + 1, 1.005, f"S{s}", transform=axb.get_xaxis_transform(),
                 ha="center", fontsize=7, fontweight="bold")
    axb.set_ylabel("Stage-plateau ΔB (µT), mean ± s.d.")
    axb.set_xlabel("Channel")
    axb.legend(loc="upper left", handletextpad=0.4)
    nt = feats["trial"].nunique()
    axb.set_title(f"b   Cycle-averaged stage response ({nt} trials)", loc="left",
                  fontsize=8, pad=10)

    fig.tight_layout(pad=0.6)
    fig.savefig(FIG_BASE + ".png")
    fig.savefig(FIG_BASE + ".pdf")
    plt.close(fig)


def main():
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    os.makedirs(os.path.dirname(FIG_BASE), exist_ok=True)
    pairs, med_off, offsets = pair_files()

    all_rows, aligns, all_anom, traces = [], {}, [], {}
    for mp, fp, off in pairs:
        trial = re.search(r"(\d{8}_\d{6})", os.path.basename(mp)).group(1)
        rows, trace, al, anom = process_pair(mp, fp, trial)
        for a in anom:
            all_anom.append(f"{trial}: {a}")
        if rows is None or not np.isfinite(al["r"]) or al["r_pos"] < 0.6:
            rs = al.get("r_pos", np.nan)
            all_anom.append(f"{trial}: alignment failed or r_pos={rs:.3f} "
                            "< 0.6; trial excluded")
            continue
        aligns[trial] = al
        traces[trial] = trace
        all_rows += rows

    feats = pd.DataFrame(all_rows)
    cols = ["trial", "cycle", "stage"] + CH_NAMES + ["Fx", "Fy", "Fz", "peakF"]
    feats = feats[cols]

    # data-driven outlier screen on the left-stage per-trial mean vector
    # (12 channels) vs the cross-trial median:
    #  - a single sensor deviating > 300 uT while >= 2 sensors sit < 30 uT
    #    = instrument fault -> trial removed from the CSV and aggregates;
    #  - otherwise max channel deviation > 100 uT = coherent mechanical
    #    offset (e.g. first-trial settling) -> kept in the CSV, excluded
    #    from the aggregate statistics and panel b.
    lm = feats[feats["stage"] == "left"].groupby("trial")[CH_NAMES].mean()
    med = lm.median(axis=0)
    dev = (lm - med).abs()
    sensor_dev = pd.DataFrame(
        {s: dev[[f"dB{a}{s}" for a in "xyz"]].max(axis=1) for s in range(4)})
    fault, mech = [], []
    for t in lm.index:
        sd_t = sensor_dev.loc[t]
        if (sd_t > 300).any() and (sd_t < 30).sum() >= 2:
            fault.append(t)
            all_anom.append(
                f"{t}: instrument fault - sensor deviation vs cross-trial "
                f"median (uT): " +
                ", ".join(f"S{s}={sd_t[s]:.0f}" for s in range(4)) +
                "; trial removed from CSV and aggregates")
        elif dev.loc[t].max() > 100:
            mech.append(t)
            all_anom.append(
                f"{t}: coherent offset vs cross-trial median (max channel "
                f"deviation {dev.loc[t].max():.0f} uT, spread across "
                "sensors; first-trial settling); kept in CSV, excluded "
                "from aggregates")
    feats = feats[~feats["trial"].isin(fault)]
    agg = feats[~feats["trial"].isin(mech)]
    feats.to_csv(OUT_CSV, index=False)

    rvals = np.array([a["r"] for a in aligns.values()])
    rst = np.array([a["r_pos"] for a in aligns.values()])
    shifts = np.array([a["shift"] for a in aligns.values()])
    print(f"\ntrials processed: {len(aligns)} / {len(pairs)} paired")
    print(f"alignment r (Fy vs best dB ch): median {np.median(rvals):.3f}, "
          f"min {rvals.min():.3f}, max {rvals.max():.3f}")
    print(f"alignment r_pos (position wave vs dB): median {np.median(rst):.3f},"
          f" min {rst.min():.3f}, max {rst.max():.3f}")
    print(f"shifts (mag - force clock): {np.min(shifts):.2f} .. "
          f"{np.max(shifts):.2f} s (median {np.median(shifts):.2f})")
    print(f"feature rows: {len(feats)} (CSV, {feats['trial'].nunique()} "
          f"trials); aggregates over {agg['trial'].nunique()} trials")

    # key amplitudes
    for st in ["down", "left", "right"]:
        sub = agg[agg["stage"] == st]
        mu = sub[CH_NAMES].mean(0)
        c = mu.abs().idxmax()
        print(f"stage {st:5s}: largest |dB| channel {c} = {mu[c]:+.1f} uT "
              f"(mean across {len(sub)} cycle-stages); "
              f"mean F=({sub['Fx'].mean():+.2f},{sub['Fy'].mean():+.2f},"
              f"{sub['Fz'].mean():+.2f}) N, peak|F| {sub['peakF'].mean():.2f} N")

    # fingerprint quantification: right - left differential per channel
    piv = agg[agg["stage"].isin(["left", "right"])].pivot_table(
        index=["trial", "cycle"], columns="stage", values=CH_NAMES)
    print("\nshear differential (right - left), mean +- s.d. across "
          "trials x cycles:")
    diffs = {}
    for ch in CH_NAMES:
        d = piv[(ch, "right")] - piv[(ch, "left")]
        diffs[ch] = (d.mean(), d.std())
        print(f"  {ch}: {d.mean():+7.1f} +- {d.std():4.1f} uT")
    for s in range(4):
        dx, dy = diffs[f"dBx{s}"][0], diffs[f"dBy{s}"][0]
        rel = "together" if np.sign(dx) == np.sign(dy) else "opposite"
        print(f"  sensor{s}: dX={dx:+.1f}, dY={dy:+.1f} -> X,Y move {rel}"
              f" (|dX|/max ratio {abs(dx)/max(abs(v[0]) for k,v in diffs.items() if k.startswith('dBx')):.2f})")

    if all_anom:
        print("\nanomalies:")
        for a in all_anom:
            print("  " + a)
    else:
        print("\nanomalies: none")

    # representative trial = best stage-wave alignment
    ok_trials = set(agg["trial"].unique())
    rep = max((t for t in aligns if t in ok_trials),
              key=lambda t: aligns[t]["r_pos"])
    make_figure(traces[rep], agg, rep)
    print(f"\nfigure ({rep} as representative) -> {FIG_BASE}.png/.pdf")


if __name__ == "__main__":
    main()

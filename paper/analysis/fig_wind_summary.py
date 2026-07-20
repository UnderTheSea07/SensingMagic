#!/usr/bin/env python3
"""Preliminary wind-data summary figure (internal / Extended Data candidate).

Honest presentation of the 2025-07-17 single-sensor session showing WHY the
current data cannot yield a dB-v calibration:
  a  band-limited RMS vs reference speed (non-monotonic; trial-9 outlier
     excluded and marked)
  b  PSD of clean windows at three speeds: response is narrowband at fan
     rotation harmonics (EMI-confounded), broadband floor speed-independent
  c  steady DC mean shift vs speed with trial order: masked by drift and a
     setup discontinuity at trial 9
Reads out/wind_summary.csv (panel a) and recomputes b, c from raw files.
"""
import csv
import glob
import os
import re

import numpy as np
from scipy import signal
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
OUT = os.path.join(os.path.dirname(__file__), "out")
FIGS = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(FIGS, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 7.5, "axes.labelsize": 8, "axes.titlesize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.8,
    "axes.linewidth": 0.8,
    "xtick.direction": "out", "ytick.direction": "out",
    "xtick.major.width": 0.8, "ytick.major.width": 0.8,
    "xtick.major.size": 2.8, "ytick.major.size": 2.8,
    "xtick.minor.width": 0.6, "ytick.minor.width": 0.6,
    "xtick.minor.size": 1.6, "ytick.minor.size": 1.6,
    "pdf.fonttype": 42, "figure.dpi": 300,
})
ANN = "#555555"
COLS = {0.0: "#000000", 2.43: "#0072B2", 10.0: "#D55E00"}
# Okabe-Ito-blue sequential ramp, same as fig2e (geometry sensitivity)
OI_BLUES = matplotlib.colors.LinearSegmentedColormap.from_list(
    "oi_blues", ["#E7F1F9", "#A8D2EC", "#56B4E9", "#0072B2", "#014A73"])

STARTUP_S, WIN_S, K, GUARD = 15.0, 1.0, 3.0, 1


def load(f):
    d = np.genfromtxt(f, delimiter=",", skip_header=1)
    d = d[~np.isnan(d).any(axis=1)]
    return (d[:, 0] - d[0, 0]) / 1000.0, d[:, 1:4]


def clean_idx(Bhp, fs):
    w = int(WIN_S * fs)
    start = int(STARTUP_S * fs)
    n = (len(Bhp) - start) // w
    stds = np.array([Bhp[start + i * w : start + (i + 1) * w].std(0).max()
                     for i in range(n)])
    bad = stds > K * np.median(stds)
    for i in np.where(bad)[0]:
        bad[max(0, i - GUARD) : i + GUARD + 1] = True
    return [start + i * w for i in np.where(~bad)[0]], w


def main():
    # ---- panel a: summary csv --------------------------------------------
    rows = []
    with open(os.path.join(OUT, "wind_summary.csv")) as fh:
        for r in csv.DictReader(fh):
            rows.append((float(r["v_mps"]), float(r["rmsY"]), float(r["sdY"]),
                         r["file"]))
    rows.sort()
    outlier = [r for r in rows if "trial9_" in r[3]]
    good = [r for r in rows if "trial9_" not in r[3]]

    # ---- panels b/c: recompute -------------------------------------------
    psds, dc = [], []
    files = sorted(glob.glob(os.path.join(ROOT, "*20250717*wind*.csv")))
    for f in files:
        base = os.path.basename(f)
        v = float(re.search(r"wind([0-9.]+)", base).group(1))
        trial = int(re.search(r"trial(\d+)", base).group(1))
        try:
            t, B = load(f)
        except Exception:
            continue
        if len(t) < 20000:
            continue
        fs = 1.0 / np.median(np.diff(t))
        sos_hp = signal.butter(2, 0.5, "highpass", fs=fs, output="sos")
        Bhp = signal.sosfiltfilt(sos_hp, B, axis=0)
        wins, w = clean_idx(Bhp, fs)
        if len(wins) < 10:
            continue
        if v in COLS and not any(abs(p[0] - v) < 1e-6 for p in psds):
            seg = np.concatenate([Bhp[i : i + w] for i in wins], axis=0)
            fr, px = signal.welch(seg[:, 1], fs=fs, nperseg=4096)
            psds.append((v, fr, px))
        means = np.array([B[i : i + w].mean(0) for i in wins])
        dc.append((trial, v, means.mean(0)))

    dc.sort(key=lambda d: d[0])
    ref = np.mean([m for _, v, m in dc if v == 0], axis=0)

    # ---- figure ----------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.2))

    axa = axes[0]
    vv = [r[0] for r in good]
    axa.errorbar(vv, [r[1] for r in good], yerr=[r[2] for r in good],
                 fmt="o", ms=4, color="#0072B2", markeredgecolor="white",
                 markeredgewidth=0.5, capsize=2.5, elinewidth=0.8,
                 capthick=0.8)
    for v, rms, sd, _ in outlier:
        axa.plot(v, rms, "x", color="#D55E00", ms=5, markeredgewidth=1.0)
        axa.annotate("trial 9 (contaminated)", (v, rms), fontsize=6.5,
                     color=ANN, textcoords="offset points", xytext=(0, -9),
                     ha="center", va="top")
    axa.set_xlabel("Reference wind speed (m s$^{-1}$)")
    axa.set_ylabel(u"Band-limited RMS, $B_y$ 5–100 Hz (µT)")
    axa.set_title("a  Fluctuation metric: non-monotonic", loc="left")

    axb = axes[1]
    for v, fr, px in sorted(psds):
        axb.loglog(fr[1:], px[1:], lw=1.0, color=COLS[v],
                   label=f"{v:g} m s$^{{-1}}$")
    axb.set_xlabel("Frequency (Hz)")
    axb.set_ylabel(u"PSD, $B_y$ (µT$^2$ Hz$^{-1}$)")
    ylo, yhi = axb.get_ylim()
    axb.set_ylim(ylo, yhi * 1000)  # headroom so legend clears the peaks
    axb.legend(frameon=False, loc="upper right", handlelength=1.3,
               handletextpad=0.5, borderaxespad=0.1, labelspacing=0.25)
    axb.set_title("b  Fan harmonics: EMI-confounded", loc="left")

    axc = axes[2]
    tr = [d[0] for d in dc if d[0] != 9]
    mag = [np.linalg.norm(d[2] - ref) for d in dc if d[0] != 9]
    vs = [d[1] for d in dc if d[0] != 9]
    sc = axc.scatter(tr, mag, c=vs, cmap=OI_BLUES, s=16, vmin=0, vmax=10,
                     edgecolors="#555555", linewidths=0.4, zorder=3)
    ylo, yhi = axc.set_ylim(top=max(mag) * 1.22)  # headroom for annotation
    # dashed marker line stops below the annotation text so it never
    # strikes through it
    axc.axvline(9, color="#AAAAAA", lw=0.6, ls="--", zorder=1,
                ymax=(max(mag) * 1.02 - ylo) / (yhi - ylo))
    axc.annotate("setup disturbed\n(trial 9, >1000 µT)",
                 (9, max(mag) * 1.19), fontsize=6.5, color=ANN,
                 ha="center", va="top", linespacing=1.25)
    cb = plt.colorbar(sc, ax=axc, label="v (m s$^{-1}$)", pad=0.02)
    cb.outline.set_linewidth(0.6)
    axc.set_xlabel("Trial index (time order)")
    axc.set_ylabel(u"|steady mean shift| vs v=0 (µT)")
    axc.set_title("c  DC shift tracks time, not speed", loc="left")

    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(pad=0.6)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIGS, f"edx_wind_preliminary.{ext}"))
    print("wrote", os.path.join(FIGS, "edx_wind_preliminary.pdf"))


if __name__ == "__main__":
    main()

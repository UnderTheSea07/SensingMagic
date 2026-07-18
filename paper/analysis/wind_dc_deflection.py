#!/usr/bin/env python3
"""Steady-state DC deflection vs wind speed.

Rationale: fan-motor EMI is AC (narrow peaks at rotation harmonics) and does
not shift the mean field; static drag deflection of magnetized cilia does.
All trials were recorded consecutively in one session (2025-07-17, trials
0-15) with speeds interleaved non-monotonically in time, so if the steady
mean tracks speed rather than clock time, drift is excluded as explanation.

Metric: per-file steady-state mean of raw X/Y/Z over artifact-free windows
(same window logic as wind_pipeline), referenced to the v=0 trials of the
same session; also reports trial time order to check against drift.
"""
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

STARTUP_S = 15.0
WIN_S = 1.0
ARTIFACT_K = 3.0
GUARD = 1


def load(f):
    d = np.genfromtxt(f, delimiter=",", skip_header=1)
    d = d[~np.isnan(d).any(axis=1)]
    t = (d[:, 0] - d[0, 0]) / 1000.0
    return t, d[:, 1:4]


def steady_mean(t, B, fs):
    sos_hp = signal.butter(2, 0.5, "highpass", fs=fs, output="sos")
    Bhp = signal.sosfiltfilt(sos_hp, B, axis=0)
    w = int(WIN_S * fs)
    start = int(STARTUP_S * fs)
    n = (len(B) - start) // w
    if n < 10:
        return None, 0
    stds = np.array([Bhp[start + i * w : start + (i + 1) * w].std(axis=0).max()
                     for i in range(n)])
    med = np.median(stds)
    bad = stds > ARTIFACT_K * med
    for i in np.where(bad)[0]:
        bad[max(0, i - GUARD) : i + GUARD + 1] = True
    idx = np.where(~bad)[0]
    means = np.array([B[start + i * w : start + (i + 1) * w].mean(axis=0)
                      for i in idx])
    return means, len(idx)


def main():
    files = sorted(glob.glob(os.path.join(ROOT, "*20250717*wind*.csv")))
    recs = []
    for f in files:
        base = os.path.basename(f)
        v = float(re.search(r"wind([0-9.]+)", base).group(1))
        trial = int(re.search(r"trial(\d+)", base).group(1))
        clock = re.search(r"20250717_(\d{6})", base).group(1)
        try:
            t, B = load(f)
        except Exception:
            continue
        if len(t) < 20000:
            continue
        fs = 1.0 / np.median(np.diff(t))
        means, nw = steady_mean(t, B, fs)
        if means is None or nw < 10:
            continue
        mu = means.mean(axis=0)
        sd = means.std(axis=0)
        recs.append((trial, clock, v, mu, sd, nw))

    recs.sort()  # by trial = time order
    # reference: mean of v=0 trials
    zeros = [r[3] for r in recs if r[2] == 0]
    if not zeros:
        print("no v=0 reference in session")
        return
    ref = np.mean(zeros, axis=0)

    print(f"{'trial':>5} {'clock':>7} {'v':>6} | dX(uT)   dY      dZ    |dB|   (nwin)")
    rows = []
    for trial, clock, v, mu, sd, nw in recs:
        d = mu - ref
        mag = np.linalg.norm(d)
        rows.append((v, d, mag, trial))
        print(f"{trial:5d} {clock:>7} {v:6.2f} | {d[0]:7.2f} {d[1]:7.2f} "
              f"{d[2]:7.2f}  {mag:6.2f}  ({nw})")

    # plot |dB| and components vs v, annotated with trial order
    rows.sort(key=lambda r: r[0])
    vv = np.array([r[0] for r in rows])
    dd = np.array([r[1] for r in rows])
    mm = np.array([r[2] for r in rows])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for k, lab, c in zip(range(3), ["$\\Delta B_x$", "$\\Delta B_y$", "$\\Delta B_z$"],
                         ["#1f77b4", "#ff7f0e", "#2ca02c"]):
        axes[0].plot(vv, dd[:, k], "o-", ms=5, lw=0.8, color=c, label=lab)
    axes[0].axhline(0, color="k", lw=0.5)
    axes[0].set_xlabel("reference wind speed (m/s)")
    axes[0].set_ylabel("steady mean shift vs v=0 (uT)")
    axes[0].legend(fontsize=8)
    axes[1].plot(vv, mm, "ks", ms=6)
    for (v, d, mag, trial) in rows:
        axes[1].annotate(f"t{trial}", (v, mag), fontsize=6,
                         textcoords="offset points", xytext=(4, 3))
    from scipy.stats import spearmanr
    sel = vv > 0
    rho, p = spearmanr(vv[sel], mm[sel])
    axes[1].set_title(f"|dB| vs v (v>0): Spearman rho={rho:.3f}, P={p:.1e}")
    axes[1].set_xlabel("reference wind speed (m/s)")
    axes[1].set_ylabel("|steady mean shift| (uT)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "wind_dc_deflection.png"), dpi=130)
    print("\nSpearman (v>0):", f"rho={rho:.3f} P={p:.2e}")
    print("wrote", os.path.join(OUT, "wind_dc_deflection.png"))


if __name__ == "__main__":
    main()

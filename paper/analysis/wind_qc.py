#!/usr/bin/env python3
"""QC + time-structure scan for wind-speed calibration files.

Step 1 of the airflow pipeline: for each *wind*.csv, report sampling rate,
duration, dropouts, and the rolling mean/std of each axis over time so we can
see whether wind is on for the whole file or switched on/off, and where
operator transients sit. No conclusions drawn here — inspection only.
"""
import glob
import os
import re

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
OUT = os.path.join(os.path.dirname(__file__), "out")
os.makedirs(OUT, exist_ok=True)


def load(f):
    d = np.genfromtxt(f, delimiter=",", skip_header=1)
    d = d[~np.isnan(d).any(axis=1)]
    t = (d[:, 0] - d[0, 0]) / 1000.0  # s
    return t, d[:, 1], d[:, 2], d[:, 3]


def main():
    files = sorted(glob.glob(os.path.join(ROOT, "*wind*.csv")))
    rows = []
    for f in files:
        m = re.search(r"wind([0-9.]+)", os.path.basename(f))
        v = float(m.group(1)) if m else np.nan
        try:
            t, x, y, z = load(f)
        except Exception as e:
            print(f"SKIP {os.path.basename(f)}: {e}")
            continue
        if len(t) < 1000:
            print(f"SKIP {os.path.basename(f)}: only {len(t)} rows")
            continue
        dt = np.diff(t)
        fs = 1.0 / np.median(dt)
        gaps = int((dt > 5 * np.median(dt)).sum())
        rows.append((v, os.path.basename(f), fs, t[-1], len(t), gaps))

        # rolling stats, 1 s window
        w = max(1, int(fs))
        n = len(x) // w
        rt = t[: n * w : w] + 0.5
        rmean = {a: sig[: n * w].reshape(n, w).mean(1) for a, sig in (("X", x), ("Y", y), ("Z", z))}
        rstd = {a: sig[: n * w].reshape(n, w).std(1) for a, sig in (("X", x), ("Y", y), ("Z", z))}

        fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
        for a in ("X", "Y", "Z"):
            axes[0].plot(rt, rmean[a] - rmean[a][0], lw=0.8, label=a)
            axes[1].semilogy(rt, rstd[a] + 1e-3, lw=0.8, label=a)
        axes[0].set_ylabel("1-s mean − start (uT)")
        axes[1].set_ylabel("1-s std (uT)")
        axes[1].set_xlabel("time (s)")
        axes[0].legend(fontsize=7)
        axes[0].set_title(f"v={v} m/s  {os.path.basename(f)[:60]}", fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, f"qc_v{v:05.2f}_" + os.path.basename(f)[:40] + ".png"), dpi=110)
        plt.close(fig)

    rows.sort()
    print(f"{'v':>6} {'fs(Hz)':>7} {'dur(s)':>8} {'N':>8} {'gaps':>5}  file")
    for v, name, fs, dur, n, gaps in rows:
        print(f"{v:6.2f} {fs:7.0f} {dur:8.1f} {n:8d} {gaps:5d}  {name[:55]}")


if __name__ == "__main__":
    main()

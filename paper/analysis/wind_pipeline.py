#!/usr/bin/env python3
"""Airflow calibration pipeline for magnetic cilia sensor (single MLX90393).

Pipeline (honest, artifact-aware):
 1. Load each *wind*.csv (fs ~= 1000 Hz, ground-truth speed v parsed from
    filename, measured by reference anemometer per lab notes).
 2. Discard the first STARTUP_S seconds (fan spin-up / handling transient).
 3. Split the remainder into 1-s windows; reject artifact windows whose
    broadband std exceeds ARTIFACT_K x the file's median window std, plus
    one guard window on each side (operator-touch spikes ~10 uT).
 4. Welch PSD per file on clean windows -> identify wind-sensitive band.
 5. Metric: band-limited RMS per axis (Butterworth band-pass, applied on
    clean segments) -> per-window values -> per-file mean +/- s.d.
 6. Aggregate across files, fit RMS vs v, report monotonicity + R^2.

Outputs: out/wind_psd.png, out/wind_calibration.png, out/wind_summary.csv
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
os.makedirs(OUT, exist_ok=True)

STARTUP_S = 15.0     # discard fan spin-up / handling at file start
WIN_S = 1.0          # analysis window length
ARTIFACT_K = 3.0     # reject windows with std > K x median window std
GUARD = 1            # extra windows rejected around an artifact window
BAND = (5.0, 100.0)  # wind-sensitive band; confirmed against PSD output
HP_DETREND = 0.5     # Hz, high-pass to remove drift before PSD


def load(f):
    d = np.genfromtxt(f, delimiter=",", skip_header=1)
    d = d[~np.isnan(d).any(axis=1)]
    t = (d[:, 0] - d[0, 0]) / 1000.0
    return t, d[:, 1:4]


def clean_windows(sig3, fs):
    """Return list of artifact-free windows (start indices) after startup."""
    w = int(WIN_S * fs)
    start = int(STARTUP_S * fs)
    n = (len(sig3) - start) // w
    if n < 10:
        return [], w
    stds = np.array([sig3[start + i * w : start + (i + 1) * w].std(axis=0).max()
                     for i in range(n)])
    med = np.median(stds)
    bad = stds > ARTIFACT_K * med
    for i in np.where(bad)[0]:
        bad[max(0, i - GUARD) : i + GUARD + 1] = True
    return [start + i * w for i in np.where(~bad)[0]], w


def main():
    files = sorted(glob.glob(os.path.join(ROOT, "*wind*.csv")))
    sos_band = None
    psd_fig, psd_axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    cmap = plt.cm.viridis

    summary = []  # (v, file, n_windows, rms_x, sd_x, rms_y, sd_y, rms_z, sd_z)
    psd_store = []

    for f in files:
        m = re.search(r"wind([0-9.]+)", os.path.basename(f))
        v = float(m.group(1)) if m else np.nan
        try:
            t, B = load(f)
        except Exception:
            continue
        if len(t) < 20000:
            print(f"SKIP short {os.path.basename(f)}")
            continue
        fs = 1.0 / np.median(np.diff(t))
        if sos_band is None:
            sos_band = signal.butter(4, BAND, "bandpass", fs=fs, output="sos")
            sos_hp = signal.butter(2, HP_DETREND, "highpass", fs=fs, output="sos")

        # artifact-free windows on high-passed signal
        Bhp = signal.sosfiltfilt(sos_hp, B, axis=0)
        wins, w = clean_windows(Bhp, fs)
        if len(wins) < 10:
            print(f"SKIP {os.path.basename(f)}: {len(wins)} clean windows")
            continue

        # PSD over concatenated clean windows
        seg = np.concatenate([Bhp[i : i + w] for i in wins], axis=0)
        freqs, pxx = signal.welch(seg, fs=fs, nperseg=4096, axis=0)
        psd_store.append((v, freqs, pxx))

        # band-limited RMS per clean window
        Bband = signal.sosfiltfilt(sos_band, B, axis=0)
        rms = np.array([np.sqrt((Bband[i : i + w] ** 2).mean(axis=0)) for i in wins])
        mu, sd = rms.mean(axis=0), rms.std(axis=0)
        summary.append((v, os.path.basename(f), len(wins), *mu, *sd))
        print(f"v={v:5.2f}  windows={len(wins):4d}  bandRMS X/Y/Z = "
              f"{mu[0]:.3f}/{mu[1]:.3f}/{mu[2]:.3f} uT")

    # ---- PSD figure ----------------------------------------------------
    vmax = max(v for v, _, _ in psd_store)
    for v, freqs, pxx in sorted(psd_store, key=lambda s: s[0]):
        c = cmap(v / vmax if vmax else 0)
        for k, ax in enumerate(psd_axes):
            ax.loglog(freqs[1:], pxx[1:, k], color=c, lw=0.8)
    for k, ax, lab in zip(range(3), psd_axes, "XYZ"):
        ax.set_title(f"B{lab.lower()}")
        ax.set_xlabel("frequency (Hz)")
        ax.axvspan(*BAND, alpha=0.08, color="red")
    psd_axes[0].set_ylabel("PSD (uT$^2$/Hz)")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, vmax))
    psd_fig.colorbar(sm, ax=psd_axes, label="wind speed (m/s)")
    psd_fig.suptitle("Welch PSD, clean steady windows (red = analysis band)")
    psd_fig.savefig(os.path.join(OUT, "wind_psd.png"), dpi=130)

    # ---- calibration figure -------------------------------------------
    summary.sort()
    arr = np.array([[s[0], *s[3:6], *s[6:9]] for s in summary])
    fig, ax = plt.subplots(figsize=(6, 4.5))
    labels = ["$B_x$", "$B_y$", "$B_z$"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    for k in range(3):
        ax.errorbar(arr[:, 0], arr[:, 1 + k], yerr=arr[:, 4 + k], fmt="o",
                    ms=5, capsize=3, color=colors[k], label=labels[k])
    # fit on the best axis (highest dynamic range): choose by span ratio
    span = [(arr[:, 1 + k].max() - arr[:, 1 + k].min()) / arr[:, 1 + k].min()
            for k in range(3)]
    kbest = int(np.argmax(span))
    vv, rr = arr[:, 0], arr[:, 1 + kbest]
    # power-law fit  RMS = a * v^b + c  (c = still-air floor)
    from scipy.optimize import curve_fit
    def model(v, a, b, c):
        return a * np.maximum(v, 0) ** b + c
    try:
        p0 = [0.1, 1.5, rr[vv == 0].mean() if (vv == 0).any() else rr.min()]
        popt, _ = curve_fit(model, vv, rr, p0=p0, maxfev=20000)
        vf = np.linspace(0, vv.max(), 200)
        pred = model(vv, *popt)
        ss_res = ((rr - pred) ** 2).sum()
        ss_tot = ((rr - rr.mean()) ** 2).sum()
        r2 = 1 - ss_res / ss_tot
        ax.plot(vf, model(vf, *popt), "--", color=colors[kbest], lw=1,
                label=f"{labels[kbest]}: $av^b+c$, $R^2$={r2:.3f}")
        print(f"\nFIT axis {['X','Y','Z'][kbest]}: a={popt[0]:.4f} b={popt[1]:.3f} "
              f"c={popt[2]:.4f}  R2={r2:.4f}")
        # Spearman monotonicity across files
        from scipy.stats import spearmanr
        rho, pval = spearmanr(vv, rr)
        print(f"Spearman rho={rho:.3f}  P={pval:.2e}  (n={len(vv)} files)")
    except Exception as e:
        print("fit failed:", e)
    ax.set_xlabel("reference wind speed (m/s)")
    ax.set_ylabel(f"band-limited RMS {BAND[0]:.0f}-{BAND[1]:.0f} Hz (uT)")
    ax.legend(fontsize=8)
    ax.set_title("Airflow calibration, L8/d0.8 cilia, single MLX90393")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "wind_calibration.png"), dpi=130)

    with open(os.path.join(OUT, "wind_summary.csv"), "w") as fh:
        fh.write("v_mps,file,n_clean_windows,rmsX,rmsY,rmsZ,sdX,sdY,sdZ\n")
        for s in summary:
            fh.write(f"{s[0]},{s[1]},{s[2]},"
                     + ",".join(f"{x:.4f}" for x in s[3:]) + "\n")
    print("\nwrote", os.path.join(OUT, "wind_summary.csv"))


if __name__ == "__main__":
    main()

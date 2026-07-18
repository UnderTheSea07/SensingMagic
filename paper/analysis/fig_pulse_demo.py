#!/usr/bin/env python3
"""Fig. 5d candidate (preliminary, single sensor): pulse micro-vibration demo.

Uses the best-quality real recording (single_pulse0120_66_500hz, beat-SNR
+0.6 dB, 93 s). Panels:
  a  raw Bz (drift visible) + band-passed trace, 12-s excerpt
  b  band-passed with detected beats, 12-s excerpt
  c  beat-aligned individual beats + mean waveform
Qualitative demonstration only — no HR validation claim (no ECG).
Nature style: 7 pt, thin lines, colorblind-safe, vector PDF + 300 dpi PNG.
"""
import os

import numpy as np
from scipy import signal
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
FIGS = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(FIGS, exist_ok=True)

FILE = "single_pulse0120_66_500hz_20260120_235652.csv"
BAND = (0.5, 10.0)

plt.rcParams.update({
    "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7,
    "xtick.labelsize": 6, "ytick.labelsize": 6, "legend.fontsize": 6,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "pdf.fonttype": 42, "figure.dpi": 300,
})
C_RAW, C_FILT, C_MEAN, C_BEAT = "#888888", "#0072B2", "#D55E00", "#BBBBBB"


def load(f):
    with open(f) as fh:
        first = fh.readline()
    skip = 1 if first.lower().startswith("ms") else 0
    d = np.genfromtxt(f, delimiter=",", skip_header=skip)
    d = d[~np.isnan(d).any(axis=1)]
    return (d[:, 0] - d[0, 0]) / 1000.0, d[:, 1:4]


def main():
    t, B = load(os.path.join(ROOT, FILE))
    fs = 1.0 / np.median(np.diff(t))
    sos = signal.butter(3, BAND, "bandpass", fs=fs, output="sos")
    Bf = signal.sosfiltfilt(sos, B, axis=0)

    # axis with max power in HR band
    freqs, pxx = signal.welch(Bf, fs=fs, nperseg=int(30 * fs), axis=0)
    hrband = (freqs >= 0.7) & (freqs <= 2.0)
    k = int(np.argmax(pxx[hrband].sum(axis=0)))
    s, raw = Bf[:, k], B[:, k]
    ax_name = "xyz"[k]

    prom = 2.5 * np.median(np.abs(s - np.median(s)))
    peaks, _ = signal.find_peaks(s, distance=int(fs * 60 / 140), prominence=prom)

    w_pre, w_post = int(0.25 * fs), int(0.70 * fs)
    segs = np.array([s[p - w_pre : p + w_post] for p in peaks[2:-2]
                     if p - w_pre >= 0 and p + w_post <= len(s)])
    mean_beat = segs.mean(axis=0)
    tb = (np.arange(-w_pre, w_post)) / fs

    t0, t1 = 20.0, 32.0
    sl = slice(int(t0 * fs), int(t1 * fs))

    fig, axes = plt.subplots(1, 3, figsize=(7.1, 1.9))

    axa = axes[0]
    axa.plot(t[sl], raw[sl] - raw[sl].mean(), color=C_RAW, lw=0.5,
             label=f"raw $B_{ax_name}$ (mean-removed)")
    axa.plot(t[sl], s[sl], color=C_FILT, lw=0.6,
             label=f"band-passed {BAND[0]}–{BAND[1]} Hz")
    axa.set_xlabel("Time (s)")
    axa.set_ylabel(u"ΔB (µT)")
    axa.legend(frameon=False, loc="upper right")
    axa.set_title("Radial-artery recording, single MLX90393", loc="left")

    axb = axes[1]
    axb.plot(t[sl], s[sl], color=C_FILT, lw=0.6)
    pk_in = peaks[(peaks >= sl.start) & (peaks < sl.stop)]
    axb.plot(t[pk_in], s[pk_in], "v", color=C_MEAN, ms=3,
             label="detected beats")
    axb.set_xlabel("Time (s)")
    axb.set_ylabel(u"ΔB (µT)")
    axb.legend(frameon=False, loc="upper right")
    axb.set_title("Beat detection", loc="left")

    axc = axes[2]
    step = max(1, len(segs) // 60)
    axc.plot(tb, segs[::step].T, color=C_BEAT, lw=0.25, zorder=1)
    axc.plot(tb, mean_beat, color=C_MEAN, lw=1.2, zorder=2,
             label=f"mean of {len(segs)} beats")
    axc.set_xlabel("Time from beat peak (s)")
    axc.set_ylabel(u"ΔB (µT)")
    axc.legend(frameon=False, loc="upper right")
    axc.set_title("Beat-aligned average", loc="left")

    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout(pad=0.4)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIGS, f"fig5d_pulse_demo.{ext}"))
    print(f"axis={ax_name} fs={fs:.0f} Hz beats={len(segs)} "
          f"mean-beat peak={mean_beat.max():.2f} uT")
    print("wrote", os.path.join(FIGS, "fig5d_pulse_demo.pdf"))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Fig. 5d candidate (preliminary, single sensor): pulse micro-vibration demo.

Uses the best-quality real recording (single_pulse0120_66_500hz, beat-SNR
+0.6 dB, 93 s). Panels:
  a  raw Bz (drift visible) + band-passed trace, 12-s excerpt
  b  band-passed with detected beats, 12-s excerpt
  c  beat-aligned individual beats + mean waveform
Qualitative demonstration only — no HR validation claim (no ECG).
Nature style: Okabe-Ito palette, thin lines, vector PDF + 300 dpi PNG.
"""
import os

import numpy as np
from scipy import signal
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
FIGS = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(FIGS, exist_ok=True)

FILE = "single_pulse0120_66_500hz_20260120_235652.csv"
BAND = (0.5, 10.0)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 7.5, "axes.labelsize": 8, "axes.titlesize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.8,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8, "ytick.major.width": 0.8,
    "xtick.major.size": 2.8, "ytick.major.size": 2.8,
    "xtick.direction": "out", "ytick.direction": "out",
    "pdf.fonttype": 42, "figure.dpi": 300,
})
C_RAW, C_FILT, C_MEAN, C_BEAT = "#888888", "#0072B2", "#D55E00", "#888888"
C_REF = "#AAAAAA"


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

    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.0))

    # -- a: raw + band-passed -------------------------------------------------
    axa = axes[0]
    r = raw[sl] - raw[sl].mean()
    axa.plot(t[sl], r, color=C_RAW, lw=0.7, alpha=0.35,
             label=f"raw $B_{ax_name}$ (mean-removed)")
    axa.plot(t[sl], s[sl], color=C_FILT, lw=1.2,
             label=f"band-passed {BAND[0]:g}–{BAND[1]:g} Hz")
    lo, hi = r.min(), r.max()
    rng = hi - lo
    axa.set_ylim(lo - 0.04 * rng, hi + 0.46 * rng)   # headroom for legend
    axa.set_xlabel("Time (s)")
    axa.set_ylabel(u"ΔB (µT)")
    axa.legend(frameon=False, loc="upper left", handlelength=1.3,
               labelspacing=0.25, borderaxespad=0.15)
    axa.set_title("Radial-artery recording", loc="left")

    # -- b: beat detection ----------------------------------------------------
    axb = axes[1]
    axb.plot(t[sl], s[sl], color=C_FILT, lw=1.0)
    pk_in = peaks[(peaks >= sl.start) & (peaks < sl.stop)]
    lo, hi = s[sl].min(), s[sl].max()
    rng = hi - lo
    off = 0.09 * rng                                  # lift markers off trace
    axb.plot(t[pk_in], s[pk_in] + off, "v", color=C_MEAN, ms=2.8,
             mec="white", mew=0.4, ls="none", label="detected beats")
    axb.set_ylim(lo - 0.04 * rng, hi + 0.34 * rng)
    axb.set_xlabel("Time (s)")
    axb.set_ylabel(u"ΔB (µT)")
    axb.legend(frameon=False, loc="upper left", handlelength=1.0,
               borderaxespad=0.15)
    axb.set_title("Beat detection", loc="left")

    # -- c: beat-aligned average ---------------------------------------------
    axc = axes[2]
    axc.axhline(0.0, color=C_REF, lw=0.6, ls=(0, (4, 3)), zorder=0)
    step = max(1, len(segs) // 60)
    axc.plot(tb, segs[::step].T, color=C_BEAT, lw=0.4, alpha=0.3, zorder=1)
    axc.plot(tb, mean_beat, color=C_MEAN, lw=1.8, zorder=3,
             label=f"mean of {len(segs)} beats")
    lo, hi = segs[::step].min(), segs[::step].max()
    rng = hi - lo
    axc.set_ylim(lo - 0.04 * rng, hi + 0.28 * rng)   # clean corner for legend
    axc.set_xlabel("Time from beat peak (s)")
    axc.set_ylabel(u"ΔB (µT)")
    axc.legend(frameon=False, loc="upper left", handlelength=1.3,
               borderaxespad=0.15)
    axc.set_title("Beat-aligned average", loc="left")

    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
    for ax in (axa, axb):
        ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))

    fig.tight_layout(pad=0.6)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIGS, f"fig5d_pulse_demo.{ext}"))
    print(f"axis={ax_name} fs={fs:.0f} Hz beats={len(segs)} "
          f"mean-beat peak={mean_beat.max():.2f} uT")
    print("wrote", os.path.join(FIGS, "fig5d_pulse_demo.pdf"))


if __name__ == "__main__":
    main()

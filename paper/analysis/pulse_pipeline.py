#!/usr/bin/env python3
"""Pulse (Fig. 5d-f) feasibility analysis.

Dataset facts (verified):
 - No ECG/PPG files exist anywhere in the repo -> the paper's HR-MAE /
   ECG-delay claims cannot be computed. What CAN be tested:
 - single_pulse0120_* / single_XYZ_pulse* filenames encode a manually
   noted reference HR (e.g. _65, _68to70) from an unnamed commercial
   device -> coarse HR agreement check (magnetic HR vs label).
 - Waveform quality: band-passed signal, beat detection, beat-aligned
   average, SNR -> supports "pulse micro-vibrations are recorded", not
   full physiological validation.

Pipeline per file:
 1. Load (headerless or ms,X,Y,Z header), fs from timestamps.
 2. Band-pass 0.5-10 Hz; pick axis with max power in 0.7-2.0 Hz.
 3. HR: Welch peak in 0.7-2.0 Hz (42-120 bpm), fundamental-vs-harmonic
    guard; also beat count via find_peaks for cross-check.
 4. Beat-aligned mean waveform + SNR (mean beat energy / residual).
 5. Compare with filename HR label where present.
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

HR_BAND = (0.7, 2.0)   # 42-120 bpm
PULSE_BAND = (0.5, 10.0)


def load(f):
    with open(f) as fh:
        first = fh.readline()
    skip = 1 if first.lower().startswith("ms") else 0
    d = np.genfromtxt(f, delimiter=",", skip_header=skip)
    d = d[~np.isnan(d).any(axis=1)]
    t = (d[:, 0] - d[0, 0]) / 1000.0
    return t, d[:, 1:4]


def hr_label(name):
    m = re.search(r"(?:pulse\d*|0120)_(\d{2,3})(?:to(\d{2,3}))?(?:_|\.)", name)
    if not m:
        return None
    lo = float(m.group(1))
    hi = float(m.group(2)) if m.group(2) else lo
    return (min(lo, hi), max(lo, hi))


def analyze(f, make_plot=False):
    t, B = load(f)
    if len(t) < 3000:
        return None
    fs = 1.0 / np.median(np.diff(t))
    if not (20 < fs < 2000):
        return None
    dur = t[-1]
    sos = signal.butter(3, [max(PULSE_BAND[0], 0.5), min(PULSE_BAND[1], 0.45 * fs)],
                        "bandpass", fs=fs, output="sos")
    Bf = signal.sosfiltfilt(sos, B, axis=0)

    # axis with max HR-band power
    freqs, pxx = signal.welch(Bf, fs=fs, nperseg=min(len(Bf), int(30 * fs)), axis=0)
    band = (freqs >= HR_BAND[0]) & (freqs <= HR_BAND[1])
    if band.sum() < 3:
        return None
    k = int(np.argmax(pxx[band].sum(axis=0)))
    p = pxx[:, k]
    # fundamental: strongest in-band peak; guard against picking a harmonic:
    # if f/2 also has a local peak with power > 0.3x, take f/2.
    fpk = freqs[band][np.argmax(p[band])]
    half = fpk / 2
    if half >= HR_BAND[0]:
        i_half = np.argmin(np.abs(freqs - half))
        if p[i_half] > 0.3 * p[np.argmin(np.abs(freqs - fpk))]:
            fpk = half
    hr_psd = fpk * 60

    # beat detection on chosen axis
    s = Bf[:, k]
    prom = 2.5 * np.median(np.abs(s - np.median(s)))
    peaks, _ = signal.find_peaks(s, distance=int(fs * 60 / 140), prominence=prom)
    hr_beats = 60 * (len(peaks) - 1) / (t[peaks[-1]] - t[peaks[0]]) if len(peaks) > 5 else np.nan
    ibi = np.diff(t[peaks]) if len(peaks) > 5 else np.array([np.nan])
    cv = 100 * ibi.std() / ibi.mean() if len(ibi) > 4 else np.nan

    # beat-aligned average + SNR
    snr = np.nan
    if len(peaks) > 10:
        w = int(0.7 * fs)
        segs = np.array([s[p0 - w // 3 : p0 + w] for p0 in peaks[2:-2]
                         if p0 - w // 3 >= 0 and p0 + w <= len(s)])
        if len(segs) > 8:
            mean_beat = segs.mean(axis=0)
            resid = segs - mean_beat
            snr = 10 * np.log10((mean_beat ** 2).mean() / (resid ** 2).mean())
            if make_plot:
                fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
                sl = slice(int(10 * fs), int(20 * fs))
                axes[0].plot(t[sl], s[sl], lw=0.6)
                axes[0].set_title(f"filtered {'XYZ'[k]} (10 s)")
                axes[0].set_xlabel("time (s)")
                axes[0].set_ylabel("uT")
                tb = (np.arange(len(mean_beat)) - w // 3) / fs
                axes[1].plot(tb, segs[::max(1, len(segs)//40)].T, color="0.8", lw=0.3)
                axes[1].plot(tb, mean_beat, "r", lw=1.5)
                axes[1].set_title(f"beat-aligned n={len(segs)}, SNR={snr:.1f} dB")
                axes[1].set_xlabel("time from peak (s)")
                axes[2].semilogy(freqs, p, lw=0.8)
                axes[2].axvspan(*HR_BAND, alpha=0.1, color="red")
                axes[2].set_xlim(0, 12)
                axes[2].set_title(f"PSD, HR={hr_psd:.0f} bpm")
                axes[2].set_xlabel("Hz")
                fig.suptitle(os.path.basename(f)[:70], fontsize=8)
                fig.tight_layout()
                fig.savefig(os.path.join(OUT, "pulse_" +
                            os.path.basename(f)[:45] + ".png"), dpi=120)
                plt.close(fig)

    return dict(file=os.path.basename(f), fs=fs, dur=dur, axis="XYZ"[k],
                hr_psd=hr_psd, hr_beats=hr_beats, n_beats=len(peaks),
                cv=cv, snr=snr, label=hr_label(os.path.basename(f)))


def main():
    pats = ["single_pulse0120_*.csv", "single_XYZ*pulse*.csv",
            "*l8_d0.8_num1_trial*_pulse_fastmode.csv"]
    files = []
    for p in pats:
        files += sorted(glob.glob(os.path.join(ROOT, p)))
    print(f"analyzing {len(files)} files\n")
    results, errs = [], []
    for i, f in enumerate(files):
        try:
            r = analyze(f, make_plot=(i % 8 == 0))
        except Exception as e:
            errs.append((os.path.basename(f), str(e)))
            continue
        if r:
            results.append(r)

    print(f"{'file':<52} {'fs':>5} {'dur':>6} {'ax':>2} {'HRpsd':>6} "
          f"{'HRbeat':>6} {'CV%':>5} {'SNR':>5}  label")
    ok = 0
    for r in results:
        lab = f"{r['label'][0]:.0f}-{r['label'][1]:.0f}" if r['label'] else "-"
        inlab = ""
        if r['label']:
            lo, hi = r['label']
            inlab = "  MATCH" if lo - 5 <= r['hr_psd'] <= hi + 5 else "  MISS"
            ok += inlab == "  MATCH"
        print(f"{r['file'][:52]:<52} {r['fs']:5.0f} {r['dur']:6.0f} "
              f"{r['axis']:>2} {r['hr_psd']:6.1f} {r['hr_beats']:6.1f} "
              f"{r['cv']:5.1f} {r['snr']:5.1f}  {lab}{inlab}")
    labeled = [r for r in results if r['label']]
    if labeled:
        err = [abs(r['hr_psd'] - np.mean(r['label'])) for r in labeled]
        print(f"\nlabeled files: {len(labeled)}, within label±5 bpm: {ok}, "
              f"median |HR err| vs label midpoint = {np.median(err):.1f} bpm")
    for name, e in errs:
        print("ERR", name, e[:60])


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Fig. 2f (cyclic stability): within-checkpoint per-cycle stability of the
5x5 cilia array (L = 5 mm, d = 0.5 mm, pitch 0.5 mm) during fatigue cycling.

Source (headerless CSV, columns [t_ms, Bx, By, Bz] in µT, sampled ~31 Hz —
median dt = 32 ms):
  /Users/arielzhang/Desktop/磁毛传感器可靠性分析原始数据/5x5_0.5_5_0.5/10000z.csv
recorded AT the 10,000-cycle checkpoint. The time column can jump between
recording segments, so rows are split into contiguous segments (dt < 100 ms)
and only segments longer than 60 s are analysed (here: one 87-min segment
that contains exactly 10,000 loading cycles).

Pipeline (all annotated numbers computed here, nothing hardcoded):
  1. bad-packet removal: drop rows where any axis deviates > 8x its global
     MAD from a centred 5-sample rolling median (removes single-sample
     spikes, e.g. Bx = +1228 µT in an otherwise -110..-37 µT channel).
  2. interpolate Bx onto a uniform grid, low-pass (Butterworth, 3rd order)
     at 4 Hz for cycle detection and 8 Hz for amplitude read-out; subtract
     a 4-s rolling-median baseline.
  3. cycle detection: scipy.signal.find_peaks on the negated 4-Hz signal
     (loading dips, prominence >= 20 µT, min spacing 0.3 s).
  4. per-cycle peak-to-peak amplitude = max - min of the 8-Hz signal
     between consecutive dips.

Outputs: paper/figures/fig2f_cyclic_stability.{pdf,png}
"""
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks

SRC = ("/Users/arielzhang/Desktop/磁毛传感器可靠性分析原始数据/"
       "5x5_0.5_5_0.5/10000z.csv")
FIGS = "/Users/arielzhang/Desktop/SensingMagic/paper/figures"
os.makedirs(FIGS, exist_ok=True)

plt.rcParams.update({
    "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7,
    "xtick.labelsize": 6, "ytick.labelsize": 6, "legend.fontsize": 6,
    "axes.linewidth": 0.6, "pdf.fonttype": 42, "figure.dpi": 300,
})

BLUE, ORANGE = "#0072B2", "#D55E00"

# ---------------------------------------------------------------- load + QC
df = pd.read_csv(SRC, header=None, names=["t_ms", "Bx", "By", "Bz"])

# bad packets: any axis > 8x global MAD away from 5-sample rolling median
bad = np.zeros(len(df), bool)
for c in ["Bx", "By", "Bz"]:
    x = df[c]
    rmed = x.rolling(5, center=True, min_periods=1).median()
    mad = np.median(np.abs(x - x.median()))
    bad |= (np.abs(x - rmed) > 8 * mad).values
n_bad = int(bad.sum())
df = df[~bad].reset_index(drop=True)

# contiguous segments: split where dt >= 100 ms, keep those >= 60 s long
gap = np.where(np.diff(df.t_ms.values) >= 100.0)[0]
bounds = np.concatenate([[0], gap + 1, [len(df)]])
segs = [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)
        if df.t_ms.values[bounds[i + 1] - 1] - df.t_ms.values[bounds[i]]
        >= 60_000]
assert segs, "no usable contiguous segment found"

# ------------------------------------------------- per-segment cycle metrics
pp_all, rate_all, excerpt = [], [], None
for s, e in segs:
    t = df.t_ms.values[s:e] / 1000.0
    fs = 1000.0 / np.median(np.diff(df.t_ms.values[s:e]))
    tu = np.arange(t[0], t[-1], 1.0 / fs)
    bxu = np.interp(tu, t, df.Bx.values[s:e])

    b4, a4 = butter(3, 4.0 / (fs / 2), "low")
    b8, a8 = butter(3, 8.0 / (fs / 2), "low")
    base = pd.Series(bxu).rolling(int(fs * 4) | 1, center=True,
                                  min_periods=1).median().values
    sig4 = filtfilt(b4, a4, bxu) - base    # for dip detection
    sig8 = filtfilt(b8, a8, bxu) - base    # for amplitude read-out
    raw = bxu - base                        # for the raw excerpt panel

    dips, _ = find_peaks(-sig4, distance=int(0.3 * fs), prominence=20.0)
    pp = np.array([np.ptp(sig8[dips[i]:dips[i + 1] + 1])
                   for i in range(len(dips) - 1)])
    pp_all.append(pp)
    rate_all.append(1.0 / np.median(np.diff(tu[dips])))

    if excerpt is None:                    # 20-s excerpt from mid-segment
        t0 = tu[0] + 0.5 * (tu[-1] - tu[0])
        m = (tu >= t0) & (tu < t0 + 20.0)
        excerpt = (tu[m] - t0, raw[m])

pp = np.concatenate(pp_all)
n_cyc = len(pp)
rate = float(np.median(rate_all))
pp_med = float(np.median(pp))
k = max(1, n_cyc // 10)
pp_first, pp_last = float(np.median(pp[:k])), float(np.median(pp[-k:]))
drift_pct = 100.0 * (pp_last - pp_first) / pp_first
roll = pd.Series(pp).rolling(301, center=True, min_periods=50).median()

print(f"bad packets removed : {n_bad} rows "
      f"({100 * n_bad / (len(df) + n_bad):.3f}%)")
print(f"segments used       : {len(segs)}  (longest "
      f"{(df.t_ms.values[segs[0][1]-1] - df.t_ms.values[segs[0][0]])/1000:.0f} s)")
print(f"cycles detected     : {n_cyc}  at {rate:.2f} Hz")
print(f"median p-p amplitude: {pp_med:.1f} µT "
      f"(IQR {np.percentile(pp, 25):.1f}-{np.percentile(pp, 75):.1f})")
print(f"drift first->last 10%: {drift_pct:+.1f}% "
      f"({pp_first:.1f} -> {pp_last:.1f} µT)")

# ------------------------------------------------------------------- figure
fig, (axa, axb) = plt.subplots(
    1, 2, figsize=(7.0, 2.3), gridspec_kw={"width_ratios": [1.0, 1.55]})

# panel a: raw 20-s excerpt
axa.plot(excerpt[0], excerpt[1], color=BLUE, lw=0.6)
axa.set_xlabel("Time within excerpt (s)")
axa.set_ylabel(r"$\Delta B_x$ ($\mu$T)")
axa.set_xlim(0, 20)
axa.set_xticks([0, 5, 10, 15, 20])
axa.set_title(f"a  Raw $B_x$ during cyclic loading ({rate:.2f} Hz, "
              "20 s excerpt)", loc="left")

# panel b: per-cycle peak-to-peak amplitude, y-axis from 0 (not truncated)
idx = np.arange(1, n_cyc + 1)
axb.plot(idx, pp, ".", color="0.62", ms=1.4, alpha=0.30, rasterized=True,
         label="per-cycle p–p")
axb.plot(idx, roll, color=ORANGE, lw=1.1, label="rolling median (301 cycles)")
axb.set_xlabel("Cycle index within recording")
axb.set_ylabel(r"Peak-to-peak $B_x$ ($\mu$T)")
axb.set_ylim(0, np.percentile(pp, 99.5) * 1.28)
axb.set_xlim(0, n_cyc * 1.02)
axb.set_title(f"b  Within-checkpoint stability ($n$ = {n_cyc:,} cycles)",
              loc="left")
drift_txt = f"{drift_pct:+.1f}".replace("-", "−")
axb.annotate(f"median p–p = {pp_med:.1f} $\\mu$T\n"
             f"drift, first→last 10% of cycles: {drift_txt}%",
             xy=(0.03, 0.10), xycoords="axes fraction", fontsize=6,
             va="bottom")
axb.legend(frameon=False, loc="upper right", handletextpad=0.4,
           borderaxespad=0.2, markerscale=4)

for ax in (axa, axb):
    ax.spines[["top", "right"]].set_visible(False)

fig.tight_layout(pad=0.5)
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(FIGS, f"fig2f_cyclic_stability.{ext}"))
print("saved fig2f_cyclic_stability.{pdf,png}")

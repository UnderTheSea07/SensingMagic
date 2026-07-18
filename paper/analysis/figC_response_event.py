#!/usr/bin/env python
"""
figC_response_event.py

Fig. 2f — dynamic response and repeatability of the magnetic-cilia patch
(manual loading, hand-pressed taps).

Sub-panels are labelled f1/f2/f3 (neutral, to avoid clashing with the
top-level a-f letters of the Fig. 2 composite).

Panel f1: representative tap-response event from the raw high-rate recording
         (headerless CSV [t_ms, Bx, By, Bz], fs ~ 1 kHz). Bad packets (rows
         where the axes jump to fixed clipped levels) are DROPPED (never
         interpolated): a row is discarded when any axis deviates by more
         than 8 x the global robust MAD (1.4826*MAD) from a long centred
         rolling median. The 10-90 % rise time and 90-10 % recovery time are
         computed from the data (crossings on a 7-sample moving average,
         linear interpolation between samples).
Panel f2: 5 aligned press-release cycles (Fig4a_overlay.csv) + their mean.
Panel f3: peak |dB| across 20 manual trials (Fig4b_peaks.csv) with mean,
         +-1 s.d. band and the computed CV%.

All annotated numbers are computed here from the source data. Units: uT.

Outputs:
  /Users/arielzhang/Desktop/SensingMagic/paper/figures/fig2f_response_event.pdf
  /Users/arielzhang/Desktop/SensingMagic/paper/figures/fig2f_response_event.png
"""

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# ----------------------------------------------------------------------------
# Style (paper-wide)
# ----------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 7.5, "axes.labelsize": 8, "axes.titlesize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.8,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8, "ytick.major.width": 0.8,
    "xtick.major.size": 2.8, "ytick.major.size": 2.8,
    "xtick.direction": "out", "ytick.direction": "out",
    "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42, "figure.dpi": 300,
})
# Okabe-Ito palette
BLUE, VERM, BLACK, GREY = "#0072B2", "#D55E00", "#000000", "#888888"
ANN = "#555555"        # annotation text
REF = "#AAAAAA"        # dashed reference lines

RAW_HIGHRATE = "/Users/arielzhang/Desktop/磁毛传感器可靠性分析原始数据/高采样率_z轴.csv"
OVERLAY_CSV = ("/Users/arielzhang/Desktop/磁毛传感器可靠性分析原始数据/"
               "处理结果与图表_20260717/figures/data/Fig4a_overlay.csv")
PEAKS_CSV = ("/Users/arielzhang/Desktop/磁毛传感器可靠性分析原始数据/"
             "处理结果与图表_20260717/figures/data/Fig4b_peaks.csv")
OUT_PDF = "/Users/arielzhang/Desktop/SensingMagic/paper/figures/fig2f_response_event.pdf"
OUT_PNG = "/Users/arielzhang/Desktop/SensingMagic/paper/figures/fig2f_response_event.png"

# ----------------------------------------------------------------------------
# Panel a — load raw high-rate file, drop bad packets, find a clean tap event
# ----------------------------------------------------------------------------
df = pd.read_csv(RAW_HIGHRATE, header=None, names=["t_ms", "Bx", "By", "Bz"])

# Bad packets: runs where all axes stick to fixed clipped levels. Drop any
# row where an axis deviates > 8 x global robust MAD from a centred rolling
# median (window 2001 samples ~ 2 s, longer than the worst bad run). No
# interpolation - the rows are simply removed.
keep = np.ones(len(df), dtype=bool)
for col in ("Bx", "By", "Bz"):
    v = df[col]
    roll_med = v.rolling(2001, center=True, min_periods=100).median()
    mad = np.median(np.abs(v - v.median()))          # global MAD
    keep &= ((v - roll_med).abs() <= 8 * 1.4826 * mad).values
n_dropped = int((~keep).sum())
d = df[keep].reset_index(drop=True)

t_s = (d["t_ms"].values - d["t_ms"].values[0]) / 1000.0   # s
bz = d["Bz"].values                                        # uT

# Detect tap events on the slowly-detrended -dBz (taps are dips in Bz).
trend = pd.Series(bz).rolling(3001, center=True, min_periods=100).median().values
depth = -(bz - trend)
res = bz - trend
noise_sigma = 1.4826 * np.median(np.abs(res - np.median(res)))  # robust noise
peaks, props = find_peaks(depth, prominence=6 * noise_sigma, distance=300)

# Choose a clean, well-isolated event:
#  - sample times continuous (no dropped packets) within +-0.5 s
#  - previous peak >= 0.45 s away, next peak >= 0.40 s away
#  - amplitude closest to the median tap amplitude (representative)
amps = depth[peaks]
med_amp = np.median(amps)
candidates = []
for k, i in enumerate(peaks):
    lo, hi = np.searchsorted(t_s, t_s[i] - 0.5), np.searchsorted(t_s, t_s[i] + 0.5)
    if lo == 0 or hi >= len(t_s):
        continue
    if np.max(np.diff(t_s[lo:hi])) > 0.003:      # gap => dropped packets nearby
        continue
    gap_prev = t_s[i] - t_s[peaks[k - 1]] if k > 0 else np.inf
    gap_next = t_s[peaks[k + 1]] - t_s[i] if k < len(peaks) - 1 else np.inf
    if gap_prev < 0.45 or gap_next < 0.40:
        continue
    candidates.append((abs(amps[k] - med_amp), i))
assert candidates, "no clean isolated tap event found"
sel = min(candidates)[1]                          # representative clean event

# Local pre-event baseline (median of a quiet window before the tap).
tc = t_s[sel]
base_mask = (t_s >= tc - 0.30) & (t_s <= tc - 0.12)
baseline = np.median(bz[base_mask])
win = (t_s >= tc - 0.20) & (t_s <= tc + 0.20)
te = (t_s[win] - tc) * 1000.0                     # ms, 0 at peak
dbz = bz[win] - baseline                          # signed dBz (uT)

# 7-sample centred moving average (~7 ms << rise time) for level crossings.
s = -pd.Series(dbz).rolling(7, center=True, min_periods=1).mean().values
ipk = np.argmax(s[(te > -60) & (te < 60)]) + np.searchsorted(te, -60)
A = s[ipk]                                        # event amplitude (uT)


def cross_time(sig, tt, start, level, direction):
    """Time where sig crosses `level`, walking from `start` (interp)."""
    i = start
    while 0 < i < len(sig) - 1:
        j = i + direction
        if (sig[i] - level) * (sig[j] - level) <= 0:
            if tt[j] == tt[i] or sig[j] == sig[i]:
                return tt[j]
            return tt[i] + (level - sig[i]) * (tt[j] - tt[i]) / (sig[j] - sig[i])
        i = j
    return tt[i]


t90r = cross_time(s, te, ipk, 0.9 * A, -1)        # rising edge 90 %
t10r = cross_time(s, te, ipk, 0.1 * A, -1)        # rising edge 10 %
t90f = cross_time(s, te, ipk, 0.9 * A, +1)        # falling edge 90 %
t10f = cross_time(s, te, ipk, 0.1 * A, +1)        # falling edge 10 %
rise_ms = t90r - t10r
recov_ms = t10f - t90f

# ----------------------------------------------------------------------------
# Panels b, c — repeatability data
# ----------------------------------------------------------------------------
ov = pd.read_csv(OVERLAY_CSV)
cyc_cols = [c for c in ov.columns if c.startswith("dBz_cycle")]
ov_mean = ov[cyc_cols].mean(axis=1)

pk = pd.read_csv(PEAKS_CSV)
pk_abs = np.abs(pk["dB_peak"].values)
pk_mean, pk_sd = pk_abs.mean(), pk_abs.std(ddof=1)
cv_pct = 100.0 * pk_sd / pk_mean

# ----------------------------------------------------------------------------
# Figure
# ----------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.35), constrained_layout=True,
                         gridspec_kw={"width_ratios": [1.15, 1.0, 1.0]})

# --- panel f1 ----------------------------------------------------------------
ax = axes[0]
ax.plot(te, np.abs(dbz), color=BLACK, alpha=0.30, lw=0.7, label="raw")
ax.plot(te, np.abs(s), color=BLUE, lw=1.5, label="7-ms avg.")

# 10 % / 90 % guide lines (dashed, subtle) + level tag in the empty right area
for lv in (0.1 * A, 0.9 * A):
    ax.axhline(lv, color=REF, lw=0.6, ls=(0, (3, 2)), zorder=0)
ax.text(146, 0.9 * A + 1.2, "90%", ha="right", va="bottom",
        fontsize=6.5, color=ANN)

# crossing markers (white-edged for separation from the trace)
ax.plot([t10r, t90r], [0.1 * A, 0.9 * A], "o", color=BLACK, ms=2.6,
        mec="white", mew=0.5, zorder=5)
ax.plot([t90f, t10f], [0.9 * A, 0.1 * A], "s", color=BLACK, ms=2.4,
        mec="white", mew=0.5, zorder=5)

# compact double-headed arrows above the trace, one per interval
ymax_win = np.abs(dbz[(te >= -150) & (te <= 150)]).max()
y_arr = ymax_win + 4.5
arrow_kw = dict(arrowstyle="<->", lw=0.6, color=ANN,
                shrinkA=0, shrinkB=0, mutation_scale=6)
ax.annotate("", xy=(t10r, y_arr), xytext=(t90r, y_arr),
            arrowprops=dict(**arrow_kw))
ax.text(0.5 * (t10r + t90r), y_arr + 1.8, f"rise 10–90%\n{rise_ms:.0f} ms",
        ha="center", va="bottom", fontsize=6.5, color=ANN, linespacing=1.2)
ax.annotate("", xy=(t90f, y_arr), xytext=(t10f, y_arr),
            arrowprops=dict(**arrow_kw))
ax.text(t10f + 7, y_arr, f"recovery\n{recov_ms:.0f} ms",
        ha="left", va="center", fontsize=6.5, color=ANN, linespacing=1.2)

ax.set_xlabel("Time (ms)")
ax.set_ylabel("|ΔB$_z$| (µT)")
ax.set_title("f1  Tap response event", loc="left")
ax.set_xlim(-150, 150)
ax.set_ylim(0, y_arr + 13.5)
ax.legend(frameon=False, loc="center right", bbox_to_anchor=(1.0, 0.30),
          handlelength=1.2, borderaxespad=0.2, labelspacing=0.3)

# --- panel f2 ----------------------------------------------------------------
ax = axes[1]
for j, c in enumerate(cyc_cols):
    ax.plot(ov["time_s"], ov[c], color=GREY, alpha=0.35, lw=0.7,
            label=f"cycles 1–{len(cyc_cols)}" if j == 0 else None)
ax.plot(ov["time_s"], ov_mean, color=VERM, lw=1.6, label="mean")
ax.set_xlabel("Time (s)")
ax.set_ylabel("ΔB$_z$ (µT)")
ax.set_title(f"f2  Cycle overlay (n = {len(cyc_cols)})", loc="left")
leg = ax.legend(frameon=False, loc="upper right", handlelength=1.4,
                borderaxespad=0.2, labelspacing=0.3)
leg.legend_handles[0].set_alpha(0.7)      # keep the legend key legible

# --- panel f3 ----------------------------------------------------------------
ax = axes[2]
ax.axhspan(pk_mean - pk_sd, pk_mean + pk_sd, color=BLUE, alpha=0.12,
           lw=0, zorder=0)
ax.axhline(pk_mean, color=VERM, lw=1.4, zorder=1)
ax.plot(pk["trial"], pk_abs, "o", color=BLUE, ms=3.4,
        mec="white", mew=0.5, zorder=3)
ax.text(0.03, 0.05,
        f"mean = {pk_mean:.0f} µT, s.d. = {pk_sd:.0f} µT\n"
        f"CV = {cv_pct:.1f}%",
        transform=ax.transAxes, ha="left", va="bottom",
        fontsize=6.5, color=ANN)
ax.set_xlabel("Trial")
ax.set_ylabel("|ΔB| peak (µT)")
ax.set_title(f"f3  Peak amplitude (n = {len(pk_abs)})", loc="left")
ax.set_xlim(0, 21)
ax.set_ylim(0, 1300)
ax.set_xticks([1, 5, 10, 15, 20])

fig.savefig(OUT_PDF)
fig.savefig(OUT_PNG)

print(f"dropped bad-packet rows : {n_dropped} / {len(df)}")
print(f"detected tap events     : {len(peaks)}")
print(f"selected event          : t = {tc:.2f} s, amplitude {A:.1f} uT")
print(f"rise time (10-90%)      : {rise_ms:.1f} ms")
print(f"recovery time (90-10%)  : {recov_ms:.1f} ms")
print(f"peaks: mean {pk_mean:.1f} uT, sd {pk_sd:.1f} uT, CV {cv_pct:.2f} %")
print(f"saved: {OUT_PDF}")
print(f"saved: {OUT_PNG}")

#!/usr/bin/env python
"""
fig2d_stress_strain_direction: tensile stress-strain curves of PDMS/magnetic
composite specimens grouped by magnetization direction.

Data source: pdms0131_1.xlsx (Zwick tensile export).
  - Sheet "测试结果": per-specimen summary (Et MPa, sm MPa, em %, b, h, A0 mm^2).
  - Sheets "试样 N": raw curves, columns [strain %, standard load N] after
    3 preamble rows (title, names, units).

Stress(MPa) = Load(N) / A0(mm^2), using each specimen's own A0 from 测试结果.

Break criterion: each curve is truncated at the last point of its running
stress maximum, so any post-peak drop (specimen failure beginning inside the
0-100 % window, e.g. specimen 6 breaking at ~98.6 %) is removed and every
curve ends cleanly at its last pre-break point -- no end hooks or stub tails.

Groups by specimen number:
  1-6  Non-magnetized, 7-12 x-magnetized, 13-18 y-magnetized, 19-24 z-magnetized.
Only sheets actually present in the workbook are used.

Group mean: specimens are interpolated onto a common 0-100 % strain grid
(0.2 % step); the mean is drawn only over strains where ALL specimens of the
group still have data (i.e. up to min over specimens of min(failure strain,
100 %)), so no specimen is extrapolated past its own failure. If a specimen
actually fails inside the 0-100 % window (raw data ends before 100 %, e.g.
specimen 6 at ~98.6 %), the group mean additionally stops FAIL_MARGIN before
that failure: the failing specimen's last ~1 % of strain shows pre-break
softening that would bias the mean end downward and leave a stub at the axis.

Legend Et values are the mean +/- s.d. of the Et column of 测试结果 for each
group (sheet values used as-is; Et is the machine-computed linear slope over
the 20-40 % strain window, strain dimensionless). Nothing is hardcoded.
"""

import os
import re

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

XLSX = ("/private/tmp/claude-502/-Users-arielzhang-Desktop-SensingMagic/"
        "617d7206-542d-4a74-bd36-6e9400bc4ca0/scratchpad/tensile/349959/"
        "349959/pdms0131_1.xlsx")
FIGDIR = "/Users/arielzhang/Desktop/SensingMagic/paper/figures"
BASE = "fig2d_stress_strain_direction" + ("" if __import__("os").environ.get("SS_FONT", "serif") == "serif" else "_sans")

STRAIN_MAX = 100.0          # % ; truncate curves here, no failure tail
GRID = np.linspace(0.0, STRAIN_MAX, 501)  # common strain grid, 0.2 % step
SMOOTH_WIN = 9              # centered moving average, 9 pts = 1.8 % strain
FAIL_MARGIN = 1.0           # % strain; mean stops this far before an
                            # in-window specimen failure (pre-break softening)

GROUPS = [
    # (label stem, specimen number range, color) -- Okabe-Ito palette
    ("Non-mag.", range(1, 7),   "#000000"),   # black
    ("x-mag.",   range(7, 13),  "#0072B2"),   # blue
    ("y-mag.",   range(13, 19), "#D55E00"),   # vermillion
    ("z-mag.",   range(19, 25), "#009E73"),   # green
]


def setup_style():
    """Sans-serif (DejaVu Sans), matching the rcParams family used by the
    e/f source figures so the Fig. 2 composite has one typeface throughout.
    Boxed axes with inward ticks are kept (layout convention, not a font)."""
    plt.rcParams.update({
        # SS_FONT=serif (default, user spec: Times/STIX) or sans (composite)
        **({"font.family": "serif",
            "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
            "mathtext.fontset": "stix"}
           if __import__("os").environ.get("SS_FONT", "serif") == "serif"
           else {"font.family": "sans-serif",
                 "font.sans-serif": ["DejaVu Sans"],
                 "mathtext.fontset": "dejavusans"}),
        "font.size": 8.5,
        "axes.labelsize": 9.5,
        "axes.linewidth": 0.8,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.grid": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def pad_stems(stems, fontsize):
    """Pad the group-name stems with spaces so the $E_t$ values in the legend
    start at a common x -- the four entries read as a neat table. Widths are
    measured with the actually-resolved font (TextPath), and the space width
    by ink difference (trailing spaces carry no ink of their own)."""
    from matplotlib.font_manager import FontProperties
    from matplotlib.textpath import TextPath
    fp = FontProperties(family=plt.rcParams["font.family"], size=fontsize)
    w = lambda t: TextPath((0, 0), t, prop=fp).get_extents().width
    space = w("i i") - w("ii")
    target = max(w(s) for s in stems)
    return [s + " " * (2 + max(0, int(round((target - w(s)) / space))))
            for s in stems]


def load_summary():
    """Return {spec_no: {"Et": float, "A0": float}} from 测试结果."""
    df = pd.read_excel(XLSX, sheet_name="测试结果", header=None)
    out = {}
    for _, row in df.iterrows():
        m = re.fullmatch(r"试样\s*(\d+)", str(row[0]).strip())
        if not m:
            continue
        out[int(m.group(1))] = {"Et": float(row[1]), "A0": float(row[8])}
    return out


def load_curve(xl, spec_no, a0):
    """Return (strain %, stress MPa, fails_inside), cleaned, monotonic,
    truncated at 100 % and at the specimen's break point (running-max
    criterion). fails_inside is True when the specimen's raw record ends
    before 100 % strain, i.e. it actually broke inside the plotted window."""
    df = pd.read_excel(xl, sheet_name=f"试样 {spec_no}", header=None)
    strain = pd.to_numeric(df[0], errors="coerce")
    load = pd.to_numeric(df[1], errors="coerce")
    ok = strain.notna() & load.notna()
    s = strain[ok].to_numpy(float)
    f = load[ok].to_numpy(float)
    # keep only points on the running strain maximum -> strictly increasing x
    cummax = np.maximum.accumulate(s)
    keep = s >= cummax
    s, f = s[keep], f[keep]
    inc = np.r_[True, np.diff(s) > 0]
    s, f = s[inc], f[inc]
    s = np.clip(s, 0.0, None)
    stress = f / a0
    fails_inside = s[-1] < STRAIN_MAX
    # truncate at 100 % strain (drop failure tail)
    m = s <= STRAIN_MAX
    s, stress = s[m], stress[m]
    # break criterion: end at the last point of the running stress maximum,
    # dropping any post-peak drop (start of specimen failure) -> no end hooks
    i_end = stress.size - 1 - int(np.argmax(stress[::-1]))
    return s[:i_end + 1], stress[:i_end + 1], fails_inside


def centered_smooth(v, win):
    """Centered moving average; the window shrinks symmetrically at both
    edges (no padding, no phase shift), so curve ends stay anchored to the
    data with no smoothing artifacts."""
    n = v.size
    h = win // 2
    c = np.r_[0.0, np.cumsum(v)]
    out = np.empty(n)
    for i in range(n):
        k = min(h, i, n - 1 - i)
        out[i] = (c[i + k + 1] - c[i - k]) / (2 * k + 1)
    return out


def resample_smooth(s, y):
    """Resample onto GRID (bin means, anti-aliased) + centered smoothing.

    Raw curves carry ~6000 points; averaging the samples that fall into each
    0.2 % grid bin (instead of picking single interpolated samples) removes
    sensor jitter, then a centered ~1.8 % strain window (edge-shrunk) gives
    visual continuity. Symmetric smoothing preserves the near-linear 20-40 %
    region, so curve shapes there are unchanged.
    Returns a full-length array aligned to GRID, NaN past specimen break.
    """
    out = np.full(GRID.shape, np.nan)
    m = GRID <= s[-1] + 1e-9
    gx = GRID[m]
    step = GRID[1] - GRID[0]
    idx = np.clip(np.round(s / step).astype(int), 0, GRID.size - 1)
    sums = np.bincount(idx, weights=y, minlength=GRID.size)
    cnts = np.bincount(idx, minlength=GRID.size)
    binned = np.where(cnts > 0, sums / np.maximum(cnts, 1), np.nan)
    v = binned[:gx.size].copy()
    gaps = np.isnan(v)
    if gaps.any():
        v[gaps] = np.interp(gx[gaps], s, y)
    if v.size > SMOOTH_WIN:
        v = centered_smooth(v, SMOOTH_WIN)
    out[m] = v
    return out


def main():
    setup_style()
    os.makedirs(FIGDIR, exist_ok=True)
    summary = load_summary()
    xl = pd.ExcelFile(XLSX)
    present = {int(m.group(1)) for name in xl.sheet_names
               if (m := re.fullmatch(r"试样\s*(\d+)", name))}

    fig, ax = plt.subplots(figsize=(3.6, 3.0), dpi=300)

    handles, labels = [], []
    report = []
    for stem, rng, color in GROUPS:
        specs = sorted(n for n in rng if n in present and n in summary)
        if not specs:
            continue
        curves = []
        mean_stop = STRAIN_MAX
        for n in specs:
            s, y, fails_inside = load_curve(xl, n, summary[n]["A0"])
            g = resample_smooth(s, y)
            curves.append(g)
            if fails_inside:  # keep mean clear of pre-break softening
                mean_stop = min(mean_stop, s[-1] - FAIL_MARGIN)
            ax.plot(GRID, g, color=color, lw=0.7, alpha=0.30,
                    solid_capstyle="round", zorder=2)
        arr = np.vstack(curves)
        all_ok = ~np.isnan(arr).any(axis=0)   # strains where every specimen has data
        all_ok &= GRID <= mean_stop
        mean = np.nanmean(arr[:, all_ok], axis=0)
        ax.plot(GRID[all_ok], mean, color=color, lw=1.8, zorder=3,
                solid_capstyle="round")

        et = np.array([summary[n]["Et"] for n in specs])
        handles.append(Line2D([], [], color=color, lw=1.8))
        labels.append((stem, et.mean(), et.std(ddof=1)))
        report.append((stem, specs, et.mean(), et.std(ddof=1),
                       GRID[all_ok][-1]))

    # legend as a neat table: stems padded to equal width so Et values align
    LEG_FS = 7.6
    padded = pad_stems([s for s, _, _ in labels], LEG_FS)
    labels = [f"{p}$E_t$ = {m:.2f} ± {s:.2f} MPa"
              for p, (_, m, s) in zip(padded, labels)]

    ax.set_xlim(0, STRAIN_MAX)
    data_top = max(np.nanmax(l.get_ydata()) for l in ax.get_lines())
    ymax = np.ceil((data_top * 1.04) / 0.5) * 0.5   # round up to 0.5 MPa
    ax.set_ylim(0, ymax)
    ax.xaxis.set_major_locator(plt.MultipleLocator(20))
    ax.xaxis.set_minor_locator(plt.MultipleLocator(10))
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.5))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(0.25))
    ax.set_xlabel("Strain (%)")
    ax.set_ylabel("Stress (MPa)")
    ax.tick_params(direction="in", top=True, right=True, length=3.2, width=0.8)
    ax.tick_params(which="minor", direction="in", top=True, right=True,
                   length=1.8, width=0.6)
    for sp in ax.spines.values():
        sp.set_visible(True)
    ax.legend(handles, labels, loc="lower right", frameon=False,
              fontsize=LEG_FS, handlelength=1.6, handletextpad=0.6,
              borderaxespad=0.6, labelspacing=0.5)

    fig.tight_layout(pad=0.4)
    for ext in ("svg", "pdf", "png"):
        fig.savefig(os.path.join(FIGDIR, f"{BASE}.{ext}"),
                    dpi=300, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)

    for stem, specs, m, sd, cut in report:
        print(f"{stem:10s} specimens={specs}  Et = {m:.3f} +/- {sd:.3f} MPa"
              f"  mean-curve cutoff = {cut:.1f} %")


if __name__ == "__main__":
    main()

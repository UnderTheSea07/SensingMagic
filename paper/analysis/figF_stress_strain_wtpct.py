#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
figF_stress_strain_wtpct.py

Companion composition-series stress-strain figure:
Ecoflex 00-30 + NdFeB magnetic powder dogbone tension, 50/60/70/80 wt% NdFeB.

Inputs (Zwick exports):
  tensile/20241121/20241121/{50_1,60_1,70_2,80_5}.xlsx
  - sheet "测试结果": per-specimen Et / sm / em / A0 (two header rows, col0 "试样 N")
  - specimen sheets "试样 N": columns [应变 %, 标准载荷 <unit>]
    NOTE: in these exports the "load" column unit row reads "kPa", i.e. the
    machine already divided by A0 -> the column IS stress. The unit row is read
    at runtime; if a sheet were exported in N the script divides by that
    specimen's A0 instead. Nothing is hardcoded.

Outputs:
  paper/figures/fig2d_stress_strain_wtpct.{svg,pdf,png}

Style: sans-serif (DejaVu Sans, matching the other Fig. 2 panels), white
background, no grid, four-sided box, inward ticks on all sides, light raw
per-specimen curves + bold group mean, frameless legend inside upper left
(empty region) with Et = mean ± s.d. computed from the summary sheet at
runtime.

Each specimen's curve is truncated at its break point (global maximum of the
lightly smoothed stress): the Zwick export keeps a short post-break drop tail
(up to ~50% strain of falling stress) which is NOT drawn. Visual smoothing is
a centered moving average whose window shrinks at the ends (normalized 'same'
convolution), so the curve ends exactly at the break point with no edge hooks.
The group mean is drawn only up to the group's earliest break.
"""

import os
import numpy as np
import openpyxl
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

DATA_DIR = ("/private/tmp/claude-502/-Users-arielzhang-Desktop-SensingMagic/"
            "617d7206-542d-4a74-bd36-6e9400bc4ca0/scratchpad/tensile/20241121/20241121")
FIG_DIR = "/Users/arielzhang/Desktop/SensingMagic/paper/figures"
BASENAME = "fig2d_stress_strain_wtpct" + ("" if __import__("os").environ.get("SS_FONT", "serif") == "serif" else "_sans")

# colorblind-safe ordered ramp (Okabe-Ito hues, cool -> warm with filler
# loading: sky, blue, orange, vermillion)
GROUPS = [  # (file, wt%, color)
    ("50_1.xlsx", 50, "#56B4E9"),
    ("60_1.xlsx", 60, "#0072B2"),
    ("70_2.xlsx", 70, "#E69F00"),
    ("80_5.xlsx", 80, "#D55E00"),
]

SUMMARY_SHEET = "测试结果"
N_GRID = 900          # resample points per curve (visual only)
SMOOTH_WIN = 41       # mild moving-average window on the raw trace (visual only)
DETECT_WIN = 11       # light smoothing used only to locate the break point
MIN_BREAK_STRAIN = 50.0   # ignore the initial toe/yield hump when locating it


# ----------------------------------------------------------------------------- fonts
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
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.grid": False,
        "axes.linewidth": 0.9,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    })


# ----------------------------------------------------------------------------- io
def read_summary(wb):
    """Parse the 测试结果 sheet -> {specimen_name: {Et, sm, em, A0, ...}}."""
    ws = wb[SUMMARY_SHEET]
    rows = [r for r in ws.iter_rows(values_only=True)]
    header = rows[0]  # (None, 'Et', 'sm', 'em', 'sb', 'eb', 'b', 'h', 'A0')
    col = {str(h): i for i, h in enumerate(header) if h is not None}
    out = {}
    for r in rows[2:]:
        if r is None or r[0] is None:
            continue
        name = str(r[0]).strip()
        if not name.startswith("试样"):
            continue
        out[name] = {k: float(r[i]) for k, i in col.items()
                     if r[i] is not None}
    return out


def read_specimen(ws, A0):
    """Read one 试样 sheet -> (strain %, stress MPa). Unit row decides scaling."""
    it = ws.iter_rows(values_only=True)
    next(it)                      # row 0: ('试样 N', '试样 N')
    next(it)                      # row 1: ('应变', '标准载荷')
    units = next(it)              # row 2: ('%', 'kPa' | 'N' | 'MPa')
    u = str(units[1]).strip().lower()
    strain, load = [], []
    for r in it:
        if r[0] is None or r[1] is None:
            continue
        strain.append(float(r[0]))
        load.append(float(r[1]))
    strain = np.asarray(strain)
    load = np.asarray(load)
    if u == "kpa":                # machine already normalized by A0
        stress = load / 1000.0
    elif u == "mpa":
        stress = load
    elif u == "n":                # true load -> engineering stress
        stress = load / A0
    else:
        raise ValueError(f"unknown load unit {u!r}")
    keep = strain >= 0.0
    return strain[keep], stress[keep]


def centered_smooth(y, win):
    """Centered moving average whose window shrinks symmetrically toward the
    ends (normalized 'same' convolution) -> no edge bias, no end hooks."""
    if win <= 1 or len(y) <= 3 * win:
        return y
    k = np.ones(win)
    return np.convolve(y, k, mode="same") / np.convolve(np.ones(len(y)), k,
                                                        mode="same")


def truncate_at_break(strain, stress):
    """Cut the curve at the specimen's break point and drop everything after.

    The break point is the global maximum of the lightly smoothed stress past
    MIN_BREAK_STRAIN. For a terminal fracture drop this is exactly the last
    point before stress falls >3% below its running maximum, but unlike the
    running-max test it is immune to the small yield hump these composites
    show near ~50-90% strain (stress dips >3% there, then rises much higher).
    """
    det = centered_smooth(stress, DETECT_WIN)
    if strain[-1] > MIN_BREAK_STRAIN:
        det = np.where(strain < MIN_BREAK_STRAIN, -np.inf, det)
    ib = int(np.argmax(det))
    return strain[:ib + 1], stress[:ib + 1]


def resample(strain, stress, n=N_GRID, win=SMOOTH_WIN):
    """Smooth (edge-aware, centered) + downsample onto a uniform strain grid
    that ends exactly at the break strain (visual only). Called on already
    truncated data, so no post-break samples can leak into the smoothing.

    After smoothing, the curve is re-cut at the maximum of the SMOOTHED
    stress: the detection window (DETECT_WIN) and the visual window (win)
    peak a few samples apart, so without this the trailing average could dip
    slightly and leave a micro-hook. Cutting at the smoothed maximum makes
    'last point = highest point' true by construction -> no end hooks."""
    stress = centered_smooth(stress, win)
    order = np.argsort(strain, kind="stable")
    s_sorted, y_sorted = strain[order], stress[order]
    if s_sorted[-1] > MIN_BREAK_STRAIN:
        masked = np.where(s_sorted < MIN_BREAK_STRAIN, -np.inf, y_sorted)
        im = int(np.argmax(masked))
        s_sorted, y_sorted = s_sorted[:im + 1], y_sorted[:im + 1]
    grid = np.linspace(0.0, s_sorted[-1], n)
    return grid, np.interp(grid, s_sorted, y_sorted)


# ----------------------------------------------------------------------------- main
def main():
    setup_style()
    os.makedirs(FIG_DIR, exist_ok=True)

    # 3.9 x 2.7 in: in the Fig. 2 composite this panel is placed ~2.75 in
    # wide (row height fixed by the SEM group), so a smaller canvas keeps the
    # downscale near 0.7x and the tick/legend text print-legible (~6-8 pt
    # effective instead of ~4-5 pt at the former 4.6 x 3.6 in canvas).
    fig, ax = plt.subplots(figsize=(3.9, 2.7), dpi=300)

    handles, labels = [], []
    print("group summary (computed at runtime):")
    for fname, wt, color in GROUPS:
        wb = openpyxl.load_workbook(os.path.join(DATA_DIR, fname),
                                    read_only=True, data_only=True)
        summary = read_summary(wb)
        spec_sheets = [s for s in wb.sheetnames if s.strip().startswith("试样")]

        curves = []
        for sheet in spec_sheets:
            A0 = summary.get(sheet.strip(), {}).get("A0", np.nan)
            s, y = read_specimen(wb[sheet], A0)
            s, y = truncate_at_break(s, y)   # nothing after break is kept
            curves.append(resample(s, y))
        wb.close()

        # light per-specimen raw curves, each ending at its break point
        for g, y in curves:
            ax.plot(g, y, color=color, lw=0.7, alpha=0.3, zorder=2)

        # bold group mean, only where ALL specimens are still intact
        cutoff = min(g[-1] for g, _ in curves)
        grid = np.linspace(0.0, cutoff, N_GRID)
        mean = np.mean([np.interp(grid, g, y) for g, y in curves], axis=0)
        ax.plot(grid, mean, color=color, lw=1.8, zorder=3)

        et = np.array([v["Et"] for v in summary.values()])
        et_m, et_s = et.mean(), et.std(ddof=1)
        breaks = ", ".join(f"{g[-1]:.0f}" for g, _ in curves)
        print(f"  {wt} wt%: n={len(curves)}, Et = {et_m:.3f} ± {et_s:.3f} MPa, "
              f"breaks at [{breaks}]% strain, mean cut at {cutoff:.0f}%")

        handles.append(Line2D([0], [0], color=color, lw=1.8))
        labels.append(f"{wt} wt%  $E_t$ = {et_m:.3f} $\\pm$ {et_s:.3f} MPa")

    ax.set_xlabel("Strain (%)", fontsize=11)
    ax.set_ylabel("Stress (MPa)", fontsize=11)
    ax.set_xlim(0, 750)   # last break is at 713% strain (763% was drop tail)
    ax.set_ylim(0, 1.25)
    # 6 majors on each axis, ending exactly at the box corners (0/750, 0/1.25)
    ax.xaxis.set_major_locator(plt.MultipleLocator(150))
    ax.xaxis.set_minor_locator(plt.MultipleLocator(50))
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.25))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(0.125))
    ax.tick_params(direction="in", top=True, right=True, labelsize=9.5,
                   length=3.5, width=0.8)
    ax.tick_params(which="minor", direction="in", top=True, right=True,
                   length=1.9, width=0.6)
    for sp in ax.spines.values():
        sp.set_visible(True)

    # Frameless legend in the empty upper-left region (all curves rise to the
    # right; the 80 wt% raw curves peak at ~0.81 MPa, well below the legend).
    # Lower right would sit on top of the 60/70 wt% raw curves.
    ax.legend(handles, labels, loc="upper left", fontsize=8.3,
              frameon=False, borderaxespad=0.6,
              handlelength=1.6, handletextpad=0.6, labelspacing=0.5)

    fig.tight_layout(pad=0.4)
    for ext in ("svg", "pdf", "png"):
        fig.savefig(os.path.join(FIG_DIR, f"{BASENAME}.{ext}"),
                    dpi=300, facecolor="white")
    plt.close(fig)
    print("saved:", os.path.join(FIG_DIR, BASENAME) + ".{svg,pdf,png}")


if __name__ == "__main__":
    main()

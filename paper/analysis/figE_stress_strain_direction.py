#!/usr/bin/env python
"""
fig2d_stress_strain_direction: tensile stress-strain curves of PDMS/magnetic
composite specimens grouped by magnetization direction.

Data source: pdms0131_1.xlsx (Zwick tensile export).
  - Sheet "测试结果": per-specimen summary (Et MPa, sm MPa, em %, b, h, A0 mm^2).
  - Sheets "试样 N": raw curves, columns [strain %, standard load N] after
    3 preamble rows (title, names, units).

Stress(MPa) = Load(N) / A0(mm^2), using each specimen's own A0 from 测试结果.

Groups by specimen number:
  1-6  Non-magnetized, 7-12 x-magnetized, 13-18 y-magnetized, 19-24 z-magnetized.
Only sheets actually present in the workbook are used.

Group mean: specimens are interpolated onto a common 0-100 % strain grid
(0.2 % step); the mean is drawn only over strains where ALL specimens of the
group still have data (i.e. up to min over specimens of min(failure strain,
100 %)), so no specimen is extrapolated past its own failure.

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
from matplotlib import font_manager
from matplotlib.lines import Line2D

XLSX = ("/private/tmp/claude-502/-Users-arielzhang-Desktop-SensingMagic/"
        "617d7206-542d-4a74-bd36-6e9400bc4ca0/scratchpad/tensile/349959/"
        "349959/pdms0131_1.xlsx")
FIGDIR = "/Users/arielzhang/Desktop/SensingMagic/paper/figures"
BASE = "fig2d_stress_strain_direction"

STRAIN_MAX = 100.0          # % ; truncate curves here, no failure tail
GRID = np.linspace(0.0, STRAIN_MAX, 501)  # common strain grid, 0.2 % step
SMOOTH_WIN = 5              # light moving average on resampled curve (1 % strain)

GROUPS = [
    # (label stem, specimen number range, color)
    ("Non-mag.", range(1, 7),   "#1a1a1a"),
    ("x-mag.",   range(7, 13),  "#c62828"),
    ("y-mag.",   range(13, 19), "#1565c0"),
    ("z-mag.",   range(19, 25), "#2e7d32"),
]


def setup_style():
    names = {f.name for f in font_manager.fontManager.ttflist}
    serif = ["Times New Roman"] if "Times New Roman" in names else ["STIXGeneral"]
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": serif + ["STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
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
    """Return (strain %, stress MPa), cleaned, monotonic, truncated at 100 %."""
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
    # truncate at 100 % strain (drop failure tail)
    m = s <= STRAIN_MAX
    return s[m], stress[m]


def resample_smooth(s, y):
    """Resample onto GRID (within specimen range) + light moving average.

    Smoothing window ~1 % strain: purely visual, does not change trends.
    Returns full-length arrays aligned to GRID with NaN past specimen failure.
    """
    out = np.full(GRID.shape, np.nan)
    m = GRID <= s[-1]
    out[m] = np.interp(GRID[m], s, y)
    v = out[m]
    if v.size > SMOOTH_WIN:
        k = np.ones(SMOOTH_WIN) / SMOOTH_WIN
        sm = np.convolve(v, k, mode="same")
        # fix edge bias of 'same' convolution
        h = SMOOTH_WIN // 2
        sm[:h], sm[-h:] = v[:h], v[-h:]
        out[m] = sm
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
        for n in specs:
            s, y = load_curve(xl, n, summary[n]["A0"])
            g = resample_smooth(s, y)
            curves.append(g)
            ax.plot(GRID, g, color=color, lw=0.6, alpha=0.35,
                    solid_capstyle="round", zorder=2)
        arr = np.vstack(curves)
        all_ok = ~np.isnan(arr).any(axis=0)   # strains where every specimen has data
        mean = np.nanmean(arr[:, all_ok], axis=0)
        ax.plot(GRID[all_ok], mean, color=color, lw=1.8, zorder=3)

        et = np.array([summary[n]["Et"] for n in specs])
        handles.append(Line2D([], [], color=color, lw=1.8))
        labels.append(f"{stem}  $E_t$ = {et.mean():.2f} ± {et.std(ddof=1):.2f} MPa")
        report.append((stem, specs, et.mean(), et.std(ddof=1),
                       GRID[all_ok][-1]))

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
              fontsize=7.6, handlelength=1.5, borderaxespad=0.6,
              labelspacing=0.45)

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

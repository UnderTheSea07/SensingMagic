#!/usr/bin/env python
"""fig3c_event_alignment: force-magnetic correspondence of a magnetic-cilia patch
under manual normal loading (single MLX90393 + ATI F/T reference).

Panel a: single load-hold-unload event (Fig2_single_event.csv)
Panel b: five consecutive load-unload cycles (Fig3_five_cycles.csv)

Sign convention (stated on the axes, nothing silently flipped):
  Fz from the ATI sensor is negative in compression, so the left axis plots
  -Fz labelled "Compressive force -Fz (mN)".
  dBz decreases on loading, so the right axis plots -dBz labelled
  "-[Delta]Bz (uT)"; both traces therefore read upward during loading.

All annotated numbers are computed from the source CSVs at run time.

Style: Okabe-Ito palette (black force / #D55E00 magnetic), twin axes with
matching tick/label colors, grouped stat blocks in #555, light phase shading
(loading / hold / unloading) in panel a.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np
import pandas as pd

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 7.5,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 6.8,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.major.size": 2.8,
    "ytick.major.size": 2.8,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "pdf.fonttype": 42,
    "figure.dpi": 300,
})

BLACK = "#000000"
VERMILLION = "#D55E00"
ANNOT = "#555555"
PHASE_GREY = "#888888"

SINGLE = ("/Users/arielzhang/Desktop/磁毛传感器可靠性分析原始数据/"
          "处理结果与图表_20260717/figures/data/Fig2_single_event.csv")
CYCLES = ("/Users/arielzhang/Desktop/磁毛传感器可靠性分析原始数据/"
          "处理结果与图表_20260717/figures/data/Fig3_five_cycles.csv")
OUTDIR = "/Users/arielzhang/Desktop/SensingMagic/paper/figures"


def plot_pair(ax, df):
    """Twin-y plot of -Fz (mN, black, left) and -dBz (uT, vermillion, right)."""
    t = df["time_s"].to_numpy()
    f_mN = -df["Fz_N"].to_numpy() * 1e3          # compressive force, mN
    b_uT = -df["dBz_uT"].to_numpy()              # -dBz, uT

    axr = ax.twinx()
    lf, = ax.plot(t, f_mN, color=BLACK, lw=1.1, zorder=3)
    lb, = axr.plot(t, b_uT, color=VERMILLION, lw=1.1, zorder=2)

    # headroom above the traces so legends/annotations never overlap data
    for a, y in ((ax, f_mN), (axr, b_uT)):
        span = y.max() - y.min()
        a.set_ylim(y.min() - 0.05 * span, y.max() + 0.30 * span)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel(r"Compressive force $-F_z$ (mN)", color=BLACK)
    axr.set_ylabel(r"$-\Delta B_z$ ($\mu$T)", color=VERMILLION)

    # matching tick/label colors on each y-axis
    ax.tick_params(axis="y", colors=BLACK)
    axr.tick_params(axis="y", colors=VERMILLION)
    axr.spines["right"].set_color(VERMILLION)

    ax.spines[["top", "right"]].set_visible(False)
    axr.spines[["top", "left"]].set_visible(False)

    r = np.corrcoef(f_mN, b_uT)[0, 1]
    return lf, lb, axr, t, f_mN, b_uT, r


def shade_phases(ax, t, f_mN):
    """Light loading / hold / unloading phase shading for a single press.

    Phase boundaries are detected from the force trace at run time:
    event onset/offset at the first/last 5 %-of-peak crossing, hold between
    the first and last 50 %-of-peak crossings (robust to the brief manual
    overshoot).  The ramps are fast (~1 s), so the hold band carries the
    grey alpha-0.05 shading and all three phases are labelled at the top.
    """
    peak = f_mN.max()
    above_on = np.where(f_mN > 0.05 * peak)[0]
    above_hold = np.where(f_mN > 0.50 * peak)[0]
    i_on, i_off = above_on[0], above_on[-1]
    i_h0, i_h1 = above_hold[0], above_hold[-1]

    ax.axvspan(t[i_h0], t[i_h1], facecolor="grey", alpha=0.05,
               edgecolor="none", zorder=0)

    bands = [
        (t[i_on], t[i_h0], "loading"),
        (t[i_h0], t[i_h1], "hold"),
        (t[i_h1], t[i_off], "unloading"),
    ]
    trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    for x0, x1, name in bands:
        ax.text(0.5 * (x0 + x1), 0.995, name, transform=trans,
                ha="center", va="top", fontsize=6.5, color=PHASE_GREY,
                zorder=1)


def main():
    df_a = pd.read_csv(SINGLE)
    df_b = pd.read_csv(CYCLES)

    fig, (axa, axb) = plt.subplots(
        1, 2, figsize=(7.1, 2.3), gridspec_kw={"width_ratios": [1.0, 1.9]})

    # --- panel a: single load-hold-unload event -------------------------
    lf, lb, axar, t_a, f_a, b_a, r_a = plot_pair(axa, df_a)
    shade_phases(axa, t_a, f_a)
    axa.set_title("a  Single load–hold–unload event", loc="left")
    # x=0.48 centers the block in the gap between the loading ramp
    # (ends ~x=0.15 in axes fraction) and the unloading edge (~x=0.82),
    # so no line of text touches either transient.
    axa.text(0.48, 0.05,
             (f"peak $-F_z$ = {f_a.max():.0f} mN\n"
              f"peak $-\\Delta B_z$ = {b_a.max():.0f} $\\mu$T\n"
              f"Pearson $r$ = {r_a:.2f}"),
             transform=axa.transAxes, ha="center", va="bottom",
             fontsize=6.8, color=ANNOT, linespacing=1.35)

    # --- panel b: five consecutive load-unload cycles -------------------
    lf2, lb2, axbr, t_b, f_b, b_b, r_b = plot_pair(axb, df_b)
    axb.set_title("b  Five consecutive load–unload cycles (manual loading)",
                  loc="left")
    axb.legend([lf2, lb2], [r"$-F_z$ (ATI)", r"$-\Delta B_z$ (MLX90393)"],
               frameon=False, loc="upper left", handlelength=1.3,
               borderaxespad=0.2, labelspacing=0.3)
    axb.text(0.99, 0.97, f"Pearson $r$ = {r_b:.2f}",
             transform=axb.transAxes, ha="right", va="top",
             fontsize=6.8, color=ANNOT)

    fig.tight_layout(pad=0.6, w_pad=2.4)
    for ext in ("pdf", "png"):
        fig.savefig(f"{OUTDIR}/fig3c_event_alignment.{ext}",
                    facecolor="white", bbox_inches="tight")
    plt.close(fig)

    print(f"panel a: n={len(df_a)}, span {t_a[-1]:.1f} s, "
          f"peak -Fz {f_a.max():.1f} mN, peak -dBz {b_a.max():.1f} uT, "
          f"r={r_a:.3f}")
    print(f"panel b: n={len(df_b)}, span {t_b[-1]:.1f} s, "
          f"peak -Fz {f_b.max():.1f} mN, peak -dBz {b_b.max():.1f} uT, "
          f"r={r_b:.3f}")


if __name__ == "__main__":
    main()

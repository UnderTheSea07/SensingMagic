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
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({
    "font.size": 7,
    "axes.labelsize": 7,
    "axes.titlesize": 7,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "legend.fontsize": 6,
    "axes.linewidth": 0.6,
    "pdf.fonttype": 42,
    "figure.dpi": 300,
})

BLACK = "#000000"
ORANGE = "#D55E00"

SINGLE = ("/Users/arielzhang/Desktop/磁毛传感器可靠性分析原始数据/"
          "处理结果与图表_20260717/figures/data/Fig2_single_event.csv")
CYCLES = ("/Users/arielzhang/Desktop/磁毛传感器可靠性分析原始数据/"
          "处理结果与图表_20260717/figures/data/Fig3_five_cycles.csv")
OUTDIR = "/Users/arielzhang/Desktop/SensingMagic/paper/figures"


def styled(ax):
    ax.spines[["top", "right"]].set_visible(False)


def plot_pair(ax, df):
    """Twin-y plot of -Fz (mN, black, left) and -dBz (uT, orange, right)."""
    t = df["time_s"].to_numpy()
    f_mN = -df["Fz_N"].to_numpy() * 1e3          # compressive force, mN
    b_uT = -df["dBz_uT"].to_numpy()              # -dBz, uT

    axr = ax.twinx()
    lf, = ax.plot(t, f_mN, color=BLACK, lw=0.9, zorder=3)
    lb, = axr.plot(t, b_uT, color=ORANGE, lw=0.8, zorder=2)

    # headroom above the traces so legends/annotations never overlap data
    for a, y in ((ax, f_mN), (axr, b_uT)):
        span = y.max() - y.min()
        a.set_ylim(y.min() - 0.05 * span, y.max() + 0.28 * span)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel(r"Compressive force $-F_z$ (mN)", color=BLACK)
    axr.set_ylabel(r"$-\Delta B_z$ ($\mu$T)", color=ORANGE)
    axr.tick_params(axis="y", colors=ORANGE)
    axr.spines["right"].set_color(ORANGE)

    ax.spines[["top", "right"]].set_visible(False)
    axr.spines[["top", "left"]].set_visible(False)

    r = np.corrcoef(f_mN, b_uT)[0, 1]
    return lf, lb, axr, t, f_mN, b_uT, r


def main():
    df_a = pd.read_csv(SINGLE)
    df_b = pd.read_csv(CYCLES)

    fig, (axa, axb) = plt.subplots(
        1, 2, figsize=(7.1, 2.3), gridspec_kw={"width_ratios": [1.0, 1.9]})

    # --- panel a: single load-hold-unload event -------------------------
    lf, lb, axar, t_a, f_a, b_a, r_a = plot_pair(axa, df_a)
    axa.set_title("a  Single load-hold-unload event (manual loading)",
                  loc="left")
    axa.legend([lf, lb], [r"$-F_z$ (ATI)", r"$-\Delta B_z$ (MLX90393)"],
               frameon=False, loc="upper left", handlelength=1.4,
               borderaxespad=0.2)
    axa.text(0.52, 0.08,
             (f"peak $-F_z$ = {f_a.max():.0f} mN\n"
              f"peak $-\\Delta B_z$ = {b_a.max():.0f} $\\mu$T\n"
              f"Pearson $r$ = {r_a:.2f}"),
             transform=axa.transAxes, ha="center", va="bottom", fontsize=6)

    # --- panel b: five consecutive load-unload cycles -------------------
    lf2, lb2, axbr, t_b, f_b, b_b, r_b = plot_pair(axb, df_b)
    axb.set_title("b  Five consecutive load-unload cycles (manual loading)",
                  loc="left")
    axb.text(0.99, 0.97, f"Pearson $r$ = {r_b:.2f}",
             transform=axb.transAxes, ha="right", va="top", fontsize=6)

    fig.tight_layout(w_pad=2.2)
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

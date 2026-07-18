#!/usr/bin/env python
"""Assemble the Figure 2 composite page (raster draft, 300 dpi).

Grid (Nature double column, 183 mm / 7.2 in wide, <= 9.5 in tall):
    Row 1:  a  composition-series stress-strain   | b placeholder | c SEM group
    Row 2:  d  magnetization-direction stress-strain | e geometry-sensitivity heatmaps
    Row 3:  f  tap response / repeatability (full width; sub-panels f1-f3)
            f  150k-cycle durability + response-time histogram (centred;
               sub-panels f4-f5)

Sub-panel letters inside the f-row source rasters are neutral (f1..f5,
set in figC_response_event.py and fig2ef_characterization.py) so they can
never collide with the top-level a-f letters drawn here; the SEM group
uses the same convention (c1..c3, drawn by this script).

Raster assembly draft: panels are placed with imshow at native resolution;
vector originals ship separately. No content is redrawn or invented; the
b panel is an explicit placeholder, and no scale bars are synthesised on
the SEM micrographs (see on-figure note).

Excluded assets (disclosed in the manuscript notes):
    sem_extra1.png  - low-res draft stress-strain chart, not an SEM (redundant with a)
    sem_extra2.png  - oblique top view, redundant with c2 at this panel size
    fig2f_cyclic_stability.png - within-checkpoint stability, overlaps the
                                 150k-cycle durability panel chosen for f
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt

FIG_DIR = Path(__file__).resolve().parent.parent / "figures"
SEM_DIR = FIG_DIR / "sem"
OUT_STEM = FIG_DIR / "fig2_composite"

DPI = 300

# ---------------------------------------------------------------- geometry (inches)
W = 7.2          # Nature double-column width (183 mm)
M_L = 0.05       # outer margins
M_R = 0.05
M_T = 0.06
M_B = 0.10
HDR = 0.20       # header strip per row for the bold panel letters
GAP_ROW = 0.12
GAP_COL = 0.12
NOTE_H = 0.14    # strip under row 1 for the SEM scale note
WC = 1.66        # width of the c (SEM) group
GC = 0.06        # inner gap inside the c group
H2 = 2.00        # row-2 image height
F2_W = 4.70      # width of the second (centred) f strip
GAP_F = 0.10     # gap between the two f strips

LETTER_KW = dict(fontsize=10, fontweight="bold", family="sans-serif",
                 ha="left", va="top", color="black")


def load(path):
    img = mpimg.imread(str(path))
    h, w = img.shape[:2]
    return img, w / h


def main():
    img_a, ar_a = load(FIG_DIR / "fig2d_stress_strain_wtpct_sans.png")
    img_d, ar_d = load(FIG_DIR / "fig2d_stress_strain_direction_sans.png")
    img_e, ar_e = load(FIG_DIR / "fig2e_geometry_sensitivity.png")
    img_f1, ar_f1 = load(FIG_DIR / "fig2f_response_event.png")
    img_f2, ar_f2 = load(FIG_DIR / "fig2f_dynamics_durability.png")
    img_c1, ar_c1 = load(SEM_DIR / "sem_array_stage.png")
    img_c2, ar_c2 = load(SEM_DIR / "sem_array_side.png")
    img_c3, ar_c3 = load(SEM_DIR / "sem_cross_section_402um.png")

    usable = W - M_L - M_R

    # -- row 1: height set by the SEM group (c1 on top, c2|c3 below)
    c1_h = WC / ar_c1
    c23_h = (WC - GC) / (ar_c2 + ar_c3)
    h1 = c1_h + GC + c23_h
    a_w = ar_a * h1
    b_w = usable - a_w - WC - 2 * GAP_COL

    # -- row 2: d | e at common height H2, leftover split into pads + mid gap
    d_w = ar_d * H2
    e_w = ar_e * H2
    leftover = usable - d_w - e_w
    pad2 = leftover * 0.25
    gap2 = leftover * 0.50

    # -- row 3: f1 full width, f2 centred
    f1_h = usable / ar_f1
    f2_h = F2_W / ar_f2

    H = (M_T + HDR + h1 + NOTE_H + GAP_ROW
         + HDR + H2 + GAP_ROW
         + HDR + f1_h + GAP_F + f2_h + M_B)
    print(f"figure size: {W:.2f} x {H:.2f} in  (limit 9.5 in)")

    fig = plt.figure(figsize=(W, H))

    def ax_at(x, y_top, w, h):
        """Axes from top-left corner in inches."""
        ax = fig.add_axes([x / W, (H - y_top - h) / H, w / W, h / H])
        ax.set_axis_off()
        return ax

    def letter(x, y_top, s):
        fig.text(x / W, (H - y_top) / H, s, **LETTER_KW)

    # ---------------------------------------------------------------- row 1
    y = M_T
    letter(M_L, y, "a")
    x_b = M_L + a_w + GAP_COL
    letter(x_b, y, "b")
    x_c = x_b + b_w + GAP_COL
    letter(x_c, y, "c")
    y += HDR

    ax_at(M_L, y, a_w, h1).imshow(img_a, aspect="auto")

    # b: team-made fabrication workflow schematic (real asset).
    # Known typos in the source art ("Vaccum", "Demode") to fix in the
    # final vector redraw; used as-is for the assembly draft.
    img_b, _ = load(FIG_DIR / "assets" / "fig2b_fabrication_workflow.jpg")
    ax_at(x_b, y, b_w, h1).imshow(img_b, aspect="auto")

    # c group: c1 stage overview on top, c2 side view | c3 cross-section below
    sub_kw = dict(fontsize=6.5, fontweight="bold", family="sans-serif",
                  color="white", ha="left", va="top",
                  path_effects=[pe.withStroke(linewidth=1.4, foreground="black")])
    axc1 = ax_at(x_c, y, WC, c1_h)
    axc1.imshow(img_c1, aspect="auto")
    axc1.text(0.035, 0.955, "c1", transform=axc1.transAxes, **sub_kw)

    y_c23 = y + c1_h + GC
    c2_w = ar_c2 * c23_h
    c3_w = ar_c3 * c23_h
    axc2 = ax_at(x_c, y_c23, c2_w, c23_h)
    axc2.imshow(img_c2, aspect="auto")
    axc2.text(0.05, 0.93, "c2", transform=axc2.transAxes, **sub_kw)
    axc3 = ax_at(x_c + c2_w + GC, y_c23, c3_w, c23_h)
    axc3.imshow(img_c3, aspect="auto")
    axc3.text(0.05, 0.93, "c3", transform=axc3.transAxes, **sub_kw)

    # SEM scale note (no invented scale bars). When real bar lengths arrive
    # from the SEM metadata, draw the bars HERE at composite scale (axes
    # overlay on axc1/axc2/axc3) rather than relying on the tiny burned-in
    # '402.5 um' annotation inside the c3 raster.
    y_note = y + h1 + 0.03
    fig.text((x_c + WC / 2) / W, (H - y_note) / H,
             "scale: on-image annotation (cross-section);\n"
             "bars to be added from SEM metadata",
             ha="center", va="top", fontsize=5.0, color="#666666",
             family="sans-serif", linespacing=1.25)

    # ---------------------------------------------------------------- row 2
    y = M_T + HDR + h1 + NOTE_H + GAP_ROW
    x_d = M_L + pad2
    x_e = x_d + d_w + gap2
    letter(x_d, y, "d")
    letter(x_e, y, "e")
    y += HDR
    ax_at(x_d, y, d_w, H2).imshow(img_d, aspect="auto")
    ax_at(x_e, y, e_w, H2).imshow(img_e, aspect="auto")

    # ---------------------------------------------------------------- row 3
    y = M_T + HDR + h1 + NOTE_H + GAP_ROW + HDR + H2 + GAP_ROW
    letter(M_L, y, "f")
    y += HDR
    ax_at(M_L, y, usable, f1_h).imshow(img_f1, aspect="auto")
    x_f2 = M_L + (usable - F2_W) / 2
    ax_at(x_f2, y + f1_h + GAP_F, F2_W, f2_h).imshow(img_f2, aspect="auto")

    # ---------------------------------------------------------------- save
    for ext in ("png", "pdf"):
        out = OUT_STEM.with_suffix("." + ext)
        fig.savefig(out, dpi=DPI, facecolor="white")
        print("wrote", out)
    plt.close(fig)


if __name__ == "__main__":
    main()

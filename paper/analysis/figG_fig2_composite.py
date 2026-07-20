#!/usr/bin/env python
"""Assemble the Figure 2 composite page (raster draft, 300 dpi).

Grid (Nature double column, 183 mm / 7.2 in wide, <= 9.5 in tall):
    Row 1:  a  composition-series stress-strain | b fabrication workflow | c SEM group
    Row 2:  d  magnetization-direction stress-strain | e geometry-sensitivity heatmaps
    Row 3:  f  tap response / repeatability (full width; sub-panels f1-f3)
            f  150k-cycle durability + response-time histogram (centred;
               sub-panels f4-f5)

Layout rhythm: every row is a letter strip (HDR) over content; rows are
separated by one uniform GAP_ROW and columns within a row by one uniform
GAP_COL.  Row 1 solves the SEM-group width WC so that a | b | c share a
single common height and fill the page width exactly; row 2 derives its
height the same way for d | e.  The b JPG ships with white padding, so it
is cropped to content (trim_white) before its aspect ratio enters the
solve -- it is then placed undistorted.

Sub-panel letters inside the f-row source rasters are neutral (f1..f5,
set in figC_response_event.py and fig2ef_characterization.py) so they can
never collide with the top-level a-f letters drawn here; the SEM group
uses the same convention (c1..c3, drawn by this script).

Raster assembly draft: panels are placed with imshow at native resolution;
vector originals ship separately. No content is redrawn or invented; the
b panel is the team-made fabrication schematic used as-is (typos noted
below), and no scale bars are synthesised on the SEM micrographs (see
on-figure note).

Excluded assets (disclosed in the manuscript notes):
    sem_extra1.png  - low-res draft stress-strain chart, not an SEM (redundant with a)
    sem_extra2.png  - oblique top view, redundant with c2 at this panel size
    fig2f_cyclic_stability.png - within-checkpoint stability, overlaps the
                                 150k-cycle durability panel chosen for f
"""

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt

FIG_DIR = Path(__file__).resolve().parent.parent / "figures"
SEM_DIR = FIG_DIR / "sem_hires"
OUT_STEM = FIG_DIR / "fig2_composite"

DPI = 300

# ---------------------------------------------------------------- geometry (inches)
W = 7.2          # Nature double-column width (183 mm)
M_L = 0.05       # outer margins
M_R = 0.05
M_T = 0.06
M_B = 0.10
HDR = 0.20       # header strip per row for the bold panel letters
GAP_ROW = 0.24   # uniform gap between rows (content bottom -> next letter strip);
                 # the SEM scale note lives inside the first one
GAP_COL = 0.12   # uniform gap between columns within a row
GC = 0.06        # inner gap inside the c (SEM) group
F2_W = 4.70      # width of the second (centred) f strip
GAP_F = 0.12     # inner gap between the two f strips (sub-panels of one f)

LETTER_KW = dict(fontsize=10, fontweight="bold", family="sans-serif",
                 ha="left", va="top", color="black")


def load(path):
    img = mpimg.imread(str(path))
    h, w = img.shape[:2]
    return img, w / h


def trim_white(img, thr=245, pad=6, min_px=2):
    """Crop near-white borders (exported-JPG padding) via numpy slicing.

    A row/column counts as content when at least `min_px` of its pixels
    have a min-over-RGB value below `thr` (JPG noise tolerance); `pad`
    pixels of breathing room are kept on every side.
    """
    a = img
    if a.dtype != np.uint8:
        a = (np.clip(a, 0.0, 1.0) * 255).astype(np.uint8)
    g = a[..., :3].min(axis=2) if a.ndim == 3 else a
    mask = g < thr
    rows = np.where(mask.sum(axis=1) >= min_px)[0]
    cols = np.where(mask.sum(axis=0) >= min_px)[0]
    if rows.size == 0 or cols.size == 0:
        return img
    r0 = max(int(rows[0]) - pad, 0)
    r1 = min(int(rows[-1]) + pad + 1, g.shape[0])
    c0 = max(int(cols[0]) - pad, 0)
    c1 = min(int(cols[-1]) + pad + 1, g.shape[1])
    return img[r0:r1, c0:c1]


def main():
    img_a, ar_a = load(FIG_DIR / "fig2d_stress_strain_wtpct_sans.png")
    img_d, ar_d = load(FIG_DIR / "fig2d_stress_strain_direction_sans.png")
    img_e, ar_e = load(FIG_DIR / "fig2e_geometry_sensitivity.png")
    img_f1, ar_f1 = load(FIG_DIR / "fig2f_response_event.png")
    img_f2, ar_f2 = load(FIG_DIR / "fig2f_dynamics_durability.png")
    img_c1, ar_c1 = load(SEM_DIR / "101.jpg")          # stage overview, 400 um bar
    img_c2, ar_c2 = load(SEM_DIR / "08-8-1-01.jpg")    # cilia close-up, 300 um bar
    img_c3, ar_c3 = load(SEM_DIR / "05-3-04.jpg")      # cross-section, 100 um bar + 402.5 um

    # b: team-made fabrication workflow schematic (real asset).  The export
    # ships with white padding, so crop to content before its aspect ratio
    # enters the row-1 solve.  Known typos in the source art ("Vaccum",
    # "Demode") to fix in the final vector redraw; used as-is here.
    img_b = trim_white(mpimg.imread(str(FIG_DIR / "assets" / "fig2b_fabrication_workflow.jpg")))
    ar_b = img_b.shape[1] / img_b.shape[0]

    usable = W - M_L - M_R

    # -- row 1: solve the SEM-group width WC so that a | b | c share one
    #    common height h1 and fill the usable width exactly (no distortion,
    #    no slack).  The c group is affine in WC:  h1 = p * WC + q.
    p = 1.0 / ar_c1 + 1.0 / (ar_c2 + ar_c3)
    q = GC * (1.0 - 1.0 / (ar_c2 + ar_c3))
    U = usable - 2 * GAP_COL
    WC = (U - (ar_a + ar_b) * q) / ((ar_a + ar_b) * p + 1.0)
    h1 = p * WC + q
    a_w = ar_a * h1
    b_w = ar_b * h1
    c1_h = WC / ar_c1
    c23_h = (WC - GC) / (ar_c2 + ar_c3)

    # -- row 2: d | e share height H2 and fill the width with one GAP_COL
    H2 = (usable - GAP_COL) / (ar_d + ar_e)
    d_w = ar_d * H2
    e_w = ar_e * H2

    # -- row 3: f1 full width, f2 centred
    f1_h = usable / ar_f1
    f2_h = F2_W / ar_f2

    H = (M_T + HDR + h1 + GAP_ROW
         + HDR + H2 + GAP_ROW
         + HDR + f1_h + GAP_F + f2_h + M_B)
    print(f"figure size: {W:.2f} x {H:.2f} in  (limit 9.5 in)")
    print(f"row 1: h1 = {h1:.3f} in, WC = {WC:.3f} in;  row 2: H2 = {H2:.3f} in")

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
    # '402.5 um' annotation inside the c3 raster.  The note sits inside the
    # uniform row gap, flush with the page's right edge under the c group.
    y_note = y + h1 + 0.035
    fig.text((W - M_R) / W, (H - y_note) / H,
             "on-image scale bars: c1 400 \u03bcm, c2 300 \u03bcm,\n"
             "c3 100 \u03bcm (+402.5 \u03bcm width annotation)",
             ha="right", va="top", fontsize=5.5, color="#666666",
             family="sans-serif", linespacing=1.25)

    # ---------------------------------------------------------------- row 2
    y = M_T + HDR + h1 + GAP_ROW
    x_e = M_L + d_w + GAP_COL
    letter(M_L, y, "d")
    letter(x_e, y, "e")
    y += HDR
    ax_at(M_L, y, d_w, H2).imshow(img_d, aspect="auto")
    ax_at(x_e, y, e_w, H2).imshow(img_e, aspect="auto")

    # ---------------------------------------------------------------- row 3
    y = M_T + HDR + h1 + GAP_ROW + HDR + H2 + GAP_ROW
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

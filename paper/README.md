# Magnetic cilia tactile skin — manuscript (Nature Sensors target)

LaTeX project scaffold. **No experimental numbers are invented** — every missing
value is `\dataneeded{...}` (renders red) and every unverified citation is
`\refneeded{...}` (renders orange).

## Build
```
cd paper
pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```
Requires only base TeX Live (authblk removed). Produces `main.pdf` (~12 pp).

## Layout
- `main.tex` — master file, title (claim-gated), placeholder macros
- `sections/00_abstract.tex` … `06_extended_data.tex` — one file per section
- `sections/05_figure_captions.tex` — Fig. 1–5 panel-by-panel captions
- `references.bib` — placeholder only; add verified refs, do not fabricate
- `notes/claims_table.md` — claim → minimum-evidence gate; controls the title

## Fill-in workflow (per data batch)
QC → sync (MLX90393/Nano17/xArm/video) → clean → baseline/drift → split
(no leakage) → baseline model → improved model → unseen test → figures →
replace matching `\dataneeded{}` → update caption + claims table → list next
experiments.

## Claim gate (see notes/claims_table.md)
Title stays "magnetic cilia tactile skin" until the flat-film control + gain
statistics justify "cilia-enhanced"/"cilia-amplified".

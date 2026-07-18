# Claims admission table — gate every claim on evidence

Downgrade automatically when evidence is insufficient. Never over-state for tone.

| Claim | Minimum evidence | Current status |
|---|---|---|
| magnetic cilia tactile skin | basic response recorded | pending data |
| cilia-enhanced | significantly > flat control | **not yet met** (needs Fig. 2e controls) |
| cilia-amplified | defined gain, ≥3 independent devices, statistical significance, magnetic-material controlled | **not yet met** — title stays "cilia" not "cilia-amplified" until met |
| distributed full-hand readout | 44 nodes read stably | pending data |
| spatially resolved | unseen-position localization error reported | pending data |
| force map | Nano17 calibration + independent test | pending data |
| normal/shear map | tri-axis decoupling + direction error | pending data |
| airflow map | at least airflow-induced tactile response map (NOT flow field w/o PIV/CFD) | pending data |
| physiological pulse sensing | multi-subject + ECG/PPG | pending data |
| closed-loop control | feedback vs no-feedback comparison | **not attempted** — use "grasp-state discrimination" |

## Title decision
- Default (current `main.tex`): "A full-hand magnetic cilia tactile skin for spatially distributed perception of subtle mechanical stimuli"
- Upgrade to "Cilia-amplified ... embodied spatial perception ..." ONLY after the `cilia-amplified` row is met.

## Words banned until data justify them
ultrahigh, unprecedented, breakthrough, ultra-sensitive, world-first.

## Existing-data status (2026-07-18 review of repo CSVs)
All 1880 CSVs are **single-MLX90393** (`ms,X,Y,Z`). No 2×2 / 1×4 / 44-node
array data, no Nano17 force column, no ECG/PPG. Consequence:
- **Fig. 1, 3, 4 (array, force decoding, full-hand): zero backing data.** All
  those Results/claims are structural placeholders only.
- **Fig. 2 (cilia): SUBSTANTIALLY BACKED as of 2026-07-18** by the
  reliability-project dataset (see `reliability_dataset_mapping.md`):
  9-geometry sensitivity surface (R²=0.92 tangential), 150k-cycle fatigue
  (n=3, stable ±4%), response 25 ms / recovery 26 ms, ATI force reference.
  Still missing: flat/pillar/non-magnetic/bare controls — the
  `cilia-enhanced`/`cilia-amplified` gate REMAINS CLOSED; manual-loading CV
  13.4% means force-calibrated claims should be redone per SOP-1.
- **Fig. 5a airflow: DATA INSUFFICIENT** — see `wind_data_verdict.md`. No
  credible ΔB–v (fan-EMI confound + drift + non-monotonic). Placeholders stay.
- **Fig. 5d–f pulse: QUALITATIVE ONLY** — see `pulse_data_verdict.md`.
  Coherent averaged waveform exists; no ECG → HR-MAE/delay/multi-subject
  claims unsupported. One raw+beat-average demo panel is defensible.
- **Reusable real number:** single-sensor fastmode logging = 1000 Hz measured.

# Pulse (Fig. 5d-f) data verdict — 2026-07-18

**Verdict: data supports ONLY a qualitative "pulse micro-vibration is
captured" demonstration (one raw + beat-averaged panel, Fig. 5d or a Fig. 1a
inset). The quantitative Fig. 5e/f claims — HR MAE vs ECG, ECG-to-pulse delay,
multi-subject, multi-position — remain unsupported. Keep those `\dataneeded`
placeholders empty.**

## Dataset (65 files analyzed; paper/analysis/pulse_pipeline.py)
- Three eras: `single_pulse0120_*` (500 Hz & 50 Hz), `single_XYZ_pulse*`
  (1000 Hz), `l8_d0.8_num1_trial*_pulse_fastmode` (1000 Hz). L8/d0.8 cilia,
  single MLX90393, all `ms,X,Y,Z`.
- **No ECG or PPG file exists anywhere in the repo** (verified by search).
- Subject appears to be a single person ("num1"); no subject IDs, no ethics
  metadata, no attachment-pressure/position log.

## What the data DOES show (real, positive)
- Beat-aligned averaging reveals a **coherent systolic waveform** in most
  trials: clear upstroke at the detected peak, ~0.3-2 uT amplitude
  (e.g. l8 trial5, single_pulse0120_65). This is genuine evidence that
  cilia + single MLX90393 pick up radial-pulse micro-vibration.
- Sampling rates verified: 50 / 500 / 1000 Hz depending on file.

## What the data does NOT support (fails honestly)
1. **No reference → no HR MAE, no pulse-transit delay.** The paper's core
   Fig. 5e numbers require synchronized ECG (and ideally PPG). None exists.
2. **Standalone HR extraction is unreliable.** PSD-peak HR (44-100 bpm) and
   beat-count HR (63-100 bpm) disagree by 20-40 bpm on the same file; the
   spectral picker is fooled by harmonics/broadband noise (no sharp
   fundamental in most PSDs).
3. **Filename HR labels are not ground truth.** They come from an unnamed,
   unsynchronized device. Only 8/35 labeled files fall within label±5 bpm of
   the magnetic PSD-HR; median |error| = 10 bpm. Cannot be used to validate.
4. **Beat-to-beat SNR is low.** Mean-beat energy < across-beat residual in
   nearly all files (SNR -5 to -15 dB by that definition): the averaged
   waveform is coherent but individual beats are noisy → morphology features
   (dicrotic notch) NOT reliably resolvable.
5. Single subject, single sensor, single wrist position → none of the
   multi-subject / multi-position panels (Fig. 5f) are backed.
6. One file unreadable: `single_XYZ_0120pulse04_20260120_214024.csv`
   (non-standard column layout).

## Re-measurement SOP (to make Fig. 5d-f credible)
1. **Synchronized ECG on every recording** (hardware trigger or shared
   clock); add PPG for waveform-morphology cross-check.
2. **Multi-subject cohort**: exploratory n=5, main n=10-20; 3 sessions each
   with re-attachment; log subject ID, wrist side, posture, attachment
   pressure, patch location.
3. **Multi-position** per protocol: centre, ±5 mm radial/ulnar, ±10 mm
   proximal/distal, 3×60 s each.
4. Report HR MAE vs ECG, ECG-to-pulse delay, waveform correlation, SNR,
   beat-to-beat CV — only after (1).
5. Ethics approval + informed consent before any of the above.

## For the manuscript now
- Fig. 5d may show ONE real trial: raw + band-passed + beat-aligned average,
  labelled as a qualitative micro-vibration demonstration.
- Do NOT state a heart rate as validated; do NOT claim ECG synchronization.

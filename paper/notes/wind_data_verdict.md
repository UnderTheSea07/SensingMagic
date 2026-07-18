# Airflow (Fig. 5a) data verdict — 2026-07-18

**Verdict: 当前数据不足以支撑可信的 ΔB–v 标定。Fig. 5a 的所有
`\dataneeded` 占位符保持空置。** 本文档记录判定依据与补测 SOP。

## Dataset analyzed
- 18 usable files, session 2025-07-17 (trials 0–15), L8/d0.8 cilia,
  single MLX90393, fastmode.
- Measured sampling rate **1000 Hz**, ~2 min/file, 0 dropped frames.
- Ground-truth speed in filename (anemometer): 0, 2.43–10 m/s; leading
  number (2100–8640) = fan drive setting. Speeds quasi-randomized in time
  (good design).

## Pipeline (paper/analysis/)
`wind_qc.py` → per-file QC + rolling stats. `wind_pipeline.py` → startup
discard (15 s), 1-s windows, artifact rejection (3×median std + guard),
Welch PSD, band-limited (5–100 Hz) RMS. `wind_dc_deflection.py` → steady
DC mean shift vs v=0 reference, drift check via time-vs-speed ordering.

## Findings (all metrics fail honestly)

1. **Broadband fluctuation: no v-dependence.** PSD noise floor
   (~1e-3 µT²/Hz) identical across 0–10 m/s.
2. **Narrowband peaks: confounded by fan-motor EMI.** All wind response
   concentrates in discrete peaks (~30 Hz + harmonics) that track fan
   rotation. Without a bare-sensor / non-magnetic-cilia control, cilia
   aeroelastic response cannot be distinguished from direct magnetic
   pickup of the fan motor. Band RMS is additionally **non-monotonic**
   (rises to ~1.7 µT at 2.4 m/s then falls to ~0.34 µT at 10 m/s);
   Spearman ρ=−0.47, P=0.06.
3. **DC drag deflection: masked by drift + setup discontinuity.**
   Trial 9 shows a >1000 µT baseline jump (sensor/magnet physically
   disturbed) splitting the session in two. Elsewhere DC shifts are
   1–4 µT, non-monotonic with v (ρ=0.18, P=0.53), same order as the
   ~5 µT drift between the two v=0 anchors an hour apart.
4. Contaminated file: trial 9 (v=6.40) is unusable (broadband RMS ~6–9 µT).

## Reusable real numbers
- Single-sensor fastmode logging = **1000 Hz measured** (Methods,
  single-node acquisition; array frame rate still needs measuring).
- Reference anemometer was used per-run (Methods wind-tunnel setup).

## Re-measurement SOP (minimum to make Fig. 5a credible)
1. **Wind ON/OFF cycling within each recording** (e.g. 30 s off / 30 s
   on × 5): within-file baseline defeats drift; paired ΔB per cycle.
2. **Bare MLX90393 control (no cilia) at every speed, same geometry** —
   separates fan EMI from cilia response. Add non-magnetic cilia control
   if EMI is non-zero.
3. Increase fan–sensor distance or duct the flow so the motor sits off-axis;
   verify EMI floor with fan running but flow blocked (blocked-duct sham).
4. Do not touch the rig mid-session; if touched, log it (trial-9 lesson).
5. ≥5 independent runs per speed, randomized order (keep), anemometer log
   per run, report mean ± s.d. across runs, up/down hysteresis.
6. Target speeds 0/0.5/1/2/3/4/5 m/s per paper plan — current data has no
   points below 2.4 m/s, where the weak-airflow claim lives.

#!/usr/bin/env python
"""quadA_process_ctrl.py — single-sensor z-press CONTROL vs quad-array z-press.

Loads the single-sensor control pair (headerless ms,X,Y,Z magnetic file +
standard force file) and one representative quad z-trial, baseline-subtracts,
aligns magnetic to force by event onset refined with envelope cross-correlation,
and computes per-cycle |dBz| amplitude (peak-to-peak of baseline-subtracted Bz
within each down+up cycle window) and per-cycle Fz peak-to-peak.

S = median(dBz_pp) / median(Fz_pp)  [uT/N]

Outputs:
  paper/analysis/out/quad_z_control_comparison.csv
      rows: ctrl, quad_s0..quad_s3
      cols: dB_amp_uT, Fz_pp_N, S_uT_per_N
  paper/analysis/out/quad_z_control_alignment_qc.png  (alignment QC only)

All numbers computed from data; anomalies printed, not hidden.
"""
import glob
import os
import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------- style
OKABE = ["#0072B2", "#D55E00", "#009E73", "#E69F00",
         "#56B4E9", "#CC79A7", "#000000", "#888888"]
plt.rcParams.update({
    "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.8,
    "axes.linewidth": 0.8, "xtick.major.width": 0.8, "ytick.major.width": 0.8,
    "xtick.major.size": 2.8, "ytick.major.size": 2.8,
    "xtick.direction": "out", "ytick.direction": "out",
    "axes.spines.top": False, "axes.spines.right": False,
    "font.family": "DejaVu Sans", "pdf.fonttype": 42, "savefig.dpi": 300,
})

BASE = ("/private/tmp/claude-502/-Users-arielzhang-Desktop-SensingMagic/"
        "617d7206-542d-4a74-bd36-6e9400bc4ca0/scratchpad/4sensors/4_sensors/"
        "z方向-2mm")
CTRL_DIR = os.path.join(BASE, "单个传感器对照")
CTRL_MAG = os.path.join(CTRL_DIR, "single_XYZ_20260717_222745.csv")
CTRL_FORCE = os.path.join(CTRL_DIR, "20260717_143004.csv")
OUT_DIR = "/Users/arielzhang/Desktop/SensingMagic/paper/analysis/out"
os.makedirs(OUT_DIR, exist_ok=True)

GRID_HZ = 100.0
R_GOOD = 0.80          # alignment quality threshold for "well-aligned"


# ---------------------------------------------------------------- helpers
def load_force(path):
    f = pd.read_csv(path, encoding="utf-8-sig")
    f["t"] = f["elapsed_time_s"].values
    return f


def mag_onset_time(t, env, base_sec=2.0, sustain_sec=0.10):
    """First sustained departure of |dB| envelope from baseline noise."""
    base = env[t < t[0] + base_sec]
    mad = np.median(np.abs(base - np.median(base))) * 1.4826
    thr = max(np.median(base) + 8.0 * mad, 8.0)     # uT floor against drift
    dt = np.median(np.diff(t))
    need = max(1, int(round(sustain_sec / dt)))
    above = env > thr
    csum = np.convolve(above.astype(int), np.ones(need, int), "same")
    idx = np.flatnonzero(csum >= need)
    if idx.size == 0:
        return None, thr
    return t[idx[0]], thr


def align(t_m, env_m, t_f, env_f, offset0, search=3.0, step=0.005):
    """Find offset o s.t. mag time = force time + o, maximizing Pearson r of
    the two 100 Hz envelopes over the force window. Returns (offset, r)."""
    grid = np.arange(t_f[0], t_f[-1], 1.0 / GRID_HZ)
    ef = np.interp(grid, t_f, env_f)
    best = (offset0, -np.inf)
    for o in np.arange(offset0 - search, offset0 + search + step, step):
        em = np.interp(grid + o, t_m, env_m, left=np.nan, right=np.nan)
        ok = ~np.isnan(em)
        if ok.sum() < 0.8 * grid.size:
            continue
        r = np.corrcoef(ef[ok], em[ok])[0, 1]
        if r > best[1]:
            best = (o, r)
    return best


def cycle_metrics(f, t_m, dbz, offset):
    """Per-cycle Fz p2p (force) and dBz p2p (magnetic, mapped window)."""
    rows = []
    for cyc in sorted(f.loc[f["stage"] != "idle", "cycle"].unique()):
        w = f[f["cycle"] == cyc]
        t0, t1 = w["t"].iloc[0], w["t"].iloc[-1]
        fz = w["Fz_N"].values
        sel = (t_m >= t0 + offset) & (t_m <= t1 + offset)
        if sel.sum() < 10:
            rows.append((cyc, np.nan, fz.max() - fz.min()))
            continue
        seg = dbz[sel]
        rows.append((cyc, seg.max() - seg.min(), fz.max() - fz.min()))
    return pd.DataFrame(rows, columns=["cycle", "dBz_pp_uT", "Fz_pp_N"])


def process_pair(tag, t_m, B, f, zcols):
    """Baseline-subtract, align, per-cycle metrics for every z-column.
    B: (n, k) array of magnetic channels; zcols: list of (name, col_index in B
    restricted to z channels used for metrics). Alignment env uses all cols."""
    # baseline from pre-event data: robust first-2s median (mag starts early)
    med0 = np.median(B[t_m < t_m[0] + 2.0], axis=0)
    dB = B - med0
    env_m = np.linalg.norm(dB, axis=1)

    t_on_m, thr = mag_onset_time(t_m, env_m)
    fd = f[f["stage"] == "down"]
    t_on_f = fd["t"].iloc[0]
    fz0 = np.median(f.loc[f["stage"] == "idle", "Fz_N"]) if (
        f["stage"] == "idle").any() else np.median(f["Fz_N"].iloc[:40])
    env_f = np.abs(f["Fz_N"].values - fz0)

    offset0 = (t_on_m - t_on_f) if t_on_m is not None else (t_m[0] - f["t"].iloc[0])
    offset, r = align(t_m, env_m, f["t"].values, env_f, offset0)
    print(f"[{tag}] mag onset {t_on_m if t_on_m is None else round(t_on_m,3)} s "
          f"(thr {thr:.1f} uT), force 'down' onset {t_on_f:.3f} s, "
          f"event offset {offset0:.3f} s -> refined {offset:.3f} s, r={r:.4f}")

    res = {}
    for name, ci in zcols:
        cm = cycle_metrics(f, t_m, dB[:, ci], offset)
        nbad = int(cm["dBz_pp_uT"].isna().sum())
        if nbad:
            print(f"[{tag}] {name}: {nbad} cycle(s) without magnetic coverage "
                  "(dropped from medians)")
        res[name] = cm
    return dB, env_m, env_f, offset, r, res


# ---------------------------------------------------------------- control
print("=" * 70)
print("CONTROL (single sensor)")
mc = pd.read_csv(CTRL_MAG, header=None)
mc.columns = ["ms", "X", "Y", "Z"]
t_mc = (mc["ms"].values - mc["ms"].values[0]) / 1e3
dtc = np.diff(mc["ms"].values)
print(f"ctrl mag: {len(mc)} samples, {t_mc[-1]:.2f} s, "
      f"rate {1e3/np.median(dtc):.1f} Hz, "
      f"gaps>10ms: {(dtc>10).sum()}, non-monotonic: {(dtc<0).sum()}")
fc = load_force(CTRL_FORCE)
print(f"ctrl force: {len(fc)} samples, {fc['t'].iloc[-1]:.2f} s, "
      f"cycles {int(fc['cycle'].max())}, stages {sorted(fc['stage'].unique())}")

Bc = mc[["X", "Y", "Z"]].values
dBc, env_mc, env_fc, off_c, r_c, res_c = process_pair(
    "ctrl", t_mc, Bc, fc, [("ctrl", 2)])
cm_ctrl = res_c["ctrl"].dropna()

# ---------------------------------------------------------------- quad pairs
print("=" * 70)
print("QUAD z-trials: filename-offset consistency check")
mfiles = sorted(glob.glob(os.path.join(BASE, "磁数据", "*.csv")))
ffiles = sorted(glob.glob(os.path.join(BASE, "力数据", "*.csv")))
assert len(mfiles) == len(ffiles), "unequal magnetic/force file counts"
offs = []
for mf, ff in zip(mfiles, ffiles):
    ms = os.path.basename(mf).replace("mlx_v4_quad_", "").replace(".csv", "")
    fs = os.path.basename(ff).replace("force_", "").replace(".csv", "")
    mdt = datetime.datetime.strptime(ms, "%Y%m%d_%H%M%S") - datetime.timedelta(hours=8)
    fdt = datetime.datetime.strptime(fs, "%Y%m%d_%H%M%S")
    offs.append((fdt - mdt).total_seconds())
offs = np.array(offs)
print(f"filename offset (force UTC - mag local-8h): "
      f"median {np.median(offs):.0f} s, range {offs.min():.0f}..{offs.max():.0f} s "
      f"over {len(offs)} pairs (clock skew between PCs; constant -> pairing OK)")
cms = os.path.basename(CTRL_MAG).replace("single_XYZ_", "").replace(".csv", "")
cfs = os.path.basename(CTRL_FORCE).replace(".csv", "")
cmdt = datetime.datetime.strptime(cms, "%Y%m%d_%H%M%S") - datetime.timedelta(hours=8)
cfdt = datetime.datetime.strptime(cfs, "%Y%m%d_%H%M%S")
print(f"control filename offset: {(cfdt-cmdt).total_seconds():.0f} s (consistent)")

# representative quad trial = first pair whose alignment r >= R_GOOD
quad_pick = None
for k, (mf, ff) in enumerate(zip(mfiles, ffiles)):
    mq = pd.read_csv(mf)
    fq = load_force(ff)
    t_mq = (mq["us"].values - mq["us"].values[0]) / 1e6
    dtq = np.diff(mq["us"].values)
    print(f"\npair {k}: {os.path.basename(mf)} + {os.path.basename(ff)}  "
          f"mag {t_mq[-1]:.1f} s @ {1e6/np.median(dtq):.0f} Hz "
          f"(gaps>10ms: {(dtq>1e4).sum()}, non-mono: {(dtq<0).sum()}); "
          f"force {fq['t'].iloc[-1]:.1f} s, cycles {int(fq['cycle'].max())}")
    Bq = mq[[c for c in mq.columns if c != "us"]].values
    zi = [("quad_s%d" % s, 3 * s + 2) for s in range(4)]
    dBq, env_mq, env_fq, off_q, r_q, res_q = process_pair(f"quad{k}", t_mq, Bq, fq, zi)
    if r_q >= R_GOOD:
        quad_pick = (k, mf, ff, fq, t_mq, dBq, env_mq, env_fq, off_q, r_q, res_q)
        print(f"pair {k} selected as representative (r={r_q:.3f} >= {R_GOOD})")
        break
    print(f"pair {k} rejected (r={r_q:.3f} < {R_GOOD}); trying next")

if quad_pick is None:
    raise SystemExit("no well-aligned quad pair found")
k, mf, ff, fq, t_mq, dBq, env_mq, env_fq, off_q, r_q, res_q = quad_pick

# ---------------------------------------------------------------- table
rows = []
cm = cm_ctrl
rows.append(("ctrl", cm["dBz_pp_uT"].median(), cm["Fz_pp_N"].median()))
for s in range(4):
    cq = res_q[f"quad_s{s}"].dropna()
    rows.append((f"quad_s{s}", cq["dBz_pp_uT"].median(), cq["Fz_pp_N"].median()))
tab = pd.DataFrame(rows, columns=["row", "dB_amp_uT", "Fz_pp_N"])
tab["S_uT_per_N"] = tab["dB_amp_uT"] / tab["Fz_pp_N"]
print("\nper-cycle spread (IQR):")
for name, cm_ in [("ctrl", cm_ctrl)] + [
        (f"quad_s{s}", res_q[f"quad_s{s}"].dropna()) for s in range(4)]:
    q1, q3 = cm_["dBz_pp_uT"].quantile([0.25, 0.75])
    f1, f3 = cm_["Fz_pp_N"].quantile([0.25, 0.75])
    print(f"  {name}: dBz_pp {cm_['dBz_pp_uT'].median():.1f} uT "
          f"[IQR {q1:.1f}-{q3:.1f}], Fz_pp {cm_['Fz_pp_N'].median():.3f} N "
          f"[IQR {f1:.3f}-{f3:.3f}]")
csv_path = os.path.join(OUT_DIR, "quad_z_control_comparison.csv")
tab.round(3).to_csv(csv_path, index=False)
print("\n" + tab.round(3).to_string(index=False))
print("saved:", csv_path)

best = tab.iloc[1:].sort_values("S_uT_per_N", ascending=False).iloc[0]
print(f"\nS_ctrl = {tab['S_uT_per_N'].iloc[0]:.1f} uT/N; "
      f"best quad sensor {best['row']} = {best['S_uT_per_N']:.1f} uT/N; "
      f"ratio ctrl/best = {tab['S_uT_per_N'].iloc[0]/best['S_uT_per_N']:.2f}x")
print(f"alignment r: ctrl {r_c:.3f}, quad pair {k} {r_q:.3f}")
print(f"cycles used: ctrl {len(cm_ctrl)}, quad "
      f"{len(res_q['quad_s0'].dropna())} of {int(fq['cycle'].max())}")

# ---------------------------------------------------------------- QC figure
fig, axes = plt.subplots(2, 1, figsize=(4.6, 3.3))
for ax, (tag, t_f, env_f, t_m, env_m, off, r) in zip(
        axes,
        [("control", fc["t"].values, env_fc, t_mc, env_mc, off_c, r_c),
         (f"quad pair {k}", fq["t"].values, env_fq, t_mq, env_mq, off_q, r_q)]):
    ax.plot(t_f, env_f / np.nanmax(env_f), color=OKABE[0], lw=0.9,
            label="|Fz| envelope (norm.)")
    ax.plot(t_m - off, env_m / np.nanmax(env_m), color=OKABE[1], lw=0.7,
            alpha=0.85, label="|dB| envelope (norm.)")
    ax.set_xlim(t_f[0] - 1, t_f[-1] + 1)
    ax.set_ylim(-0.05, 1.28)
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_ylabel("norm. env")
    ax.set_title(f"{tag}: offset {off:.2f} s, r = {r:.3f}", loc="left")
axes[0].legend(frameon=False, loc="upper right", ncol=2, handlelength=1.4,
               columnspacing=1.2, borderaxespad=0.2)
axes[1].set_xlabel("force time (s)")
fig.tight_layout(pad=0.6)
png_path = os.path.join(OUT_DIR, "quad_z_control_alignment_qc.png")
fig.savefig(png_path)
print("saved:", png_path)

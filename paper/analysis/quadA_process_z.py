#!/usr/bin/env python
"""quadA_process_z: quad-MLX90393 z-direction press trials (2 mm press cycles).

Dataset: <scratchpad>/4sensors/4_sensors/z方向-2mm with subfolders
  磁数据/mlx_v4_quad_*.csv  -- us,X0..Z3 (12 channels, 4 sensors x 3 axes),
                              ~559 Hz, device microsecond clock (no unix base)
  力数据/force_*.csv        -- sample,elapsed_time_s,system_time,cycle,stage,
                              Fx_N..Tz_Nm,raw_*; 200 Hz; stages idle/down/up
                              (z press protocol), cycles 0..20 (cycle 0 = idle)

Pipeline
 1. Pair magnetic and force files in filename-time order. Magnetic filename is
    LOCAL time (UTC+8), force filename is UTC. Verify the per-pair offset
    mag_local - (force_utc + 8 h) is constant; exclude pairs deviating > 10 s
    from the median offset (with disclosure).
 2. Per pair: per-channel baseline-subtracted dB (baseline = per-channel median
    of a pre-event magnetic window). Align force time to magnetic time by
    matching the press-event sequences (peaks of the 100 Hz |dB| and |F|
    envelopes; anchor = median per-event lag, tolerant to spurious extra
    magnetic events via best contiguous-subset matching), then refine with
    cross-correlation of the envelopes within +-0.45 s (< half a press cycle,
    so no cycle-slip). Matched event onsets (first sustained departure from
    baseline) are the fallback anchor if event matching fails. Record best r
    and the median absolute per-event timing residual. Rationale: pure
    onset anchoring is broken by pre-protocol disturbances in the magnetic
    record (observed in one trial), whereas event-sequence matching is not.
 3. Per press cycle (1..20): the stage labels are motion COMMANDS and the
    measured force peak lands at the very end of 'down' or spills into 'up'
    (F/T readout lags the labels), so the press plateau is defined from the
    force signal itself: the contiguous run of samples around the cycle's
    peak |Fz| (down+up rows) with |Fz| >= 80% of that peak. Mean Fx/Fy/Fz
    over the plateau, mean dB per channel over the plateau window mapped into
    magnetic time, plus per-cycle peak |F| (over the full cycle). Rows are
    labelled stage='down' (the press event).
 4. Tidy table -> paper/analysis/out/quad_features_z.csv
    columns: trial,cycle,stage,dBx0..dBz3,Fx,Fy,Fz,peakF
 5. Diagnostic figure -> paper/figures/fig3_quad_z_overview.{pdf,png}
    a: representative trial, 12-channel dB time series (4 sensor rows,
       axes color-coded), 'down' stages shaded.
    b: cycle-averaged down-plateau response per channel (mean +- s.d. across
       all cycles x trials).

Units: magnetic CSV values are raw integer device counts; the dataset notes
say "uT presumably". Axes are labelled uT per those notes, but the uT
interpretation is presumed (MLX90393 LSB->uT scale not recorded); all numbers
are in the file's native units either way.

Every reported number is computed from the data at run time.
"""

import glob
import os
import re
from datetime import datetime, timedelta

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

# ----------------------------------------------------------------------------
# paths / constants
# ----------------------------------------------------------------------------
DATA = ("/private/tmp/claude-502/-Users-arielzhang-Desktop-SensingMagic/"
        "617d7206-542d-4a74-bd36-6e9400bc4ca0/scratchpad/4sensors/4_sensors/"
        "z方向-2mm")
MAG_DIR = os.path.join(DATA, "磁数据")
FORCE_DIR = os.path.join(DATA, "力数据")
OUT_DIR = "/Users/arielzhang/Desktop/SensingMagic/paper/analysis/out"
FIG_DIR = "/Users/arielzhang/Desktop/SensingMagic/paper/figures"
FEATURES_CSV = os.path.join(OUT_DIR, "quad_features_z.csv")
FIG_BASE = os.path.join(FIG_DIR, "fig3_quad_z_overview")

GRID_HZ = 100.0           # common envelope grid
ENV_SMOOTH_S = 0.05       # envelope rolling-mean window
ONSET_K = 8.0             # onset threshold = base + K*MAD
ONSET_SUSTAIN_S = 0.10    # must stay above threshold this long
LAG_SEARCH_S = 0.45       # refine window around anchor (< half press cycle)
LAG_STEP_S = 0.005
PLATEAU_FRAC = 0.80       # |Fz| >= frac * cycle peak |Fz|
OFFSET_TOL_S = 10.0       # pairing filename-offset tolerance vs median
R_MIN = 0.50              # drop trial from features below this (disclosed)

CH_NAMES = [f"dB{a}{s}" for s in range(4) for a in ("x", "y", "z")]

# Okabe-Ito
OI = ["#0072B2", "#D55E00", "#009E73", "#E69F00",
      "#56B4E9", "#CC79A7", "#000000", "#888888"]
AX_COLORS = {"x": OI[0], "y": OI[1], "z": OI[2]}

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
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
    "figure.dpi": 300,
})


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def fname_time(path):
    m = re.search(r"(\d{8}_\d{6})", os.path.basename(path))
    return datetime.strptime(m.group(1), "%Y%m%d_%H%M%S")


def smooth(x, n):
    return pd.Series(x).rolling(max(int(n), 1), center=True,
                                min_periods=1).mean().values


def onset_time(t, env, base_n):
    """First sustained departure of env from its own leading baseline."""
    base = np.median(env[:base_n])
    mad = np.median(np.abs(env[:base_n] - base))
    thr = base + ONSET_K * max(mad, 1e-3 * max(np.ptp(env), 1e-9))
    above = env > thr
    need = max(int(ONSET_SUSTAIN_S * GRID_HZ), 1)
    run = 0
    for i, a in enumerate(above):
        run = run + 1 if a else 0
        if run >= need:
            return t[i - need + 1], thr
    return None, thr


def env_events(t, env, base_n, min_period=0.5):
    """Times of press-event peaks in an envelope."""
    base = np.median(env[:base_n])
    thr = base + 0.5 * (env.max() - base)
    pk, _ = find_peaks(env, height=thr, distance=int(min_period * GRID_HZ))
    return t[pk]


def event_anchor(ev_m, ev_f):
    """Median per-event lag; tolerates spurious extra magnetic events by
    best contiguous-subset matching. Returns (lag, n_matched, mad) or None."""
    if len(ev_m) < 5 or len(ev_f) < 5:
        return None
    if len(ev_m) < len(ev_f):
        return None                    # missing magnetic events: bail out
    n = len(ev_f)
    best = None
    for s in range(len(ev_m) - n + 1):
        d = ev_m[s:s + n] - ev_f
        spread = np.median(np.abs(d - np.median(d)))
        if best is None or spread < best[2]:
            best = (float(np.median(d)), n, float(spread))
    return best


def xcorr_refine(t_m, env_m, t_f, env_f, lag0, half_win):
    """Pearson r between env_f(t) and env_m(t+lag) over force support."""
    lags = np.arange(lag0 - half_win, lag0 + half_win + LAG_STEP_S / 2,
                     LAG_STEP_S)
    best = (-2.0, lag0)
    ef = (env_f - env_f.mean()) / (env_f.std() + 1e-12)
    for lag in lags:
        em = np.interp(t_f + lag, t_m, env_m, left=np.nan, right=np.nan)
        ok = np.isfinite(em)
        if ok.sum() < 0.9 * len(t_f):
            continue
        em = em[ok]
        em = (em - em.mean()) / (em.std() + 1e-12)
        r = float(np.mean(em * ef[ok]))
        if r > best[0]:
            best = (r, float(lag))
    return best  # (r, lag)


# ----------------------------------------------------------------------------
# 1. pairing + filename-offset verification
# ----------------------------------------------------------------------------
def pair_files():
    mags = sorted(glob.glob(os.path.join(MAG_DIR, "*.csv")), key=fname_time)
    forces = sorted(glob.glob(os.path.join(FORCE_DIR, "*.csv")),
                    key=fname_time)
    assert len(mags) == len(forces), (
        f"unequal file counts: {len(mags)} magnetic vs {len(forces)} force")
    offsets = np.array([
        (fname_time(m) - (fname_time(f) + timedelta(hours=8))).total_seconds()
        for m, f in zip(mags, forces)])
    med = float(np.median(offsets))
    keep, dropped = [], []
    for m, f, off in zip(mags, forces, offsets):
        (keep if abs(off - med) <= OFFSET_TOL_S else dropped).append(
            (m, f, off))
    return keep, dropped, offsets, med


# ----------------------------------------------------------------------------
# 2. load + align one pair
# ----------------------------------------------------------------------------
def process_pair(mag_path, force_path, log):
    trial = re.search(r"(\d{8}_\d{6})", os.path.basename(mag_path)).group(1)
    dm = pd.read_csv(mag_path)
    df = pd.read_csv(force_path, encoding="utf-8-sig")

    us = dm["us"].to_numpy(np.int64)
    wraps = int((np.diff(us) < 0).sum())
    if wraps:
        us = us + (np.cumsum(np.r_[0, np.diff(us) < 0]) << 32)
        log.append(f"{trial}: fixed {wraps} us-clock wrap(s)")
    t_m = (us - us[0]) / 1e6
    B = dm.iloc[:, 1:13].to_numpy(float)          # 12 channels
    fs_m = len(t_m) / (t_m[-1] - t_m[0])

    t_f = df["elapsed_time_s"].to_numpy(float)
    F = df[["Fx_N", "Fy_N", "Fz_N"]].to_numpy(float)

    # provisional baseline (first 2 s) for the alignment envelope
    b0 = np.median(B[: int(2 * fs_m)], axis=0)
    env_norm = np.linalg.norm(B - b0, axis=1)

    # 100 Hz envelopes
    tg_m = np.arange(t_m[0], t_m[-1], 1 / GRID_HZ)
    tg_f = np.arange(t_f[0], t_f[-1], 1 / GRID_HZ)
    em = np.interp(tg_m, t_m, smooth(env_norm, ENV_SMOOTH_S * fs_m))
    ef = np.interp(tg_f, t_f, smooth(np.linalg.norm(F, axis=1),
                                     ENV_SMOOTH_S * 200))

    # anchor lag (t_mag = t_force + lag): press-event sequence matching,
    # onset matching as fallback
    on_m, _ = onset_time(tg_m, em, int(2 * GRID_HZ))
    on_f, _ = onset_time(tg_f, ef, int(0.15 * GRID_HZ))
    ev_m = env_events(tg_m, em, int(2 * GRID_HZ))
    ev_f = env_events(tg_f, ef, int(0.15 * GRID_HZ))
    anchor = event_anchor(ev_m, ev_f)
    if anchor is not None:
        lag0, n_ev, ev_spread = anchor
        if on_m is not None and on_f is not None and \
                abs((on_m - on_f) - lag0) > 0.5:
            log.append(f"{trial}: onset anchor {on_m - on_f:+.2f} s differs "
                       f"from event anchor {lag0:+.2f} s (pre-protocol "
                       "magnetic disturbance suspected); event anchor used")
    elif on_m is not None and on_f is not None:
        lag0, n_ev, ev_spread = on_m - on_f, 0, np.nan
        log.append(f"{trial}: event matching failed "
                   f"({len(ev_m)} mag / {len(ev_f)} force events), "
                   "onset anchor used")
    else:
        log.append(f"{trial}: DROPPED, no anchor (events "
                   f"{len(ev_m)}/{len(ev_f)}, onsets {on_m}/{on_f})")
        return None
    r, lag = xcorr_refine(tg_m, em, tg_f, ef, lag0, LAG_SEARCH_S)
    if r < R_MIN:
        log.append(f"{trial}: DROPPED, alignment r={r:.3f} < {R_MIN}")
        return None
    # per-event timing residuals at the refined lag
    if n_ev:
        n = len(ev_f)
        s = int(np.argmin([np.abs(ev_m[i:i + n] - ev_f - lag).sum()
                           for i in range(len(ev_m) - n + 1)]))
        ev_resid = float(np.median(np.abs(ev_m[s:s + n] - ev_f - lag)))
    else:
        ev_resid = np.nan

    # final baseline: pre-onset magnetic window [pre-3.0, pre-0.5] s;
    # fallback (magnetic recording started mid-activity): quiet window
    # AFTER the force protocol ends; last resort: first 2 s (flagged).
    pre = on_m if on_m is not None else (ev_m[0] - 0.6 if len(ev_m) else 2.5)
    m0, m1 = max(pre - 3.0, t_m[0]), pre - 0.5
    sel = (t_m >= m0) & (t_m <= m1)
    if sel.sum() < fs_m:                      # need >= 1 s
        f_end = t_f[-1] + lag
        sel = (t_m >= f_end + 0.5) & (t_m <= f_end + 2.5)
        if sel.sum() >= fs_m:
            log.append(f"{trial}: pre-onset window too short, baseline = "
                       "post-protocol quiet window")
        else:
            sel = t_m <= t_m[0] + 2.0
            log.append(f"{trial}: WARNING no quiet window, baseline = "
                       "first 2 s (may be contaminated)")
    base = np.median(B[sel], axis=0)
    dB = B - base

    return dict(trial=trial, t_m=t_m, dB=dB, fs_m=fs_m, df=df, t_f=t_f,
                F=F, lag=lag, r=r, lag0=lag0, n_ev=n_ev, ev_resid=ev_resid)


# ----------------------------------------------------------------------------
# 3. per-cycle features
# ----------------------------------------------------------------------------
def extract_features(p, log):
    rows = []
    df, t_f, F = p["df"], p["t_f"], p["F"]
    lag = p["lag"]
    Fmag = np.linalg.norm(F, axis=1)
    stg = df["stage"].to_numpy()
    aFz = np.abs(F[:, 2])
    for cyc in sorted(df["cycle"].unique()):
        cm = df["cycle"].to_numpy() == cyc
        moving = cm & ((stg == "down") | (stg == "up"))
        if not moving.any():
            continue                       # cycle 0 = idle
        # press plateau: contiguous run around the cycle's peak |Fz|
        idx = np.flatnonzero(moving)       # contiguous (down then up)
        pk = idx[np.argmax(aFz[idx])]
        hi = aFz >= PLATEAU_FRAC * aFz[pk]
        i0 = pk
        while i0 - 1 >= idx[0] and hi[i0 - 1]:
            i0 -= 1
        i1 = pk
        while i1 + 1 <= idx[-1] and hi[i1 + 1]:
            i1 += 1
        plateau = np.zeros(len(df), bool)
        plateau[i0:i1 + 1] = True
        if plateau.sum() < 3:
            log.append(f"{p['trial']} cycle {cyc}: plateau <3 samples, "
                       "skipped")
            continue
        w0, w1 = t_f[plateau].min() + lag, t_f[plateau].max() + lag
        msel = (p["t_m"] >= w0) & (p["t_m"] <= w1)
        if msel.sum() < 3:
            log.append(f"{p['trial']} cycle {cyc}: no magnetic samples in "
                       "plateau window, skipped")
            continue
        row = dict(trial=p["trial"], cycle=int(cyc), stage="down")
        row.update({n: float(v) for n, v in
                    zip(CH_NAMES, p["dB"][msel].mean(axis=0))})
        row["Fx"], row["Fy"], row["Fz"] = F[plateau].mean(axis=0)
        row["peakF"] = float(Fmag[cm].max())
        rows.append(row)
    return rows


# ----------------------------------------------------------------------------
# 5. figure
# ----------------------------------------------------------------------------
def make_figure(rep, feat):
    fig = plt.figure(figsize=(7.05, 3.6))
    gs = fig.add_gridspec(4, 2, width_ratios=[1.55, 1.0])

    # ---- panel a: representative trial, 4 sensor rows ----
    t0 = rep["df"].loc[rep["df"]["stage"] == "down",
                       "elapsed_time_s"].iloc[0] + rep["lag"]
    t = rep["t_m"] - t0
    tf = rep["t_f"] + rep["lag"] - t0
    xlim = (-1.0, rep["t_f"][-1] + rep["lag"] - t0 + 0.5)

    # down-stage shading intervals (force time -> magnetic time)
    stg = rep["df"]["stage"].to_numpy()
    dn = (stg == "down").astype(int)
    edges = np.flatnonzero(np.diff(np.r_[0, dn, 0]))
    spans = [(tf[i0], tf[i1 - 1]) for i0, i1 in
             zip(edges[::2], edges[1::2])]

    win = (rep["t_m"] >= t0 + xlim[0]) & (rep["t_m"] <= t0 + xlim[1])
    ylo = rep["dB"][win].min() * 1.06
    yhi = rep["dB"][win].max() * 1.06
    axs = []
    for s in range(4):
        ax = fig.add_subplot(gs[s, 0], sharex=axs[0] if axs else None)
        axs.append(ax)
        for x0, x1 in spans:
            ax.axvspan(x0, x1, color=OI[7], alpha=0.18, lw=0, zorder=0)
        for a in ("x", "y", "z"):
            ch = CH_NAMES.index(f"dB{a}{s}")
            ax.plot(t, rep["dB"][:, ch], color=AX_COLORS[a], lw=0.7,
                    label=rf"$\Delta B_{a}$" if s == 0 else None)
        ax.text(0.006, 0.90, f"S{s}", transform=ax.transAxes, fontsize=7,
                fontweight="bold", va="top")
        ax.set_xlim(*xlim)
        ax.set_ylim(ylo, yhi)
        ax.axhline(0, color=OI[6], lw=0.5, alpha=0.35, zorder=1)
        if s < 3:
            plt.setp(ax.get_xticklabels(), visible=False)
            ax.tick_params(axis="x", length=0)
    axs[0].legend(ncol=3, frameon=False, loc="lower left",
                  bbox_to_anchor=(0.0, 1.00), borderpad=0.1,
                  handlelength=1.2, columnspacing=0.9, borderaxespad=0.1)
    axs[-1].set_xlabel("Time from first press onset (s)")
    fig.text(0.009, 0.5, r"$\Delta B$ ($\mu$T)", rotation=90,
             va="center", ha="center", fontsize=8)

    # ---- panel b: cycle-averaged down-plateau response per channel ----
    axb = fig.add_subplot(gs[:, 1])
    mu = feat[CH_NAMES].mean()
    sd = feat[CH_NAMES].std()
    xpos, cols, labels = [], [], []
    for s in range(4):
        for k, a in enumerate(("x", "y", "z")):
            xpos.append(s + (k - 1) * 0.22)
            cols.append(AX_COLORS[a])
            labels.append(f"dB{a}{s}")
    order = [CH_NAMES.index(l) for l in labels]
    axb.axhline(0, color=OI[6], lw=0.6, alpha=0.4, zorder=1)
    for xp, c, l in zip(xpos, cols, labels):
        axb.errorbar(xp, mu[l], yerr=sd[l], fmt="o", ms=4.2, color=c,
                     mec="white", mew=0.6, elinewidth=0.9, capsize=2,
                     capthick=0.9, zorder=3)
    for s in range(1, 4):
        axb.axvline(s - 0.5, color=OI[7], lw=0.5, alpha=0.35, zorder=0)
    axb.set_xticks(range(4))
    axb.set_xticklabels([f"S{s}" for s in range(4)])
    axb.set_xlim(-0.55, 3.55)
    axb.set_xlabel("Sensor")
    axb.set_ylabel(r"Down-plateau $\Delta B$ ($\mu$T)")
    n_cyc = len(feat)
    n_tr = feat["trial"].nunique()
    axb.set_title(f"mean $\\pm$ s.d., {n_cyc} cycles / {n_tr} trials",
                  fontsize=7, pad=3)
    handles = [plt.Line2D([], [], marker="o", ls="", ms=4.2,
                          color=AX_COLORS[a], mec="white", mew=0.6,
                          label=rf"$\Delta B_{a}$") for a in ("x", "y", "z")]
    axb.legend(handles=handles, frameon=False, loc="lower left",
               borderpad=0.2, handletextpad=0.4)

    for lab, x in (("a", 0.012), ("b", 0.615)):
        fig.text(x, 0.972, lab, fontsize=8, fontweight="bold")
    fig.tight_layout(pad=0.6, h_pad=0.35, w_pad=1.6,
                     rect=(0.022, 0, 1, 0.95))
    fig.savefig(FIG_BASE + ".png", dpi=300)
    fig.savefig(FIG_BASE + ".pdf")
    plt.close(fig)


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)
    log = []

    keep, dropped, offsets, med = pair_files()
    print(f"[pairing] {len(keep) + len(dropped)} pairs; filename offset "
          f"(mag_local - force_utc+8h): median {med:+.1f} s, "
          f"range {offsets.min():+.1f}..{offsets.max():+.1f} s, "
          f"spread {offsets.max() - offsets.min():.1f} s")
    for m, f, off in dropped:
        log.append(f"EXCLUDED pair {os.path.basename(m)} / "
                   f"{os.path.basename(f)}: offset {off:+.1f} s deviates "
                   f">{OFFSET_TOL_S:.0f} s from median {med:+.1f} s")
    print(f"[pairing] kept {len(keep)}, excluded {len(dropped)}")

    results, rows = [], []
    for m, f, _ in keep:
        p = process_pair(m, f, log)
        if p is None:
            continue
        results.append(p)
        rows.extend(extract_features(p, log))
        print(f"  {p['trial']}: lag {p['lag']:+7.3f} s "
              f"(anchor {p['lag0']:+7.3f}, {p['n_ev']} events, "
              f"resid {p['ev_resid']*1e3:5.0f} ms), align r = {p['r']:.4f}")

    feat = pd.DataFrame(rows, columns=["trial", "cycle", "stage",
                                       *CH_NAMES, "Fx", "Fy", "Fz", "peakF"])
    feat.to_csv(FEATURES_CSV, index=False)
    print(f"[features] {len(feat)} cycle rows from {len(results)} trials "
          f"-> {FEATURES_CSV}")

    rs = np.array([p["r"] for p in results])
    resid = np.array([p["ev_resid"] for p in results])
    print(f"[align] r: mean {rs.mean():.4f}, min {rs.min():.4f}, "
          f"max {rs.max():.4f}; per-event timing residual (median abs): "
          f"mean {np.nanmean(resid)*1e3:.0f} ms, "
          f"max {np.nanmax(resid)*1e3:.0f} ms")

    mu = feat[CH_NAMES].mean()
    big = mu.abs().idxmax()
    print(f"[amplitude] largest |mean down-plateau dB|: {big} = "
          f"{mu[big]:+.1f} +- {feat[big].std():.1f} (s.d.)")
    print("[amplitude] per-channel mean down-plateau dB:")
    for n in CH_NAMES:
        print(f"    {n}: {mu[n]:+8.2f} +- {feat[n].std():6.2f}")
    print(f"[force] plateau Fz: {feat['Fz'].mean():+.3f} +- "
          f"{feat['Fz'].std():.3f} N; per-cycle peak |F|: "
          f"{feat['peakF'].mean():.3f} +- {feat['peakF'].std():.3f} N "
          f"(max {feat['peakF'].max():.3f} N)")

    # representative trial: alignment r closest to the median r
    rep = results[int(np.argmin(np.abs(rs - np.median(rs))))]
    print(f"[figure] representative trial {rep['trial']} "
          f"(r = {rep['r']:.4f})")
    make_figure(rep, feat)
    print(f"[figure] -> {FIG_BASE}.png/.pdf")

    print("[anomalies]" if log else "[anomalies] none")
    for line in log:
        print("   ", line)


if __name__ == "__main__":
    main()

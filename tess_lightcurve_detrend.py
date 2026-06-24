#!/usr/bin/env python3
"""Standalone detrending for light curves produced by tess_voronoi_lc_pipeline.py."""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import pickle

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator

SAVE_FIGURE_PICKLES = False


def robust_wls(X, y, n_iter: int = 8, huber_k: float = 1.5):
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    good = np.all(np.isfinite(X), axis=1) & np.isfinite(y)
    X = X[good]
    y = y[good]
    if len(y) == 0:
        return np.zeros(X.shape[1], float)

    w = np.ones(len(y), float)
    beta = np.zeros(X.shape[1], float)

    for _ in range(max(1, int(n_iter))):
        sw = np.sqrt(w)
        Xw = X * sw[:, None]
        yw = y * sw
        beta, *_ = np.linalg.lstsq(Xw, yw, rcond=None)

        resid = y - X @ beta
        mad = np.nanmedian(np.abs(resid - np.nanmedian(resid)))
        sigma = 1.4826 * mad if np.isfinite(mad) and mad > 0 else np.nanstd(resid)
        if not np.isfinite(sigma) or sigma <= 0:
            break
        u = resid / (huber_k * sigma)
        w = np.where(np.abs(u) <= 1, 1.0, 1.0 / np.abs(u))

    return beta


def build_design_matrix(t, x, y, knot_spacing_days=np.inf):
    t = np.asarray(t, float)
    x = np.asarray(x, float)
    y = np.asarray(y, float)

    cols = [np.ones_like(t), x - np.nanmedian(x), y - np.nanmedian(y)]

    if np.isfinite(knot_spacing_days) and knot_spacing_days > 0:
        t0 = np.nanmin(t)
        t1 = np.nanmax(t)
        knots = np.arange(t0, t1 + knot_spacing_days, knot_spacing_days)
        for tk in knots[1:-1]:
            cols.append(np.maximum(0.0, t - tk))

    return np.column_stack(cols)


def find_matching_file(diag_root: Path, prefix: str, stem: str):
    matches = list(diag_root.rglob(f"{prefix}_{stem}.npy"))
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        return sorted(matches)[0]
    return None


def rms_ppm(x):
    x = np.asarray(x, float)
    return 1e6 * np.nanstd(x - np.nanmedian(x))






def normalize_lc_stem(stem: str):
    if stem.startswith("preferred_lc_"):
        return stem[len("preferred_lc_"):]
    return stem

def parse_stem_metadata(stem: str):
    """
    Parse stems like:
      s0019_sig_Sco_target1_matlab_pure
    Returns dict with sector, source, target, method.
    """
    m = re.match(r'^(s\d{4})_(.+)_(target\d+)_(.+)$', stem)
    if not m:
        return None
    return {
        "sector": m.group(1),
        "source": m.group(2),
        "target": m.group(3),
        "method": m.group(4),
    }


def simple_output_stem(stem: str):
    meta = parse_stem_metadata(stem)
    if meta is None:
        return stem
    return f'{meta["source"]}_{meta["target"]}_{meta["method"]}_{meta["sector"]}'


def combined_output_stem(stem: str):
    meta = parse_stem_metadata(stem)
    if meta is None:
        return stem
    return f'{meta["source"]}_{meta["target"]}_{meta["method"]}'


def load_sector_orbtable(csv_path: str | Path):
    p = Path(csv_path).expanduser()
    if not p.exists():
        raise FileNotFoundError(f"Orbital-frequency table not found: {p}")
    df = pd.read_csv(p)
    cols = {c.strip(): c for c in df.columns}
    need = ["sector", "mid_tjd", "freq_cyc/day"]
    missing = [c for c in need if c not in cols]
    if missing:
        raise ValueError(f"Orbital-frequency table missing columns: {missing}. Found: {list(df.columns)}")
    out = {}
    for _, r in df.iterrows():
        try:
            sec = int(r[cols["sector"]])
            mid_btjd = float(r[cols["mid_tjd"]])
            freq = float(r[cols["freq_cyc/day"]])
        except Exception:
            continue
        if np.isfinite(mid_btjd) and np.isfinite(freq):
            out[sec] = (mid_btjd, freq)
    if not out:
        raise ValueError(f"No valid rows found in orbital-frequency table: {p}")
    return out


def infer_sector_from_csv_stem(stem: str):
    m = re.search(r'(^|[_-])s(\d{4})([_-]|$)', stem.lower())
    if m:
        return int(m.group(2))
    m = re.search(r's(\d{4})', stem.lower())
    if m:
        return int(m.group(1))
    return None


def phase_template_detrend(flux, time_btjd, freq_cyc_per_day, phase_bin: float = 0.01):
    f = np.asarray(flux, float)
    t = np.asarray(time_btjd, float)
    ph = (t * float(freq_cyc_per_day)) % 1.0
    binw = float(phase_bin)
    edges = np.arange(0.0, 1.0 + binw, binw)
    centers = edges[:-1] + 0.5 * binw
    idx = np.digitize(ph, edges) - 1
    b = np.full_like(centers, np.nan, dtype=float)
    for i in range(len(centers)):
        m = idx == i
        if np.any(m):
            b[i] = np.nanmean(f[m])
    ok = np.isfinite(b)
    if ok.sum() < 4:
        return f.copy(), ph, None
    x = centers[ok]
    y = b[ok]
    x2 = np.concatenate([x - 1.0, x, x + 1.0])
    y2 = np.concatenate([y, y, y])
    o = np.argsort(x2)
    x2, y2 = x2[o], y2[o]
    pchip = PchipInterpolator(x2, y2, extrapolate=True)
    trend = pchip(ph)
    med = np.nanmedian(f)
    f_det = (f - trend) + med
    return f_det, ph, trend


def save_figure_with_optional_pickle(fig, outpng: Path, **savefig_kwargs):
    outpng = Path(outpng)
    fig.savefig(outpng, **savefig_kwargs)
    if SAVE_FIGURE_PICKLES:
        try:
            pkl = outpng.with_suffix(outpng.suffix + ".pickle")
            with open(pkl, "wb") as fh:
                pickle.dump(fig, fh, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as exc:
            print(f"  [WARN] Could not save pickled figure for {outpng.name}: {exc}")




def iterative_sigma_keep(y, sigma_thresh: float = 5.0, n_iter: int = 1):
    y = np.asarray(y, float)
    keep = np.isfinite(y).copy()
    sigma_thresh = float(sigma_thresh)
    if not np.isfinite(sigma_thresh) or sigma_thresh <= 0:
        return keep
    for _ in range(max(1, int(n_iter))):
        if keep.sum() < 3:
            break
        yy = y[keep]
        med = np.nanmedian(yy)
        mad = np.nanmedian(np.abs(yy - med))
        sigma = 1.4826 * mad if np.isfinite(mad) and mad > 0 else np.nanstd(yy)
        if not np.isfinite(sigma) or sigma <= 0:
            break
        new_keep = keep & (np.abs(y - med) <= sigma_thresh * sigma)
        if np.array_equal(new_keep, keep):
            break
        keep = new_keep
    return keep


def build_binned_pchip_trend(t, y, bin_days: float = 0.25, stat: str = "median",
                             min_points: int = 3, sigma_clip: float = 0.0,
                             sigma_clip_iters: int = 1):
    t = np.asarray(t, float)
    y = np.asarray(y, float)
    full_m = np.isfinite(t) & np.isfinite(y)
    full_trend = np.full_like(np.asarray(y, float), np.nan)

    tt = t[full_m]
    yy = y[full_m]
    if tt.size < max(3, int(min_points)):
        return None, full_trend

    stat = str(stat).lower().strip()
    if not np.isfinite(bin_days) or float(bin_days) <= 0:
        return None, full_trend

    t0 = np.nanmin(tt)
    t1 = np.nanmax(tt)
    edges = np.arange(t0, t1 + float(bin_days), float(bin_days))
    if edges.size < 2:
        edges = np.array([t0, t1 + float(bin_days)], float)

    bt = []
    by = []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        mm = (tt >= lo) & (tt <= hi) if i == len(edges) - 2 else (tt >= lo) & (tt < hi)
        if mm.sum() < int(min_points):
            continue
        tbin = tt[mm].copy()
        ybin = yy[mm].copy()

        if float(sigma_clip) > 0:
            keep = iterative_sigma_keep(ybin, sigma_thresh=float(sigma_clip), n_iter=int(sigma_clip_iters))
            tbin = tbin[keep]
            ybin = ybin[keep]
        if ybin.size < int(min_points):
            continue

        bt.append(np.nanmedian(tbin))
        by.append(np.nanmean(ybin) if stat == "mean" else np.nanmedian(ybin))

    if len(bt) < 3:
        return None, full_trend

    bt = np.asarray(bt, float)
    by = np.asarray(by, float)
    order = np.argsort(bt)
    bt = bt[order]
    by = by[order]

    ubt, inv = np.unique(bt, return_inverse=True)
    if len(ubt) != len(bt):
        by2 = np.full(len(ubt), np.nan)
        for i in range(len(ubt)):
            by2[i] = np.nanmedian(by[inv == i])
        bt = ubt
        by = by2

    if len(bt) < 3:
        return None, full_trend

    pchip = PchipInterpolator(bt, by, extrapolate=True)
    full_trend[full_m] = pchip(tt)
    return pchip, full_trend


def save_detrend_plot(t, flux_raw, flux_decor, flux_final, outpng: Path, title: str):
    fig, axes = plt.subplots(3, 1, figsize=(10.5, 7.0), sharex=True)
    series = [
        (flux_raw, "Raw"),
        (flux_decor, "Decorrelated"),
        (flux_final, "Final detrended"),
    ]
    for ax, (y, lab) in zip(axes, series):
        ax.plot(t, y, ".", ms=2.2)
        ax.set_ylabel(lab)
        ax.grid(alpha=0.25)
    axes[-1].set_xlabel("Time [BTJD]")
    fig.suptitle(title, y=0.995)
    fig.tight_layout()
    save_figure_with_optional_pickle(fig, outpng, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_combined_plot(df_all: pd.DataFrame, outpng: Path, title: str):
    fig, ax = plt.subplots(figsize=(10.5, 4.0))
    ax.plot(df_all["time_btjd"].to_numpy(float), df_all["flux_detrend_rel"].to_numpy(float), ".", ms=2.2)
    ax.set_xlabel("Time [BTJD]")
    ax.set_ylabel("Final detrended flux")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    save_figure_with_optional_pickle(fig, outpng, dpi=180, bbox_inches="tight")
    plt.close(fig)

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Detrend raw light curves produced by tess_voronoi_lc_pipeline.py.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--lightcurve-dir", type=str, default="LC_products_multi",
                   help="Directory containing raw light-curve CSV files from the extraction pipeline.")
    p.add_argument("--diagnostics-dir", type=str, default="LC_products_multi",
                   help="Directory containing centroids/background/aperture diagnostics from the extraction pipeline.")
    p.add_argument("--output-dir", type=str, default="LC_products_multi",
                   help="Directory for detrended light-curve CSV files.")
    p.add_argument("--prefix", type=str, default="detrended_",
                   help="Prefix added to all detrended output filenames.")
    p.add_argument("--recursive", action="store_true",
                   help="Search for CSV files recursively under --lightcurve-dir.")
    p.add_argument("--pattern", type=str, default="*.csv",
                   help="Filename pattern for input raw light-curve CSV files.")
    p.add_argument("--use-background", action="store_true",
                   help="Include the saved background series as a decorrelation regressor when available.")
    p.add_argument("--knot-spacing-days", type=float, default=np.inf,
                   help="Optional spline-like time basis spacing in days; inf disables time-basis terms.")
    p.add_argument("--robust-iters", type=int, default=8,
                   help="Number of robust weighted least-squares iterations.")
    p.add_argument("--huber-k", type=float, default=1.5,
                   help="Huber tuning constant for the robust fit.")
    p.add_argument("--use-pchip-highpass", action="store_true",
                   help="Apply an additional PCHIP high-pass step after centroid/background decorrelation.")
    p.add_argument("--pchip-knot-spacing", type=float, default=0.5,
                   help="PCHIP knot spacing in days when --use-pchip-highpass is enabled.")
    p.add_argument("--pre-model-pchip", action="store_true",
                   help="Fit and subtract a binned PCHIP variability model before decorrelation, then add it back afterward.")
    p.add_argument("--pre-model-bin-days", type=float, default=0.25,
                   help="Time-bin size in days for the optional pre-model PCHIP variability fit.")
    p.add_argument("--pre-model-stat", choices=["median", "mean"], default="median",
                   help="Statistic used within each time bin for the pre-model PCHIP fit.")
    p.add_argument("--pre-model-min-points", type=int, default=3,
                   help="Minimum number of points required in a bin before it is used in the pre-model PCHIP fit.")
    p.add_argument("--pre-model-sigma-clip", type=float, default=0.0,
                   help="Optional sigma-clipping threshold inside each pre-model time bin; <=0 disables clipping.")
    p.add_argument("--pre-model-sigma-iters", type=int, default=1,
                   help="Number of sigma-clipping iterations inside each pre-model time bin.")
    p.add_argument("--clip-residuals-before-detrend", action="store_true",
                   help="Sigma-clip residuals before solving the centroid/background decorrelation fit.")
    p.add_argument("--clip-residuals-sigma", type=float, default=5.0,
                   help="Sigma threshold for residual clipping before decorrelation.")
    p.add_argument("--clip-residuals-iters", type=int, default=1,
                   help="Number of iterations for residual clipping before decorrelation.")
    p.add_argument("--gap-days", type=float, default=0.5,
                   help="Gap threshold reserved for chunk-wise workflows and compatibility with the GUI.")
    p.add_argument("--save-figure-pickles", action="store_true",
                   help="Save pickled Matplotlib figure objects alongside PNG plots.")
    p.add_argument("--skip-xybg-decorrelation", action="store_true",
                   help="Skip centroid/background decorrelation entirely, but still allow orbital and optional PCHIP corrections.")
    p.add_argument("--apply-orbital-phase-template", action="store_true",
                   help="Apply the MATLAB-style orbital phase-template correction in the detrending stage.")
    p.add_argument("--orbtable", type=str, default="",
                   help="Path to tess_sector_orbfreq_midpoints.csv used for orbital phase-template correction.")
    p.add_argument("--phase-bin", type=float, default=0.01,
                   help="Phase bin width for orbital phase-template subtraction.")
    p.add_argument("--no-combine-sectors", action="store_true",
                   help="Do not write combined multi-sector CSV/PNG products.")
    return p.parse_args(argv)


def main(argv=None):
    global SAVE_FIGURE_PICKLES
    args = parse_args(argv)
    SAVE_FIGURE_PICKLES = bool(getattr(args, "save_figure_pickles", False))

    lc_root = Path(args.lightcurve_dir).expanduser().resolve()

    # Path policy:
    # - If the user leaves --diagnostics-dir / --output-dir at their defaults,
    #   place them alongside --lightcurve-dir (i.e. under lc_root.parent).
    # - If the user explicitly supplies a relative path such as
    #   "LC_products/lightcurves_detrended", respect it relative to the
    #   current working directory instead of prepending lc_root.parent again.
    default_diag = "LC_products_multi"
    default_out = "LC_products_multi"

    diag_candidate = Path(args.diagnostics_dir).expanduser()
    if diag_candidate.is_absolute():
        diag_root = diag_candidate.resolve()
    elif args.diagnostics_dir == default_diag:
        diag_root = (lc_root.parent / diag_candidate).resolve()
    else:
        diag_root = diag_candidate.resolve()

    out_candidate = Path(args.output_dir).expanduser()
    if out_candidate.is_absolute():
        out_root = out_candidate.resolve()
    elif args.output_dir == default_out:
        out_root = (lc_root.parent / out_candidate).resolve()
    else:
        out_root = out_candidate.resolve()

    out_root.mkdir(parents=True, exist_ok=True)

    sector_orb = None
    if getattr(args, "apply_orbital_phase_template", False):
        if not getattr(args, "orbtable", ""):
            raise ValueError("--apply-orbital-phase-template requires --orbtable.")
        sector_orb = load_sector_orbtable(args.orbtable)

    if not lc_root.exists():
        raise FileNotFoundError(f"--lightcurve-dir not found: {lc_root}")
    if not diag_root.exists():
        raise FileNotFoundError(f"--diagnostics-dir not found: {diag_root}")

    csv_files = sorted(lc_root.rglob(args.pattern) if args.recursive else lc_root.glob(args.pattern))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {lc_root} matching {args.pattern!r}")

    print(f"Raw light curves to detrend: {len(csv_files)}")
    combined_rows = {}

    for csv_path in csv_files:
        stem = csv_path.stem
        stem_core = normalize_lc_stem(stem)
        if stem.startswith(str(args.prefix)) or "_detrend" in stem or "_combined_detrend" in stem or "_combined" in stem and stem.startswith(str(args.prefix)):
            print(f"Skipping already-detrended file: {csv_path.name}")
            continue
        print(f"\nProcessing: {csv_path.name}")

        df = pd.read_csv(csv_path)
        if "time_btjd" not in df.columns:
            print(f"Skipping non-light-curve CSV: {csv_path.name}")
            continue
        flux_col = (
            "flux_rel" if "flux_rel" in df.columns else
            ("flux_detrended_rel" if "flux_detrended_rel" in df.columns else
             ("flux_medscaled" if "flux_medscaled" in df.columns else None))
        )
        if flux_col is None:
            print(f"Skipping CSV without usable flux column: {csv_path.name}")
            continue

        t = np.asarray(df["time_btjd"], float)
        flux = np.asarray(df[flux_col], float)

        crow_file = find_matching_file(diag_root, "centroid_row", stem_core)
        ccol_file = find_matching_file(diag_root, "centroid_col", stem_core)
        if crow_file is None or ccol_file is None:
            raise FileNotFoundError(f"Could not find centroid files for {stem_core} under {diag_root}")

        crow = np.load(crow_file)
        ccol = np.load(ccol_file)

        bg = None
        if args.use_background:
            bg_file = find_matching_file(diag_root, "background", stem_core)
            if bg_file is not None:
                bg = np.load(bg_file)

        n = min(len(t), len(flux), len(crow), len(ccol), len(bg) if bg is not None else 10**12)
        t = t[:n]
        flux = flux[:n]
        crow = crow[:n]
        ccol = ccol[:n]
        if bg is not None:
            bg = bg[:n]

        mask = np.isfinite(t) & np.isfinite(flux) & np.isfinite(crow) & np.isfinite(ccol)
        if bg is not None:
            mask &= np.isfinite(bg)

        t = t[mask]
        flux = flux[mask]
        crow = crow[mask]
        ccol = ccol[mask]
        if bg is not None:
            bg = bg[mask]

        if len(t) == 0:
            print(f"Skipping empty/fully-invalid light curve after masking: {csv_path.name}")
            continue
        if len(t) < 3:
            print(f"Skipping too-short light curve after masking: {csv_path.name}")
            continue

        flux_for_fit = flux.copy()
        variability_model = np.full_like(flux, np.nan)
        flux_variability_resid = flux.copy()
        if getattr(args, "pre_model_pchip", False):
            _pchip0, trend0 = build_binned_pchip_trend(
                t, flux_for_fit,
                bin_days=float(args.pre_model_bin_days),
                stat=str(args.pre_model_stat),
                min_points=int(args.pre_model_min_points),
                sigma_clip=float(args.pre_model_sigma_clip),
                sigma_clip_iters=int(args.pre_model_sigma_iters),
            )
            if np.any(np.isfinite(trend0)):
                variability_model = np.asarray(trend0, float)
                flux_variability_resid = flux_for_fit - variability_model
            else:
                flux_variability_resid = flux_for_fit.copy()
        else:
            flux_variability_resid = flux_for_fit.copy()

        if getattr(args, "skip_xybg_decorrelation", False):
            flux_decor = flux_variability_resid.copy()
        else:
            X = build_design_matrix(t, ccol, crow, knot_spacing_days=args.knot_spacing_days)
            if bg is not None:
                dbg = bg - np.nanmedian(bg)
                X = np.column_stack([X, dbg])

            y = flux_variability_resid - np.nanmedian(flux_variability_resid)
            fit_keep = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
            if getattr(args, "clip_residuals_before_detrend", False):
                fit_keep &= iterative_sigma_keep(
                    y,
                    sigma_thresh=float(args.clip_residuals_sigma),
                    n_iter=int(args.clip_residuals_iters),
                )
            if fit_keep.sum() >= max(5, X.shape[1] + 1):
                beta = robust_wls(X[fit_keep], y[fit_keep], n_iter=args.robust_iters, huber_k=args.huber_k)
            else:
                beta = robust_wls(X, y, n_iter=args.robust_iters, huber_k=args.huber_k)
            flux_decor = y - (X @ beta) + np.nanmedian(flux_variability_resid)

        if np.any(np.isfinite(variability_model)):
            flux_decor = flux_decor + variability_model - np.nanmedian(variability_model)

        orbital_phase = np.full_like(flux_decor, np.nan)
        orbital_trend = np.full_like(flux_decor, np.nan)
        flux_orbital = flux_decor.copy()
        if getattr(args, "apply_orbital_phase_template", False):
            sec = infer_sector_from_csv_stem(stem_core)
            if sec is not None and sector_orb is not None and sec in sector_orb:
                _mid_btjd, freq_cpd = sector_orb[sec]
                flux_orbital, orbital_phase, orbital_trend = phase_template_detrend(
                    flux_decor, t, freq_cpd, phase_bin=float(args.phase_bin)
                )
            else:
                print(f"  [WARN] Could not apply orbital phase-template correction for {csv_path.name}: sector not found in orbtable.")
                flux_orbital = flux_decor.copy()

        trend = np.full_like(flux_orbital, np.nan)
        flux_final = flux_orbital.copy()

        if args.use_pchip_highpass:
            knots = np.arange(np.nanmin(t), np.nanmax(t) + args.pchip_knot_spacing, args.pchip_knot_spacing)
            knot_t, knot_y = [], []
            half = 0.5 * args.pchip_knot_spacing
            for tk in knots:
                m = (t >= tk - half) & (t < tk + half)
                if np.any(m):
                    knot_t.append(tk)
                    knot_y.append(np.nanmedian(flux_orbital[m]))
            if len(knot_t) >= 3:
                pchip = PchipInterpolator(np.asarray(knot_t), np.asarray(knot_y))
                trend = pchip(t)
                flux_final = flux_orbital - trend + np.nanmedian(trend)

        out_df = pd.DataFrame({
            "time_btjd": t,
            "flux_rel": flux,
            "flux_variability_resid_rel": flux_variability_resid,
            "flux_detrend_rel": flux_final,
            "flux_decor_only_rel": flux_decor,
            "centroid_col": ccol,
            "centroid_row": crow,
        })
        if bg is not None:
            out_df["background"] = bg
        if np.any(np.isfinite(variability_model)):
            out_df["pre_model_pchip_trend"] = variability_model
        if np.any(np.isfinite(orbital_trend)):
            out_df["orbital_phase"] = orbital_phase
            out_df["orbital_phase_trend"] = orbital_trend
            out_df["flux_orbital_corrected_rel"] = flux_orbital
        if np.any(np.isfinite(trend)):
            out_df["pchip_trend"] = trend

        for col in df.columns:
            if col not in out_df.columns and len(df[col]) >= n:
                try:
                    out_df[col] = np.asarray(df[col], object)[:n][mask]
                except Exception:
                    pass

        simple_stem = simple_output_stem(stem_core)
        out_path = out_root / f"{args.prefix}{simple_stem}.csv"
        out_df.to_csv(out_path, index=False)

        plot_path = out_root / f"{args.prefix}{simple_stem}.png"
        save_detrend_plot(
            t,
            flux,
            flux_decor,
            flux_final,
            plot_path,
            f"{simple_stem} detrending",
        )

        comb_key = combined_output_stem(stem_core)
        combined_rows.setdefault(comb_key, []).append(out_df.copy())

        print(f"  RMS raw     : {rms_ppm(flux):.1f} ppm")
        print(f"  RMS decor   : {rms_ppm(flux_decor):.1f} ppm")
        if getattr(args, "apply_orbital_phase_template", False):
            print(f"  RMS orbital : {rms_ppm(flux_orbital):.1f} ppm")
        if args.use_pchip_highpass:
            print(f"  RMS detrend : {rms_ppm(flux_final):.1f} ppm")
        print(f"  Wrote       : {out_path.name}")
        print(f"  Wrote plot  : {plot_path.name}")

    if not args.no_combine_sectors:
        for comb_key, frames in combined_rows.items():
            if len(frames) == 0:
                continue
            df_all = pd.concat(frames, ignore_index=True).sort_values("time_btjd").reset_index(drop=True)
            comb_csv = out_root / f"{args.prefix}{comb_key}_combined.csv"
            df_all.to_csv(comb_csv, index=False)
            comb_png = out_root / f"{args.prefix}{comb_key}_combined.png"
            save_combined_plot(df_all, comb_png, f"{comb_key} combined detrended light curve")
            print(f"Combined      : {comb_csv.name}")
            print(f"Combined plot : {comb_png.name}")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

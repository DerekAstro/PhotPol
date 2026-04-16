#!/usr/bin/env python3
"""Standalone detrending/leveling for light curves produced by the TESS extraction pipelines."""

from __future__ import annotations

import argparse
from pathlib import Path
import re

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator


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


def segment_indices_by_gaps(time_btjd, gap_days: float = 0.5):
    t = np.asarray(time_btjd, float)
    good = np.isfinite(t)
    idx = np.where(good)[0]
    if idx.size == 0:
        return []
    tt = t[idx]
    breaks = np.where(np.diff(tt) > float(gap_days))[0]
    starts = np.r_[0, breaks + 1]
    ends = np.r_[breaks + 1, tt.size]
    return [idx[s:e] for s, e in zip(starts, ends) if (e - s) > 0]


def median_level_by_segment(time_btjd, flux_rel, gap_days: float = 0.5):
    f = np.asarray(flux_rel, float).copy()
    for ii in segment_indices_by_gaps(time_btjd, gap_days=gap_days):
        med = np.nanmedian(f[ii])
        if np.isfinite(med) and med != 0:
            f[ii] /= med
    return f






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
    fig.savefig(outpng, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_combined_plot(df_all: pd.DataFrame, outpng: Path, title: str):
    fig, ax = plt.subplots(figsize=(10.5, 4.0))
    ax.plot(df_all["time_btjd"].to_numpy(float), df_all["flux_detrend_rel"].to_numpy(float), ".", ms=2.2)
    ax.set_xlabel("Time [BTJD]")
    ax.set_ylabel("Final detrended flux")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(outpng, dpi=180, bbox_inches="tight")
    plt.close(fig)

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Detrend or gap-level raw light curves produced by the TESS extraction pipelines.",
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
    p.add_argument("--skip-xybg-decorrelation", action="store_true",
                   help="Skip centroid/background decorrelation, but still level gap-separated chunks, optionally apply PCHIP high-pass, and combine sectors.")
    p.add_argument("--gap-days", type=float, default=0.5,
                   help="Gap threshold in days used for chunk-wise median leveling.")
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
    p.add_argument("--no-combine-sectors", action="store_true",
                   help="Do not write combined multi-sector CSV/PNG products.")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

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

        flux_leveled = median_level_by_segment(t, flux, gap_days=args.gap_days)

        if args.skip_xybg_decorrelation:
            flux_decor = flux_leveled.copy()
        else:
            X = build_design_matrix(t, ccol, crow, knot_spacing_days=args.knot_spacing_days)
            labels = ["offset", "dx", "dy"]
            if np.isfinite(args.knot_spacing_days) and args.knot_spacing_days > 0:
                t0 = np.nanmin(t)
                t1 = np.nanmax(t)
                knots = np.arange(t0, t1 + args.knot_spacing_days, args.knot_spacing_days)
                labels.extend([f"hinge_{i}" for i, _ in enumerate(knots[1:-1], start=1)])

            if bg is not None:
                dbg = bg - np.nanmedian(bg)
                X = np.column_stack([X, dbg])
                labels.append("bg")

            y = flux - np.nanmedian(flux)
            beta = robust_wls(X, y, n_iter=args.robust_iters, huber_k=args.huber_k)
            flux_decor = y - (X @ beta) + np.nanmedian(flux)
            flux_decor = median_level_by_segment(t, flux_decor, gap_days=args.gap_days)

        trend = np.full_like(flux_decor, np.nan)
        flux_final = flux_decor.copy()

        if args.use_pchip_highpass:
            knots = np.arange(np.nanmin(t), np.nanmax(t) + args.pchip_knot_spacing, args.pchip_knot_spacing)
            knot_t, knot_y = [], []
            half = 0.5 * args.pchip_knot_spacing
            for tk in knots:
                m = (t >= tk - half) & (t < tk + half)
                if np.any(m):
                    knot_t.append(tk)
                    knot_y.append(np.nanmedian(flux_decor[m]))
            if len(knot_t) >= 3:
                pchip = PchipInterpolator(np.asarray(knot_t), np.asarray(knot_y))
                trend = pchip(t)
                flux_final = flux_decor - trend + np.nanmedian(trend)

        out_df = pd.DataFrame({
            "time_btjd": t,
            "flux_rel": flux,
            "flux_leveled_rel": flux_leveled,
            "flux_detrend_rel": flux_final,
            "flux_decor_only_rel": flux_decor,
            "centroid_col": ccol,
            "centroid_row": crow,
        })
        if bg is not None:
            out_df["background"] = bg
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
        print(f"  RMS leveled : {rms_ppm(flux_leveled):.1f} ppm")
        print(f"  RMS decor   : {rms_ppm(flux_decor):.1f} ppm")
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

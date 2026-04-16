
#!/usr/bin/env python3
"""
Convert standard SPOC light-curve FITS products (*lc.fits) into CSV files that can be
used by the user's detrending and TESS-guided polarimetry-analysis workflows.

Outputs per input file:
- <base>.csv with columns including:
    time_btjd
    flux_rel              (SAP_FLUX normalized by its median, if available)
    flux_detrended_rel    (PDCSAP_FLUX normalized by its median, if available)
    flux_medscaled        (same as flux_detrended_rel if available, else flux_rel)
    sap_flux
    sap_flux_err
    pdcsap_flux
    pdcsap_flux_err
    quality
    cadence_num
    sector
    ticid
    source_file

Optional combined output per TIC:
- spoc_combined_tic<TIC>.csv

These CSVs are designed to be readable by:
- tess_guided_analysis.py
- tess_lightcurve_detrend_simpler_v3.py (if you want to compare detrending approaches)

Examples
--------
Convert all lc.fits files in the current directory:
    python spoc_lightcurve_converter.py --input-dir . --pattern "*lc.fits"

Convert recursively and also make combined TIC files:
    python spoc_lightcurve_converter.py --input-dir . --recursive --combine-by-tic

Keep all cadences, even with nonzero QUALITY:
    python spoc_lightcurve_converter.py --input-dir . --keep-all-quality
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re
from collections import defaultdict

import numpy as np
import pandas as pd
from astropy.io import fits


def median_normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, float)
    med = np.nanmedian(x)
    if not np.isfinite(med) or med == 0:
        return x.copy()
    return x / med


def first_present(header, keys, default=None):
    for k in keys:
        if k in header:
            try:
                val = header[k]
                if val not in (None, ""):
                    return val
            except Exception:
                pass
    return default


def sanitize_token(s: str) -> str:
    s = str(s).strip()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^0-9A-Za-z_+\-\.]", "_", s)
    s = re.sub(r"_+", "_", s)
    return s.strip("_")


def read_spoc_lc_fits(path: Path, keep_all_quality: bool = False) -> pd.DataFrame:
    with fits.open(path, memmap=True) as hdul:
        # Standard SPOC LC files store table in HDU 1.
        data = hdul[1].data
        hdr0 = hdul[0].header
        hdr1 = hdul[1].header

        names = set(data.names)

        if "TIME" not in names:
            raise KeyError(f"{path.name}: missing TIME column")

        time_btjd = np.asarray(data["TIME"], dtype=float)

        quality = np.asarray(data["QUALITY"], dtype=int) if "QUALITY" in names else np.zeros(len(time_btjd), dtype=int)
        cadence_num = np.asarray(data["CADENCENO"]) if "CADENCENO" in names else np.arange(len(time_btjd))

        sap_flux = np.asarray(data["SAP_FLUX"], dtype=float) if "SAP_FLUX" in names else np.full(len(time_btjd), np.nan)
        sap_flux_err = np.asarray(data["SAP_FLUX_ERR"], dtype=float) if "SAP_FLUX_ERR" in names else np.full(len(time_btjd), np.nan)

        pdcsap_flux = np.asarray(data["PDCSAP_FLUX"], dtype=float) if "PDCSAP_FLUX" in names else np.full(len(time_btjd), np.nan)
        pdcsap_flux_err = np.asarray(data["PDCSAP_FLUX_ERR"], dtype=float) if "PDCSAP_FLUX_ERR" in names else np.full(len(time_btjd), np.nan)

        sector = first_present(hdr0, ["SECTOR"], default=first_present(hdr1, ["SECTOR"], default=np.nan))
        ticid = first_present(hdr0, ["TICID", "TARGETID", "OBJECT"], default=first_present(hdr1, ["TICID", "TARGETID", "OBJECT"], default="unknown"))

    if not keep_all_quality:
        keep = np.isfinite(time_btjd) & (quality == 0)
    else:
        keep = np.isfinite(time_btjd)

    # Require at least one finite flux stream
    keep &= (
        np.isfinite(sap_flux) |
        np.isfinite(pdcsap_flux)
    )

    time_btjd = time_btjd[keep]
    quality = quality[keep]
    cadence_num = cadence_num[keep]
    sap_flux = sap_flux[keep]
    sap_flux_err = sap_flux_err[keep]
    pdcsap_flux = pdcsap_flux[keep]
    pdcsap_flux_err = pdcsap_flux_err[keep]

    flux_rel = median_normalize(sap_flux)
    flux_detrended_rel = median_normalize(pdcsap_flux)

    # Prefer PDCSAP as the "main" already-detrended light curve if present.
    if np.isfinite(flux_detrended_rel).any():
        flux_medscaled = flux_detrended_rel.copy()
    else:
        flux_medscaled = flux_rel.copy()

    df = pd.DataFrame({
        "time_btjd": time_btjd,
        "flux_rel": flux_rel,
        "flux_detrended_rel": flux_detrended_rel,
        "flux_medscaled": flux_medscaled,
        "sap_flux": sap_flux,
        "sap_flux_err": sap_flux_err,
        "pdcsap_flux": pdcsap_flux,
        "pdcsap_flux_err": pdcsap_flux_err,
        "quality": quality,
        "cadence_num": cadence_num,
        "sector": sector,
        "ticid": ticid,
        "source_file": path.name,
    })

    return df


def output_basename(df: pd.DataFrame, path: Path) -> str:
    tic = sanitize_token(df["ticid"].iloc[0]) if len(df) else "unknown"
    sec = df["sector"].iloc[0] if len(df) else np.nan
    sec_tag = f"s{int(sec):04d}" if np.isfinite(sec) else "sXXXX"
    return f"spoc_{sec_tag}_{tic}"


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Convert standard SPOC *lc.fits files to CSV for downstream detrending / guided analysis.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input-dir", type=str, default=".", help="Directory containing SPOC *lc.fits files.")
    p.add_argument("--pattern", type=str, default="*lc.fits", help="Glob pattern for input FITS files.")
    p.add_argument("--recursive", action="store_true", help="Search recursively under --input-dir.")
    p.add_argument("--output-dir", type=str, default="spoc_csvs", help="Directory for converted CSV files.")
    p.add_argument("--keep-all-quality", action="store_true", help="Keep nonzero-QUALITY cadences instead of filtering to QUALITY==0.")
    p.add_argument("--combine-by-tic", action="store_true", help="Also write a combined CSV per TIC after per-file conversion.")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    in_dir = Path(args.input_dir).expanduser().resolve()
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    fits_files = sorted(in_dir.rglob(args.pattern) if args.recursive else in_dir.glob(args.pattern))
    if not fits_files:
        raise FileNotFoundError(f"No files found in {in_dir} matching {args.pattern!r}")

    print(f"SPOC lc.fits files found: {len(fits_files)}")

    by_tic = defaultdict(list)

    for fits_path in fits_files:
        print(f"\nConverting: {fits_path.name}")
        df = read_spoc_lc_fits(fits_path, keep_all_quality=args.keep_all_quality)

        if len(df) == 0:
            print("  [WARN] No usable cadences after filtering; skipping.")
            continue

        base = output_basename(df, fits_path)
        out_csv = out_dir / f"{base}.csv"
        df.to_csv(out_csv, index=False)

        tic = sanitize_token(df["ticid"].iloc[0])
        by_tic[tic].append(df)

        print(f"  Rows kept    : {len(df)}")
        print(f"  Sector       : {df['sector'].iloc[0]}")
        print(f"  TIC          : {df['ticid'].iloc[0]}")
        print(f"  Wrote        : {out_csv.name}")

    if args.combine_by_tic:
        print("\nWriting combined TIC files...")
        for tic, dfs in sorted(by_tic.items()):
            if not dfs:
                continue
            df_all = pd.concat(dfs, ignore_index=True)
            df_all = df_all.sort_values("time_btjd").reset_index(drop=True)
            out_csv = out_dir / f"spoc_combined_tic{tic}.csv"
            df_all.to_csv(out_csv, index=False)
            print(f"  TIC {tic}: {out_csv.name} ({len(df_all)} rows)")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

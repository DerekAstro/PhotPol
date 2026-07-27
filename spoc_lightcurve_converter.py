#!/usr/bin/env python3
"""
Convert standard SPOC light-curve FITS products (``*lc.fits``) to CSV files for
PhotPol detrending and guided photometry/polarimetry analysis.

The converter remains a fully standalone command-line program. It can process a
single FITS file, a directory, or one or more glob patterns; optionally write
per-sector CSVs; combine multiple sectors by TIC; prefer the shortest cadence
when duplicate products exist for one sector; and write a JSON manifest that a
GUI can use to locate the correct analysis CSV.

Examples
--------
Single file::

    python spoc_lightcurve_converter.py \
        --input "/path with spaces/tess...-lc.fits" \
        --output-dir converted

All sectors for one target, recursively, with a combined CSV::

    python spoc_lightcurve_converter.py \
        --input-dir "/path/to/downloads" \
        --pattern "*lc.fits" \
        --recursive \
        --combine-by-tic \
        --cadence-policy shortest \
        --normalization sigma-clipped-median \
        --output-dir converted

The legacy ``--input-dir``, ``--pattern``, ``--recursive``,
``--keep-all-quality``, and ``--combine-by-tic`` options remain supported.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import glob
import json
from pathlib import Path
import re
from typing import Iterable

import numpy as np
import pandas as pd

try:
    from astropy.io import fits
except Exception as exc:  # pragma: no cover - exercised on user systems
    fits = None
    _ASTROPY_IMPORT_ERROR = exc
else:
    _ASTROPY_IMPORT_ERROR = None


DEFAULT_PATTERN = "*lc.fits"
NORMALIZATION_CHOICES = ("sigma-clipped-median", "median", "none")
FLUX_CHOICES = ("auto", "pdcsap", "sap")
QUALITY_CHOICES = ("good", "all")
CADENCE_POLICIES = ("shortest", "longest", "all")


def is_tess_target_pixel_path(path: str | Path) -> bool:
    """Identify standard TESS target-pixel product names."""
    name = Path(str(path)).name.lower()
    return name.endswith(("-tp.fits", "-tp.fits.gz", "_tp.fits", "_tp.fits.gz"))


def target_pixel_error_message(path: str | Path) -> str:
    name = Path(str(path)).name
    return (
        f"{name}: this is a TESS target-pixel file (*-tp.fits), not a SPOC "
        "light-curve file. Choose the corresponding *-lc.fits or *-fast-lc.fits "
        "product, or process the target-pixel file through the PhotPol extractor first."
    )


def sanitize_token(value: object) -> str:
    text = str(value).strip()
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"[^0-9A-Za-z_+\-.]", "_", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_") or "unknown"


def first_present(header, keys: Iterable[str], default=None):
    for key in keys:
        try:
            value = header.get(key)
        except Exception:
            continue
        if value not in (None, ""):
            return value
    return default


def robust_sigma_clipped_median(
    values: np.ndarray,
    sigma: float = 4.0,
    max_iters: int = 5,
) -> float:
    arr = np.asarray(values, dtype=float)
    keep = np.isfinite(arr)
    if keep.sum() == 0:
        return np.nan

    sigma = float(sigma)
    max_iters = max(1, int(max_iters))
    for _ in range(max_iters):
        sample = arr[keep]
        if sample.size == 0:
            return np.nan
        med = float(np.nanmedian(sample))
        mad = float(np.nanmedian(np.abs(sample - med)))
        scale = 1.4826 * mad if np.isfinite(mad) and mad > 0 else float(np.nanstd(sample))
        if not np.isfinite(scale) or scale <= 0 or not np.isfinite(sigma) or sigma <= 0:
            break
        new_keep = np.isfinite(arr) & (np.abs(arr - med) <= sigma * scale)
        if np.array_equal(new_keep, keep):
            break
        keep = new_keep

    return float(np.nanmedian(arr[keep])) if keep.any() else np.nan


def normalization_factor(
    values: np.ndarray,
    method: str = "sigma-clipped-median",
    sigma: float = 4.0,
    max_iters: int = 5,
) -> float:
    method = str(method).strip().lower()
    arr = np.asarray(values, dtype=float)
    if method == "none":
        return 1.0
    if method == "median":
        factor = float(np.nanmedian(arr))
    elif method == "sigma-clipped-median":
        factor = robust_sigma_clipped_median(arr, sigma=sigma, max_iters=max_iters)
    else:
        raise ValueError(f"Unknown normalization method: {method!r}")
    if not np.isfinite(factor) or factor == 0:
        return np.nan
    return factor


def normalize_flux_and_error(
    flux: np.ndarray,
    error: np.ndarray,
    method: str,
    sigma: float,
    max_iters: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    flux = np.asarray(flux, dtype=float)
    error = np.asarray(error, dtype=float)
    factor = normalization_factor(flux, method=method, sigma=sigma, max_iters=max_iters)
    if not np.isfinite(factor) or factor == 0:
        return flux.copy(), error.copy(), np.nan
    return flux / factor, error / abs(factor), factor


def infer_cadence_seconds(time_btjd: np.ndarray, hdr0, hdr1) -> float:
    # TIMEDEL is in days in standard TESS products.
    for header in (hdr1, hdr0):
        value = first_present(header, ["TIMEDEL"], default=None)
        try:
            seconds = float(value) * 86400.0
        except Exception:
            seconds = np.nan
        if np.isfinite(seconds) and seconds > 0:
            return seconds

    for header in (hdr1, hdr0):
        value = first_present(header, ["EXPOSURE", "EXPTIME", "FRAMETIM"], default=None)
        try:
            seconds = float(value)
        except Exception:
            seconds = np.nan
        if np.isfinite(seconds) and seconds > 0:
            return seconds

    t = np.asarray(time_btjd, dtype=float)
    dt = np.diff(np.sort(t[np.isfinite(t)]))
    dt = dt[np.isfinite(dt) & (dt > 0)]
    return float(np.nanmedian(dt) * 86400.0) if dt.size else np.nan


def _tic_token(value: object) -> str:
    text = str(value).strip()
    match = re.search(r"(?:TIC\s*)?(\d+)", text, flags=re.IGNORECASE)
    return match.group(1) if match else sanitize_token(text)


def read_spoc_lc_fits(
    path: Path,
    *,
    quality_policy: str = "good",
    normalization: str = "sigma-clipped-median",
    normalization_sigma: float = 4.0,
    normalization_iters: int = 5,
    flux_choice: str = "auto",
) -> pd.DataFrame:
    path = Path(path).expanduser().resolve()
    if is_tess_target_pixel_path(path):
        raise ValueError(target_pixel_error_message(path))

    if fits is None:
        raise ImportError(
            "spoc_lightcurve_converter.py requires astropy. "
            f"Original import error: {_ASTROPY_IMPORT_ERROR}"
        )

    quality_policy = str(quality_policy).strip().lower()
    normalization = str(normalization).strip().lower()
    flux_choice = str(flux_choice).strip().lower()
    if quality_policy not in QUALITY_CHOICES:
        raise ValueError(f"quality_policy must be one of {QUALITY_CHOICES}")
    if normalization not in NORMALIZATION_CHOICES:
        raise ValueError(f"normalization must be one of {NORMALIZATION_CHOICES}")
    if flux_choice not in FLUX_CHOICES:
        raise ValueError(f"flux_choice must be one of {FLUX_CHOICES}")

    with fits.open(path, memmap=True) as hdul:
        if len(hdul) < 2 or getattr(hdul[1], "data", None) is None:
            raise ValueError(f"{path.name}: missing standard light-curve table in HDU 1")
        data = hdul[1].data
        hdr0 = hdul[0].header
        hdr1 = hdul[1].header
        names = set(data.names or [])

        if "TIME" not in names:
            raise KeyError(f"{path.name}: missing TIME column")

        # Some target-pixel files may have nonstandard names. Detect their
        # table signature as well as the standard *-tp.fits filename.
        has_lc_flux = ("SAP_FLUX" in names) or ("PDCSAP_FLUX" in names)
        if not has_lc_flux and "FLUX" in names:
            raise ValueError(target_pixel_error_message(path))

        n = len(data)
        time_btjd = np.asarray(data["TIME"], dtype=float)
        quality = np.asarray(data["QUALITY"], dtype=np.int64) if "QUALITY" in names else np.zeros(n, dtype=np.int64)
        cadence_num = np.asarray(data["CADENCENO"], dtype=np.int64) if "CADENCENO" in names else np.arange(n, dtype=np.int64)

        def col(name: str) -> np.ndarray:
            return np.asarray(data[name], dtype=float) if name in names else np.full(n, np.nan, dtype=float)

        sap_flux = col("SAP_FLUX")
        sap_flux_err = col("SAP_FLUX_ERR")
        pdcsap_flux = col("PDCSAP_FLUX")
        pdcsap_flux_err = col("PDCSAP_FLUX_ERR")

        sector = first_present(hdr0, ["SECTOR"], default=first_present(hdr1, ["SECTOR"], default=np.nan))
        camera = first_present(hdr0, ["CAMERA"], default=first_present(hdr1, ["CAMERA"], default=np.nan))
        ccd = first_present(hdr0, ["CCD"], default=first_present(hdr1, ["CCD"], default=np.nan))
        ticid = first_present(
            hdr0,
            ["TICID", "TARGETID", "OBJECT"],
            default=first_present(hdr1, ["TICID", "TARGETID", "OBJECT"], default="unknown"),
        )
        object_name = first_present(hdr0, ["OBJECT"], default=first_present(hdr1, ["OBJECT"], default=""))
        cadence_sec = infer_cadence_seconds(time_btjd, hdr0, hdr1)

    keep = np.isfinite(time_btjd)
    if quality_policy == "good":
        keep &= quality == 0
    keep &= np.isfinite(sap_flux) | np.isfinite(pdcsap_flux)

    time_btjd = time_btjd[keep]
    quality = quality[keep]
    cadence_num = cadence_num[keep]
    sap_flux = sap_flux[keep]
    sap_flux_err = sap_flux_err[keep]
    pdcsap_flux = pdcsap_flux[keep]
    pdcsap_flux_err = pdcsap_flux_err[keep]

    sap_rel, sap_err_rel, sap_norm = normalize_flux_and_error(
        sap_flux,
        sap_flux_err,
        normalization,
        normalization_sigma,
        normalization_iters,
    )
    pdc_rel, pdc_err_rel, pdc_norm = normalize_flux_and_error(
        pdcsap_flux,
        pdcsap_flux_err,
        normalization,
        normalization_sigma,
        normalization_iters,
    )

    have_sap = bool(np.isfinite(sap_rel).any())
    have_pdc = bool(np.isfinite(pdc_rel).any())
    if flux_choice == "pdcsap":
        if not have_pdc:
            raise ValueError(f"{path.name}: PDCSAP_FLUX was requested but contains no finite values")
        selected_rel, selected_err, selected_source, selected_norm = pdc_rel, pdc_err_rel, "pdcsap", pdc_norm
    elif flux_choice == "sap":
        if not have_sap:
            raise ValueError(f"{path.name}: SAP_FLUX was requested but contains no finite values")
        selected_rel, selected_err, selected_source, selected_norm = sap_rel, sap_err_rel, "sap", sap_norm
    elif have_pdc:
        selected_rel, selected_err, selected_source, selected_norm = pdc_rel, pdc_err_rel, "pdcsap", pdc_norm
    elif have_sap:
        selected_rel, selected_err, selected_source, selected_norm = sap_rel, sap_err_rel, "sap", sap_norm
    else:
        raise ValueError(f"{path.name}: neither SAP_FLUX nor PDCSAP_FLUX contains finite values")

    tic_token = _tic_token(ticid)
    df = pd.DataFrame(
        {
            "time_btjd": time_btjd,
            # Backward-compatible names used by the existing PhotPol tools.
            "flux_rel": sap_rel,
            "flux_detrended_rel": pdc_rel,
            "flux_medscaled": selected_rel,
            # Explicit selected science stream and uncertainty.
            "flux_selected_rel": selected_rel,
            "flux_selected_err_rel": selected_err,
            "flux_source": selected_source,
            # Both normalized streams and uncertainties.
            "sap_flux_rel": sap_rel,
            "sap_flux_err_rel": sap_err_rel,
            "pdcsap_flux_rel": pdc_rel,
            "pdcsap_flux_err_rel": pdc_err_rel,
            # Original SPOC values.
            "sap_flux": sap_flux,
            "sap_flux_err": sap_flux_err,
            "pdcsap_flux": pdcsap_flux,
            "pdcsap_flux_err": pdcsap_flux_err,
            "quality": quality,
            "cadence_num": cadence_num,
            "cadence_sec": cadence_sec,
            "sector": sector,
            "camera": camera,
            "ccd": ccd,
            "ticid": tic_token,
            "object": object_name,
            "source_file": path.name,
            "source_path": str(path),
            "normalization_method": normalization,
            "sap_normalization_factor": sap_norm,
            "pdcsap_normalization_factor": pdc_norm,
            "selected_normalization_factor": selected_norm,
        }
    )
    return df


def output_basename(df: pd.DataFrame, path: Path) -> str:
    tic = sanitize_token(df["ticid"].iloc[0]) if len(df) else "unknown"
    sec = pd.to_numeric(df["sector"], errors="coerce").iloc[0] if len(df) else np.nan
    cadence = pd.to_numeric(df["cadence_sec"], errors="coerce").iloc[0] if len(df) else np.nan
    sec_tag = f"s{int(sec):04d}" if np.isfinite(sec) else "sXXXX"
    cadence_tag = f"{int(round(cadence))}s" if np.isfinite(cadence) else "cadence_unknown"
    return f"spoc_{sec_tag}_tic{tic}_{cadence_tag}"


def _looks_like_glob(text: str) -> bool:
    return any(ch in text for ch in "*?[")


def _expand_one_input(spec: str, pattern: str, recursive: bool) -> list[Path]:
    expanded = str(Path(spec).expanduser()) if not _looks_like_glob(spec) else str(Path(spec).expanduser())
    if _looks_like_glob(expanded):
        return [Path(p).expanduser().resolve() for p in glob.glob(expanded, recursive=recursive) if Path(p).is_file()]

    path = Path(expanded).expanduser()
    if path.is_file():
        return [path.resolve()]
    if path.is_dir():
        files = path.rglob(pattern) if recursive else path.glob(pattern)
        return [p.resolve() for p in files if p.is_file()]
    return []


def discover_input_files(args) -> list[Path]:
    specs: list[str] = []
    specs.extend(args.input or [])
    specs.extend(args.input_file or [])
    if args.input_dir:
        specs.append(args.input_dir)
    if not specs:
        specs = ["."]

    files: list[Path] = []
    for spec in specs:
        files.extend(_expand_one_input(spec, args.pattern, args.recursive))
    unique = sorted({p.resolve() for p in files})
    if not unique:
        raise FileNotFoundError(
            f"No SPOC light-curve FITS files found for inputs {specs!r} "
            f"with pattern {args.pattern!r}"
        )
    return unique


def _sector_key(df: pd.DataFrame) -> str:
    value = df["sector"].iloc[0] if len(df) else np.nan
    try:
        return f"{int(float(value)):04d}"
    except Exception:
        return "XXXX"


def _cadence_value(df: pd.DataFrame) -> float:
    values = pd.to_numeric(df.get("cadence_sec", pd.Series(dtype=float)), errors="coerce")
    return float(np.nanmedian(values)) if np.isfinite(values).any() else np.inf


def select_sector_products(dfs: list[pd.DataFrame], cadence_policy: str) -> tuple[list[pd.DataFrame], list[dict]]:
    cadence_policy = str(cadence_policy).strip().lower()
    if cadence_policy not in CADENCE_POLICIES:
        raise ValueError(f"cadence_policy must be one of {CADENCE_POLICIES}")
    if cadence_policy == "all":
        return list(dfs), []

    by_sector: dict[str, list[pd.DataFrame]] = defaultdict(list)
    for df in dfs:
        by_sector[_sector_key(df)].append(df)

    selected: list[pd.DataFrame] = []
    skipped: list[dict] = []
    for sector, candidates in sorted(by_sector.items()):
        def rank(df: pd.DataFrame):
            cadence = _cadence_value(df)
            primary = cadence if cadence_policy == "shortest" else -cadence
            source = str(df["source_path"].iloc[0]) if len(df) and "source_path" in df else ""
            return (primary, -len(df), source)

        chosen = sorted(candidates, key=rank)[0]
        selected.append(chosen)
        chosen_path = str(chosen["source_path"].iloc[0]) if len(chosen) else ""
        for other in candidates:
            if other is chosen:
                continue
            skipped.append(
                {
                    "sector": sector,
                    "source_path": str(other["source_path"].iloc[0]) if len(other) else "",
                    "cadence_sec": _cadence_value(other),
                    "reason": f"cadence-policy={cadence_policy}; selected {chosen_path}",
                }
            )
    return selected, skipped


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Convert standard SPOC *lc.fits files to CSV for PhotPol Guided and Joint analysis.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        action="append",
        default=[],
        help="Input FITS file, directory, or glob. May be supplied more than once.",
    )
    parser.add_argument(
        "--input-file",
        action="append",
        default=[],
        help="Explicit input FITS file. May be supplied more than once; alias for --input.",
    )
    parser.add_argument("--input-dir", type=str, default="", help="Legacy directory input option.")
    parser.add_argument("--pattern", type=str, default=DEFAULT_PATTERN, help="Glob pattern used for directory inputs.")
    parser.add_argument("--recursive", action="store_true", help="Search directory inputs recursively.")
    parser.add_argument("--output-dir", type=str, default="spoc_csvs", help="Directory for converted CSV files.")

    parser.add_argument("--flux", choices=FLUX_CHOICES, default="auto", help="Science flux stream used for flux_medscaled/flux_selected_rel.")
    parser.add_argument("--quality", choices=QUALITY_CHOICES, default="good", help="QUALITY filtering policy.")
    parser.add_argument("--keep-all-quality", action="store_true", help="Legacy alias for --quality all.")
    parser.add_argument("--normalization", choices=NORMALIZATION_CHOICES, default="sigma-clipped-median", help="Per-file/per-sector normalization method.")
    parser.add_argument("--normalization-sigma", type=float, default=4.0, help="Sigma threshold for sigma-clipped-median normalization.")
    parser.add_argument("--normalization-iters", type=int, default=5, help="Maximum clipping iterations for sigma-clipped-median normalization.")

    parser.add_argument("--cadence-policy", choices=CADENCE_POLICIES, default="shortest", help="How to handle multiple cadence products for one TIC and sector when combining.")
    parser.add_argument("--combine-by-tic", action="store_true", help="Write one time-sorted combined CSV per TIC.")
    parser.add_argument("--write-combined", action="store_true", help="Alias for --combine-by-tic.")
    parser.add_argument("--write-per-sector", dest="write_per_sector", action="store_true", default=True, help="Write converted CSV for every input product.")
    parser.add_argument("--no-write-per-sector", dest="write_per_sector", action="store_false", help="Do not write individual product CSVs.")
    parser.add_argument("--manifest-json", type=str, default="", help="Optional JSON manifest listing all generated outputs and the preferred analysis CSV.")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.keep_all_quality:
        args.quality = "all"
    args.combine_by_tic = bool(args.combine_by_tic or args.write_combined)
    if not args.write_per_sector and not args.combine_by_tic:
        raise ValueError("At least one output type must be enabled: per-sector and/or combined.")

    fits_files = discover_input_files(args)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"SPOC lc.fits files found: {len(fits_files)}")
    print(f"Output directory: {output_dir}")
    print(
        "Settings: "
        f"flux={args.flux}, quality={args.quality}, normalization={args.normalization}, "
        f"cadence_policy={args.cadence_policy}, combine_by_tic={args.combine_by_tic}"
    )

    manifest: dict = {
        "version": 2,
        "input_files": [str(p) for p in fits_files],
        "settings": {
            "flux": args.flux,
            "quality": args.quality,
            "normalization": args.normalization,
            "normalization_sigma": float(args.normalization_sigma),
            "normalization_iters": int(args.normalization_iters),
            "cadence_policy": args.cadence_policy,
            "write_per_sector": bool(args.write_per_sector),
            "combine_by_tic": bool(args.combine_by_tic),
        },
        "per_sector_outputs": [],
        "combined_outputs": [],
        "cadence_skips": [],
        "failures": [],
        "preferred_analysis_csv": None,
    }

    by_tic: dict[str, list[pd.DataFrame]] = defaultdict(list)
    for index, fits_path in enumerate(fits_files, start=1):
        print(f"\n[{index}/{len(fits_files)}] Converting: {fits_path}")
        try:
            df = read_spoc_lc_fits(
                fits_path,
                quality_policy=args.quality,
                normalization=args.normalization,
                normalization_sigma=args.normalization_sigma,
                normalization_iters=args.normalization_iters,
                flux_choice=args.flux,
            )
        except Exception as exc:
            manifest["failures"].append({"input": str(fits_path), "error": f"{type(exc).__name__}: {exc}"})
            print(f"  [FAIL] {type(exc).__name__}: {exc}")
            continue

        if df.empty:
            manifest["failures"].append({"input": str(fits_path), "error": "No usable cadences after filtering"})
            print("  [WARN] No usable cadences after filtering; skipping.")
            continue

        tic = sanitize_token(df["ticid"].iloc[0])
        by_tic[tic].append(df)
        base = output_basename(df, fits_path)
        out_csv = output_dir / f"{base}.csv"
        if args.write_per_sector:
            df.to_csv(out_csv, index=False)
            manifest["per_sector_outputs"].append(str(out_csv))
            print(f"  Wrote per-sector CSV: {out_csv.name}")

        print(f"  TIC={tic} sector={df['sector'].iloc[0]} cadence={_cadence_value(df):.3f} s rows={len(df)}")

    if not by_tic:
        if args.manifest_json:
            manifest_path = Path(args.manifest_json).expanduser().resolve()
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        if len(manifest["failures"]) == 1:
            detail = manifest["failures"][0]["error"]
            raise RuntimeError(f"No input FITS files were converted successfully. {detail}")
        raise RuntimeError("No input FITS files were converted successfully.")

    if args.combine_by_tic:
        print("\nWriting combined TIC files...")
        for tic, dfs in sorted(by_tic.items()):
            selected, skipped = select_sector_products(dfs, args.cadence_policy)
            manifest["cadence_skips"].extend(skipped)
            if not selected:
                continue
            combined = pd.concat(selected, ignore_index=True, sort=False)
            combined = combined.sort_values("time_btjd", kind="mergesort").reset_index(drop=True)
            combined["combined_by_tic"] = True
            combined["cadence_policy"] = args.cadence_policy
            out_csv = output_dir / f"spoc_combined_tic{tic}.csv"
            combined.to_csv(out_csv, index=False)
            manifest["combined_outputs"].append(str(out_csv))
            sectors = sorted({str(v) for v in combined["sector"].dropna().unique()})
            print(f"  TIC {tic}: {out_csv.name} ({len(combined)} rows; sectors={','.join(sectors)})")
            for item in skipped:
                if sanitize_token(Path(item["source_path"]).name):
                    print(f"    [cadence skip] sector {item['sector']}: {Path(item['source_path']).name}")

    analysis_candidates = manifest["combined_outputs"] if args.combine_by_tic else manifest["per_sector_outputs"]
    if len(analysis_candidates) == 1:
        manifest["preferred_analysis_csv"] = analysis_candidates[0]
    elif len(analysis_candidates) > 1:
        print(
            "[WARN] More than one possible analysis CSV was produced. "
            "The manifest leaves preferred_analysis_csv unset; select one TIC/output explicitly."
        )

    if args.manifest_json:
        manifest_path = Path(args.manifest_json).expanduser().resolve()
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"Manifest: {manifest_path}")

    print("\nDone.")
    if manifest["failures"]:
        print(f"Completed with {len(manifest['failures'])} failed input file(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

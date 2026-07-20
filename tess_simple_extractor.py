#!/usr/bin/env python3
"""
tess_simple_extractor.py

Robust aperture light-curve extractor for TESS, Kepler, and K2 Target Pixel Files (TPFs) with:
  • image-based seeding (no Gaia/WCS needed for seeds)
  • three extraction modes:
      - fullstamp : sum all pixels in the stamp
      - fixedap   : sum a large fixed circular aperture around a seed
      - apgrow    : image peaks + pixel Voronoi + simple region-growing apertures
  • auto-switching logic (default):
      - if --n-targets > 1  -> apgrow
      - if --n-targets == 1:
          - if not saturated -> apgrow
          - if saturated     -> choose fixedap vs fullstamp based on how much of the stamp
                               is “very bright” (bleed dominates)
  • outputs are MEDIAN-SCALED (relative flux; median ≈ 1)

Outputs (per input TPF):
  - <stem>_preferred_lc_target{k}_{mode}.csv  (mission-aware time, flux_medscaled, npix)

Notes:
  - TIME is saved in its native BTJD or BKJD convention; Kepler/K2 files also include a BTJD conversion for downstream compatibility.
  - Flux comes from FLUX (e-/s) and is then median-scaled.

Example:
  python tess_simple_extractor.py --input "/path/*_tp.fits" --n-targets 1 --outdir lc_out --save-plots
"""

import argparse
from datetime import datetime, timezone
import glob
import json
from pathlib import Path

import numpy as np

# Avoid Qt/Wayland issues when saving plots in batch runs.
# If you *want* interactive windows, remove these two lines or set MPLBACKEND=TkAgg.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from astropy.io import fits

try:
    from tess_prf_photometry import (
        SceneSource,
        PRFPhotometryConfig,
        assess_saturation,
        extract_jitter_aware_prf,
        save_prf_products,
    )
    _HAVE_PRF_MODULE = True
except Exception as _prf_import_error:
    SceneSource = PRFPhotometryConfig = None
    extract_jitter_aware_prf = save_prf_products = assess_saturation = None
    _HAVE_PRF_MODULE = False
    _PRF_IMPORT_ERROR = _prf_import_error


# ----------------------------
#  Heuristics / configuration
# ----------------------------

def is_saturated(mean_img: np.ndarray,
                 top_frac: float = 0.02,
                 flat_frac_of_max: float = 0.5,
                 min_flat_pixels: int = 8) -> bool:
    """Heuristic saturation test for very bright/saturated targets.

    - Take the brightest top_frac of pixels (at least min_flat_pixels)
    - If their median is > flat_frac_of_max * max, we likely have a flat-topped
      saturated core / bleed structure.
    """
    img = np.asarray(mean_img, float)
    v = img[np.isfinite(img)]
    if v.size == 0:
        return False
    v = np.sort(v)
    n = max(1, int(round(top_frac * v.size)))
    n = max(n, min_flat_pixels)
    top = v[-n:]
    vmax = float(np.nanmax(v))
    if not np.isfinite(vmax) or vmax <= 0:
        return False
    return float(np.nanmedian(top)) > (flat_frac_of_max * vmax)


def very_bright_fraction(mean_img: np.ndarray, frac_of_peak: float = 0.10) -> float:
    """Fraction of finite pixels above frac_of_peak * peak."""
    img = np.asarray(mean_img, float)
    m = np.isfinite(img)
    if not np.any(m):
        return 0.0
    vmax = float(np.nanmax(img[m]))
    if not np.isfinite(vmax) or vmax <= 0:
        return 0.0
    return float(np.mean(img[m] > (frac_of_peak * vmax)))


# ----------------------------
#  Seeding and apertures
# ----------------------------

def find_peak_seeds(mean_img: np.ndarray,
                    n_peaks: int,
                    min_sep: int = 6,
                    edge: int = 1) -> list[tuple[int, int]]:
    """Find up to n_peaks local maxima in the mean image (NumPy only).

    Returns list of (iy, ix) seeds.
    """
    img = np.asarray(mean_img, float)
    ny, nx = img.shape
    work = img.copy()
    work[~np.isfinite(work)] = -np.inf
    if edge > 0:
        work[:edge, :] = -np.inf
        work[-edge:, :] = -np.inf
        work[:, :edge] = -np.inf
        work[:, -edge:] = -np.inf

    peaks = []
    for iy in range(1, ny - 1):
        for ix in range(1, nx - 1):
            v = work[iy, ix]
            if not np.isfinite(v):
                continue
            nb = work[iy - 1:iy + 2, ix - 1:ix + 2]
            if v >= np.nanmax(nb):
                peaks.append((v, iy, ix))

    peaks.sort(key=lambda t: t[0], reverse=True)

    seeds: list[tuple[int, int]] = []
    for _, iy, ix in peaks:
        ok = True
        for sy, sx in seeds:
            if (iy - sy) ** 2 + (ix - sx) ** 2 < (min_sep ** 2):
                ok = False
                break
        if ok:
            seeds.append((iy, ix))
        if len(seeds) >= int(n_peaks):
            break

    return seeds


def select_peak_seeds(mean_img: np.ndarray, n_peaks: int, min_sep: int = 6,
                      edge: int = 1, prefer_center_first: bool = False) -> list[tuple[int, int]]:
    """Select peak seeds, placing the nearest central peak first for TESSCut."""
    if not prefer_center_first:
        return find_peak_seeds(mean_img, n_peaks=n_peaks, min_sep=min_sep, edge=edge)
    n_candidates = max(int(n_peaks), 32)
    candidates = find_peak_seeds(
        mean_img, n_peaks=n_candidates, min_sep=min_sep, edge=edge
    )
    if not candidates:
        return []
    ny, nx = np.asarray(mean_img).shape
    center_row, center_column = (ny - 1) / 2.0, (nx - 1) / 2.0
    nearest = min(
        range(len(candidates)),
        key=lambda i: (candidates[i][0] - center_row) ** 2 + (candidates[i][1] - center_column) ** 2,
    )
    ordered = [candidates[nearest]] + [seed for i, seed in enumerate(candidates) if i != nearest]
    return ordered[:max(1, int(n_peaks))]


def target_local_image(mean_img: np.ndarray, seed: tuple[int, int] | None,
                       radius: int = 8) -> np.ndarray:
    """Return a target-centred image used by simple-extractor auto decisions."""
    img = np.asarray(mean_img, float)
    if seed is None:
        return img
    row, column = map(int, seed)
    radius = max(3, int(radius))
    return img[
        max(0, row - radius):min(img.shape[0], row + radius + 1),
        max(0, column - radius):min(img.shape[1], column + radius + 1),
    ]


def circular_aperture_mask(ny: int, nx: int, center: tuple[int, int], radius: float) -> np.ndarray:
    """Boolean mask for a circular aperture in pixel coordinates."""
    cy, cx = center
    yy, xx = np.indices((ny, nx))
    return ((yy - cy) ** 2 + (xx - cx) ** 2) <= (radius ** 2)


def voronoi_owner_map_pixels(ny: int, nx: int, seeds: list[tuple[int, int]]) -> np.ndarray:
    """Pixel-space Voronoi partition: each pixel assigned to nearest seed."""
    yy, xx = np.indices((ny, nx))
    if len(seeds) == 0:
        return np.zeros((ny, nx), dtype=int)

    d2_stack = []
    for (sy, sx) in seeds:
        d2_stack.append((yy - sy) ** 2 + (xx - sx) ** 2)
    d2 = np.stack(d2_stack, axis=0)  # (k, ny, nx)
    return np.argmin(d2, axis=0).astype(int)


def region_grow_aperture(mean_img: np.ndarray,
                         seed: tuple[int, int],
                         owner: np.ndarray,
                         owner_id: int,
                         max_npix: int = 45,
                         min_frac_of_seed: float = 0.05) -> np.ndarray:
    """Simple region-growing aperture within a seed's Voronoi region."""
    img = np.asarray(mean_img, float)
    ny, nx = img.shape
    sy, sx = seed
    if sy < 0 or sy >= ny or sx < 0 or sx >= nx:
        raise ValueError("Seed out of bounds")

    mask = np.zeros((ny, nx), dtype=bool)

    seed_val = img[sy, sx]
    if not np.isfinite(seed_val):
        seed_val = np.nanmax(img)

    thresh = min_frac_of_seed * seed_val if np.isfinite(seed_val) else -np.inf

    mask[sy, sx] = True
    frontier = {(sy, sx)}

    def neighbors(p):
        y, x = p
        for yy, xx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
            if 0 <= yy < ny and 0 <= xx < nx:
                yield yy, xx

    while mask.sum() < max_npix:
        candidates = set()
        for p in frontier:
            for q in neighbors(p):
                if mask[q]:
                    continue
                if owner[q] != owner_id:
                    continue
                candidates.add(q)

        if not candidates:
            break

        best = None
        best_val = -np.inf
        for q in candidates:
            v = img[q]
            if np.isfinite(v) and v > best_val:
                best_val = float(v)
                best = q

        if best is None:
            break
        if np.isfinite(best_val) and best_val < thresh:
            break

        mask[best] = True
        frontier = {best}

    return mask


# ----------------------------
#  I/O helpers
# ----------------------------

def load_tpf_cube(path: str):
    """Load TIME, FLUX, optional FLUX_ERR, QUALITY, and headers from a TESS TPF."""
    with fits.open(path, memmap=True) as hdul:
        data = hdul[1].data
        hdr0 = hdul[0].header
        hdr1 = hdul[1].header
        names = set(data.columns.names)
        time = np.array(data["TIME"], dtype=float)
        flux = np.array(data["FLUX"], dtype=float)  # (nt, ny, nx)
        flux_err = np.array(data["FLUX_ERR"], dtype=float) if "FLUX_ERR" in names else None
        quality = np.array(data["QUALITY"], dtype=int) if "QUALITY" in names else None
        try:
            flux_unit = hdul[1].columns["FLUX"].unit
        except Exception:
            flux_unit = None
    return time, flux, flux_err, quality, hdr0, hdr1, flux_unit



def infer_mission(path: str | Path, hdr0=None, hdr1=None) -> str:
    """Infer TESS, KEPLER, or K2 from FITS metadata and filename."""
    values = []
    for hdr in (hdr1, hdr0):
        if hdr is None:
            continue
        for key in ("MISSION", "TELESCOP", "OBSERVAT"):
            try:
                val = hdr.get(key)
                if val not in (None, ""):
                    values.append(str(val))
            except Exception:
                pass
    text = " ".join(values).upper()
    name = Path(path).name.lower()
    campaign = None
    for hdr in (hdr1, hdr0):
        try:
            val = hdr.get("CAMPAIGN") if hdr is not None else None
            if val not in (None, ""):
                campaign = val
                break
        except Exception:
            pass
    if "K2" in text or name.startswith("ktwo") or campaign not in (None, ""):
        return "K2"
    if "TESS" in text or name.startswith("tess") or "astrocut" in name:
        return "TESS"
    if "KEPLER" in text or name.startswith("kplr"):
        return "KEPLER"
    return "UNKNOWN"


def is_tesscut_product(path: str | Path, hdr0=None, hdr1=None) -> bool:
    """Identify TESSCut/Astrocut stamps, which lack a unique SPOC target."""
    if "astrocut" in Path(path).name.lower():
        return True
    for hdr in (hdr1, hdr0):
        if hdr is None:
            continue
        text = " ".join(str(hdr.get(key, "")) for key in ("CREATOR", "PROCNAME", "ORIGIN")).lower()
        if "astrocut" in text or "tesscut" in text:
            return True
    return False


def infer_time_system(hdr0=None, hdr1=None, mission: str = "UNKNOWN") -> str:
    for hdr in (hdr1, hdr0):
        try:
            ref = hdr.get("BJDREFI") if hdr is not None else None
            if ref is None:
                continue
            ref = int(round(float(ref)))
            if ref == 2457000:
                return "BTJD"
            if ref == 2454833:
                return "BKJD"
        except Exception:
            pass
    return "BTJD" if str(mission).upper() == "TESS" else ("BKJD" if str(mission).upper() in {"KEPLER", "K2"} else "JD")


def discover_tpf_paths(input_value: str, recursive: bool = False) -> list[str]:
    """Resolve a path, glob, or directory containing TESS/Kepler/K2 TPFs."""
    candidate = Path(input_value).expanduser()
    if candidate.is_dir():
        patterns = (
            "*tp.fits", "*tpf.fits", "*tp.fits.gz", "*tpf.fits.gz",
            "*_astrocut.fits", "*_astrocut.fits.gz",
            "*_lpd-targ.fits", "*_spd-targ.fits",
            "*_lpd-targ.fits.gz", "*_spd-targ.fits.gz",
        )
        found = []
        for pat in patterns:
            found.extend(candidate.rglob(pat) if recursive else candidate.glob(pat))
        return sorted({str(x.resolve()) for x in found})
    paths = sorted(glob.glob(str(candidate), recursive=bool(recursive)))
    if not paths and candidate.exists():
        paths = [str(candidate)]
    return paths


def target_pixel_stem(path: str | Path) -> str:
    """Strip FITS and optional compression suffixes from a product name."""
    name = Path(path).name
    lower = name.lower()
    for suffix in (".fits.gz", ".fit.gz", ".fits", ".fit"):
        if lower.endswith(suffix):
            return name[:-len(suffix)]
    return Path(name).stem


def segment_indices_by_gaps(time, gap_days=0.5):
    """Return list of index arrays, split where time gaps exceed gap_days."""
    t = np.asarray(time, float)
    good = np.isfinite(t)
    idx = np.where(good)[0]
    if idx.size == 0:
        return []
    tt = t[idx]
    breaks = np.where(np.diff(tt) > gap_days)[0]
    starts = np.r_[0, breaks + 1]
    ends   = np.r_[breaks + 1, tt.size]
    return [idx[s:e] for s, e in zip(starts, ends) if (e - s) > 0]

def median_scale_by_segment(time, flux_1d, gap_days=0.5):
    """Median-scale within each continuous segment (split by time gaps)."""
    f = np.asarray(flux_1d, float).copy()
    for ii in segment_indices_by_gaps(time, gap_days=gap_days):
        med = np.nanmedian(f[ii])
        if np.isfinite(med) and med != 0:
            f[ii] /= med
    return f


def save_lc_csv(out_path: Path, time: np.ndarray, flux_medscaled: np.ndarray, npix: int,
                mission: str = "TESS", time_system: str = "BTJD"):
    import pandas as pd
    t = np.asarray(time, float)
    data = {}
    system = str(time_system).upper()
    if system == "BKJD":
        data["time_bkjd"] = t
        data["time_btjd"] = t - 2167.0
    elif system == "BTJD":
        data["time_btjd"] = t
    elif system == "MJD":
        data["time_mjd"] = t
        data["time_btjd"] = t - 56999.5
    else:
        data["time_jd"] = t
        data["time_btjd"] = t - 2457000.0
    data["flux_medscaled"] = np.asarray(flux_medscaled, float)
    data["npix"] = np.full(len(t), int(npix), dtype=int)
    data["mission"] = np.full(len(t), str(mission).upper(), dtype=object)
    data["time_system"] = np.full(len(t), system, dtype=object)
    pd.DataFrame(data).to_csv(out_path, index=False)


def quicklook_plot(out_png: Path, time: np.ndarray, flux_medscaled: np.ndarray, title: str, time_system: str = "BTJD"):
    plt.figure(figsize=(10, 3))
    plt.plot(time, flux_medscaled, ".", ms=2)
    plt.xlabel(f"Time [{str(time_system).upper()}]")
    plt.ylabel("Relative flux (median=1)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def mask_plot(out_png: Path, mean_img: np.ndarray, masks: list[np.ndarray], seeds: list[tuple[int, int]], title: str):
    plt.figure(figsize=(5.5, 5.5))
    plt.imshow(mean_img, origin="lower")
    for k, m in enumerate(masks):
        yy, xx = np.where(m)
        plt.plot(xx, yy, ".", ms=1)
        sy, sx = seeds[k]
        plt.plot([sx], [sy], "o", ms=6)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


# ----------------------------
#  Mode selection
# ----------------------------

def choose_mode(args, mean_img: np.ndarray, target_seed: tuple[int, int] | None = None) -> str:
    """Return one of: 'fullstamp', 'fixedap', 'apgrow'."""
    if args.aperture_mode != "auto":
        return args.aperture_mode

    if args.n_targets > 1:
        return "apgrow"

    # n_targets == 1: decide based on saturation
    decision_img = target_local_image(mean_img, target_seed)
    sat = is_saturated(decision_img,
                       top_frac=args.sat_top_frac,
                       flat_frac_of_max=args.sat_flat_frac_of_max,
                       min_flat_pixels=args.sat_min_flat_pixels)
    if not sat:
        return "apgrow"

    # Saturated single target: choose fullstamp vs fixedap
    vb = very_bright_fraction(decision_img, frac_of_peak=args.fullstamp_bright_frac_of_peak)
    # If bleed dominates a big chunk of the stamp, fullstamp is usually safest.
    if vb >= args.fullstamp_if_bright_frac_ge:
        return "fullstamp"
    return "fixedap"


def write_run_configuration(outdir: Path, args, paths: list[str]) -> Path:
    """Record resolved simple-extractor settings and discovered inputs."""
    created = datetime.now(timezone.utc)
    payload = {
        "schema_version": 1,
        "created_utc": created.isoformat(),
        "extractor": Path(__file__).name,
        "settings": vars(args).copy(),
        "input_files": list(paths),
    }
    stamp = created.strftime("%Y%m%dT%H%M%S_%fZ")
    path = outdir / f"extraction_run_config_{stamp}.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


# ----------------------------
#  Main
# ----------------------------

def main(argv=None):
    """Parse command-line options and extract every matched target-pixel file."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True,
                    help="TPF path, glob pattern, or directory containing TESS/Kepler/K2 target-pixel files.")
    ap.add_argument("--outdir", default="lc_out", help="Output directory.")
    ap.add_argument("--gap-days", type=float, default=0.5,
                    help="Gap threshold (days) for segment-wise median scaling.")
    ap.add_argument("--no-quality0", action="store_true",
                    help="Disable filtering to QUALITY==0 cadences.")
    ap.add_argument("--save-plots", action="store_true",
                    help="Save quicklook PNGs (light curves and aperture overlays).")
    ap.add_argument("--recursive", action="store_true",
                    help="Enable recursive ** glob expansion when --input contains **.")

    # Optional jitter-aware PRF photometry.  This writes an additional PRF
    # product and never replaces the selected aperture product.
    ap.add_argument("--prf-photometry", action="store_true",
                    help="Also extract a cadence-dependent, jitter-aware PRF-weighted light curve.")
    ap.add_argument("--prf-backend", choices=["auto", "lkprf", "tess_prf", "gaussian"], default="auto",
                    help="PRF backend. Auto prefers official lkprf/TESS_PRF engineering models.")
    ap.add_argument("--prf-motion-source", choices=["auto", "poscorr", "ensemble", "target", "fixed"], default="auto",
                    help="Motion source. Auto uses varying POS_CORR, then ensemble centroid, target centroid, and fixed position; constant placeholders are rejected.")
    ap.add_argument("--prf-scene-mode", choices=["single", "gaia"], default="single",
                    help="PRF scene mode. The simple extractor has no Gaia catalog and therefore falls back to single-source mode for 'gaia'.")
    ap.add_argument("--prf-neighbor-treatment", choices=["fixed"], default="fixed")
    ap.add_argument("--prf-source-output", choices=["primary", "all"], default="primary",
                    help="Accepted for GUI compatibility. The simple extractor has no Gaia scene, so only the primary source is written.")
    ap.add_argument("--prf-neighbor-dmag", type=float, default=8.0)
    ap.add_argument("--prf-neighbor-margin", type=float, default=6.0)
    ap.add_argument("--prf-max-scene-sources", type=int, default=20)
    ap.add_argument("--prf-min-neighbor-fraction", type=float, default=1e-4)
    ap.add_argument("--prf-background", choices=["none", "constant", "plane"], default="plane",
                    help="Per-cadence background terms fitted with the PRF source flux.")
    ap.add_argument("--prf-min-weight", type=float, default=1e-5,
                    help="Minimum relative PRF weight used when defining the fitting region.")
    ap.add_argument("--prf-fit-radius", type=float, default=6.0,
                    help="Radius in pixels included around each PRF source in the scene fit.")
    ap.add_argument("--prf-shift-quantization", type=float, default=0.01,
                    help="Requested regular-grid spacing for precomputed shifted PRFs; cadence PRFs are bilinearly interpolated and each axis is capped at 21 nodes.")
    ap.add_argument("--prf-max-shift", type=float, default=2.0,
                    help="Maximum absolute cadence shift in pixels after robust cleaning.")
    ap.add_argument("--prf-no-diagnostics", action="store_true",
                    help="Do not write the PRF diagnostic PNG (motion data and metadata are still saved).")
    ap.add_argument("--prf-no-gaussian-fallback", action="store_true",
                    help="Fail PRF extraction rather than use a Gaussian if official PRF packages are unavailable.")

    ap.add_argument("--n-targets", type=int, default=1,
                    help="Number of targets to extract. TESSCut places the detected peak nearest the cutout centre first.")

    ap.add_argument("--aperture-mode", choices=["auto", "fullstamp", "fixedap", "apgrow"],
                    default="auto",
                    help="Extraction mode. 'auto' switches among fullstamp/fixedap/apgrow.")

    # Fixed aperture options (used in fixedap mode)
    ap.add_argument("--fixed-radius", type=float, default=12.0,
                    help="Fixed circular aperture radius in pixels (fixedap mode).")
    ap.add_argument("--fixed-center", choices=["peak", "center"], default="peak",
                    help="Where to center fixed aperture: 'peak' uses brightest peak; 'center' uses stamp center.")
    ap.add_argument("--fixed-edge", type=int, default=1,
                    help="Edge mask (pixels) when locating the peak for fixed aperture.")

    # apgrow options
    ap.add_argument("--max-npix", type=int, default=45,
                    help="Max pixels per aperture (apgrow mode).")
    ap.add_argument("--min-sep", type=int, default=6,
                    help="Minimum separation (pixels) between peak seeds (apgrow).")
    ap.add_argument("--edge", type=int, default=1,
                    help="Edge mask (pixels) for peak finding (apgrow).")
    ap.add_argument("--min-frac-of-seed", type=float, default=0.05,
                    help="Stop adding pixels when candidate is below this fraction of seed pixel value (apgrow).")

    # Saturation and auto-switch tuning
    ap.add_argument("--sat-top-frac", type=float, default=0.02,
                    help="Saturation test: fraction of brightest pixels to consider.")
    ap.add_argument("--sat-flat-frac-of-max", type=float, default=0.5,
                    help="Saturation test: median(top) > this * max implies saturation.")
    ap.add_argument("--sat-min-flat-pixels", type=int, default=8,
                    help="Saturation test: minimum number of pixels in 'top' set.")

    ap.add_argument("--fullstamp-bright-frac-of-peak", type=float, default=0.10,
                    help="Auto-switch: define 'very bright' as > this * peak.")
    ap.add_argument("--fullstamp-if-bright-frac-ge", type=float, default=0.25,
                    help="Auto-switch (saturated single): if very-bright fraction >= this, use fullstamp else fixedap.")

    args = ap.parse_args(argv)
    if int(args.n_targets) < 1:
        ap.error("--n-targets must be at least 1")

    paths = discover_tpf_paths(args.input, recursive=bool(args.recursive))
    if len(paths) == 0:
        raise SystemExit(f"No target-pixel files matched: {args.input}")

    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    args.outdir = str(outdir)
    config_path = write_run_configuration(outdir, args, paths)
    print(f"Resolved extraction settings: {config_path}")

    for path in paths:
        base = target_pixel_stem(path)
        print(f"TPF: {path}")

        time, flux, flux_err, quality, _hdr0, _hdr1, flux_unit = load_tpf_cube(path)
        mission = infer_mission(path, _hdr0, _hdr1)
        tesscut_product = is_tesscut_product(path, _hdr0, _hdr1)
        time_system = infer_time_system(_hdr0, _hdr1, mission)
        print(f"  Mission: {mission}; native time: {time_system}")
        if args.prf_photometry and mission in {"KEPLER", "K2"}:
            print(
                f"  [WARN] {mission} target-pixel data detected: the PRF module is TESS-only "
                "and will be skipped. Aperture extraction will continue normally."
            )
        good = np.isfinite(time)
        if (quality is not None) and (not args.no_quality0):
            good &= (quality == 0)
            if not np.any(good) and np.any(np.isfinite(time)):
                print("  [WARN] No QUALITY==0 cadences; retaining finite-time cadences instead.")
                good = np.isfinite(time)
        if not np.any(good):
            print("  [WARN] No finite cadences; skipping this file.")
            continue
        cadence_indices = np.flatnonzero(good)
        if not np.all(good):
            dropped = int(np.size(good) - np.sum(good))
            print(f"  [INFO] Dropping {dropped} cadences (non-finite time and/or QUALITY!=0)")
            time = time[good]
            flux = flux[good, :, :]
            if flux_err is not None:
                flux_err = flux_err[good, :, :]


        mean_img = np.nanmean(flux, axis=0)
        ny, nx = mean_img.shape
        target_seeds = select_peak_seeds(
            mean_img,
            n_peaks=max(1, int(args.n_targets)),
            min_sep=max(1, int(args.min_sep)),
            edge=max(0, int(args.edge)),
            prefer_center_first=tesscut_product,
        )
        if not target_seeds:
            target_seeds = [(ny // 2, nx // 2)]
        if tesscut_product:
            print(
                "  TESSCut target policy: target 1 is the detected peak nearest "
                f"the requested cutout centre (seed={target_seeds[0]})."
            )

        # Optional additional jitter-aware PRF extraction.  It is intentionally
        # independent of the aperture mode selected below.
        if args.prf_photometry and mission == "TESS":
            if not _HAVE_PRF_MODULE:
                print(f"  [WARN] PRF photometry unavailable: {_PRF_IMPORT_ERROR}")
            else:
                tessmag = None
                if not tesscut_product:
                    for hdr in (_hdr1, _hdr0):
                        try:
                            value = hdr.get("TESSMAG")
                            if value is not None:
                                tessmag = float(value)
                                break
                        except Exception:
                            pass
                if str(args.prf_scene_mode).strip().lower() != "single":
                    print("  [WARN] Gaia multi-source PRF mode requires tess_watershed_extractor.py; the simple extractor will use single-source PRF fits.")
                if str(args.prf_source_output).strip().lower() != "primary":
                    print("  [WARN] All-scene-source PRF output requires the watershed extractor; the simple extractor will write only primary-source light curves.")
                prf_seeds = target_seeds
                for prf_k, (prf_row, prf_col) in enumerate(prf_seeds, start=1):
                    sat_info = assess_saturation(
                        mean_img,
                        tessmag=tessmag,
                        source_row=float(prf_row),
                        source_column=float(prf_col),
                        local_radius=max(8.0, float(args.prf_fit_radius)),
                        flux_unit=str(flux_unit) if flux_unit is not None else None,
                        product_type="TESSCUT" if tesscut_product else "TPF",
                    )
                    if sat_info.get("saturated", False):
                        print(
                            f"  [WARN] PRF photometry skipped for target {prf_k}: "
                            "the source appears saturated or bleed-dominated. "
                            + "; ".join(sat_info.get("reasons", []))
                        )
                        continue
                    try:
                        prf_cfg = PRFPhotometryConfig(
                            backend=args.prf_backend,
                            motion_source=args.prf_motion_source,
                            background=args.prf_background,
                            min_prf_weight=float(args.prf_min_weight),
                            fit_radius=float(args.prf_fit_radius),
                            shift_quantization=float(args.prf_shift_quantization),
                            max_abs_shift=float(args.prf_max_shift),
                            allow_gaussian_fallback=(not args.prf_no_gaussian_fallback),
                            scene_mode="single",
                            neighbor_treatment=str(args.prf_neighbor_treatment),
                            source_output_mode="primary",
                            neighbor_min_contribution_fraction=float(args.prf_min_neighbor_fraction),
                            save_diagnostics=(not args.prf_no_diagnostics),
                            gap_days_for_scaling=float(args.gap_days),
                        )
                        prf_result = extract_jitter_aware_prf(
                            time, flux,
                            [SceneSource(float(prf_row), float(prf_col), f"target{prf_k}")],
                            config=prf_cfg,
                            tpf_path=path,
                            cadence_indices=cadence_indices,
                            flux_err_cube=flux_err,
                        )
                        prf_stem = f"{base}_preferred_lc_target{prf_k}_prf"
                        paths_written = save_prf_products(
                            prf_result, mean_img, outdir, prf_stem,
                            source_index=0,
                            extra_columns={"target_index": prf_k, "method": "prf"},
                            extra_metadata={"tpf_path": str(path), "simple_extractor": True, "mission": mission},
                            save_diagnostics=(not args.prf_no_diagnostics),
                        )
                        for msg in prf_result.metadata.get("backend_messages", []):
                            print(f"  [PRF] {msg}")
                        coupling = prf_result.metadata.get("motion_coupling", [{}])[0]
                        quality = "PASS" if coupling.get("quality_pass", True) else "FLAGGED"
                        print(
                            f"  PRF target{prf_k}: backend={prf_result.backend}, "
                            f"motion={prf_result.motion_source}, quality={quality}, "
                            f"wrote={paths_written['csv'].name}"
                        )
                        if not coupling.get("quality_pass", True):
                            print(
                                "  [WARN] PRF extraction retained for diagnostics but aperture "
                                "photometry remains preferred: "
                                + "; ".join(coupling.get("quality_reasons", []))
                            )
                    except Exception as exc:
                        print(f"  [WARN] PRF target{prf_k} failed ({type(exc).__name__}: {exc}); continuing with aperture extraction.")

        primary_seed = target_seeds[0]
        mode = choose_mode(args, mean_img, target_seed=primary_seed)
        decision_img = target_local_image(mean_img, primary_seed)
        vb = very_bright_fraction(decision_img, frac_of_peak=args.fullstamp_bright_frac_of_peak)
        if args.aperture_mode == "auto":
            sat_flag = is_saturated(decision_img,
                                    top_frac=args.sat_top_frac,
                                    flat_frac_of_max=args.sat_flat_frac_of_max,
                                    min_flat_pixels=args.sat_min_flat_pixels)
            print(f"  [AUTO] saturated={sat_flag}  very_bright_frac={vb:.3f}  -> mode={mode}")

        if mode == "fullstamp":
            lc_raw = np.nansum(flux, axis=(1, 2))
            lc = median_scale_by_segment(time, lc_raw, gap_days=args.gap_days)
            npix = int(np.isfinite(mean_img).sum())
            out_csv = outdir / f"{base}_preferred_lc_target1_fullstamp.csv"
            save_lc_csv(out_csv, time, lc, npix=npix, mission=mission, time_system=time_system)
            if args.save_plots:
                out_png = outdir / f"{base}_target1_fullstamp.png"
                quicklook_plot(out_png, time, lc, f"{base} : full-stamp", time_system=time_system)
            continue

        if mode == "fixedap":
            if args.fixed_center == "center":
                seed = (ny // 2, nx // 2)
            else:
                seed = primary_seed

            mask = circular_aperture_mask(ny, nx, seed, radius=args.fixed_radius)
            npix = int(mask.sum())
            lc_raw = np.nansum(flux[:, mask], axis=1)
            lc = median_scale_by_segment(time, lc_raw, gap_days=args.gap_days)

            tag = f"fixedapR{args.fixed_radius:g}"
            out_csv = outdir / f"{base}_preferred_lc_target1_{tag}.csv"
            save_lc_csv(out_csv, time, lc, npix=npix, mission=mission, time_system=time_system)

            print(f"  Fixed aperture: center={seed}  radius={args.fixed_radius:g}  npix={npix}")

            if args.save_plots:
                out_png = outdir / f"{base}_aperture_{tag}.png"
                mask_plot(out_png, mean_img, [mask], [seed], f"{base} : {tag}")
                out_png_lc = outdir / f"{base}_target1_{tag}.png"
                quicklook_plot(out_png_lc, time, lc, f"{base} : {tag} (seed={seed})", time_system=time_system)
            continue

        # mode == "apgrow"
        seeds = target_seeds
        if len(seeds) == 0:
            print("  [WARN] No peaks found; falling back to full-stamp")
            lc_raw = np.nansum(flux, axis=(1, 2))
            lc = median_scale_by_segment(time, lc_raw, gap_days=args.gap_days)
            out_csv = outdir / f"{base}_preferred_lc_target1_fullstamp_fallback.csv"
            save_lc_csv(out_csv, time, lc, npix=int(np.isfinite(mean_img).sum()), mission=mission, time_system=time_system)
            if args.save_plots:
                out_png = outdir / f"{base}_target1_fullstamp_fallback.png"
                quicklook_plot(out_png, time, lc, f"{base} : full-stamp fallback", time_system=time_system)
            continue

        owner = voronoi_owner_map_pixels(ny, nx, seeds)

        masks = []
        for k, seed in enumerate(seeds):
            m = region_grow_aperture(
                mean_img,
                seed=seed,
                owner=owner,
                owner_id=k,
                max_npix=args.max_npix,
                min_frac_of_seed=args.min_frac_of_seed,
            )
            masks.append(m)

        if args.save_plots:
            out_png = outdir / f"{base}_apertures_apgrow.png"
            mask_plot(out_png, mean_img, masks, seeds, f"{base} : apgrow apertures")

        print("  Targets (apgrow):")
        for k, seed in enumerate(seeds):
            npix = int(masks[k].sum())
            print(f"    [{k+1}] seed={seed}  npix={npix}")

            lc_raw = np.nansum(flux[:, masks[k]], axis=1)
            lc = median_scale_by_segment(time, lc_raw, gap_days=args.gap_days)

            out_csv = outdir / f"{base}_preferred_lc_target{k+1}_apgrow.csv"
            save_lc_csv(out_csv, time, lc, npix=npix, mission=mission, time_system=time_system)

            if args.save_plots:
                out_png = outdir / f"{base}_target{k+1}_apgrow.png"
                quicklook_plot(out_png, time, lc, f"{base} : target{k+1} apgrow (seed={seed})", time_system=time_system)

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)

#!/usr/bin/env python3
"""
Watershed light-curve pipeline.

Extract aperture light curves from TESS, Kepler, and K2 target-pixel files, including TESSCut astrocut FITS files.
The optional engineering-PRF branch is TESS-only and is skipped with a warning for Kepler/K2.
This version uses watershed segmentation for initial target regions and writes flat outputs directly into --output-root.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import pickle

import numpy as np
import pandas as pd

from scipy.interpolate import PchipInterpolator
from scipy.ndimage import gaussian_filter

# Safe non-interactive backend for saving plots
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits

import lightkurve as lk
from skimage.segmentation import watershed

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

try:
    from astroquery.simbad import Simbad
except Exception:
    Simbad = None

try:
    from astroquery.gaia import Gaia
except Exception:
    # Gaia is optional for explicit single-target/no-Gaia workflows.  Keeping
    # the import optional also lets users inspect --help without installing
    # astroquery; a clear error is raised only if a Gaia query is requested.
    Gaia = None


APERTURE_FOM_MODE = "stddiff"
SAVE_FIGURE_PICKLES = False
DEFAULT_ORBITAL_TABLE = Path(__file__).resolve().with_name("tess_sector_orbfreq_midpoints.csv")


# =============================================================================
# User-supplied fixed aperture masks
# =============================================================================

_MASK_TRUE_TOKENS = {"1", "y", "yes", "t", "true"}
_MASK_FALSE_TOKENS = {"0", "n", "no", "f", "false"}


def load_external_aperture_mask(
    mask_path: str | Path,
    expected_shape: tuple[int, int] | None = None,
):
    """Read a human-editable, row-major Boolean aperture mask.

    The file contains one image row per nonblank text line. Values may be
    separated by commas or whitespace, and ``#`` starts a comment. Accepted
    true values are 1/Y/YES/T/TRUE; accepted false values are
    0/N/NO/F/FALSE, case-insensitively.

    No automatic transpose, flip, padding, or cropping is performed. This is
    deliberate: silently changing an aperture orientation would produce a
    plausible-looking but scientifically incorrect light curve. The first
    data line maps to local image row 0 and the first token maps to column 0.
    """
    path = Path(mask_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"External aperture mask not found: {path}")
    if not path.is_file():
        raise ValueError(f"External aperture mask is not a regular file: {path}")

    rows: list[list[bool]] = []
    row_line_numbers: list[int] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), start=1):
        # Inline comments make it possible to annotate a hand-maintained mask
        # without adding a sidecar file. Empty/comment-only lines are ignored.
        data_text = raw_line.split("#", 1)[0].strip()
        if not data_text:
            continue
        tokens = [token for token in re.split(r"[\s,]+", data_text) if token]
        parsed_row: list[bool] = []
        for token in tokens:
            normalized = token.strip().lower()
            if normalized in _MASK_TRUE_TOKENS:
                parsed_row.append(True)
            elif normalized in _MASK_FALSE_TOKENS:
                parsed_row.append(False)
            else:
                allowed = "0/1, Y/N, YES/NO, T/F, or TRUE/FALSE"
                raise ValueError(
                    f"Invalid external-mask token {token!r} on line {line_number} "
                    f"of {path.name}; expected {allowed}."
                )
        if parsed_row:
            rows.append(parsed_row)
            row_line_numbers.append(line_number)

    if not rows:
        raise ValueError(f"External aperture mask contains no data rows: {path}")

    n_columns = len(rows[0])
    for parsed_row, line_number in zip(rows, row_line_numbers):
        if len(parsed_row) != n_columns:
            raise ValueError(
                f"External aperture mask is not rectangular: line {line_number} "
                f"has {len(parsed_row)} columns, expected {n_columns}."
            )

    mask = np.asarray(rows, dtype=bool)
    if expected_shape is not None and tuple(mask.shape) != tuple(expected_shape):
        raise ValueError(
            f"External aperture mask shape {tuple(mask.shape)} does not match "
            f"the TPF image shape {tuple(expected_shape)}. Rows and columns are "
            "not transposed automatically."
        )
    if not np.any(mask):
        raise ValueError(f"External aperture mask selects zero pixels: {path}")

    metadata = {
        "schema_version": 1,
        "source_path": str(path),
        "source_name": path.name,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "shape_rows_columns": [int(mask.shape[0]), int(mask.shape[1])],
        "selected_pixels": int(np.count_nonzero(mask)),
        "orientation": "row-major; first data row is local image row 0; first token is local column 0",
        "accepted_values": "1/0, Y/N, YES/NO, T/F, TRUE/FALSE (case-insensitive)",
        "comments": "# begins an inline comment; blank and comment-only lines are ignored",
    }
    return mask, metadata

# =============================================================================
# Saturation-optimized single-target aperture extraction
# =============================================================================

def read_tpf_arrays_for_saturated_aperture(path: Path):
    """Read cadence arrays needed by the saturation-optimized extractor."""
    with fits.open(path, memmap=True) as hdul:
        data = hdul[1].data
        names = set(data.names)

        time = np.array(data["TIME"], dtype=float)
        if "QUALITY" in names:
            quality = np.array(data["QUALITY"])
        elif "DQUALITY" in names:
            quality = np.array(data["DQUALITY"])
        else:
            raise KeyError(f"No QUALITY/DQUALITY column found in {path.name}")

        flux = np.array(data["FLUX"], dtype=float)
        if flux.ndim != 3:
            raise ValueError(f"Expected FLUX to have ndim=3, got shape {flux.shape} in {path.name}")

        _, nrow, ncol = flux.shape
        flux2d = flux.reshape(flux.shape[0], nrow * ncol)

    return time, flux2d, quality, nrow, ncol


def extract_saturation_optimized_aperture(
    path: Path,
    threshold: float = 3000.0,
    nback: int = 20,
    filter_quality: bool = True,
    verbose: bool = True,
):
    """Build a single-target aperture by minimizing high-frequency scatter.

    Pixels above ``threshold`` seed the aperture.  Remaining positive pixels
    are then tested one at a time, and the candidate with the lowest
    first-difference figure of merit is appended.  The best intermediate
    aperture is selected after the full growth sequence.  This unrestricted
    image geometry works for arbitrary TPF and TESSCut stamp dimensions.
    """
    time, flux, quality, nrow, ncol = read_tpf_arrays_for_saturated_aperture(path)
    npix = nrow * ncol

    keep = np.isfinite(time)
    if filter_quality:
        quality_keep = quality == 0
        if np.any(keep & quality_keep):
            keep &= quality_keep
        else:
            print("  [WARN] No QUALITY==0 cadences; retaining finite-time cadences instead.")
    time = time[keep]
    flux = flux[keep, :]

    if len(time) < 3:
        raise ValueError(f"Not enough valid cadences after filtering in {path.name}")

    mean_image = np.nanmean(flux, axis=0)
    if not np.isfinite(mean_image).any():
        raise ValueError(f"Mean image is all-NaN for {path.name}")

    nback = min(int(nback), npix)
    idx_sorted = np.argsort(np.where(np.isfinite(mean_image), mean_image, np.inf))
    back_idx = idx_sorted[:nback]
    back = np.nanmean(flux[:, back_idx], axis=1)

    els = np.where(np.isfinite(mean_image) & (mean_image > threshold))[0]
    if len(els) == 0:
        brightest = int(np.nanargmax(mean_image))
        els = np.array([brightest], dtype=int)

    flag = np.zeros(npix, dtype=int)
    flag[els] = 1

    ts_flux = [np.nansum(flux[:, els], axis=1)]
    ts_pixels = list(els.astype(int))
    ts_diff = []

    pixel_means = np.nanmean(flux, axis=0)

    cur = ts_flux[0]
    denom = np.nansum(cur)
    if denom == 0 or not np.isfinite(denom):
        raise ValueError(f"Initial aperture flux sum is invalid for {path.name}")
    fom0 = np.nansum(np.abs(np.diff(cur))) / denom
    ts_diff.append(fom0)

    for _ in range(npix):
        test_fom = np.full(npix, np.nan, dtype=float)

        for ii in range(npix):
            if flag[ii] == 0 and np.isfinite(pixel_means[ii]) and pixel_means[ii] > 0:
                temp_flux = ts_flux[-1] + flux[:, ii]
                denom = np.nansum(temp_flux)
                if denom != 0 and np.isfinite(denom):
                    test_fom[ii] = np.nansum(np.abs(np.diff(temp_flux))) / denom

        if not np.isfinite(test_fom).any():
            break

        bb = int(np.nanargmin(test_fom))
        aa = float(test_fom[bb])

        flag[bb] = 1
        ts_flux.append(ts_flux[-1] + flux[:, bb])
        ts_diff.append(aa)
        ts_pixels.append(bb)

    fom = []
    for arr in ts_flux:
        mu = np.nanmean(arr)
        if mu == 0 or not np.isfinite(mu):
            fom.append(np.nan)
        else:
            fom.append(np.nanstd(np.diff(arr)) / mu)
    fom = np.array(fom, dtype=float)

    if not np.isfinite(fom).any():
        raise ValueError(f"No finite FOM values for {path.name}")

    best_idx = int(np.nanargmin(fom))
    best_flux = np.array(ts_flux[best_idx], dtype=float)

    med = np.nanmedian(best_flux)
    if med == 0 or not np.isfinite(med):
        raise ValueError(f"Median best flux invalid for {path.name}")

    rel_flux = best_flux / med

    best_pixels_linear = np.array(ts_pixels[: best_idx + 1], dtype=int)
    best_mask = np.zeros(npix, dtype=bool)
    best_mask[best_pixels_linear] = True
    best_mask_2d = best_mask.reshape(nrow, ncol)

    # Flux-weighted aperture center-of-light series. For saturated targets this is
    # best interpreted as a motion/systematics proxy rather than a precise astrometric centroid.
    yy2d, xx2d = np.indices((nrow, ncol))
    ypix = yy2d.ravel()[best_mask][None, :]
    xpix = xx2d.ravel()[best_mask][None, :]
    f_ap = flux[:, best_mask]
    denom = np.nansum(f_ap, axis=1)
    denom = np.where(np.isfinite(denom) & (denom != 0), denom, np.nan)
    crow = np.nansum(f_ap * ypix, axis=1) / denom
    ccol = np.nansum(f_ap * xpix, axis=1) / denom

    meta = {
        "file": str(path),
        "sector": infer_sector_from_tpf(path),
        "nrow": nrow,
        "ncol": ncol,
        "npix": npix,
        "n_cadences_used": len(time),
        "n_initial_pixels": len(els),
        "n_pixels_in_best_curve": best_idx + 1,
        "best_fom": float(fom[best_idx]),
        "threshold": float(threshold),
        "nback": int(nback),
        "aperture_geometry": "unrestricted",
    }

    if verbose:
        print(
            f"  Saturation-optimized aperture: shape={nrow}x{ncol}, cadences={len(time)}, "
            f"best_pixels={best_idx + 1}, best_fom={fom[best_idx]:.6g}"
        )

    # Mission-specific time naming belongs in ``main``, where the input has
    # already been classified as TESS, Kepler, or K2. Keeping this internal
    # table neutral avoids accidentally labeling native BKJD values as BTJD.
    out = pd.DataFrame({"time_native": time, "flux_detrended_rel": rel_flux})
    return out, meta, mean_image.reshape(nrow, ncol), best_mask_2d, back, keep, crow, ccol


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


def save_aperture_plot(mean_image_2d, ap_mask_2d, outpng: Path, title: str):
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(mean_image_2d, origin="lower", aspect="equal", cmap="gray", interpolation="nearest")
    yy, xx = np.where(ap_mask_2d)
    if len(xx):
        ax.plot(xx, yy, "rs", ms=6, mfc="none")
    ax.set_title(title)
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Mean Flux")
    fig.tight_layout()
    save_figure_with_optional_pickle(fig, outpng, dpi=150)
    plt.close(fig)


def save_lightcurve_plot(time_values, flux_rel, outpng: Path, title: str, time_label: str = "Time [BTJD]"):
    fig, ax = plt.subplots(figsize=(8.0, 3.8))
    ax.plot(np.asarray(time_values, float), np.asarray(flux_rel, float), ".", ms=2.5)
    ax.set_xlabel(str(time_label))
    ax.set_ylabel("Relative Flux")
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    save_figure_with_optional_pickle(fig, outpng, dpi=180, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# Robust regression + design matrix (decorrelation)
# =============================================================================

def robust_wls(X, y, n_iter: int = 8, huber_k: float = 1.5):
    """Iteratively reweighted least squares with Huber-like weights."""
    y = np.asarray(y, float)
    X = np.asarray(X, float)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    for _ in range(int(n_iter)):
        r = y - X @ beta
        s = 1.4826 * np.nanmedian(np.abs(r - np.nanmedian(r)))
        if not np.isfinite(s) or s == 0:
            break
        uvec = r / (huber_k * s)
        w = np.ones_like(uvec)
        m = np.abs(uvec) > 1
        w[m] = 1.0 / np.abs(uvec[m])
        sw = np.sqrt(w)
        Xw = X * sw[:, None]
        yw = y * sw
        beta_new, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
        if np.allclose(beta_new, beta, rtol=1e-6, atol=1e-9):
            beta = beta_new
            break
        beta = beta_new
    return beta


def build_design_matrix(t, x, y, knot_spacing_days: float = 1.0, psf_sigma=None):
    """Design matrix for decorrelating against centroid motion + low-order time trend.

    Includes optional PSF-width proxy (psf_sigma), if provided.
    """
    t = np.asarray(t, float)
    x = np.asarray(x, float)
    y = np.asarray(y, float)

    t0 = np.nanmedian(t)
    tt = t - t0
    x0 = np.nanmedian(x)
    xx = x - x0
    y0 = np.nanmedian(y)
    yy = y - y0

    cols = [
        np.ones_like(tt),
        xx,
        yy,
        xx**2,
        yy**2,
        xx * yy,
        tt,
        tt**2,
        tt**3,
    ]

    if psf_sigma is not None:
        ss = np.asarray(psf_sigma, float)
        s0 = np.nanmedian(ss)
        cols.append(ss - s0)
        cols.append((ss - s0) ** 2)

    if np.isfinite(tt).any():
        tmin, tmax = np.nanmin(tt), np.nanmax(tt)
        if (tmax - tmin) > 2 * knot_spacing_days:
            knots = np.arange(tmin + knot_spacing_days, tmax, knot_spacing_days)
            for k in knots:
                cols.append(np.maximum(0.0, tt - k))

    return np.vstack(cols).T



# =============================================================================
# Sector orbital-frequency helpers for saturated-target systematics correction
# =============================================================================

def load_sector_orbtable(csv_path: str | Path):
    """Load sector midpoint time + orbital frequency table.

    Preferred columns (as in the bundled table):
      - sector (int)
      - mid_btjd (sector midpoint in BJD - 2457000 days)
      - freq_cyc_per_day (positive orbital frequency in cycles/day)

    The historical ``mid_tjd`` and ``freq_cyc/day`` headings remain accepted
    so existing user tables continue to work.

    Returns a dict: sector -> (mid_btjd, freq_cyc_per_day)
    """
    p = Path(csv_path).expanduser()
    if not p.exists():
        raise FileNotFoundError(f"Orbital-frequency table not found: {p}")
    df = pd.read_csv(p)
    # Strip surrounding whitespace while retaining the user's original labels
    # for clear error messages.
    cols = {c.strip(): c for c in df.columns}
    mid_key = "mid_btjd" if "mid_btjd" in cols else "mid_tjd" if "mid_tjd" in cols else None
    freq_key = (
        "freq_cyc_per_day"
        if "freq_cyc_per_day" in cols
        else "freq_cyc/day"
        if "freq_cyc/day" in cols
        else None
    )
    missing = []
    if "sector" not in cols:
        missing.append("sector")
    if mid_key is None:
        missing.append("mid_btjd (or legacy mid_tjd)")
    if freq_key is None:
        missing.append("freq_cyc_per_day (or legacy freq_cyc/day)")
    if missing:
        raise ValueError(f"Orbital-frequency table missing columns: {missing}. Found: {list(df.columns)}")

    out = {}
    invalid_rows = []
    for row_index, r in df.iterrows():
        try:
            sec = int(r[cols["sector"]])
            mid_btjd = float(r[cols[mid_key]])
            freq = float(r[cols[freq_key]])
        except Exception as exc:
            invalid_rows.append(f"row {int(row_index) + 2}: {type(exc).__name__}")
            continue
        if sec < 1 or not np.isfinite(mid_btjd) or not np.isfinite(freq) or freq <= 0:
            invalid_rows.append(
                f"row {int(row_index) + 2}: sector={sec}, mid_btjd={mid_btjd}, frequency={freq}"
            )
            continue
        if sec in out:
            raise ValueError(f"Orbital-frequency table contains duplicate sector {sec}: {p}")
        out[sec] = (mid_btjd, freq)

    if invalid_rows:
        preview = "; ".join(invalid_rows[:5])
        if len(invalid_rows) > 5:
            preview += f"; plus {len(invalid_rows) - 5} more"
        raise ValueError(f"Orbital-frequency table contains invalid rows ({preview}): {p}")
    if not out:
        raise ValueError(f"Orbital-frequency table loaded but no valid rows found: {p}")
    return out


def infer_sector_from_tpf(tpf_path: Path, tpf_obj=None):
    """Infer TESS sector number from TPF metadata or filename."""
    # 1) Try Lightkurve meta/header
    if tpf_obj is not None:
        for key in ("SECTOR", "sector"):
            try:
                if hasattr(tpf_obj, "meta") and key in tpf_obj.meta:
                    return int(tpf_obj.meta[key])
            except Exception:
                pass
        # Some LK objects expose a header-like dict
        for attr in ("header", "hdu"):
            try:
                h = getattr(tpf_obj, attr, None)
                if h is None:
                    continue
                if isinstance(h, dict) and "SECTOR" in h:
                    return int(h["SECTOR"])
            except Exception:
                pass

    # 2) Parse from filename patterns: ...-s0074-... or ..._s0074_... or ...s0074...
    s = tpf_path.name
    m = re.search(r"[\-_]s(\d{4})[\-_]", s)
    if m:
        return int(m.group(1))
    m = re.search(r"s(\d{4})", s)
    if m:
        return int(m.group(1))
    return None


def is_heavily_saturated(mean_img, thresh: float, min_npix: int):
    """Heuristic for heavy saturation: many pixels above a high flux threshold."""
    img = np.asarray(mean_img, float)
    n = int(np.count_nonzero(np.isfinite(img) & (img >= thresh)))
    return n >= int(min_npix), n


def estimate_background_faint_pixels(flux_cube, n_faint: int = 20):
    """Per-cadence background estimate = mean of faintest n_faint pixels."""
    f = np.asarray(flux_cube, float)
    nt, ny, nx = f.shape
    flat = f.reshape(nt, ny * nx)
    # Replace non-finite with +inf so they won't appear among faintest.
    flat2 = np.where(np.isfinite(flat), flat, np.inf)
    n_use = max(1, min(int(n_faint), flat2.shape[1]))
    # Use partial selection (O(N)) per row
    part = np.partition(flat2, n_use - 1, axis=1)[:, :n_use]
    # If a cadence is all inf (unlikely), back becomes inf; fix to nan
    back = np.nanmean(np.where(np.isfinite(part), part, np.nan), axis=1)
    return back


def optimize_background_scale(lc_raw, back, k_max_factor: float = 2.0, n_grid: int = 200):
    """Choose scalar k to minimize HF metric of (lc_raw - k*back)."""
    lc = np.asarray(lc_raw, float)
    b = np.asarray(back, float)
    good = np.isfinite(lc) & np.isfinite(b)
    if good.sum() < 100:
        return 0.0
    med_lc = np.nanmedian(lc[good])
    med_b = np.nanmedian(b[good])
    if not np.isfinite(med_lc) or not np.isfinite(med_b) or med_b == 0:
        return 0.0
    scale = med_lc / med_b
    # Search around 0..k_max_factor*scale
    ks = np.linspace(0.0, float(k_max_factor) * float(scale), int(n_grid))
    best_k = 0.0
    best_m = np.inf
    for k in ks:
        temp = lc - k * b
        med = np.nanmedian(temp[good])
        temp_rel = temp / med if np.isfinite(med) and med != 0 else temp
        m = hf_metric_from_flux(temp_rel)
        if np.isfinite(m) and m < best_m:
            best_m = m
            best_k = float(k)
    return best_k


def phase_template_detrend(flux, time_btjd, freq_cyc_per_day, phase_bin: float = 0.01):
    """Detrend flux by subtracting a phase-binned PCHIP template."""
    f = np.asarray(flux, float)
    t = np.asarray(time_btjd, float)
    ph = (t * float(freq_cyc_per_day)) % 1.0

    # Bin edges/centers
    binw = float(phase_bin)
    edges = np.arange(0.0, 1.0 + binw, binw)
    centers = edges[:-1] + 0.5 * binw

    # Assign to bins
    idx = np.digitize(ph, edges) - 1
    b = np.full_like(centers, np.nan, dtype=float)
    for i in range(len(centers)):
        m = idx == i
        if np.any(m):
            b[i] = np.nanmean(f[m])

    # Fill gaps by interpolation on the circle:
    # Do a simple fill: linear interp across valid centers, then PCHIP.
    ok = np.isfinite(b)
    if ok.sum() < 4:
        # Too few points for a meaningful template
        return f.copy(), ph, None

    x = centers[ok]
    y = b[ok]
    # Ensure periodic continuity by duplicating around 0/1 if needed
    x2 = np.concatenate([x - 1.0, x, x + 1.0])
    y2 = np.concatenate([y, y, y])
    o = np.argsort(x2)
    x2, y2 = x2[o], y2[o]

    pchip = PchipInterpolator(x2, y2, extrapolate=True)
    trend = pchip(ph)
    med = np.nanmedian(f)
    f_det = (f - trend) + med
    return f_det, ph, trend


def build_design_matrix_with_back(t, x, y, back, knot_spacing_days: float = 1.0, psf_sigma=None):
    """Design matrix including background regressor (and quadratic term)."""
    X = build_design_matrix(t, x, y, knot_spacing_days=knot_spacing_days, psf_sigma=psf_sigma)
    b = np.asarray(back, float)
    b0 = np.nanmedian(b)
    bb = b - b0
    # Append background terms
    return np.hstack([X, bb[:, None], (bb**2)[:, None]])

# =============================================================================
# Metrics + helpers
# =============================================================================

def amp_metric_from_flux(flux_rel, q_lo: float = 1, q_hi: float = 99):
    f = np.asarray(flux_rel, float)
    f = f[np.isfinite(f)]
    if f.size < 10:
        return np.nan
    return np.nanpercentile(f, q_hi) - np.nanpercentile(f, q_lo)


def hf_metric_from_flux(flux_rel):
    f = np.asarray(flux_rel, float)
    f = f[np.isfinite(f)]
    if f.size < 10:
        return np.nan
    return np.nanstd(np.diff(f))


def std_metric_from_flux(flux_rel):
    f = np.asarray(flux_rel, float)
    f = f[np.isfinite(f)]
    if f.size < 10:
        return np.nan
    return np.nanstd(f)


def mad_metric_from_flux(flux_rel):
    f = np.asarray(flux_rel, float)
    f = f[np.isfinite(f)]
    if f.size < 10:
        return np.nan
    med = np.nanmedian(f)
    return np.nanmean(np.abs(f - med))


def aperture_metric_from_flux(flux_rel, mode: str | None = None):
    mode = str(APERTURE_FOM_MODE if mode is None else mode).strip().lower()
    if mode == "std":
        return std_metric_from_flux(flux_rel)
    if mode == "mad":
        return mad_metric_from_flux(flux_rel)
    return hf_metric_from_flux(flux_rel)


def metric_from_lc(lc):
    med = np.nanmedian(lc)
    if not np.isfinite(med) or med == 0:
        return np.inf
    rel = lc / med
    return aperture_metric_from_flux(rel)


def get_neighbor_pixels(ap_set, ny: int, nx: int, allowed_mask):
    neighbors = set()
    for (iy, ix) in ap_set:
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                jy, jx = iy + dy, ix + dx
                if 0 <= jy < ny and 0 <= jx < nx and (jy, jx) not in ap_set:
                    if allowed_mask[jy, jx]:
                        neighbors.add((jy, jx))
    return neighbors


def pixel_radius_from_seed(seed_pix, iy: int, ix: int):
    return np.hypot(float(iy - seed_pix[0]), float(ix - seed_pix[1]))

# =============================================================================
# Gaia + Voronoi (version-safe cone search)
# =============================================================================

def gaia_brightest_sources_near(tpf, radius_arcmin: float = 6.0):
    if Gaia is None:
        raise ImportError(
            "Gaia target discovery requires astroquery. Install astroquery, "
            "or use --no-gaia for a single-target image-based extraction."
        )

    ra0 = dec0 = None
    for k in ("RA_OBJ", "RA", "ra"):
        if hasattr(tpf, "meta") and k in tpf.meta:
            try:
                ra0 = float(tpf.meta[k])
                break
            except Exception:
                pass
    for k in ("DEC_OBJ", "DEC", "dec"):
        if hasattr(tpf, "meta") and k in tpf.meta:
            try:
                dec0 = float(tpf.meta[k])
                break
            except Exception:
                pass

    if ra0 is None or dec0 is None:
        cg = tpf.get_coordinates(cadence=0)
        if not isinstance(cg, SkyCoord):
            ra_grid, dec_grid = cg
            cg = SkyCoord(
                ra=np.asarray(ra_grid, float) * u.deg,
                dec=np.asarray(dec_grid, float) * u.deg,
            )
        ra0 = np.nanmedian(cg.ra.deg)
        dec0 = np.nanmedian(cg.dec.deg)

    center = SkyCoord(ra=ra0 * u.deg, dec=dec0 * u.deg)
    radius = float(radius_arcmin) * u.arcmin

    # Make sure astroquery does not silently truncate the Gaia result set.
    Gaia.ROW_LIMIT = -1

    # Use keyword args (astroquery signature changed across versions)
    job = Gaia.cone_search_async(coordinate=center, radius=radius)
    tab = job.get_results()
    print(f"  Gaia cone search returned {len(tab)} rows before inside-stamp filtering")
    if "phot_g_mean_mag" not in tab.colnames:
        raise RuntimeError("Gaia result missing phot_g_mean_mag")
    tab = tab[np.argsort(tab["phot_g_mean_mag"])]
    return tab


def sources_inside_stamp(tpf, gaia_tab, cadence_idx: int):
    ny, nx = tpf.flux.shape[1], tpf.flux.shape[2]

    keep = []
    for row in gaia_tab:
        tc = SkyCoord(ra=float(row["ra"]) * u.deg, dec=float(row["dec"]) * u.deg)
        try:
            x, y = tpf.wcs.world_to_pixel(tc)
        except Exception:
            keep.append(False)
            continue

        inside = (
            np.isfinite(x) and np.isfinite(y) and
            (-0.5 <= x <= nx - 0.5) and
            (-0.5 <= y <= ny - 0.5)
        )
        keep.append(bool(inside))

    keep = np.array(keep, dtype=bool)
    gaia_in = gaia_tab[keep]

    if len(gaia_in) > 0 and "phot_g_mean_mag" in gaia_in.colnames:
        gaia_in = gaia_in[np.argsort(gaia_in["phot_g_mean_mag"])]

    coords_grid = tpf.get_coordinates(cadence=cadence_idx)
    if not isinstance(coords_grid, SkyCoord):
        ra_grid, dec_grid = coords_grid
        coords_grid = SkyCoord(
            ra=np.asarray(ra_grid, float) * u.deg,
            dec=np.asarray(dec_grid, float) * u.deg,
        )

    return gaia_in, coords_grid


def coordinate_grid_for_flux(tpf, cadence_idx: int, flux: np.ndarray, mean_img: np.ndarray):
    """Return a SkyCoord pixel grid, cropping it and the images consistently."""
    coords_grid = tpf.get_coordinates(cadence=cadence_idx)
    if not isinstance(coords_grid, SkyCoord):
        ra_grid, dec_grid = coords_grid
        coords_grid = SkyCoord(
            ra=np.asarray(ra_grid, float) * u.deg,
            dec=np.asarray(dec_grid, float) * u.deg,
        )

    ny_flux, nx_flux = flux.shape[1], flux.shape[2]
    ny_cg, nx_cg = coords_grid.shape
    if (ny_cg, nx_cg) != (ny_flux, nx_flux):
        ny0 = min(ny_cg, ny_flux)
        nx0 = min(nx_cg, nx_flux)
        print(
            f"  [WARN] Shape mismatch coords_grid={coords_grid.shape} vs "
            f"flux={(ny_flux, nx_flux)}; cropping to {(ny0, nx0)}"
        )
        coords_grid = coords_grid[:ny0, :nx0]
        flux = flux[:, :ny0, :nx0]
        mean_img = mean_img[:ny0, :nx0]
    return coords_grid, flux, mean_img


def nearest_pixel_to_coord(coords_grid: SkyCoord, target_coord: SkyCoord):
    sep = coords_grid.separation(target_coord).arcsec
    idx = int(np.nanargmin(sep))
    ny, nx = coords_grid.shape
    return divmod(idx, nx)


def watershed_owner_map(mean_image: np.ndarray, seeds: list[tuple[int, int]], smooth_sigma: float = 1.0):
    """
    Build a watershed segmentation map from the mean image.

    The watershed is used only to define an initial candidate region for each target.
    Final apertures are still chosen by the existing connected pixel-growth optimizer.
    """
    img = np.asarray(mean_image, float)
    ny, nx = img.shape

    finite = np.isfinite(img)
    if not finite.any():
        raise RuntimeError("Mean image has no finite pixels.")

    vals = img[finite]
    bg = np.nanmedian(vals)
    mad = np.nanmedian(np.abs(vals - bg))
    sigma = 1.4826 * mad if np.isfinite(mad) else 0.0
    peak = np.nanmax(vals)

    # Conservative segmentation mask: keep significant positive structure but do not
    # exclude saturated plateaus. This only defines the candidate pool.
    thresh1 = bg + sigma
    thresh2 = bg + 0.02 * (peak - bg) if np.isfinite(peak) else thresh1
    thresh = max(thresh1, thresh2) if np.isfinite(thresh1) and np.isfinite(thresh2) else thresh1
    mask = finite & (img > thresh)

    # Never lose the seeds.
    for (iy, ix) in seeds:
        if 0 <= iy < ny and 0 <= ix < nx:
            mask[iy, ix] = True

    # If the threshold was too aggressive, fall back to all finite positive pixels.
    if not np.any(mask):
        mask = finite & (img > bg)
        for (iy, ix) in seeds:
            if 0 <= iy < ny and 0 <= ix < nx:
                mask[iy, ix] = True

    # Final fallback: all finite pixels.
    if not np.any(mask):
        mask = finite.copy()

    img_fill = np.array(img, copy=True)
    img_fill[~finite] = bg if np.isfinite(bg) else 0.0
    img_smooth = gaussian_filter(img_fill, smooth_sigma)

    markers = np.zeros((ny, nx), dtype=np.int32)
    for k, (iy, ix) in enumerate(seeds, start=1):
        if not (0 <= iy < ny and 0 <= ix < nx):
            continue
        markers[iy, ix] = k

    if not np.any(markers):
        raise RuntimeError("No valid watershed markers were created from the target seeds.")

    # Watershed on the negative image so bright structures define basins.
    labels = watershed(-img_smooth, markers=markers, mask=mask)

    # Convert to zero-based owner map for downstream code.
    owner = labels.astype(int) - 1

    # Ensure every seed belongs to its own region even in pathological cases.
    for k, (iy, ix) in enumerate(seeds):
        if 0 <= iy < ny and 0 <= ix < nx:
            owner[iy, ix] = k

    return owner
# =============================================================================
# Optional PSF-width proxy
# =============================================================================

def psf_width_sigma(frames, ap_mask, crow, ccol):
    """Second-moment PSF-width proxy in pixels, within an aperture.

    frames: (nt, ny, nx)
    ap_mask: (ny, nx) bool
    crow, ccol: (nt,) centroid row/col (float)

    Returns: (nt,) sigma-like width.
    """
    frames = np.asarray(frames, float)
    ap = np.asarray(ap_mask, bool)
    nt, ny, nx = frames.shape
    yy, xx = np.indices((ny, nx))

    f = frames[:, ap]  # (nt, npix)
    if f.size == 0:
        return np.full(nt, np.nan)

    ypix = yy[ap][None, :]
    xpix = xx[ap][None, :]

    denom = np.nansum(f, axis=1)
    denom = np.where(np.isfinite(denom) & (denom != 0), denom, np.nan)

    dy2 = (ypix - np.asarray(crow, float)[:, None]) ** 2
    dx2 = (xpix - np.asarray(ccol, float)[:, None]) ** 2
    r2 = dy2 + dx2

    m2 = np.nansum(f * r2, axis=1) / denom
    return np.sqrt(m2)


# =============================================================================
# Aperture growth methods
# =============================================================================

def grow_aperture_in_region(
    flux_good,
    time_good,
    seed_pix,
    allowed_mask,
    min_pixels=10,
    amp_q_lo=1,
    amp_q_hi=99,
    amp_min_frac=0.01,
    max_radius_pix=np.inf,
):
    nt, ny, nx = flux_good.shape
    mean_img = np.nanmean(flux_good, axis=0)

    lc_ref = np.nansum(flux_good[:, allowed_mask], axis=1)
    med_ref = np.nanmedian(lc_ref)
    lc_ref_rel = lc_ref / med_ref if np.isfinite(med_ref) and med_ref != 0 else lc_ref
    A_ref = amp_metric_from_flux(lc_ref_rel, amp_q_lo, amp_q_hi)
    N_ref = hf_metric_from_flux(lc_ref_rel)
    A_min_allowed = amp_min_frac * A_ref if np.isfinite(A_ref) else -np.inf

    iy0, ix0 = seed_pix
    if not allowed_mask[iy0, ix0]:
        raise RuntimeError("Seed pixel is not inside allowed region.")

    ap_set = {(iy0, ix0)}
    lc_current = flux_good[:, iy0, ix0].copy()
    metric_current = metric_from_lc(lc_current)

    while True:
        neighbors = get_neighbor_pixels(ap_set, ny, nx, allowed_mask)
        if not neighbors:
            break

        best_neighbor = None
        best_metric = np.inf
        best_lc = None

        for (iy, ix) in neighbors:
            if pixel_radius_from_seed(seed_pix, iy, ix) > max_radius_pix:
                continue
            lc_trial = lc_current + flux_good[:, iy, ix]
            med_trial = np.nanmedian(lc_trial)
            if not np.isfinite(med_trial) or med_trial == 0:
                continue
            lc_trial_rel = lc_trial / med_trial
            A_trial = amp_metric_from_flux(lc_trial_rel, amp_q_lo, amp_q_hi)
            if (not np.isfinite(A_trial)) or (A_trial < A_min_allowed):
                continue

            m_trial = metric_from_lc(lc_trial)
            if m_trial < best_metric:
                best_metric = m_trial
                best_neighbor = (iy, ix)
                best_lc = lc_trial

        if best_neighbor is None:
            break

        improved = best_metric < metric_current
        if improved or (len(ap_set) < min_pixels):
            ap_set.add(best_neighbor)
            lc_current = best_lc
            metric_current = best_metric
        else:
            break

    ap_mask = np.zeros((ny, nx), bool)
    for (iy, ix) in ap_set:
        ap_mask[iy, ix] = True

    lc_final = lc_current
    med_final = np.nanmedian(lc_final)
    lc_final_rel = lc_final / med_final if np.isfinite(med_final) and med_final != 0 else lc_final
    A_final = amp_metric_from_flux(lc_final_rel, amp_q_lo, amp_q_hi)
    N_final = hf_metric_from_flux(lc_final_rel)

    return time_good, lc_final, ap_mask, A_ref, N_ref, A_final, N_final, mean_img

def grow_aperture_multi_component_in_region(
    flux_good,
    time_good,
    seed_pix,
    allowed_mask,
    min_pixels=10,
    amp_q_lo=1,
    amp_q_hi=99,
    amp_min_frac=0.01,
    max_components=3,
    min_seed_frac_of_peak=0.15,
    min_new_pixels_per_component=1,
    max_radius_pix=np.inf,
):
    nt, ny, nx = flux_good.shape
    mean_img = np.nanmean(flux_good, axis=0)

    lc_ref = np.nansum(flux_good[:, allowed_mask], axis=1)
    med_ref = np.nanmedian(lc_ref)
    lc_ref_rel = lc_ref / med_ref if np.isfinite(med_ref) and med_ref != 0 else lc_ref
    A_ref = amp_metric_from_flux(lc_ref_rel, amp_q_lo, amp_q_hi)
    N_ref = hf_metric_from_flux(lc_ref_rel)
    A_min_allowed = amp_min_frac * A_ref if np.isfinite(A_ref) else -np.inf

    ap_union = np.zeros((ny, nx), dtype=bool)

    allowed_vals = mean_img[allowed_mask & np.isfinite(mean_img)]
    peak_allowed = np.nanmax(allowed_vals) if allowed_vals.size else np.nan

    next_seed = seed_pix
    n_comp = 0

    while n_comp < max_components:
        n_comp += 1

        if (
            next_seed is None
            or (not allowed_mask[next_seed[0], next_seed[1]])
            or ap_union[next_seed[0], next_seed[1]]
        ):
            rem = allowed_mask & (~ap_union) & np.isfinite(mean_img)
            if not np.any(rem):
                break
            idx = int(np.nanargmax(mean_img * rem))
            next_seed = divmod(idx, nx)

        if np.isfinite(peak_allowed):
            rem = allowed_mask & (~ap_union) & np.isfinite(mean_img)
            if np.any(rem):
                rem_peak = np.nanmax(mean_img[rem])
                if rem_peak < min_seed_frac_of_peak * peak_allowed:
                    break

        allowed_this = allowed_mask & (~ap_union)
        if not allowed_this[next_seed[0], next_seed[1]]:
            break

        t_g, lc_comp, ap_comp, *_ = grow_aperture_in_region(
            flux_good,
            time_good,
            seed_pix=next_seed,
            allowed_mask=allowed_this,
            min_pixels=min_pixels,
            amp_q_lo=amp_q_lo,
            amp_q_hi=amp_q_hi,
            amp_min_frac=amp_min_frac,
            max_radius_pix=max_radius_pix,
        )

        new_pix = ap_comp & (~ap_union)
        if int(np.count_nonzero(new_pix)) < min_new_pixels_per_component:
            break

        ap_union |= ap_comp

        rem = allowed_mask & (~ap_union) & np.isfinite(mean_img)
        if not np.any(rem):
            break
        idx = int(np.nanargmax(mean_img * rem))
        next_seed = divmod(idx, nx)

    lc_final = np.nansum(flux_good[:, ap_union], axis=1)

    med_final = np.nanmedian(lc_final)
    lc_final_rel = lc_final / med_final if np.isfinite(med_final) and med_final != 0 else lc_final
    A_final = amp_metric_from_flux(lc_final_rel, amp_q_lo, amp_q_hi)
    N_final = hf_metric_from_flux(lc_final_rel)

    return time_good, lc_final, ap_union, A_ref, N_ref, A_final, N_final, mean_img

def grow_aperture_bright_core_preseed(
    flux_good,
    time_good,
    seed_pix,
    allowed_mask,
    min_pixels=10,
    amp_q_lo=1,
    amp_q_hi=99,
    amp_min_frac=0.01,
    core_npix=12,
    core_min_frac_of_peak=0.25,
    max_radius_pix=np.inf,
):
    nt, ny, nx = flux_good.shape
    mean_img = np.nanmean(flux_good, axis=0)

    region_vals = mean_img[allowed_mask & np.isfinite(mean_img)]
    if region_vals.size == 0:
        raise RuntimeError("Allowed region has no finite mean_img pixels.")
    peak = np.nanmax(region_vals)

    cand_mask = allowed_mask & np.isfinite(mean_img) & (mean_img >= core_min_frac_of_peak * peak)
    if not np.any(cand_mask):
        cand_mask = allowed_mask & np.isfinite(mean_img)
    if not cand_mask[seed_pix]:
        cand_mask = cand_mask.copy()
        cand_mask[seed_pix] = bool(allowed_mask[seed_pix] and np.isfinite(mean_img[seed_pix]))

    core_pix = [seed_pix]
    core_set = {seed_pix}

    while len(core_pix) < core_npix:
        neighbors = get_neighbor_pixels(core_set, ny, nx, cand_mask)
        if not neighbors:
            break

        best = None
        best_val = -np.inf

        for (iy, ix) in neighbors:
            if pixel_radius_from_seed(seed_pix, iy, ix) > max_radius_pix:
                continue
            val = mean_img[iy, ix]
            if np.isfinite(val) and val > best_val:
                best_val = val
                best = (iy, ix)

        if best is None:
            break

        core_pix.append(best)
        core_set.add(best)

    ap_set = set(core_pix)
    lc_current = np.nansum(
        flux_good[:, [p[0] for p in ap_set], [p[1] for p in ap_set]], axis=1
    )
    metric_current = metric_from_lc(lc_current)

    lc_ref = np.nansum(flux_good[:, allowed_mask], axis=1)
    med_ref = np.nanmedian(lc_ref)
    lc_ref_rel = lc_ref / med_ref if np.isfinite(med_ref) and med_ref != 0 else lc_ref
    A_ref = amp_metric_from_flux(lc_ref_rel, amp_q_lo, amp_q_hi)
    N_ref = hf_metric_from_flux(lc_ref_rel)
    A_min_allowed = amp_min_frac * A_ref if np.isfinite(A_ref) else -np.inf

    while True:
        neighbors = get_neighbor_pixels(ap_set, ny, nx, allowed_mask)
        if not neighbors:
            break

        best_neighbor = None
        best_metric = np.inf
        best_lc = None

        for (iy, ix) in neighbors:
            if pixel_radius_from_seed(seed_pix, iy, ix) > max_radius_pix:
                continue
            lc_trial = lc_current + flux_good[:, iy, ix]
            med_trial = np.nanmedian(lc_trial)
            if not np.isfinite(med_trial) or med_trial == 0:
                continue
            lc_trial_rel = lc_trial / med_trial
            A_trial = amp_metric_from_flux(lc_trial_rel, amp_q_lo, amp_q_hi)
            if (not np.isfinite(A_trial)) or (A_trial < A_min_allowed):
                continue

            m_trial = metric_from_lc(lc_trial)
            if m_trial < best_metric:
                best_metric = m_trial
                best_neighbor = (iy, ix)
                best_lc = lc_trial

        if best_neighbor is None:
            break

        improved = best_metric < metric_current
        if improved or (len(ap_set) < min_pixels):
            ap_set.add(best_neighbor)
            lc_current = best_lc
            metric_current = best_metric
        else:
            break

    ap_mask = np.zeros((ny, nx), bool)
    for (iy, ix) in ap_set:
        ap_mask[iy, ix] = True

    lc_final = lc_current
    med_final = np.nanmedian(lc_final)
    lc_final_rel = lc_final / med_final if np.isfinite(med_final) and med_final != 0 else lc_final
    A_final = amp_metric_from_flux(lc_final_rel, amp_q_lo, amp_q_hi)
    N_final = hf_metric_from_flux(lc_final_rel)

    return time_good, lc_final, ap_mask, A_ref, N_ref, A_final, N_final, mean_img

# =============================================================================
# TPF discovery
# =============================================================================

def find_tpfs(search_dir: Path, recursive: bool = False):
    pats = [
        # TESS SPOC/TESS-SPOC target-pixel products and TESSCut/Astrocut files.
        "*tp.fits", "*tpf.fits", "*tp.fits.gz", "*tpf.fits.gz",
        "*_astrocut.fits", "*_astrocut.fits.gz",
        # Kepler and K2 long-/short-cadence target-pixel products.
        "*_lpd-targ.fits", "*_spd-targ.fits",
        "*_lpd-targ.fits.gz", "*_spd-targ.fits.gz",
    ]
    out = []
    if recursive:
        for pat in pats:
            out.extend(search_dir.rglob(pat))
    else:
        for pat in pats:
            out.extend(search_dir.glob(pat))
    return sorted({p.resolve() for p in out})


def _first_tpf_metadata_value(tpf, keys):
    """Return the first non-empty value from Lightkurve metadata/FITS headers."""
    if tpf is None:
        return None
    meta = getattr(tpf, "meta", None)
    if meta is not None:
        for key in keys:
            try:
                if key in meta and meta[key] not in (None, ""):
                    return meta[key]
            except Exception:
                pass
    try:
        hdu = getattr(tpf, "hdu", None)
        if hdu is not None:
            for item in hdu[:3]:
                hdr = getattr(item, "header", None)
                if hdr is None:
                    continue
                for key in keys:
                    val = hdr.get(key)
                    if val not in (None, ""):
                        return val
    except Exception:
        pass
    return None


def _first_fits_header_value(path: Path, keys):
    try:
        with fits.open(path, memmap=True) as hdul:
            for hdu in hdul[:3]:
                for key in keys:
                    val = hdu.header.get(key)
                    if val not in (None, ""):
                        return val
    except Exception:
        pass
    return None


def infer_mission(tpf_path: Path, tpf=None) -> str:
    """Infer TESS, KEPLER, or K2 from metadata, class, and filename."""
    raw = _first_tpf_metadata_value(tpf, ("MISSION", "TELESCOP", "OBSERVAT"))
    if raw is None:
        raw = _first_fits_header_value(tpf_path, ("MISSION", "TELESCOP", "OBSERVAT"))
    text = str(raw or "").strip().upper()
    name = tpf_path.name.lower()
    cls = type(tpf).__name__.lower() if tpf is not None else ""

    # K2 uses KeplerTargetPixelFile, so campaign/filename checks must precede
    # the generic Kepler class/name check.
    campaign = _first_tpf_metadata_value(tpf, ("CAMPAIGN",))
    if campaign is None:
        campaign = _first_fits_header_value(tpf_path, ("CAMPAIGN",))
    if "K2" in text or name.startswith("ktwo") or campaign not in (None, ""):
        return "K2"
    if "TESS" in text or "tess" in cls or name.startswith("tess") or "astrocut" in name:
        return "TESS"
    if "KEPLER" in text or "kepler" in cls or name.startswith("kplr"):
        return "KEPLER"
    return "UNKNOWN"


def is_tesscut_product(tpf_path: Path, tpf=None) -> bool:
    """Identify TESSCut/Astrocut stamps, which do not have a SPOC target."""
    if "astrocut" in tpf_path.name.lower():
        return True
    raw = _first_tpf_metadata_value(tpf, ("CREATOR", "PROCNAME", "ORIGIN"))
    if raw is None:
        raw = _first_fits_header_value(tpf_path, ("CREATOR", "PROCNAME", "ORIGIN"))
    text = str(raw or "").lower()
    return "astrocut" in text or "tesscut" in text


def prf_saturation_tessmag(tpf_path: Path, tpf=None):
    """Return a reliable target TESS magnitude, or None for targetless cutouts."""
    if is_tesscut_product(tpf_path, tpf):
        return None
    value = _first_tpf_metadata_value(tpf, ("TESSMAG", "TMAG"))
    if value is None:
        value = _first_fits_header_value(tpf_path, ("TESSMAG", "TMAG"))
    try:
        value = float(value)
        return value if np.isfinite(value) else None
    except Exception:
        return None


def tpf_flux_unit(tpf):
    """Return the Lightkurve flux-cube unit when available."""
    try:
        unit = getattr(getattr(tpf, "flux", None), "unit", None)
        return str(unit) if unit is not None and str(unit).strip() else None
    except Exception:
        return None


def infer_native_time_system(tpf_path: Path, tpf=None, mission: str | None = None) -> str:
    """Return the native numeric time convention used by the loaded TPF."""
    try:
        fmt = str(getattr(getattr(tpf, "time", None), "format", "")).strip().upper()
        if fmt in {"BTJD", "BKJD", "JD", "MJD"}:
            return fmt
    except Exception:
        pass
    bjdref = _first_tpf_metadata_value(tpf, ("BJDREFI",))
    if bjdref is None:
        bjdref = _first_fits_header_value(tpf_path, ("BJDREFI",))
    try:
        bjdref = int(round(float(bjdref)))
        if bjdref == 2457000:
            return "BTJD"
        if bjdref == 2454833:
            return "BKJD"
    except Exception:
        pass
    mission = str(mission or infer_mission(tpf_path, tpf)).upper()
    if mission == "TESS":
        return "BTJD"
    if mission in {"KEPLER", "K2"}:
        return "BKJD"
    return "JD"


def infer_observation_tag(tpf_path: Path, tpf=None) -> str:
    """Return s#### for TESS, q## for Kepler, or c## for K2."""
    mission = infer_mission(tpf_path, tpf)
    if mission == "TESS":
        sec = infer_sector_from_tpf(tpf_path, tpf)
        if sec is not None:
            return f"s{int(sec):04d}"
        m = re.search(r"s(\d{4})", tpf_path.name.lower())
        return m.group(0) if m else "tess_no_sector"

    if mission == "KEPLER":
        q = _first_tpf_metadata_value(tpf, ("QUARTER",))
        if q is None:
            q = _first_fits_header_value(tpf_path, ("QUARTER",))
        try:
            return f"q{int(q):02d}"
        except Exception:
            return "kepler_no_quarter"

    if mission == "K2":
        c = _first_tpf_metadata_value(tpf, ("CAMPAIGN",))
        if c is None:
            c = _first_fits_header_value(tpf_path, ("CAMPAIGN",))
        try:
            return f"c{int(c):02d}"
        except Exception:
            m = re.search(r"[-_]c(\d{1,3})", tpf_path.name.lower())
            return f"c{int(m.group(1)):02d}" if m else "k2_no_campaign"

    return "unknown_observation"


def make_lightcurve_dataframe(time, flux, mission: str, time_system: str, flux_name: str):
    """Create a mission-aware, backward-compatible light-curve table.

    Kepler/K2 retain native BKJD and also include an exact BTJD conversion so
    the existing TESS-oriented detrender can continue to consume the files.
    """
    t = np.asarray(time, float)
    f = np.asarray(flux, float)
    mission = str(mission or "UNKNOWN").upper()
    system = str(time_system or "JD").upper()
    data = {}
    if system == "BKJD":
        data["time_bkjd"] = t
        data["time_btjd"] = t - 2167.0  # (BJD-2454833) -> (BJD-2457000)
    elif system == "BTJD":
        data["time_btjd"] = t
    elif system == "MJD":
        data["time_mjd"] = t
        data["time_btjd"] = t - 56999.5
    else:
        data["time_jd"] = t
        data["time_btjd"] = t - 2457000.0
    data[flux_name] = f
    data["mission"] = np.full(len(t), mission, dtype=object)
    data["time_system"] = np.full(len(t), system, dtype=object)
    return pd.DataFrame(data)


def mission_time_axis_label(time_system: str) -> str:
    return f"Time [{str(time_system or 'JD').upper()}]"


def sanitize_token(text: str) -> str:
    text = str(text).strip()
    text = text.replace("Gaia DR3 ", "GaiaDR3_")
    text = text.replace("Gaia DR2 ", "GaiaDR2_")
    text = text.replace("HD ", "HD_")
    text = re.sub(r"[^A-Za-z0-9._+-]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("._")
    return text or "unknown"


def is_catalog_like_name(name: str) -> bool:
    n = str(name).strip().upper()
    prefixes = (
        "GAIA", "HD ", "HIP ", "TYC ", "TIC ", "2MASS ", "UCAC", "WISE ", "BD ",
        "CD ", "CPD ", "GSC ", "APASS ", "SDSS ", "USNO ", "NAME ", "TOI ", "KIC ",
        "EPIC ", "ASAS ", "GJ ", "IRAS "
    )
    return n.startswith(prefixes)


def normalize_simbad_main_id(name: str) -> str:
    """Turn SIMBAD-style MAIN_ID values into stable filename-safe labels.

    This intentionally prefers the cleaned SIMBAD MAIN_ID itself (for example
    ``* alf UMi`` -> ``alf_UMi``) rather than searching for a more colloquial
    alias such as a familiar stellar name. That keeps filenames consistent
    across stars: e.g. ``alf_UMi``, ``alf_Lyr``, ``alf_Cyg``.
    """
    n = str(name).strip()
    for prefix in ("NAME ", "* ", "V* ", "EM* "):
        if n.upper().startswith(prefix.upper()):
            n = n[len(prefix):].strip()
            break
    return sanitize_token(n)


_SIMBAD_CACHE = {}


def _first_matching_col(table, candidates):
    lower_map = {str(c).lower(): str(c) for c in table.colnames}
    for cand in candidates:
        key = str(cand).lower()
        if key in lower_map:
            return lower_map[key]
    return None


def resolve_target_label(target_coord: SkyCoord, gaia_source_id: str | None = None) -> str:
    cache_key = gaia_source_id or f"{target_coord.ra.deg:.8f}_{target_coord.dec.deg:.8f}"
    if cache_key in _SIMBAD_CACHE:
        return _SIMBAD_CACHE[cache_key]

    fallback = sanitize_token(f"GaiaDR3_{gaia_source_id}" if gaia_source_id else "unknown_target")
    if Simbad is None:
        _SIMBAD_CACHE[cache_key] = fallback
        return fallback

    try:
        sim = Simbad()
        sim.add_votable_fields("ids")
        tab = sim.query_region(target_coord, radius=5 * u.arcsec)
    except Exception:
        _SIMBAD_CACHE[cache_key] = fallback
        return fallback

    if tab is None or len(tab) < 1:
        _SIMBAD_CACHE[cache_key] = fallback
        return fallback

    row = tab[0]

    ids_col = _first_matching_col(tab, ["IDS", "ids"])
    main_id_col = _first_matching_col(tab, ["MAIN_ID", "main_id", "MATCHED_ID", "matched_id"])

    ids = []
    if ids_col is not None and row[ids_col] is not None:
        ids = [x.strip() for x in str(row[ids_col]).split("|") if str(x).strip()]
    main_id = str(row[main_id_col]).strip() if main_id_col is not None else ""

    if main_id:
        val = normalize_simbad_main_id(main_id)
        if val and val.lower() not in ("name", "unknown") and not is_catalog_like_name(main_id):
            _SIMBAD_CACHE[cache_key] = val
            return val

    for ident in ids:
        if ident.upper().startswith("NAME "):
            val = sanitize_token(ident[5:])
            _SIMBAD_CACHE[cache_key] = val
            return val

    for ident in ids:
        if ident.upper().startswith("HD "):
            val = sanitize_token(ident)
            _SIMBAD_CACHE[cache_key] = val
            return val

    for ident in ids:
        if ident.upper().startswith("GAIA DR3 "):
            val = sanitize_token(ident)
            _SIMBAD_CACHE[cache_key] = val
            return val

    _SIMBAD_CACHE[cache_key] = fallback
    return fallback


def build_gaia_prf_scene(
    tpf,
    gaia_tab,
    *,
    target_source_id: str,
    target_label: str,
    target_gmag: float,
    target_row: float,
    target_column: float,
    shape: tuple[int, int],
    margin_pixels: float = 6.0,
    max_delta_mag: float = 8.0,
    max_sources: int = 20,
    duplicate_distance_pixels: float = 0.05,
):
    """Build a target-first Gaia source list for constrained PRF fitting.

    Sources just outside the stamp are retained when their centers lie within
    ``margin_pixels`` of an edge.  Neighbor labels deliberately use stable Gaia
    identifiers rather than making one SIMBAD query per contaminant.
    """
    ny, nx = map(int, shape)
    margin = max(0.0, float(margin_pixels))
    dmag = max(0.0, float(max_delta_mag))
    max_sources = max(1, int(max_sources))
    duplicate_distance = max(0.0, float(duplicate_distance_pixels))
    target_source_id = str(target_source_id)
    candidates = []
    for row in gaia_tab:
        try:
            sid = str(row["source_id"])
            mag = float(row["phot_g_mean_mag"])
            coord = SkyCoord(float(row["ra"]) * u.deg, float(row["dec"]) * u.deg)
            xpix, ypix = tpf.wcs.world_to_pixel(coord)
            xpix, ypix = float(xpix), float(ypix)
        except Exception:
            continue
        if not (np.isfinite(xpix) and np.isfinite(ypix) and np.isfinite(mag)):
            continue
        is_target = sid == target_source_id
        in_extended = (
            -margin <= xpix <= (nx - 1 + margin)
            and -margin <= ypix <= (ny - 1 + margin)
        )
        if not in_extended:
            continue
        if not is_target and np.isfinite(target_gmag) and mag > float(target_gmag) + dmag:
            continue
        label = target_label if is_target else sanitize_token(f"GaiaDR3_{sid}")
        candidates.append({
            "row": ypix, "column": xpix, "label": label,
            "source_id": sid, "magnitude": mag,
            "role": "target" if is_target else "neighbor",
            "is_target": is_target,
        })

    if not any(c["is_target"] for c in candidates):
        candidates.insert(0, {
            "row": float(target_row), "column": float(target_column),
            "label": target_label, "source_id": target_source_id,
            "magnitude": float(target_gmag), "role": "target", "is_target": True,
        })
    candidates.sort(key=lambda c: (not c["is_target"], c["magnitude"]))

    # Remove effectively duplicate catalog entries, always preserving the target
    # and otherwise the brighter source.
    deduped = []
    for candidate in candidates:
        duplicate = None
        for j, kept in enumerate(deduped):
            distance = np.hypot(candidate["row"] - kept["row"], candidate["column"] - kept["column"])
            if distance < duplicate_distance:
                duplicate = j
                break
        if duplicate is None:
            deduped.append(candidate)
        elif candidate["is_target"] and not deduped[duplicate]["is_target"]:
            deduped[duplicate] = candidate
    deduped.sort(key=lambda c: (not c["is_target"], c["magnitude"]))
    deduped = deduped[:max_sources]

    # Guarantee target-first ordering after the source cap.
    target_candidates = [c for c in deduped if c["is_target"]]
    neighbors = [c for c in deduped if not c["is_target"]]
    if target_candidates:
        ordered = [target_candidates[0]] + neighbors[:max_sources - 1]
    else:
        ordered = [{
            "row": float(target_row), "column": float(target_column),
            "label": target_label, "source_id": target_source_id,
            "magnitude": float(target_gmag), "role": "target", "is_target": True,
        }] + neighbors[:max_sources - 1]
    return [
        SceneSource(
            float(c["row"]), float(c["column"]), str(c["label"]),
            str(c["source_id"]), float(c["magnitude"]), str(c["role"]),
        )
        for c in ordered
    ]


def infer_sector_tag(tpf_path: Path, tpf=None) -> str:
    # Historical function name retained because many output-building call sites
    # use ``sector_tag``.  It now returns a mission-aware observation tag.
    return infer_observation_tag(tpf_path, tpf)


def build_output_stem(sector_tag: str, source_label: str, target_idx: int, method_tag: str) -> str:
    return f"{sanitize_token(sector_tag)}_{sanitize_token(source_label)}_target{int(target_idx)}_{sanitize_token(method_tag)}"




def build_prf_scene_output_label(source, *, is_primary: bool) -> str:
    """Filename label for a written PRF scene-source light curve.

    The selected primary keeps the established human-readable filename.  Other
    scene sources include both a resolved label, when available, and the Gaia
    DR3 identifier so output names remain unambiguous and reproducible.
    """
    label = sanitize_token(getattr(source, "label", "") or "unknown_source")
    source_id = str(getattr(source, "source_id", "") or "").strip()
    if is_primary or not source_id:
        return label
    gaia_label = sanitize_token(f"GaiaDR3_{source_id}")
    if label.lower() == gaia_label.lower() or label.lower().startswith("gaiadr3_"):
        return gaia_label
    return sanitize_token(f"{label}_{gaia_label}")


def resolve_prf_scene_output_names(tpf, prf_result, source_indices):
    """Resolve human-readable names only for scene sources being written.

    SIMBAD failures are harmless: the stable Gaia identifier remains the label.
    This is intentionally deferred until after output-eligibility pruning so an
    all-source run does not issue catalog queries for sources that will be skipped.
    """
    scene_rows = prf_result.metadata.get("reference_scene", {}).get("sources", [])
    positions = prf_result.metadata.get("source_positions", [])
    for j in source_indices:
        j = int(j)
        if j <= 0 or j >= len(prf_result.sources):
            continue
        source = prf_result.sources[j]
        try:
            coord = tpf.wcs.pixel_to_world(float(source.column), float(source.row))
            resolved = resolve_target_label(coord, source.source_id)
        except Exception:
            resolved = source.label
        if resolved:
            source.label = sanitize_token(resolved)
        if j < len(scene_rows):
            scene_rows[j]["label"] = source.label
        if j < len(positions):
            positions[j]["label"] = source.label

def fast_object_label_from_fits(tpf_path: Path) -> str:
    try:
        with fits.open(tpf_path, memmap=True) as hdul:
            for hdu in hdul[:2]:
                hdr = hdu.header
                for key in ("OBJECT", "TARGNAME", "TARGET", "TICID", "KEPLERID", "EPICID", "EPIC"):
                    val = hdr.get(key)
                    if val is None:
                        continue
                    sval = str(val).strip()
                    if sval and sval.lower() not in {"none", "nan"}:
                        return sanitize_token(sval)
    except Exception:
        pass
    m = re.search(r"-(\d{16})-", tpf_path.name)
    if m:
        return sanitize_token(f"TIC{m.group(1).lstrip('0')}")
    m = re.search(r"kplr(\d{9})", tpf_path.name.lower())
    if m:
        return sanitize_token(f"KIC_{int(m.group(1))}")
    m = re.search(r"ktwo(\d{9})", tpf_path.name.lower())
    if m:
        return sanitize_token(f"EPIC_{int(m.group(1))}")
    return sanitize_token("unknown_target")


def fast_full_region_sum_from_fits(tpf_path: Path, no_quality0: bool = False, block_cadences: int = 256):
    """Memory-safe full-stamp summation using FITS memmap blocks."""
    with fits.open(tpf_path, memmap=True) as hdul:
        data = hdul[1].data
        names = set(data.names)
        time = np.asarray(data["TIME"], dtype=float)
        if "QUALITY" in names:
            qual = np.asarray(data["QUALITY"])
        elif "DQUALITY" in names:
            qual = np.asarray(data["DQUALITY"])
        else:
            qual = np.zeros(len(time), dtype=int)

        keep = np.isfinite(time)
        if not no_quality0:
            keep &= (np.asarray(qual) == 0)
            if not np.any(keep):
                keep = np.isfinite(time)

        keep_idx = np.flatnonzero(keep)
        if keep_idx.size < 1:
            raise ValueError(f"No cadences available after filtering in {tpf_path.name}")

        flux_ref = data["FLUX"]
        if len(flux_ref.shape) != 3:
            raise ValueError(f"Expected 3D FLUX in {tpf_path.name}, got shape {flux_ref.shape}")
        _, ny, nx = flux_ref.shape

        yy, xx = np.indices((ny, nx), dtype=float)
        npix = ny * nx
        t_g = time[keep_idx].astype(float)
        lc_raw = np.empty(keep_idx.size, dtype=float)
        crow = np.full(keep_idx.size, np.nan, dtype=float)
        ccol = np.full(keep_idx.size, np.nan, dtype=float)
        sum_img = np.zeros((ny, nx), dtype=float)
        count_img = np.zeros((ny, nx), dtype=float)

        pos = 0
        for j0 in range(0, keep_idx.size, int(block_cadences)):
            rows = keep_idx[j0:j0 + int(block_cadences)]
            block = np.asarray(flux_ref[rows, :, :], dtype=float)
            finite = np.isfinite(block)
            sum_img += np.where(finite, block, 0.0).sum(axis=0)
            count_img += finite.sum(axis=0)

            flat = block.reshape(block.shape[0], -1)
            sums = np.nansum(flat, axis=1)
            lc_raw[pos:pos + len(rows)] = sums

            rownum = np.nansum(block * yy[None, :, :], axis=(1, 2))
            colnum = np.nansum(block * xx[None, :, :], axis=(1, 2))
            good = np.isfinite(sums) & (sums != 0)
            crow[pos:pos + len(rows)][good] = rownum[good] / sums[good]
            ccol[pos:pos + len(rows)][good] = colnum[good] / sums[good]
            pos += len(rows)

        mean_img = np.divide(sum_img, count_img, out=np.full_like(sum_img, np.nan), where=(count_img > 0))
        ap_mask = np.ones((ny, nx), dtype=bool)
        flat_idx = int(np.nanargmax(mean_img)) if np.isfinite(mean_img).any() else 0
        seed = (flat_idx // nx, flat_idx % nx)
        return t_g, lc_raw, mean_img, ap_mask, crow, ccol, seed


def masked_sum_and_centroid_blockwise(flux: np.ndarray, ap_mask: np.ndarray, yy: np.ndarray, xx: np.ndarray,
                                      target_block_elems: int = 2_000_000):
    """Memory-safe masked flux sum and centroid for large/full-stamp apertures.

    Works on flux with shape (ntime, nrow, ncol) and ap_mask with shape (nrow, ncol).
    Processes cadences in blocks to avoid materializing a giant boolean-indexed copy.
    """
    flux = np.asarray(flux, float)
    ap_mask = np.asarray(ap_mask, bool)
    if flux.ndim != 3:
        raise ValueError(f"Expected 3D flux cube, got shape {flux.shape}")
    if ap_mask.ndim != 2:
        raise ValueError(f"Expected 2D aperture mask, got shape {ap_mask.shape}")

    nt, nr, nc = flux.shape
    if ap_mask.shape != (nr, nc):
        raise ValueError(f"Mask shape {ap_mask.shape} does not match flux spatial shape {(nr, nc)}")

    sel = np.flatnonzero(ap_mask.ravel())
    if sel.size == 0:
        raise ValueError("Aperture mask contains no selected pixels.")

    ysel = np.asarray(yy, float).ravel()[sel]
    xsel = np.asarray(xx, float).ravel()[sel]
    flat_flux = flux.reshape(nt, nr * nc)

    block_nt = max(1, int(target_block_elems // max(1, sel.size)))
    sums = np.zeros(nt, dtype=float)
    crow = np.full(nt, np.nan, dtype=float)
    ccol = np.full(nt, np.nan, dtype=float)

    for start in range(0, nt, block_nt):
        stop = min(nt, start + block_nt)
        block = np.array(flat_flux[start:stop][:, sel], dtype=float, copy=True)
        np.nan_to_num(block, copy=False, nan=0.0)
        bsum = np.sum(block, axis=1)
        sums[start:stop] = bsum

        good = np.isfinite(bsum) & (bsum != 0)
        if np.any(good):
            brow = block @ ysel
            bcol = block @ xsel
            crow[start:stop][good] = brow[good] / bsum[good]
            ccol[start:stop][good] = bcol[good] / bsum[good]

    return sums, crow, ccol


def infer_header_target_coord(tpf):
    """Return the intended target coordinate from FITS metadata if available.

    Preference is given to target/object-specific header keywords rather than
    generic image-center information, because saturated stars can have bleed
    structures whose brightest pixel is offset from the true stellar position.
    """
    if not hasattr(tpf, "meta"):
        return None

    meta = tpf.meta

    ra_keys = ("RA_OBJ", "RA_TARG", "TARGRA", "OBJRA", "RA")
    dec_keys = ("DEC_OBJ", "DEC_TARG", "TARGDEC", "OBJDEC", "DEC")

    ra = dec = None

    for key in ra_keys:
        if key in meta and meta[key] not in (None, ""):
            try:
                ra = float(meta[key])
                break
            except Exception:
                pass

    for key in dec_keys:
        if key in meta and meta[key] not in (None, ""):
            try:
                dec = float(meta[key])
                break
            except Exception:
                pass

    if ra is None or dec is None:
        return None

    try:
        return SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")
    except Exception:
        return None


def prioritize_tesscut_center_source(gaia_in, tpf, image_shape):
    """Place the Gaia source nearest the requested TESSCut centre first.

    TESSCut stamps are sky cutouts rather than target-specific SPOC products.
    Their brightest in-stamp source can be far from the requested coordinate,
    especially for large cutouts.  Remaining sources retain the Gaia query's
    brightness order so multi-target behavior stays deterministic.
    """
    if gaia_in is None or len(gaia_in) < 2:
        return gaia_in, None

    center_coord = infer_header_target_coord(tpf)
    if center_coord is None:
        try:
            ny, nx = map(int, image_shape)
            center_coord = tpf.wcs.pixel_to_world((nx - 1) / 2.0, (ny - 1) / 2.0)
        except Exception:
            center_coord = None
    if center_coord is None:
        return gaia_in, None

    try:
        coords = SkyCoord(
            ra=np.asarray(gaia_in["ra"], float) * u.deg,
            dec=np.asarray(gaia_in["dec"], float) * u.deg,
        )
        separations = np.asarray(center_coord.separation(coords).arcsec, float)
        finite = np.isfinite(separations)
        if not np.any(finite):
            return gaia_in, None
        nearest = int(np.nanargmin(np.where(finite, separations, np.nan)))
        order = [nearest] + [i for i in range(len(gaia_in)) if i != nearest]
        return gaia_in[np.asarray(order, dtype=int)], float(separations[nearest])
    except Exception:
        return gaia_in, None


def infer_single_target_label(
    tpf,
    mean_image_2d=None,
    gaia_radius_arcmin: float = 12.0,
    allow_catalog: bool = True,
):
    """Infer a human-friendly label for a single-target or saturated-target run.

    Preference order:
      1) OBJECT/TARGNAME-style header metadata, if genuinely human-friendly
      2) SIMBAD/common-name resolution from the intended target coordinate in the FITS header
      3) SIMBAD/common-name resolution from the brightest image pixel (fallback only)
      4) TIC-style header metadata
      5) brightest in-stamp Gaia source
      6) unknown_target

    Returns (label, gaia_source_id_or_none, gaia_g_mag_or_nan).

    ``allow_catalog=False`` honors an explicit no-Gaia/offline workflow by
    skipping SIMBAD and Gaia network lookups while retaining local FITS
    metadata choices.
    """
    # First choice: useful human-friendly metadata if present.
    # Reject purely catalog-like values here so that SIMBAD/common-name
    # resolution gets a chance before we fall back to TIC/Gaia-style labels.
    for key in ("OBJECT", "OBJECT_ID", "TARGNAME", "TARGET", "LABEL"):
        try:
            if hasattr(tpf, "meta") and key in tpf.meta and tpf.meta[key] not in (None, ""):
                raw_val = str(tpf.meta[key]).strip()
                if not raw_val:
                    continue
                if is_catalog_like_name(raw_val):
                    continue
                val = sanitize_token(raw_val)
                if val and val.lower() not in ("unknown", "nan", "none"):
                    return val, None, np.nan
        except Exception:
            pass

    # Second choice: resolve the intended target coordinate from header metadata.
    # This is the right thing for saturated single-target files, where the
    # brightest pixel can be displaced from the stellar photocenter.
    if allow_catalog:
        try:
            tc_hdr = infer_header_target_coord(tpf)
            if tc_hdr is not None:
                label = resolve_target_label(tc_hdr, None)
                if label != "unknown_target":
                    return label, None, np.nan
        except Exception:
            pass

    # Third choice: resolve the brightest pixel position directly through SIMBAD.
    # This is only a fallback when there is no useful target coordinate in the
    # header or that lookup fails.
    if allow_catalog:
        try:
            if mean_image_2d is None:
                mean_image_2d = np.nanmean(np.asarray(tpf.flux, float), axis=0)
            if np.isfinite(mean_image_2d).any():
                iy, ix = np.unravel_index(int(np.nanargmax(mean_image_2d)), mean_image_2d.shape)
                tc = tpf.wcs.pixel_to_world(float(ix), float(iy))
                label = resolve_target_label(tc, None)
                if label != "unknown_target":
                    return label, None, np.nan
        except Exception:
            pass

    # Fourth choice: TIC-like metadata if present. This is preferred to a raw
    # Gaia DR3 id in single-target saturated workflows because it usually
    # reflects the intended target for the cutout.
    for key, prefix in (("TICID", "TIC"), ("TARGETID", "TIC"),
                        ("KEPLERID", "KIC"), ("KEPLER_ID", "KIC"),
                        ("EPICID", "EPIC"), ("EPIC", "EPIC")):
        try:
            if hasattr(tpf, "meta") and key in tpf.meta and tpf.meta[key] not in (None, ""):
                return sanitize_token(f"{prefix}_{tpf.meta[key]}"), None, np.nan
        except Exception:
            pass

    # Fifth choice: brightest Gaia source actually inside the stamp.
    if allow_catalog:
        try:
            gaia_tab = gaia_brightest_sources_near(tpf, radius_arcmin=max(float(gaia_radius_arcmin), 6.0))
            gaia_in, _ = sources_inside_stamp(tpf, gaia_tab, cadence_idx=0)
            if len(gaia_in) > 0:
                row = gaia_in[0]
                tc = SkyCoord(float(row["ra"]) * u.deg, float(row["dec"]) * u.deg)
                sid = str(row["source_id"])
                try:
                    gmag = float(row["phot_g_mean_mag"])
                except Exception:
                    gmag = np.nan
                return resolve_target_label(tc, sid), sid, gmag
        except Exception:
            pass

    return "unknown_target", None, np.nan


# =============================================================================
# Main
# =============================================================================

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Extract multi-target aperture light curves from TESS, Kepler, K2, and TESSCut TPFs (Voronoi-then-grow).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument(
        "--tpf-dir",
        type=str,
        default=".",
        help="Directory to search for TPF files. Default: current directory.",
    )
    p.add_argument(
        "--recursive",
        action="store_true",
        help="Search for TPF files recursively under --tpf-dir.",
    )
    p.add_argument(
        "--single",
        type=str,
        default="",
        help="If set, process only this one TPF path (overrides directory search).",
    )

    p.add_argument(
        "--output-root",
        type=str,
        default="LC_products_multi",
        help="Output root directory.",
    )
    p.add_argument(
        "--method",
        choices=["jump", "core", "both"],
        default="jump",
        help="Aperture-growth method to run.",
    )

    p.add_argument(
        "--n-targets",
        type=int,
        default=2,
        help=("Max number of Gaia targets to extract per stamp. TESSCut target 1 is "
              "nearest the requested cutout centre; otherwise brightness order is used."),
    )
    p.add_argument(
        "--gaia-radius-arcmin",
        type=float,
        default=6.0,
        help="Gaia cone-search radius around stamp center.",
    )


    p.add_argument(
        "--no-gaia",
        action="store_true",
        help="Do not query the Gaia Archive. Intended for single-target runs (e.g., saturated stars) or when Gaia is unavailable.",
    )
    p.add_argument(
        "--gaia-fallback",
        action="store_true",
        help="If Gaia query fails, fall back to no-Gaia single-target mode (only allowed when --n-targets=1).",
    )
    p.add_argument(
        "--no-quality0",
        action="store_true",
        help="Disable filtering to quality==0 cadences.",
    )

    # Growth knobs
    p.add_argument("--min-pixels", type=int, default=10)
    p.add_argument("--amp-q-lo", type=float, default=1.0)
    p.add_argument("--amp-q-hi", type=float, default=99.0)
    p.add_argument("--amp-min-frac", type=float, default=0.01)

    # Jump-growth extra knobs
    p.add_argument("--max-components", type=int, default=3)
    p.add_argument("--min-seed-frac-of-peak", type=float, default=0.15)
    p.add_argument("--min-new-pixels-per-component", type=int, default=1)

    # Core-preseed knobs
    p.add_argument("--core-npix", type=int, default=12)
    p.add_argument("--core-min-frac-of-peak", type=float, default=0.25)
    p.add_argument(
        "--max-radius-pix",
        type=float,
        default=np.inf,
        help="Maximum distance in pixels that an aperture may grow from its original seed.",
    )

    # Decorrelation knobs
    p.add_argument("--knot-spacing-days", type=float, default='inf') #turned off by default
    p.add_argument("--robust-iters", type=int, default=8)
    p.add_argument("--huber-k", type=float, default=1.5)
    p.add_argument(
        "--psf-proxy",
        action="store_true",
        help="Include PSF-width proxy terms in the decorrelation model.",
    )

    p.add_argument(
        "--no-aperture-plots",
        action="store_true",
        help="Do not save aperture overlay PNGs.",
    )
    p.add_argument(
        "--save-figure-pickles",
        action="store_true",
        help=(
            "Save each Matplotlib figure as a .png.pickle companion to its PNG. "
            "Only open figure pickles produced by trusted runs."
        ),
    )

    # Optional jitter-aware PRF photometry. This is an additional extraction
    # product; it never replaces the selected aperture product.
    p.add_argument(
        "--prf-photometry", action="store_true",
        help="Also extract cadence-dependent, jitter-aware TESS PRF photometry. Kepler/K2 inputs are skipped with a warning.",
    )
    p.add_argument(
        "--prf-backend", choices=["auto", "lkprf", "tess_prf", "gaussian"], default="auto",
        help="PRF backend. Auto prefers the official lkprf/TESS_PRF engineering models.",
    )
    p.add_argument(
        "--prf-motion-source", choices=["auto", "poscorr", "ensemble", "target", "fixed"], default="auto",
        help="Motion source. Auto uses varying POS_CORR, then ensemble centroid, target centroid, and fixed position; constant placeholders are rejected.",
    )
    p.add_argument(
        "--prf-scene-mode", choices=["single", "gaia"], default="single",
        help="Single-source PRF extraction or a target-first Gaia scene with fixed fitted neighbors.",
    )
    p.add_argument(
        "--prf-neighbor-treatment", choices=["fixed"], default="fixed",
        help="Cadence treatment for Gaia neighbors. Fixed holds non-active sources at their reference-scene fluxes.",
    )
    p.add_argument(
        "--prf-source-output", choices=["primary", "all"], default="primary",
        help=("Which Gaia-scene light curves to write. Primary (default) writes only the selected target. "
              "All extracts each usable retained scene source separately while holding all other sources fixed."),
    )
    p.add_argument("--prf-neighbor-dmag", type=float, default=8.0,
                   help="Maximum Gaia G magnitude difference for initial scene neighbors.")
    p.add_argument("--prf-neighbor-margin", type=float, default=6.0,
                   help="Include Gaia sources this many pixels outside the stamp edge.")
    p.add_argument("--prf-max-scene-sources", type=int, default=20,
                   help="Maximum target-plus-neighbor sources in the initial Gaia PRF scene.")
    p.add_argument("--prf-min-neighbor-fraction", type=float, default=1e-4,
                   help="Drop fitted neighbors contributing less than this fraction of target flux inside the stamp.")
    p.add_argument(
        "--prf-background", choices=["none", "constant", "plane"], default="plane",
        help="Per-cadence background terms fitted simultaneously with the PRF flux; plane is the recommended default.",
    )
    p.add_argument("--prf-min-weight", type=float, default=1e-5,
                   help="Minimum relative PRF weight used to define the fitting region.")
    p.add_argument("--prf-fit-radius", type=float, default=6.0,
                   help="Radius in pixels included around each modeled PRF source.")
    p.add_argument("--prf-shift-quantization", type=float, default=0.01,
                   help="Requested regular-grid spacing for precomputed shifted PRFs; cadence PRFs are bilinearly interpolated and each axis is capped at 21 nodes.")
    p.add_argument("--prf-max-shift", type=float, default=2.0,
                   help="Maximum absolute cadence motion retained after robust cleaning [pixel].")
    p.add_argument("--prf-no-diagnostics", action="store_true",
                   help="Do not write the PRF diagnostic PNG (motion NPZ and metadata are still written).")
    p.add_argument("--prf-no-gaussian-fallback", action="store_true",
                   help="Fail PRF extraction instead of using a Gaussian if official PRF packages are unavailable.")
    
    p.add_argument(
        "--full-region-sum",
        dest="full_region_sum",
        action="store_true",
        help="Sum every pixel in the allowed target region without aperture optimization.",
    )
    # One-release command-line compatibility for saved scripts.  The alias is
    # deliberately hidden so new users see only descriptive terminology.
    p.add_argument(
        "--pure-sum",
        dest="full_region_sum",
        action="store_true",
        default=argparse.SUPPRESS,
        help=argparse.SUPPRESS,
    )

    p.add_argument(
        "--external-mask-file",
        type=str,
        default="",
        help=(
            "Fixed single-target aperture mask in a text file. Use one image row per line "
            "with comma- or whitespace-separated 1/0 or Y/N values. The mask must exactly "
            "match the TPF row/column shape and overrides --method."
        ),
    )

    p.add_argument(
        "--aperture-fom",
        choices=["stddiff", "std", "mad"],
        default="stddiff",
        help="Figure of merit used during jump/core aperture growth.",
    )

    # Saturated-target workflows.  The first augments standard aperture
    # extraction; the second is a separate, single-target extraction approach.
    p.add_argument(
        "--saturated-systematics-correction",
        dest="saturated_systematics_correction",
        action="store_true",
        help="Apply the saturated-target background, orbital-phase, and split-sector systematics correction.",
    )
    p.add_argument(
        "--matlab-sat-mode",
        dest="saturated_systematics_correction",
        action="store_true",
        default=argparse.SUPPRESS,
        help=argparse.SUPPRESS,
    )
    p.add_argument(
        "--orbtable",
        type=str,
        default=str(DEFAULT_ORBITAL_TABLE),
        help=(
            "Sector orbital-frequency/midpoint CSV used by saturated-target systematics correction. "
            "The bundled tess_sector_orbfreq_midpoints.csv is used by default."
        ),
    )
    p.add_argument("--sat-thresh", type=float, default=1e5, help="Mean-image threshold to consider a pixel saturated (counts).")
    p.add_argument("--sat-min-npix", type=int, default=20, help="Minimum number of pixels above --sat-thresh to treat target as heavily saturated.")
    p.add_argument("--back-nfaint", type=int, default=20, help="Number of faintest pixels to use for per-cadence background estimate.")
    p.add_argument("--phase-bin", type=float, default=0.01, help="Phase bin width for orbital-phase template subtraction.")
    p.add_argument(
        "--saturation-optimized-aperture",
        dest="saturation_optimized_aperture",
        action="store_true",
        help="For one heavily saturated source, bypass Gaia and jump/core growth and use scatter-optimized aperture growth.",
    )
    p.add_argument(
        "--matlab-pure-single-sat",
        dest="saturation_optimized_aperture",
        action="store_true",
        default=argparse.SUPPRESS,
        help=argparse.SUPPRESS,
    )
    p.add_argument(
        "--saturated-aperture-threshold",
        dest="saturated_aperture_threshold",
        type=float,
        default=3000.0,
        help="Initial mean-image threshold for the saturation-optimized aperture seed.",
    )
    p.add_argument(
        "--matlab-ap-thresh",
        dest="saturated_aperture_threshold",
        type=float,
        default=argparse.SUPPRESS,
        help=argparse.SUPPRESS,
    )

    return p.parse_args(argv)


def validate_args(args) -> None:
    """Fail early for option combinations that would otherwise fail mid-run."""
    if int(args.n_targets) < 1:
        raise ValueError("--n-targets must be at least 1.")
    if args.saturation_optimized_aperture and int(args.n_targets) != 1:
        raise ValueError("--saturation-optimized-aperture requires --n-targets=1.")
    if args.saturation_optimized_aperture:
        incompatible = []
        if args.full_region_sum:
            incompatible.append("--full-region-sum")
        if getattr(args, "gaia_region_sum", False):
            incompatible.append("--gaia-region-sum")
        if getattr(args, "external_mask_file", ""):
            incompatible.append("--external-mask-file")
        if args.saturated_systematics_correction:
            incompatible.append("--saturated-systematics-correction")
        if args.prf_photometry:
            incompatible.append("--prf-photometry")
        if incompatible:
            raise ValueError(
                "--saturation-optimized-aperture is a separate extraction path "
                "and cannot be combined with " + ", ".join(incompatible) + "."
            )
    if args.saturated_systematics_correction and int(args.n_targets) != 1:
        raise ValueError("--saturated-systematics-correction requires --n-targets=1.")
    if getattr(args, "external_mask_file", "") and int(args.n_targets) != 1:
        raise ValueError("--external-mask-file requires --n-targets=1.")
    if getattr(args, "external_mask_file", "") and args.full_region_sum:
        raise ValueError("--external-mask-file cannot be combined with --full-region-sum.")
    if args.no_gaia and int(args.n_targets) != 1:
        raise ValueError("--no-gaia requires --n-targets=1.")
    if not (0.0 <= float(args.amp_q_lo) < float(args.amp_q_hi) <= 100.0):
        raise ValueError("Aperture amplitude percentiles must satisfy 0 <= low < high <= 100.")
    if int(args.min_pixels) < 1 or int(args.max_components) < 1:
        raise ValueError("Aperture sizes and component counts must be positive.")
    if int(args.sat_min_npix) < 1 or int(args.back_nfaint) < 1:
        raise ValueError("Saturation/background pixel counts must be positive.")
    if not (0.0 < float(args.phase_bin) <= 1.0):
        raise ValueError("--phase-bin must be in the interval (0, 1].")
    if float(args.prf_fit_radius) <= 0 or float(args.prf_max_shift) <= 0:
        raise ValueError("PRF fit radius and maximum shift must be positive.")


def write_run_configuration(
    output_root: Path,
    args,
    tpf_paths: list[Path],
    external_mask_metadata: dict | None = None,
) -> Path:
    """Write the fully resolved extraction options beside the data products."""
    created = datetime.now(timezone.utc)
    settings = {}
    for key, value in vars(args).items():
        if isinstance(value, float) and not np.isfinite(value):
            value = str(value)
        settings[key] = value
    payload = {
        "schema_version": 1,
        "created_utc": created.isoformat(),
        "extractor": Path(__file__).name,
        "settings": settings,
        "input_files": [str(path) for path in tpf_paths],
    }
    if external_mask_metadata is not None:
        # The checksum and explicit orientation make a hand-edited aperture
        # reproducible even if its source file is later renamed or changed.
        payload["external_aperture_mask"] = external_mask_metadata
    stamp = created.strftime("%Y%m%dT%H%M%S_%fZ")
    path = output_root / f"extraction_run_config_{stamp}.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def main(argv=None):
    global APERTURE_FOM_MODE, SAVE_FIGURE_PICKLES
    args = parse_args(argv)
    validate_args(args)
    APERTURE_FOM_MODE = str(getattr(args, "aperture_fom", "stddiff")).strip().lower()
    SAVE_FIGURE_PICKLES = bool(getattr(args, "save_figure_pickles", False))

    # Parse a fixed external aperture once. Shape validation is repeated for
    # each input TPF so directory runs fail clearly if stamp dimensions differ.
    external_aperture_mask = None
    external_mask_metadata = None
    if getattr(args, "external_mask_file", ""):
        external_aperture_mask, external_mask_metadata = load_external_aperture_mask(
            args.external_mask_file
        )
        args.external_mask_file = external_mask_metadata["source_path"]
        print(
            "External aperture mask loaded: "
            f"{external_mask_metadata['source_path']} "
            f"shape={tuple(external_aperture_mask.shape)} "
            f"Npix={external_mask_metadata['selected_pixels']}"
        )

    # Optional: load sector orbital-frequency table once
    sector_orb = None
    if args.saturated_systematics_correction and not args.saturation_optimized_aperture:
        if not args.orbtable:
            raise ValueError(
                "--saturated-systematics-correction requires --orbtable "
                "unless --saturation-optimized-aperture is used"
            )
        args.orbtable = str(Path(args.orbtable).expanduser().resolve())
        sector_orb = load_sector_orbtable(args.orbtable)

    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    # Flat output structure: all products go directly into --output-root.
    outdir = output_root

    # TPF list
    if args.single:
        tpf_paths = [Path(args.single).expanduser().resolve()]
        if not tpf_paths[0].exists():
            raise FileNotFoundError(f"--single path not found: {tpf_paths[0]}")
    else:
        search_dir = Path(args.tpf_dir).expanduser().resolve()
        if not search_dir.exists():
            raise FileNotFoundError(f"--tpf-dir not found: {search_dir}")
        tpf_paths = find_tpfs(search_dir, recursive=args.recursive)

    if not tpf_paths:
        raise FileNotFoundError("No TPF files found.")

    args.output_root = str(output_root)
    config_path = write_run_configuration(
        output_root,
        args,
        tpf_paths,
        external_mask_metadata=external_mask_metadata,
    )
    print(f"Resolved extraction settings: {config_path}")

    print(f"TPFs to process: {len(tpf_paths)}")
    for i, p in enumerate(tpf_paths, 1):
        print(f"  [{i}] {p}")

    if external_aperture_mask is not None:
        # A fixed mask is itself the aperture-selection method. Running both
        # jump and core would only duplicate the same summed light curve.
        methods_to_run = ["external_aperture"]
    elif args.full_region_sum:
        methods_to_run = ["full_region_sum"]
    else:
        methods_to_run = [args.method] if args.method in ("jump", "core") else ["jump", "core"]

    for tpf_path in tpf_paths:
        print("\n" + "#" * 80)
        print("TPF:", tpf_path)

        outdir = output_root

        # Fast full-region path: avoid loading the full flux cube through Lightkurve.
        # When PRF photometry is requested we use the normal path because the PRF
        # branch needs WCS/metadata and cadence-level motion information.
        if (
            args.full_region_sum
            and not args.prf_photometry
            and not args.saturated_systematics_correction
            and not args.saturation_optimized_aperture
        ):
            mission = infer_mission(tpf_path, None)
            time_system = infer_native_time_system(tpf_path, None, mission)
            sector_tag = infer_sector_tag(tpf_path, None)
            source_label = fast_object_label_from_fits(tpf_path)
            t_g, lc_raw, mean_img, ap_mask, crow, ccol, seed = fast_full_region_sum_from_fits(
                tpf_path,
                no_quality0=bool(args.no_quality0),
            )
            print("  Gaia skipped/unavailable: using pixel-defined single target")
            print(f"    [1] seed={seed}  (brightest pixel in mean image)")

            med = np.nanmedian(lc_raw)
            lc_xy = lc_raw / med if np.isfinite(med) and med != 0 else lc_raw
            output_stem = build_output_stem(sector_tag, source_label, 1, "full_region_sum")
            make_lightcurve_dataframe(
                t_g, lc_xy, mission, time_system, "flux_detrended_rel"
            ).to_csv(outdir / f"preferred_lc_{output_stem}.csv", index=False)
            save_lightcurve_plot(
                t_g, lc_xy,
                outdir / f"preferred_lc_{output_stem}.png",
                f"{tpf_path.stem} — {source_label} / target1 (full_region_sum)",
                time_label=mission_time_axis_label(time_system),
            )
            np.save(outdir / f"aperture_mask_{output_stem}.npy", ap_mask.astype(bool))
            np.save(outdir / f"centroid_row_{output_stem}.npy", np.asarray(crow, float))
            np.save(outdir / f"centroid_col_{output_stem}.npy", np.asarray(ccol, float))
            if not args.no_aperture_plots:
                save_aperture_plot(
                    mean_img,
                    ap_mask.astype(bool),
                    outdir / f"aperture_{output_stem}.png",
                    f"{tpf_path.stem} — {source_label} / target1 (full_region_sum)\nGaia G=NA",
                )
            print(f"  Wrote target1 (full_region_sum): preferred_lc_{output_stem}.csv  Npix={int(np.count_nonzero(ap_mask))}")
            continue

        tpf = lk.read(str(tpf_path))
        mission = infer_mission(tpf_path, tpf)
        tesscut_product = is_tesscut_product(tpf_path, tpf)
        time_system = infer_native_time_system(tpf_path, tpf, mission)
        print(f"  Mission: {mission}; native time: {time_system}; observation: {infer_observation_tag(tpf_path, tpf)}")
        if getattr(args, "prf_photometry", False) and mission in {"KEPLER", "K2"}:
            print(
                f"  [WARN] {mission} target-pixel data detected: the PRF module is TESS-only "
                "and will be skipped. Aperture extraction will continue normally."
            )
        # Sector inference is needed by the optional orbital-phase correction.
        sector_num = infer_sector_from_tpf(tpf_path, tpf)
        if args.saturated_systematics_correction and sector_num is None:
            print(
                "  [WARN] Saturated-target systematics correction requested, but "
                "SECTOR could not be inferred; using the standard correction."
            )


        # Cadence selection
        native_time = np.asarray(tpf.time.value, float)
        finite_time = np.isfinite(native_time)
        if args.no_quality0:
            keep_idx = np.flatnonzero(finite_time)
        else:
            try:
                qual = np.asarray(tpf.quality, int)
                keep_idx = np.flatnonzero((qual == 0) & finite_time)
                if keep_idx.size == 0:
                    print("  [WARN] No finite QUALITY==0 cadences; retaining finite-time cadences instead.")
                    keep_idx = np.flatnonzero(finite_time)
            except Exception:
                keep_idx = np.flatnonzero(finite_time)

        if keep_idx.size == 0:
            print("  [WARN] No finite cadences; skipping this file.")
            continue

        mid = int(keep_idx[len(keep_idx) // 2]) if len(keep_idx) else 0

        flux = np.asarray(tpf.flux, float)[keep_idx, :, :]
        time = native_time[keep_idx]
        mean_img = np.nanmean(flux, axis=0)

        # This approach is intentionally separate from Gaia and jump/core
        # extraction because it grows a single aperture over the whole stamp.
        if args.saturation_optimized_aperture:
            if int(args.n_targets) != 1:
                raise ValueError("--saturation-optimized-aperture requires --n-targets=1.")
            sat_ok, sat_npix = is_heavily_saturated(mean_img, thresh=args.sat_thresh, min_npix=args.sat_min_npix)
            if sat_ok:
                print(f"  Using saturation-optimized aperture extraction (npix_above_thresh={sat_npix})")
                lc_df, meta, mean_image_2d, ap_mask_2d, back_series, keep_mask, crow, ccol = extract_saturation_optimized_aperture(
                    tpf_path,
                    threshold=args.saturated_aperture_threshold,
                    nback=args.back_nfaint,
                    filter_quality=(not args.no_quality0),
                    verbose=True,
                )
                sector_tag = infer_sector_tag(tpf_path, tpf)
                target_label, target_gaia_id, target_gaia_g = infer_single_target_label(
                    tpf,
                    mean_image_2d=mean_image_2d,
                    gaia_radius_arcmin=max(float(args.gaia_radius_arcmin), 12.0),
                    allow_catalog=(not args.no_gaia),
                )
                output_stem = build_output_stem(sector_tag, target_label, 1, "saturated_aperture")
                raw_csv_path = outdir / f"preferred_lc_{output_stem}.csv"
                # Every product is labeled; never remove unrelated products
                # that may already exist in a user-selected output directory.
                np.save(outdir / f"keep_idx_{output_stem}.npy", np.flatnonzero(keep_mask))
                np.save(outdir / f"mean_image_{output_stem}.npy", np.asarray(mean_image_2d, float))

                # Apply the common mission-aware time schema. Kepler/K2 retain
                # native BKJD and gain the exact compatible BTJD conversion;
                # TESS retains its native BTJD values directly.
                native_sat_time = lc_df["time_native"].to_numpy(float)
                lc_df = make_lightcurve_dataframe(
                    native_sat_time,
                    lc_df["flux_detrended_rel"].to_numpy(float),
                    mission,
                    time_system,
                    "flux_rel",
                )
                lc_df["target_label"] = target_label
                lc_df["target_index"] = 1
                lc_df["method"] = "saturated_aperture"
                lc_df["sector"] = sector_tag
                lc_df["tpf_name"] = tpf_path.name
                lc_df["gaia_source_id"] = target_gaia_id if target_gaia_id is not None else ""
                lc_df["gaia_g_mag"] = target_gaia_g
                lc_df.to_csv(raw_csv_path, index=False)
                save_lightcurve_plot(
                    native_sat_time,
                    lc_df["flux_rel"].to_numpy(float),
                    outdir / f"preferred_lc_{output_stem}.png",
                    f"{tpf_path.stem} — {target_label} / target1 (saturated_aperture)",
                    time_label=mission_time_axis_label(time_system),
                )
                pd.DataFrame([meta]).to_csv(outdir / f"saturated_aperture_meta_{output_stem}.csv", index=False)
                np.save(outdir / f"aperture_mask_{output_stem}.npy", np.asarray(ap_mask_2d, bool))
                np.save(outdir / f"background_{output_stem}.npy", np.asarray(back_series, float))
                np.save(outdir / f"keep_mask_{output_stem}.npy", np.asarray(keep_mask, bool))
                np.save(outdir / f"centroid_row_{output_stem}.npy", np.asarray(crow, float))
                np.save(outdir / f"centroid_col_{output_stem}.npy", np.asarray(ccol, float))
                if not args.no_aperture_plots:
                    save_aperture_plot(
                        mean_image_2d,
                        ap_mask_2d,
                        outdir / f"aperture_{output_stem}.png",
                        f"{tpf_path.stem} — {target_label} / target1 (saturated_aperture)",
                    )
                print(
                    f"  Wrote {target_label} / target1 (saturated_aperture): {raw_csv_path.name}"
                    f"  Npix={int(np.count_nonzero(ap_mask_2d))}"
                )
                if getattr(args, "prf_photometry", False):
                    print("  [WARN] PRF photometry skipped for saturation-optimized aperture extraction.")
                continue
            else:
                print(
                    "  [INFO] --saturation-optimized-aperture requested but the "
                    f"saturation gate failed (npix_above_thresh={sat_npix}); using the standard pipeline."
                )

        # Target definition / Gaia usage
        use_gaia = not (
            args.no_gaia
            or args.full_region_sum
            or args.saturated_systematics_correction
            or external_aperture_mask is not None
        )

        coords_grid = None
        if use_gaia:
            # Gaia seed placement needs a sky coordinate at every pixel.  An
            # explicit no-Gaia run does not, so it remains usable for files
            # with incomplete or invalid WCS metadata.
            try:
                coords_grid, flux, mean_img = coordinate_grid_for_flux(
                    tpf, mid, flux, mean_img
                )
            except Exception as exc:
                if args.gaia_fallback and int(args.n_targets) == 1:
                    print(
                        f"  [WARN] WCS coordinate grid failed ({type(exc).__name__}: {exc}). "
                        "Falling back to no-Gaia single-target mode."
                    )
                    use_gaia = False
                    coords_grid = None
                else:
                    raise

        gaia_tab = None
        gaia_use = None
        sid = None
        gmag = None
        source_labels = None
        target_coords = None
        seeds = None
        owner = None
        n_use = 1
        sector_tag = infer_sector_tag(tpf_path, tpf)

        if use_gaia:
            try:
                gaia_tab = gaia_brightest_sources_near(tpf, radius_arcmin=args.gaia_radius_arcmin)
            except Exception as e:
                if args.gaia_fallback and int(args.n_targets) == 1:
                    print(f"  [WARN] Gaia query failed ({type(e).__name__}: {e}). Falling back to no-Gaia single-target mode.")
                    use_gaia = False
                else:
                    raise

        if use_gaia:
            # Gaia-based multi-/single-target identification (standard behavior)
            gaia_in, _ = sources_inside_stamp(tpf, gaia_tab, mid)
            if len(gaia_in) < 1:
                print("  No Gaia sources inside stamp; skipping.")
                continue

            center_separation_arcsec = None
            if tesscut_product:
                gaia_in, center_separation_arcsec = prioritize_tesscut_center_source(
                    gaia_in, tpf, mean_img.shape
                )
                if center_separation_arcsec is not None:
                    print(
                        "  TESSCut target policy: target 1 is the Gaia source nearest "
                        f"the requested cutout centre ({center_separation_arcsec:.1f} arcsec); "
                        "remaining targets retain brightness order."
                    )

            print("  Gaia sources inside stamp after pixel filter:")
            for i, row in enumerate(gaia_in[:10], start=1):
                tc = SkyCoord(float(row["ra"]) * u.deg, float(row["dec"]) * u.deg)
                try:
                    x, y = tpf.wcs.world_to_pixel(tc)
                    print(f"    [{i}] source_id={row['source_id']}  G={row['phot_g_mean_mag']:.3f}  x={x:.2f}  y={y:.2f}")
                except Exception as e:
                    print(f"    [{i}] source_id={row['source_id']}  G={row['phot_g_mean_mag']:.3f}  pixel_error={type(e).__name__}: {e}")

            n_use = min(int(args.n_targets), len(gaia_in))
            gaia_use = gaia_in[:n_use]

            target_coords = [
                SkyCoord(ra=float(r) * u.deg, dec=float(d) * u.deg)
                for r, d in zip(gaia_use["ra"], gaia_use["dec"])
            ]
            gmag = [float(m) for m in gaia_use["phot_g_mean_mag"]]
            sid = [str(s) for s in gaia_use["source_id"]]
            source_labels = [resolve_target_label(tc, gs) for tc, gs in zip(target_coords, sid)]

            # For a genuine single-target TPF, prefer the stable human-readable
            # OBJECT/TARGNAME label recorded in the file when the selected Gaia
            # source matches the intended target coordinate.  This avoids
            # network-dependent filename changes: a transient SIMBAD failure
            # should not turn, for example, L_98-59_b into GaiaDR3_<source_id>.
            # Do not apply this blindly to multi-target stamps, where the TPF
            # header name may refer to a different source than target k.
            if n_use == 1:
                try:
                    single_label, _, _ = infer_single_target_label(
                        tpf,
                        mean_img,
                        gaia_radius_arcmin=float(args.gaia_radius_arcmin),
                    )
                    header_coord = infer_header_target_coord(tpf)
                    selected_matches_header = False
                    if header_coord is not None and target_coords and target_coords[0] is not None:
                        sep_arcsec = float(header_coord.separation(target_coords[0]).arcsec)
                        selected_matches_header = np.isfinite(sep_arcsec) and sep_arcsec <= 45.0

                    force_single_label = bool(
                        args.saturated_systematics_correction
                        or getattr(args, "no_gaia", False)
                        or args.saturation_optimized_aperture
                    )
                    if single_label and (selected_matches_header or force_single_label):
                        source_labels[0] = single_label
                except Exception:
                    pass

            seeds = [nearest_pixel_to_coord(coords_grid, tc) for tc in target_coords]
            owner = watershed_owner_map(mean_img, seeds)

            if tesscut_product:
                print("  Targets (TESSCut centre target first; remaining Gaia sources by brightness):")
            else:
                print("  Targets (brightest Gaia sources in stamp):")
            for k in range(n_use):
                label_txt = source_labels[k] if source_labels is not None else f"target{k+1}"
                print(f"    [{k+1}] label={label_txt}  source_id={sid[k]}  G={gmag[k]:.3f}  seed={seeds[k]}")

        else:
            # No-Gaia single-target mode: define target by pixels (useful for saturated stars / Gaia outages)
            if int(args.n_targets) != 1:
                raise ValueError("No-Gaia mode requires --n-targets=1.")
            # Seed at the brightest pixel in the mean image
            flat_idx = int(np.nanargmax(mean_img))
            ny, nx = mean_img.shape
            seed = (flat_idx // nx, flat_idx % nx)

            target_coords = [None]
            seeds = [seed]
            owner = np.zeros(mean_img.shape, dtype=int)
            sid = ["pixel_seed"]
            gmag = [np.nan]
            try:
                single_label, _, _ = infer_single_target_label(
                    tpf,
                    mean_img,
                    gaia_radius_arcmin=float(args.gaia_radius_arcmin),
                    allow_catalog=(not args.no_gaia),
                )
                source_labels = [single_label if single_label else sanitize_token("unknown_target")]
            except Exception:
                source_labels = [sanitize_token("unknown_target")]

            if external_aperture_mask is not None:
                print("  External aperture defines a fixed pixel-selected single target")
            else:
                print("  Gaia skipped/unavailable: using pixel-defined single target")
            print(f"    [1] seed={seed}  (brightest pixel in mean image)")

        yy, xx = np.indices(mean_img.shape)

        # -----------------------------------------------------------------
        # Optional additional jitter-aware PRF extraction.
        # -----------------------------------------------------------------
        if getattr(args, "prf_photometry", False) and mission == "TESS":
            if not _HAVE_PRF_MODULE:
                print(f"  [WARN] PRF photometry unavailable: {_PRF_IMPORT_ERROR}")
            else:
                tessmag = prf_saturation_tessmag(tpf_path, tpf)
                flux_unit = tpf_flux_unit(tpf)
                try:
                    prf_flux_err = np.asarray(tpf.flux_err, float)[keep_idx, :, :]
                    if prf_flux_err.shape != flux.shape:
                        prf_flux_err = prf_flux_err[:, : flux.shape[1], : flux.shape[2]]
                    if prf_flux_err.shape != flux.shape:
                        prf_flux_err = None
                except Exception:
                    prf_flux_err = None

                for prf_k in range(n_use):
                    source_label = source_labels[prf_k] if source_labels is not None else f"target{prf_k+1}"
                    # Prefer the subpixel WCS position for Gaia-defined targets;
                    # fall back to the image seed if WCS conversion is unavailable.
                    prf_row, prf_col = map(float, seeds[prf_k])
                    try:
                        if target_coords is not None and target_coords[prf_k] is not None:
                            xpix, ypix = tpf.wcs.world_to_pixel(target_coords[prf_k])
                            if (np.isfinite(xpix) and np.isfinite(ypix) and
                                    -float(args.prf_neighbor_margin) <= xpix <= mean_img.shape[1] - 1 + float(args.prf_neighbor_margin) and
                                    -float(args.prf_neighbor_margin) <= ypix <= mean_img.shape[0] - 1 + float(args.prf_neighbor_margin)):
                                prf_row, prf_col = float(ypix), float(xpix)
                    except Exception:
                        pass

                    sat_info = assess_saturation(
                        mean_img,
                        tessmag=tessmag,
                        absolute_threshold=float(args.sat_thresh),
                        source_row=prf_row,
                        source_column=prf_col,
                        local_radius=max(8.0, float(args.prf_fit_radius)),
                        flux_unit=flux_unit,
                        product_type="TESSCUT" if tesscut_product else "TPF",
                    )
                    if sat_info.get("saturated", False):
                        print(
                            f"  [WARN] PRF photometry skipped for target {prf_k+1}: "
                            "saturation/bleeding is not represented by the engineering PRF. "
                            + "; ".join(sat_info.get("reasons", []))
                        )
                        continue

                    requested_scene = str(args.prf_scene_mode).strip().lower()
                    scene_sources = [SceneSource(
                        prf_row, prf_col, source_label,
                        sid[prf_k] if sid is not None else None,
                        gmag[prf_k] if gmag is not None else None,
                        "target",
                    )]
                    actual_scene = "single"
                    if requested_scene == "gaia":
                        if use_gaia and gaia_tab is not None and sid is not None:
                            scene_sources = build_gaia_prf_scene(
                                tpf, gaia_tab,
                                target_source_id=sid[prf_k], target_label=source_label,
                                target_gmag=gmag[prf_k], target_row=prf_row,
                                target_column=prf_col, shape=mean_img.shape,
                                margin_pixels=float(args.prf_neighbor_margin),
                                max_delta_mag=float(args.prf_neighbor_dmag),
                                max_sources=int(args.prf_max_scene_sources),
                            )
                            actual_scene = "gaia" if len(scene_sources) > 1 else "single"
                            print(
                                f"  [PRF] Target {prf_k+1} Gaia scene: "
                                f"{len(scene_sources)} candidate source(s), including target."
                            )
                            for scene_j, src in enumerate(scene_sources[:10]):
                                role = "target" if scene_j == 0 else "neighbor"
                                magtxt = f"{src.magnitude:.3f}" if src.magnitude is not None and np.isfinite(src.magnitude) else "nan"
                                print(
                                    f"  [PRF]   [{scene_j+1}] {role} {src.label} "
                                    f"G={magtxt} row={src.row:.2f} col={src.column:.2f}"
                                )
                        else:
                            print("  [WARN] Gaia PRF scene requested but Gaia data are unavailable; using single-source mode.")

                    try:
                        prf_cfg = PRFPhotometryConfig(
                            backend=args.prf_backend,
                            motion_source=args.prf_motion_source,
                            background=args.prf_background,
                            saturation_absolute_threshold=float(args.sat_thresh),
                            min_prf_weight=float(args.prf_min_weight),
                            fit_radius=float(args.prf_fit_radius),
                            shift_quantization=float(args.prf_shift_quantization),
                            max_abs_shift=float(args.prf_max_shift),
                            allow_gaussian_fallback=(not args.prf_no_gaussian_fallback),
                            scene_mode=actual_scene,
                            neighbor_treatment=str(args.prf_neighbor_treatment),
                            source_output_mode=str(args.prf_source_output),
                            neighbor_min_contribution_fraction=float(args.prf_min_neighbor_fraction),
                            save_diagnostics=(not args.prf_no_diagnostics),
                        )
                        prf_result = extract_jitter_aware_prf(
                            time, flux, scene_sources,
                            config=prf_cfg,
                            tpf_obj=tpf,
                            tpf_path=tpf_path,
                            cadence_indices=keep_idx,
                            flux_err_cube=prf_flux_err,
                        )
                        output_indices = list(map(
                            int, prf_result.metadata.get("extracted_source_indices", [0])
                        ))
                        if not output_indices:
                            output_indices = [0]
                        resolve_prf_scene_output_names(tpf, prf_result, output_indices)

                        for msg in prf_result.metadata.get("backend_messages", []):
                            print(f"  [PRF] {msg}")
                        scene_info = prf_result.metadata.get("reference_scene", {})
                        written_csvs = []
                        for scene_index in output_indices:
                            source = prf_result.sources[scene_index]
                            is_primary_output = scene_index == 0
                            if is_primary_output:
                                prf_output_stem = "preferred_lc_" + build_output_stem(
                                    sector_tag, source_label, prf_k + 1, "prf"
                                )
                            else:
                                file_label = build_prf_scene_output_label(source, is_primary=False)
                                prf_output_stem = (
                                    f"preferred_lc_{sanitize_token(sector_tag)}_"
                                    f"{file_label}_parenttarget{prf_k + 1}_"
                                    f"scene{scene_index + 1}_prf"
                                )

                            paths_written = save_prf_products(
                                prf_result, mean_img, outdir, prf_output_stem,
                                source_index=scene_index,
                                extra_columns={
                                    "target_label": source.label,
                                    "target_index": prf_k + 1,
                                    "scene_source_index": scene_index,
                                    "scene_source_role": "primary" if is_primary_output else "neighbor",
                                    "method": "prf",
                                    "sector": sector_tag,
                                    "tpf_name": tpf_path.name,
                                    "gaia_source_id": source.source_id or "",
                                    "gaia_g_mag": source.magnitude if source.magnitude is not None else np.nan,
                                },
                                extra_metadata={
                                    "tpf_path": str(tpf_path),
                                    "watershed_extractor": True,
                                    "single_source_fit": len(prf_result.sources) == 1,
                                    "requested_prf_scene_mode": requested_scene,
                                    "actual_prf_scene_mode": prf_result.metadata.get("scene_mode", actual_scene),
                                    "requested_prf_source_output": str(args.prf_source_output),
                                    "scene_source_index": int(scene_index),
                                    "scene_source_role": "primary" if is_primary_output else "neighbor",
                                    "parent_primary_label": source_label,
                                    "parent_primary_index": int(prf_k + 1),
                                },
                                save_diagnostics=(not args.prf_no_diagnostics),
                                save_figure_pickle=SAVE_FIGURE_PICKLES,
                            )
                            written_csvs.append(paths_written["csv"].name)
                            coupling_all = prf_result.metadata.get("motion_coupling", [])
                            coupling = coupling_all[scene_index] if scene_index < len(coupling_all) else {}
                            quality_ok = coupling.get("quality_pass", True) and scene_info.get("scene_quality_pass", True)
                            quality = "PASS" if quality_ok else "FLAGGED"
                            print(
                                f"  PRF scene source {scene_index + 1}: label={source.label}, "
                                f"backend={prf_result.backend}, motion={prf_result.motion_source}, "
                                f"scene={prf_result.metadata.get('scene_mode','single')}, "
                                f"sources={len(prf_result.sources)}, quality={quality}, "
                                f"wrote={paths_written['csv'].name}"
                            )
                            if not quality_ok:
                                reasons = (
                                    coupling.get("quality_reasons", [])
                                    + scene_info.get("scene_quality_reasons", [])
                                )
                                print(
                                    "  [WARN] PRF light curve retained for diagnostics but should be "
                                    "treated cautiously: " + "; ".join(reasons)
                                )

                        skipped = prf_result.metadata.get("skipped_output_sources", [])
                        for item in skipped:
                            print(
                                f"  [PRF] Skipped scene source {int(item.get('scene_index', -1)) + 1} "
                                f"({item.get('label', 'unknown')}): {item.get('reason', 'not usable')}"
                            )
                        if len(written_csvs) > 1:
                            print(f"  [PRF] Wrote {len(written_csvs)} constrained scene-source light curves.")
                    except Exception as exc:
                        print(
                            f"  [WARN] PRF target{prf_k+1} failed "
                            f"({type(exc).__name__}: {exc}); continuing with aperture extraction."
                        )

        # Per-target processing
        for k in range(n_use):
            tag = f"target{k+1}"
            source_label = source_labels[k] if source_labels is not None else sanitize_token(tag)
            allowed = (owner == k)
            if external_aperture_mask is None and not np.any(allowed):
                print(f"  [WARN] {tag}: empty watershed region; skipping.")
                continue

            for meth in methods_to_run:
                crow = ccol = None
                active_external_metadata = None
                if meth == "external_aperture":
                    # Validate against the actual post-read image shape. The
                    # selected pixels deliberately ignore the watershed owner
                    # map: the user's text matrix is the complete aperture.
                    if tuple(external_aperture_mask.shape) != tuple(mean_img.shape):
                        raise ValueError(
                            f"External aperture mask shape {tuple(external_aperture_mask.shape)} "
                            f"does not match the TPF image shape {tuple(mean_img.shape)} for "
                            f"{tpf_path.name}. Rows and columns are not transposed automatically."
                        )
                    ap_mask = external_aperture_mask.copy()
                    active_external_metadata = dict(external_mask_metadata)
                    t_g = time
                    lc_raw, crow, ccol = masked_sum_and_centroid_blockwise(
                        flux, ap_mask, yy, xx
                    )
                    meth_tag = "external_aperture"
                elif meth == "full_region_sum":
                    ap_mask = allowed.copy()
                    t_g = time
                    lc_raw, crow, ccol = masked_sum_and_centroid_blockwise(flux, ap_mask, yy, xx)
                    meth_tag = "full_region_sum"
                elif meth == "jump":
                    t_g, lc_raw, ap_mask, *_ = grow_aperture_multi_component_in_region(
                        flux,
                        time,
                        seed_pix=seeds[k],
                        allowed_mask=allowed,
                        min_pixels=args.min_pixels,
                        amp_q_lo=args.amp_q_lo,
                        amp_q_hi=args.amp_q_hi,
                        amp_min_frac=args.amp_min_frac,
                        max_components=args.max_components,
                        min_seed_frac_of_peak=args.min_seed_frac_of_peak,
                        min_new_pixels_per_component=args.min_new_pixels_per_component,
                        max_radius_pix=args.max_radius_pix,
                    )
                    meth_tag = "jump"
                else:
                    t_g, lc_raw, ap_mask, *_ = grow_aperture_bright_core_preseed(
                        flux,
                        time,
                        seed_pix=seeds[k],
                        allowed_mask=allowed,
                        min_pixels=args.min_pixels,
                        amp_q_lo=args.amp_q_lo,
                        amp_q_hi=args.amp_q_hi,
                        amp_min_frac=args.amp_min_frac,
                        core_npix=args.core_npix,
                        core_min_frac_of_peak=args.core_min_frac_of_peak,
                        max_radius_pix=args.max_radius_pix,
                    )
                    meth_tag = "core"

                # Median-normalize raw
                med = np.nanmedian(lc_raw)
                lc_rel = lc_raw / med if np.isfinite(med) and med != 0 else lc_raw

                # Flux-weighted centroid in this aperture
                ap = ap_mask
                if crow is None or ccol is None:
                    denom = np.nansum(flux[:, ap], axis=1)
                    crow = np.nansum(flux[:, ap] * yy[ap][None, :], axis=1) / denom
                    ccol = np.nansum(flux[:, ap] * xx[ap][None, :], axis=1) / denom

                # Optional PSF-width proxy
                psf_sig = None
                if args.psf_proxy:
                    psf_sig = psf_width_sigma(flux, ap, crow, ccol)
                # Saturated-target correction is gated on both morphology and
                # a valid sector entry so an unsuitable request falls back to
                # the standard correction without discarding the extraction.
                use_saturated_correction = bool(
                    args.saturated_systematics_correction
                    and sector_orb is not None
                    and n_use == 1
                )
                if use_saturated_correction:
                    sat_ok, sat_npix = is_heavily_saturated(mean_img, thresh=args.sat_thresh, min_npix=args.sat_min_npix)
                    if not sat_ok:
                        print(
                            "  [INFO] Saturated-target correction requested but the "
                            f"saturation gate failed (npix_above_thresh={sat_npix}); using the standard correction."
                        )
                        use_saturated_correction = False
                if use_saturated_correction:
                    if sector_num not in sector_orb:
                        print(
                            f"  [WARN] Sector {sector_num} is absent from the orbital table; "
                            "using the standard correction."
                        )
                        use_saturated_correction = False

                # Build per-cadence background from faint pixels (if needed)
                back = None
                orbital_phase = None
                orbital_trend = None
                background_scale = None
                if use_saturated_correction:
                    back = estimate_background_faint_pixels(flux, n_faint=args.back_nfaint)
                    # Optimize background scaling and subtract
                    background_scale = optimize_background_scale(lc_raw, back)
                    lc_raw2 = lc_raw - background_scale * back
                    med2 = np.nanmedian(lc_raw2)
                    lc_rel2 = lc_raw2 / med2 if np.isfinite(med2) and med2 != 0 else lc_raw2
                    # Orbit-phase template detrend
                    mid_btjd, freq_cpd = sector_orb[sector_num]
                    lc_rel2_det, orbital_phase, orbital_trend = phase_template_detrend(
                        lc_rel2, t_g, freq_cpd, phase_bin=args.phase_bin
                    )
                    lc_work = lc_rel2_det
                else:
                    lc_work = lc_rel

                # -----------------------------------------------------------------
                # Decorrelation (standard or split-at-mid-sector with background term)
                # -----------------------------------------------------------------
                if args.full_region_sum:
                    lc_xy = lc_work.copy()
                else:
                    if use_saturated_correction:
                        mid_btjd, freq_cpd = sector_orb[sector_num]
                        # Split at mid-sector (downlink break proxy)
                        m1 = t_g < mid_btjd
                        m2 = ~m1
                        lc_xy = lc_work.copy()

                        for mask in (m1, m2):
                            if mask.sum() < 200:
                                continue
                            # Design matrix with background term
                            X = build_design_matrix_with_back(
                                t_g[mask],
                                ccol[mask],
                                crow[mask],
                                back[mask] if back is not None else np.zeros(mask.sum()),
                                knot_spacing_days=args.knot_spacing_days,
                                psf_sigma=(psf_sig[mask] if psf_sig is not None else None),
                            )
                            yseg = lc_work[mask] - np.nanmedian(lc_work[mask])
                            good = np.isfinite(yseg) & np.all(np.isfinite(X), axis=1)
                            if good.sum() > 300:
                                beta = robust_wls(X[good], yseg[good], n_iter=args.robust_iters, huber_k=args.huber_k)
                                lc_xy[mask] = (yseg - (X @ beta)) + np.nanmedian(lc_work[mask])
                    else:
                        X = build_design_matrix(
                            t_g,
                            ccol,
                            crow,
                            knot_spacing_days=args.knot_spacing_days,
                            psf_sigma=psf_sig,
                        )
                        y = lc_work - np.nanmedian(lc_work)
                        good = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
                        if good.sum() > 1000:
                            beta = robust_wls(X[good], y[good], n_iter=args.robust_iters, huber_k=args.huber_k)
                            lc_xy = (y - (X @ beta)) + np.nanmedian(lc_work)
                        else:
                            lc_xy = lc_work.copy()


                # Write outputs (flat structure, labeled by sector/source/target/method).
                output_stem = build_output_stem(sector_tag, source_label, k + 1, meth_tag)
                output_df = make_lightcurve_dataframe(
                    t_g, lc_xy, mission, time_system, "flux_detrended_rel"
                )
                output_df["saturated_systematics_correction"] = bool(use_saturated_correction)
                if active_external_metadata is not None:
                    # Repeated identity columns keep the CSV self-describing;
                    # full provenance, orientation, and checksum are also
                    # written once in the companion JSON below.
                    output_df["aperture_definition"] = "external_text_mask"
                    output_df["external_mask_file"] = active_external_metadata["source_name"]
                    output_df["external_mask_sha256"] = active_external_metadata["sha256"]
                if use_saturated_correction:
                    output_df["background"] = np.asarray(back, float)
                    output_df["background_scale"] = float(background_scale)
                    output_df["orbital_phase"] = np.asarray(orbital_phase, float)
                    if orbital_trend is not None:
                        output_df["orbital_phase_trend"] = np.asarray(orbital_trend, float)
                    np.save(outdir / f"background_{output_stem}.npy", np.asarray(back, float))
                output_df.to_csv(outdir / f"preferred_lc_{output_stem}.csv", index=False)
                save_lightcurve_plot(
                    t_g,
                    lc_xy,
                    outdir / f"preferred_lc_{output_stem}.png",
                    f"{tpf_path.stem} — {source_label} / target{k+1} ({meth_tag})",
                    time_label=mission_time_axis_label(time_system),
                )
                np.save(outdir / f"aperture_mask_{output_stem}.npy", ap_mask.astype(bool))
                np.save(outdir / f"centroid_row_{output_stem}.npy", np.asarray(crow, float))
                np.save(outdir / f"centroid_col_{output_stem}.npy", np.asarray(ccol, float))
                if active_external_metadata is not None:
                    product_metadata = dict(active_external_metadata)
                    product_metadata.update({
                        "tpf_path": str(tpf_path),
                        "output_stem": output_stem,
                        "target_label": source_label,
                        "target_index": int(k + 1),
                        "method": meth_tag,
                        "tpf_shape_rows_columns": [int(mean_img.shape[0]), int(mean_img.shape[1])],
                    })
                    metadata_path = outdir / f"external_aperture_meta_{output_stem}.json"
                    metadata_path.write_text(
                        json.dumps(product_metadata, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8",
                    )

                # Aperture image
                if not args.no_aperture_plots:
                    # gmag may be unavailable if Gaia was skipped/unavailable
                    gtxt = "Gaia G=NA"
                    try:
                        if gmag is not None and np.isfinite(float(gmag[k])):
                            gtxt = f"Gaia G={float(gmag[k]):.2f}"
                    except Exception:
                        pass
                    save_aperture_plot(
                        mean_img,
                        ap_mask.astype(bool),
                        outdir / f"aperture_{output_stem}.png",
                        (
                            f"{tpf_path.stem} — {source_label} / target{k+1} ({meth_tag})\n"
                            + (
                                f"External mask: {active_external_metadata['source_name']}; "
                                f"Npix={active_external_metadata['selected_pixels']}"
                                if active_external_metadata is not None
                                else gtxt
                            )
                        ),
                    )

                print(
                    f"  Wrote {tag} ({meth_tag}): preferred_lc_{output_stem}.csv"
                    f"  Npix={int(np.count_nonzero(ap_mask))}"
                )

    print("\nDone.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)

#!/usr/bin/env python3
"""Jitter-aware TESS PRF photometry helpers.

This module is intentionally independent of the existing aperture-growth code.
It provides a small scene-model interface so the current one-source extractor can
later be extended to simultaneous multi-source PRF fitting without replacing the
motion, PRF, background, or output machinery.

Preferred PRF backends
----------------------
1. ``lkprf`` (``pip install lkprf``)
2. ``TESS_PRF`` (``pip install TESS_PRF``)
3. A pixel-integrated Gaussian approximation, used only as an explicit or
   warned fallback when an official engineering PRF backend is unavailable.

Motion-source hierarchy in ``auto`` mode
----------------------------------------
1. POS_CORR1/POS_CORR2 from the TPF
2. Ensemble image centroid motion
3. Target-local centroid motion
4. Fixed PRF position

Quaternion motion is deliberately not implemented here.  It can be added later
as another MotionProvider without changing the scene or flux-solving classes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence
import json
import pickle
import time

import numpy as np


# -----------------------------------------------------------------------------
# Public data structures
# -----------------------------------------------------------------------------


@dataclass
class SceneSource:
    """One point source in local postage-stamp coordinates.

    Parameters
    ----------
    row, column
        Zero-based local pixel coordinates. Integer coordinates are pixel
        centers, matching the convention used by lkprf and TESS_PRF.
    label
        Human-readable source label used in metadata and output columns.
    source_id, magnitude
        Optional catalog information retained in scene tables and metadata.
    role
        ``"target"`` for the cadence-variable source, otherwise ``"neighbor"``.
    """

    row: float
    column: float
    label: str = "target"
    source_id: str | None = None
    magnitude: float | None = None
    role: str = "neighbor"


@dataclass
class PRFPhotometryConfig:
    backend: str = "auto"  # auto | lkprf | tess_prf | gaussian
    motion_source: str = "auto"  # auto | poscorr | ensemble | target | fixed
    background: str = "plane"  # none | constant | plane
    min_prf_weight: float = 1.0e-5
    fit_radius: float = 6.0
    # Requested regular-grid spacing for the precomputed moving PRF.  The
    # actual spacing may be made coarser so neither axis exceeds
    # ``grid_max_points_per_axis``.  A value <= 0 requests automatic spacing.
    shift_quantization: float = 0.01
    grid_max_points_per_axis: int = 21
    interpolation_chunk_size: int = 4096
    max_abs_shift: float = 2.0
    centroid_radius: float = 3.5
    ensemble_bright_fraction: float = 0.20
    min_motion_finite_fraction: float = 0.50
    allow_gaussian_fallback: bool = True
    refine_reference_position: bool = True
    reference_search_radius: float = 0.75
    reference_search_step: float = 0.10
    reference_fit_radius: float = 5.0
    reference_background: str = "plane"
    motion_coupling_warn_correlation: float = 0.50
    motion_coupling_warn_peak_to_peak: float = 0.005
    saturation_tessmag_limit: float = 7.5
    saturation_absolute_threshold: float | None = None
    # Scene controls.  ``gaia`` means the extractor supplied a target-first
    # Gaia source list.  The initial multi-source release holds neighbors fixed
    # at their jointly fitted reference-image fluxes while fitting the target
    # and background at every cadence.
    scene_mode: str = "single"  # single | gaia
    neighbor_treatment: str = "fixed"  # fixed (future: chunk, selected_free)
    # Which cadence light curves to produce from a Gaia scene.  ``primary`` is
    # deliberately the default.  ``all`` extracts each usable retained source
    # in turn while holding every other source at its reference-scene flux.
    source_output_mode: str = "primary"  # primary | all
    neighbor_min_contribution_fraction: float = 1.0e-4
    all_source_min_captured_fraction: float = 0.10
    scene_duplicate_distance: float = 0.05
    scene_degeneracy_correlation: float = 0.9995
    scene_condition_warn: float = 1.0e8
    save_diagnostics: bool = True
    gap_days_for_scaling: float = 0.5


@dataclass
class TPFMetadata:
    mission: str = "UNKNOWN"
    product_type: str = "TPF"
    sector: int | None = None
    camera: int | None = None
    ccd: int | None = None
    tessmag: float | None = None
    flux_unit: str | None = None
    origin_row: float = 0.0
    origin_column: float = 0.0
    source: str = "unknown"


@dataclass
class MotionSolution:
    row_shift: np.ndarray
    column_shift: np.ndarray
    source: str
    details: dict = field(default_factory=dict)


@dataclass
class PRFPhotometryResult:
    time: np.ndarray
    source_flux: np.ndarray  # shape (ntime, nsource)
    source_flux_err: np.ndarray  # shape (ntime, nsource)
    background: np.ndarray
    row_shift: np.ndarray
    column_shift: np.ndarray
    captured_fraction: np.ndarray  # shape (ntime, nsource)
    residual_rms: np.ndarray
    sources: list[SceneSource]
    backend: str
    motion_source: str
    metadata: dict
    reference_prfs: np.ndarray | None = None  # shape (nsource, ny, nx)
    reference_source_fluxes: np.ndarray | None = None
    reference_model_image: np.ndarray | None = None
    contaminant_model_image: np.ndarray | None = None
    mean_residual_image: np.ndarray | None = None
    # Optional source-specific cadence products.  The one-dimensional fields
    # above remain the primary-source values for backward compatibility.
    source_background: np.ndarray | None = None  # shape (ntime, nsource)
    source_residual_rms: np.ndarray | None = None  # shape (ntime, nsource)
    source_mean_residual_images: np.ndarray | None = None  # shape (nsource, ny, nx)


# -----------------------------------------------------------------------------
# General helpers
# -----------------------------------------------------------------------------


def _as_float_array(value) -> np.ndarray | None:
    if value is None:
        return None
    try:
        if hasattr(value, "value"):
            value = value.value
        return np.asarray(value, dtype=float)
    except Exception:
        return None


def _finite_fraction(x: np.ndarray | None) -> float:
    if x is None:
        return 0.0
    x = np.asarray(x)
    return float(np.mean(np.isfinite(x))) if x.size else 0.0


def _robust_motion_range(x: np.ndarray | None) -> float:
    """Central 98% range of a motion series, in pixels."""
    if x is None:
        return np.nan
    values = np.asarray(x, float)
    values = values[np.isfinite(values)]
    if values.size < 2:
        return np.nan
    lo, hi = np.nanpercentile(values, [1.0, 99.0])
    return float(hi - lo)


def _motion_pair_diagnostics(pair) -> dict:
    if pair is None:
        return {
            "row_finite_fraction": 0.0,
            "column_finite_fraction": 0.0,
            "row_robust_range_pix": None,
            "column_robust_range_pix": None,
        }
    row, column = pair
    row_range = _robust_motion_range(row)
    column_range = _robust_motion_range(column)
    return {
        "row_finite_fraction": _finite_fraction(row),
        "column_finite_fraction": _finite_fraction(column),
        "row_robust_range_pix": float(row_range) if np.isfinite(row_range) else None,
        "column_robust_range_pix": float(column_range) if np.isfinite(column_range) else None,
    }


def _motion_pair_has_variation(pair, minimum_range: float = 1.0e-7) -> bool:
    """Reject constant placeholder motion while retaining effectively 1-D jitter."""
    if pair is None:
        return False
    ranges = [_robust_motion_range(pair[0]), _robust_motion_range(pair[1])]
    finite = [value for value in ranges if np.isfinite(value)]
    return bool(finite and max(finite) > float(minimum_range))


def _robust_sigma(x: np.ndarray) -> float:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return np.nan
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    if np.isfinite(mad) and mad > 0:
        return float(1.4826 * mad)
    return float(np.nanstd(x))


def _fill_interpolate(x: np.ndarray, default: float = 0.0) -> np.ndarray:
    x = np.asarray(x, float).copy()
    idx = np.arange(x.size)
    good = np.isfinite(x)
    if good.sum() == 0:
        return np.full_like(x, float(default))
    if good.sum() == 1:
        return np.full_like(x, float(x[good][0]))
    x[~good] = np.interp(idx[~good], idx[good], x[good])
    return x


def _clean_shift(x: np.ndarray, max_abs_shift: float) -> np.ndarray:
    x = _fill_interpolate(x, default=0.0)
    x -= np.nanmedian(x)
    sig = _robust_sigma(x)
    if np.isfinite(sig) and sig > 0:
        bad = np.abs(x - np.nanmedian(x)) > max(8.0 * sig, 0.25)
        if np.any(bad) and np.any(~bad):
            good_idx = np.flatnonzero(~bad)
            bad_idx = np.flatnonzero(bad)
            x[bad] = np.interp(bad_idx, good_idx, x[~bad])
    if np.isfinite(max_abs_shift) and max_abs_shift > 0:
        x = np.clip(x, -float(max_abs_shift), float(max_abs_shift))
    return x


def segment_indices_by_gaps(time: np.ndarray, gap_days: float = 0.5) -> list[np.ndarray]:
    t = np.asarray(time, float)
    good = np.isfinite(t)
    idx = np.flatnonzero(good)
    if idx.size == 0:
        return []
    tt = t[idx]
    breaks = np.where(np.diff(tt) > float(gap_days))[0]
    starts = np.r_[0, breaks + 1]
    ends = np.r_[breaks + 1, tt.size]
    return [idx[s:e] for s, e in zip(starts, ends) if e > s]


def median_scale_by_segment(time: np.ndarray, flux: np.ndarray, gap_days: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """Return segment-normalized flux and the normalization used per cadence."""
    f = np.asarray(flux, float).copy()
    norm = np.ones_like(f, dtype=float)
    for ii in segment_indices_by_gaps(time, gap_days=gap_days):
        med = np.nanmedian(f[ii])
        if np.isfinite(med) and med != 0:
            f[ii] /= med
            norm[ii] = med
        else:
            f[ii] = np.nan
            norm[ii] = np.nan
    return f, norm


def representative_image(flux_cube: np.ndarray, max_cadences: int = 2048) -> np.ndarray:
    """Robust representative image from an evenly sampled cadence subset."""
    flux = np.asarray(flux_cube, float)
    if flux.ndim != 3:
        raise ValueError(f"Expected a 3-D flux cube, got {flux.shape}.")
    nt = len(flux)
    if nt == 0:
        return np.full(flux.shape[1:], np.nan)
    nuse = min(nt, max(32, int(max_cadences)))
    indices = np.unique(np.linspace(0, nt - 1, nuse).astype(int))
    return np.nanmedian(flux[indices], axis=0)


def refine_reference_position(mean_image: np.ndarray, row: float, column: float, radius: float = 2.5) -> tuple[float, float]:
    """Refine a nominal source position with a local background-subtracted centroid."""
    img = np.asarray(mean_image, float)
    ny, nx = img.shape
    yy, xx = np.indices(img.shape, dtype=float)
    mask = ((yy - float(row)) ** 2 + (xx - float(column)) ** 2) <= float(radius) ** 2
    if mask.sum() < 3:
        return float(row), float(column)
    vals = img[mask]
    finite = np.isfinite(vals)
    if finite.sum() < 3:
        return float(row), float(column)
    background = np.nanpercentile(vals[finite], 15.0)
    weights = np.where(np.isfinite(img) & mask, np.maximum(img - background, 0.0), 0.0)
    denom = np.sum(weights)
    if not np.isfinite(denom) or denom <= 0:
        return float(row), float(column)
    rr = float(np.sum(weights * yy) / denom)
    cc = float(np.sum(weights * xx) / denom)
    if not (np.isfinite(rr) and np.isfinite(cc)):
        return float(row), float(column)
    # Avoid allowing a confused centroid to jump to a different source.
    if np.hypot(rr - row, cc - column) > max(1.5, radius):
        return float(row), float(column)
    return rr, cc


def _longest_true_run(values: np.ndarray) -> int:
    values = np.asarray(values, bool)
    best = cur = 0
    for value in values:
        cur = cur + 1 if value else 0
        best = max(best, cur)
    return int(best)


def _longest_true_run_near(values: np.ndarray, index: float, tolerance: int = 2) -> int:
    """Longest true run that crosses, or nearly crosses, a target row."""
    values = np.asarray(values, bool)
    if values.size == 0 or not np.isfinite(index):
        return 0
    best = 0
    start = None
    for i, value in enumerate(np.r_[values, False]):
        if value and start is None:
            start = i
        elif not value and start is not None:
            stop = i - 1
            if (start - int(tolerance)) <= float(index) <= (stop + int(tolerance)):
                best = max(best, stop - start + 1)
            start = None
    return int(best)


def _electron_rate_unit(unit: str | None) -> bool:
    """Return True when a unit string denotes electrons per second."""
    if unit is None:
        return False
    text = str(unit).strip().lower()
    compact = text.replace(" ", "").replace("electrons", "electron")
    return any(token in compact for token in (
        "electron/s", "electron/sec", "electronsecond-1", "electron*s-1",
        "e-/s", "e-/sec", "e-/second", "e/s", "e/sec",
    ))


def assess_saturation(
    mean_image: np.ndarray,
    *,
    tessmag: float | None = None,
    tessmag_limit: float = 7.5,
    absolute_threshold: float | None = None,
    absolute_min_pixels: int = 3,
    source_row: float | None = None,
    source_column: float | None = None,
    local_radius: float = 8.0,
    flux_unit: str | None = None,
    product_type: str | None = None,
) -> dict:
    """Return a conservative saturation/bleed assessment.

    The engineering PRF describes unsaturated images.  A fixed count threshold
    alone is brittle, so the decision combines TESS magnitude, a vertical
    bleed-column morphology test, a small absolute-threshold test, and a
    flat-core diagnostic.  The returned dictionary is JSON serializable and is
    written into PRF metadata.
    """
    full_img = np.asarray(mean_image, float)
    if full_img.ndim != 2:
        raise ValueError(f"Expected a 2-D mean image, got shape {full_img.shape}.")

    # Saturation is a property of the requested source, not of an unrelated
    # bright object elsewhere in a large TESSCut.  Restrict every pixel-based
    # diagnostic to a target-centred box when a source position is available.
    local_used = False
    y0 = x0 = 0
    local_row = local_column = None
    img = full_img
    try:
        row = float(source_row)
        column = float(source_column)
        radius = max(3.0, float(local_radius))
        if np.isfinite(row) and np.isfinite(column) and np.isfinite(radius):
            y0 = max(0, int(np.floor(row - radius)))
            y1 = min(full_img.shape[0], int(np.ceil(row + radius)) + 1)
            x0 = max(0, int(np.floor(column - radius)))
            x1 = min(full_img.shape[1], int(np.ceil(column + radius)) + 1)
            if y1 > y0 and x1 > x0:
                img = full_img[y0:y1, x0:x1]
                local_row = row - y0
                local_column = column - x0
                local_used = True
            else:
                img = full_img
    except Exception:
        img = full_img

    finite = np.isfinite(img)
    vals = img[finite]
    if vals.size == 0:
        return {"saturated": False, "reasons": [], "status": "empty_image"}

    background = float(np.nanpercentile(vals, 20.0))
    mad = float(np.nanmedian(np.abs(vals - np.nanmedian(vals))))
    sigma = 1.4826 * mad if np.isfinite(mad) and mad > 0 else float(np.nanstd(vals))
    peak = float(np.nanmax(vals))
    signal_peak = max(peak - background, 0.0)

    # Bleed trails are long, nearly vertical runs of high-signal pixels.  The
    # threshold is relative to the source peak but is also kept well above the
    # robust background scatter.
    bright_level = background + max(0.08 * signal_peak, 20.0 * sigma if np.isfinite(sigma) else 0.0)
    bright = finite & (img >= bright_level)
    if local_used:
        # Only columns passing through the target core are allowed to define a
        # target bleed trail, and the run must cross the target row.  This
        # prevents a distant saturated star or detector column from vetoing a
        # PRF fit to the requested TESSCut target.
        candidate_columns = np.flatnonzero(
            np.abs(np.arange(img.shape[1], dtype=float) - float(local_column)) <= 2.5
        )
        column_counts = np.zeros(img.shape[1], dtype=int)
        longest_runs = np.zeros(img.shape[1], dtype=int)
        for j in candidate_columns:
            column_counts[j] = int(np.count_nonzero(bright[:, j]))
            longest_runs[j] = _longest_true_run_near(bright[:, j], float(local_row), tolerance=2)
        candidate_rows = np.flatnonzero(
            np.abs(np.arange(img.shape[0], dtype=float) - float(local_row)) <= 2.5
        )
        horizontal_runs = np.zeros(img.shape[0], dtype=int)
        for i in candidate_rows:
            horizontal_runs[i] = _longest_true_run_near(
                bright[i, :], float(local_column), tolerance=2
            )
    else:
        column_counts = np.sum(bright, axis=0)
        longest_runs = np.array([_longest_true_run(bright[:, j]) for j in range(img.shape[1])], int)
        horizontal_runs = np.array([_longest_true_run(bright[i, :]) for i in range(img.shape[0])], int)
    max_column_count = int(np.max(column_counts)) if column_counts.size else 0
    max_vertical_run = int(np.max(longest_runs)) if longest_runs.size else 0
    max_horizontal_run = int(np.max(horizontal_runs)) if horizontal_runs.size else 0
    # An ordinary undersampled PRF commonly has five bright pixels across both
    # axes.  Require a longer and distinctly vertical structure before calling
    # it a bleed trail.
    bleed_column = bool(
        max_vertical_run >= 7 and max_column_count >= 7 and
        max_vertical_run >= max_horizontal_run + 2
    )

    # A normal undersampled star can put comparable flux in four pixels.  Call
    # the core flat/clipped only when at least six pixels lie within 2% of the
    # background-subtracted peak.
    flat_core_pixels = int(np.count_nonzero(
        finite & (img >= background + 0.98 * signal_peak)
    )) if signal_peak > 0 else 0
    flat_core = bool(flat_core_pixels >= 6)

    above_absolute = 0
    absolute_trigger = False
    unit_known = flux_unit is not None and str(flux_unit).strip() != ""
    absolute_unit_ok = (not unit_known) or _electron_rate_unit(flux_unit)
    if absolute_threshold is not None and absolute_unit_ok:
        try:
            threshold = float(absolute_threshold)
        except Exception:
            threshold = np.nan
        if np.isfinite(threshold) and threshold > 0:
            above_absolute = int(np.count_nonzero(finite & (img >= threshold)))
            absolute_trigger = above_absolute >= max(1, int(absolute_min_pixels))

    magnitude_trigger = False
    tessmag_value = None
    try:
        # TESSCut/Astrocut has no unique SPOC target.  Its header TESSMAG is a
        # placeholder and is invalid even if a caller passes it explicitly.
        if str(product_type or "").strip().upper() == "TESSCUT":
            tessmag = None
        tessmag_value = float(tessmag) if tessmag is not None else None
        magnitude_trigger = bool(
            tessmag_value is not None and np.isfinite(tessmag_value) and
            tessmag_value <= float(tessmag_limit)
        )
    except Exception:
        tessmag_value = None

    # Bright count levels by themselves do not prove clipping.  They become a
    # veto only with a genuinely flat core; a target-centred bleed trail or a
    # reliable target magnitude remains sufficient on its own.
    saturated = bool(magnitude_trigger or bleed_column or (absolute_trigger and flat_core))
    reasons = []
    if magnitude_trigger:
        reasons.append(f"TESSMAG={tessmag_value:.3f} <= {float(tessmag_limit):.2f}")
    if bleed_column:
        reasons.append(f"vertical bleed morphology (run={max_vertical_run} pixels)")
    if absolute_trigger and (flat_core or bleed_column or magnitude_trigger):
        reasons.append(f"{above_absolute} pixels >= {float(absolute_threshold):g}")
    # A source centered near a pixel corner can naturally illuminate four
    # pixels almost equally, so a flat core is supporting evidence rather than
    # a standalone saturation trigger.
    if flat_core and (absolute_trigger or bleed_column):
        reasons.append("flat/clipped bright core")

    return {
        "saturated": saturated,
        "reasons": reasons,
        "tessmag": tessmag_value,
        "tessmag_limit": float(tessmag_limit),
        "background": background,
        "robust_sigma": float(sigma) if np.isfinite(sigma) else None,
        "peak": peak,
        "bright_level": float(bright_level),
        "max_vertical_run": max_vertical_run,
        "max_horizontal_run": max_horizontal_run,
        "max_bright_pixels_in_column": max_column_count,
        "flat_core": flat_core,
        "flat_core_pixels": flat_core_pixels,
        "absolute_threshold": absolute_threshold,
        "absolute_threshold_applied": bool(absolute_threshold is not None and absolute_unit_ok),
        "flux_unit": str(flux_unit) if flux_unit is not None else None,
        "product_type": str(product_type) if product_type is not None else None,
        "pixels_above_absolute_threshold": above_absolute,
        "target_local_assessment": local_used,
        "assessment_origin_row": int(y0),
        "assessment_origin_column": int(x0),
        "assessment_shape": [int(img.shape[0]), int(img.shape[1])],
    }


# -----------------------------------------------------------------------------
# TPF metadata and auxiliary arrays
# -----------------------------------------------------------------------------


def _first_header_value(headers: Sequence, keys: Sequence[str], default=None):
    for header in headers:
        if header is None:
            continue
        for key in keys:
            try:
                val = header.get(key)
            except Exception:
                try:
                    val = header[key]
                except Exception:
                    continue
            if val not in (None, ""):
                return val
    return default


def infer_tpf_metadata(tpf_obj=None, tpf_path: str | Path | None = None) -> TPFMetadata:
    headers = []
    fits_flux_unit = None
    meta = getattr(tpf_obj, "meta", None) if tpf_obj is not None else None
    if meta is not None:
        headers.append(meta)

    origin_row = None
    origin_col = None
    for attr, dest in (("row", "row"), ("column", "col")):
        if tpf_obj is None:
            continue
        try:
            val = float(getattr(tpf_obj, attr))
            if dest == "row":
                origin_row = val
            else:
                origin_col = val
        except Exception:
            pass

    if tpf_obj is not None:
        try:
            hdu = getattr(tpf_obj, "hdu", None)
            if hdu is not None:
                for item in hdu:
                    headers.append(getattr(item, "header", None))
        except Exception:
            pass

    if tpf_path is not None:
        try:
            from astropy.io import fits
            with fits.open(Path(tpf_path), memmap=True) as hdul:
                headers.extend([h.header for h in hdul[:3]])
                for hdu in hdul:
                    columns = getattr(hdu, "columns", None)
                    if columns is None or "FLUX" not in (getattr(columns, "names", None) or []):
                        continue
                    try:
                        fits_flux_unit = columns["FLUX"].unit
                    except Exception:
                        fits_flux_unit = None
                    break
        except Exception:
            pass

    def _int_or_none(value):
        try:
            return int(value)
        except Exception:
            return None

    raw_mission = _first_header_value(headers, ["MISSION", "TELESCOP", "OBSERVAT"])
    mission_text = str(raw_mission or "").strip().upper()
    filename_text = Path(tpf_path).name.lower() if tpf_path is not None else ""
    class_text = type(tpf_obj).__name__.lower() if tpf_obj is not None else ""
    campaign_value = _first_header_value(headers, ["CAMPAIGN"])
    creator_text = str(_first_header_value(headers, ["CREATOR", "PROCNAME", "ORIGIN"], "") or "").lower()
    product_type = "TESSCUT" if (
        "astrocut" in filename_text or "astrocut" in creator_text or "tesscut" in creator_text
    ) else "TPF"
    # K2 files are represented by Lightkurve's KeplerTargetPixelFile class, so
    # campaign and filename checks intentionally precede generic Kepler checks.
    if "K2" in mission_text or filename_text.startswith("ktwo") or campaign_value not in (None, ""):
        mission = "K2"
    elif "TESS" in mission_text or "tess" in class_text or filename_text.startswith("tess") or "astrocut" in filename_text:
        mission = "TESS"
    elif "KEPLER" in mission_text or "kepler" in class_text or filename_text.startswith("kplr"):
        mission = "KEPLER"
    else:
        mission = "UNKNOWN"

    sector = _int_or_none(_first_header_value(headers, ["SECTOR"]))
    camera = _int_or_none(_first_header_value(headers, ["CAMERA"]))
    ccd = _int_or_none(_first_header_value(headers, ["CCD"]))
    tessmag = None
    try:
        tessmag = float(_first_header_value(headers, ["TESSMAG", "TMAG"]))
        if not np.isfinite(tessmag):
            tessmag = None
    except Exception:
        tessmag = None
    # Astrocut stamps do not have a unique SPOC target; their placeholder
    # TESSMAG (commonly 0.0) describes neither the requested central source nor
    # any Gaia-selected source and must never trigger a saturation veto.
    if product_type == "TESSCUT":
        tessmag = None

    flux_unit = None
    try:
        unit = getattr(getattr(tpf_obj, "flux", None), "unit", None)
        if unit is not None and str(unit).strip():
            flux_unit = str(unit)
    except Exception:
        pass
    if flux_unit is None and fits_flux_unit is not None and str(fits_flux_unit).strip():
        flux_unit = str(fits_flux_unit)
    if flux_unit is None:
        value = _first_header_value(headers, ["FLUXUNIT", "BUNIT"])
        if value not in (None, ""):
            flux_unit = str(value)

    # Pixel-coordinate WCS reference values.  Lightkurve's .row/.column are
    # preferred because they already account for TPF conventions.
    if origin_col is None:
        val = _first_header_value(headers, ["1CRV4P", "CRVAL1P", "COLSTART", "COLUMN"])
        try:
            origin_col = float(val)
        except Exception:
            origin_col = 0.0
    if origin_row is None:
        val = _first_header_value(headers, ["2CRV4P", "CRVAL2P", "ROWSTART", "ROW"])
        try:
            origin_row = float(val)
        except Exception:
            origin_row = 0.0

    return TPFMetadata(
        mission=mission,
        product_type=product_type,
        sector=sector,
        camera=camera,
        ccd=ccd,
        tessmag=tessmag,
        flux_unit=flux_unit,
        origin_row=float(origin_row),
        origin_column=float(origin_col),
        source="lightkurve" if tpf_obj is not None else "fits",
    )


def _select_cadences(arr: np.ndarray | None, cadence_indices: np.ndarray | None, ntime: int) -> np.ndarray | None:
    if arr is None:
        return None
    arr = np.asarray(arr)
    if cadence_indices is not None and arr.shape[0] != ntime:
        try:
            arr = arr[np.asarray(cadence_indices, dtype=int)]
        except Exception:
            pass
    if arr.shape[0] != ntime:
        return None
    return np.asarray(arr, float)


def load_tpf_auxiliary(
    ntime: int,
    tpf_obj=None,
    tpf_path: str | Path | None = None,
    cadence_indices: np.ndarray | None = None,
) -> dict:
    """Load POS_CORR and FLUX_ERR arrays without requiring a specific TPF class."""
    pos1 = pos2 = flux_err = None

    if tpf_obj is not None:
        for name in ("pos_corr1", "POS_CORR1"):
            try:
                pos1 = _as_float_array(getattr(tpf_obj, name))
                if pos1 is not None:
                    break
            except Exception:
                pass
        for name in ("pos_corr2", "POS_CORR2"):
            try:
                pos2 = _as_float_array(getattr(tpf_obj, name))
                if pos2 is not None:
                    break
            except Exception:
                pass
        try:
            flux_err = _as_float_array(getattr(tpf_obj, "flux_err"))
        except Exception:
            pass

        try:
            data = tpf_obj.hdu[1].data
            names = set(data.names)
            if pos1 is None and "POS_CORR1" in names:
                pos1 = np.asarray(data["POS_CORR1"], float)
            if pos2 is None and "POS_CORR2" in names:
                pos2 = np.asarray(data["POS_CORR2"], float)
            if flux_err is None and "FLUX_ERR" in names:
                flux_err = np.asarray(data["FLUX_ERR"], float)
        except Exception:
            pass

    if tpf_path is not None and (pos1 is None or pos2 is None or flux_err is None):
        try:
            from astropy.io import fits
            with fits.open(Path(tpf_path), memmap=True) as hdul:
                data = hdul[1].data
                names = set(data.names)
                if pos1 is None and "POS_CORR1" in names:
                    pos1 = np.asarray(data["POS_CORR1"], float)
                if pos2 is None and "POS_CORR2" in names:
                    pos2 = np.asarray(data["POS_CORR2"], float)
                if flux_err is None and "FLUX_ERR" in names:
                    flux_err = np.asarray(data["FLUX_ERR"], float)
        except Exception:
            pass

    return {
        "pos_corr1": _select_cadences(pos1, cadence_indices, ntime),
        "pos_corr2": _select_cadences(pos2, cadence_indices, ntime),
        "flux_err": _select_cadences(flux_err, cadence_indices, ntime),
    }


# -----------------------------------------------------------------------------
# Motion providers
# -----------------------------------------------------------------------------


def _centroid_series(flux_cube: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    f = np.asarray(flux_cube, float)
    mask = np.asarray(mask, bool)
    nt, ny, nx = f.shape
    yy, xx = np.indices((ny, nx), dtype=float)
    sel = np.flatnonzero(mask.ravel())
    if sel.size < 2:
        return np.full(nt, np.nan), np.full(nt, np.nan)
    flat = f.reshape(nt, ny * nx)[:, sel]
    # Cadence background from the lower quartile of selected pixels.
    background = np.nanpercentile(flat, 20.0, axis=1)
    work = flat - background[:, None]
    work = np.where(np.isfinite(work), np.maximum(work, 0.0), 0.0)
    denom = np.sum(work, axis=1)
    rows = yy.ravel()[sel]
    cols = xx.ravel()[sel]
    r = np.full(nt, np.nan)
    c = np.full(nt, np.nan)
    good = np.isfinite(denom) & (denom > 0)
    if np.any(good):
        r[good] = (work[good] @ rows) / denom[good]
        c[good] = (work[good] @ cols) / denom[good]
    return r, c


def image_motion_candidates(
    flux_cube: np.ndarray,
    source: SceneSource,
    config: PRFPhotometryConfig,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    mean_image = np.nanmean(np.asarray(flux_cube, float), axis=0)
    ny, nx = mean_image.shape
    yy, xx = np.indices((ny, nx), dtype=float)

    # Target-local centroid.
    target_mask = ((yy - source.row) ** 2 + (xx - source.column) ** 2) <= float(config.centroid_radius) ** 2
    tr, tc = _centroid_series(flux_cube, target_mask)
    tr -= np.nanmedian(tr)
    tc -= np.nanmedian(tc)

    # Ensemble centroid from the brightest fraction of finite mean-image pixels.
    finite_vals = mean_image[np.isfinite(mean_image)]
    if finite_vals.size:
        q = 100.0 * (1.0 - np.clip(config.ensemble_bright_fraction, 0.02, 0.8))
        threshold = np.nanpercentile(finite_vals, q)
        ensemble_mask = np.isfinite(mean_image) & (mean_image >= threshold)
    else:
        ensemble_mask = np.zeros_like(mean_image, dtype=bool)
    if ensemble_mask.sum() < 4:
        ensemble_mask = target_mask
    er, ec = _centroid_series(flux_cube, ensemble_mask)
    er -= np.nanmedian(er)
    ec -= np.nanmedian(ec)

    return {"target": (tr, tc), "ensemble": (er, ec)}


def _poscorr_transform_score(row: np.ndarray, col: np.ndarray, image_row: np.ndarray, image_col: np.ndarray) -> float:
    good = np.isfinite(row) & np.isfinite(col) & np.isfinite(image_row) & np.isfinite(image_col)
    if good.sum() < 20:
        return np.inf
    dr = image_row[good] - row[good]
    dc = image_col[good] - col[good]
    sr = _robust_sigma(dr)
    sc = _robust_sigma(dc)
    return float(np.hypot(sr if np.isfinite(sr) else 1e9, sc if np.isfinite(sc) else 1e9))


def _robust_linear_motion_map(X: np.ndarray, Y: np.ndarray, initial: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Robustly fit Y = X @ A.T, initialized near a sign/swap mapping."""
    X = np.asarray(X, float)
    Y = np.asarray(Y, float)
    good = np.all(np.isfinite(X), axis=1) & np.all(np.isfinite(Y), axis=1)
    if good.sum() < 50:
        return np.asarray(initial, float), good
    keep = good.copy()
    A = np.asarray(initial, float).copy()
    for _ in range(6):
        # Tiny ridge toward the selected sign/swap convention prevents a noisy
        # centroid series from producing an implausible transform.
        x = X[keep]
        y = Y[keep]
        ridge = max(1.0e-8, 1.0e-4 * float(len(x)))
        lhs = x.T @ x + ridge * np.eye(2)
        rhs = x.T @ y + ridge * initial.T
        try:
            B = np.linalg.solve(lhs, rhs)  # X @ B = Y
        except np.linalg.LinAlgError:
            break
        A_new = B.T
        resid = Y - X @ A_new.T
        radius = np.hypot(resid[:, 0], resid[:, 1])
        med = np.nanmedian(radius[keep])
        sig = _robust_sigma(radius[keep])
        if not np.isfinite(sig) or sig <= 0:
            A = A_new
            break
        new_keep = good & (radius <= med + 5.0 * sig)
        A = A_new
        if np.array_equal(new_keep, keep):
            break
        keep = new_keep
    return A, keep


def calibrate_poscorr(
    pos1: np.ndarray,
    pos2: np.ndarray,
    image_row: np.ndarray | None,
    image_col: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, str, float | None, list[list[float]]]:
    p1 = np.asarray(pos1, float) - np.nanmedian(pos1)
    p2 = np.asarray(pos2, float) - np.nanmedian(pos2)
    X = np.column_stack([p1, p2])
    canonical = np.array([[0.0, 1.0], [1.0, 0.0]])
    if image_row is None or image_col is None:
        mapped = X @ canonical.T
        return mapped[:, 0], mapped[:, 1], "canonical POS_CORR row/column mapping", None, canonical.tolist()

    candidates = []
    for swap in (False, True):
        for sr in (-1.0, 1.0):
            for sc in (-1.0, 1.0):
                if swap:
                    A0 = np.array([[sr, 0.0], [0.0, sc]])
                    label = f"row={sr:+g}*POS_CORR1,col={sc:+g}*POS_CORR2"
                else:
                    A0 = np.array([[0.0, sr], [sc, 0.0]])
                    label = f"row={sr:+g}*POS_CORR2,col={sc:+g}*POS_CORR1"
                mapped = X @ A0.T
                score = _poscorr_transform_score(mapped[:, 0], mapped[:, 1], image_row, image_col)
                candidates.append((score, A0, label))
    initial_score, initial, initial_label = min(candidates, key=lambda item: item[0])

    Y = np.column_stack([np.asarray(image_row, float), np.asarray(image_col, float)])
    A, keep = _robust_linear_motion_map(X, Y, initial)
    mapped = X @ A.T
    score = _poscorr_transform_score(mapped[:, 0], mapped[:, 1], image_row, image_col)

    # Reject pathological solutions and retain the discrete convention.
    det = float(np.linalg.det(A))
    norm = float(np.linalg.norm(A))
    improves = (
        np.isfinite(score) and np.isfinite(initial_score) and
        score <= 0.99 * float(initial_score)
    )
    if (not np.all(np.isfinite(A)) or norm > 4.0 or abs(det) < 0.05 or not improves):
        A = initial
        mapped = X @ A.T
        score = _poscorr_transform_score(mapped[:, 0], mapped[:, 1], image_row, image_col)
        reason = "implausible fit" if (norm > 4.0 or abs(det) < 0.05) else "no material improvement"
        label = initial_label + f" (discrete fallback: {reason})"
    else:
        label = (
            "full linear POS_CORR mapping: "
            f"row={A[0,0]:+.5f}*POS_CORR1{A[0,1]:+.5f}*POS_CORR2; "
            f"col={A[1,0]:+.5f}*POS_CORR1{A[1,1]:+.5f}*POS_CORR2"
        )
    return mapped[:, 0], mapped[:, 1], label, float(score) if np.isfinite(score) else None, A.tolist()


def choose_motion_solution(
    flux_cube: np.ndarray,
    source: SceneSource,
    config: PRFPhotometryConfig,
    pos_corr1: np.ndarray | None = None,
    pos_corr2: np.ndarray | None = None,
) -> MotionSolution:
    nt = len(flux_cube)
    image_candidates = image_motion_candidates(flux_cube, source, config)
    requested = str(config.motion_source).strip().lower()
    if requested not in {"auto", "poscorr", "ensemble", "target", "fixed"}:
        raise ValueError(f"Unknown PRF motion source: {config.motion_source!r}")

    def valid_pair(pair) -> bool:
        return bool(
            pair is not None and
            min(_finite_fraction(pair[0]), _finite_fraction(pair[1])) >= config.min_motion_finite_fraction and
            _motion_pair_has_variation(pair)
        )

    fallback_reasons = []

    if requested == "fixed":
        return MotionSolution(
            row_shift=np.zeros(nt, dtype=float),
            column_shift=np.zeros(nt, dtype=float),
            source="fixed",
            details={"reason": "Fixed PRF position explicitly requested."},
        )

    if requested in {"auto", "poscorr"} and pos_corr1 is not None and pos_corr2 is not None:
        poscorr_pair = (pos_corr1, pos_corr2)
        poscorr_diag = _motion_pair_diagnostics(poscorr_pair)
        coverage_ok = min(
            poscorr_diag["row_finite_fraction"],
            poscorr_diag["column_finite_fraction"],
        ) >= config.min_motion_finite_fraction
        variation_ok = _motion_pair_has_variation(poscorr_pair)
        if coverage_ok and variation_ok:
            ref_pair = image_candidates.get("ensemble") if valid_pair(image_candidates.get("ensemble")) else image_candidates.get("target")
            ref_row, ref_col = ref_pair if valid_pair(ref_pair) else (None, None)
            rr, cc, transform, score, matrix = calibrate_poscorr(pos_corr1, pos_corr2, ref_row, ref_col)
            rr = _clean_shift(rr, config.max_abs_shift)
            cc = _clean_shift(cc, config.max_abs_shift)
            output_diag = _motion_pair_diagnostics((rr, cc))
            if not _motion_pair_has_variation((rr, cc)):
                fallback_reasons.append(
                    "POS_CORR mapping produced constant/near-constant shifts and was rejected."
                )
            else:
                details = {
                    "transform": transform,
                    "matrix_poscorr1_poscorr2_to_row_col": matrix,
                    "image_calibration_residual_pix": score,
                    "input_diagnostics": poscorr_diag,
                    "output_diagnostics": output_diag,
                }
                return MotionSolution(
                    row_shift=rr,
                    column_shift=cc,
                    source="poscorr",
                    details=details,
                )
        elif not coverage_ok:
            fallback_reasons.append(
                "POS_CORR rejected because finite coverage is below "
                f"{float(config.min_motion_finite_fraction):.0%} "
                f"(POS_CORR1={poscorr_diag['row_finite_fraction']:.1%}, "
                f"POS_CORR2={poscorr_diag['column_finite_fraction']:.1%})."
            )
        else:
            r1 = poscorr_diag.get("row_robust_range_pix")
            r2 = poscorr_diag.get("column_robust_range_pix")
            fallback_reasons.append(
                "POS_CORR rejected as a constant/near-constant placeholder "
                f"(central 98% ranges: POS_CORR1={float(r1 or 0.0):.3g} pix, "
                f"POS_CORR2={float(r2 or 0.0):.3g} pix)."
            )

        if requested == "poscorr":
            raise ValueError(fallback_reasons[-1])
    elif requested == "poscorr":
        raise ValueError("POS_CORR motion was requested, but POS_CORR1/POS_CORR2 were not both available.")

    order = [requested] if requested in {"ensemble", "target"} else ["ensemble", "target"]
    for name in order:
        pair = image_candidates.get(name)
        if valid_pair(pair):
            rr, cc = pair
            return MotionSolution(
                row_shift=_clean_shift(rr, config.max_abs_shift),
                column_shift=_clean_shift(cc, config.max_abs_shift),
                source=name,
                details={
                    "selection_reason": (
                        "Selected by the configured image-motion hierarchy."
                        if not fallback_reasons else
                        "Selected after a higher-priority motion source was rejected."
                    ),
                    "fallback_reasons": list(fallback_reasons),
                    "motion_diagnostics": _motion_pair_diagnostics(pair),
                },
            )
        if requested == name:
            diag = _motion_pair_diagnostics(pair)
            raise ValueError(
                f"{name} centroid motion was requested, but it lacks sufficient finite "
                f"coverage or measurable variation (diagnostics={diag})."
            )

    return MotionSolution(
        row_shift=np.zeros(nt, dtype=float),
        column_shift=np.zeros(nt, dtype=float),
        source="fixed",
        details={
            "reason": "No usable cadence-dependent motion source.",
            "fallback_reasons": list(fallback_reasons),
        },
    )


# -----------------------------------------------------------------------------
# PRF backends
# -----------------------------------------------------------------------------


class BasePRFBackend:
    name = "base"
    native_orientation = True

    def evaluate(self, row: float, column: float, shape: tuple[int, int]) -> np.ndarray:
        raise NotImplementedError


class LKPRFBackend(BasePRFBackend):
    name = "lkprf"

    def __init__(self, metadata: TPFMetadata):
        if metadata.camera is None or metadata.ccd is None:
            raise ValueError("lkprf requires CAMERA and CCD metadata.")
        import lkprf

        kwargs = {"camera": int(metadata.camera), "ccd": int(metadata.ccd)}
        if metadata.sector is not None:
            kwargs["sector"] = int(metadata.sector)
        self.model = lkprf.TESSPRF(**kwargs)
        self.metadata = metadata

    def evaluate(self, row: float, column: float, shape: tuple[int, int]) -> np.ndarray:
        # lkprf accepts subpixel target coordinates, but its image origin and
        # shape are used internally as array indices and therefore must be
        # ordinary Python integers.  Lightkurve exposes TPF row/column origins
        # as numeric scalars, and infer_tpf_metadata stores them as floats;
        # passing values such as 1045.0 through to lkprf can trigger
        # ``IndexError: arrays used as indices must be of integer type``.
        ny, nx = int(shape[0]), int(shape[1])
        origin_row = int(np.rint(float(self.metadata.origin_row)))
        origin_column = int(np.rint(float(self.metadata.origin_column)))
        target = (origin_row + float(row), origin_column + float(column))
        origin = (origin_row, origin_column)
        arr = self.model.evaluate(
            targets=[target],
            origin=origin,
            shape=(ny, nx),
        )
        return np.asarray(arr[0], float)


class TESSPRFBackend(BasePRFBackend):
    name = "TESS_PRF"

    def __init__(self, metadata: TPFMetadata, shape: tuple[int, int]):
        if metadata.camera is None or metadata.ccd is None or metadata.sector is None:
            raise ValueError("TESS_PRF requires SECTOR, CAMERA, and CCD metadata.")
        constructor = None
        errors = []
        for import_style in ("PRF", "TESS_PRF"):
            try:
                if import_style == "PRF":
                    import PRF
                    constructor = PRF.TESS_PRF
                else:
                    from TESS_PRF import TESS_PRF
                    constructor = TESS_PRF
                break
            except Exception as exc:
                errors.append(str(exc))
        if constructor is None:
            raise ImportError("Could not import TESS_PRF (tried PRF.TESS_PRF and TESS_PRF.TESS_PRF): " + "; ".join(errors))
        ny, nx = shape
        center_col = metadata.origin_column + 0.5 * (nx - 1)
        center_row = metadata.origin_row + 0.5 * (ny - 1)
        self.model = constructor(
            int(metadata.camera), int(metadata.ccd), int(metadata.sector), center_col, center_row
        )

    def evaluate(self, row: float, column: float, shape: tuple[int, int]) -> np.ndarray:
        # TESS_PRF.locate follows (column, row, shape).
        return np.asarray(self.model.locate(float(column), float(row), shape), float)


class GaussianPRFBackend(BasePRFBackend):
    name = "gaussian"

    def __init__(self, sigma: float = 0.65):
        self.sigma = float(sigma)

    def evaluate(self, row: float, column: float, shape: tuple[int, int]) -> np.ndarray:
        ny, nx = shape
        yy, xx = np.indices((ny, nx), dtype=float)
        arr = np.exp(-0.5 * (((yy - float(row)) / self.sigma) ** 2 + ((xx - float(column)) / self.sigma) ** 2))
        total = np.sum(arr)
        return arr / total if np.isfinite(total) and total > 0 else arr


def _center_of_mass(arr: np.ndarray) -> tuple[float, float]:
    a = np.asarray(arr, float)
    a = np.where(np.isfinite(a), np.maximum(a, 0.0), 0.0)
    total = np.sum(a)
    if total <= 0:
        return np.nan, np.nan
    yy, xx = np.indices(a.shape, dtype=float)
    return float(np.sum(a * yy) / total), float(np.sum(a * xx) / total)


def validate_prf_image(arr: np.ndarray, shape: tuple[int, int], backend_name: str) -> np.ndarray:
    """Validate a backend PRF without changing its documented orientation."""
    a = np.asarray(arr, float)
    if a.shape != tuple(shape):
        raise ValueError(f"{backend_name} returned PRF shape {a.shape}; expected {shape}.")
    return a


def orient_prf_image(arr: np.ndarray, expected_row: float, expected_col: float, shape: tuple[int, int]) -> np.ndarray:
    """Legacy orientation helper, retained only for explicitly uncertain backends.

    Official lkprf and TESS_PRF outputs are now trusted in their documented
    native orientation.  Selecting flips/transposes by center of light can
    silently corrupt asymmetric PRFs.
    """
    a = np.asarray(arr, float)
    candidates = [a, np.flipud(a), np.fliplr(a), np.flipud(np.fliplr(a))]
    if a.T.shape == shape:
        at = a.T
        candidates.extend([at, np.flipud(at), np.fliplr(at), np.flipud(np.fliplr(at))])
    candidates = [c for c in candidates if c.shape == shape]
    if not candidates:
        raise ValueError(f"PRF backend returned shape {a.shape}; expected {shape}.")
    scored = []
    for c in candidates:
        rr, cc = _center_of_mass(c)
        score = np.hypot(rr - expected_row, cc - expected_col) if np.isfinite(rr + cc) else np.inf
        scored.append((score, c))
    return np.asarray(min(scored, key=lambda x: x[0])[1], float)


def create_prf_backend(
    config: PRFPhotometryConfig,
    metadata: TPFMetadata,
    shape: tuple[int, int],
) -> tuple[BasePRFBackend, list[str]]:
    requested = str(config.backend).strip().lower()
    if requested not in {"auto", "lkprf", "tess_prf", "gaussian"}:
        raise ValueError(f"Unknown PRF backend: {config.backend!r}")
    attempts = [requested] if requested != "auto" else ["lkprf", "tess_prf"]
    messages = []
    for name in attempts:
        try:
            if name == "lkprf":
                return LKPRFBackend(metadata), messages
            if name == "tess_prf":
                return TESSPRFBackend(metadata, shape), messages
            if name == "gaussian":
                return GaussianPRFBackend(), messages
        except Exception as exc:
            messages.append(f"{name}: {type(exc).__name__}: {exc}")
            if requested != "auto":
                raise
    if config.allow_gaussian_fallback:
        messages.append("Official TESS PRF backend unavailable; using Gaussian approximation.")
        return GaussianPRFBackend(), messages
    raise ImportError(
        "No usable TESS PRF backend. Install lkprf (`pip install lkprf`) or TESS_PRF (`pip install TESS_PRF`). "
        + " | ".join(messages)
    )



def _reference_fit_mask(shape: tuple[int, int], row: float, column: float, radius: float) -> np.ndarray:
    yy, xx = np.indices(shape, dtype=float)
    return ((yy - float(row)) ** 2 + (xx - float(column)) ** 2) <= float(radius) ** 2


def _solve_reference_image_model(
    mean_image: np.ndarray,
    prf: np.ndarray,
    mask: np.ndarray,
    background_mode: str = "plane",
) -> tuple[float, np.ndarray, np.ndarray]:
    image = np.asarray(mean_image, float)
    prf = np.asarray(prf, float)
    sel = np.flatnonzero(np.asarray(mask, bool).ravel())
    y = image.ravel()[sel]
    p = prf.ravel()[sel]
    yy, xx = np.indices(image.shape, dtype=float)
    ycoord = yy.ravel()[sel] - np.nanmean(yy.ravel()[sel])
    xcoord = xx.ravel()[sel] - np.nanmean(xx.ravel()[sel])
    cols = [p]
    mode = str(background_mode).strip().lower()
    if mode in {"constant", "plane"}:
        cols.append(np.ones_like(p))
    if mode == "plane":
        cols.extend([ycoord, xcoord])
    X = np.column_stack(cols)
    good = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    if good.sum() < X.shape[1] + 3:
        return np.inf, np.full(X.shape[1], np.nan), np.full(image.shape, np.nan)
    beta, *_ = np.linalg.lstsq(X[good], y[good], rcond=None)
    resid = y[good] - X[good] @ beta
    # Use the full residual sum for registration.  Clipping the largest core
    # residuals can make an incorrectly shifted PRF appear artificially good.
    score = float(np.nanmean(np.square(resid))) if resid.size else np.inf
    model_full = np.full(image.size, np.nan)
    finite_all = np.all(np.isfinite(X), axis=1)
    model_full[sel[finite_all]] = X[finite_all] @ beta
    residual_full = image - model_full.reshape(image.shape)
    return score, beta, residual_full


def fit_reference_position_with_prf(
    mean_image: np.ndarray,
    source: SceneSource,
    backend: BasePRFBackend,
    config: PRFPhotometryConfig,
) -> tuple[SceneSource, dict]:
    """Fit the static subpixel location directly to the median/mean image."""
    shape = np.asarray(mean_image).shape
    start_row, start_col = float(source.row), float(source.column)
    radius = max(0.05, float(config.reference_search_radius))
    step = max(0.02, float(config.reference_search_step))
    offsets = np.arange(-radius, radius + 0.5 * step, step)
    if offsets.size > 31:
        offsets = np.linspace(-radius, radius, 31)
    mask = _reference_fit_mask(shape, start_row, start_col, float(config.reference_fit_radius))
    best = (np.inf, start_row, start_col, None, None)
    neval = 0

    def objective(row: float, col: float):
        nonlocal neval
        arr = backend.evaluate(float(row), float(col), shape)
        arr = validate_prf_image(arr, shape, backend.name)
        arr = np.where(np.isfinite(arr) & (arr > 0), arr, 0.0)
        score, beta, residual = _solve_reference_image_model(
            mean_image, arr, mask, background_mode=config.reference_background
        )
        neval += 1
        return score, beta, residual

    for dr in offsets:
        for dc in offsets:
            row = start_row + float(dr)
            col = start_col + float(dc)
            score, beta, residual = objective(row, col)
            if score < best[0]:
                best = (score, row, col, beta, residual)

    # Continuous bounded refinement after the coarse search.
    try:
        from scipy.optimize import minimize

        lo = np.array([start_row - radius, start_col - radius], float)
        hi = np.array([start_row + radius, start_col + radius], float)

        def scalar(z):
            if np.any(z < lo) or np.any(z > hi):
                return 1.0e30
            return objective(float(z[0]), float(z[1]))[0]

        opt = minimize(
            scalar,
            x0=np.array([best[1], best[2]], float),
            method="Powell",
            bounds=list(zip(lo, hi)),
            options={"xtol": 2.0e-4, "ftol": 1.0e-8, "maxiter": 80},
        )
        if opt.success and np.all(np.isfinite(opt.x)):
            score, beta, residual = objective(float(opt.x[0]), float(opt.x[1]))
            if score < best[0]:
                best = (score, float(opt.x[0]), float(opt.x[1]), beta, residual)
    except Exception:
        pass

    fitted = SceneSource(best[1], best[2], source.label, source.source_id, source.magnitude, source.role)
    details = {
        "method": "direct_prf_fit_to_mean_image",
        "initial_row": start_row,
        "initial_column": start_col,
        "fitted_row": float(best[1]),
        "fitted_column": float(best[2]),
        "delta_row": float(best[1] - start_row),
        "delta_column": float(best[2] - start_col),
        "search_radius_pixels": radius,
        "coarse_step_pixels": step,
        "backend_evaluations": int(neval),
        "mean_squared_residual": float(best[0]),
        "background_model": str(config.reference_background),
        "linear_coefficients": None if best[3] is None else np.asarray(best[3], float).tolist(),
    }
    return fitted, details



def _scene_source_copy(source: SceneSource, row: float, column: float) -> SceneSource:
    return SceneSource(
        float(row), float(column), str(source.label),
        source.source_id, source.magnitude, str(source.role),
    )


def _scene_linear_model(
    image: np.ndarray,
    prfs: np.ndarray,
    background_mode: str = "plane",
    mask: np.ndarray | None = None,
) -> dict:
    """Fit non-negative source amplitudes plus an unconstrained background."""
    image = np.asarray(image, float)
    prfs = np.asarray(prfs, float)
    if prfs.ndim != 3 or prfs.shape[1:] != image.shape:
        raise ValueError("PRF scene array must have shape (source,row,column).")
    nsrc = prfs.shape[0]
    if mask is None:
        mask = np.ones(image.shape, dtype=bool)
    mask = np.asarray(mask, bool) & np.isfinite(image)
    sel = np.flatnonzero(mask.ravel())
    yy, xx = np.indices(image.shape, dtype=float)
    y0 = float(np.nanmean(yy.ravel()[sel])) if sel.size else 0.0
    x0 = float(np.nanmean(xx.ravel()[sel])) if sel.size else 0.0
    source_cols = prfs.reshape(nsrc, -1)[:, sel].T
    columns = [source_cols]
    mode = str(background_mode).strip().lower()
    if mode not in {"none", "constant", "plane"}:
        raise ValueError(f"Unknown reference background model: {background_mode!r}")
    if mode in {"constant", "plane"}:
        columns.append(np.ones((sel.size, 1), float))
    if mode == "plane":
        columns.extend([
            (yy.ravel()[sel] - y0)[:, None],
            (xx.ravel()[sel] - x0)[:, None],
        ])
    X = np.hstack(columns)
    y = image.ravel()[sel]
    good = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    if good.sum() < X.shape[1] + 3:
        return {
            "score": np.inf, "beta": np.full(X.shape[1], np.nan),
            "model": np.full(image.shape, np.nan),
            "residual": np.full(image.shape, np.nan),
            "condition_number": np.inf, "max_source_correlation": np.nan,
            "source_correlation_matrix": np.full((nsrc, nsrc), np.nan),
        }
    Xg, yg = X[good], y[good]
    lower = np.r_[np.zeros(nsrc), np.full(X.shape[1] - nsrc, -np.inf)]
    upper = np.full(X.shape[1], np.inf)
    try:
        from scipy.optimize import lsq_linear
        fit = lsq_linear(Xg, yg, bounds=(lower, upper), method="bvls", tol=1e-10, max_iter=500)
        beta = np.asarray(fit.x, float)
    except Exception:
        # Small active-set fallback: repeatedly remove source columns whose
        # unconstrained coefficient is negative, while retaining backgrounds.
        active = list(range(nsrc))
        bg_idx = list(range(nsrc, X.shape[1]))
        beta = np.zeros(X.shape[1], float)
        for _ in range(nsrc + 1):
            cols = active + bg_idx
            trial, *_ = np.linalg.lstsq(Xg[:, cols], yg, rcond=None)
            beta[:] = 0.0
            beta[cols] = trial
            negatives = [j for j in active if beta[j] < 0]
            if not negatives:
                break
            worst = min(negatives, key=lambda j: beta[j])
            active.remove(worst)
        beta[:nsrc] = np.maximum(beta[:nsrc], 0.0)

    model_vec = X @ beta
    resid_good = yg - Xg @ beta
    score = float(np.nanmean(resid_good ** 2)) if resid_good.size else np.inf
    model = np.full(image.size, np.nan)
    model[sel] = model_vec
    model = model.reshape(image.shape)
    residual = image - model

    src = source_cols[good]
    norms = np.sqrt(np.sum(src * src, axis=0))
    normalized = np.divide(src, norms[None, :], out=np.zeros_like(src), where=norms[None, :] > 0)
    corr = normalized.T @ normalized
    if nsrc > 1:
        tri = corr[np.triu_indices(nsrc, 1)]
        max_corr = float(np.nanmax(np.abs(tri))) if tri.size else np.nan
    else:
        max_corr = 0.0
    try:
        condition = float(np.linalg.cond(src)) if src.size else np.inf
    except Exception:
        condition = np.inf
    return {
        "score": score,
        "beta": beta,
        "model": model,
        "residual": residual,
        "condition_number": condition,
        "max_source_correlation": max_corr,
        "source_correlation_matrix": corr,
        "coordinate_center": [y0, x0],
    }


def fit_reference_scene_with_prf(
    mean_image: np.ndarray,
    sources: Sequence[SceneSource],
    backend: BasePRFBackend,
    config: PRFPhotometryConfig,
    target_index: int = 0,
) -> tuple[list[SceneSource], np.ndarray, dict, np.ndarray, np.ndarray, np.ndarray]:
    """Jointly register and fit a target-first multi-source reference scene.

    Gaia relative positions remain fixed.  Only one common row/column translation
    is fitted, followed by non-negative source amplitudes and the requested
    reference background model.  Negligible neighbors are pruned and the final
    scene is refitted at the selected translation.
    """
    sources = list(sources)
    if not sources:
        raise ValueError("Reference scene requires at least one source.")
    if not (0 <= int(target_index) < len(sources)):
        raise ValueError("target_index is outside the scene source list.")
    shape = np.asarray(mean_image).shape
    radius = max(0.05, float(config.reference_search_radius))
    step = max(0.03, float(config.reference_search_step))
    offsets = np.arange(-radius, radius + 0.5 * step, step)
    if offsets.size > 17:
        offsets = np.linspace(-radius, radius, 17)
    full_mask = np.isfinite(mean_image)
    neval = 0

    def evaluate(source_list: Sequence[SceneSource], dr: float, dc: float):
        nonlocal neval
        prfs = []
        for source in source_list:
            arr = backend.evaluate(float(source.row + dr), float(source.column + dc), shape)
            arr = validate_prf_image(arr, shape, backend.name)
            prfs.append(np.where(np.isfinite(arr) & (arr > 0), arr, 0.0))
            neval += 1
        prfs = np.asarray(prfs, float)
        fit = _scene_linear_model(
            mean_image, prfs, background_mode=config.reference_background, mask=full_mask
        )
        return fit, prfs

    best = None
    for dr in offsets:
        for dc in offsets:
            fit, prfs = evaluate(sources, float(dr), float(dc))
            if best is None or fit["score"] < best[0]:
                best = (float(fit["score"]), float(dr), float(dc), fit, prfs)

    # Refine the common translation continuously.  The bounds ensure the scene
    # cannot walk away from the WCS solution to explain unrelated image flux.
    try:
        from scipy.optimize import minimize
        def objective(z):
            if np.any(np.abs(z) > radius):
                return 1.0e30
            return evaluate(sources, float(z[0]), float(z[1]))[0]["score"]
        opt = minimize(
            objective, np.array([best[1], best[2]], float), method="Powell",
            bounds=[(-radius, radius), (-radius, radius)],
            options={"xtol": 2.0e-4, "ftol": 1.0e-8, "maxiter": 60},
        )
        if opt.success and np.all(np.isfinite(opt.x)):
            fit, prfs = evaluate(sources, float(opt.x[0]), float(opt.x[1]))
            if fit["score"] < best[0]:
                best = (float(fit["score"]), float(opt.x[0]), float(opt.x[1]), fit, prfs)
    except Exception:
        pass

    _, dr, dc, fit, prfs = best
    beta = np.asarray(fit["beta"], float)
    ref_flux = beta[:len(sources)]
    captured = np.sum(prfs, axis=(1, 2))
    contribution = ref_flux * captured
    target_contribution = float(contribution[int(target_index)])
    if not np.isfinite(target_contribution) or target_contribution <= 0:
        target_contribution = float(np.nanmax(contribution)) if np.isfinite(contribution).any() else 1.0
    ratios = np.divide(
        contribution, target_contribution,
        out=np.zeros_like(contribution), where=np.isfinite(contribution),
    )
    min_fraction = max(0.0, float(config.neighbor_min_contribution_fraction))
    keep = np.asarray(ratios >= min_fraction, bool)
    keep[int(target_index)] = True
    kept_indices = np.flatnonzero(keep)
    dropped_indices = np.flatnonzero(~keep)

    # Preserve target-first ordering.  Extractors are expected to supply the
    # target at index zero, but this also handles a nonzero target_index safely.
    ordered = [int(target_index)] + [int(i) for i in kept_indices if int(i) != int(target_index)]
    retained_original = [sources[i] for i in ordered]
    fit, prfs = evaluate(retained_original, dr, dc)
    ref_flux = np.asarray(fit["beta"], float)[:len(retained_original)]
    refined = [_scene_source_copy(s, s.row + dr, s.column + dc) for s in retained_original]
    captured = np.sum(prfs, axis=(1, 2))
    contribution = ref_flux * captured
    target_contribution = float(contribution[0]) if len(contribution) else np.nan
    fractions = np.divide(
        contribution, target_contribution,
        out=np.full_like(contribution, np.nan),
        where=np.isfinite(target_contribution) & (target_contribution != 0),
    )
    source_model_images = ref_flux[:, None, None] * prfs
    target_model = source_model_images[0]
    contaminant_model = np.sum(source_model_images[1:], axis=0) if len(refined) > 1 else np.zeros(shape, float)

    source_rows = []
    corr_matrix = np.asarray(fit["source_correlation_matrix"], float)
    min_captured = max(0.0, float(config.all_source_min_captured_fraction))
    eligible_indices = []
    for j, (source, fref, cap, frac) in enumerate(zip(refined, ref_flux, captured, fractions)):
        if len(refined) > 1 and corr_matrix.shape == (len(refined), len(refined)):
            others = np.delete(np.abs(corr_matrix[j]), j)
            max_other_corr = float(np.nanmax(others)) if others.size and np.isfinite(others).any() else 0.0
        else:
            max_other_corr = 0.0
        output_reasons = []
        if not np.isfinite(fref) or fref <= 0:
            output_reasons.append("non-positive reference-scene flux")
        if not np.isfinite(cap) or cap < min_captured:
            output_reasons.append(
                f"captured PRF fraction {float(cap):.4g} is below {min_captured:.4g}"
            )
        if np.isfinite(max_other_corr) and max_other_corr >= float(config.scene_degeneracy_correlation):
            output_reasons.append(
                f"PRF column is degenerate with another source (correlation {max_other_corr:.5f})"
            )
        # The selected primary is always retained as an output, but warnings are
        # recorded so the diagnostic remains honest about a difficult scene.
        output_eligible = (j == 0) or (not output_reasons)
        if output_eligible:
            eligible_indices.append(j)
        source_rows.append({
            "scene_index": j,
            "role": "target" if j == 0 else "neighbor",
            "label": source.label,
            "source_id": source.source_id,
            "magnitude": source.magnitude,
            "row": float(source.row),
            "column": float(source.column),
            "reference_flux": float(fref),
            "captured_fraction": float(cap),
            "contribution_in_stamp": float(fref * cap),
            "contribution_relative_to_target": float(frac),
            "cadence_treatment": "free" if j == 0 else str(config.neighbor_treatment),
            "max_correlation_with_other_source": max_other_corr,
            "lightcurve_output_eligible": bool(output_eligible),
            "lightcurve_output_reason": "; ".join(output_reasons),
        })
    scene_reasons = []
    if np.isfinite(fit["condition_number"]) and fit["condition_number"] >= float(config.scene_condition_warn):
        scene_reasons.append(f"reference source matrix condition number {fit['condition_number']:.3g}")
    if np.isfinite(fit["max_source_correlation"]) and fit["max_source_correlation"] >= float(config.scene_degeneracy_correlation):
        scene_reasons.append(f"nearly degenerate PRF columns (max correlation {fit['max_source_correlation']:.5f})")
    details = {
        "method": "joint_gaia_reference_scene",
        "common_delta_row": float(dr),
        "common_delta_column": float(dc),
        "search_radius_pixels": radius,
        "coarse_step_pixels": step,
        "backend_evaluations": int(neval),
        "mean_squared_residual": float(fit["score"]),
        "background_model": str(config.reference_background),
        "condition_number": float(fit["condition_number"]),
        "max_source_correlation": float(fit["max_source_correlation"]),
        "source_correlation_matrix": np.asarray(fit["source_correlation_matrix"], float).tolist(),
        "n_input_sources": int(len(sources)),
        "n_retained_sources": int(len(refined)),
        "retained_original_indices": ordered,
        "dropped_original_indices": dropped_indices.astype(int).tolist(),
        "minimum_neighbor_contribution_fraction": min_fraction,
        "minimum_output_captured_fraction": min_captured,
        "output_eligible_source_indices": [int(i) for i in eligible_indices],
        "sources": source_rows,
        "scene_quality_pass": not scene_reasons,
        "scene_quality_reasons": scene_reasons,
    }
    return refined, ref_flux, details, fit["model"], fit["residual"], contaminant_model

def _center_by_segments(time: np.ndarray, values: np.ndarray, gap_days: float) -> np.ndarray:
    out = np.asarray(values, float).copy()
    for ii in segment_indices_by_gaps(time, gap_days=gap_days):
        med = np.nanmedian(out[ii])
        if np.isfinite(med):
            out[ii] -= med
    return out


def motion_coupling_diagnostics(
    time: np.ndarray,
    source_flux: np.ndarray,
    row_shift: np.ndarray,
    column_shift: np.ndarray,
    gap_days: float,
    warn_correlation: float,
    warn_peak_to_peak: float,
) -> dict:
    raw = np.asarray(source_flux, float)
    rel, _ = median_scale_by_segment(time, raw, gap_days=gap_days)
    y = rel - 1.0
    row = _center_by_segments(time, row_shift, gap_days)
    col = _center_by_segments(time, column_shift, gap_days)
    good = np.isfinite(y) & np.isfinite(row) & np.isfinite(col)
    diagnostics = {
        "n_valid": int(good.sum()),
        "row_correlation": None,
        "column_correlation": None,
        "row_slope_relative_flux_per_pixel": None,
        "column_slope_relative_flux_per_pixel": None,
        "predicted_motion_peak_to_peak_relative_flux": None,
        "quality_pass": True,
        "quality_reasons": [],
    }
    if good.sum() < 30:
        diagnostics["quality_reasons"] = ["too few valid cadences for motion-coupling diagnostic"]
        return diagnostics

    def corr(a, b):
        aa = a - np.nanmean(a)
        bb = b - np.nanmean(b)
        den = np.sqrt(np.nansum(aa * aa) * np.nansum(bb * bb))
        return float(np.nansum(aa * bb) / den) if den > 0 else np.nan

    diagnostics["row_correlation"] = corr(y[good], row[good])
    diagnostics["column_correlation"] = corr(y[good], col[good])
    X = np.column_stack([row[good], col[good]])
    yy = y[good]
    keep = np.ones(len(yy), bool)
    beta = np.zeros(2)
    for _ in range(6):
        if keep.sum() < 10:
            break
        beta, *_ = np.linalg.lstsq(X[keep], yy[keep], rcond=None)
        resid = yy - X @ beta
        sig = _robust_sigma(resid[keep])
        if not np.isfinite(sig) or sig <= 0:
            break
        new_keep = np.abs(resid - np.nanmedian(resid[keep])) <= 5.0 * sig
        if np.array_equal(new_keep, keep):
            break
        keep = new_keep
    diagnostics["row_slope_relative_flux_per_pixel"] = float(beta[0])
    diagnostics["column_slope_relative_flux_per_pixel"] = float(beta[1])
    model = X @ beta
    ptp = float(np.nanpercentile(model, 99) - np.nanpercentile(model, 1))
    diagnostics["predicted_motion_peak_to_peak_relative_flux"] = ptp
    corr_values = np.asarray([
        diagnostics["row_correlation"], diagnostics["column_correlation"]
    ], float)
    max_corr = float(np.nanmax(np.abs(corr_values))) if np.isfinite(corr_values).any() else np.nan
    reasons = []
    if np.isfinite(max_corr) and max_corr >= float(warn_correlation):
        reasons.append(f"strong flux-motion correlation ({max_corr:.3f})")
    if np.isfinite(ptp) and ptp >= float(warn_peak_to_peak):
        reasons.append(f"motion model spans {1e6 * ptp:.0f} ppm (1st-99th percentile)")
    diagnostics["quality_pass"] = not reasons
    diagnostics["quality_reasons"] = reasons
    return diagnostics


# -----------------------------------------------------------------------------
# Scene model and flux extraction
# -----------------------------------------------------------------------------


class PRFSceneModel:
    """Cadence-dependent linear PRF scene model.

    Expensive engineering-PRF evaluations are performed only on a bounded,
    regular two-dimensional grid spanning the observed spacecraft motion.  The
    cadence PRF is then obtained by bilinear interpolation between the four
    surrounding grid nodes.  This keeps the number of backend evaluations
    independent of the number of cadences while preserving a general
    multi-source design matrix.

    The current extractors pass one source at a time.  Multiple ``SceneSource``
    entries already produce a multi-source design matrix; adding a Gaia scene
    later therefore requires source construction and policy choices rather than
    a new solver.
    """

    def __init__(
        self,
        sources: Sequence[SceneSource],
        backend: BasePRFBackend,
        shape: tuple[int, int],
        config: PRFPhotometryConfig,
    ):
        if not sources:
            raise ValueError("At least one SceneSource is required.")
        self.sources = list(sources)
        self.backend = backend
        self.shape = tuple(shape)
        self.config = config
        self._row_grid: np.ndarray | None = None
        self._column_grid: np.ndarray | None = None
        self._grid_prfs: np.ndarray | None = None
        self._grid_info: dict = {}

    @staticmethod
    def _finite_motion(values: np.ndarray) -> np.ndarray:
        a = np.asarray(values, float)
        return a[np.isfinite(a)]

    def _make_grid_axis(self, values: np.ndarray) -> tuple[np.ndarray, float]:
        finite = self._finite_motion(values)
        if finite.size == 0:
            finite = np.array([0.0])
        lo = float(min(np.nanmin(finite), 0.0))
        hi = float(max(np.nanmax(finite), 0.0))
        span = hi - lo
        if not np.isfinite(span) or span <= 1.0e-12:
            return np.array([0.5 * (lo + hi)], dtype=float), 0.0

        max_points = max(2, int(self.config.grid_max_points_per_axis))
        requested = abs(float(self.config.shift_quantization))
        if not np.isfinite(requested) or requested <= 0:
            npoint = max_points
        else:
            npoint = int(np.ceil(span / requested)) + 1
            npoint = min(max_points, max(2, npoint))
        axis = np.linspace(lo, hi, npoint, dtype=float)
        actual_step = float(np.nanmedian(np.diff(axis))) if len(axis) > 1 else 0.0
        return axis, actual_step

    def prepare_grid(self, motion: MotionSolution) -> None:
        """Precompute official PRFs on a bounded regular motion grid."""
        if self._grid_prfs is not None:
            return

        row_grid, row_step = self._make_grid_axis(motion.row_shift)
        col_grid, col_step = self._make_grid_axis(motion.column_shift)
        nr, nc = len(row_grid), len(col_grid)
        nsrc = len(self.sources)
        ny, nx = self.shape
        neval = nr * nc * nsrc

        print(
            f"  [PRF] Precomputing {nr} x {nc} motion grid "
            f"({neval} backend evaluation{'s' if neval != 1 else ''}; "
            f"row step={row_step:.4g} pix, column step={col_step:.4g} pix)...",
            flush=True,
        )
        t0 = time.perf_counter()
        grid = np.empty((nr, nc, nsrc, ny, nx), dtype=float)
        progress_stride = max(1, nr // 5)
        for ir, dr in enumerate(row_grid):
            if nr >= 8 and (ir == 0 or ir % progress_stride == 0):
                print(f"  [PRF]   grid row {ir + 1}/{nr}", flush=True)
            for ic, dc in enumerate(col_grid):
                for js, source in enumerate(self.sources):
                    rr = source.row + float(dr)
                    cc = source.column + float(dc)
                    arr = self.backend.evaluate(rr, cc, self.shape)
                    if getattr(self.backend, "native_orientation", True):
                        arr = validate_prf_image(arr, self.shape, self.backend.name)
                    else:
                        arr = orient_prf_image(arr, rr, cc, self.shape)
                    grid[ir, ic, js] = np.where(
                        np.isfinite(arr) & (arr > 0), arr, 0.0
                    )

        self._row_grid = row_grid
        self._column_grid = col_grid
        self._grid_prfs = grid
        elapsed = time.perf_counter() - t0
        self._grid_info = {
            "row_nodes": int(nr),
            "column_nodes": int(nc),
            "backend_evaluations": int(neval),
            "requested_step_pixels": float(self.config.shift_quantization),
            "row_step_pixels": float(row_step),
            "column_step_pixels": float(col_step),
            "row_min_pixels": float(row_grid[0]),
            "row_max_pixels": float(row_grid[-1]),
            "column_min_pixels": float(col_grid[0]),
            "column_max_pixels": float(col_grid[-1]),
            "precompute_seconds": float(elapsed),
            "interpolation": "bilinear",
        }
        print(f"  [PRF] Motion grid ready in {elapsed:.1f} s.", flush=True)

    @staticmethod
    def _axis_brackets(axis: np.ndarray, values: np.ndarray):
        vals = np.asarray(values, float)
        if len(axis) == 1:
            z = np.zeros(vals.shape, dtype=int)
            return z, z, np.zeros(vals.shape, dtype=float)
        clipped = np.clip(vals, axis[0], axis[-1])
        hi = np.searchsorted(axis, clipped, side="right")
        hi = np.clip(hi, 1, len(axis) - 1)
        lo = hi - 1
        denom = axis[hi] - axis[lo]
        frac = np.divide(
            clipped - axis[lo],
            denom,
            out=np.zeros_like(clipped, dtype=float),
            where=denom != 0,
        )
        return lo.astype(int), hi.astype(int), np.clip(frac, 0.0, 1.0)

    def interpolate_prfs(
        self,
        row_shift: np.ndarray | float,
        column_shift: np.ndarray | float,
    ) -> np.ndarray:
        """Return bilinearly interpolated PRFs.

        Returns an array with shape ``(ncadence, nsource, nrow, ncolumn)``.
        """
        if self._grid_prfs is None or self._row_grid is None or self._column_grid is None:
            raise RuntimeError("prepare_grid() must be called before PRF interpolation.")
        rr = np.atleast_1d(np.asarray(row_shift, float))
        cc = np.atleast_1d(np.asarray(column_shift, float))
        if rr.shape != cc.shape:
            raise ValueError("Row and column shift arrays must have the same shape.")
        rr = np.where(np.isfinite(rr), rr, 0.0)
        cc = np.where(np.isfinite(cc), cc, 0.0)
        r0, r1, fr = self._axis_brackets(self._row_grid, rr)
        c0, c1, fc = self._axis_brackets(self._column_grid, cc)
        g = self._grid_prfs
        p00 = g[r0, c0]
        p10 = g[r1, c0]
        p01 = g[r0, c1]
        p11 = g[r1, c1]
        wr0 = (1.0 - fr)[:, None, None, None]
        wr1 = fr[:, None, None, None]
        wc0 = (1.0 - fc)[:, None, None, None]
        wc1 = fc[:, None, None, None]
        return (
            wr0 * wc0 * p00
            + wr1 * wc0 * p10
            + wr0 * wc1 * p01
            + wr1 * wc1 * p11
        )

    def reference_prfs(self) -> np.ndarray:
        if self._grid_prfs is None:
            # This path is mainly useful for diagnostics before extraction.
            dummy = MotionSolution(
                row_shift=np.array([0.0]),
                column_shift=np.array([0.0]),
                source="fixed",
            )
            self.prepare_grid(dummy)
        return self.interpolate_prfs(0.0, 0.0)[0]

    def grid_metadata(self) -> dict:
        return dict(self._grid_info)

    def fit_mask(self) -> np.ndarray:
        ny, nx = self.shape
        yy, xx = np.indices(self.shape, dtype=float)
        mask = np.zeros(self.shape, dtype=bool)
        for source in self.sources:
            mask |= ((yy - source.row) ** 2 + (xx - source.column) ** 2) <= float(self.config.fit_radius) ** 2
        ref = self.reference_prfs()
        peak = np.nanmax(ref, axis=(1, 2))
        for j in range(len(self.sources)):
            threshold = max(float(self.config.min_prf_weight), float(peak[j]) * float(self.config.min_prf_weight))
            mask |= ref[j] >= threshold
        return mask

    def extract(
        self,
        flux_cube: np.ndarray,
        motion: MotionSolution,
        flux_err_cube: np.ndarray | None = None,
        *,
        free_source_indices: Sequence[int] | None = None,
        fixed_source_fluxes: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
        """Extract a cadence-dependent scene.

        ``free_source_indices`` identifies source amplitudes solved at every
        cadence. Other source columns are held at ``fixed_source_fluxes``. This
        makes the target-free / fixed-neighbor mode a special case of the same
        design matrix used by future selectively variable-neighbor modes.
        """
        flux = np.asarray(flux_cube, float)
        nt, ny, nx = flux.shape
        nsrc = len(self.sources)
        if free_source_indices is None:
            free = np.arange(nsrc, dtype=int)
        else:
            free = np.unique(np.asarray(list(free_source_indices), int))
        if free.size == 0 or np.any((free < 0) | (free >= nsrc)):
            raise ValueError("free_source_indices must identify at least one valid scene source.")
        fixed_mask = np.ones(nsrc, dtype=bool)
        fixed_mask[free] = False
        fixed = np.flatnonzero(fixed_mask)
        if fixed.size:
            if fixed_source_fluxes is None:
                raise ValueError("Fixed scene sources require fixed_source_fluxes.")
            fixed_flux = np.asarray(fixed_source_fluxes, float)
            if fixed_flux.shape != (nsrc,):
                raise ValueError(f"fixed_source_fluxes must have shape ({nsrc},).")
            if not np.all(np.isfinite(fixed_flux[fixed])):
                raise ValueError("Fixed neighbor fluxes contain non-finite values.")
        else:
            fixed_flux = np.zeros(nsrc, float)

        self.prepare_grid(motion)
        fit_mask = self.fit_mask()
        sel = np.flatnonzero(fit_mask.ravel())
        background_mode = str(self.config.background).strip().lower()
        nbg = 0 if background_mode == "none" else (1 if background_mode == "constant" else 3)
        if sel.size < free.size + nbg + 2:
            raise ValueError("Too few pixels in PRF fitting region.")

        yy, xx = np.indices((ny, nx), dtype=float)
        ycoord = yy.ravel()[sel] - np.nanmean(yy.ravel()[sel])
        xcoord = xx.ravel()[sel] - np.nanmean(xx.ravel()[sel])
        ydata = flux.reshape(nt, ny * nx)[:, sel]
        edata = None
        if flux_err_cube is not None:
            e = np.asarray(flux_err_cube, float)
            if e.shape == flux.shape:
                edata = e.reshape(nt, ny * nx)[:, sel]

        source_flux = np.full((nt, nsrc), np.nan)
        source_err = np.full((nt, nsrc), np.nan)
        if fixed.size:
            source_flux[:, fixed] = fixed_flux[fixed][None, :]
        background = np.full(nt, np.nan)
        captured = np.full((nt, nsrc), np.nan)
        residual_rms = np.full(nt, np.nan)
        residual_sum = np.zeros((ny, nx), dtype=float)
        residual_count = np.zeros((ny, nx), dtype=float)
        if background_mode not in {"none", "constant", "plane"}:
            raise ValueError(f"Unknown PRF background model: {self.config.background!r}")

        fixed_cols = []
        if background_mode in {"constant", "plane"}:
            fixed_cols.append(np.ones(sel.size))
        if background_mode == "plane":
            fixed_cols.extend([ycoord, xcoord])

        chunk_size = max(1, int(self.config.interpolation_chunk_size))
        for start in range(0, nt, chunk_size):
            stop = min(nt, start + chunk_size)
            idx = np.arange(start, stop, dtype=int)
            prfs = self.interpolate_prfs(motion.row_shift[idx], motion.column_shift[idx])
            captured[idx, :] = np.sum(prfs, axis=(2, 3))
            p_all = prfs.reshape(len(idx), nsrc, ny * nx)[:, :, sel]
            fixed_model = np.zeros((len(idx), sel.size), float)
            if fixed.size:
                fixed_model = np.einsum(
                    "csp,s->cp", p_all[:, fixed, :], fixed_flux[fixed], optimize=True
                )
            Y_original = np.asarray(ydata[idx], float)
            Y = Y_original - fixed_model

            # Fast batched path for the default target-free/fixed-neighbor mode.
            if free.size == 1:
                P = np.asarray(p_all[:, free[0], :], float)
                X = np.concatenate(
                    [P[:, :, None]] + [
                        np.broadcast_to(c[None, :, None], (len(idx), sel.size, 1))
                        for c in fixed_cols
                    ], axis=2,
                )
                finite = np.isfinite(Y) & np.all(np.isfinite(X), axis=2)
                if edata is not None:
                    E = np.asarray(edata[idx], float)
                    valid = finite & np.isfinite(E) & (E > 0)
                    W = np.where(valid, 1.0 / np.square(E), 0.0)
                else:
                    valid = finite
                    W = valid.astype(float)
                X0 = np.where(valid[:, :, None], X, 0.0)
                Y0 = np.where(valid, Y, 0.0)
                normal = np.einsum("cpi,cp,cpj->cij", X0, W, X0, optimize=True)
                rhs = np.einsum("cpi,cp,cp->ci", X0, W, Y0, optimize=True)
                npar = X.shape[2]
                nvalid = np.sum(valid, axis=1)
                beta = np.full((len(idx), npar), np.nan)
                ok = nvalid >= npar + 2
                for local in np.flatnonzero(ok):
                    try:
                        beta[local] = np.linalg.solve(normal[local], rhs[local])
                    except np.linalg.LinAlgError:
                        ok[local] = False
                variable_model = np.einsum("cpi,ci->cp", X, beta, optimize=True)
                model_all = fixed_model + variable_model
                resid_all = np.where(valid, Y_original - model_all, np.nan)
                source_flux[idx[ok], free[0]] = beta[ok, 0]
                if background_mode != "none":
                    background[idx[ok]] = beta[ok, 1]
                residual_rms[idx[ok]] = np.sqrt(np.nanmean(np.square(resid_all[ok]), axis=1))
                weighted_rss = np.nansum(
                    W * np.square(np.where(np.isfinite(resid_all), resid_all, 0.0)), axis=1
                )
                dof = np.maximum(1, nvalid - npar)
                scale2 = weighted_rss / dof
                for local in np.flatnonzero(ok):
                    try:
                        cov = np.linalg.inv(normal[local]) * scale2[local]
                        source_err[idx[local], free[0]] = np.sqrt(max(float(cov[0, 0]), 0.0))
                    except np.linalg.LinAlgError:
                        pass
                rsum = np.nansum(resid_all, axis=0)
                rcount = np.sum(np.isfinite(resid_all), axis=0)
                flat_sum = np.zeros(ny * nx, dtype=float)
                flat_count = np.zeros(ny * nx, dtype=float)
                flat_sum[sel] = rsum
                flat_count[sel] = rcount
                residual_sum += flat_sum.reshape(ny, nx)
                residual_count += flat_count.reshape(ny, nx)
                continue

            # General path for several cadence-variable sources.
            for local, i in enumerate(idx):
                p_free = p_all[local, free, :].T
                base = np.column_stack([p_free] + [c[:, None] for c in fixed_cols]) if fixed_cols else p_free
                y = Y[local]
                good = np.isfinite(y) & np.all(np.isfinite(base), axis=1)
                if edata is not None:
                    err = edata[i]
                    good &= np.isfinite(err) & (err > 0)
                if good.sum() < base.shape[1] + 2:
                    continue
                X = base[good]
                yy_i = y[good]
                if edata is not None:
                    ww = 1.0 / np.square(edata[i][good])
                    sw = np.sqrt(ww)
                    Xw, yw = X * sw[:, None], yy_i * sw
                else:
                    ww = np.ones_like(yy_i)
                    Xw, yw = X, yy_i
                try:
                    beta, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
                except np.linalg.LinAlgError:
                    continue
                variable_model = X @ beta
                model = fixed_model[local, good] + variable_model
                resid = Y_original[local, good] - model
                source_flux[i, free] = beta[:free.size]
                if background_mode != "none":
                    background[i] = beta[free.size]
                residual_rms[i] = np.sqrt(np.nanmean(resid ** 2))
                dof = max(1, good.sum() - X.shape[1])
                scale2 = np.sum(ww * resid ** 2) / dof
                try:
                    cov = np.linalg.inv(Xw.T @ Xw) * scale2
                    source_err[i, free] = np.sqrt(np.maximum(np.diag(cov)[:free.size], 0.0))
                except np.linalg.LinAlgError:
                    pass
                resid_full = np.full(sel.size, np.nan)
                resid_full[good] = resid
                tmp = np.full(ny * nx, np.nan)
                tmp[sel] = resid_full
                finite = np.isfinite(tmp)
                residual_sum.ravel()[finite] += tmp[finite]
                residual_count.ravel()[finite] += 1

        mean_residual = np.divide(
            residual_sum, residual_count,
            out=np.full_like(residual_sum, np.nan), where=residual_count > 0,
        )
        return source_flux, source_err, background, captured, residual_rms, mean_residual


# -----------------------------------------------------------------------------
# High-level extraction and outputs
# -----------------------------------------------------------------------------


def _validate_config(config: PRFPhotometryConfig) -> None:
    """Validate public configuration values before allocating PRF grids."""
    if config.backend not in {"auto", "lkprf", "tess_prf", "gaussian"}:
        raise ValueError(f"Unknown PRF backend: {config.backend!r}")
    if config.motion_source not in {"auto", "poscorr", "ensemble", "target", "fixed"}:
        raise ValueError(f"Unknown PRF motion source: {config.motion_source!r}")
    if config.scene_mode not in {"single", "gaia"}:
        raise ValueError(f"Unknown PRF scene mode: {config.scene_mode!r}")
    if config.neighbor_treatment != "fixed":
        raise ValueError("Only fixed-neighbor cadence treatment is currently supported.")
    if config.source_output_mode not in {"primary", "all"}:
        raise ValueError(f"Unknown PRF source output mode: {config.source_output_mode!r}")
    if config.background not in {"none", "constant", "plane"}:
        raise ValueError(f"Unknown PRF background model: {config.background!r}")
    if config.reference_background not in {"none", "constant", "plane"}:
        raise ValueError(f"Unknown reference-image background model: {config.reference_background!r}")
    if float(config.fit_radius) <= 0 or float(config.reference_fit_radius) <= 0:
        raise ValueError("PRF fit radii must be positive.")
    if float(config.max_abs_shift) <= 0:
        raise ValueError("PRF maximum absolute shift must be positive.")
    if float(config.reference_search_radius) < 0 or float(config.reference_search_step) <= 0:
        raise ValueError("Reference-position search radius/step are invalid.")
    if float(config.min_prf_weight) < 0:
        raise ValueError("Minimum PRF weight cannot be negative.")
    if int(config.grid_max_points_per_axis) < 2:
        raise ValueError("PRF motion grids require at least two points per axis.")
    if int(config.interpolation_chunk_size) < 1:
        raise ValueError("PRF interpolation chunk size must be positive.")


def extract_jitter_aware_prf(
    time: np.ndarray,
    flux_cube: np.ndarray,
    sources: Sequence[SceneSource],
    config: PRFPhotometryConfig | None = None,
    *,
    tpf_obj=None,
    tpf_path: str | Path | None = None,
    cadence_indices: np.ndarray | None = None,
    flux_err_cube: np.ndarray | None = None,
) -> PRFPhotometryResult:
    config = config or PRFPhotometryConfig()
    _validate_config(config)
    time = np.asarray(time, float)
    flux = np.asarray(flux_cube, float)
    sources = list(sources)
    if flux.ndim != 3:
        raise ValueError(f"Expected flux cube with shape (cadence,row,column), got {flux.shape}.")
    if len(time) != len(flux):
        raise ValueError("Time and flux-cube cadence counts do not match.")
    if len(time) == 0 or not np.isfinite(time).any():
        raise ValueError("PRF extraction requires at least one finite-time cadence.")
    if not sources:
        raise ValueError("At least one PRF scene source is required.")
    if not np.isfinite([sources[0].row, sources[0].column]).all():
        raise ValueError("The primary PRF source position must be finite.")

    mean_image = representative_image(flux)
    metadata = infer_tpf_metadata(tpf_obj=tpf_obj, tpf_path=tpf_path)
    if metadata.mission in {"KEPLER", "K2"}:
        raise ValueError(
            f"{metadata.mission} target-pixel data detected. The current PRF module "
            "contains TESS engineering PRFs only, so PRF extraction is disabled for "
            "Kepler/K2. Aperture extraction may still be used."
        )
    saturation = assess_saturation(
        mean_image, tessmag=metadata.tessmag,
        tessmag_limit=float(config.saturation_tessmag_limit),
        absolute_threshold=config.saturation_absolute_threshold,
        source_row=float(sources[0].row),
        source_column=float(sources[0].column),
        local_radius=max(8.0, float(config.fit_radius)),
        flux_unit=metadata.flux_unit,
        product_type=metadata.product_type,
    )
    if saturation.get("saturated", False):
        raise ValueError(
            "Source appears saturated or bleed-dominated; ordinary engineering-PRF "
            "photometry is disabled. Reasons: " + "; ".join(saturation.get("reasons", []))
        )

    backend, backend_messages = create_prf_backend(config, metadata, mean_image.shape)
    requested_scene = str(config.scene_mode).strip().lower()
    multi_scene = requested_scene == "gaia" and len(sources) > 1
    reference_model = None
    contaminant_model = np.zeros(mean_image.shape, float)
    reference_fits = []
    scene_details = {"method": "single_source", "n_input_sources": len(sources), "n_retained_sources": 1}

    if multi_scene:
        refined_sources, reference_fluxes, scene_details, reference_model, reference_residual, contaminant_model = (
            fit_reference_scene_with_prf(mean_image, sources, backend, config, target_index=0)
        )
        reference_fits = [scene_details]
        print(
            f"  [PRF] Joint reference scene: {scene_details['n_input_sources']} input, "
            f"{scene_details['n_retained_sources']} retained; "
            f"delta=({scene_details['common_delta_row']:+.4f}, "
            f"{scene_details['common_delta_column']:+.4f}) pix; "
            f"condition={scene_details['condition_number']:.3g}", flush=True,
        )
        if not scene_details.get("scene_quality_pass", True):
            print("  [WARN] PRF reference scene is poorly conditioned: " + "; ".join(scene_details.get("scene_quality_reasons", [])))
    else:
        source = sources[0]
        rr, cc = source.row, source.column
        centroid_start = _scene_source_copy(source, rr, cc)
        if config.refine_reference_position:
            rr0, cc0 = refine_reference_position(mean_image, rr, cc)
            centroid_start = _scene_source_copy(source, rr0, cc0)
            fitted, fit_details = fit_reference_position_with_prf(mean_image, centroid_start, backend, config)
            fit_details["input_wcs_or_seed_row"] = float(rr)
            fit_details["input_wcs_or_seed_column"] = float(cc)
            refined_sources = [fitted]
            reference_fits = [fit_details]
            print(
                f"  [PRF] Reference position {source.label}: "
                f"({float(rr):.4f}, {float(cc):.4f}) -> "
                f"({fitted.row:.4f}, {fitted.column:.4f}); "
                f"delta=({fit_details['delta_row']:+.4f}, {fit_details['delta_column']:+.4f}) pix",
                flush=True,
            )
        else:
            refined_sources = [centroid_start]
            reference_fits = [{"method": "disabled", "fitted_row": rr, "fitted_column": cc}]
        arr = backend.evaluate(refined_sources[0].row, refined_sources[0].column, mean_image.shape)
        arr = validate_prf_image(arr, mean_image.shape, backend.name)
        arr = np.where(np.isfinite(arr) & (arr > 0), arr, 0.0)
        mask = _reference_fit_mask(mean_image.shape, refined_sources[0].row, refined_sources[0].column, float(config.reference_fit_radius))
        score, beta, reference_residual = _solve_reference_image_model(
            mean_image, arr, mask, background_mode=config.reference_background
        )
        reference_fluxes = np.array([float(beta[0]) if beta is not None and len(beta) else np.nan])
        # Construct a full diagnostic model using the fitted source plus a simple
        # residual-derived background.  The exact cadence extraction still fits
        # its configured background independently.
        reference_model = mean_image - reference_residual
        scene_details.update({
            "mean_squared_residual": float(score),
            "sources": [{
                "scene_index": 0, "role": "target", "label": refined_sources[0].label,
                "source_id": refined_sources[0].source_id, "magnitude": refined_sources[0].magnitude,
                "row": refined_sources[0].row, "column": refined_sources[0].column,
                "reference_flux": float(reference_fluxes[0]),
                "captured_fraction": float(np.sum(arr)),
                "contribution_in_stamp": float(reference_fluxes[0] * np.sum(arr)),
                "contribution_relative_to_target": 1.0, "cadence_treatment": "free",
                "max_correlation_with_other_source": 0.0,
                "lightcurve_output_eligible": True,
                "lightcurve_output_reason": "",
            }],
            "output_eligible_source_indices": [0],
            "minimum_output_captured_fraction": float(config.all_source_min_captured_fraction),
            "scene_quality_pass": True, "scene_quality_reasons": [],
        })

    aux = load_tpf_auxiliary(
        len(time), tpf_obj=tpf_obj, tpf_path=tpf_path, cadence_indices=cadence_indices
    )
    if flux_err_cube is None:
        flux_err_cube = aux.get("flux_err")
    if flux_err_cube is not None and np.asarray(flux_err_cube).shape != flux.shape:
        flux_err_cube = None

    motion = choose_motion_solution(
        flux, refined_sources[0], config,
        pos_corr1=aux.get("pos_corr1"), pos_corr2=aux.get("pos_corr2"),
    )
    for reason in motion.details.get("fallback_reasons", []):
        print(f"  [PRF] Motion fallback: {reason}", flush=True)
    if motion.source in {"ensemble", "target"}:
        diag = motion.details.get("motion_diagnostics", {})
        print(
            f"  [PRF] Image motion source: {motion.source}; central 98% ranges "
            f"row={float(diag.get('row_robust_range_pix') or 0.0):.5g} pix, "
            f"column={float(diag.get('column_robust_range_pix') or 0.0):.5g} pix",
            flush=True,
        )
    if motion.source == "poscorr":
        print(f"  [PRF] POS_CORR calibration: {motion.details.get('transform', 'unknown')}", flush=True)
        residual = motion.details.get("image_calibration_residual_pix")
        if residual is not None:
            print(f"  [PRF] POS_CORR/image motion residual: {float(residual):.5f} pix", flush=True)

    scene = PRFSceneModel(refined_sources, backend, mean_image.shape, config)
    fixed_scene = multi_scene and str(config.neighbor_treatment).strip().lower() == "fixed"
    output_mode = str(config.source_output_mode).strip().lower()
    if output_mode not in {"primary", "all"}:
        raise ValueError(f"Unknown PRF source output mode: {config.source_output_mode!r}")

    nsrc = len(refined_sources)
    extracted_indices = [0]
    skipped_output_sources = []
    if fixed_scene and output_mode == "all":
        candidates = scene_details.get("output_eligible_source_indices", list(range(nsrc)))
        extracted_indices = sorted({0, *[int(i) for i in candidates if 0 <= int(i) < nsrc]})
        for row in scene_details.get("sources", []):
            j = int(row.get("scene_index", -1))
            if j > 0 and j not in extracted_indices:
                skipped_output_sources.append({
                    "scene_index": j,
                    "label": row.get("label", refined_sources[j].label if j < nsrc else "unknown"),
                    "reason": row.get("lightcurve_output_reason", "source did not pass output eligibility checks"),
                })
    elif not fixed_scene:
        # Single-source mode, and any future jointly-free mode, already has its
        # variable source set defined by the scene itself.
        extracted_indices = list(range(nsrc)) if not multi_scene else [0]

    source_background = np.full((len(time), nsrc), np.nan)
    source_residual_rms = np.full((len(time), nsrc), np.nan)
    source_mean_residual_images = np.full((nsrc, *mean_image.shape), np.nan)

    if fixed_scene:
        fixed_fluxes = np.asarray(reference_fluxes, float)
        source_flux = np.full((len(time), nsrc), np.nan)
        source_err = np.full((len(time), nsrc), np.nan)
        captured = None
        successful_indices = []
        for j in extracted_indices:
            try:
                sf, se, bg, cap, rrms, mres = scene.extract(
                    flux, motion, flux_err_cube=flux_err_cube,
                    free_source_indices=[j], fixed_source_fluxes=fixed_fluxes,
                )
            except Exception as exc:
                if j == 0:
                    raise
                skipped_output_sources.append({
                    "scene_index": int(j), "label": refined_sources[j].label,
                    "reason": f"cadence extraction failed: {type(exc).__name__}: {exc}",
                })
                print(
                    f"  [WARN] PRF scene source {j + 1} ({refined_sources[j].label}) "
                    f"was not written: {type(exc).__name__}: {exc}", flush=True,
                )
                continue
            source_flux[:, j] = sf[:, j]
            source_err[:, j] = se[:, j]
            source_background[:, j] = bg
            source_residual_rms[:, j] = rrms
            if mres is not None:
                source_mean_residual_images[j] = mres
            if captured is None:
                captured = cap
            successful_indices.append(int(j))
            if output_mode == "all" and j > 0:
                print(
                    f"  [PRF] Extracted scene source {j + 1}/{nsrc}: "
                    f"{refined_sources[j].label} with all other source fluxes fixed.",
                    flush=True,
                )
        extracted_indices = successful_indices
        if captured is None:
            raise RuntimeError("No PRF scene source light curve was successfully extracted.")
        background = source_background[:, 0]
        residual_rms = source_residual_rms[:, 0]
        mean_residual = source_mean_residual_images[0]
    else:
        free_indices = list(range(nsrc))
        source_flux, source_err, background, captured, residual_rms, mean_residual = scene.extract(
            flux, motion, flux_err_cube=flux_err_cube,
            free_source_indices=free_indices, fixed_source_fluxes=None,
        )
        for j in free_indices:
            source_background[:, j] = background
            source_residual_rms[:, j] = residual_rms
            if mean_residual is not None:
                source_mean_residual_images[j] = mean_residual
        extracted_indices = list(map(int, free_indices))

    # Mark the final per-source output decision in the scene table.
    extracted_set = set(map(int, extracted_indices))
    for row in scene_details.get("sources", []):
        j = int(row.get("scene_index", -1))
        row["lightcurve_extracted"] = j in extracted_set

    coupling = []
    for j in range(source_flux.shape[1]):
        if j in extracted_set:
            coupling.append(motion_coupling_diagnostics(
                time, source_flux[:, j], motion.row_shift, motion.column_shift,
                gap_days=float(config.gap_days_for_scaling),
                warn_correlation=float(config.motion_coupling_warn_correlation),
                warn_peak_to_peak=float(config.motion_coupling_warn_peak_to_peak),
            ))
        else:
            coupling.append({
                "n_valid": 0, "row_correlation": None, "column_correlation": None,
                "row_slope_relative_flux_per_pixel": None,
                "column_slope_relative_flux_per_pixel": None,
                "predicted_motion_peak_to_peak_relative_flux": None,
                "quality_pass": False,
                "quality_reasons": ["cadence light curve was not extracted"],
            })
    result_meta = {
        "mission": metadata.mission,
        "sector": metadata.sector, "camera": metadata.camera, "ccd": metadata.ccd,
        "tessmag": metadata.tessmag, "origin_row": metadata.origin_row,
        "origin_column": metadata.origin_column, "backend_messages": backend_messages,
        "motion_details": motion.details, "saturation_assessment": saturation,
        "reference_position_fits": reference_fits, "reference_scene": scene_details,
        "motion_coupling": coupling, "config": config.__dict__.copy(),
        "source_positions": [s.__dict__.copy() for s in refined_sources],
        "reference_source_fluxes": np.asarray(reference_fluxes, float).tolist(),
        "target_source_index": 0,
        "source_output_mode": output_mode,
        "extracted_source_indices": list(map(int, extracted_indices)),
        "skipped_output_sources": skipped_output_sources,
        "n_cadences": int(len(time)), "stamp_shape": list(mean_image.shape),
        "prf_grid": scene.grid_metadata(), "scene_mode": "gaia" if multi_scene else "single",
    }
    return PRFPhotometryResult(
        time=time, source_flux=source_flux, source_flux_err=source_err,
        background=background, row_shift=motion.row_shift, column_shift=motion.column_shift,
        captured_fraction=captured, residual_rms=residual_rms, sources=refined_sources,
        backend=backend.name, motion_source=motion.source, metadata=result_meta,
        reference_prfs=scene.reference_prfs(),
        reference_source_fluxes=np.asarray(reference_fluxes, float),
        reference_model_image=reference_model,
        contaminant_model_image=contaminant_model,
        mean_residual_image=mean_residual,
        source_background=source_background,
        source_residual_rms=source_residual_rms,
        source_mean_residual_images=source_mean_residual_images,
    )


def result_source_columns(
    result: PRFPhotometryResult,
    source_index: int = 0,
    gap_days: float | None = None,
) -> dict[str, np.ndarray | str]:
    i = int(source_index)
    raw = np.asarray(result.source_flux[:, i], float)
    err = np.asarray(result.source_flux_err[:, i], float)
    gap = result.metadata.get("config", {}).get("gap_days_for_scaling", 0.5) if gap_days is None else gap_days
    rel, norm = median_scale_by_segment(result.time, raw, gap_days=float(gap))
    err_rel = np.divide(err, norm, out=np.full_like(err, np.nan), where=np.isfinite(norm) & (norm != 0))
    coupling_all = result.metadata.get("motion_coupling", [])
    coupling = coupling_all[i] if i < len(coupling_all) else {}
    quality_pass = bool(coupling.get("quality_pass", True))
    quality_reason = "; ".join(coupling.get("quality_reasons", []))
    ref = result.reference_source_fluxes
    if ref is None:
        ref = np.nanmedian(result.source_flux, axis=0)
    ref = np.asarray(ref, float)
    captured = np.asarray(result.captured_fraction, float)
    target_reference_in_stamp = ref[i] * captured[:, i]
    neighbor_reference_in_stamp = np.zeros(len(result.time), float)
    if len(ref) > 1:
        other = [j for j in range(len(ref)) if j != i]
        neighbor_reference_in_stamp = np.nansum(captured[:, other] * ref[other][None, :], axis=1)
    denom = target_reference_in_stamp + neighbor_reference_in_stamp
    contamination = np.divide(
        neighbor_reference_in_stamp, denom,
        out=np.full_like(denom, np.nan), where=np.isfinite(denom) & (denom != 0),
    )
    scene_mode = str(result.metadata.get("scene_mode", "single"))
    treatment = str(result.metadata.get("config", {}).get("neighbor_treatment", "fixed"))
    output_mode = str(result.metadata.get("source_output_mode", "primary"))
    if result.source_background is not None and np.asarray(result.source_background).shape == result.source_flux.shape:
        source_background = np.asarray(result.source_background[:, i], float)
    else:
        source_background = np.asarray(result.background, float)
    if result.source_residual_rms is not None and np.asarray(result.source_residual_rms).shape == result.source_flux.shape:
        source_residual_rms = np.asarray(result.source_residual_rms[:, i], float)
    else:
        source_residual_rms = np.asarray(result.residual_rms, float)
    source = result.sources[i]
    return {
        "time_btjd": np.asarray(result.time, float),
        "flux_rel": rel, "flux_prf_rel": rel, "flux_prf_raw": raw,
        "flux_prf_err": err, "flux_prf_err_rel": err_rel,
        "background_prf": source_background,
        "motion_row": np.asarray(result.row_shift, float),
        "motion_column": np.asarray(result.column_shift, float),
        "prf_captured_fraction": captured[:, i],
        "prf_neighbor_model_flux_in_stamp": neighbor_reference_in_stamp,
        "prf_contamination_fraction": contamination,
        "prf_residual_rms": source_residual_rms,
        "prf_backend": np.full(len(result.time), result.backend, dtype=object),
        "prf_motion_source": np.full(len(result.time), result.motion_source, dtype=object),
        "prf_scene_mode": np.full(len(result.time), scene_mode, dtype=object),
        "prf_neighbor_treatment": np.full(len(result.time), treatment, dtype=object),
        "prf_source_output_mode": np.full(len(result.time), output_mode, dtype=object),
        "prf_source_index": np.full(len(result.time), i, dtype=int),
        "prf_source_label": np.full(len(result.time), source.label, dtype=object),
        "prf_source_id": np.full(len(result.time), source.source_id or "", dtype=object),
        "prf_source_magnitude": np.full(len(result.time), source.magnitude if source.magnitude is not None else np.nan),
        "prf_scene_source_count": np.full(len(result.time), len(result.sources), dtype=int),
        "prf_neighbor_count": np.full(len(result.time), max(0, len(result.sources) - 1), dtype=int),
        "prf_quality_pass": np.full(len(result.time), quality_pass, dtype=bool),
        "prf_quality_reason": np.full(len(result.time), quality_reason, dtype=object),
        "prf_flux_motion_row_correlation": np.full(len(result.time), coupling.get("row_correlation", np.nan)),
        "prf_flux_motion_column_correlation": np.full(len(result.time), coupling.get("column_correlation", np.nan)),
        "prf_motion_model_peak_to_peak": np.full(len(result.time), coupling.get("predicted_motion_peak_to_peak_relative_flux", np.nan)),
    }


def save_prf_diagnostic(
    result: PRFPhotometryResult,
    mean_image: np.ndarray,
    output_path: str | Path,
    source_index: int = 0,
    title: str = "Jitter-aware PRF photometry",
    save_figure_pickle: bool = False,
) -> None:
    """Write the six-panel PRF diagnostic and, optionally, its Figure pickle."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cols = result_source_columns(result, source_index=source_index)
    target = result.sources[source_index]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    image = np.asarray(mean_image, float)

    ax = axes[0, 0]
    ax.imshow(image, origin="lower", cmap="gray", interpolation="nearest")
    if result.reference_prfs is not None:
        prf = np.asarray(result.reference_prfs[source_index], float)
        finite = prf[np.isfinite(prf) & (prf > 0)]
        if finite.size:
            levels = np.nanmax(finite) * np.array([0.01, 0.05, 0.20, 0.50])
            try:
                ax.contour(prf, levels=np.unique(levels), origin="lower", linewidths=1.0)
            except Exception:
                pass
    for j, source in enumerate(result.sources):
        marker = "+" if j == source_index else "x"
        label = "target" if j == source_index else ("neighbors" if j == 1 else None)
        ax.plot(source.column, source.row, marker=marker, ms=10 if j == source_index else 6,
                mew=2 if j == source_index else 1, label=label)
    fits = result.metadata.get("reference_position_fits", [])
    if len(result.sources) == 1 and source_index < len(fits):
        fit_info = fits[source_index]
        initial_row = fit_info.get("input_wcs_or_seed_row", fit_info.get("initial_row"))
        initial_col = fit_info.get("input_wcs_or_seed_column", fit_info.get("initial_column"))
        if initial_row is not None and initial_col is not None:
            ax.plot(float(initial_col), float(initial_row), marker="s", ms=5, mfc="none", label="initial")
    if len(result.sources) > 1:
        ax.legend(loc="best", fontsize=8)
    ax.set_title(f"Observed reference image + scene\nbackend={result.backend}, sources={len(result.sources)}")
    ax.set_xlabel("Local column"); ax.set_ylabel("Local row")

    ax = axes[0, 1]
    if result.reference_model_image is not None and np.isfinite(result.reference_model_image).any():
        im = ax.imshow(result.reference_model_image, origin="lower", cmap="gray", interpolation="nearest")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title("Joint reference-scene model")
    else:
        ax.text(0.5, 0.5, "Reference model unavailable", ha="center", va="center", transform=ax.transAxes)
    ax.set_xlabel("Local column"); ax.set_ylabel("Local row")

    ax = axes[0, 2]
    residual = None
    scene_ref = result.metadata.get("reference_scene", {})
    if result.reference_model_image is not None:
        residual = image - np.asarray(result.reference_model_image, float)
    elif result.mean_residual_image is not None:
        residual = result.mean_residual_image
    if residual is not None and np.isfinite(residual).any():
        im = ax.imshow(residual, origin="lower", cmap="gray", interpolation="nearest")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(
        "Reference residual\n"
        f"condition={scene_ref.get('condition_number', np.nan):.3g}, "
        f"max corr={scene_ref.get('max_source_correlation', np.nan):.4f}"
    )
    ax.set_xlabel("Local column"); ax.set_ylabel("Local row")

    ax = axes[1, 0]
    ax.plot(result.time, result.column_shift, ".", ms=1.3, label="column")
    ax.plot(result.time, result.row_shift, ".", ms=1.3, label="row")
    ax.set_title(f"Image motion: {result.motion_source}")
    ax.set_xlabel("Time [BTJD]"); ax.set_ylabel("Shift [pixel]"); ax.legend(loc="best")

    ax = axes[1, 1]
    ax.plot(result.time, cols["flux_prf_rel"], ".", ms=1.3)
    coupling_all = result.metadata.get("motion_coupling", [])
    coupling = coupling_all[source_index] if source_index < len(coupling_all) else {}
    status = "PASS" if coupling.get("quality_pass", True) and scene_ref.get("scene_quality_pass", True) else "FLAGGED"
    def _fmt(value, scale=1.0):
        try:
            value = float(value) * scale
            return f"{value:.3f}" if np.isfinite(value) else "nan"
        except Exception:
            return "nan"
    ax.set_title(
        f"Source PRF light curve — {status}\n"
        f"corr(row,col)=({_fmt(coupling.get('row_correlation'))}, {_fmt(coupling.get('column_correlation'))}); "
        f"contam={_fmt(np.nanmedian(cols['prf_contamination_fraction']), 100)}%"
    )
    ax.set_xlabel("Time [BTJD]"); ax.set_ylabel("Relative flux")

    ax = axes[1, 2]
    contaminant = None
    if result.reference_prfs is not None and result.reference_source_fluxes is not None:
        prfs = np.asarray(result.reference_prfs, float)
        refs = np.asarray(result.reference_source_fluxes, float)
        if prfs.ndim == 3 and len(refs) == prfs.shape[0]:
            other = [j for j in range(len(refs)) if j != source_index]
            contaminant = np.nansum(refs[other, None, None] * prfs[other], axis=0) if other else np.zeros(prfs.shape[1:], float)
    if contaminant is None:
        contaminant = result.contaminant_model_image
    if contaminant is not None and np.isfinite(contaminant).any() and np.nanmax(np.abs(contaminant)) > 0:
        im = ax.imshow(contaminant, origin="lower", cmap="gray", interpolation="nearest")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title("Other-source reference model")
        ax.set_xlabel("Local column"); ax.set_ylabel("Local row")
    else:
        source_rows = scene_ref.get("sources", [])
        lines = [f"Scene mode: {result.metadata.get('scene_mode', 'single')}",
                 f"Sources: {len(result.sources)}", f"Neighbor treatment: {result.metadata.get('config', {}).get('neighbor_treatment', 'fixed')}"]
        for row in source_rows[:8]:
            lines.append(f"{row.get('role','?')}: {row.get('label','?')}  rel={row.get('contribution_relative_to_target', np.nan):.3g}")
        ax.axis("off")
        ax.text(0.02, 0.98, "\n".join(lines), va="top", family="monospace", transform=ax.transAxes)

    fig.suptitle(title)
    fig.tight_layout()
    output_path = Path(output_path)
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    if save_figure_pickle:
        try:
            pickle_path = output_path.with_suffix(output_path.suffix + ".pickle")
            with pickle_path.open("wb") as handle:
                pickle.dump(fig, handle, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as exc:
            # A serialization problem must not discard the scientifically
            # useful PNG or abort an otherwise successful extraction run.
            print(
                f"  [WARN] Could not save pickled figure for "
                f"{output_path.name}: {exc}"
            )
    plt.close(fig)


def save_prf_products(
    result: PRFPhotometryResult,
    mean_image: np.ndarray,
    output_dir: str | Path,
    output_stem: str,
    *,
    source_index: int = 0,
    extra_columns: dict | None = None,
    extra_metadata: dict | None = None,
    save_diagnostics: bool | None = None,
    save_figure_pickle: bool = False,
) -> dict[str, Path]:
    """Write a downstream-compatible target light curve and scene diagnostics."""
    import pandas as pd

    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    columns = result_source_columns(result, source_index=source_index)
    if extra_columns:
        for key, value in extra_columns.items():
            if np.isscalar(value) or isinstance(value, str):
                columns[key] = np.full(len(result.time), value, dtype=object)
            else:
                arr = np.asarray(value)
                if len(arr) != len(result.time):
                    raise ValueError(f"Extra column {key!r} has length {len(arr)}, expected {len(result.time)}.")
                columns[key] = arr
    csv_path = outdir / f"{output_stem}.csv"
    pd.DataFrame(columns).to_csv(csv_path, index=False)

    metadata = dict(result.metadata)
    metadata.update({
        "backend": result.backend, "motion_source": result.motion_source,
        "source_index": int(source_index), "source": result.sources[source_index].__dict__.copy(),
    })
    if extra_metadata:
        metadata.update(extra_metadata)
    json_path = outdir / f"{output_stem}_prf_meta.json"
    json_path.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")

    scene_rows = result.metadata.get("reference_scene", {}).get("sources", [])
    scene_path = outdir / f"{output_stem}_prf_scene.csv"
    if scene_rows:
        pd.DataFrame(scene_rows).to_csv(scene_path, index=False)
    else:
        pd.DataFrame([s.__dict__ for s in result.sources]).to_csv(scene_path, index=False)

    motion_path = outdir / f"{output_stem}_prf_motion.npz"
    reference_fluxes = result.reference_source_fluxes
    if reference_fluxes is None:
        reference_fluxes = np.nanmedian(np.asarray(result.source_flux, float), axis=0)
    np.savez_compressed(
        motion_path, time=np.asarray(result.time, float),
        row_shift=np.asarray(result.row_shift, float),
        column_shift=np.asarray(result.column_shift, float),
        captured_fraction=np.asarray(result.captured_fraction, float),
        residual_rms=np.asarray(result.residual_rms, float),
        source_flux=np.asarray(result.source_flux, float),
        source_flux_err=np.asarray(result.source_flux_err, float),
        source_background=(np.asarray(result.source_background, float) if result.source_background is not None else np.asarray(result.background, float)[:, None]),
        source_residual_rms=(np.asarray(result.source_residual_rms, float) if result.source_residual_rms is not None else np.asarray(result.residual_rms, float)[:, None]),
        extracted_source_indices=np.asarray(result.metadata.get("extracted_source_indices", [source_index]), int),
        reference_source_fluxes=np.asarray(reference_fluxes, float),
        source_rows=np.asarray([s.row for s in result.sources], float),
        source_columns=np.asarray([s.column for s in result.sources], float),
    )

    paths = {"csv": csv_path, "metadata": json_path, "motion": motion_path, "scene": scene_path}
    if save_diagnostics is None:
        save_diagnostics = bool(result.metadata.get("config", {}).get("save_diagnostics", True))
    if save_diagnostics:
        png_path = outdir / f"{output_stem}_prf_diagnostic.png"
        save_prf_diagnostic(
            result, mean_image, png_path, source_index=source_index,
            title=f"{result.sources[source_index].label}: jitter-aware PRF photometry",
            save_figure_pickle=save_figure_pickle,
        )
        paths["diagnostic"] = png_path
        if save_figure_pickle:
            pickle_path = png_path.with_suffix(png_path.suffix + ".pickle")
            if pickle_path.exists():
                paths["diagnostic_pickle"] = pickle_path
    return paths


__all__ = [
    "SceneSource",
    "PRFPhotometryConfig",
    "TPFMetadata",
    "MotionSolution",
    "PRFPhotometryResult",
    "PRFSceneModel",
    "fit_reference_scene_with_prf",
    "extract_jitter_aware_prf",
    "save_prf_products",
    "save_prf_diagnostic",
    "assess_saturation",
    "representative_image",
    "refine_reference_position",
    "fit_reference_position_with_prf",
    "motion_coupling_diagnostics",
]

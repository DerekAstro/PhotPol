
#!/usr/bin/env python3
"""Guided TESS-photometry and polarimetry frequency analysis.

The module supports a TESS-only discovery pass followed by local searches in
one or more polarimetry channels.  It can be imported and configured by the
GUI, or executed directly with the configuration constants below.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from pathlib import Path
import json
import time
import matplotlib.pyplot as plt
import re
import importlib.util
import sys

try:
    from IPython.display import display
except Exception:
    display = None

try:
    from scipy.signal import lombscargle as _scipy_lomb
except Exception:
    _scipy_lomb = None

try:
    from scipy.optimize import least_squares, curve_fit
    _HAVE_LSQ = True
except Exception:
    least_squares = None
    curve_fit = None
    _HAVE_LSQ = False

try:
    from scipy.interpolate import PchipInterpolator
except Exception:
    PchipInterpolator = None

try:
    from astropy.timeseries import LombScargle
    _HAVE_ASTROPY = True
except Exception:
    LombScargle = None
    _HAVE_ASTROPY = False

try:
    from polarimetry_quadrature import (
        fit_quadrature_multisin,
        normalize_quadrature_mode,
        phase_at_reference,
        wrap_phase_radians,
    )
except ModuleNotFoundError:
    # GUI runners import this file by absolute path, which does not always put
    # the backend directory on sys.path.  Load the colocated shared fitter
    # explicitly so direct, GUI, and notebook execution behave identically.
    _quadrature_path = Path(__file__).resolve().with_name("polarimetry_quadrature.py")
    _quadrature_spec = importlib.util.spec_from_file_location(
        "photpol_polarimetry_quadrature", _quadrature_path
    )
    if _quadrature_spec is None or _quadrature_spec.loader is None:
        raise
    _quadrature_module = importlib.util.module_from_spec(_quadrature_spec)
    sys.modules[_quadrature_spec.name] = _quadrature_module
    _quadrature_spec.loader.exec_module(_quadrature_module)
    fit_quadrature_multisin = _quadrature_module.fit_quadrature_multisin
    normalize_quadrature_mode = _quadrature_module.normalize_quadrature_mode
    phase_at_reference = _quadrature_module.phase_at_reference
    wrap_phase_radians = _quadrature_module.wrap_phase_radians

print("Astropy:", _HAVE_ASTROPY, "| SciPy least_squares:", _HAVE_LSQ)

# ----------------------------------------------------------------------
# User configuration
# ----------------------------------------------------------------------

# --- File paths / input mode ---
TESS_INPUT_MODE = "spoc_csv"              # "spoc_csv" | "pipeline_dir"
TESS_CSV = Path("combined_filtered.csv")  # used if TESS_INPUT_MODE == "spoc_csv"

# If TESS_INPUT_MODE == "pipeline_dir", point this at a directory containing
# CSV outputs from your Voronoi/raw or detrend pipelines.
TESS_PIPELINE_DIR = Path("tess_pipeline_lcs")
TESS_PIPELINE_PATTERN = "*.csv"
TESS_PIPELINE_RECURSIVE = False
TESS_PIPELINE_FLUX = "raw"                # "raw" | "detrended" | "auto"

POL_CSV = Path("target_IND.csv")   # raw/basic polarimetry CSV OR precomputed *_analysis_frame.csv

# --- Polarimetry product to analyze ---
POL_PRODUCT = "resid_nm_pchip"  # 'nm' | 'resid_nm_pchip' | 'pw_resid_nm_pchip'
USE_POLARIMETRY = True

# --- Raw-polarimetry preprocessing (used automatically if POL_CSV is raw/basic) ---
POL_SAVE_GENERATED_ANALYSIS_FRAME = True
POL_GENERATED_ANALYSIS_DIR = None  # None => write next to POL_CSV
POL_NIGHT_GAP_THRESHOLD_DAYS = 0.4
POL_PCHIP_BIN_WIDTH = 0.05
POL_PCHIP_MIN_POINTS_PER_BIN = 1
POL_PREWHITEN_ITERS = 3
POL_PREWHITEN_HARMONICS = 5
POL_PREWHITEN_FREQ_WINDOW_FRAC = 0.05
POL_PREWHITEN_GRID_MODE = "longest_chunk"   # "longest_chunk" | "global_baseline"
POL_PREWHITEN_LS_OVERSAMPLE = 4
POL_PREWHITEN_STOP_RULE = True
POL_PREWHITEN_MIN_ITERS = 1
POL_PREWHITEN_SNR_LS_MIN = 3.0
POL_PREWHITEN_MAX_FMAX_CPD = 20.0
POL_PREWHITEN_MIN_SAMPLES = 1000
POL_PREWHITEN_MAX_SAMPLES = 100_000

# --- Frequency range (cycles/day) ---
FMIN = 0.01
FMAX = 50.0
TESS_CAP_FMAX_TO_NYQUIST = True
TESS_WEIGHT_MODE = "sector_rescaled"  # "none" | "formal" | "sector_rescaled"
TESS_ERROR_FLOOR_FRAC = 0.25
TESS_SECTOR_RESCALE_MIN_POINTS = 20
TESS_SECTOR_SCALE_MIN = 0.10
TESS_SECTOR_SCALE_MAX = 100.0

# --- TESS-only extraction ---
TESS_OVERSAMPLE = 1.0
TESS_GRID_MODE = "full_baseline"   # "full_baseline" | "longest_chunk"
MAX_TESS_MODES = 20
TESS_SNR_STOP = 4.0
KS_TESS = 15.0
TRIM_TOP_FRAC = 0.10
KFIT = 3.0
LOCAL_FIT_STEPS = 1001
TESS_MAX_GRID_POINTS = 60000
PLOT_MAX_GRID_POINTS = 6000
POL_LOCAL_MAX_POINTS = 1201
POL_FULL_RANGE_PLOT_POINTS = 2400
POL_FULL_RANGE_PLOT_OVERSAMPLE = 1.0

# Smoothing changes only periodogram-based candidate selection.  All final
# frequency, amplitude, and phase fits continue to use the unsmoothed
# time-series data.  The width is tied to the independent polarimetry
# resolution 1/T, not to an arbitrary oversampled grid.
POL_SMOOTH_ENABLED = False
POL_SMOOTH_KERNEL = "gaussian"       # "gaussian" | "boxcar"
POL_SMOOTH_WIDTH_RES_ELEMS = 10.0    # Gaussian FWHM or boxcar full width

# Optional visual comparison.  The background spectrum never enters candidate
# selection, fitting, SNR calculation, or prewhitening.
SHOW_TESS_SPECTRUM_BACKGROUND = False
TESS_SPECTRUM_BACKGROUND_SCALE = "right_axis"  # "right_axis" | "normalized"

# Optional physically constrained Stokes q/u fit.  ``auto`` tests both
# quadrature signs separately for every accepted frequency.
QU_PHASE_MODE = "free"  # "free" | "auto" | "force_plus" | "force_minus"
QUADRATURE_MAX_NFEV = 300

# --- Polarimetry guided extraction ---
POL_CHANNELS = ["q", "u", "p"]  # p is analyzed directly from the observed CSV column
POL_SNR_STOP = 2.0
POL_SNR_SOFT_MIN = 1.5      # admit marginal guided polarimetric modes into the global fit if they are close to the TESS template
POL_POSTFIT_SNR_MIN = 1.5   # after the global fit, keep marginal modes only if their leave-one-out local SNR clears this value
POL_SEED_MAX_OFFSET_MULT = 2.0  # require marginal seeds to lie within this many resolution elements of the TESS template
POL_OVERSAMPLE = 2.0
POL_SEARCH_WINDOW_MULT = 10.0  # half-window in units of max(1/T_tess, 1/T_pol)
POL_LOCAL_NOISE_KS = 3.0      # sideband half-width for guided local SNR, in units of 1/T
POL_LOCAL_NOISE_SIDE_BINS = 6 # minimum sideband width for guided local SNR, in local-grid bins
MAX_POL_MODES = 99            # hard cap; actual search count is limited by TESS mode list
GUIDED_POL_FMIN = 0.2       # minimum TESS/template frequency [c/d] to test in guided q/u/p matching

# --- Phase-plot summaries ---
PHASE_BIN_N = 16             # number of phase bins for overplotted median points in phased plots
PHASE_ZERO_MODE = "local_start"  # "local_start" | "btjd_zero" | "custom_btjd"
PHASE_ZERO_BTJD = 0.0

# --- Final global multisinusoid fit ---
GLOBAL_FREQ_BOUND_MULT = 3.0  # each frequency may move by +/- this x frequency resolution
GLOBAL_MAX_NFEV = 300
MIN_FREQ_SEP_MULT = 5.0        # minimum separation for distinct fitted modes, in units of 1/T_full
TESS_DISCOVERY_EXCLUSION_MULT = 5.0  # block re-discovery within this many resolution elements during sequential TESS search

# --- Night-aware polarimetry baseline model ---
USE_POL_NIGHT_OFFSETS = True
USE_POL_NIGHT_SLOPES = False
POL_NIGHT_GROUP_MODE = "gap"     # 'gap' | 'integer_jd' | 'run' | 'subrun'
POL_NIGHT_GAP_HOURS = 8.0

# --- Optional broad detrending hook (defaults off) ---
DO_DETREND = False
DETREND_POLY_ORDER = 0

# --- Phased plots ---
N_PHASE_PLOTS = 3
PHASE_SORT_BY = "amp"            # "amp" | "snr" | "mode"

# --- Interactive/script display ---
SHOW_PLOTS_INLINE = True
VERBOSE = 1
LSQ_VERBOSE = 0

# --- Output ---
OUTROOT = Path("guided_phot_pol_outputs")
OUTPUT_TARGET_SUBDIR = False


POL_FILENAME_SUFFIXES = [
    "_IND_nm_pchip_analysis_frame",
    "_nm_pchip_analysis_frame",
    "_analysis_frame",
    "_IND",
]

def prefixed_output_name(prefix: str, basename: str) -> str:
    prefix = str(prefix).strip()
    return f"{prefix}_{basename}" if prefix else basename

def infer_star_labels_from_pol_path(path: Path | str) -> tuple[str, str]:
    stem = Path(path).stem
    for suff in sorted(POL_FILENAME_SUFFIXES, key=len, reverse=True):
        if stem.endswith(suff):
            stem = stem[:-len(suff)]
            break
    stem = stem.strip(" _-")
    display = re.sub(r"[_]+", " ", stem).strip()
    display = re.sub(r"\s+", " ", display)
    safe = re.sub(r"[^A-Za-z0-9.+-]+", "_", display).strip("_")
    if not display:
        display = "target"
    if not safe:
        safe = "target"
    return display, safe

def infer_star_labels_from_tess_input() -> tuple[str, str]:
    if str(TESS_INPUT_MODE).lower() == "pipeline_dir":
        patt = str(TESS_PIPELINE_PATTERN).strip()
        if patt and ("*" not in patt) and ("?" not in patt) and ("[" not in patt):
            stem = Path(patt).stem
        else:
            stem = Path(TESS_PIPELINE_DIR).name
    else:
        stem = Path(TESS_CSV).stem
    for suff in ["_combined_filtered", "_combined", "_lc", "_lightcurve"]:
        if stem.endswith(suff):
            stem = stem[:-len(suff)]
            break
    stem = stem.strip(" _-")
    display = re.sub(r"[_]+", " ", stem).strip()
    display = re.sub(r"\s+", " ", display)
    safe = re.sub(r"[^A-Za-z0-9.+-]+", "_", display).strip("_")
    if not display:
        display = "target"
    if not safe:
        safe = "target"
    return display, safe

def resolve_analysis_outroot(star_safe: str) -> Path:
    base = OUTROOT / star_safe if OUTPUT_TARGET_SUBDIR else OUTROOT
    base.mkdir(parents=True, exist_ok=True)
    return base

def compute_tess_nyquist_cpd(tess) -> float:
    try:
        t = np.asarray(tess.t, float)
    except Exception:
        return np.nan
    t = t[np.isfinite(t)]
    if t.size < 3:
        return np.nan
    dt = np.diff(np.sort(t))
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if dt.size < 1:
        return np.nan
    med_dt = np.nanmedian(dt)
    if not np.isfinite(med_dt) or med_dt <= 0:
        return np.nan
    return 0.5 / med_dt


# ----------------------------------------------------------------------
# Data structures
# ----------------------------------------------------------------------

@dataclass
class TimeSeries:
    t: np.ndarray
    y: np.ndarray
    yerr: np.ndarray | None
    name: str
    t_abs: np.ndarray | None = None
    group_id: np.ndarray | None = None

@dataclass
class NightTrendConfig:
    use_offsets: bool = True
    use_slopes: bool = False
    group_mode: str = "gap"
    gap_days: float = 8.0 / 24.0

# ----------------------------------------------------------------------
# I/O helpers
# ----------------------------------------------------------------------

TESS_TIME_COL_CANDIDATES = ["time_btjd", "btjd", "time", "bjd", "jd"]
TESS_FORCE_Y_COL = None

# Ordered from the most final/science-ready product to progressively earlier
# pipeline stages. This covers the standard detrender, quaternion branch,
# orbital-template branch, watershed/saturation-optimized extractors, simple
# extractor, and
# SPOC converter outputs.
TESS_Y_COL_CANDIDATES = [
    "flux_detrend_rel",
    "flux_orbital_corrected_rel",
    "flux_quaternion_corrected_rel",
    "flux_decor_only_rel",
    "flux_quaternion_only_corrected_rel",
    # Explicit science stream written by the updated SPOC converter. Keep this
    # ahead of the legacy SAP/PDCSAP compatibility columns so a user-selected
    # SAP stream is not silently replaced by PDCSAP during analysis.
    "flux_selected_rel",
    "flux_detrended_sub",
    "flux_detrended_div",
    "flux_detrended_rel",
    "flux_medscaled",
    "pdcsap_flux_rel",
    "sap_flux_rel",
    "flux_rel",
    "relative_flux",
    "normalized_flux",
    "flux",
]
TESS_ERROR_COL_CANDIDATES_BY_FLUX = {
    "flux_selected_rel": ["flux_selected_err_rel"],
    "pdcsap_flux_rel": ["pdcsap_flux_err_rel", "flux_selected_err_rel"],
    "sap_flux_rel": ["sap_flux_err_rel", "flux_selected_err_rel"],
    "flux_detrended_rel": ["pdcsap_flux_err_rel", "flux_selected_err_rel", "flux_err_rel"],
    "flux_medscaled": ["flux_selected_err_rel", "pdcsap_flux_err_rel", "sap_flux_err_rel", "flux_err_rel"],
    "flux_rel": ["sap_flux_err_rel", "flux_selected_err_rel", "flux_err_rel"],
    "relative_flux": ["relative_flux_err", "flux_err_rel", "flux_error_rel"],
    "normalized_flux": ["normalized_flux_err", "flux_err_rel", "flux_error_rel"],
    "flux_detrend_rel": ["flux_detrend_err_rel", "flux_err_rel", "flux_error_rel"],
    "flux_orbital_corrected_rel": ["flux_orbital_corrected_err_rel", "flux_err_rel", "flux_error_rel"],
    "flux_quaternion_corrected_rel": ["flux_quaternion_corrected_err_rel", "flux_err_rel", "flux_error_rel"],
    "flux_decor_only_rel": ["flux_decor_only_err_rel", "flux_err_rel", "flux_error_rel"],
    "flux_quaternion_only_corrected_rel": ["flux_quaternion_only_corrected_err_rel", "flux_err_rel", "flux_error_rel"],
    "flux_detrended_sub": ["flux_detrended_sub_err", "flux_err", "flux_error"],
    "flux_detrended_div": ["flux_detrended_div_err_rel", "flux_err_rel", "flux_error_rel"],
}
TESS_GENERIC_ERROR_COL_CANDIDATES = [
    "flux_selected_err_rel", "flux_err_rel", "flux_error_rel",
    "relative_flux_err", "normalized_flux_err",
]
TESS_PIPELINE_RAW_Y_COLS = [
    "flux_rel",
    "flux_medscaled",
    "flux_detrended_rel",  # saturation-optimized extractor uses this name
    "relative_flux",
    "normalized_flux",
    "flux",
]
TESS_PIPELINE_DETRENDED_Y_COLS = [
    "flux_detrend_rel",
    "flux_orbital_corrected_rel",
    "flux_quaternion_corrected_rel",
    "flux_decor_only_rel",
    "flux_quaternion_only_corrected_rel",
    "flux_detrended_sub",
    "flux_detrended_div",
    "flux_detrended_rel",
    "flux_medscaled",
    "flux_rel",
]

POL_TIME_COL = "jd"
POL_ERR_COLS = {"q": "q_err", "u": "u_err", "p": "p_err"}

POL_PRODUCTS = {
    "nm":               {"q": "q_nm",               "u": "u_nm",               "p": "p_nm"},
    "resid_nm_pchip":   {"q": "q_resid_nm_pchip",   "u": "u_resid_nm_pchip",   "p": "p_resid_nm_pchip"},
    "pw_resid_nm_pchip":{"q": "q_pw_resid_nm_pchip","u": "u_pw_resid_nm_pchip","p": "p_pw_resid_nm_pchip"},
}


RAW_POL_CANDIDATES = {
    "jd": ["jd", "JD"],
    "q": ["q"],
    "u": ["u"],
    "p": ["p"],
    "q_err": ["q_err", "qe"],
    "u_err": ["u_err", "ue"],
    "p_err": ["p_err", "pe"],
}

# Common column names used for observing-run or subrun labels.  A label such
# as 12A is retained as subrun_label and reduced to 12 for run_label.
RAW_POL_RUN_CODE_CANDIDATES = [
    "run", "Run", "RUN", "run_id", "RunID", "runid",
    "run_code", "RunCode", "RUN_CODE", "run_label", "RunLabel",
    "subrun", "Subrun", "subrun_label", "SubrunLabel",
]

def _first_present(df: pd.DataFrame, names: list[str]) -> str | None:
    for n in names:
        if n in df.columns:
            return n
    return None

def _normalize_run_token(value) -> str | None:
    if pd.isna(value):
        return None
    s = str(value).strip()
    if not s:
        return None
    return s

def _base_run_token(token: str | None) -> str | None:
    if token is None:
        return None
    tok = str(token).strip()
    if not tok:
        return None
    if tok[-1].isalpha():
        return tok[:-1] or tok
    return tok

def normalize_raw_polarimetry_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize core raw-polarimetry columns and preserve run/subrun labels."""
    cols = {}
    for key, names in RAW_POL_CANDIDATES.items():
        col = _first_present(df, names)
        if col is None:
            raise ValueError(f"Raw polarimetry input is missing a column for {key!r}. Looked for {names}. Found: {list(df.columns)}")
        cols[key] = col
    out = pd.DataFrame({k: pd.to_numeric(df[v], errors="coerce") for k, v in cols.items()})

    run_col = _first_present(df, RAW_POL_RUN_CODE_CANDIDATES)
    if run_col is not None:
        subrun = df[run_col].map(_normalize_run_token)
        run = subrun.map(_base_run_token)
        out["subrun_label"] = subrun
        out["run_label"] = run
    return out

def pol_assign_night_ids(df: pd.DataFrame, gap_days: float = POL_NIGHT_GAP_THRESHOLD_DAYS) -> pd.DataFrame:
    df = df.sort_values("jd").reset_index(drop=True)
    jd = df["jd"].to_numpy(float)
    night_ids = np.zeros(len(df), dtype=int)
    if len(jd) > 1:
        jd_diff = np.diff(jd)
        for i in range(1, len(df)):
            if np.isfinite(jd_diff[i-1]) and (jd_diff[i-1] > gap_days):
                night_ids[i:] += 1
    df["night_id"] = night_ids
    return df

def _factorize_group_labels(labels: np.ndarray) -> np.ndarray:
    ser = pd.Series(labels, dtype="object")
    ser = ser.fillna("__MISSING_GROUP__")
    codes, _ = pd.factorize(ser, sort=False)
    return codes.astype(int)

def get_polarimetry_group_labels(df: pd.DataFrame, mode: str) -> np.ndarray:
    """Return label values used by the run/subrun baseline-group modes."""
    if mode == "run":
        if "run_label" in df.columns:
            vals = df["run_label"].map(_normalize_run_token).to_numpy(object)
        elif "subrun_label" in df.columns:
            vals = df["subrun_label"].map(_base_run_token).to_numpy(object)
        else:
            # Also accept the original run-like columns in precomputed frames.
            run_col = _first_present(df, RAW_POL_RUN_CODE_CANDIDATES)
            if run_col is None:
                raise ValueError(
                    "POL_NIGHT_GROUP_MODE='run' requires a run-like column in the polarimetry CSV. "
                    "Expected one of run/run_id/run_code/run_label/subrun/subrun_label, "
                    "or a generated analysis frame that preserves run_label."
                )
            vals = df[run_col].map(_normalize_run_token).map(_base_run_token).to_numpy(object)
        return vals

    if mode == "subrun":
        if "subrun_label" in df.columns:
            vals = df["subrun_label"].map(_normalize_run_token).to_numpy(object)
        elif "run_label" in df.columns:
            vals = df["run_label"].map(_normalize_run_token).to_numpy(object)
        else:
            run_col = _first_present(df, RAW_POL_RUN_CODE_CANDIDATES)
            if run_col is None:
                raise ValueError(
                    "POL_NIGHT_GROUP_MODE='subrun' requires a run/subrun-like column in the polarimetry CSV. "
                    "Expected one of run/run_id/run_code/run_label/subrun/subrun_label, "
                    "or a generated analysis frame that preserves subrun_label."
                )
            vals = df[run_col].map(_normalize_run_token).to_numpy(object)
        return vals

    raise ValueError(f"Unsupported label-based group mode: {mode}")

def pol_compute_phase(df: pd.DataFrame) -> pd.DataFrame:
    jd0 = np.floor(df["jd"].min()) + 0.5
    df["phase1d"] = (df["jd"] - jd0) % 1.0
    return df

def pol_phase_bin_stats(phase: np.ndarray, values: np.ndarray,
                        width: float = POL_PCHIP_BIN_WIDTH,
                        min_count: int = POL_PCHIP_MIN_POINTS_PER_BIN):
    phase = np.mod(phase, 1.0)
    bin_edges = np.arange(0.0, 1.0 + 1e-12, width)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    bin_idx = np.floor(phase / width).astype(int)
    means = np.full(len(bin_centers), np.nan)
    counts = np.zeros(len(bin_centers), dtype=int)
    for k in range(len(bin_centers)):
        m = (bin_idx == k) & np.isfinite(values)
        if m.sum() >= min_count:
            means[k] = np.nanmean(values[m])
            counts[k] = int(m.sum())
    good = np.isfinite(means)
    return bin_centers[good], means[good], counts[good]

def pol_build_pchip_from_binned(phase: np.ndarray, y: np.ndarray):
    if PchipInterpolator is None:
        raise ImportError("scipy.interpolate.PchipInterpolator is required for raw-polarimetry preprocessing.")
    m = np.isfinite(phase) & np.isfinite(y)
    phase, y = np.mod(phase[m], 1.0), y[m]
    bx, by, _ = pol_phase_bin_stats(phase, y)
    if len(bx) < 3:
        return None, None, None
    x_fit = np.concatenate([bx - 1.0, bx, bx + 1.0])
    y_fit = np.concatenate([by, by, by])
    return PchipInterpolator(x_fit, y_fit), bx, by

def pol_center_time(t: np.ndarray):
    t = np.asarray(t, float)
    t0 = np.nanmedian(t)
    return t - t0, t0

def pol_contiguous_chunks(t: np.ndarray, gap_days: float = POL_NIGHT_GAP_THRESHOLD_DAYS):
    t = np.asarray(t, float)
    idx = np.argsort(t)
    t_sorted = t[idx]
    if t_sorted.size == 0:
        return []
    breaks = np.where(np.diff(t_sorted) > gap_days)[0]
    starts = np.r_[0, breaks + 1]
    ends = np.r_[breaks, len(t_sorted) - 1]
    return [(int(s), int(e), t_sorted[s], t_sorted[e]) for s, e in zip(starts, ends)]

def pol_longest_chunk_duration_days(t: np.ndarray, gap_days: float = POL_NIGHT_GAP_THRESHOLD_DAYS):
    ch = pol_contiguous_chunks(t, gap_days=gap_days)
    if not ch:
        return 0.0
    return max(e_t - s_t for _, _, s_t, e_t in ch)

def pol_suggest_frequency_grid(t: np.ndarray,
                               mode: str = POL_PREWHITEN_GRID_MODE,
                               oversample: int = POL_PREWHITEN_LS_OVERSAMPLE):
    t = np.asarray(t, float)
    t = t[np.isfinite(t)]
    if t.size < 3:
        return None
    baseline = t.max() - t.min()
    dt = np.diff(np.sort(t))
    med_dt = np.nanmedian(dt) if dt.size else 1.0
    f_nyq = 0.5 / med_dt if med_dt > 0 else POL_PREWHITEN_MAX_FMAX_CPD
    f_max = min(POL_PREWHITEN_MAX_FMAX_CPD, max(5.0 / max(baseline, 1e-6), f_nyq))
    if mode == "longest_chunk":
        Tchunk = max(pol_longest_chunk_duration_days(t), 1e-6)
        df = (1.0 / Tchunk) / max(1, oversample)
        f_min = 1.0 / (10.0 * baseline) if baseline > 0 else df
        freqs = np.arange(max(f_min, df), f_max, df)
    else:
        f_min = 1.0 / (10.0 * baseline) if baseline > 0 else 1e-3
        n_samples = int(5 * max(500, np.ceil((f_max - f_min) * baseline)))
        n_samples = int(np.clip(n_samples, POL_PREWHITEN_MIN_SAMPLES, POL_PREWHITEN_MAX_SAMPLES))
        freqs = np.linspace(f_min, f_max, n_samples)
    if freqs.size < POL_PREWHITEN_MIN_SAMPLES and mode == "longest_chunk":
        freqs = np.linspace(max(1e-6, freqs.min() if freqs.size else 1e-6), f_max, POL_PREWHITEN_MIN_SAMPLES)
    return freqs

def pol_ls_power_multiharmonic(t_shifted: np.ndarray, y_det: np.ndarray, freqs_cpd: np.ndarray,
                               yerr: np.ndarray | None = None, nterms: int = POL_PREWHITEN_HARMONICS):
    if _HAVE_ASTROPY:
        ls = LombScargle(t_shifted, y_det, dy=yerr, nterms=max(1, int(nterms)))
        P = ls.power(freqs_cpd)
        return P, ls
    if _scipy_lomb is None:
        raise ImportError("Need astropy or scipy.signal.lombscargle for prewhitening.")
    w = 2 * np.pi * freqs_cpd
    y0 = y_det - np.nanmean(y_det)
    P = _scipy_lomb(t_shifted, y0, w, precenter=False, normalize=True)
    return P, None

def pol_local_ls_snr(freqs: np.ndarray, power: np.ndarray, idx_pk: int,
                     window_frac: float = 0.02, guard_bins: int = 5):
    fpk = freqs[idx_pk]
    f_span = freqs[-1] - freqs[0]
    if f_span <= 0:
        return np.nan
    dfreq = f_span * window_frac
    idx = np.where((freqs >= fpk - dfreq) & (freqs <= fpk + dfreq))[0]
    if idx.size < 20:
        return np.nan
    keep = (idx < idx_pk - guard_bins) | (idx > idx_pk + guard_bins)
    idx_keep = idx[keep]
    if idx_keep.size < 10:
        return np.nan
    loc = power[idx_keep]
    med = np.nanmedian(loc)
    mad = np.nanmedian(np.abs(loc - med))
    if not np.isfinite(mad) or mad <= 0:
        return np.nan
    return (power[idx_pk] - med) / mad

def pol_harmonic_design_matrix(t: np.ndarray, f: float, H: int):
    cols = []
    w = 2 * np.pi * f
    for k in range(1, H + 1):
        wt = w * k * t
        cols.append(np.sin(wt))
        cols.append(np.cos(wt))
    return np.vstack(cols).T

def pol_linear_harmonic_fit(t: np.ndarray, y: np.ndarray, f: float, H: int):
    A = pol_harmonic_design_matrix(t, f, H)
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return coef

def pol_model_from_coef(t: np.ndarray, f: float, coef: np.ndarray):
    H = len(coef) // 2
    y_hat = np.zeros_like(t, dtype=float)
    w = 2 * np.pi * f
    for k in range(1, H + 1):
        a = coef[2*(k-1)]
        b = coef[2*(k-1)+1]
        y_hat += a * np.sin(w*k*t) + b * np.cos(w*k*t)
    return y_hat

def pol_prewhiten_series_harmonic(jd: np.ndarray, y: np.ndarray, yerr: np.ndarray | None = None,
                                  n_iter: int = POL_PREWHITEN_ITERS,
                                  H: int = POL_PREWHITEN_HARMONICS,
                                  f_window_frac: float = POL_PREWHITEN_FREQ_WINDOW_FRAC,
                                  grid_mode: str = POL_PREWHITEN_GRID_MODE):
    tt_raw = np.asarray(jd, float)
    yy_raw = np.asarray(y, float)
    good = np.isfinite(tt_raw) & np.isfinite(yy_raw)
    tt_raw, yy_raw = tt_raw[good], yy_raw[good]
    tt, _ = pol_center_time(tt_raw)
    dy = None
    if yerr is not None:
        dy_raw = np.asarray(yerr, float)[good]
        if np.isfinite(dy_raw).any():
            med_dy = np.nanmedian(dy_raw[np.isfinite(dy_raw)])
            dy = np.where(np.isfinite(dy_raw) & (dy_raw > 0), dy_raw, med_dy)
    freqs0 = pol_suggest_frequency_grid(tt, mode=grid_mode, oversample=POL_PREWHITEN_LS_OVERSAMPLE)
    if freqs0 is None or len(freqs0) == 0:
        return yy_raw.copy()
    resid = yy_raw.copy()
    for k in range(1, n_iter + 1):
        Pk, _ = pol_ls_power_multiharmonic(tt, resid, freqs0, yerr=dy, nterms=H)
        idx_pk = int(np.nanargmax(Pk))
        f_peak = float(freqs0[idx_pk])
        if POL_PREWHITEN_STOP_RULE and (k >= POL_PREWHITEN_MIN_ITERS):
            snr_ls_now = pol_local_ls_snr(freqs0, Pk, idx_pk, window_frac=0.02, guard_bins=5)
            if not np.isfinite(snr_ls_now) or (snr_ls_now < POL_PREWHITEN_SNR_LS_MIN):
                break
        coef0 = pol_linear_harmonic_fit(tt, resid, f_peak, H)
        model = pol_model_from_coef(tt, f_peak, coef0)
        # Optional nonlinear refinement if curve_fit is available
        if curve_fit is not None:
            f_lo = max(freqs0[0],  f_peak * (1 - f_window_frac))
            f_hi = min(freqs0[-1], f_peak * (1 + f_window_frac))
            def model_func(tfit, *params):
                *c, f = params
                return pol_model_from_coef(tfit, f, np.array(c, dtype=float))
            p0 = np.r_[coef0, f_peak]
            lower = np.r_[np.full_like(coef0, -np.inf), f_lo]
            upper = np.r_[np.full_like(coef0,  np.inf), f_hi]
            try:
                popt, _ = curve_fit(
                    model_func, tt, resid, p0=p0, bounds=(lower, upper),
                    sigma=dy, absolute_sigma=True if dy is not None else False, maxfev=40000
                )
                coef = popt[:-1]
                f_ref = float(popt[-1])
                model = pol_model_from_coef(tt, f_ref, coef)
            except Exception:
                pass
        resid = resid - model
    return resid

def build_analysis_frame_from_raw_pol(path: Path, save_generated: bool = POL_SAVE_GENERATED_ANALYSIS_FRAME) -> pd.DataFrame:
    """Build nm/resid_nm_pchip/pw_resid_nm_pchip columns from a raw/basic polarimetry CSV."""
    raw = pd.read_csv(path)
    df = normalize_raw_polarimetry_dataframe(raw)
    df = pol_assign_night_ids(df, gap_days=POL_NIGHT_GAP_THRESHOLD_DAYS)
    df = pol_compute_phase(df)

    for c in ["q", "u", "p"]:
        df[f"{c}_nm"] = df[c] - df.groupby("night_id")[c].transform("mean")

    for lab in ["q", "u", "p"]:
        ycol = f"{lab}_nm"
        yerrcol = f"{lab}_err"
        built = pol_build_pchip_from_binned(df["phase1d"].to_numpy(float), df[ycol].to_numpy(float))
        if built[0] is None:
            # Fallback: if too sparse to build PCHIP, pass through the night-mean-subtracted series.
            df[f"{lab}_resid_nm_pchip"] = df[ycol].to_numpy(float)
        else:
            f_pchip, _, _ = built
            m = df[["phase1d", ycol]].notna().all(axis=1)
            phi = df.loc[m, "phase1d"].to_numpy(float)
            y = df.loc[m, ycol].to_numpy(float)
            resid = y - f_pchip(phi)
            df.loc[m, f"{lab}_resid_nm_pchip"] = resid

    for lab in ["q", "u", "p"]:
        resid = pol_prewhiten_series_harmonic(
            df["jd"].to_numpy(float),
            df[f"{lab}_resid_nm_pchip"].to_numpy(float),
            yerr=df[f"{lab}_err"].to_numpy(float),
        )
        valid = np.isfinite(df[f"{lab}_resid_nm_pchip"].to_numpy(float)) & np.isfinite(df["jd"].to_numpy(float))
        tmp = np.full(len(df), np.nan, dtype=float)
        tmp[valid] = resid
        df[f"{lab}_pw_resid_nm_pchip"] = tmp

    if save_generated:
        stem = path.stem + "_nm_pchip_analysis_frame.csv"
        outdir = path.parent if POL_GENERATED_ANALYSIS_DIR is None else Path(POL_GENERATED_ANALYSIS_DIR)
        outdir.mkdir(parents=True, exist_ok=True)
        outpath = outdir / stem
        df.to_csv(outpath, index=False)
        print(f"[raw pol] wrote generated analysis frame: {outpath}")
    return df

def _clean_sort_with_group(t, y, yerr=None, t_abs=None, group_labels=None):
    """Apply the same finite-row filtering and sorting to optional group labels."""
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(t) & np.isfinite(y)
    if yerr is not None:
        yerr = np.asarray(yerr, dtype=float)
        yerr = np.where(np.isfinite(yerr) & (yerr > 0), yerr, np.nan)
        m &= np.isfinite(yerr)
    if t_abs is not None:
        t_abs = np.asarray(t_abs, dtype=float)
        m &= np.isfinite(t_abs)
    if group_labels is not None:
        group_labels = np.asarray(group_labels, dtype=object)

    t = t[m]
    y = y[m]
    if yerr is not None:
        yerr = yerr[m]
    if t_abs is not None:
        t_abs = t_abs[m]
    if group_labels is not None:
        group_labels = group_labels[m]

    idx = np.argsort(t)
    t = t[idx]
    y = y[idx]
    if yerr is not None:
        yerr = yerr[idx]
    if t_abs is not None:
        t_abs = t_abs[idx]
    if group_labels is not None:
        group_labels = group_labels[idx]
    return t, y, yerr, t_abs, group_labels

def _pick_first_existing(columns, candidates):
    column_map = {str(c).strip().lower(): c for c in columns}
    for candidate in candidates:
        found = column_map.get(str(candidate).strip().lower())
        if found is not None:
            return found
    return None


def _infer_numeric_flux_column(df: pd.DataFrame, excluded=()) -> str | None:
    """Conservative fallback for pipeline CSVs with an unfamiliar flux name."""
    excluded_lower = {str(c).strip().lower() for c in excluded}
    reject_tokens = (
        "time", "err", "error", "sigma", "uncert", "quality", "flag",
        "model", "trend", "systematic", "background", "centroid", "phase",
        "valid", "sample", "count", "npix", "camera", "sector", "cadence",
    )
    ranked = []
    for col in df.columns:
        name = str(col).strip()
        lname = name.lower()
        if lname in excluded_lower or "flux" not in lname:
            continue
        if any(token in lname for token in reject_tokens):
            continue
        numeric = pd.to_numeric(df[col], errors="coerce")
        finite_fraction = float(np.mean(np.isfinite(numeric.to_numpy(float)))) if len(df) else 0.0
        if finite_fraction < 0.5:
            continue
        score = 0
        if "detrend" in lname or "corrected" in lname:
            score += 40
        if "relative" in lname or lname.endswith("_rel") or "normalized" in lname or "medscaled" in lname:
            score += 20
        if lname == "flux":
            score += 5
        ranked.append((score, finite_fraction, name))
    if not ranked:
        return None
    ranked.sort(key=lambda item: (-item[0], -item[1], item[2]))
    return ranked[0][2]


def _tess_error_candidates_for_flux(y_col: str) -> list[str]:
    candidates = list(TESS_ERROR_COL_CANDIDATES_BY_FLUX.get(str(y_col).strip().lower(), []))
    for name in TESS_GENERIC_ERROR_COL_CANDIDATES:
        if name not in candidates:
            candidates.append(name)
    return candidates


def _tess_group_labels_from_frame(df: pd.DataFrame, path: Path) -> np.ndarray:
    """Use sector/source metadata to keep uncertainty rescaling independent."""
    sector_col = _pick_first_existing(df.columns, ["sector", "tess_sector"])
    source_col = _pick_first_existing(df.columns, ["source_file"])
    if sector_col is not None:
        labels = []
        for i, value in enumerate(df[sector_col]):
            if pd.notna(value) and str(value).strip():
                try:
                    fval = float(value)
                    label = f"sector:{int(fval)}" if np.isfinite(fval) and fval.is_integer() else f"sector:{str(value).strip()}"
                except Exception:
                    label = f"sector:{str(value).strip()}"
            elif source_col is not None and pd.notna(df[source_col].iloc[i]):
                label = f"file:{str(df[source_col].iloc[i]).strip()}"
            else:
                label = f"file:{path.name}"
            labels.append(label)
        return np.asarray(labels, dtype=object)
    if source_col is not None:
        return np.asarray([
            f"file:{str(v).strip()}" if pd.notna(v) and str(v).strip() else f"file:{path.name}"
            for v in df[source_col]
        ], dtype=object)
    return np.full(len(df), f"file:{path.name}", dtype=object)


def _robust_first_difference_sigma(t: np.ndarray, y: np.ndarray) -> float:
    """Estimate point noise while rejecting differences across large gaps."""
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    good = np.isfinite(t) & np.isfinite(y)
    t, y = t[good], y[good]
    if y.size < 3:
        return np.nan
    order = np.argsort(t)
    dt, dy = np.diff(t[order]), np.diff(y[order])
    good = np.isfinite(dt) & (dt > 0) & np.isfinite(dy)
    if np.any(good):
        med_dt = np.nanmedian(dt[good])
        if np.isfinite(med_dt) and med_dt > 0:
            good &= dt <= 10.0 * med_dt
    dy = dy[good]
    if dy.size < 2:
        return np.nan
    mad = np.nanmedian(np.abs(dy - np.nanmedian(dy)))
    if np.isfinite(mad) and mad > 0:
        return float((1.4826 * mad) / np.sqrt(2.0))
    std = np.nanstd(dy)
    return float(std / np.sqrt(2.0)) if np.isfinite(std) and std > 0 else np.nan


def _prepare_tess_errors(
    t: np.ndarray,
    y: np.ndarray,
    yerr_raw: np.ndarray | None,
    group_id: np.ndarray | None,
) -> np.ndarray | None:
    """Validate, floor, and optionally sector-rescale formal uncertainties."""
    mode = str(TESS_WEIGHT_MODE).strip().lower()
    if mode not in {"none", "formal", "sector_rescaled"}:
        raise ValueError("TESS_WEIGHT_MODE must be none, formal, or sector_rescaled.")
    if mode == "none":
        return None
    if yerr_raw is None:
        print(f"[TESS weights] mode={mode}: no matching uncertainty column; using equal weights.")
        return None
    err = np.asarray(yerr_raw, dtype=float).copy()
    usable = np.isfinite(err) & (err > 0)
    if np.count_nonzero(usable) < 3:
        print(f"[TESS weights] mode={mode}: fewer than three usable values; using equal weights.")
        return None
    gid = np.zeros(len(err), dtype=int) if group_id is None else np.asarray(group_id, dtype=int)
    floor_frac = float(TESS_ERROR_FLOOR_FRAC)
    if not np.isfinite(floor_frac) or floor_frac < 0:
        raise ValueError("TESS_ERROR_FLOOR_FRAC must be finite and non-negative.")
    global_median = float(np.nanmedian(err[usable]))
    for group in np.unique(gid):
        mask = gid == group
        values = err[mask]
        good = np.isfinite(values) & (values > 0)
        group_median = float(np.nanmedian(values[good])) if np.any(good) else global_median
        values = np.where(good, values, group_median)
        scale = 1.0
        scatter = _robust_first_difference_sigma(np.asarray(t)[mask], np.asarray(y)[mask])
        if mode == "sector_rescaled" and np.count_nonzero(mask) >= int(TESS_SECTOR_RESCALE_MIN_POINTS):
            if np.isfinite(scatter) and scatter > 0 and group_median > 0:
                scale = float(np.clip(scatter / group_median, TESS_SECTOR_SCALE_MIN, TESS_SECTOR_SCALE_MAX))
        values *= scale
        median_scaled = float(np.nanmedian(values[np.isfinite(values) & (values > 0)]))
        if floor_frac > 0 and np.isfinite(median_scaled):
            values = np.maximum(values, floor_frac * median_scaled)
        err[mask] = values
        print(
            f"[TESS weights] group={int(group)} N={np.count_nonzero(mask)} "
            f"formal_med={group_median:.6g} fd_sigma={scatter:.6g} scale={scale:.6g}"
        )
    final_good = np.isfinite(err) & (err > 0)
    final_median = float(np.nanmedian(err[final_good])) if np.any(final_good) else global_median
    return np.where(final_good, err, final_median)


def _read_tess_csv_with_candidates(path: Path, y_candidates):
    df = pd.read_csv(path)
    vprint(1, f"Read TESS CSV | rows={len(df)}")

    time_col = _pick_first_existing(df.columns, TESS_TIME_COL_CANDIDATES)
    if time_col is None:
        raise ValueError(
            f"{path}: missing a recognized TESS time column. "
            f"Expected one of {TESS_TIME_COL_CANDIDATES}; found: {list(df.columns)}"
        )

    forced_col = TESS_FORCE_Y_COL
    if forced_col is not None and str(forced_col).strip():
        y_col = _pick_first_existing(df.columns, [forced_col])
        if y_col is None:
            raise ValueError(
                f"{path}: required TESS column '{forced_col}' not found. "
                f"Found: {list(df.columns)}"
            )
    else:
        y_col = _pick_first_existing(df.columns, y_candidates)
        if y_col is None:
            y_col = _infer_numeric_flux_column(df, excluded=[time_col])
        if y_col is None:
            raise ValueError(
                f"{path}: no usable photometric flux column found. "
                f"Recognized candidates: {list(y_candidates)}; found: {list(df.columns)}"
            )

    err_col = _pick_first_existing(df.columns, _tess_error_candidates_for_flux(str(y_col)))
    vprint(1, f"Using TESS file: {path} | time={time_col} | flux={y_col} | error={err_col or 'none'}")
    t_abs = pd.to_numeric(df[time_col], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(df[y_col], errors="coerce").to_numpy(dtype=float)
    yerr_raw = pd.to_numeric(df[err_col], errors="coerce").to_numpy(dtype=float) if err_col is not None else None
    group_labels = _tess_group_labels_from_frame(df, Path(path))
    t_rel = t_abs - np.nanmin(t_abs)
    t_rel, y, yerr_raw, t_abs, group_labels = _clean_sort_with_group(
        t_rel, y, yerr_raw, t_abs=t_abs, group_labels=group_labels
    )
    return t_rel, y, yerr_raw, t_abs, group_labels, str(y_col), (None if err_col is None else str(err_col))

def load_tess_csv(path: Path) -> TimeSeries:
    t_rel, y, yerr_raw, t_abs, labels, y_col, err_col = _read_tess_csv_with_candidates(path, TESS_Y_COL_CANDIDATES)
    group_id = _factorize_group_labels(labels)
    yerr = _prepare_tess_errors(t_rel, y, yerr_raw, group_id)
    return TimeSeries(
        t=t_rel,
        y=y,
        yerr=yerr,
        name=f"tess({y_col}; err={err_col or 'none'}; weights={TESS_WEIGHT_MODE})",
        t_abs=t_abs,
        group_id=group_id,
    )

def load_tess_pipeline_dir(dirpath: Path, flux_mode: str = "raw", pattern: str = "*.csv", recursive: bool = False) -> TimeSeries:
    dirpath = Path(dirpath)
    if not dirpath.exists():
        raise FileNotFoundError(f"TESS_PIPELINE_DIR not found: {dirpath}")
    files = sorted(dirpath.rglob(pattern) if recursive else dirpath.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {dirpath} matching {pattern!r}")

    mode = str(flux_mode).lower()
    if mode == "raw":
        candidates = TESS_PIPELINE_RAW_Y_COLS
    elif mode == "detrended":
        candidates = TESS_PIPELINE_DETRENDED_Y_COLS
    elif mode == "auto":
        candidates = TESS_PIPELINE_DETRENDED_Y_COLS + TESS_PIPELINE_RAW_Y_COLS + TESS_Y_COL_CANDIDATES
    else:
        raise ValueError(f"TESS_PIPELINE_FLUX must be 'raw', 'detrended', or 'auto'. Got: {flux_mode}")

    t_all, y_all, err_all, label_all = [], [], [], []
    file_names, used_cols, used_err_cols = [], [], []
    any_error_column = False
    skipped = []
    for path in files:
        try:
            _, y, yerr_raw, t_abs, labels, y_col, err_col = _read_tess_csv_with_candidates(path, candidates)
        except Exception as exc:
            skipped.append((path.name, str(exc)))
            continue
        if len(t_abs) == 0:
            skipped.append((path.name, "no finite rows after cleaning"))
            continue
        t_all.append(t_abs)
        y_all.append(y)
        err_all.append(yerr_raw if yerr_raw is not None else np.full(len(y), np.nan, dtype=float))
        label_all.append(labels)
        any_error_column = any_error_column or (yerr_raw is not None)
        file_names.append(path.name)
        used_cols.append(y_col)
        used_err_cols.append(err_col or "none")

    if not t_all:
        msg = f"No usable TESS pipeline CSVs found in {dirpath} matching {pattern!r}"
        if skipped:
            msg += ". First skip: " + skipped[0][0] + " -> " + skipped[0][1]
        raise ValueError(msg)

    t_abs = np.concatenate(t_all)
    y = np.concatenate(y_all)
    yerr_raw = np.concatenate(err_all) if any_error_column else None
    labels = np.concatenate(label_all)
    t_rel = t_abs - np.nanmin(t_abs)
    t_rel, y, yerr_raw, t_abs, labels = _clean_sort_with_group(
        t_rel, y, yerr_raw, t_abs=t_abs, group_labels=labels
    )
    group_id = _factorize_group_labels(labels)
    yerr = _prepare_tess_errors(t_rel, y, yerr_raw, group_id)

    print(f"Loaded {len(file_names)} TESS pipeline CSV file(s) from {dirpath}")
    print("  flux columns used:", sorted(set(used_cols)))
    print("  error columns used:", sorted(set(used_err_cols)))
    if skipped:
        print(f"  skipped {len(skipped)} file(s) that did not match the requested format")
        for name, reason in skipped[:5]:
            print("   -", name, "->", reason)

    return TimeSeries(
        t=t_rel, y=y, yerr=yerr,
        name=f"tess_pipeline({mode}; {len(file_names)} files; weights={TESS_WEIGHT_MODE})",
        t_abs=t_abs, group_id=group_id,
    )

def load_tess_input() -> TimeSeries:
    vprint(1, f"Loading TESS input in mode: {TESS_INPUT_MODE}")
    mode = str(TESS_INPUT_MODE).lower()
    if mode == "spoc_csv":
        return load_tess_csv(TESS_CSV)
    if mode == "pipeline_dir":
        return load_tess_pipeline_dir(
            TESS_PIPELINE_DIR,
            flux_mode=TESS_PIPELINE_FLUX,
            pattern=TESS_PIPELINE_PATTERN,
            recursive=TESS_PIPELINE_RECURSIVE,
        )
    raise ValueError(f"TESS_INPUT_MODE must be 'spoc_csv' or 'pipeline_dir'. Got: {TESS_INPUT_MODE}")

def build_night_groups(t_abs: np.ndarray, mode: str = "gap", gap_days: float = 8.0/24.0,
                       group_labels: np.ndarray | None = None) -> np.ndarray:
    t_abs = np.asarray(t_abs, dtype=float)
    if t_abs.size == 0:
        return np.array([], dtype=int)
    if mode == "integer_jd":
        _, inv = np.unique(np.floor(t_abs).astype(int), return_inverse=True)
        return inv.astype(int)
    if mode in ("run", "subrun"):
        if group_labels is None:
            raise ValueError(f"Group mode {mode!r} requires run/subrun labels.")
        if len(group_labels) != len(t_abs):
            raise ValueError("Group-label array length does not match the polarimetry time array.")
        return _factorize_group_labels(group_labels)
    if mode != "gap":
        raise ValueError(f"Unsupported group mode: {mode}")
    dt = np.diff(t_abs)
    group_id = np.zeros_like(t_abs, dtype=int)
    g = 0
    for i, dti in enumerate(dt, start=1):
        if np.isfinite(dti) and dti > gap_days:
            g += 1
        group_id[i] = g
    return group_id

def load_polarimetry_csv(
    path: Path,
    product: str,
    trend_cfg: NightTrendConfig,
    channels: list[str] | tuple[str, ...] | None = None,
) -> dict:
    """Load selected channels from an analysis frame or raw polarimetry CSV."""
    if product not in POL_PRODUCTS:
        raise ValueError(f"POL_PRODUCT must be one of {list(POL_PRODUCTS.keys())}. Got: {product}")
    selected = [str(k).lower() for k in (channels or ("q", "u", "p"))]
    invalid = [k for k in selected if k not in {"q", "u", "p"}]
    if invalid:
        raise ValueError(f"Unsupported polarimetry channel(s): {invalid}")

    df = pd.read_csv(path)
    vprint(1, f"Read polarimetry CSV | rows={len(df)}")

    # Minimal compatibility aliases for common raw/basic files.
    # This is intentionally conservative so it does not change behavior for files that
    # already use the canonical names.
    for src_name, dst_name in [("JD", "jd"), ("qe", "q_err"), ("ue", "u_err"), ("pe", "p_err")]:
        if src_name in df.columns and dst_name not in df.columns:
            df[dst_name] = df[src_name]

    # If the requested processed product is already present, use it directly.
    required = [POL_TIME_COL]
    required += [POL_PRODUCTS[product][k] for k in selected]
    required += [POL_ERR_COLS[k] for k in selected]
    if all(c in df.columns for c in required):
        t_abs = df[POL_TIME_COL].to_numpy(dtype=float)
        t0 = np.nanmin(t_abs)
        t_rel = t_abs - t0
        out = {}
        group_labels = None
        if trend_cfg.group_mode in ("run", "subrun"):
            group_labels = get_polarimetry_group_labels(df, trend_cfg.group_mode)
        for k in selected:
            y = df[POL_PRODUCTS[product][k]].to_numpy(dtype=float)
            yerr = df[POL_ERR_COLS[k]].to_numpy(dtype=float)
            tt, yy, ee, ta, glab = _clean_sort_with_group(
                t_rel, y, yerr, t_abs=t_abs, group_labels=group_labels
            )
            gid = build_night_groups(
                ta, mode=trend_cfg.group_mode, gap_days=trend_cfg.gap_days,
                group_labels=glab,
            )
            out[k] = TimeSeries(t=tt, y=yy, yerr=ee, name=f"pol_{k}", t_abs=ta, group_id=gid)
        return out

    # Otherwise, if a raw/basic set of columns is present, build the analysis-frame
    # products first and continue exactly as before.
    raw_required = ["jd", "q", "u", "p", "q_err", "u_err", "p_err"]
    if all(c in df.columns for c in raw_required):
        vprint(1, "Detected raw/basic polarimetry input; building nm/resid_nm_pchip/pw_resid_nm_pchip products in memory.")
        df_proc = build_analysis_frame_from_raw_pol(path, save_generated=POL_SAVE_GENERATED_ANALYSIS_FRAME)
    else:
        missing = [c for c in required if c not in df.columns]
        raise ValueError(f"{path}: missing columns: {missing}. Found: {list(df.columns)}")

    required2 = [POL_TIME_COL]
    required2 += [POL_PRODUCTS[product][k] for k in selected]
    required2 += [POL_ERR_COLS[k] for k in selected]
    missing2 = [c for c in required2 if c not in df_proc.columns]
    if missing2:
        raise ValueError(f"Processed polarimetry dataframe is missing columns: {missing2}")

    t_abs = df_proc[POL_TIME_COL].to_numpy(dtype=float)
    t0 = np.nanmin(t_abs)
    t_rel = t_abs - t0
    out = {}
    group_labels = None
    if trend_cfg.group_mode in ("run", "subrun"):
        group_labels = get_polarimetry_group_labels(df_proc, trend_cfg.group_mode)
    for k in selected:
        y = df_proc[POL_PRODUCTS[product][k]].to_numpy(dtype=float)
        yerr = df_proc[POL_ERR_COLS[k]].to_numpy(dtype=float)
        tt, yy, ee, ta, glab = _clean_sort_with_group(
            t_rel, y, yerr, t_abs=t_abs, group_labels=group_labels
        )
        gid = build_night_groups(
            ta, mode=trend_cfg.group_mode, gap_days=trend_cfg.gap_days,
            group_labels=glab,
        )
        out[k] = TimeSeries(t=tt, y=yy, yerr=ee, name=f"pol_{k}", t_abs=ta, group_id=gid)
    return out

# ----------------------------------------------------------------------
# General utilities
# ----------------------------------------------------------------------

def vprint(level: int, *args, **kwargs):
    if VERBOSE >= level:
        print(*args, **kwargs)

def detrend_poly(ts: TimeSeries, order: int = 1) -> TimeSeries:
    vprint(2, f"Detrending {ts.name} with polynomial order {order}")
    if order < 0 or len(ts.t) < max(3, order + 2):
        return ts
    if ts.yerr is not None:
        w = 1.0 / np.square(ts.yerr)
        w = np.where(np.isfinite(w), w, 0.0)
    else:
        w = np.ones_like(ts.y)
    coeff = np.polyfit(ts.t, ts.y, deg=order, w=np.sqrt(np.maximum(w, 0.0)))
    trend = np.polyval(coeff, ts.t)
    return TimeSeries(
        t=ts.t.copy(),
        y=ts.y - trend,
        yerr=None if ts.yerr is None else ts.yerr.copy(),
        name=ts.name,
        t_abs=None if ts.t_abs is None else ts.t_abs.copy(),
        group_id=None if ts.group_id is None else ts.group_id.copy(),
    )

def compute_T_full(ts: TimeSeries) -> float:
    return float(ts.t[-1] - ts.t[0]) if len(ts.t) >= 2 else 0.0

def longest_contiguous_segment_duration(t: np.ndarray, gap_days: float) -> float:
    if len(t) < 2:
        return 0.0
    ts = np.sort(t)
    dt = np.diff(ts)
    breaks = np.where(dt > gap_days)[0]
    starts = np.r_[0, breaks + 1]
    ends = np.r_[breaks + 1, len(ts)]
    seg_durs = []
    for s, e in zip(starts, ends):
        if e - s >= 2:
            seg_durs.append(ts[e - 1] - ts[s])
    return float(np.max(seg_durs)) if seg_durs else float(ts[-1] - ts[0])

def compute_Tseg(ts: TimeSeries, gap_days: float = 1.0) -> float:
    return longest_contiguous_segment_duration(ts.t, gap_days=gap_days)

def make_frequency_grid(fmin: float, fmax: float, df: float, max_points: int = 60000) -> np.ndarray:
    if not np.isfinite(df) or df <= 0:
        raise ValueError("Frequency spacing df must be finite and positive.")
    n = int(np.floor((fmax - fmin) / df)) + 1
    if n <= 1:
        return np.array([fmin], dtype=float)
    if n > max_points:
        return np.linspace(fmin, fmax, max_points, dtype=float)
    return fmin + df * np.arange(n, dtype=float)


def _same_length_convolve(values: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Return a centered convolution with exactly ``len(values)`` samples."""
    full = np.convolve(values, kernel, mode="full")
    start = (len(kernel) - 1) // 2
    return full[start:start + len(values)]


def smooth_polarimetry_periodogram(
    freqs: np.ndarray,
    power: np.ndarray,
    baseline_days: float,
    *,
    enabled: bool | None = None,
    kernel_name: str | None = None,
    width_resolution_elements: float | None = None,
) -> tuple[np.ndarray, dict]:
    """Smooth power with a NaN-aware, edge-corrected convolution.

    The requested width is the boxcar full width or Gaussian FWHM in units of
    the independent frequency resolution ``1 / baseline_days``.  This makes a
    saved width reproducible even if frequency-grid oversampling later changes.
    """
    f = np.asarray(freqs, dtype=float)
    p = np.asarray(power, dtype=float)
    use = POL_SMOOTH_ENABLED if enabled is None else bool(enabled)
    name = (POL_SMOOTH_KERNEL if kernel_name is None else str(kernel_name)).strip().lower()
    width_re = (
        POL_SMOOTH_WIDTH_RES_ELEMS
        if width_resolution_elements is None
        else float(width_resolution_elements)
    )
    if name not in {"gaussian", "boxcar"}:
        raise ValueError("Polarimetry smoothing kernel must be 'gaussian' or 'boxcar'.")
    if not np.isfinite(width_re) or width_re <= 0:
        raise ValueError("Polarimetry smoothing width must be positive.")

    resolution_cpd = 1.0 / float(baseline_days) if np.isfinite(baseline_days) and baseline_days > 0 else np.nan
    steps = np.diff(f)
    steps = steps[np.isfinite(steps) & (steps > 0)]
    grid_step_cpd = float(np.nanmedian(steps)) if steps.size else np.nan
    width_cpd = width_re * resolution_cpd if np.isfinite(resolution_cpd) else np.nan
    metadata = {
        "enabled": bool(use),
        "applied": False,
        "kernel": name,
        "width_resolution_elements": float(width_re),
        "resolution_cpd": float(resolution_cpd) if np.isfinite(resolution_cpd) else None,
        "effective_width_cpd": float(width_cpd) if np.isfinite(width_cpd) else None,
        "grid_step_cpd": float(grid_step_cpd) if np.isfinite(grid_step_cpd) else None,
        "width_definition": "Gaussian FWHM or boxcar full width",
        "fit_uses_unsmoothed_time_series": True,
    }
    if not use or len(p) < 3 or not np.isfinite(width_cpd) or not np.isfinite(grid_step_cpd):
        return p.copy(), metadata

    width_bins = width_cpd / grid_step_cpd
    metadata["requested_width_grid_samples"] = float(width_bins)
    if width_bins < 2.0:
        # Convolution cannot represent a sub-grid kernel. Returning the input
        # is more honest than silently broadening the requested physical width.
        metadata["not_applied_reason"] = "requested width is under-sampled on this frequency grid"
        return p.copy(), metadata
    if name == "boxcar":
        n = max(1, int(round(width_bins)))
        if n % 2 == 0:
            n += 1
        n = min(n, len(p) if len(p) % 2 == 1 else max(1, len(p) - 1))
        kernel = np.ones(n, dtype=float)
    else:
        sigma_bins = max(width_bins / 2.354820045, 1e-6)
        radius = max(1, int(np.ceil(4.0 * sigma_bins)))
        radius = min(radius, max(1, len(p) - 1))
        x = np.arange(-radius, radius + 1, dtype=float)
        kernel = np.exp(-0.5 * np.square(x / sigma_bins))
    kernel /= np.sum(kernel)

    valid = np.isfinite(p).astype(float)
    numerator = _same_length_convolve(np.where(np.isfinite(p), p, 0.0), kernel)
    denominator = _same_length_convolve(valid, kernel)
    smoothed = np.divide(
        numerator,
        denominator,
        out=np.full_like(p, np.nan, dtype=float),
        where=denominator > 1e-12,
    )
    metadata["kernel_samples"] = int(len(kernel))
    metadata["applied"] = True
    return smoothed, metadata


def polarimetry_smoothing_metadata(baseline_days: float) -> dict:
    """Return the smoothing configuration in JSON-safe form."""
    _, metadata = smooth_polarimetry_periodogram(
        np.array([0.0, 1.0, 2.0]),
        np.ones(3),
        baseline_days,
    )
    metadata.pop("kernel_samples", None)
    metadata.pop("grid_step_cpd", None)
    metadata.pop("applied", None)
    metadata.pop("requested_width_grid_samples", None)
    metadata.pop("not_applied_reason", None)
    return metadata

def make_local_frequency_grid(f_center: float, half_window: float, df: float,
                              fmin: float | None = None, fmax: float | None = None,
                              max_points: int = POL_LOCAL_MAX_POINTS) -> np.ndarray:
    fmin = FMIN if fmin is None else float(fmin)
    fmax = FMAX if fmax is None else float(fmax)
    flo = max(fmin, float(f_center) - float(half_window))
    fhi = min(fmax, float(f_center) + float(half_window))
    if fhi <= flo:
        return np.array([max(fmin, min(fmax, float(f_center)))], dtype=float)
    return make_frequency_grid(flo, fhi, df, max_points=max_points)

def make_plot_frequency_grid(ts: TimeSeries, oversample: float = 1.0, gap_days: float = 1.0,
                             max_points: int = PLOT_MAX_GRID_POINTS) -> np.ndarray:
    Tseg = max(compute_Tseg(ts, gap_days=gap_days), 1e-8)
    df = (1.0 / Tseg) / max(oversample, 1e-8)
    return make_frequency_grid(FMIN, FMAX, df, max_points=max_points)

def make_fullbaseline_plot_frequency_grid(ts: TimeSeries, oversample: float = 1.0,
                                          max_points: int = POL_FULL_RANGE_PLOT_POINTS) -> np.ndarray:
    Tfull = max(compute_T_full(ts), 1e-8)
    df = (1.0 / Tfull) / max(oversample, 1e-8)
    return make_frequency_grid(FMIN, FMAX, df, max_points=max_points)

def make_tess_baseline_matrix(ts: TimeSeries) -> np.ndarray:
    return np.ones((len(ts.t), 1), dtype=float)

def make_pol_baseline_matrix(ts: TimeSeries, cfg: NightTrendConfig) -> np.ndarray:
    if ts.group_id is None:
        raise ValueError(f"{ts.name}: group_id is required for night-aware polarimetry fits.")
    gid = np.asarray(ts.group_id, dtype=int)
    groups = np.unique(gid)
    cols = []
    if cfg.use_offsets:
        for g in groups:
            cols.append((gid == g).astype(float))
    if cfg.use_slopes:
        for g in groups:
            m = gid == g
            x = np.zeros_like(ts.t, dtype=float)
            tg = ts.t[m]
            if tg.size > 0:
                x[m] = tg - np.nanmean(tg)
            cols.append(x)
    if not cols:
        cols.append(np.ones_like(ts.t, dtype=float))
    return np.column_stack(cols)

def weighted_linear_solve(y: np.ndarray, X: np.ndarray, w: np.ndarray | None = None) -> dict:
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    if w is None:
        w = np.ones_like(y, dtype=float)
    else:
        w = np.asarray(w, dtype=float)
        w = np.where(np.isfinite(w) & (w > 0), w, 0.0)
    sw = np.sqrt(w)
    Xw = X * sw[:, None]
    yw = y * sw
    beta, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
    yhat = X @ beta
    resid = y - yhat
    rss = float(np.sum(w * resid * resid))
    return {"beta": beta, "yhat": yhat, "rss": rss, "resid": resid}

def signal_weights(ts: TimeSeries) -> np.ndarray:
    if ts.yerr is None:
        return np.ones_like(ts.y, dtype=float)
    w = 1.0 / np.square(ts.yerr)
    return np.where(np.isfinite(w) & (w > 0), w, 0.0)

def design_matrix_multi(t: np.ndarray, freqs: np.ndarray, baseline: np.ndarray) -> np.ndarray:
    cols = []
    for f in np.asarray(freqs, dtype=float):
        ang = 2.0 * np.pi * f * t
        cols.append(np.sin(ang))
        cols.append(np.cos(ang))
    if cols:
        Xsig = np.column_stack(cols)
        return np.column_stack([Xsig, baseline])
    return np.asarray(baseline, dtype=float)

def signal_from_beta(t: np.ndarray, freqs: np.ndarray, beta: np.ndarray) -> np.ndarray:
    freqs = np.asarray(freqs, dtype=float)
    beta = np.asarray(beta, dtype=float)
    if freqs.size == 0:
        return np.zeros_like(t, dtype=float)
    nmode = freqs.size
    out = np.zeros_like(t, dtype=float)
    for i, f in enumerate(freqs):
        s_coeff = beta[2 * i]
        c_coeff = beta[2 * i + 1]
        out += s_coeff * np.sin(2.0 * np.pi * f * t) + c_coeff * np.cos(2.0 * np.pi * f * t)
    return out

def component_signal(t: np.ndarray, f: float, s_coeff: float, c_coeff: float) -> np.ndarray:
    return s_coeff * np.sin(2.0 * np.pi * f * t) + c_coeff * np.cos(2.0 * np.pi * f * t)

def baseline_from_beta(beta: np.ndarray, nfreq: int, baseline_matrix: np.ndarray) -> np.ndarray:
    return baseline_matrix @ beta[2 * nfreq:]

def sc_to_amp_phase(s_coeff: float, c_coeff: float) -> tuple[float, float]:
    amp = float(np.hypot(s_coeff, c_coeff))
    phase = float(np.arctan2(c_coeff, s_coeff))
    return amp, phase

def lomb_scargle_power(ts: TimeSeries, freqs: np.ndarray) -> np.ndarray:
    if _HAVE_ASTROPY:
        if ts.yerr is not None:
            ls = LombScargle(ts.t, ts.y, dy=ts.yerr)
        else:
            ls = LombScargle(ts.t, ts.y)
        return np.asarray(ls.power(freqs, normalization="psd"), dtype=float)
    baseline = make_tess_baseline_matrix(ts)
    return nuisance_periodogram(ts, freqs, baseline_matrix=baseline)

def nuisance_periodogram(ts: TimeSeries, freqs: np.ndarray, baseline_matrix: np.ndarray) -> np.ndarray:
    w = signal_weights(ts)
    null_fit = weighted_linear_solve(ts.y, baseline_matrix, w)
    rss0 = max(null_fit["rss"], 1e-30)
    power = np.full_like(freqs, np.nan, dtype=float)
    for i, f in enumerate(freqs):
        X = design_matrix_multi(ts.t, np.array([f]), baseline_matrix)
        alt_fit = weighted_linear_solve(ts.y, X, w)
        power[i] = max(0.0, (rss0 - alt_fit["rss"]) / rss0)
    return power

def spectral_window(ts: TimeSeries, freqs: np.ndarray) -> np.ndarray:
    t = np.asarray(ts.t, dtype=float)
    w = np.ones_like(t, dtype=float)
    z = np.array([np.abs(np.sum(w * np.exp(-2j * np.pi * f * t))) ** 2 for f in freqs], dtype=float)
    m = np.nanmax(z)
    return z / m if np.isfinite(m) and m > 0 else z

def trimmed_median(x: np.ndarray, trim_top_frac: float = 0.1) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan
    if trim_top_frac > 0:
        k = int(np.floor((1.0 - trim_top_frac) * x.size))
        k = min(max(k, 1), x.size)
        x = np.partition(x, k - 1)[:k]
    return float(np.nanmedian(x))


def tess_local_snr_grid(f_fit: float, T_noise: float, ks: float,
                        fmin: float | None = None, fmax: float | None = None,
                        max_points: int = POL_LOCAL_MAX_POINTS) -> tuple[np.ndarray, float, float]:
    """
    Build a dedicated local frequency grid for the TESS SNR estimate.

    The window is defined entirely in frequency units:
      - a central guard region around the fitted peak
      - two sidebands extending out by ks * (1/T_noise)
    """
    fmin = FMIN if fmin is None else float(fmin)
    fmax = FMAX if fmax is None else float(fmax)
    T_noise = max(float(T_noise), 1e-8)
    res = 1.0 / T_noise
    guard_half = max(KFIT * (2.0 / T_noise), 2.0 * res)
    side_half = max(float(ks) * res, 4.0 * res)
    half_window = guard_half + side_half
    df_local = max(res / 5.0, 1e-8)
    freqs_local = make_local_frequency_grid(
        f_fit, half_window, df_local, fmin=fmin, fmax=fmax, max_points=max_points
    )
    return freqs_local, guard_half, side_half

def tess_local_noise_floor_from_local_periodogram(freqs: np.ndarray, power: np.ndarray, f_fit: float,
                                                  guard_half: float, side_half: float,
                                                  trim_top_frac: float = TRIM_TOP_FRAC,
                                                  min_points: int = 12) -> tuple[float, int]:
    """
    Estimate a genuinely local noise floor from sidebands around f_fit.

    No whole-spectrum fallback is used. If the initial sidebands are too sparse,
    the sideband width is widened locally in a controlled way.
    """
    freqs = np.asarray(freqs, dtype=float)
    power = np.asarray(power, dtype=float)
    if freqs.size < 5 or power.size != freqs.size:
        return np.nan, 0

    expand = 1.0
    n_used = 0
    for _ in range(6):
        outer = guard_half + expand * side_half
        m = np.isfinite(power) & (np.abs(freqs - f_fit) >= guard_half) & (np.abs(freqs - f_fit) <= outer)
        vals = power[m]
        n_used = int(np.count_nonzero(np.isfinite(vals)))
        noise = trimmed_median(vals, trim_top_frac=trim_top_frac)
        if np.isfinite(noise) and (noise > 0) and (n_used >= min_points):
            return float(noise), n_used
        expand *= 1.5
    return np.nan, n_used

def tess_local_snr_from_fit(ts: TimeSeries, f_fit: float, T_noise: float, ks: float,
                            trim_top_frac: float = TRIM_TOP_FRAC) -> tuple[float, float]:
    """
    Compute the TESS SNR from a dedicated local periodogram around the fitted peak.

    This avoids dependence on the broad discovery grid and uses only local sidebands
    around the fitted frequency.
    """
    freqs_local, guard_half, side_half = tess_local_snr_grid(f_fit, T_noise=T_noise, ks=ks)
    power_local = lomb_scargle_power(ts, freqs_local)
    if not np.isfinite(power_local).any():
        return np.nan, np.nan

    m_peak = np.isfinite(power_local) & (np.abs(freqs_local - f_fit) <= guard_half)
    if np.any(m_peak):
        p0 = float(np.nanmax(power_local[m_peak]))
    else:
        j = int(np.nanargmin(np.abs(freqs_local - f_fit)))
        p0 = float(power_local[j])

    noise, _n_used = tess_local_noise_floor_from_local_periodogram(
        freqs_local, power_local, f_fit=f_fit, guard_half=guard_half,
        side_half=side_half, trim_top_frac=trim_top_frac
    )
    W = float(p0 / noise) if (np.isfinite(noise) and noise > 0) else np.nan
    snr = float(np.sqrt(W)) if np.isfinite(W) and W > 0 else np.nan
    return snr, W



def pol_local_snr_grid(f_fit: float, T_noise: float, ks: float,
                       fmin: float | None = None, fmax: float | None = None,
                       max_points: int = POL_LOCAL_MAX_POINTS) -> tuple[np.ndarray, float, float]:
    """
    Build a dedicated local frequency grid for the polarimetric SNR estimate.

    Compared with the TESS version, use a somewhat wider guard and sideband
    region to better sample the more structured alias/window environment.
    """
    fmin = FMIN if fmin is None else float(fmin)
    fmax = FMAX if fmax is None else float(fmax)
    T_noise = max(float(T_noise), 1e-8)
    res = 1.0 / T_noise
    guard_half = max(KFIT * (2.0 / T_noise), 3.0 * res)
    side_half = max(1.5 * float(ks) * res, 8.0 * res)
    half_window = guard_half + side_half
    df_local = max(res / 5.0, 1e-8)
    freqs_local = make_local_frequency_grid(
        f_fit, half_window, df_local, fmin=fmin, fmax=fmax, max_points=max_points
    )
    return freqs_local, guard_half, side_half

def pol_local_noise_floor_from_local_periodogram(freqs: np.ndarray, power: np.ndarray, f_fit: float,
                                                 guard_half: float, side_half: float,
                                                 trim_top_frac: float = TRIM_TOP_FRAC,
                                                 min_points: int = 12) -> tuple[float, int]:
    """
    Estimate a local polarimetric noise floor from sidebands around f_fit,
    widening only locally if needed. No whole-spectrum fallback is used.
    """
    freqs = np.asarray(freqs, dtype=float)
    power = np.asarray(power, dtype=float)
    if freqs.size < 5 or power.size != freqs.size:
        return np.nan, 0

    expand = 1.0
    n_used = 0
    for _ in range(6):
        outer = guard_half + expand * side_half
        m = np.isfinite(power) & (np.abs(freqs - f_fit) >= guard_half) & (np.abs(freqs - f_fit) <= outer)
        vals = power[m]
        n_used = int(np.count_nonzero(np.isfinite(vals)))
        noise = trimmed_median(vals, trim_top_frac=trim_top_frac)
        if np.isfinite(noise) and (noise > 0) and (n_used >= min_points):
            return float(noise), n_used
        expand *= 1.5
    return np.nan, n_used

def pol_local_snr_from_fit(ts: TimeSeries, f_fit: float, T_noise: float, ks: float,
                           baseline_matrix: np.ndarray,
                           trim_top_frac: float = TRIM_TOP_FRAC) -> tuple[float, float]:
    """
    Recompute the polarimetric SNR on a fresh ultra-local nuisance periodogram
    centered on the fitted frequency, with frequency-defined guard/sideband
    widths and no whole-spectrum fallback.
    """
    freqs_local, guard_half, side_half = pol_local_snr_grid(f_fit, T_noise=T_noise, ks=ks)
    power_local = nuisance_periodogram(ts, freqs_local, baseline_matrix=baseline_matrix)
    if not np.isfinite(power_local).any():
        return np.nan, np.nan

    m_peak = np.isfinite(power_local) & (np.abs(freqs_local - f_fit) <= guard_half)
    if np.any(m_peak):
        p0 = float(np.nanmax(power_local[m_peak]))
    else:
        j = int(np.nanargmin(np.abs(freqs_local - f_fit)))
        p0 = float(power_local[j])

    noise, _n_used = pol_local_noise_floor_from_local_periodogram(
        freqs_local, power_local, f_fit=f_fit, guard_half=guard_half,
        side_half=side_half, trim_top_frac=trim_top_frac
    )
    W = float(p0 / noise) if (np.isfinite(noise) and noise > 0) else np.nan
    snr = float(np.sqrt(W)) if np.isfinite(W) and W > 0 else np.nan
    return snr, W


def fit_frequency_with_design(ts: TimeSeries, f0: float, T: float, kfit: float, fmin: float, fmax: float,
                              baseline_matrix: np.ndarray, n_steps: int = 1001,
                              hard_fmin: float | None = None, hard_fmax: float | None = None) -> dict:
    w = signal_weights(ts)
    df_fit = kfit * (2.0 / T)
    f_lo = max(fmin, f0 - df_fit)
    f_hi = min(fmax, f0 + df_fit)
    if hard_fmin is not None:
        f_lo = max(f_lo, float(hard_fmin))
    if hard_fmax is not None:
        f_hi = min(f_hi, float(hard_fmax))
    if f_hi <= f_lo:
        f_lo = max(fmin, f0 - 0.5 * df_fit)
        f_hi = min(fmax, f0 + 0.5 * df_fit)
        if hard_fmin is not None:
            f_lo = max(f_lo, float(hard_fmin))
        if hard_fmax is not None:
            f_hi = min(f_hi, float(hard_fmax))
    freqs = np.linspace(f_lo, f_hi, max(11, int(n_steps)))
    rss = np.full_like(freqs, np.nan, dtype=float)
    fits = [None] * len(freqs)
    for i, f in enumerate(freqs):
        X = design_matrix_multi(ts.t, np.array([f]), baseline_matrix)
        fit = weighted_linear_solve(ts.y, X, w)
        fits[i] = fit
        rss[i] = fit["rss"]
    j = int(np.nanargmin(rss))
    f_best = float(freqs[j])
    fit_best = fits[j]
    beta = np.asarray(fit_best["beta"], dtype=float)
    s_coeff, c_coeff = float(beta[0]), float(beta[1])
    amp, phase = sc_to_amp_phase(s_coeff, c_coeff)
    signal_model = component_signal(ts.t, f_best, s_coeff, c_coeff)
    baseline_model = baseline_from_beta(beta, 1, baseline_matrix)
    full_model = signal_model + baseline_model
    return {
        "best_f": f_best,
        "amp": amp,
        "phase": phase,
        "rss": float(fit_best["rss"]),
        "beta": beta,
        "signal_model": signal_model,
        "baseline_model": baseline_model,
        "full_model": full_model,
        "resid": ts.y - full_model,
        "df_fit": float(df_fit),
        "s_coeff": s_coeff,
        "c_coeff": c_coeff,
    }

def prewhiten(ts: TimeSeries, model: np.ndarray) -> TimeSeries:
    return TimeSeries(
        t=ts.t.copy(),
        y=ts.y - np.asarray(model, dtype=float),
        yerr=None if ts.yerr is None else ts.yerr.copy(),
        name=ts.name,
        t_abs=None if ts.t_abs is None else ts.t_abs.copy(),
        group_id=None if ts.group_id is None else ts.group_id.copy(),
    )

def dedupe_frequency_rows(df: pd.DataFrame, fcol: str, score_cols=("snr_local", "amp_local", "W_local"),
                         min_sep: float = 0.0) -> pd.DataFrame:
    if df is None or len(df) == 0:
        return df.copy() if isinstance(df, pd.DataFrame) else pd.DataFrame()
    out = df.copy().reset_index(drop=True)
    if fcol not in out.columns:
        return out
    score = np.zeros(len(out), dtype=float)
    for col in score_cols:
        if col in out.columns:
            vals = pd.to_numeric(out[col], errors="coerce").to_numpy(dtype=float)
            vals = np.where(np.isfinite(vals), vals, -np.inf)
            score = np.maximum(score, vals)
    if not np.isfinite(score).any():
        score[:] = np.arange(len(out), 0, -1, dtype=float)
    out["_score"] = score
    out = out.sort_values(["_score", fcol], ascending=[False, True]).reset_index(drop=True)
    keep_rows = []
    keep_freqs = []
    for _, row in out.iterrows():
        f = float(row[fcol])
        if not np.isfinite(f):
            continue
        if all(abs(f - fk) >= min_sep for fk in keep_freqs):
            keep_rows.append(row.drop(labels=["_score"]).to_dict())
            keep_freqs.append(f)
    keep = pd.DataFrame(keep_rows)
    if len(keep) == 0:
        return keep
    if "mode" in keep.columns:
        keep = keep.sort_values("mode").reset_index(drop=True)
    return keep


def collapse_close_frequency_components(freqs: np.ndarray, beta: np.ndarray, min_sep: float) -> tuple[np.ndarray, list[int], list[list[int]]]:
    freqs = np.asarray(freqs, dtype=float)
    beta = np.asarray(beta, dtype=float)
    nmode = len(freqs)
    if nmode <= 1:
        return freqs.copy(), [0] if nmode == 1 else [], [[0]] if nmode == 1 else []
    amps = np.hypot(beta[0:2*nmode:2], beta[1:2*nmode:2])
    order = np.argsort(freqs)
    groups = []
    current = [int(order[0])]
    for idx in order[1:]:
        idx = int(idx)
        if abs(freqs[idx] - freqs[current[-1]]) < min_sep:
            current.append(idx)
        else:
            groups.append(current)
            current = [idx]
    groups.append(current)
    keep = [max(g, key=lambda j: float(amps[j])) for g in groups]
    keep = sorted(keep, key=lambda j: freqs[j])
    freqs_keep = freqs[keep]
    return freqs_keep, keep, groups

# ----------------------------------------------------------------------
# Global multisinusoid fit
# ----------------------------------------------------------------------

def fit_global_multisin(ts: TimeSeries, seed_freqs: np.ndarray, baseline_matrix: np.ndarray,
                        freq_resolution: float, bound_mult: float = 3.0) -> dict:
    seed_freqs = np.asarray(seed_freqs, dtype=float)
    w = signal_weights(ts)
    min_sep = max(float(freq_resolution) * float(MIN_FREQ_SEP_MULT), 1e-8)

    if seed_freqs.size == 0:
        vprint(1, f"[{ts.name}] global fit skipped: no frequencies")
        fit = weighted_linear_solve(ts.y, baseline_matrix, w)
        return {
            "success": True,
            "message": "no frequencies",
            "freqs": np.array([], dtype=float),
            "beta": fit["beta"],
            "rss": float(fit["rss"]),
            "full_model": fit["yhat"],
            "signal_model": np.zeros_like(ts.y),
            "baseline_model": fit["yhat"],
            "resid": fit["resid"],
            "component_rows": [],
            "seed_freqs_used": np.array([], dtype=float),
            "min_sep": min_sep,
        }

    seed_df = pd.DataFrame({"f_seed": seed_freqs, "seed_rank": np.arange(len(seed_freqs)), "snr_local": np.arange(len(seed_freqs), 0, -1, dtype=float)})
    seed_df = dedupe_frequency_rows(seed_df, "f_seed", score_cols=("snr_local",), min_sep=min_sep)
    seed_freqs = seed_df["f_seed"].to_numpy(dtype=float) if not seed_df.empty else np.array([], dtype=float)
    if seed_freqs.size == 0:
        fit = weighted_linear_solve(ts.y, baseline_matrix, w)
        return {
            "success": True,
            "message": "all seed frequencies merged away",
            "freqs": np.array([], dtype=float),
            "beta": fit["beta"],
            "rss": float(fit["rss"]),
            "full_model": fit["yhat"],
            "signal_model": np.zeros_like(ts.y),
            "baseline_model": fit["yhat"],
            "resid": fit["resid"],
            "component_rows": [],
            "seed_freqs_used": np.array([], dtype=float),
            "min_sep": min_sep,
        }

    delta = max(bound_mult * float(freq_resolution), 1e-8)
    lb = np.maximum(FMIN, seed_freqs - delta)
    ub = np.minimum(FMAX, seed_freqs + delta)

    def solve_linear(freqs_now):
        X = design_matrix_multi(ts.t, freqs_now, baseline_matrix)
        return weighted_linear_solve(ts.y, X, w)

    vprint(1, f"[{ts.name}] starting global multisin fit | npts={len(ts.t)} | nmodes={seed_freqs.size} | baseline_cols={baseline_matrix.shape[1]} | df={freq_resolution:.6g} c/d | bound=±{delta:.6g} c/d | min_sep={min_sep:.6g} c/d")
    t0 = time.time()
    nfev_limit = None if GLOBAL_MAX_NFEV is None else int(GLOBAL_MAX_NFEV)

    if _HAVE_LSQ and seed_freqs.size > 0:
        def fun(freqs_now):
            fit = solve_linear(freqs_now)
            sw = np.sqrt(w)
            return fit["resid"] * sw

        try:
            opt = least_squares(
                fun, x0=seed_freqs, bounds=(lb, ub), method="trf",
                xtol=1e-12, ftol=1e-12, gtol=1e-12,
                max_nfev=nfev_limit, verbose=LSQ_VERBOSE
            )
            freqs_best = np.asarray(opt.x, dtype=float)
            success = bool(opt.success)
            message = str(opt.message)
            vprint(1, f"[{ts.name}] global multisin fit finished in {time.time()-t0:.1f}s | success={success} | nfev={getattr(opt, 'nfev', 'NA')} | message={message}")
        except Exception as exc:
            freqs_best = seed_freqs.copy()
            success = False
            message = f"least_squares failed: {exc}"
            vprint(1, f"[{ts.name}] global multisin fit failed after {time.time()-t0:.1f}s | {message}")
    else:
        freqs_best = seed_freqs.copy()
        success = False
        message = "SciPy least_squares unavailable; kept seed frequencies fixed."
        vprint(1, f"[{ts.name}] global multisin fit skipped optimization | {message}")

    fit = solve_linear(freqs_best)
    beta = np.asarray(fit["beta"], dtype=float)
    freqs_keep, keep_idx, groups = collapse_close_frequency_components(freqs_best, beta, min_sep=min_sep)
    collapsed = any(len(g) > 1 for g in groups)
    if collapsed:
        ngroup = sum(len(g) > 1 for g in groups)
        message = (message + f" | merged {ngroup} close-frequency group(s)").strip()
        vprint(1, f"[{ts.name}] merged {ngroup} close-frequency group(s) with min_sep={min_sep:.6g} c/d")
        fit = solve_linear(freqs_keep)
        beta = np.asarray(fit["beta"], dtype=float)
        freqs_best = np.asarray(freqs_keep, dtype=float)

    nmode = freqs_best.size
    signal_model = signal_from_beta(ts.t, freqs_best, beta)
    baseline_model = baseline_from_beta(beta, nmode, baseline_matrix)
    rows = []
    for i, f in enumerate(freqs_best):
        s_coeff = float(beta[2 * i])
        c_coeff = float(beta[2 * i + 1])
        amp, phase = sc_to_amp_phase(s_coeff, c_coeff)
        rows.append({
            "mode": i + 1,
            "f": float(f),
            "s_coeff": s_coeff,
            "c_coeff": c_coeff,
            "amp": amp,
            "phase": phase,
        })
    return {
        "success": success,
        "message": message,
        "freqs": freqs_best,
        "beta": beta,
        "rss": float(fit["rss"]),
        "full_model": signal_model + baseline_model,
        "signal_model": signal_model,
        "baseline_model": baseline_model,
        "resid": fit["resid"],
        "component_rows": rows,
        "seed_freqs_used": seed_freqs,
        "min_sep": min_sep,
    }

# ----------------------------------------------------------------------
# Extraction logic
# ----------------------------------------------------------------------
# Extraction logic
# ----------------------------------------------------------------------

def extract_tess_modes(tess0: TimeSeries) -> tuple[pd.DataFrame, dict, list[dict], TimeSeries, np.ndarray, np.ndarray, np.ndarray]:
    tess = TimeSeries(
        t=tess0.t.copy(), y=tess0.y.copy(),
        yerr=None if tess0.yerr is None else tess0.yerr.copy(),
        name=tess0.name, t_abs=None if tess0.t_abs is None else tess0.t_abs.copy(),
        group_id=None
    )
    Tfull = max(compute_T_full(tess0), 1e-8)
    Tseg = max(compute_Tseg(tess0, gap_days=1.0), 1e-8)
    if TESS_GRID_MODE == "full_baseline":
        Tgrid = Tfull
    elif TESS_GRID_MODE == "longest_chunk":
        Tgrid = Tseg
    else:
        raise ValueError(f"Unknown TESS_GRID_MODE={TESS_GRID_MODE!r}; expected 'full_baseline' or 'longest_chunk'")
    df_grid = (1.0 / Tgrid) / max(TESS_OVERSAMPLE, 1e-8)
    freqs = make_frequency_grid(FMIN, FMAX, df_grid, max_points=TESS_MAX_GRID_POINTS)
    print(f"[TESS] discovery grid: mode={TESS_GRID_MODE} | npts={len(tess0.t)} | Tfull={Tfull:.3f} d | Tseg={Tseg:.3f} d | Tgrid={Tgrid:.3f} d | nfreq={len(freqs)} | df={np.nanmedian(np.diff(freqs)):.6g} c/d")
    rows = []
    snapshots = []
    exclude_half_window = max((1.0 / Tfull) * TESS_DISCOVERY_EXCLUSION_MULT, np.nanmedian(np.diff(freqs)))

    for n in range(1, MAX_TESS_MODES + 1):
        vprint(1, f"[TESS] period search iteration {n}/{MAX_TESS_MODES}")
        power = lomb_scargle_power(tess, freqs)
        if not np.isfinite(power).any():
            print(f"[TESS] stopping at iter {n}: no finite periodogram values")
            break

        power_pick = np.asarray(power, dtype=float).copy()
        if rows:
            for rr in rows:
                f_prev = float(rr["f_local"])
                power_pick[np.abs(freqs - f_prev) < exclude_half_window] = np.nan
        if not np.isfinite(power_pick).any():
            print(f"[TESS] stopping at iter {n}: all remaining peaks blocked by exclusion radius {exclude_half_window:.6g} c/d")
            break

        j = int(np.nanargmax(power_pick))
        f0 = float(freqs[j])
        fit = fit_frequency_with_design(
            tess, f0=f0, T=Tfull, kfit=KFIT, fmin=FMIN, fmax=FMAX,
            baseline_matrix=make_tess_baseline_matrix(tess), n_steps=LOCAL_FIT_STEPS
        )
        snr, W = tess_local_snr_from_fit(tess, fit["best_f"], T_noise=Tseg, ks=KS_TESS)
        if (not np.isfinite(snr)) or (snr < TESS_SNR_STOP):
            print(f"[TESS] stopping at iter {n}: SNR={snr:.2f} < {TESS_SNR_STOP:.2f}")
            break
        rows.append({
            "mode": n,
            "f_local": fit["best_f"],
            "amp_local": fit["amp"],
            "phase_local": fit["phase"],
            "snr_local": snr,
            "W_local": W,
        })
        snapshots.append({
            "mode": n,
            "prefit": TimeSeries(
                t=tess.t.copy(), y=tess.y.copy(),
                yerr=None if tess.yerr is None else tess.yerr.copy(),
                name=tess.name,
                t_abs=None if tess.t_abs is None else tess.t_abs.copy(),
                group_id=None
            ),
            "fit": fit,
        })
        tess = prewhiten(tess, fit["full_model"])
        print(f"[TESS iter {n}] f={fit['best_f']:.6f} c/d | amp={fit['amp']:.4g} | SNR={snr:.2f}")

    local_df_raw = pd.DataFrame(rows)
    min_sep_global = max((1.0 / Tfull) * MIN_FREQ_SEP_MULT, 1e-8)
    local_df = dedupe_frequency_rows(local_df_raw, "f_local", score_cols=("snr_local", "amp_local", "W_local"), min_sep=min_sep_global)
    if len(local_df_raw) != len(local_df):
        print(f"[TESS] merged {len(local_df_raw)-len(local_df)} near-duplicate sequential mode(s) before global fit (min_sep={min_sep_global:.6g} c/d)")
    keep_modes = set(pd.to_numeric(local_df.get("mode", pd.Series(dtype=float)), errors="coerce").dropna().astype(int).tolist())
    snapshots = [s for s in snapshots if int(s.get("mode", -1)) in keep_modes]
    vprint(1, f"[TESS] sequential extraction found {len(local_df_raw)} modes; using {len(local_df)} distinct seed mode(s) for global fit")
    seed_freqs = local_df["f_local"].to_numpy(dtype=float) if not local_df.empty else np.array([], dtype=float)
    global_fit = fit_global_multisin(
        tess0, seed_freqs=seed_freqs,
        baseline_matrix=make_tess_baseline_matrix(tess0),
        freq_resolution=(1.0 / Tfull),
        bound_mult=GLOBAL_FREQ_BOUND_MULT
    )
    final_resid = prewhiten(tess0, global_fit["full_model"])
    freqs_plot = make_plot_frequency_grid(tess0, oversample=1.0, gap_days=1.0, max_points=PLOT_MAX_GRID_POINTS)
    start_power = lomb_scargle_power(tess0, freqs_plot)
    end_power = lomb_scargle_power(final_resid, freqs_plot)
    return local_df, global_fit, snapshots, final_resid, freqs_plot, start_power, end_power

def match_global_components_with_local(local_df: pd.DataFrame, global_fit: dict) -> pd.DataFrame:
    if local_df.empty:
        return pd.DataFrame(columns=["mode", "f_local", "amp_local", "phase_local", "snr_local", "W_local",
                                     "f", "amp", "phase", "s_coeff", "c_coeff"])
    g_rows = pd.DataFrame(global_fit["component_rows"])
    if g_rows.empty:
        out = local_df.copy()
        out["f"] = np.nan
        out["amp"] = np.nan
        out["phase"] = np.nan
        out["s_coeff"] = np.nan
        out["c_coeff"] = np.nan
        return out

    remaining = list(range(len(g_rows)))
    merged = []
    for _, lr in local_df.iterrows():
        f0 = float(lr["f_local"])
        if not remaining:
            grow = {k: np.nan for k in ["f", "amp", "phase", "s_coeff", "c_coeff"]}
        else:
            idx = min(remaining, key=lambda j: abs(float(g_rows.iloc[j]["f"]) - f0))
            grow = g_rows.iloc[idx].to_dict()
            remaining.remove(idx)
        merged.append({**lr.to_dict(), **grow})
    return pd.DataFrame(merged)


def _guided_seed_decision(f_tess: float, f_local: float, snr_local: float, local_res: float) -> tuple[bool, bool, bool, float]:
    freq_offset = float(abs(float(f_local) - float(f_tess))) if np.isfinite(f_local) else np.nan
    near_tess = bool(np.isfinite(freq_offset) and (freq_offset <= float(POL_SEED_MAX_OFFSET_MULT) * float(local_res)))
    detected = bool(np.isfinite(snr_local) and (snr_local >= float(POL_SNR_STOP)))
    seed_for_global = bool(detected or (near_tess and np.isfinite(snr_local) and (snr_local >= float(POL_SNR_SOFT_MIN))))
    return detected, seed_for_global, near_tess, freq_offset


def guided_decision_status(
    *,
    detected: bool,
    seed_for_global: bool,
    postfit_keep: bool,
    skip_reason: str = "",
) -> str:
    """Return a compact, unambiguous status for tables and diagnostics."""
    reason = str(skip_reason or "")
    if reason.startswith("below_GUIDED_POL_FMIN"):
        return "skipped_below_frequency_limit"
    if reason == "no_finite_power":
        return "rejected_no_finite_power"
    if detected:
        return "detected_retained" if postfit_keep else "detected_not_retained"
    if seed_for_global:
        return "marginal_seed_retained" if postfit_keep else "marginal_seed_pruned"
    return "rejected"

def _match_guided_rows_to_global_components(candidate_df: pd.DataFrame, global_fit: dict) -> pd.DataFrame:
    if candidate_df is None or candidate_df.empty or not global_fit.get("component_rows"):
        return pd.DataFrame()
    g_rows = pd.DataFrame(global_fit["component_rows"])
    remaining = list(range(len(g_rows)))
    out_rows = []
    for _, ar in candidate_df.iterrows():
        if not remaining:
            break
        f0 = float(ar["f_local"])
        idx = min(remaining, key=lambda j: abs(float(g_rows.iloc[j]["f"]) - f0))
        grow = g_rows.iloc[idx].to_dict()
        remaining.remove(idx)
        row = dict(ar)
        row.update({
            "f": float(grow["f"]),
            "amp": float(grow["amp"]),
            "phase": float(grow["phase"]),
            "s_coeff": float(grow["s_coeff"]),
            "c_coeff": float(grow["c_coeff"]),
        })
        out_rows.append(row)
    return pd.DataFrame(out_rows)

def _compute_guided_postfit_local_snr(pol0: TimeSeries, trend_cfg: NightTrendConfig, final_df: pd.DataFrame) -> pd.DataFrame:
    if final_df is None or final_df.empty:
        return final_df if final_df is not None else pd.DataFrame()

    baseline0 = make_pol_baseline_matrix(pol0, trend_cfg)
    t = np.asarray(pol0.t, dtype=float)

    component_models = []
    for _, row in final_df.iterrows():
        component_models.append(
            component_signal(t, float(row["f"]), float(row["s_coeff"]), float(row["c_coeff"]))
        )
    if component_models:
        total_signal = np.sum(np.vstack(component_models), axis=0)
    else:
        total_signal = np.zeros_like(t, dtype=float)

    Tfull_pol = max(compute_T_full(pol0), 1e-8)
    post_snr = []
    post_W = []
    for i, (_, row) in enumerate(final_df.iterrows()):
        y_keep = np.asarray(pol0.y, dtype=float) - (total_signal - component_models[i])
        ts_keep = TimeSeries(
            t=pol0.t.copy(),
            y=y_keep,
            yerr=None if pol0.yerr is None else pol0.yerr.copy(),
            name=pol0.name,
            t_abs=None if pol0.t_abs is None else pol0.t_abs.copy(),
            group_id=None if pol0.group_id is None else pol0.group_id.copy(),
        )
        snr, W = pol_local_snr_from_fit(
            ts_keep, float(row["f"]), T_noise=Tfull_pol, ks=POL_LOCAL_NOISE_KS,
            baseline_matrix=baseline0
        )
        post_snr.append(float(snr) if np.isfinite(snr) else np.nan)
        post_W.append(float(W) if np.isfinite(W) else np.nan)

    out = final_df.copy()
    out["postfit_snr_local"] = post_snr
    out["postfit_W_local"] = post_W
    out["postfit_keep"] = out["detected"].astype(bool) | (
        np.isfinite(pd.to_numeric(out["postfit_snr_local"], errors="coerce")) &
        (pd.to_numeric(out["postfit_snr_local"], errors="coerce") >= float(POL_POSTFIT_SNR_MIN))
    )
    return out

def search_guided_channel(pol0: TimeSeries, tess_table: pd.DataFrame, tess_freq_resolution: float,
                          trend_cfg: NightTrendConfig) -> tuple[
                              pd.DataFrame, pd.DataFrame, dict, list[dict], TimeSeries,
                              np.ndarray, np.ndarray, np.ndarray, list[dict],
                              np.ndarray, np.ndarray, np.ndarray,
                          ]:
    pol = TimeSeries(
        t=pol0.t.copy(), y=pol0.y.copy(),
        yerr=None if pol0.yerr is None else pol0.yerr.copy(),
        name=pol0.name, t_abs=None if pol0.t_abs is None else pol0.t_abs.copy(),
        group_id=None if pol0.group_id is None else pol0.group_id.copy()
    )
    Tfull_pol = max(compute_T_full(pol0), 1e-8)
    template_df = tess_table.copy().sort_values("mode").head(max(0, int(MAX_POL_MODES))).reset_index(drop=True)
    local_res = max(float(tess_freq_resolution), 1.0 / Tfull_pol)
    search_half_window = POL_SEARCH_WINDOW_MULT * local_res
    df_local = max(min(float(tess_freq_resolution), 1.0 / Tfull_pol) / max(POL_OVERSAMPLE, 1e-8), 1e-8)
    print(f"[{pol0.name}] guided search: npts={len(pol0.t)} | Tfull={Tfull_pol:.3f} d | template_modes={len(template_df)} | half_window={search_half_window:.6g} c/d | df_local={df_local:.6g} c/d")

    baseline0 = make_pol_baseline_matrix(pol0, trend_cfg)
    search_rows = []
    snapshots = []
    # Keep the exact spectrum used at each sequential search step.  Rebuilding
    # these later from the original or final residual can hide a peak that only
    # becomes visible after earlier detected modes have been removed.
    local_diags = []

    guided_pol_fmin = None if GUIDED_POL_FMIN is None else float(GUIDED_POL_FMIN)

    for i_tr, (_, tr) in enumerate(template_df.iterrows(), start=1):
        mode_n = int(tr["mode"])
        f_tess = float(tr["f"])
        if np.isfinite(guided_pol_fmin) and (f_tess < guided_pol_fmin):
            search_rows.append({
                "tess_mode": mode_n, "f_tess": f_tess,
                "detected": False, "seed_for_global": False, "near_tess": False,
                "freq_offset": np.nan,
                "f_local": np.nan, "amp_local": np.nan, "phase_local": np.nan,
                "snr_local": np.nan, "W_local": np.nan,
                "skip_reason": f"below_GUIDED_POL_FMIN({guided_pol_fmin:g})",
            })
            print(f"[{pol0.name}] skipping TESS mode {i_tr}/{len(template_df)} (mode={mode_n}) at f={f_tess:.6f} c/d < GUIDED_POL_FMIN={guided_pol_fmin:g}")
            continue

        print(f"[{pol0.name}] testing TESS mode {i_tr}/{len(template_df)} (mode={mode_n}) near f={f_tess:.6f} c/d")
        local_freqs = make_local_frequency_grid(
            f_tess, search_half_window, df_local,
            fmin=max(FMIN, guided_pol_fmin if np.isfinite(guided_pol_fmin) else FMIN),
            fmax=FMAX, max_points=POL_LOCAL_MAX_POINTS
        )
        baseline_now = make_pol_baseline_matrix(pol, trend_cfg)
        power_raw = nuisance_periodogram(pol, local_freqs, baseline_matrix=baseline_now)
        power_for_selection, smoothing_info = smooth_polarimetry_periodogram(
            local_freqs,
            power_raw,
            Tfull_pol,
        )
        if not np.isfinite(power_for_selection).any():
            search_rows.append({
                "tess_mode": mode_n, "f_tess": f_tess,
                "detected": False, "seed_for_global": False, "near_tess": False,
                "freq_offset": np.nan,
                "f_local": np.nan, "amp_local": np.nan, "phase_local": np.nan,
                "snr_local": np.nan, "W_local": np.nan,
                "skip_reason": "no_finite_power",
            })
            local_diags.append({
                "search_order": int(i_tr),
                "tess_mode": mode_n,
                "f_tess": f_tess,
                "freqs": local_freqs.copy(),
                "power_iteration_raw": power_raw.copy(),
                "power_iteration_smoothed": power_for_selection.copy(),
                "f_candidate": np.nan,
                "f_local": np.nan,
                "detected": False,
                "seed_for_global": False,
                "near_tess": False,
                "freq_offset": np.nan,
                "snr_local": np.nan,
                "W_local": np.nan,
                "skip_reason": "no_finite_power",
                "smoothing": dict(smoothing_info),
            })
            print(f"[{pol0.name} vs TESS mode {mode_n}] no finite local periodogram values")
            continue

        # Smoothing chooses the starting candidate only.  The local nonlinear
        # refinement below evaluates the original time-series data directly.
        jloc = int(np.nanargmax(power_for_selection))
        f0 = float(local_freqs[jloc])
        fit = fit_frequency_with_design(
            pol, f0=f0, T=Tfull_pol, kfit=KFIT, fmin=FMIN, fmax=FMAX,
            baseline_matrix=baseline_now, n_steps=LOCAL_FIT_STEPS,
            hard_fmin=float(local_freqs[0]), hard_fmax=float(local_freqs[-1])
        )
        snr, W = pol_local_snr_from_fit(
            pol, fit["best_f"], T_noise=Tfull_pol, ks=POL_LOCAL_NOISE_KS,
            baseline_matrix=baseline_now
        )
        detected, seed_for_global, near_tess, freq_offset = _guided_seed_decision(
            f_tess=f_tess, f_local=fit["best_f"], snr_local=snr, local_res=local_res
        )

        if detected:
            skip_reason = ""
        elif seed_for_global:
            skip_reason = "marginal_seed_for_global_fit"
        else:
            skip_reason = "snr_below_threshold"

        search_rows.append({
            "tess_mode": mode_n,
            "f_tess": f_tess,
            "detected": detected,
            "seed_for_global": seed_for_global,
            "near_tess": near_tess,
            "freq_offset": freq_offset,
            "f_candidate_selection_max": f0,
            "candidate_to_local_fit_offset_cpd": float(abs(float(fit["best_f"]) - f0)),
            "f_local": float(fit["best_f"]),
            "amp_local": float(fit["amp"]),
            "phase_local": float(fit["phase"]),
            "snr_local": snr,
            "W_local": W,
            "skip_reason": skip_reason,
            "pol_smoothing_enabled": bool(smoothing_info["enabled"]),
            "pol_smoothing_applied": bool(smoothing_info["applied"]),
            "pol_smoothing_kernel": smoothing_info["kernel"],
            "pol_smoothing_width_res_elements": smoothing_info["width_resolution_elements"],
            "pol_smoothing_resolution_cpd": smoothing_info["resolution_cpd"],
            "pol_smoothing_effective_width_cpd": smoothing_info["effective_width_cpd"],
        })

        local_diags.append({
            "search_order": int(i_tr),
            "tess_mode": mode_n,
            "f_tess": f_tess,
            "freqs": local_freqs.copy(),
            "power_iteration_raw": power_raw.copy(),
            "power_iteration_smoothed": power_for_selection.copy(),
            "f_candidate": f0,
            "f_local": float(fit["best_f"]),
            "detected": bool(detected),
            "seed_for_global": bool(seed_for_global),
            "near_tess": bool(near_tess),
            "freq_offset": float(freq_offset),
            "snr_local": float(snr) if np.isfinite(snr) else np.nan,
            "W_local": float(W) if np.isfinite(W) else np.nan,
            "skip_reason": skip_reason,
            "smoothing": dict(smoothing_info),
        })

        if seed_for_global:
            snapshots.append({
                "tess_mode": mode_n,
                "prefit": TimeSeries(
                    t=pol.t.copy(), y=pol.y.copy(),
                    yerr=None if pol.yerr is None else pol.yerr.copy(),
                    name=pol.name,
                    t_abs=None if pol.t_abs is None else pol.t_abs.copy(),
                    group_id=None if pol.group_id is None else pol.group_id.copy()
                ),
                "fit": fit,
            })

        if detected:
            pol = prewhiten(pol, fit["full_model"])
            print(f"[{pol0.name} vs TESS mode {mode_n}] detected f={fit['best_f']:.6f} c/d | amp={fit['amp']:.4g} | SNR={snr:.2f}")
        elif seed_for_global:
            print(f"[{pol0.name} vs TESS mode {mode_n}] marginal seed kept for global fit | f={fit['best_f']:.6f} c/d | amp={fit['amp']:.4g} | SNR={snr:.2f} | Δf={freq_offset:.6g}")
        else:
            print(f"[{pol0.name} vs TESS mode {mode_n}] no detection | best local SNR={snr:.2f}")

    search_df = pd.DataFrame(search_rows)
    min_sep_global = max((1.0 / Tfull_pol) * MIN_FREQ_SEP_MULT, float(tess_freq_resolution) * MIN_FREQ_SEP_MULT, 1e-8)

    accepted_raw = search_df.loc[search_df["seed_for_global"]].copy().reset_index(drop=True)
    if not accepted_raw.empty:
        accepted_raw["detected_score"] = accepted_raw["detected"].astype(int)
        accepted_raw["template_closeness"] = -pd.to_numeric(accepted_raw["freq_offset"], errors="coerce").fillna(np.inf)
    accepted = dedupe_frequency_rows(
        accepted_raw, "f_local",
        score_cols=("detected_score", "snr_local", "amp_local", "W_local", "template_closeness"),
        min_sep=min_sep_global
    ) if not accepted_raw.empty else accepted_raw.copy()
    if len(accepted_raw) != len(accepted):
        print(f"[{pol0.name}] merged {len(accepted_raw)-len(accepted)} near-duplicate guided candidate(s) before global fit (min_sep={min_sep_global:.6g} c/d)")
    print(f"[{pol0.name}] seeding global fit with {len(accepted)} guided candidate(s)")

    global_fit = fit_global_multisin(
        pol0,
        seed_freqs=accepted["f_local"].to_numpy(dtype=float) if not accepted.empty else np.array([], dtype=float),
        baseline_matrix=baseline0,
        freq_resolution=(1.0 / Tfull_pol),
        bound_mult=GLOBAL_FREQ_BOUND_MULT
    )

    prelim_df = _match_guided_rows_to_global_components(accepted, global_fit)
    prelim_df = _compute_guided_postfit_local_snr(pol0, trend_cfg, prelim_df)

    kept_seed_df = prelim_df.loc[prelim_df["postfit_keep"]].copy().reset_index(drop=True) if not prelim_df.empty else prelim_df
    if not prelim_df.empty and len(kept_seed_df) != len(prelim_df):
        print(f"[{pol0.name}] pruned {len(prelim_df)-len(kept_seed_df)} marginal guided candidate(s) after post-fit local-SNR check")

    if len(kept_seed_df) != len(accepted):
        global_fit = fit_global_multisin(
            pol0,
            seed_freqs=kept_seed_df["f"].to_numpy(dtype=float) if not kept_seed_df.empty else np.array([], dtype=float),
            baseline_matrix=baseline0,
            freq_resolution=(1.0 / Tfull_pol),
            bound_mult=GLOBAL_FREQ_BOUND_MULT
        )

    final_df = _match_guided_rows_to_global_components(kept_seed_df, global_fit)
    final_df = _compute_guided_postfit_local_snr(pol0, trend_cfg, final_df)
    final_resid = prewhiten(pol0, global_fit["full_model"])

    keep_modes = set()
    if final_df is not None and not final_df.empty:
        keep_modes = set(pd.to_numeric(final_df["tess_mode"], errors="coerce").dropna().astype(int).tolist())
    if "postfit_keep" not in search_df.columns:
        search_df["postfit_keep"] = False
    search_df["postfit_keep"] = search_df["tess_mode"].isin(keep_modes)

    global_frequency_by_mode = {}
    global_postfit_snr_by_mode = {}
    if final_df is not None and not final_df.empty:
        for _, row in final_df.iterrows():
            mode = int(row["tess_mode"])
            global_frequency_by_mode[mode] = float(row["f"])
            global_postfit_snr_by_mode[mode] = float(row.get("postfit_snr_local", np.nan))
    search_df["f_global"] = search_df["tess_mode"].map(global_frequency_by_mode)
    search_df["postfit_snr_global"] = search_df["tess_mode"].map(global_postfit_snr_by_mode)
    search_df["decision_status"] = [
        guided_decision_status(
            detected=bool(row.get("detected", False)),
            seed_for_global=bool(row.get("seed_for_global", False)),
            postfit_keep=bool(row.get("postfit_keep", False)),
            skip_reason=str(row.get("skip_reason", "")),
        )
        for _, row in search_df.iterrows()
    ]

    search_by_mode = {
        int(row["tess_mode"]): row
        for _, row in search_df.iterrows()
    }
    for diag in local_diags:
        row = search_by_mode.get(int(diag["tess_mode"]))
        if row is None:
            continue
        diag["postfit_keep"] = bool(row.get("postfit_keep", False))
        diag["f_global"] = float(row["f_global"]) if pd.notna(row.get("f_global", np.nan)) else np.nan
        diag["postfit_snr_global"] = (
            float(row["postfit_snr_global"])
            if pd.notna(row.get("postfit_snr_global", np.nan))
            else np.nan
        )
        diag["decision_status"] = str(row.get("decision_status", "rejected"))

    freqs_plot = make_plot_frequency_grid(pol0, oversample=1.0, gap_days=trend_cfg.gap_days, max_points=PLOT_MAX_GRID_POINTS)
    start_power = nuisance_periodogram(pol0, freqs_plot, baseline_matrix=baseline0)
    end_power = nuisance_periodogram(final_resid, freqs_plot, baseline_matrix=make_pol_baseline_matrix(final_resid, trend_cfg))

    freqs_plot_full = make_fullbaseline_plot_frequency_grid(
        pol0, oversample=POL_FULL_RANGE_PLOT_OVERSAMPLE, max_points=POL_FULL_RANGE_PLOT_POINTS
    )
    start_power_full = nuisance_periodogram(pol0, freqs_plot_full, baseline_matrix=baseline0)
    end_power_full = nuisance_periodogram(final_resid, freqs_plot_full, baseline_matrix=make_pol_baseline_matrix(final_resid, trend_cfg))

    return (search_df, final_df, global_fit, snapshots, final_resid,
            freqs_plot, start_power, end_power, local_diags,
            freqs_plot_full, start_power_full, end_power_full)


def _series_btjd_abs(ts: TimeSeries) -> np.ndarray:
    """Return absolute time in the BTJD convention when it can be inferred."""
    values = ts.t if ts.t_abs is None else np.asarray(ts.t_abs, dtype=float)
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size and float(np.nanmedian(finite)) > 1.0e6:
        values = values - 2457000.0
    return values


def _series_local_zero_btjd(ts: TimeSeries) -> float:
    absolute = _series_btjd_abs(ts)
    offset = absolute - np.asarray(ts.t, dtype=float)
    finite = offset[np.isfinite(offset)]
    return float(np.nanmedian(finite)) if finite.size else 0.0


def resolve_phase_reference_btjd(*series: TimeSeries) -> float:
    """Resolve one shared phase-reference epoch for TESS, q, and u."""
    mode = str(PHASE_ZERO_MODE).strip().lower()
    if mode == "btjd_zero":
        return 0.0
    if mode == "custom_btjd":
        value = float(PHASE_ZERO_BTJD)
        if not np.isfinite(value):
            raise ValueError("Custom phase-zero BTJD must be finite.")
        return value
    if mode != "local_start":
        raise ValueError(
            "PHASE_ZERO_MODE must be 'local_start', 'btjd_zero', or 'custom_btjd'."
        )
    starts = []
    for ts in series:
        values = _series_btjd_abs(ts)
        finite = values[np.isfinite(values)]
        if finite.size:
            starts.append(float(np.nanmin(finite)))
    return min(starts) if starts else 0.0


def _quadrature_channel_global_fit(
    pol0: TimeSeries,
    fit: dict,
    channel: str,
    rows: list[dict],
) -> dict:
    if channel == "q":
        signal = np.asarray(fit["signal_q"], dtype=float)
        baseline = np.asarray(fit["baseline_q"], dtype=float)
        model = np.asarray(fit["model_q"], dtype=float)
        resid = np.asarray(fit["resid_q"], dtype=float)
    else:
        signal = np.asarray(fit["signal_u"], dtype=float)
        baseline = np.asarray(fit["baseline_u"], dtype=float)
        model = np.asarray(fit["model_u"], dtype=float)
        resid = np.asarray(fit["resid_u"], dtype=float)
    weights = signal_weights(pol0)
    return {
        "success": bool(fit.get("success", False)),
        "message": str(fit.get("message", "")),
        "freqs": np.asarray(fit["frequencies"], dtype=float),
        "rss": float(np.sum(weights * np.square(resid))),
        "full_model": model,
        "signal_model": signal,
        "baseline_model": baseline,
        "resid": resid,
        "component_rows": rows,
        "quadrature_mode": str(fit["mode"]),
        "quadrature_signs": np.asarray(fit["signs"], dtype=int),
    }


def apply_guided_quadrature_refit(
    tess0: TimeSeries,
    tess_table: pd.DataFrame,
    channel_states: dict[str, dict],
    trend_cfg: NightTrendConfig,
) -> tuple[dict[str, dict], pd.DataFrame]:
    """Replace independent q/u final fits with one constrained global fit.

    The existing per-channel periodogram searches remain the candidate
    discovery stage.  Their accepted TESS-mode union seeds this authoritative
    fit to the original, unsmoothed q and u time series.
    """
    mode = normalize_quadrature_mode(QU_PHASE_MODE)
    if mode == "free":
        return channel_states, pd.DataFrame()
    if "q" not in channel_states or "u" not in channel_states:
        print(
            "[WARN] q/u quadrature was requested, but both q and u were not "
            "selected and available; retaining independent channel fits."
        )
        return channel_states, pd.DataFrame()

    accepted_modes = set()
    for channel in ("q", "u"):
        table = channel_states[channel].get("final_df")
        if table is not None and not table.empty and "tess_mode" in table.columns:
            accepted_modes.update(
                pd.to_numeric(table["tess_mode"], errors="coerce")
                .dropna()
                .astype(int)
                .tolist()
            )
    if not accepted_modes:
        print(
            "[q/u quadrature] No independently retained q/u candidates; "
            "there is no constrained final model to fit."
        )
        return channel_states, pd.DataFrame()

    tess_by_mode = {
        int(row["mode"]): row for _, row in tess_table.iterrows()
    }
    accepted_modes = sorted(mode for mode in accepted_modes if mode in tess_by_mode)
    if not accepted_modes:
        return channel_states, pd.DataFrame()
    seed_frequencies = np.array(
        [float(tess_by_mode[mode_n]["f"]) for mode_n in accepted_modes],
        dtype=float,
    )

    q_state = channel_states["q"]
    u_state = channel_states["u"]
    q0 = q_state["pol0"]
    u0 = u_state["pol0"]
    baseline_q = make_pol_baseline_matrix(q0, trend_cfg)
    baseline_u = make_pol_baseline_matrix(u0, trend_cfg)

    print(
        f"[q/u quadrature] fitting {len(seed_frequencies)} Guided component(s) "
        f"at the accepted TESS frequencies | mode={mode}"
    )
    fit = fit_quadrature_multisin(
        q_time=q0.t,
        q_values=q0.y,
        q_weights=signal_weights(q0),
        q_baseline=baseline_q,
        u_time=u0.t,
        u_values=u0.y,
        u_weights=signal_weights(u0),
        u_baseline=baseline_u,
        seed_frequencies=seed_frequencies,
        mode=mode,
        optimize_frequencies=False,
        max_nfev=QUADRATURE_MAX_NFEV,
    )

    reference_btjd = resolve_phase_reference_btjd(tess0, q0, u0)
    zero_tess = _series_local_zero_btjd(tess0)
    zero_q = _series_local_zero_btjd(q0)
    zero_u = _series_local_zero_btjd(u0)
    combined_rows = []
    q_rows = []
    u_rows = []

    for component, fit_row in enumerate(fit["component_rows"], start=1):
        frequency = float(fit_row["frequency_cpd"])
        nearest = int(np.argmin(np.abs(seed_frequencies - frequency)))
        tess_mode = int(accepted_modes[nearest])
        tess_row = tess_by_mode[tess_mode]
        phase_tess_local = float(tess_row.get("phase", np.nan))
        phase_tess_common = (
            phase_at_reference(
                phase_tess_local,
                frequency,
                zero_tess,
                reference_btjd,
            )
            if np.isfinite(phase_tess_local)
            else np.nan
        )
        phase_q_common = phase_at_reference(
            float(fit_row["phase_q_rad"]),
            frequency,
            zero_q,
            reference_btjd,
        )
        phase_u_common = phase_at_reference(
            float(fit_row["phase_u_rad"]),
            frequency,
            zero_u,
            reference_btjd,
        )
        snr_q, w_q = pol_local_snr_from_fit(
            q0,
            frequency,
            T_noise=max(compute_T_full(q0), 1e-8),
            ks=POL_LOCAL_NOISE_KS,
            baseline_matrix=baseline_q,
        )
        snr_u, w_u = pol_local_snr_from_fit(
            u0,
            frequency,
            T_noise=max(compute_T_full(u0), 1e-8),
            ks=POL_LOCAL_NOISE_KS,
            baseline_matrix=baseline_u,
        )
        common = {
            **fit_row,
            "tess_mode": tess_mode,
            "f_tess": float(tess_row["f"]),
            "phase_reference_btjd": float(reference_btjd),
            "phase_tess_at_reference_rad": phase_tess_common,
            "phase_q_at_reference_rad": phase_q_common,
            "phase_u_at_reference_rad": phase_u_common,
            "phase_q_minus_tess_rad": (
                float(wrap_phase_radians(phase_q_common - phase_tess_common))
                if np.isfinite(phase_tess_common)
                else np.nan
            ),
            "phase_u_minus_tess_rad": (
                float(wrap_phase_radians(phase_u_common - phase_tess_common))
                if np.isfinite(phase_tess_common)
                else np.nan
            ),
            "snr_q": float(snr_q),
            "snr_u": float(snr_u),
            "W_q": float(w_q),
            "W_u": float(w_u),
            "quadrature_mode": mode,
            "constrained_weighted_rss_total": float(fit["weighted_rss"]),
            "unconstrained_weighted_rss_total": float(
                fit["unconstrained_weighted_rss_total"]
            ),
        }
        combined_rows.append(common)

        for channel, amplitude_key, phase_key, snr, weight, phase_common in (
            ("q", "amp_q", "phase_q_rad", snr_q, w_q, phase_q_common),
            ("u", "amp_u", "phase_u_rad", snr_u, w_u, phase_u_common),
        ):
            amplitude = float(fit_row[amplitude_key])
            phase = float(fit_row[phase_key])
            channel_row = {
                "mode": component,
                "tess_mode": tess_mode,
                "f_tess": float(tess_row["f"]),
                "f": frequency,
                "amp": amplitude,
                "phase": phase,
                "s_coeff": float(amplitude * np.cos(phase)),
                "c_coeff": float(amplitude * np.sin(phase)),
                "postfit_snr_local": float(snr),
                "postfit_W_local": float(weight),
                "postfit_keep": True,
                "quadrature_mode": mode,
                "quadrature_sign": int(fit_row["quadrature_sign"]),
                "quadrature_relation": str(fit_row["quadrature_relation"]),
                "phase_diff_u_minus_q_rad": float(
                    fit_row["phase_diff_u_minus_q_rad"]
                ),
                "unconstrained_phase_diff_u_minus_q_rad": float(
                    fit_row["unconstrained_phase_diff_u_minus_q_rad"]
                ),
                "phase_reference_btjd": float(reference_btjd),
                "phase_at_reference_rad": float(phase_common),
                "phase_minus_tess_rad": (
                    float(wrap_phase_radians(phase_common - phase_tess_common))
                    if np.isfinite(phase_tess_common)
                    else np.nan
                ),
            }
            source_search = channel_states[channel]["search_df"]
            source = source_search.loc[source_search["tess_mode"] == tess_mode]
            if not source.empty:
                source = source.iloc[0]
                for name in (
                    "f_local",
                    "amp_local",
                    "phase_local",
                    "snr_local",
                    "W_local",
                    "detected",
                    "seed_for_global",
                    "near_tess",
                    "freq_offset",
                ):
                    if name in source:
                        channel_row[name] = source[name]
            (q_rows if channel == "q" else u_rows).append(channel_row)

    quadrature_table = pd.DataFrame(combined_rows).sort_values(
        "tess_mode"
    ).reset_index(drop=True)

    for channel, rows in (("q", q_rows), ("u", u_rows)):
        state = channel_states[channel]
        pol0 = state["pol0"]
        component_rows = sorted(rows, key=lambda row: row["f"])
        state["final_df"] = pd.DataFrame(component_rows).reset_index(drop=True)
        state["global_fit"] = _quadrature_channel_global_fit(
            pol0, fit, channel, component_rows
        )
        residual_values = (
            np.asarray(fit["resid_q"], dtype=float)
            if channel == "q"
            else np.asarray(fit["resid_u"], dtype=float)
        )
        state["final_resid"] = TimeSeries(
            t=pol0.t.copy(),
            y=residual_values,
            yerr=None if pol0.yerr is None else pol0.yerr.copy(),
            name=pol0.name,
            t_abs=None if pol0.t_abs is None else pol0.t_abs.copy(),
            group_id=None if pol0.group_id is None else pol0.group_id.copy(),
        )
        state["Ppo_end"] = nuisance_periodogram(
            state["final_resid"],
            state["freqs_pol"],
            baseline_matrix=make_pol_baseline_matrix(state["final_resid"], trend_cfg),
        )
        state["Ppo_end_full"] = nuisance_periodogram(
            state["final_resid"],
            state["freqs_pol_full"],
            baseline_matrix=make_pol_baseline_matrix(state["final_resid"], trend_cfg),
        )

        result_by_mode = {
            int(row["tess_mode"]): row for row in component_rows
        }
        search_df = state["search_df"].copy()
        search_df["quadrature_mode"] = mode
        search_df["quadrature_postfit_keep"] = search_df["tess_mode"].isin(
            result_by_mode
        )
        search_df["quadrature_phase_rad"] = search_df["tess_mode"].map(
            {
                tess_mode: float(row["phase"])
                for tess_mode, row in result_by_mode.items()
            }
        )
        search_df["quadrature_sign"] = search_df["tess_mode"].map(
            {
                tess_mode: int(row["quadrature_sign"])
                for tess_mode, row in result_by_mode.items()
            }
        )
        search_df["f_global"] = search_df["tess_mode"].map(
            {
                tess_mode: float(row["f"])
                for tess_mode, row in result_by_mode.items()
            }
        )
        search_df["postfit_snr_global"] = search_df["tess_mode"].map(
            {
                tess_mode: float(row["postfit_snr_local"])
                for tess_mode, row in result_by_mode.items()
            }
        )
        search_df["postfit_keep"] = search_df["quadrature_postfit_keep"]
        search_df["decision_status"] = np.where(
            search_df["quadrature_postfit_keep"],
            "retained_quadrature_global_fit",
            "not_retained_in_quadrature_global_fit",
        )
        state["search_df"] = search_df
        for diagnostic in state["guided_diags"]:
            row = result_by_mode.get(int(diagnostic["tess_mode"]))
            if row is not None:
                diagnostic["f_global"] = float(row["f"])
                diagnostic["postfit_keep"] = True
                diagnostic["postfit_snr_global"] = float(
                    row["postfit_snr_local"]
                )
                diagnostic["decision_status"] = "retained_quadrature_global_fit"
                diagnostic["quadrature_mode"] = mode
                diagnostic["quadrature_sign"] = int(row["quadrature_sign"])

    return channel_states, quadrature_table

# ----------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------

def _norm_spec(P: np.ndarray, q: float = 99.0) -> np.ndarray:
    P = np.asarray(P, dtype=float)
    finite = np.isfinite(P)
    if not finite.any():
        return P
    scale = np.nanpercentile(P[finite], q)
    if not np.isfinite(scale) or scale <= 0:
        scale = np.nanmax(P[finite])
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0
    return P / scale


def _norm_spec_shared(reference: np.ndarray, *arrays: np.ndarray, q: float = 99.0) -> tuple[np.ndarray, ...]:
    """Normalize several spectra with a scale derived from one reference."""
    ref = np.asarray(reference, dtype=float)
    finite = ref[np.isfinite(ref)]
    scale = float(np.nanpercentile(finite, q)) if finite.size else np.nan
    if not np.isfinite(scale) or scale <= 0:
        scale = float(np.nanmax(finite)) if finite.size else 1.0
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0
    return tuple(np.asarray(values, dtype=float) / scale for values in (reference, *arrays))


def normalize_tess_spectrum_background_scale(value: str) -> str:
    """Validate the user-facing TESS-spectrum background scaling mode."""
    mode = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {"right": "right_axis", "secondary_axis": "right_axis", "panel": "normalized"}
    mode = aliases.get(mode, mode)
    if mode not in {"right_axis", "normalized"}:
        raise ValueError(
            "TESS spectrum background scale must be 'right_axis' or 'normalized'."
        )
    return mode


def add_tess_spectrum_background(
    ax,
    tess_frequencies: np.ndarray,
    tess_power: np.ndarray,
    *,
    log_y: bool = False,
):
    """Add the optional faint TESS spectrum to one polarimetry spectrum axis."""
    if not SHOW_TESS_SPECTRUM_BACKGROUND:
        return None
    mode = normalize_tess_spectrum_background_scale(TESS_SPECTRUM_BACKGROUND_SCALE)
    frequencies = np.asarray(tess_frequencies, dtype=float)
    power = np.asarray(tess_power, dtype=float)
    finite = np.isfinite(frequencies) & np.isfinite(power)
    if log_y:
        finite &= power > 0
    if np.count_nonzero(finite) < 2:
        return None
    frequencies = frequencies[finite]
    power = power[finite]
    if mode == "normalized":
        scale = float(np.nanpercentile(power, 99.0))
        if not np.isfinite(scale) or scale <= 0:
            scale = float(np.nanmax(power))
        if not np.isfinite(scale) or scale <= 0:
            scale = 1.0
        power = power / scale

    background_axis = ax.twinx()
    background_axis.set_zorder(0)
    ax.set_zorder(1)
    ax.patch.set_visible(False)
    background_axis.plot(
        frequencies,
        power,
        color="#737373",
        linewidth=0.85,
        alpha=0.22,
        zorder=0,
    )
    if log_y:
        background_axis.set_yscale("log")
    label = "TESS normalized power" if mode == "normalized" else "TESS power"
    background_axis.set_ylabel(label, color="#737373", alpha=0.60, fontsize=8)
    background_axis.tick_params(
        axis="y", colors="#737373", labelsize=7, length=2, width=0.6
    )
    background_axis.spines["right"].set_color("#A6A6A6")
    background_axis.spines["right"].set_alpha(0.45)
    return background_axis


def guided_diagnostic_marker_specs(diag: dict) -> list[dict]:
    """Describe the frequency markers appropriate for one Guided decision.

    Rejected local best fits intentionally receive no detection marker.  This
    keeps a merely finite optimizer result from being presented as evidence of
    a polarimetric detection.
    """
    specs = []

    def add(value, label, **style):
        if value is not None and np.isfinite(value):
            specs.append({"frequency": float(value), "label": label, **style})

    add(diag.get("f_tess"), "TESS template", color="0.30", linestyle="-", alpha=0.65)
    smoothing = diag.get("smoothing", {}) or {}
    candidate_label = "smoothed-spectrum maximum" if smoothing.get("applied", False) else "selection maximum"
    add(diag.get("f_candidate"), candidate_label, color="#E69F00", linestyle="-.", alpha=0.90)

    if bool(diag.get("detected", False)):
        add(diag.get("f_local"), "local detection", color="#009E73", linestyle=":", alpha=0.95)
    elif bool(diag.get("seed_for_global", False)):
        add(diag.get("f_local"), "marginal local seed", color="#CC79A7", linestyle=":", alpha=0.95)

    add(diag.get("f_global"), "final global fit", color="#6A3D9A", linestyle="--", alpha=0.95)
    return specs


def guided_diagnostic_title(pol_name: str, diag: dict) -> str:
    """Build a status-rich title for a per-template diagnostic panel."""
    status = str(diag.get("decision_status", "rejected")).replace("_", " ")
    snr = diag.get("snr_local", np.nan)
    offset = diag.get("freq_offset", np.nan)
    smoothing = diag.get("smoothing", {}) or {}
    if smoothing.get("enabled", False):
        smooth_state = "applied" if smoothing.get("applied", False) else "not applied"
        smooth_text = (
            f"{smoothing.get('kernel', 'unknown')} "
            f"{float(smoothing.get('width_resolution_elements', np.nan)):g} res. elem. ({smooth_state})"
        )
    else:
        smooth_text = "off"
    snr_text = f"{float(snr):.2f}" if np.isfinite(snr) else "n/a"
    offset_text = f"{float(offset):.6g}" if np.isfinite(offset) else "n/a"
    return (
        f"{pol_name} iteration spectrum | TESS mode {int(diag['tess_mode'])} | status: {status}\n"
        f"local SNR={snr_text} | |Δf(local−TESS)|={offset_text} c/d | smoothing: {smooth_text}"
    )

def phase_fold(t: np.ndarray, f: float, t_ref: float = 0.0) -> np.ndarray:
    return ((t - t_ref) * f) % 1.0

def phase_bin_medians(phase: np.ndarray, y: np.ndarray, nbin: int = 16) -> tuple[np.ndarray, np.ndarray]:
    phase = np.asarray(phase, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(phase) & np.isfinite(y)
    phase = phase[m]
    y = y[m]
    if phase.size == 0 or nbin < 2:
        return np.array([], dtype=float), np.array([], dtype=float)
    edges = np.linspace(0.0, 1.0, int(nbin) + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    meds = np.full(int(nbin), np.nan, dtype=float)
    for i in range(int(nbin)):
        if i == int(nbin) - 1:
            sel = (phase >= edges[i]) & (phase <= edges[i + 1])
        else:
            sel = (phase >= edges[i]) & (phase < edges[i + 1])
        if np.any(sel):
            meds[i] = float(np.nanmedian(y[sel]))
    ok = np.isfinite(meds)
    return centers[ok], meds[ok]

def plot_tess_timeseries_and_spectra(tess0: TimeSeries, final_resid: TimeSeries, freqs: np.ndarray,
                                     Pstart: np.ndarray, Pend: np.ndarray, mode_table: pd.DataFrame,
                                     outdir: Path, show_plots_inline: bool = False, file_prefix: str = ""):
    outdir.mkdir(parents=True, exist_ok=True)

    C_START = "#7EC8FF"
    C_END = "#000000"
    fig, ax = plt.subplots(figsize=(11, 4.8))
    y0 = tess0.y - np.nanmedian(tess0.y)
    y1 = final_resid.y - np.nanmedian(final_resid.y)
    ax.scatter(tess0.t, y0, s=5, alpha=0.50, label="tess start", color=C_START)
    ax.scatter(final_resid.t, y1, s=5, alpha=0.55, label="tess end (resid)", color=C_END)
    ax.set_xlabel("Time [days]")
    ax.set_ylabel("TESS value (median-subtracted)")
    set_robust_time_ylim(ax, y0, y1, central_pct=98.0, pad_frac=0.12)
    ax.set_title("TESS time series")
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(outdir / prefixed_output_name(file_prefix, "timeseries_start_end.png"), dpi=180)
    if show_plots_inline:
        plt.show()
    plt.close(fig)

    P0 = _norm_spec(Pstart)
    P1 = _norm_spec(Pend)
    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.plot(freqs, P0, lw=0.9, color=C_START, label="tess start")
    ax.plot(freqs, P1 + 1.2, lw=0.9, color=C_END, linestyle="--", label="tess end (offset)")
    if not mode_table.empty:
        ytop = np.nanmax(P1 + 1.2)
        for _, r in mode_table.iterrows():
            f = float(r["f"])
            m = int(r["mode"])
            ax.axvline(f, lw=0.8, alpha=0.35, color="0.3")
            ax.text(f, ytop, str(m), rotation=90, va="top", ha="center", fontsize=8)
    ax.set_xlabel("Frequency [c/d]")
    ax.set_ylabel("Normalized power + offset")
    ax.set_title("TESS spectra")
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(outdir / prefixed_output_name(file_prefix, "spectra_start_end.png"), dpi=180)
    if show_plots_inline:
        plt.show()
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.semilogy(freqs, np.maximum(Pstart, 1e-12), lw=0.9, color=C_START, label="tess start")
    ax.semilogy(freqs, np.maximum(Pend, 1e-12), lw=0.9, color=C_END, linestyle="--", label="tess end")
    if not mode_table.empty:
        for _, r in mode_table.iterrows():
            ax.axvline(float(r["f"]), lw=0.8, alpha=0.35, color="0.3")
    ax.set_xlabel("Frequency [c/d]")
    ax.set_ylabel("Power")
    ax.set_title("TESS spectra (log y)")
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(outdir / prefixed_output_name(file_prefix, "spectra_start_end_log_offset.png"), dpi=180)
    if show_plots_inline:
        plt.show()
    plt.close(fig)

    win = spectral_window(tess0, freqs)
    fig, ax = plt.subplots(figsize=(11, 4.3))
    ax.plot(freqs, win, lw=0.9, color="0.15")
    ax.set_xlabel("Frequency [c/d]")
    ax.set_ylabel("Window power")
    ax.set_title("TESS spectral window")
    fig.tight_layout()
    fig.savefig(outdir / prefixed_output_name(file_prefix, "spectral_windows.png"), dpi=180)
    if show_plots_inline:
        plt.show()
    plt.close(fig)

def plot_channel_timeseries_and_spectra(tess0: TimeSeries, tess_final_resid: TimeSeries, tess_table: pd.DataFrame,
                                        pol0: TimeSeries, pol_final_resid: TimeSeries, pol_table: pd.DataFrame,
                                        freqs_tess: np.ndarray, Pte_start: np.ndarray, Pte_end: np.ndarray,
                                        freqs_pol: np.ndarray, Ppo_start: np.ndarray, Ppo_end: np.ndarray,
                                        freqs_pol_full: np.ndarray, Ppo_start_full: np.ndarray, Ppo_end_full: np.ndarray,
                                        outdir: Path, guided_diags: list[dict] | None = None,
                                        show_plots_inline: bool = False, file_prefix: str = ""):
    outdir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(11, 7.5), sharex=False)

    C_START = "#7EC8FF"
    C_END = "#000000"
    ax = axes[0]
    t0 = tess0.y - np.nanmedian(tess0.y)
    t1 = tess_final_resid.y - np.nanmedian(tess_final_resid.y)
    ax.scatter(tess0.t, t0, s=5, alpha=0.50, label="tess start", color=C_START)
    ax.scatter(tess_final_resid.t, t1, s=5, alpha=0.55, label="tess end (resid)", color=C_END)
    ax.set_ylabel("TESS value (median-subtracted)")
    set_robust_time_ylim(ax, t0, t1, central_pct=98.0, pad_frac=0.12)
    ax.set_title("TESS time series")
    ax.legend(loc="best", fontsize=9)

    ax = axes[1]
    p0 = pol0.y - np.nanmedian(pol0.y)
    p1 = pol_final_resid.y - np.nanmedian(pol_final_resid.y)
    ax.scatter(pol0.t, p0, s=6, alpha=0.50, label=f"{pol0.name} start", color=C_START)
    ax.scatter(pol_final_resid.t, p1, s=6, alpha=0.55, label=f"{pol0.name} end (resid)", color=C_END)
    ax.set_xlabel("Time [days]")
    ax.set_ylabel(f"{pol0.name} value (median-subtracted)")
    set_robust_time_ylim(ax, p0, p1, central_pct=98.0, pad_frac=0.12)
    ax.set_title(f"{pol0.name} time series")
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(outdir / prefixed_output_name(file_prefix, "timeseries_start_end.png"), dpi=180)
    if show_plots_inline:
        plt.show()
    plt.close(fig)

    guided_diags = guided_diags or []
    nrows = 1 + max(len(guided_diags), 1)
    fig, axes = plt.subplots(nrows=nrows, ncols=1, figsize=(11, 3.0 * nrows), sharex=False)
    if nrows == 1:
        axes = [axes]

    ax = axes[0]
    T0 = _norm_spec(Pte_start)
    T1 = _norm_spec(Pte_end)
    ax.plot(freqs_tess, T0, lw=0.9, color=C_START, label="tess start")
    ax.plot(freqs_tess, T1 + 1.2, lw=0.9, color=C_END, linestyle="--", label="tess end (offset)")
    if not tess_table.empty:
        ytop = np.nanmax(T1 + 1.2)
        for _, r in tess_table.iterrows():
            ax.axvline(float(r["f"]), lw=0.8, alpha=0.25, color="0.3")
            ax.text(float(r["f"]), ytop, str(int(r["mode"])), rotation=90, va="top", ha="center", fontsize=8)
    ax.set_ylabel("TESS norm. power + offset")
    ax.set_title("TESS spectra")
    ax.legend(loc="best", fontsize=9)

    if guided_diags:
        for i, gd in enumerate(guided_diags, start=1):
            ax = axes[i]
            freqs = gd["freqs"]
            raw, selection = _norm_spec_shared(
                gd["power_iteration_raw"],
                gd["power_iteration_smoothed"],
            )
            smoothing = gd.get("smoothing", {}) or {}
            if smoothing.get("applied", False):
                ax.plot(freqs, raw, lw=0.8, color=C_START, alpha=0.38, label=f"{pol0.name} iteration raw")
                ax.plot(freqs, selection, lw=1.25, color="#0072B2", label=f"{pol0.name} smoothed for selection")
            else:
                label = f"{pol0.name} iteration raw / selection"
                if smoothing.get("enabled", False) and not smoothing.get("applied", False):
                    label += " (kernel under-sampled)"
                ax.plot(freqs, raw, lw=1.1, color="#0072B2", label=label)
            for marker in guided_diagnostic_marker_specs(gd):
                ax.axvline(
                    marker["frequency"],
                    lw=1.05,
                    color=marker["color"],
                    linestyle=marker["linestyle"],
                    alpha=marker["alpha"],
                    label=marker["label"],
                )
            ax.set_title(guided_diagnostic_title(pol0.name, gd), fontsize=9.5)
            ax.set_ylabel(f"{pol0.name} normalized power")
            if i == len(guided_diags):
                ax.set_xlabel("Frequency [c/d]")
            add_tess_spectrum_background(ax, freqs_tess, Pte_start)
            ax.legend(loc="best", fontsize=8)
    else:
        ax = axes[1]
        P0 = _norm_spec(Ppo_start)
        P1 = _norm_spec(Ppo_end)
        ax.plot(freqs_pol, P0, lw=0.9, color=C_START, label=f"{pol0.name} start")
        ax.plot(freqs_pol, P1 + 1.2, lw=0.9, color=C_END, linestyle="--", label=f"{pol0.name} end (offset)")
        ax.set_xlabel("Frequency [c/d]")
        ax.set_ylabel(f"{pol0.name} norm. power + offset")
        ax.set_title(f"{pol0.name} spectra")
        add_tess_spectrum_background(ax, freqs_tess, Pte_start)
        ax.legend(loc="best", fontsize=9)

    fig.tight_layout()
    fig.savefig(outdir / prefixed_output_name(file_prefix, "spectra_start_end.png"), dpi=180)
    if show_plots_inline:
        plt.show()
    plt.close(fig)

    # Full-range polarimetry spectrum on a full-baseline plot grid
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(11, 7.2), sharex=False)
    ax = axes[0]
    ax.plot(freqs_tess, T0, lw=0.9, color=C_START, label="tess start")
    ax.plot(freqs_tess, T1 + 1.2, lw=0.9, color=C_END, linestyle="--", label="tess end (offset)")
    if not tess_table.empty:
        ytop = np.nanmax(T1 + 1.2)
        for _, r in tess_table.iterrows():
            ax.axvline(float(r["f"]), lw=0.8, alpha=0.25, color="0.3")
            ax.text(float(r["f"]), ytop, str(int(r["mode"])), rotation=90, va="top", ha="center", fontsize=8)
    ax.set_ylabel("TESS norm. power + offset")
    ax.set_title("TESS spectra")
    ax.legend(loc="best", fontsize=9)

    ax = axes[1]
    P0_full = _norm_spec(Ppo_start_full)
    P1_full = _norm_spec(Ppo_end_full)
    ax.plot(freqs_pol_full, P0_full, lw=0.9, color=C_START, label=f"{pol0.name} start")
    ax.plot(freqs_pol_full, P1_full + 1.2, lw=0.9, color=C_END, linestyle="--", label=f"{pol0.name} end (offset)")
    if not pol_table.empty:
        ytop = np.nanmax(P1_full + 1.2)
        for _, r in pol_table.iterrows():
            ax.axvline(float(r["f"]), lw=0.8, alpha=0.35, color="#009E73")
    ax.set_xlabel("Frequency [c/d]")
    ax.set_ylabel(f"{pol0.name} norm. power + offset")
    ax.set_title(f"{pol0.name} spectra (full-range, full-baseline grid)")
    add_tess_spectrum_background(ax, freqs_tess, Pte_start)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(outdir / prefixed_output_name(file_prefix, "spectra_fullrange_start_end.png"), dpi=180)
    if show_plots_inline:
        plt.show()
    plt.close(fig)

    # Local guided log-power diagnostics
    if guided_diags:
        fig, axes = plt.subplots(nrows=len(guided_diags), ncols=1, figsize=(11, 2.7 * len(guided_diags)), sharex=False)
        if len(guided_diags) == 1:
            axes = [axes]
        for ax, gd in zip(axes, guided_diags):
            freqs = gd["freqs"]
            raw = np.maximum(np.asarray(gd["power_iteration_raw"], dtype=float), 1e-12)
            selection = np.maximum(np.asarray(gd["power_iteration_smoothed"], dtype=float), 1e-12)
            smoothing = gd.get("smoothing", {}) or {}
            if smoothing.get("applied", False):
                ax.semilogy(freqs, raw, lw=0.8, alpha=0.42, color=C_START, label=f"{pol0.name} iteration raw")
                ax.semilogy(freqs, selection, lw=1.2, color="#0072B2", label=f"{pol0.name} smoothed for selection")
            else:
                label = f"{pol0.name} iteration raw / selection"
                if smoothing.get("enabled", False):
                    label += " (kernel under-sampled)"
                ax.semilogy(freqs, raw, lw=1.1, color="#0072B2", label=label)
            for marker in guided_diagnostic_marker_specs(gd):
                ax.axvline(
                    marker["frequency"],
                    lw=1.05,
                    color=marker["color"],
                    linestyle=marker["linestyle"],
                    alpha=marker["alpha"],
                    label=marker["label"],
                )
            ax.set_title(guided_diagnostic_title(pol0.name, gd), fontsize=9.5)
            ax.set_ylabel("Power")
            add_tess_spectrum_background(ax, freqs_tess, Pte_start, log_y=True)
            ax.legend(loc="best", fontsize=8)
        axes[-1].set_xlabel("Frequency [c/d]")
    else:
        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(11, 7.5), sharex=False)
        axes[0].semilogy(freqs_tess, np.maximum(Pte_start, 1e-12), lw=0.9, color=C_START, label="tess start")
        axes[0].semilogy(freqs_tess, np.maximum(Pte_end, 1e-12), lw=0.9, color=C_END, linestyle="--", label="tess end")
        axes[0].legend(loc="best", fontsize=9)
        axes[0].set_ylabel("TESS power")
        axes[0].set_title("TESS spectra (log y)")

        axes[1].semilogy(freqs_pol, np.maximum(Ppo_start, 1e-12), lw=0.9, color=C_START, label=f"{pol0.name} start")
        axes[1].semilogy(freqs_pol, np.maximum(Ppo_end, 1e-12), lw=0.9, color="#CC79A7", linestyle="--", label=f"{pol0.name} end")
        axes[1].legend(loc="best", fontsize=9)
        axes[1].set_xlabel("Frequency [c/d]")
        axes[1].set_ylabel(f"{pol0.name} power")
        axes[1].set_title(f"{pol0.name} spectra (log y)")
        add_tess_spectrum_background(axes[1], freqs_tess, Pte_start, log_y=True)
    fig.tight_layout()
    fig.savefig(outdir / prefixed_output_name(file_prefix, "spectra_start_end_log_offset.png"), dpi=180)
    if show_plots_inline:
        plt.show()
    plt.close(fig)

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(11, 7.2), sharex=False)
    axes[0].semilogy(freqs_tess, np.maximum(Pte_start, 1e-12), lw=0.9, color=C_START, label="tess start")
    axes[0].semilogy(freqs_tess, np.maximum(Pte_end, 1e-12), lw=0.9, color=C_END, linestyle="--", label="tess end")
    if not tess_table.empty:
        for _, r in tess_table.iterrows():
            axes[0].axvline(float(r["f"]), lw=0.8, alpha=0.25, color="0.3")
    axes[0].set_ylabel("TESS power")
    axes[0].set_title("TESS spectra (log y)")
    axes[0].legend(loc="best", fontsize=9)

    axes[1].semilogy(freqs_pol_full, np.maximum(Ppo_start_full, 1e-12), lw=0.9, color=C_START, label=f"{pol0.name} start")
    axes[1].semilogy(freqs_pol_full, np.maximum(Ppo_end_full, 1e-12), lw=0.9, color="#CC79A7", linestyle="--", label=f"{pol0.name} end")
    if not pol_table.empty:
        for _, r in pol_table.iterrows():
            axes[1].axvline(float(r["f"]), lw=0.8, alpha=0.35, color="#009E73")
    axes[1].set_xlabel("Frequency [c/d]")
    axes[1].set_ylabel(f"{pol0.name} power")
    axes[1].set_title(f"{pol0.name} spectra (full-range, log y)")
    add_tess_spectrum_background(axes[1], freqs_tess, Pte_start, log_y=True)
    axes[1].legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(outdir / prefixed_output_name(file_prefix, "spectra_fullrange_start_end_log.png"), dpi=180)
    if show_plots_inline:
        plt.show()
    plt.close(fig)

    win_t = spectral_window(tess0, freqs_tess)
    win_p = spectral_window(pol0, freqs_pol_full)
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(11, 7.0), sharex=False)
    axes[0].plot(freqs_tess, win_t, lw=0.9, color="0.2")
    axes[0].set_ylabel("Window power")
    axes[0].set_title("TESS spectral window")
    axes[1].plot(freqs_pol_full, win_p, lw=0.9, color="0.2")
    axes[1].set_xlabel("Frequency [c/d]")
    axes[1].set_ylabel("Window power")
    axes[1].set_title(f"{pol0.name} spectral window (full-baseline grid)")
    fig.tight_layout()
    fig.savefig(outdir / prefixed_output_name(file_prefix, "spectral_windows.png"), dpi=180)
    if show_plots_inline:
        plt.show()
    plt.close(fig)

def plot_tess_phased_modes(tess_snapshots: list[dict], mode_table: pd.DataFrame, outdir: Path,
                           n_phase_plots: int = 3, sort_by: str = "snr",
                           show_plots_inline: bool = False, file_prefix: str = ""):
    if int(n_phase_plots) <= 0 or mode_table.empty or not tess_snapshots:
        return
    snap_map = {int(s["mode"]): s for s in tess_snapshots if "mode" in s}
    sort_col = {"amp": "amp", "snr": "snr_local", "mode": "mode"}.get(sort_by, sort_by)
    if sort_col not in mode_table.columns:
        sort_col = "snr_local"
    pick = mode_table.sort_values(sort_col, ascending=(sort_col == "mode")).head(n_phase_plots).reset_index(drop=True)
    fig, axes = plt.subplots(nrows=len(pick), ncols=1, figsize=(8.5, 3.7 * len(pick)), squeeze=False)
    for i, (_, r) in enumerate(pick.iterrows()):
        ax = axes[i, 0]
        mode = int(r["mode"])
        snap = snap_map.get(mode)
        if snap is None:
            continue
        fit = snap["fit"]
        prefit = snap["prefit"]
        f = float(fit["best_f"])
        y = prefit.y - np.nanmedian(prefit.y)
        ph = phase_fold(prefit.t, f)
        ax.scatter(ph, y, s=6, alpha=0.65, color="0.25")
        ax.scatter(ph + 1.0, y, s=6, alpha=0.65, color="0.25")
        ph_grid = np.linspace(0.0, 2.0, 500)
        model = float(fit["s_coeff"]) * np.sin(2.0 * np.pi * ph_grid) + float(fit["c_coeff"]) * np.cos(2.0 * np.pi * ph_grid)
        model = model - np.nanmedian(model)
        ax.plot(ph_grid, model, lw=1.4, color="0.0")
        set_robust_time_ylim(ax, y, model, central_pct=98.0, pad_frac=0.12)
        ax.set_xlim(0.0, 2.0)
        ax.set_xlabel("Phase")
        ax.set_ylabel("TESS (median-subtracted)")
        ax.set_title(f"Mode {mode} | TESS | f={float(r['f']):.6f} c/d | amp={float(r['amp']):.4g} | SNR={float(r['snr_local']):.2f}")
    fig.tight_layout()
    fig.savefig(outdir / prefixed_output_name(file_prefix, "phased_top_modes.png"), dpi=180)
    if show_plots_inline:
        plt.show()
    plt.close(fig)

def plot_channel_phased_modes(
    tess_snapshots: list[dict],
    tess_table: pd.DataFrame,
    pol_search_df: pd.DataFrame,
    pol_snapshots: list[dict],
    pol0: TimeSeries,
    outdir: Path,
    n_phase_plots: int = 3,
    sort_by: str = "amp",
    show_plots_inline: bool = False,
    file_prefix: str = "",
    *,
    tess0: TimeSeries | None = None,
    tess_global_fit: dict | None = None,
    pol_final_df: pd.DataFrame | None = None,
    pol_global_fit: dict | None = None,
):
    """Plot isolated components from the authoritative final global fits.

    Older releases plotted sequential local snapshots even after a global fit
    moved or pruned a component.  The fallback remains for compatibility, but
    current callers provide the final tables and models.
    """
    if int(n_phase_plots) <= 0:
        return
    final_available = (
        pol_final_df is not None
        and not pol_final_df.empty
        and pol_global_fit is not None
        and tess0 is not None
        and tess_global_fit is not None
    )
    if final_available:
        detected = pol_final_df.copy()
        if "tess_mode" not in detected.columns:
            return
    elif pol_search_df is not None and len(pol_search_df):
        if "postfit_keep" in pol_search_df.columns:
            detected = pol_search_df.loc[pol_search_df["postfit_keep"]].copy()
        elif "seed_for_global" in pol_search_df.columns:
            detected = pol_search_df.loc[pol_search_df["seed_for_global"]].copy()
        else:
            detected = pol_search_df.loc[pol_search_df["detected"]].copy()
    else:
        detected = pd.DataFrame()
    if detected.empty:
        return

    snap_map = {int(s["tess_mode"]): s for s in pol_snapshots if "tess_mode" in s}
    tess_snap_map = {int(s["mode"]): s for s in tess_snapshots if "mode" in s}
    sort_candidates = {
        "amp": ("amp", "amp_local"),
        "snr": ("postfit_snr_local", "snr_local"),
        "mode": ("tess_mode",),
    }.get(sort_by, (sort_by,))
    sort_col = next((name for name in sort_candidates if name in detected.columns), None)
    if sort_col is None:
        sort_col = "tess_mode"
    pick = (
        detected.sort_values(sort_col, ascending=(sort_col == "tess_mode"))
        .head(n_phase_plots)
        .reset_index(drop=True)
    )
    fig, axes = plt.subplots(
        nrows=len(pick), ncols=2, figsize=(12, 3.8 * len(pick)), squeeze=False
    )

    for i, (_, pr) in enumerate(pick.iterrows()):
        tess_mode = int(pr["tess_mode"])
        tr = tess_table.loc[tess_table["mode"] == tess_mode]
        tr = None if tr.empty else tr.iloc[0]

        ax = axes[i, 0]
        if final_available and tr is not None:
            frequency_t = float(tr["f"])
            component_t = component_signal(
                tess0.t,
                frequency_t,
                float(tr["s_coeff"]),
                float(tr["c_coeff"]),
            )
            isolated_t = (
                tess0.y
                - np.asarray(tess_global_fit["baseline_model"], dtype=float)
                - (
                    np.asarray(tess_global_fit["signal_model"], dtype=float)
                    - component_t
                )
            )
            reference = float(pr.get("phase_reference_btjd", resolve_phase_reference_btjd(tess0, pol0)))
            phase_t = ((_series_btjd_abs(tess0) - reference) * frequency_t) % 1.0
            phase_t_ref = phase_at_reference(
                float(tr["phase"]),
                frequency_t,
                _series_local_zero_btjd(tess0),
                reference,
            )
            phase_grid = np.linspace(0.0, 2.0, 500)
            model_t = float(tr["amp"]) * np.sin(
                2.0 * np.pi * phase_grid + phase_t_ref
            )
            ax.scatter(phase_t, isolated_t, s=6, alpha=0.65, color="0.25")
            ax.scatter(phase_t + 1.0, isolated_t, s=6, alpha=0.65, color="0.25")
            ax.plot(phase_grid, model_t, lw=1.4, color="0.0")
            set_robust_time_ylim(
                ax, isolated_t, model_t, central_pct=98.0, pad_frac=0.12
            )
            ax.set_title(
                f"Mode {tess_mode} | TESS final global fit | "
                f"f={frequency_t:.6f} c/d | amp={float(tr['amp']):.4g}"
            )
        else:
            tsnap = tess_snap_map.get(tess_mode)
            if tsnap is not None and tr is not None:
                tfit = tsnap["fit"]
                tprefit = tsnap["prefit"]
                frequency_t = float(tfit["best_f"])
                isolated_t = tprefit.y - np.nanmedian(tprefit.y)
                phase_t = phase_fold(tprefit.t, frequency_t)
                ax.scatter(phase_t, isolated_t, s=6, alpha=0.65, color="0.25")
                ax.scatter(phase_t + 1.0, isolated_t, s=6, alpha=0.65, color="0.25")
        ax.set_xlim(0.0, 2.0)
        ax.set_xlabel("Phase")
        ax.set_ylabel("TESS isolated component")

        ax = axes[i, 1]
        binned_y = np.array([], dtype=float)
        if final_available:
            frequency_p = float(pr["f"])
            component_p = component_signal(
                pol0.t,
                frequency_p,
                float(pr["s_coeff"]),
                float(pr["c_coeff"]),
            )
            isolated_p = (
                pol0.y
                - np.asarray(pol_global_fit["baseline_model"], dtype=float)
                - (
                    np.asarray(pol_global_fit["signal_model"], dtype=float)
                    - component_p
                )
            )
            reference = float(pr.get("phase_reference_btjd", resolve_phase_reference_btjd(tess0, pol0)))
            phase_p = ((_series_btjd_abs(pol0) - reference) * frequency_p) % 1.0
            phase_p_ref = float(
                pr.get(
                    "phase_at_reference_rad",
                    phase_at_reference(
                        float(pr["phase"]),
                        frequency_p,
                        _series_local_zero_btjd(pol0),
                        reference,
                    ),
                )
            )
            phase_grid = np.linspace(0.0, 2.0, 500)
            model_p = float(pr["amp"]) * np.sin(
                2.0 * np.pi * phase_grid + phase_p_ref
            )
            ax.scatter(phase_p, isolated_p, s=8, alpha=0.50, color="0.25")
            ax.scatter(phase_p + 1.0, isolated_p, s=8, alpha=0.50, color="0.25")
            binned_phase, binned_y = phase_bin_medians(
                phase_p, isolated_p, nbin=PHASE_BIN_N
            )
            if len(binned_phase):
                ax.scatter(
                    binned_phase,
                    binned_y,
                    s=20,
                    alpha=0.95,
                    color="#D55E00",
                    zorder=3,
                )
                ax.scatter(
                    binned_phase + 1.0,
                    binned_y,
                    s=20,
                    alpha=0.95,
                    color="#D55E00",
                    zorder=3,
                    label="phase-bin median",
                )
            ax.plot(phase_grid, model_p, lw=1.4, color="0.0")
            set_robust_time_ylim(
                ax,
                isolated_p,
                model_p,
                binned_y,
                central_pct=98.0,
                pad_frac=0.12,
            )
            relation = (
                f" | {pr['quadrature_relation']}"
                if "quadrature_relation" in pr and pd.notna(pr["quadrature_relation"])
                else ""
            )
            ax.set_title(
                f"Mode {tess_mode} | {pol0.name} final global fit | "
                f"f={frequency_p:.6f} c/d | amp={float(pr['amp']):.4g}{relation}"
            )
        else:
            psnap = snap_map.get(tess_mode)
            if psnap is not None:
                pfit = psnap["fit"]
                pprefit = psnap["prefit"]
                frequency_p = float(pfit["best_f"])
                isolated_p = pprefit.y - np.asarray(pfit["baseline_model"], dtype=float)
                phase_p = phase_fold(pprefit.t, frequency_p)
                ax.scatter(phase_p, isolated_p, s=8, alpha=0.50, color="0.25")
                ax.scatter(phase_p + 1.0, isolated_p, s=8, alpha=0.50, color="0.25")
        ax.set_xlim(0.0, 2.0)
        ax.set_xlabel("Phase")
        ax.set_ylabel(f"{pol0.name} isolated component")

    fig.tight_layout()
    fig.savefig(
        outdir / prefixed_output_name(file_prefix, "phased_top_modes.png"), dpi=180
    )
    if show_plots_inline:
        plt.show()
    plt.close(fig)



def set_robust_time_ylim(ax, *arrays, central_pct: float = 98.0, pad_frac: float = 0.12):
    """Set a robust y-range using the central percentile range plus padding."""
    vals = []
    for arr in arrays:
        if arr is None:
            continue
        a = np.asarray(arr, dtype=float).ravel()
        a = a[np.isfinite(a)]
        if a.size:
            vals.append(a)
    if not vals:
        return
    y = np.concatenate(vals)
    if y.size == 0:
        return

    central_pct = float(central_pct)
    central_pct = min(99.9, max(80.0, central_pct))
    tail = 0.5 * (100.0 - central_pct)
    lo = np.nanpercentile(y, tail)
    hi = np.nanpercentile(y, 100.0 - tail)

    if not np.isfinite(lo) or not np.isfinite(hi):
        lo = np.nanmin(y)
        hi = np.nanmax(y)
    if not np.isfinite(lo) or not np.isfinite(hi):
        return

    if hi <= lo:
        med = np.nanmedian(y)
        span = np.nanpercentile(np.abs(y - med), 90.0)
        if (not np.isfinite(span)) or span <= 0:
            span = max(np.nanstd(y), 1e-6)
        lo = med - span
        hi = med + span

    span = hi - lo
    if (not np.isfinite(span)) or span <= 0:
        span = max(np.nanstd(y), 1e-6)
        if (not np.isfinite(span)) or span <= 0:
            span = 1.0

    pad = pad_frac * span
    ax.set_ylim(lo - pad, hi + pad)

def show_table(title: str, df: pd.DataFrame):
    print(f"\n--- {title} ---")
    if df is None or len(df) == 0:
        print("(empty)")
        return
    if display is not None:
        try:
            display(df)
            return
        except Exception:
            pass
    with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", 200):
        print(df.to_string(index=False))

def guided_diagnostics_to_dataframe(diags: list[dict]) -> pd.DataFrame:
    """Flatten exact per-iteration Guided spectra into a reviewable CSV."""
    frames = []
    for diag in diags:
        freqs = np.asarray(diag.get("freqs", []), dtype=float)
        raw = np.asarray(diag.get("power_iteration_raw", []), dtype=float)
        smoothed = np.asarray(diag.get("power_iteration_smoothed", []), dtype=float)
        if not (len(freqs) == len(raw) == len(smoothed)):
            raise ValueError("Guided diagnostic frequency and power arrays must have the same length.")
        smoothing = diag.get("smoothing", {}) or {}
        frames.append(pd.DataFrame({
            "search_order": int(diag.get("search_order", 0)),
            "tess_mode": int(diag["tess_mode"]),
            "frequency_cpd": freqs,
            "power_iteration_raw": raw,
            "power_for_selection": smoothed,
            "f_tess": float(diag.get("f_tess", np.nan)),
            "f_candidate_selection_max": float(diag.get("f_candidate", np.nan)),
            "f_local_fit": float(diag.get("f_local", np.nan)),
            "f_global_fit": float(diag.get("f_global", np.nan)),
            "detected": bool(diag.get("detected", False)),
            "seed_for_global": bool(diag.get("seed_for_global", False)),
            "postfit_keep": bool(diag.get("postfit_keep", False)),
            "decision_status": str(diag.get("decision_status", "rejected")),
            "snr_local": float(diag.get("snr_local", np.nan)),
            "postfit_snr_global": float(diag.get("postfit_snr_global", np.nan)),
            "frequency_offset_local_tess_cpd": float(diag.get("freq_offset", np.nan)),
            "skip_reason": str(diag.get("skip_reason", "")),
            "smoothing_enabled": bool(smoothing.get("enabled", False)),
            "smoothing_applied": bool(smoothing.get("applied", False)),
            "smoothing_kernel": str(smoothing.get("kernel", "")),
            "smoothing_width_resolution_elements": float(smoothing.get("width_resolution_elements", np.nan)),
            "smoothing_effective_width_cpd": float(smoothing.get("effective_width_cpd", np.nan)),
        }))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

# ----------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------

def _run_analysis_impl():
    print("TESS_INPUT_MODE =", TESS_INPUT_MODE)
    print("USE_POLARIMETRY =", USE_POLARIMETRY)
    print("TESS_SNR_STOP =", TESS_SNR_STOP, "| POL_SNR_STOP =", POL_SNR_STOP)
    print("GLOBAL_FREQ_BOUND_MULT =", GLOBAL_FREQ_BOUND_MULT, "| GLOBAL_MAX_NFEV =", GLOBAL_MAX_NFEV)
    print("MIN_FREQ_SEP_MULT =", MIN_FREQ_SEP_MULT, "| TESS_DISCOVERY_EXCLUSION_MULT =", TESS_DISCOVERY_EXCLUSION_MULT)
    print("POL_SEARCH_WINDOW_MULT =", POL_SEARCH_WINDOW_MULT, "| POL_LOCAL_NOISE_KS =", POL_LOCAL_NOISE_KS,
          "| POL_LOCAL_NOISE_SIDE_BINS =", POL_LOCAL_NOISE_SIDE_BINS)
    print("GUIDED_POL_FMIN =", GUIDED_POL_FMIN, "| PHASE_BIN_N =", PHASE_BIN_N)
    print(
        "POL periodogram smoothing =",
        POL_SMOOTH_ENABLED,
        "| kernel =",
        POL_SMOOTH_KERNEL,
        "| width =",
        POL_SMOOTH_WIDTH_RES_ELEMS,
        "resolution elements",
    )
    print(
        "TESS spectrum background =",
        SHOW_TESS_SPECTRUM_BACKGROUND,
        "| scale =",
        TESS_SPECTRUM_BACKGROUND_SCALE,
    )
    print(
        "q/u phase relation =",
        QU_PHASE_MODE,
        "| phase reference =",
        PHASE_ZERO_MODE,
        (
            f"(BTJD {PHASE_ZERO_BTJD})"
            if str(PHASE_ZERO_MODE).strip().lower() == "custom_btjd"
            else ""
        ),
    )
    print("VERBOSE =", VERBOSE, "| LSQ_VERBOSE =", LSQ_VERBOSE)
    print("POL channels =", POL_CHANNELS, "| p is treated as a directly observed channel in this version")

    trend_cfg = NightTrendConfig(
        use_offsets=USE_POL_NIGHT_OFFSETS,
        use_slopes=USE_POL_NIGHT_SLOPES,
        group_mode=POL_NIGHT_GROUP_MODE,
        gap_days=POL_NIGHT_GAP_HOURS / 24.0,
    )

    t_all = time.time()
    tess = load_tess_input()

    requested_fmax = float(FMAX)
    nyq_cpd = compute_tess_nyquist_cpd(tess)
    effective_fmax = requested_fmax
    if TESS_CAP_FMAX_TO_NYQUIST and np.isfinite(nyq_cpd):
        effective_fmax = min(requested_fmax, float(nyq_cpd))
        effective_fmax = max(float(FMIN), effective_fmax)
    globals()["FMAX"] = float(effective_fmax)

    if np.isfinite(nyq_cpd):
        print("Requested FMAX =", requested_fmax, "| TESS Nyquist =", f"{nyq_cpd:.6g}", "| effective FMAX =", f"{effective_fmax:.6g}", "| cap_to_nyquist =", TESS_CAP_FMAX_TO_NYQUIST)
    else:
        print("Requested FMAX =", requested_fmax, "| TESS Nyquist = unavailable | effective FMAX =", f"{effective_fmax:.6g}", "| cap_to_nyquist =", TESS_CAP_FMAX_TO_NYQUIST)

    if USE_POLARIMETRY:
        pol_dict = load_polarimetry_csv(
            POL_CSV,
            product=POL_PRODUCT,
            trend_cfg=trend_cfg,
            channels=POL_CHANNELS,
        )
        star_label, star_safe = infer_star_labels_from_pol_path(POL_CSV)
        vprint(1, f"Loaded datasets | TESS n={len(tess.t)} | " + ", ".join([f"{k} n={len(v.t)}" for k, v in pol_dict.items()]))
    else:
        pol_dict = {}
        star_label, star_safe = infer_star_labels_from_tess_input()
        vprint(1, f"Loaded datasets | TESS n={len(tess.t)} | photometry-only mode")
    print("STAR_LABEL =", star_label)
    analysis_outroot = resolve_analysis_outroot(star_safe)
    smoothing_config = polarimetry_smoothing_metadata(
        max(compute_T_full(next(iter(pol_dict.values()))), 1e-8) if pol_dict else np.nan
    )
    (analysis_outroot / "analysis_run_config.json").write_text(
        json.dumps(
            {
                "analysis_track": "guided",
                "tess_input_mode": str(TESS_INPUT_MODE),
                "frequency_range_cpd": {
                    "minimum": float(FMIN),
                    "requested_maximum": requested_fmax,
                    "effective_maximum": effective_fmax,
                    "tess_nyquist": float(nyq_cpd) if np.isfinite(nyq_cpd) else None,
                    "capped_to_nyquist": bool(TESS_CAP_FMAX_TO_NYQUIST),
                },
                "polarimetry_product": str(POL_PRODUCT),
                "polarimetry_channels": list(POL_CHANNELS),
                "polarimetry_periodogram_smoothing": smoothing_config,
                "tess_spectrum_background": {
                    "enabled": bool(SHOW_TESS_SPECTRUM_BACKGROUND),
                    "scale": normalize_tess_spectrum_background_scale(
                        TESS_SPECTRUM_BACKGROUND_SCALE
                    ),
                },
                "qu_phase_constraint": {
                    "mode": normalize_quadrature_mode(QU_PHASE_MODE),
                    "phase_zero_mode": str(PHASE_ZERO_MODE),
                    "phase_zero_btjd": float(PHASE_ZERO_BTJD),
                    "maximum_function_evaluations": int(QUADRATURE_MAX_NFEV),
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    if DO_DETREND and DETREND_POLY_ORDER > 0:
        tess_dt = detrend_poly(tess, order=DETREND_POLY_ORDER)
        pol_dt = {k: detrend_poly(v, order=DETREND_POLY_ORDER) for k, v in pol_dict.items()} if USE_POLARIMETRY else {}
    else:
        tess_dt = tess
        pol_dt = pol_dict

    # TESS-only analysis
    tess_prefix = f"{star_safe}_tess"
    tess_outdir = analysis_outroot / tess_prefix
    tess_outdir.mkdir(parents=True, exist_ok=True)
    vprint(1, "Starting TESS-only analysis...")
    tess_local_df, tess_global_fit, tess_snapshots, tess_final_resid, freqs_tess, Pte_start, Pte_end = extract_tess_modes(tess_dt)
    tess_table = match_global_components_with_local(tess_local_df, tess_global_fit)
    tess_table.to_csv(tess_outdir / prefixed_output_name(tess_prefix, "peaks_table.csv"), index=False)
    pd.DataFrame(tess_global_fit["component_rows"]).to_csv(tess_outdir / prefixed_output_name(tess_prefix, "global_components_raw.csv"), index=False)
    show_table("TESS peaks_table", tess_table)
    plot_tess_timeseries_and_spectra(tess_dt, tess_final_resid, freqs_tess, Pte_start, Pte_end, tess_table,
                                     tess_outdir, show_plots_inline=SHOW_PLOTS_INLINE, file_prefix=tess_prefix)
    plot_tess_phased_modes(tess_snapshots, tess_table, tess_outdir,
                           n_phase_plots=N_PHASE_PLOTS, sort_by=PHASE_SORT_BY,
                           show_plots_inline=SHOW_PLOTS_INLINE, file_prefix=tess_prefix)

    outputs = {"tess": tess_table}

    if not USE_POLARIMETRY:
        vprint(1, f"Photometry-only analysis complete in {time.time()-t_all:.1f}s")
        return outputs

    channel_states: dict[str, dict] = {}
    for k in POL_CHANNELS:
        if k not in pol_dt:
            print(f"Skipping channel {k!r}: not present.")
            continue
        print(f"\n=== Guided run for {star_label} {k} ===")
        (search_df, final_df, global_fit, pol_snapshots, pol_final_resid,
         freqs_pol, Ppo_start, Ppo_end, guided_diags,
         freqs_pol_full, Ppo_start_full, Ppo_end_full) = search_guided_channel(
            pol_dt[k], tess_table, 1.0 / max(compute_T_full(tess_dt), 1e-8), trend_cfg
        )
        channel_states[k] = {
            "pol0": pol_dt[k],
            "search_df": search_df,
            "final_df": final_df,
            "global_fit": global_fit,
            "pol_snapshots": pol_snapshots,
            "final_resid": pol_final_resid,
            "freqs_pol": freqs_pol,
            "Ppo_start": Ppo_start,
            "Ppo_end": Ppo_end,
            "guided_diags": guided_diags,
            "freqs_pol_full": freqs_pol_full,
            "Ppo_start_full": Ppo_start_full,
            "Ppo_end_full": Ppo_end_full,
        }

    channel_states, quadrature_table = apply_guided_quadrature_refit(
        tess_dt,
        tess_table,
        channel_states,
        trend_cfg,
    )
    if not quadrature_table.empty:
        quadrature_prefix = f"{star_safe}_qu"
        quadrature_outdir = analysis_outroot / quadrature_prefix
        quadrature_outdir.mkdir(parents=True, exist_ok=True)
        quadrature_table.to_csv(
            quadrature_outdir
            / prefixed_output_name(quadrature_prefix, "quadrature_peaks_table.csv"),
            index=False,
        )
        print(
            "[q/u quadrature] wrote",
            quadrature_outdir
            / prefixed_output_name(quadrature_prefix, "quadrature_peaks_table.csv"),
        )

    for k in POL_CHANNELS:
        if k not in channel_states:
            continue
        state = channel_states[k]
        channel_prefix = f"{star_safe}_{k}"
        outdir = analysis_outroot / channel_prefix
        outdir.mkdir(parents=True, exist_ok=True)
        search_df = state["search_df"]
        final_df = state["final_df"]
        global_fit = state["global_fit"]
        pol_snapshots = state["pol_snapshots"]
        pol_final_resid = state["final_resid"]
        freqs_pol = state["freqs_pol"]
        Ppo_start = state["Ppo_start"]
        Ppo_end = state["Ppo_end"]
        guided_diags = state["guided_diags"]
        freqs_pol_full = state["freqs_pol_full"]
        Ppo_start_full = state["Ppo_start_full"]
        Ppo_end_full = state["Ppo_end_full"]
        search_df.to_csv(outdir / prefixed_output_name(channel_prefix, "matched_search_table.csv"), index=False)
        final_df.to_csv(outdir / prefixed_output_name(channel_prefix, "peaks_table.csv"), index=False)
        pd.DataFrame(global_fit["component_rows"]).to_csv(outdir / prefixed_output_name(channel_prefix, "global_components_raw.csv"), index=False)
        start_full_smoothed, _ = smooth_polarimetry_periodogram(
            freqs_pol_full, Ppo_start_full, max(compute_T_full(pol_dt[k]), 1e-8)
        )
        end_full_smoothed, _ = smooth_polarimetry_periodogram(
            freqs_pol_full, Ppo_end_full, max(compute_T_full(pol_dt[k]), 1e-8)
        )
        pd.DataFrame(
            {
                "frequency_cpd": freqs_pol_full,
                "power_start_raw": Ppo_start_full,
                "power_start_for_selection": start_full_smoothed,
                "power_end_raw": Ppo_end_full,
                "power_end_for_selection": end_full_smoothed,
            }
        ).to_csv(
            outdir / prefixed_output_name(channel_prefix, "polarimetry_periodogram_diagnostic.csv"),
            index=False,
        )
        guided_diagnostics_to_dataframe(guided_diags).to_csv(
            outdir / prefixed_output_name(channel_prefix, "guided_search_periodogram_diagnostic.csv"),
            index=False,
        )
        show_table(f"{k} matched_search_table", search_df)
        show_table(f"{k} peaks_table", final_df)
        plot_channel_timeseries_and_spectra(
            tess_dt, tess_final_resid, tess_table,
            pol_dt[k], pol_final_resid, final_df,
            freqs_tess, Pte_start, Pte_end,
            freqs_pol, Ppo_start, Ppo_end,
            freqs_pol_full, Ppo_start_full, Ppo_end_full,
            outdir, guided_diags=guided_diags, show_plots_inline=SHOW_PLOTS_INLINE, file_prefix=channel_prefix
        )
        plot_channel_phased_modes(
            tess_snapshots, tess_table,
            search_df, pol_snapshots, pol_dt[k],
            outdir, n_phase_plots=N_PHASE_PLOTS, sort_by=PHASE_SORT_BY,
            show_plots_inline=SHOW_PLOTS_INLINE, file_prefix=channel_prefix,
            tess0=tess_dt, tess_global_fit=tess_global_fit,
            pol_final_df=final_df, pol_global_fit=global_fit,
        )
        outputs[k] = final_df
    if not quadrature_table.empty:
        outputs["qu_quadrature"] = quadrature_table

    vprint(1, f"All analysis complete in {time.time()-t_all:.1f}s")
    return outputs


def run_analysis():
    """Run Guided analysis and restore mutable frequency limits afterward."""
    requested_fmax = float(FMAX)
    try:
        return _run_analysis_impl()
    finally:
        # The implementation temporarily caps the module-level limit to the
        # current TESS cadence.  Restoration matters for repeat calls in one
        # Python process and also occurs when a run raises an exception.
        globals()["FMAX"] = requested_fmax

if __name__ == "__main__":
    outputs = run_analysis()
    if "tess" in outputs:
        print("\nTESS peaks:")
        print(outputs["tess"].head())

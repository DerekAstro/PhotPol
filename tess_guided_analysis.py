
import argparse
import numpy as np
import pandas as pd
from dataclasses import dataclass
from pathlib import Path
import re
import time
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from IPython.display import display
except Exception:
    display = None

try:
    from scipy.signal import find_peaks, lombscargle as _scipy_lomb
except Exception:
    find_peaks = None
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

# --- TESS-only extraction ---
TESS_OVERSAMPLE = 1.0
TESS_GRID_MODE = "full_baseline"   # "full_baseline" | "longest_chunk"
MAX_TESS_MODES = 20
TESS_SNR_STOP = 4.0
KS_TESS = 15.0
TRIM_TOP_FRAC = 0.10
N_SIDE_BINS = 30
KFIT = 3.0
LOCAL_FIT_STEPS = 1001
TESS_MAX_GRID_POINTS = 60000
PLOT_MAX_GRID_POINTS = 6000
POL_LOCAL_MAX_POINTS = 1201
POL_FULL_RANGE_PLOT_POINTS = 2400
POL_FULL_RANGE_PLOT_OVERSAMPLE = 1.0

# --- Polarimetry guided extraction ---
POL_CHANNELS = ["q", "u", "p"]  # p is analyzed directly from the observed CSV column
POL_SNR_STOP = 2.0
KS_POL = 10.0
POL_OVERSAMPLE = 2.0
POL_SEARCH_WINDOW_MULT = 10.0  # half-window in units of max(1/T_tess, 1/T_pol)
POL_LOCAL_NOISE_KS = 3.0      # sideband half-width for guided local SNR, in units of 1/T
POL_LOCAL_NOISE_SIDE_BINS = 6 # minimum sideband width for guided local SNR, in local-grid bins
MAX_POL_MODES = 99            # hard cap; actual search count is limited by TESS mode list
GUIDED_POL_FMIN = 0.2       # minimum TESS/template frequency [c/d] to test in guided q/u/p matching

# --- Phase-plot summaries ---
PHASE_BIN_N = 16             # number of phase bins for overplotted median points in phased plots

# --- Final global multisinusoid fit ---
GLOBAL_FREQ_BOUND_MULT = 3.0  # each frequency may move by +/- this x frequency resolution
GLOBAL_MAX_NFEV = 300
MIN_FREQ_SEP_MULT = 5.0        # minimum separation for distinct fitted modes, in units of 1/T_full
TESS_DISCOVERY_EXCLUSION_MULT = 5.0  # block re-discovery within this many resolution elements during sequential TESS search

# --- Polarimetry night-model (Option C default; D-ready) ---
USE_POL_NIGHT_OFFSETS = True
USE_POL_NIGHT_SLOPES = False
POL_NIGHT_GROUP_MODE = "gap"     # 'gap' or 'integer_jd'
POL_NIGHT_GAP_HOURS = 8.0

# --- Optional broad detrending hook (defaults off) ---
DO_DETREND = False
DETREND_POLY_ORDER = 0

# --- Phased plots ---
N_PHASE_PLOTS = 3
PHASE_SORT_BY = "amp"            # "amp" | "snr" | "mode"
PHASE_PLOT_STYLE = "isolated_mode"  # "isolated_mode" | "prefit_residual"

# --- Phase reference ---
PHASE_ZERO_MODE = "local_start"   # "local_start" | "btjd_zero" | "custom_btjd"
PHASE_ZERO_BTJD = 0.0
BTJD_OFFSET = 2457000.0

# --- Notebook/script display ---
SHOW_PLOTS_INLINE = True
VERBOSE = 1
LSQ_VERBOSE = 0

# --- Output ---
OUTROOT = Path("tess_guided_phot_pol_outputs_v12")
OUTROOT.mkdir(parents=True, exist_ok=True)

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

TESS_TIME_COL = "time_btjd"
TESS_FORCE_Y_COL = None
TESS_Y_COL_CANDIDATES = ["flux_detrend_rel", "flux_detrended_sub", "flux_detrended_div", "flux_detrended_rel", "flux_medscaled", "flux_rel"]
TESS_PIPELINE_RAW_Y_COLS = ["flux_rel", "flux_detrended_rel"]
TESS_PIPELINE_DETRENDED_Y_COLS = ["flux_detrend_rel", "flux_decor_only_rel", "flux_rel", "flux_detrended_rel"]

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


POL_FILENAME_SUFFIXES = [
    "_IND_nm_pchip_analysis_frame",
    "_nm_pchip_analysis_frame",
    "_analysis_frame",
    "_IND",
]

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

def prefixed_output_name(prefix: str, basename: str) -> str:
    prefix = str(prefix).strip()
    return f"{prefix}_{basename}" if prefix else basename

def _first_present(df: pd.DataFrame, names: list[str]) -> str | None:
    for n in names:
        if n in df.columns:
            return n
    return None

def normalize_raw_polarimetry_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Return a minimal raw-polarimetry dataframe with jd,q,u,p,q_err,u_err,p_err."""
    cols = {}
    for key, names in RAW_POL_CANDIDATES.items():
        col = _first_present(df, names)
        if col is None:
            raise ValueError(f"Raw polarimetry input is missing a column for {key!r}. Looked for {names}. Found: {list(df.columns)}")
        cols[key] = col
    out = pd.DataFrame({k: pd.to_numeric(df[v], errors="coerce") for k, v in cols.items()})
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
    star_label, star_safe = infer_star_labels_from_pol_path(path)
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
        stem = f"{star_safe}_nm_pchip_analysis_frame.csv"
        outdir = path.parent if POL_GENERATED_ANALYSIS_DIR is None else Path(POL_GENERATED_ANALYSIS_DIR)
        outdir.mkdir(parents=True, exist_ok=True)
        outpath = outdir / stem
        df.to_csv(outpath, index=False)
        print(f"[raw pol] wrote generated analysis frame: {outpath}")
    return df

def _clean_sort(t, y, yerr=None, t_abs=None):
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(t) & np.isfinite(y)
    if yerr is not None:
        yerr = np.asarray(yerr, dtype=float)
        yerr = np.where(np.isfinite(yerr) & (yerr > 0), yerr, np.nan)
        m &= np.isfinite(yerr)
        yerr = yerr[m]
    if t_abs is not None:
        t_abs = np.asarray(t_abs, dtype=float)
        m &= np.isfinite(t_abs)
        t_abs = t_abs[m]
    t, y = t[m], y[m]
    idx = np.argsort(t)
    t, y = t[idx], y[idx]
    if yerr is not None:
        yerr = yerr[idx]
    if t_abs is not None:
        t_abs = t_abs[idx]
    return t, y, yerr, t_abs

def _pick_first_existing(columns, candidates):
    for c in candidates:
        if c in columns:
            return c
    return None

def _read_tess_csv_with_candidates(path: Path, y_candidates) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    df = pd.read_csv(path)
    vprint(1, f"Read polarimetry CSV | rows={len(df)}")
    if TESS_TIME_COL not in df.columns:
        raise ValueError(f"{path}: missing '{TESS_TIME_COL}'. Found: {list(df.columns)}")
    forced_col = TESS_FORCE_Y_COL
    if forced_col is not None:
        if forced_col not in df.columns:
            raise ValueError(f"{path}: required TESS column '{forced_col}' not found. Found: {list(df.columns)}")
        y_col = forced_col
    else:
        y_col = _pick_first_existing(df.columns, y_candidates)
        if y_col is None:
            raise ValueError(f"{path}: missing any of {list(y_candidates)}. Found: {list(df.columns)}")
    vprint(1, f"Using TESS file: {path} | flux column: {y_col}")
    t_abs = df[TESS_TIME_COL].to_numpy(dtype=float)
    y = df[y_col].to_numpy(dtype=float)
    t_rel = t_abs - np.nanmin(t_abs)
    t_rel, y, _, t_abs = _clean_sort(t_rel, y, None, t_abs=t_abs)
    return t_rel, y, t_abs, y_col

def load_tess_csv(path: Path) -> TimeSeries:
    _, y, t_abs, y_col = _read_tess_csv_with_candidates(path, TESS_Y_COL_CANDIDATES)
    t_rel = t_abs - np.nanmin(t_abs)
    t_rel, y, _, t_abs = _clean_sort(t_rel, y, None, t_abs=t_abs)
    return TimeSeries(t=t_rel, y=y, yerr=None, name=f"tess({y_col})", t_abs=t_abs, group_id=None)

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

    t_all, y_all, file_names, used_cols = [], [], [], []
    skipped = []
    for path in files:
        try:
            _, y, t_abs, y_col = _read_tess_csv_with_candidates(path, candidates)
        except Exception as exc:
            skipped.append((path.name, str(exc)))
            continue
        if len(t_abs) == 0:
            skipped.append((path.name, "no finite rows after cleaning"))
            continue
        t_all.append(t_abs)
        y_all.append(y)
        file_names.append(path.name)
        used_cols.append(y_col)

    if not t_all:
        msg = f"No usable TESS pipeline CSVs found in {dirpath} matching {pattern!r}"
        if skipped:
            msg += ". First skip: " + skipped[0][0] + " -> " + skipped[0][1]
        raise ValueError(msg)

    t_abs = np.concatenate(t_all)
    y = np.concatenate(y_all)
    t_rel = t_abs - np.nanmin(t_abs)
    t_rel, y, _, t_abs = _clean_sort(t_rel, y, None, t_abs=t_abs)

    print(f"Loaded {len(file_names)} TESS pipeline CSV file(s) from {dirpath}")
    print("  flux columns used:", sorted(set(used_cols)))
    if skipped:
        print(f"  skipped {len(skipped)} file(s) that did not match the requested format")
        for name, reason in skipped[:5]:
            print("   -", name, "->", reason)

    return TimeSeries(
        t=t_rel, y=y, yerr=None,
        name=f"tess_pipeline({mode}; {len(file_names)} files)",
        t_abs=t_abs, group_id=None
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

def build_night_groups(t_abs: np.ndarray, mode: str = "gap", gap_days: float = 8.0/24.0) -> np.ndarray:
    t_abs = np.asarray(t_abs, dtype=float)
    if t_abs.size == 0:
        return np.array([], dtype=int)
    if mode == "integer_jd":
        _, inv = np.unique(np.floor(t_abs).astype(int), return_inverse=True)
        return inv.astype(int)
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

def load_polarimetry_csv(path: Path, product: str, trend_cfg: NightTrendConfig) -> dict:
    """Load either a precomputed *_analysis_frame.csv or a raw/basic polarimetry CSV."""
    star_label, _star_safe = infer_star_labels_from_pol_path(path)
    if product not in POL_PRODUCTS:
        raise ValueError(f"POL_PRODUCT must be one of {list(POL_PRODUCTS.keys())}. Got: {product}")

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
    required += [POL_PRODUCTS[product][k] for k in ["q", "u", "p"]]
    required += [POL_ERR_COLS[k] for k in ["q", "u", "p"]]
    if all(c in df.columns for c in required):
        t_abs = df[POL_TIME_COL].to_numpy(dtype=float)
        t0 = np.nanmin(t_abs)
        t_rel = t_abs - t0
        out = {}
        for k in ["q", "u", "p"]:
            y = df[POL_PRODUCTS[product][k]].to_numpy(dtype=float)
            yerr = df[POL_ERR_COLS[k]].to_numpy(dtype=float)
            tt, yy, ee, ta = _clean_sort(t_rel, y, yerr, t_abs=t_abs)
            gid = build_night_groups(ta, mode=trend_cfg.group_mode, gap_days=trend_cfg.gap_days)
            out[k] = TimeSeries(t=tt, y=yy, yerr=ee, name=f"{star_label} {k}", t_abs=ta, group_id=gid)
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
    required2 += [POL_PRODUCTS[product][k] for k in ["q", "u", "p"]]
    required2 += [POL_ERR_COLS[k] for k in ["q", "u", "p"]]
    missing2 = [c for c in required2 if c not in df_proc.columns]
    if missing2:
        raise ValueError(f"Processed polarimetry dataframe is missing columns: {missing2}")

    t_abs = df_proc[POL_TIME_COL].to_numpy(dtype=float)
    t0 = np.nanmin(t_abs)
    t_rel = t_abs - t0
    out = {}
    for k in ["q", "u", "p"]:
        y = df_proc[POL_PRODUCTS[product][k]].to_numpy(dtype=float)
        yerr = df_proc[POL_ERR_COLS[k]].to_numpy(dtype=float)
        tt, yy, ee, ta = _clean_sort(t_rel, y, yerr, t_abs=t_abs)
        gid = build_night_groups(ta, mode=trend_cfg.group_mode, gap_days=trend_cfg.gap_days)
        out[k] = TimeSeries(t=tt, y=yy, yerr=ee, name=f"{star_label} {k}", t_abs=ta, group_id=gid)
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

def make_local_frequency_grid(f_center: float, half_window: float, df: float,
                              fmin: float = FMIN, fmax: float = FMAX,
                              max_points: int = POL_LOCAL_MAX_POINTS) -> np.ndarray:
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


def weighted_linear_solve_with_cov(y: np.ndarray, X: np.ndarray, w: np.ndarray | None = None) -> dict:
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
    try:
        xtwx = Xw.T @ Xw
        cov = np.linalg.pinv(xtwx)
        dof = max(int(np.count_nonzero(w > 0)) - X.shape[1], 1)
        sigma2 = rss / float(dof)
        cov = cov * sigma2
    except Exception:
        cov = np.full((X.shape[1], X.shape[1]), np.nan, dtype=float)
    return {"beta": beta, "yhat": yhat, "rss": rss, "resid": resid, "cov": cov}

def amp_uncertainty_from_cov(s_coeff: float, c_coeff: float, cov_sc: np.ndarray | None) -> float:
    if cov_sc is None:
        return np.nan
    cov_sc = np.asarray(cov_sc, dtype=float)
    if cov_sc.shape != (2, 2) or not np.isfinite(cov_sc).any():
        return np.nan
    amp = float(np.hypot(s_coeff, c_coeff))
    if amp > 0 and np.isfinite(amp):
        grad = np.array([s_coeff / amp, c_coeff / amp], dtype=float)
    else:
        grad = np.array([1.0 / np.sqrt(2.0), 1.0 / np.sqrt(2.0)], dtype=float)
    try:
        var = float(grad @ cov_sc @ grad)
    except Exception:
        return np.nan
    if not np.isfinite(var) or var < 0:
        return np.nan
    return float(np.sqrt(var))


def phase_uncertainty_from_cov(s_coeff: float, c_coeff: float, cov_sc: np.ndarray | None) -> float:
    if cov_sc is None:
        return np.nan
    cov_sc = np.asarray(cov_sc, dtype=float)
    if cov_sc.shape != (2, 2) or not np.isfinite(cov_sc).any():
        return np.nan
    amp2 = float(s_coeff * s_coeff + c_coeff * c_coeff)
    if not np.isfinite(amp2) or amp2 <= 0:
        return np.nan
    grad = np.array([-c_coeff / amp2, s_coeff / amp2], dtype=float)
    try:
        var = float(grad @ cov_sc @ grad)
    except Exception:
        return np.nan
    if not np.isfinite(var) or var < 0:
        return np.nan
    return float(np.sqrt(var))

def wrap_phase_diff(phi_a, phi_b):
    return _wrap_phase_radians(np.asarray(phi_a, dtype=float) - np.asarray(phi_b, dtype=float))

def phase_diff_error(err_a, err_b):
    ea = pd.to_numeric(pd.Series(err_a), errors="coerce").to_numpy(dtype=float)
    eb = pd.to_numeric(pd.Series(err_b), errors="coerce").to_numpy(dtype=float)
    ea, eb = np.broadcast_arrays(ea, eb)
    out = np.full(ea.shape, np.nan, dtype=float)
    m = np.isfinite(ea) & np.isfinite(eb)
    out[m] = np.sqrt(ea[m] ** 2 + eb[m] ** 2)
    return out

def merge_tess_phase_info(df: pd.DataFrame, tess_table: pd.DataFrame) -> pd.DataFrame:
    if df is None or tess_table is None or len(df) == 0 or len(tess_table) == 0 or "tess_mode" not in df.columns or "mode" not in tess_table.columns:
        return df
    info_cols = [c for c in ["mode", "phase", "phase_err", "f"] if c in tess_table.columns]
    info = tess_table[info_cols].copy().rename(columns={"mode":"tess_mode","phase":"phase_tess","phase_err":"phase_tess_err","f":"f_tess_global"})
    out = df.merge(info, on="tess_mode", how="left")
    if "phase_local" in out.columns and "phase_tess" in out.columns:
        out["phase_diff_local_tess"] = wrap_phase_diff(out["phase_local"], out["phase_tess"])
    if "phase_local_err" in out.columns and "phase_tess_err" in out.columns:
        out["phase_diff_local_tess_err"] = phase_diff_error(out["phase_local_err"], out["phase_tess_err"])
    return out


def safe_period_days(freq):
    arr = pd.to_numeric(pd.Series(freq), errors="coerce").to_numpy(dtype=float)
    out = np.full(arr.shape, np.nan, dtype=float)
    m = np.isfinite(arr) & (arr > 0)
    out[m] = 1.0 / arr[m]
    return out

def add_period_columns(df: pd.DataFrame, mappings: list[tuple[str, str]]) -> pd.DataFrame:
    if df is None:
        return df
    out = df.copy()
    insert_offset = 0
    for fcol, pcol in mappings:
        if fcol not in out.columns:
            continue
        period = safe_period_days(out[fcol])
        loc = out.columns.get_loc(fcol) + 1 + insert_offset
        if pcol in out.columns:
            out[pcol] = period
        else:
            out.insert(loc, pcol, period)
        insert_offset += 1
    return out

def amplitude_spectrum_guided(ts: TimeSeries, freqs: np.ndarray, baseline_matrix: np.ndarray) -> np.ndarray:
    freqs = np.asarray(freqs, dtype=float)
    amps = np.full_like(freqs, np.nan, dtype=float)
    w = signal_weights(ts)
    for i, f in enumerate(freqs):
        X = design_matrix_multi(ts.t, np.array([float(f)]), baseline_matrix)
        fit = weighted_linear_solve_with_cov(ts.y, X, w)
        beta = np.asarray(fit["beta"], dtype=float)
        if beta.size >= 2:
            amps[i] = float(np.hypot(beta[0], beta[1]))
    return amps

def component_amplitude_error_rows(ts: TimeSeries, freqs: np.ndarray, baseline_matrix: np.ndarray) -> pd.DataFrame:
    freqs = np.asarray(freqs, dtype=float)
    if freqs.size == 0:
        return pd.DataFrame(columns=["f", "amp_err"])
    w = signal_weights(ts)
    X = design_matrix_multi(ts.t, freqs, baseline_matrix)
    fit = weighted_linear_solve_with_cov(ts.y, X, w)
    beta = np.asarray(fit["beta"], dtype=float)
    cov = np.asarray(fit.get("cov", np.full((len(beta), len(beta)), np.nan)), dtype=float)
    rows = []
    for i, f in enumerate(freqs):
        s_coeff = float(beta[2 * i])
        c_coeff = float(beta[2 * i + 1])
        cov_sc = cov[2 * i:2 * i + 2, 2 * i:2 * i + 2] if cov.shape[0] >= 2 * i + 2 else None
        rows.append({"f": float(f), "amp_err": amp_uncertainty_from_cov(s_coeff, c_coeff, cov_sc)})
    return pd.DataFrame(rows)

def _summary_xlim_from_table(summary_df: pd.DataFrame, freq_cols: list[str]) -> float:
    vals = []
    for c in freq_cols:
        if c in summary_df.columns:
            arr = pd.to_numeric(summary_df[c], errors="coerce").to_numpy(dtype=float)
            arr = arr[np.isfinite(arr)]
            if arr.size:
                vals.append(arr)
    if not vals:
        return min(10.0, float(FMAX))
    vmax = float(np.nanmax(np.concatenate(vals)))
    if not np.isfinite(vmax):
        return min(10.0, float(FMAX))
    return min(float(FMAX), max(10.0, 1.15 * vmax))

def _summary_ylim_from_amp(amp: np.ndarray) -> float:
    arr = np.asarray(amp, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 1.0
    hi = np.nanpercentile(arr, 99.5)
    if not np.isfinite(hi) or hi <= 0:
        hi = np.nanmax(arr)
    if not np.isfinite(hi) or hi <= 0:
        hi = 1.0
    return 1.15 * float(hi)

def build_guided_summary_table(tess_table: pd.DataFrame, channel_tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    cols = [
        "mode", "tess_f_cd", "tess_period_d", "tess_amp_ppt", "tess_amp_err_ppt", "tess_phase_rad", "tess_phase_err_rad",
        "q_f_cd", "q_period_d", "q_amp_ppm", "q_amp_err_ppm", "q_phase_rad", "q_phase_err_rad", "q_phase_diff_rad", "q_phase_diff_err_rad",
        "u_f_cd", "u_period_d", "u_amp_ppm", "u_amp_err_ppm", "u_phase_rad", "u_phase_err_rad", "u_phase_diff_rad", "u_phase_diff_err_rad",
        "p_f_cd", "p_period_d", "p_amp_ppm", "p_amp_err_ppm", "p_phase_rad", "p_phase_err_rad", "p_phase_diff_rad", "p_phase_diff_err_rad",
    ]
    if tess_table is None or tess_table.empty:
        return pd.DataFrame(columns=cols)
    tdf = tess_table.copy()
    if "snr_local" in tdf.columns:
        tdf = tdf.loc[pd.to_numeric(tdf["snr_local"], errors="coerce") >= float(TESS_SNR_STOP)].copy()
    tdf = tdf.sort_values("mode").reset_index(drop=True)
    rows = []
    for _, tr in tdf.iterrows():
        mode = int(tr["mode"])
        tf = float(tr.get("f", np.nan)) if np.isfinite(tr.get("f", np.nan)) else np.nan
        ta = float(tr.get("amp", np.nan)) if np.isfinite(tr.get("amp", np.nan)) else np.nan
        tae = float(tr.get("amp_err", np.nan)) if np.isfinite(tr.get("amp_err", np.nan)) else np.nan
        tph = float(tr.get("phase", np.nan)) if np.isfinite(tr.get("phase", np.nan)) else np.nan
        tphe = float(tr.get("phase_err", np.nan)) if np.isfinite(tr.get("phase_err", np.nan)) else np.nan
        row = {"mode": mode, "tess_f_cd": tf,
               "tess_period_d": np.nan if not (np.isfinite(tf) and tf > 0) else 1.0 / tf,
               "tess_amp_ppt": np.nan if not np.isfinite(ta) else 1000.0 * ta,
               "tess_amp_err_ppt": np.nan if not np.isfinite(tae) else 1000.0 * tae,
               "tess_phase_rad": tph, "tess_phase_err_rad": tphe}
        for ch in ["q","u","p"]:
            df = channel_tables.get(ch)
            fcol=f"{ch}_f_cd"; pcol=f"{ch}_period_d"; acol=f"{ch}_amp_ppm"; aecol=f"{ch}_amp_err_ppm"
            phcol=f"{ch}_phase_rad"; phecol=f"{ch}_phase_err_rad"; pdcol=f"{ch}_phase_diff_rad"; pdecol=f"{ch}_phase_diff_err_rad"
            row.update({fcol:np.nan,pcol:np.nan,acol:np.nan,aecol:np.nan,phcol:np.nan,phecol:np.nan,pdcol:np.nan,pdecol:np.nan})
            if df is None or len(df)==0:
                continue
            hits = df.loc[pd.to_numeric(df.get("tess_mode"), errors="coerce") == mode]
            if hits.empty:
                continue
            pr = hits.iloc[0]
            pf = float(pr.get("f", np.nan)) if np.isfinite(pr.get("f", np.nan)) else np.nan
            pa = float(pr.get("amp", np.nan)) if np.isfinite(pr.get("amp", np.nan)) else np.nan
            pae = float(pr.get("amp_err", np.nan)) if np.isfinite(pr.get("amp_err", np.nan)) else np.nan
            pph = float(pr.get("phase_local", np.nan)) if np.isfinite(pr.get("phase_local", np.nan)) else np.nan
            pphe = float(pr.get("phase_local_err", np.nan)) if np.isfinite(pr.get("phase_local_err", np.nan)) else np.nan
            row[fcol]=pf; row[pcol]=np.nan if not (np.isfinite(pf) and pf>0) else 1.0/pf; row[acol]=pa; row[aecol]=pae
            row[phcol]=pph; row[phecol]=pphe
            if np.isfinite(pph) and np.isfinite(tph):
                row[pdcol]=float(wrap_phase_diff(pph, tph))
            if np.isfinite(pphe) and np.isfinite(tphe):
                row[pdecol]=float((pphe**2 + tphe**2) ** 0.5)
        rows.append(row)
    return pd.DataFrame(rows, columns=cols)



def _summary_period_xlim_from_table(summary_df: pd.DataFrame, period_cols: list[str]) -> float:
    vals = []
    for c in period_cols:
        if c in summary_df.columns:
            arr = pd.to_numeric(summary_df[c], errors="coerce").to_numpy(dtype=float)
            arr = arr[np.isfinite(arr)]
            if arr.size:
                vals.append(arr)
    if not vals:
        return max(2.0, 1.0 / max(float(FMAX), 1e-8))
    vmax = float(np.nanmax(np.concatenate(vals)))
    if not np.isfinite(vmax) or vmax <= 0:
        return max(2.0, 1.0 / max(float(FMAX), 1e-8))
    return 1.15 * vmax

def _plot_summary_spectrum_panels(summary_df: pd.DataFrame, panels: list[tuple[np.ndarray, np.ndarray, str, str, str, str]],
                                  outpath: Path, x_mode: str = "frequency", y_scale: str = "linear",
                                  show_plots_inline: bool = False):
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(11, 9.5), sharex=True)
    x_mode = str(x_mode).lower()
    y_scale = str(y_scale).lower()
    xmax = _summary_period_xlim_from_table(summary_df, ["tess_period_d","q_period_d","u_period_d","p_period_d"]) if x_mode=="period" else _summary_xlim_from_table(summary_df, ["tess_f_cd","q_f_cd","u_f_cd","p_f_cd"])
    for ax, (freqs, amp, fcol, pcol, ylabel, title) in zip(axes, panels):
        freqs = np.asarray(freqs, dtype=float); amp = np.asarray(amp, dtype=float)
        if x_mode == "period":
            m = np.isfinite(freqs) & (freqs > 0) & np.isfinite(amp)
            x = 1.0 / freqs[m]; y = amp[m]
            order = np.argsort(x); x = x[order]; y = y[order]
            line_vals = pd.to_numeric(summary_df.get(pcol), errors="coerce").to_numpy(dtype=float) if pcol in summary_df.columns else np.array([], dtype=float)
        else:
            m = np.isfinite(freqs) & np.isfinite(amp)
            x = freqs[m]; y = amp[m]
            line_vals = pd.to_numeric(summary_df.get(fcol), errors="coerce").to_numpy(dtype=float) if fcol in summary_df.columns else np.array([], dtype=float)
        if y_scale == "log":
            m2 = np.isfinite(y) & (y > 0); x = x[m2]; y = y[m2]
            ax.semilogy(x, y, lw=0.9, color="k")
            if y.size:
                ymin = max(np.nanpercentile(y, 2.0), np.nanmin(y[y>0]))
                ymax = max(np.nanpercentile(y, 99.5) * 1.2, ymin * 5.0)
                ax.set_ylim(ymin, ymax)
        else:
            ax.plot(x, y, lw=0.9, color="k")
            ax.set_ylim(0.0, _summary_ylim_from_amp(y))
        for f in line_vals:
            if np.isfinite(f):
                ax.axvline(float(f), color="red", lw=0.8)
        ax.set_ylabel(ylabel); ax.set_title(title)
        ax.set_xlim(0.0 if x_mode=="period" else float(FMIN), xmax)
        ax.grid(alpha=0.15)
    axes[-1].set_xlabel("Period [d]" if x_mode=="period" else "Frequency [c/d]")
    fig.tight_layout()
    fig.savefig(outpath, dpi=180, bbox_inches="tight")
    if show_plots_inline:
        plt.show()
    plt.close(fig)

def plot_guided_summary_amplitude_spectra(summary_df: pd.DataFrame, tess_ts: TimeSeries, pol_dict: dict[str, TimeSeries],
                                          trend_cfg: NightTrendConfig, outpath: Path,
                                          show_plots_inline: bool = False):
    if summary_df is None or summary_df.empty:
        return
    freqs_t = make_fullbaseline_plot_frequency_grid(tess_ts, oversample=POL_FULL_RANGE_PLOT_OVERSAMPLE, max_points=POL_FULL_RANGE_PLOT_POINTS)
    amp_t = 1000.0 * amplitude_spectrum_guided(tess_ts, freqs_t, make_tess_baseline_matrix(tess_ts))
    q_ts = pol_dict.get("q"); u_ts = pol_dict.get("u")
    if q_ts is None or u_ts is None:
        return
    freqs_q = make_fullbaseline_plot_frequency_grid(q_ts, oversample=POL_FULL_RANGE_PLOT_OVERSAMPLE, max_points=POL_FULL_RANGE_PLOT_POINTS)
    amp_q = amplitude_spectrum_guided(q_ts, freqs_q, make_pol_baseline_matrix(q_ts, trend_cfg))
    freqs_u = make_fullbaseline_plot_frequency_grid(u_ts, oversample=POL_FULL_RANGE_PLOT_OVERSAMPLE, max_points=POL_FULL_RANGE_PLOT_POINTS)
    amp_u = amplitude_spectrum_guided(u_ts, freqs_u, make_pol_baseline_matrix(u_ts, trend_cfg))
    panels = [(freqs_t, amp_t, "tess_f_cd", "tess_period_d", "Amplitude (ppt)", "TESS photometry"),
              (freqs_q, amp_q, "q_f_cd", "q_period_d", "Amplitude (ppm)", "q polarimetry"),
              (freqs_u, amp_u, "u_f_cd", "u_period_d", "Amplitude (ppm)", "u polarimetry")]
    _plot_summary_spectrum_panels(summary_df, panels, outpath, x_mode="frequency", y_scale="linear", show_plots_inline=show_plots_inline)
    stem = outpath.stem; suffix = outpath.suffix or ".png"; parent = outpath.parent
    if SUMMARY_SAVE_PERIOD_VERSION:
        _plot_summary_spectrum_panels(summary_df, panels, parent / f"{stem}_period{suffix}", x_mode="period", y_scale="linear", show_plots_inline=show_plots_inline)
    if SUMMARY_SAVE_LOG_AMPLITUDE_VERSION:
        _plot_summary_spectrum_panels(summary_df, panels, parent / f"{stem}_log{suffix}", x_mode="frequency", y_scale="log", show_plots_inline=show_plots_inline)
    if SUMMARY_SAVE_PERIOD_VERSION and SUMMARY_SAVE_LOG_AMPLITUDE_VERSION:
        _plot_summary_spectrum_panels(summary_df, panels, parent / f"{stem}_period_log{suffix}", x_mode="period", y_scale="log", show_plots_inline=show_plots_inline)


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

def local_noise_floor(freqs: np.ndarray, power: np.ndarray, f0: float, T: float,
                      kfit: float, ks: float, trim_top_frac: float,
                      n_side_bins: int | None = None) -> float:
    if len(freqs) < 5 or not np.isfinite(T) or T <= 0:
        return np.nan
    df_grid = np.nanmedian(np.diff(freqs)) if len(freqs) > 1 else np.nan
    if not np.isfinite(df_grid) or df_grid <= 0:
        return np.nan
    delta_fit = kfit * (2.0 / T)
    delta_side = ks * (1.0 / T)
    if n_side_bins is None:
        n_side_bins = N_SIDE_BINS
    min_width = max(delta_side, max(int(n_side_bins), 1) * df_grid)
    lo1, hi1 = f0 - (delta_fit + min_width), f0 - delta_fit
    lo2, hi2 = f0 + delta_fit, f0 + (delta_fit + min_width)
    m = ((freqs >= lo1) & (freqs <= hi1)) | ((freqs >= lo2) & (freqs <= hi2))
    vals = power[m]
    noise = trimmed_median(vals, trim_top_frac=trim_top_frac)
    if not np.isfinite(noise) or noise <= 0:
        # Fallback: use the local window excluding the central fit zone, so the peak
        # itself does not set its own noise floor when the guided search window is small.
        m2 = np.isfinite(power) & (np.abs(freqs - f0) >= delta_fit)
        vals2 = power[m2]
        noise = trimmed_median(vals2, trim_top_frac=trim_top_frac)
    if not np.isfinite(noise) or noise <= 0:
        finite = power[np.isfinite(power)]
        noise = trimmed_median(finite, trim_top_frac=trim_top_frac)
    return noise

def local_snr_from_power(freqs: np.ndarray, power: np.ndarray, f_fit: float, T: float, ks: float,
                         n_side_bins: int | None = None, trim_top_frac: float = TRIM_TOP_FRAC) -> tuple[float, float]:
    noise = local_noise_floor(freqs, power, f_fit, T=T, kfit=KFIT, ks=ks,
                              trim_top_frac=trim_top_frac, n_side_bins=n_side_bins)
    j = int(np.argmin(np.abs(freqs - f_fit)))
    p0 = float(power[j])
    W = float(p0 / noise) if (np.isfinite(noise) and noise > 0) else np.nan
    snr = float(np.sqrt(W)) if np.isfinite(W) and W > 0 else np.nan
    return snr, W

def simple_find_peaks(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(y)
    peaks = []
    for i in range(1, len(y) - 1):
        if ok[i - 1] and ok[i] and ok[i + 1] and (y[i] > y[i - 1]) and (y[i] > y[i + 1]):
            peaks.append(i)
    return np.asarray(peaks, dtype=int)

def pick_top_peaks(freqs: np.ndarray, y: np.ndarray, k: int, min_sep: float) -> pd.DataFrame:
    y = np.asarray(y, dtype=float)
    finite = np.isfinite(y)
    fill_high = np.nanmax(y[finite]) if finite.any() else -np.inf
    y_clean = np.nan_to_num(y, nan=-np.inf, posinf=fill_high, neginf=-np.inf)
    df = np.nanmedian(np.diff(freqs))
    min_dist = int(np.ceil(min_sep / df)) if np.isfinite(df) and df > 0 else 1
    min_dist = max(1, min_dist)
    if find_peaks is not None:
        peaks, props = find_peaks(y_clean, distance=min_dist, prominence=True)
        prom = props.get("prominences", np.full_like(peaks, np.nan, dtype=float))
    else:
        peaks = simple_find_peaks(y_clean)
        prom = []
        win = max(3, min_dist * 2)
        for p in peaks:
            lo = max(0, p - win)
            hi = min(len(y), p + win + 1)
            base = np.nanmedian(y_clean[lo:hi])
            prom.append(y_clean[p] - base)
        prom = np.asarray(prom, dtype=float)
    if peaks.size == 0:
        return pd.DataFrame(columns=["idx", "f", "height", "prominence"])
    dfp = pd.DataFrame({"idx": peaks, "f": freqs[peaks], "height": y_clean[peaks], "prominence": prom})
    return dfp.sort_values(["height", "prominence"], ascending=False).head(k).reset_index(drop=True)

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
        fit = weighted_linear_solve_with_cov(ts.y, X, w)
        fits[i] = fit
        rss[i] = fit["rss"]
    j = int(np.nanargmin(rss))
    f_best = float(freqs[j])
    fit_best = fits[j]
    beta = np.asarray(fit_best["beta"], dtype=float)
    s_coeff, c_coeff = float(beta[0]), float(beta[1])
    amp, phase = sc_to_amp_phase(s_coeff, c_coeff)
    cov_sc = np.asarray(fit_best.get("cov", np.full((2, 2), np.nan)), dtype=float)[:2, :2]
    amp_err = amp_uncertainty_from_cov(s_coeff, c_coeff, cov_sc)
    phase_err = phase_uncertainty_from_cov(s_coeff, c_coeff, cov_sc)
    signal_model = component_signal(ts.t, f_best, s_coeff, c_coeff)
    baseline_model = baseline_from_beta(beta, 1, baseline_matrix)
    full_model = signal_model + baseline_model
    return {
        "best_f": f_best, "amp": amp, "phase": phase, "amp_err": amp_err, "phase_err": phase_err,
        "rss": float(fit_best["rss"]), "beta": beta, "signal_model": signal_model,
        "baseline_model": baseline_model, "full_model": full_model, "resid": ts.y - full_model,
        "df_fit": float(df_fit), "s_coeff": s_coeff, "c_coeff": c_coeff,
        "cov": np.asarray(fit_best.get("cov", np.full((len(beta), len(beta)), np.nan)), dtype=float),
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
        fit = weighted_linear_solve_with_cov(ts.y, baseline_matrix, w)
        return {"success": True, "message": "no frequencies", "freqs": np.array([], dtype=float), "beta": fit["beta"],
                "cov": np.asarray(fit.get("cov", np.full((baseline_matrix.shape[1], baseline_matrix.shape[1]), np.nan)), dtype=float),
                "rss": float(fit["rss"]), "full_model": fit["yhat"], "signal_model": np.zeros_like(ts.y),
                "baseline_model": fit["yhat"], "resid": fit["resid"], "component_rows": [],
                "seed_freqs_used": np.array([], dtype=float), "min_sep": min_sep}

    seed_df = pd.DataFrame({"f_seed": seed_freqs, "seed_rank": np.arange(len(seed_freqs)), "snr_local": np.arange(len(seed_freqs), 0, -1, dtype=float)})
    seed_df = dedupe_frequency_rows(seed_df, "f_seed", score_cols=("snr_local",), min_sep=min_sep)
    seed_freqs = seed_df["f_seed"].to_numpy(dtype=float) if not seed_df.empty else np.array([], dtype=float)
    if seed_freqs.size == 0:
        fit = weighted_linear_solve_with_cov(ts.y, baseline_matrix, w)
        return {"success": True, "message": "all seed frequencies merged away", "freqs": np.array([], dtype=float), "beta": fit["beta"],
                "cov": np.asarray(fit.get("cov", np.full((baseline_matrix.shape[1], baseline_matrix.shape[1]), np.nan)), dtype=float),
                "rss": float(fit["rss"]), "full_model": fit["yhat"], "signal_model": np.zeros_like(ts.y),
                "baseline_model": fit["yhat"], "resid": fit["resid"], "component_rows": [],
                "seed_freqs_used": np.array([], dtype=float), "min_sep": min_sep}

    delta = max(bound_mult * float(freq_resolution), 1e-8)
    lb = np.maximum(FMIN, seed_freqs - delta)
    ub = np.minimum(FMAX, seed_freqs + delta)

    def solve_linear(freqs_now):
        X = design_matrix_multi(ts.t, freqs_now, baseline_matrix)
        return weighted_linear_solve_with_cov(ts.y, X, w)

    vprint(1, f"[{ts.name}] starting global multisin fit | npts={len(ts.t)} | nmodes={seed_freqs.size} | baseline_cols={baseline_matrix.shape[1]} | df={freq_resolution:.6g} c/d | bound=±{delta:.6g} c/d | min_sep={min_sep:.6g} c/d")
    t0 = time.time()
    nfev_limit = None if GLOBAL_MAX_NFEV is None else int(GLOBAL_MAX_NFEV)

    if _HAVE_LSQ and seed_freqs.size > 0:
        def fun(freqs_now):
            fit = solve_linear(freqs_now)
            sw = np.sqrt(w)
            return fit["resid"] * sw
        try:
            opt = least_squares(fun, x0=seed_freqs, bounds=(lb, ub), method="trf",
                                xtol=1e-12, ftol=1e-12, gtol=1e-12, max_nfev=nfev_limit, verbose=LSQ_VERBOSE)
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
    cov = np.asarray(fit.get("cov", np.full((len(beta), len(beta)), np.nan)), dtype=float)
    freqs_keep, keep_idx, groups = collapse_close_frequency_components(freqs_best, beta, min_sep=min_sep)
    collapsed = any(len(g) > 1 for g in groups)
    if collapsed:
        ngroup = sum(len(g) > 1 for g in groups)
        message = (message + f" | merged {ngroup} close-frequency group(s)").strip()
        vprint(1, f"[{ts.name}] merged {ngroup} close-frequency group(s) with min_sep={min_sep:.6g} c/d")
        fit = solve_linear(freqs_keep)
        beta = np.asarray(fit["beta"], dtype=float)
        cov = np.asarray(fit.get("cov", np.full((len(beta), len(beta)), np.nan)), dtype=float)
        freqs_best = np.asarray(freqs_keep, dtype=float)

    nmode = freqs_best.size
    signal_model = signal_from_beta(ts.t, freqs_best, beta)
    baseline_model = baseline_from_beta(beta, nmode, baseline_matrix)
    rows = []
    for i, f in enumerate(freqs_best):
        s_coeff = float(beta[2 * i]); c_coeff = float(beta[2 * i + 1])
        amp, phase = sc_to_amp_phase(s_coeff, c_coeff)
        cov_sc = cov[2 * i:2 * i + 2, 2 * i:2 * i + 2] if cov.shape[0] >= 2 * i + 2 else None
        amp_err = amp_uncertainty_from_cov(s_coeff, c_coeff, cov_sc)
        phase_err = phase_uncertainty_from_cov(s_coeff, c_coeff, cov_sc)
        rows.append({"mode": i + 1, "f": float(f), "s_coeff": s_coeff, "c_coeff": c_coeff,
                     "amp": amp, "amp_err": amp_err, "phase": phase, "phase_err": phase_err})
    return {"success": success, "message": message, "freqs": freqs_best, "beta": beta, "cov": cov,
            "rss": float(fit["rss"]), "full_model": signal_model + baseline_model, "signal_model": signal_model,
            "baseline_model": baseline_model, "resid": fit["resid"], "component_rows": rows,
            "seed_freqs_used": seed_freqs, "min_sep": min_sep}


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
        snr, W = local_snr_from_power(freqs, power, fit["best_f"], T=Tseg, ks=KS_TESS)
        if (not np.isfinite(snr)) or (snr < TESS_SNR_STOP):
            print(f"[TESS] stopping at iter {n}: SNR={snr:.2f} < {TESS_SNR_STOP:.2f}")
            break
        rows.append({
            "mode": n,
            "f_local": fit["best_f"],
            "amp_local": fit["amp"],
            "amp_local_err": fit.get("amp_err", np.nan),
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
        return pd.DataFrame(columns=["mode", "f_local", "amp_local", "amp_local_err", "phase_local", "phase_local_err", "snr_local", "W_local",
                                     "f", "amp", "amp_err", "phase", "phase_err", "s_coeff", "c_coeff"])
    g_rows = pd.DataFrame(global_fit["component_rows"])
    if g_rows.empty:
        out = local_df.copy()
        out["f"] = np.nan
        out["amp"] = np.nan
        out["phase"] = np.nan
        out["phase_err"] = np.nan
        out["amp_err"] = np.nan
        out["s_coeff"] = np.nan
        out["c_coeff"] = np.nan
        return out

    remaining = list(range(len(g_rows)))
    merged = []
    for _, lr in local_df.iterrows():
        f0 = float(lr["f_local"])
        if not remaining:
            grow = {k: np.nan for k in ["f", "amp", "amp_err", "phase", "s_coeff", "c_coeff"]}
        else:
            idx = min(remaining, key=lambda j: abs(float(g_rows.iloc[j]["f"]) - f0))
            grow = g_rows.iloc[idx].to_dict()
            remaining.remove(idx)
        merged.append({**lr.to_dict(), **grow})
    return pd.DataFrame(merged)

def search_guided_channel(pol0: TimeSeries, tess_table: pd.DataFrame, tess_freq_resolution: float,
                          trend_cfg: NightTrendConfig) -> tuple[pd.DataFrame, pd.DataFrame, dict, list[dict], TimeSeries, np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    pol = TimeSeries(
        t=pol0.t.copy(), y=pol0.y.copy(),
        yerr=None if pol0.yerr is None else pol0.yerr.copy(),
        name=pol0.name, t_abs=None if pol0.t_abs is None else pol0.t_abs.copy(),
        group_id=None if pol0.group_id is None else pol0.group_id.copy()
    )
    Tfull_pol = max(compute_T_full(pol0), 1e-8)
    template_df = tess_table.copy().sort_values("mode").reset_index(drop=True)
    search_half_window = POL_SEARCH_WINDOW_MULT * max(float(tess_freq_resolution), 1.0 / Tfull_pol)
    df_local = max(min(float(tess_freq_resolution), 1.0 / Tfull_pol) / max(POL_OVERSAMPLE, 1e-8), 1e-8)
    print(f"[{pol0.name}] guided search: npts={len(pol0.t)} | Tfull={Tfull_pol:.3f} d | template_modes={len(template_df)} | half_window={search_half_window:.6g} c/d | df_local={df_local:.6g} c/d")

    baseline0 = make_pol_baseline_matrix(pol0, trend_cfg)
    search_rows = []
    snapshots = []

    guided_pol_fmin = None if GUIDED_POL_FMIN is None else float(GUIDED_POL_FMIN)

    for i_tr, (_, tr) in enumerate(template_df.iterrows(), start=1):
        mode_n = int(tr["mode"])
        f_tess = float(tr["f"])
        if np.isfinite(guided_pol_fmin) and (f_tess < guided_pol_fmin):
            search_rows.append({
                "tess_mode": mode_n, "f_tess": f_tess, "detected": False,
                "f_local": np.nan, "amp_local": np.nan, "amp_local_err": np.nan, "phase_local": np.nan, "phase_local_err": np.nan,
                "snr_local": np.nan, "W_local": np.nan,
                "skip_reason": f"below_GUIDED_POL_FMIN({guided_pol_fmin:g})",
            })
            print(f"[{pol0.name}] skipping TESS mode {i_tr}/{len(template_df)} (mode={mode_n}) at f={f_tess:.6f} c/d < GUIDED_POL_FMIN={guided_pol_fmin:g}")
            continue

        print(f"[{pol0.name}] testing TESS mode {i_tr}/{len(template_df)} (mode={mode_n}) near f={f_tess:.6f} c/d")
        local_freqs = make_local_frequency_grid(
            f_tess, search_half_window, df_local, fmin=max(FMIN, guided_pol_fmin if np.isfinite(guided_pol_fmin) else FMIN), fmax=FMAX,
            max_points=POL_LOCAL_MAX_POINTS
        )
        baseline_now = make_pol_baseline_matrix(pol, trend_cfg)
        power = nuisance_periodogram(pol, local_freqs, baseline_matrix=baseline_now)
        if not np.isfinite(power).any():
            search_rows.append({
                "tess_mode": mode_n, "f_tess": f_tess, "detected": False,
                "f_local": np.nan, "amp_local": np.nan, "amp_local_err": np.nan, "phase_local": np.nan, "phase_local_err": np.nan,
                "snr_local": np.nan, "W_local": np.nan,
                "skip_reason": "no_finite_power"
            })
            print(f"[{pol0.name} vs TESS mode {mode_n}] no finite local periodogram values")
            continue
        jloc = int(np.nanargmax(power))
        f0 = float(local_freqs[jloc])
        fit = fit_frequency_with_design(
            pol, f0=f0, T=Tfull_pol, kfit=KFIT, fmin=FMIN, fmax=FMAX,
            baseline_matrix=baseline_now, n_steps=LOCAL_FIT_STEPS,
            hard_fmin=float(local_freqs[0]), hard_fmax=float(local_freqs[-1])
        )
        snr, W = local_snr_from_power(
            local_freqs, power, fit["best_f"], T=Tfull_pol, ks=POL_LOCAL_NOISE_KS,
            n_side_bins=POL_LOCAL_NOISE_SIDE_BINS
        )
        detected = bool(np.isfinite(snr) and (snr >= POL_SNR_STOP))
        search_rows.append({
            "tess_mode": mode_n,
            "f_tess": f_tess,
            "detected": detected,
            "f_local": fit["best_f"] if detected else np.nan,
            "amp_local": fit["amp"] if detected else np.nan,
            "amp_local_err": fit.get("amp_err", np.nan) if detected else np.nan,
            "phase_local": fit["phase"] if detected else np.nan,
            "phase_local_err": fit.get("phase_err", np.nan) if detected else np.nan,
            "snr_local": snr,
            "W_local": W,
            "skip_reason": "" if detected else "snr_below_threshold",
        })
        if detected:
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
            pol = prewhiten(pol, fit["full_model"])
            print(f"[{pol0.name} vs TESS mode {mode_n}] detected f={fit['best_f']:.6f} c/d | amp={fit['amp']:.4g} | SNR={snr:.2f}")
        else:
            print(f"[{pol0.name} vs TESS mode {mode_n}] no detection | best local SNR={snr:.2f}")

    search_df = pd.DataFrame(search_rows)
    accepted_raw = search_df.loc[search_df["detected"]].copy().reset_index(drop=True)
    min_sep_global = max((1.0 / Tfull_pol) * MIN_FREQ_SEP_MULT, float(tess_freq_resolution) * MIN_FREQ_SEP_MULT, 1e-8)
    accepted = dedupe_frequency_rows(accepted_raw, "f_local", score_cols=("snr_local", "amp_local", "W_local"), min_sep=min_sep_global)
    if len(accepted_raw) != len(accepted):
        print(f"[{pol0.name}] merged {len(accepted_raw)-len(accepted)} near-duplicate guided detection(s) before global fit (min_sep={min_sep_global:.6g} c/d)")
    print(f"[{pol0.name}] accepted {len(accepted)} guided detections; starting global fit")

    global_fit = fit_global_multisin(
        pol0,
        seed_freqs=accepted["f_local"].to_numpy(dtype=float) if not accepted.empty else np.array([], dtype=float),
        baseline_matrix=baseline0,
        freq_resolution=(1.0 / Tfull_pol),
        bound_mult=GLOBAL_FREQ_BOUND_MULT
    )
    final_resid = prewhiten(pol0, global_fit["full_model"])

    freqs_plot = make_plot_frequency_grid(pol0, oversample=1.0, gap_days=trend_cfg.gap_days, max_points=PLOT_MAX_GRID_POINTS)
    start_power = nuisance_periodogram(pol0, freqs_plot, baseline_matrix=baseline0)
    end_power = nuisance_periodogram(final_resid, freqs_plot, baseline_matrix=make_pol_baseline_matrix(final_resid, trend_cfg))

    freqs_plot_full = make_fullbaseline_plot_frequency_grid(
        pol0, oversample=POL_FULL_RANGE_PLOT_OVERSAMPLE, max_points=POL_FULL_RANGE_PLOT_POINTS
    )
    start_power_full = nuisance_periodogram(pol0, freqs_plot_full, baseline_matrix=baseline0)
    end_power_full = nuisance_periodogram(final_resid, freqs_plot_full, baseline_matrix=make_pol_baseline_matrix(final_resid, trend_cfg))

    final_rows = []
    if not accepted.empty and global_fit["component_rows"]:
        g_rows = pd.DataFrame(global_fit["component_rows"])
        remaining = list(range(len(g_rows)))
        for _, ar in accepted.iterrows():
            f0 = float(ar["f_local"])
            idx = min(remaining, key=lambda j: abs(float(g_rows.iloc[j]["f"]) - f0))
            grow = g_rows.iloc[idx].to_dict()
            remaining.remove(idx)
            final_rows.append({
                "tess_mode": int(ar["tess_mode"]),
                "f_tess": float(ar["f_tess"]),
                "f_local": float(ar["f_local"]),
                "amp_local": float(ar["amp_local"]),
                "amp_local_err": float(ar.get("amp_local_err", np.nan)),
                "phase_local": float(ar["phase_local"]),
                "phase_local_err": float(ar.get("phase_local_err", np.nan)),
                "snr_local": float(ar["snr_local"]),
                "W_local": float(ar["W_local"]),
                "f": float(grow["f"]),
                "amp": float(grow["amp"]),
                "amp_err": float(grow.get("amp_err", np.nan)),
                "phase": float(grow["phase"]),
                "phase_err": float(grow.get("phase_err", np.nan)),
                "s_coeff": float(grow["s_coeff"]),
                "c_coeff": float(grow["c_coeff"]),
            })
    final_df = pd.DataFrame(final_rows)
    local_diags = build_guided_local_diagnostics(pol0, final_resid, search_df, trend_cfg, tess_freq_resolution)
    return (search_df, final_df, global_fit, snapshots, final_resid,
            freqs_plot, start_power, end_power, local_diags,
            freqs_plot_full, start_power_full, end_power_full)

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

def phase_fold(t: np.ndarray, f: float, t_ref: float = 0.0) -> np.ndarray:
    return ((t - t_ref) * f) % 1.0

def _wrap_phase_radians(phi):
    phi = np.asarray(phi, dtype=float)
    return np.arctan2(np.sin(phi), np.cos(phi))

def to_btjd_abs(ts: TimeSeries) -> np.ndarray:
    """Return an absolute BTJD-like time array for phase-reference calculations."""
    if ts.t_abs is None:
        return np.asarray(ts.t, dtype=float)
    t_abs = np.asarray(ts.t_abs, dtype=float)
    finite = t_abs[np.isfinite(t_abs)]
    if finite.size == 0:
        return np.asarray(ts.t, dtype=float)
    # Heuristic: large values are JD-like and should be converted to BTJD.
    if np.nanmedian(finite) > 1.0e6:
        return t_abs - 2457000.0
    return t_abs

def _phase_reference_epoch_btjd(ts: TimeSeries) -> float:
    mode = str(PHASE_ZERO_MODE).strip().lower()
    if mode == "local_start":
        return float(to_btjd_abs(ts)[0])
    if mode == "btjd_zero":
        return 0.0
    if mode == "custom_btjd":
        return float(PHASE_ZERO_BTJD)
    raise ValueError(f"Unknown PHASE_ZERO_MODE={PHASE_ZERO_MODE!r}")

def phase_zero_summary_text() -> str:
    mode = str(PHASE_ZERO_MODE).strip().lower()
    if mode == "local_start":
        return "local_start (each dataset uses its own first timestamp)"
    if mode == "btjd_zero":
        return "btjd_zero (shared absolute BTJD = 0.0)"
    if mode == "custom_btjd":
        return f"custom_btjd (shared BTJD = {float(PHASE_ZERO_BTJD):.6f})"
    return str(PHASE_ZERO_MODE)

def _phase_shift_days(ts: TimeSeries) -> float:
    """Return the time shift to convert local-start phase to the chosen reference."""
    mode = str(PHASE_ZERO_MODE).strip().lower()
    if mode == "local_start":
        return 0.0
    btjd = to_btjd_abs(ts)
    if btjd.size == 0:
        return 0.0
    return float(btjd[0] - _phase_reference_epoch_btjd(ts))

def phase_fold_ts(ts: TimeSeries, f: float) -> np.ndarray:
    mode = str(PHASE_ZERO_MODE).strip().lower()
    if mode == "local_start":
        return phase_fold(np.asarray(ts.t, dtype=float), f, t_ref=0.0)
    return phase_fold(to_btjd_abs(ts), f, t_ref=_phase_reference_epoch_btjd(ts))

def phase_model_on_grid(ph_grid: np.ndarray, f: float, s_coeff: float, c_coeff: float, ts: TimeSeries) -> np.ndarray:
    amp, phase0 = sc_to_amp_phase(float(s_coeff), float(c_coeff))
    phase_ref = _wrap_phase_radians(phase0 - 2.0 * np.pi * float(f) * _phase_shift_days(ts))
    return amp * np.sin(2.0 * np.pi * np.asarray(ph_grid, dtype=float) + phase_ref)

def apply_phase_reference_to_columns(df: pd.DataFrame, ts: TimeSeries, phase_freq_pairs) -> pd.DataFrame:
    if df is None:
        return df
    out = df.copy()
    delta = _phase_shift_days(ts)
    if delta == 0.0:
        return out
    for phase_col, freq_col in phase_freq_pairs:
        if (phase_col not in out.columns) or (freq_col not in out.columns):
            continue
        ph = pd.to_numeric(out[phase_col], errors="coerce").to_numpy(dtype=float)
        ff = pd.to_numeric(out[freq_col], errors="coerce").to_numpy(dtype=float)
        good = np.isfinite(ph) & np.isfinite(ff)
        if not np.any(good):
            continue
        ph_new = ph.copy()
        ph_new[good] = _wrap_phase_radians(ph[good] - 2.0 * np.pi * ff[good] * delta)
        out[phase_col] = ph_new
    return out



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

    fig, ax = plt.subplots(figsize=(11, 4.8))
    y0 = tess0.y - np.nanmedian(tess0.y)
    y1 = final_resid.y - np.nanmedian(final_resid.y)
    ax.scatter(tess0.t, y0, s=8, alpha=0.55, label="tess start", color="#0072B2")
    ax.scatter(final_resid.t, y1, s=8, alpha=0.65, label="tess end (resid)", color="#56B4E9")
    ax.set_xlabel("Time [days]")
    ax.set_ylabel("TESS value (median-subtracted)")
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
    ax.plot(freqs, P0, lw=0.9, color="#0072B2", label="tess start")
    ax.plot(freqs, P1 + 1.2, lw=0.9, color="#56B4E9", linestyle="--", label="tess end (offset)")
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
    ax.semilogy(freqs, np.maximum(Pstart, 1e-12), lw=0.9, color="#0072B2", label="tess start")
    ax.semilogy(freqs, np.maximum(Pend, 1e-12), lw=0.9, color="#56B4E9", linestyle="--", label="tess end")
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

    ax = axes[0]
    t0 = tess0.y - np.nanmedian(tess0.y)
    t1 = tess_final_resid.y - np.nanmedian(tess_final_resid.y)
    ax.scatter(tess0.t, t0, s=8, alpha=0.55, label="tess start", color="#0072B2")
    ax.scatter(tess_final_resid.t, t1, s=8, alpha=0.65, label="tess end (resid)", color="#56B4E9")
    ax.set_ylabel("TESS value (median-subtracted)")
    ax.set_title("TESS time series")
    ax.legend(loc="best", fontsize=9)

    ax = axes[1]
    p0 = pol0.y - np.nanmedian(pol0.y)
    p1 = pol_final_resid.y - np.nanmedian(pol_final_resid.y)
    ax.scatter(pol0.t, p0, s=10, alpha=0.55, label=f"{pol0.name} start", color="#E69F00")
    ax.scatter(pol_final_resid.t, p1, s=10, alpha=0.65, label=f"{pol0.name} end (resid)", color="#CC79A7")
    ax.set_xlabel("Time [days]")
    ax.set_ylabel(f"{pol0.name} value (median-subtracted)")
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
    ax.plot(freqs_tess, T0, lw=0.9, color="#0072B2", label="tess start")
    ax.plot(freqs_tess, T1 + 1.2, lw=0.9, color="#56B4E9", linestyle="--", label="tess end (offset)")
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
            P0 = _norm_spec(gd["Pstart"])
            P1 = _norm_spec(gd["Pend"])
            freqs = gd["freqs"]
            ax.plot(freqs, P0, lw=1.0, color="#E69F00", label=f"{pol0.name} start")
            ax.plot(freqs, P1 + 1.2, lw=1.0, color="#CC79A7", linestyle="--", label=f"{pol0.name} end (offset)")
            ax.axvline(gd["f_tess"], lw=0.9, alpha=0.6, color="0.3", label="TESS template" if i == 1 else None)
            if np.isfinite(gd["f_detected"]):
                ax.axvline(gd["f_detected"], lw=0.9, alpha=0.8, color="#009E73", linestyle=":", label="pol detection" if i == 1 else None)
            title = f"{pol0.name} local guided search | TESS mode {gd['tess_mode']} near f={gd['f_tess']:.6f} c/d"
            if np.isfinite(gd["f_detected"]):
                title += f" | pol f={gd['f_detected']:.6f}"
            ax.set_title(title, fontsize=10)
            ax.set_ylabel(f"{pol0.name} norm. power + offset")
            if i == len(guided_diags):
                ax.set_xlabel("Frequency [c/d]")
            ax.legend(loc="best", fontsize=8)
    else:
        ax = axes[1]
        P0 = _norm_spec(Ppo_start)
        P1 = _norm_spec(Ppo_end)
        ax.plot(freqs_pol, P0, lw=0.9, color="#E69F00", label=f"{pol0.name} start")
        ax.plot(freqs_pol, P1 + 1.2, lw=0.9, color="#CC79A7", linestyle="--", label=f"{pol0.name} end (offset)")
        ax.set_xlabel("Frequency [c/d]")
        ax.set_ylabel(f"{pol0.name} norm. power + offset")
        ax.set_title(f"{pol0.name} spectra")
        ax.legend(loc="best", fontsize=9)

    fig.tight_layout()
    fig.savefig(outdir / prefixed_output_name(file_prefix, "spectra_start_end.png"), dpi=180)
    if show_plots_inline:
        plt.show()
    plt.close(fig)

    # Full-range polarimetry spectrum on a full-baseline plot grid
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(11, 7.2), sharex=False)
    ax = axes[0]
    ax.plot(freqs_tess, T0, lw=0.9, color="#0072B2", label="tess start")
    ax.plot(freqs_tess, T1 + 1.2, lw=0.9, color="#56B4E9", linestyle="--", label="tess end (offset)")
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
    ax.plot(freqs_pol_full, P0_full, lw=0.9, color="#E69F00", label=f"{pol0.name} start")
    ax.plot(freqs_pol_full, P1_full + 1.2, lw=0.9, color="#CC79A7", linestyle="--", label=f"{pol0.name} end (offset)")
    if not pol_table.empty:
        ytop = np.nanmax(P1_full + 1.2)
        for _, r in pol_table.iterrows():
            ax.axvline(float(r["f"]), lw=0.8, alpha=0.35, color="#009E73")
    ax.set_xlabel("Frequency [c/d]")
    ax.set_ylabel(f"{pol0.name} norm. power + offset")
    ax.set_title(f"{pol0.name} spectra (full-range, full-baseline grid)")
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
            ax.semilogy(freqs, np.maximum(gd["Pstart"], 1e-12), lw=1.0, color="#E69F00", label=f"{pol0.name} start")
            ax.semilogy(freqs, np.maximum(gd["Pend"], 1e-12), lw=1.0, color="#CC79A7", linestyle="--", label=f"{pol0.name} end")
            ax.axvline(gd["f_tess"], lw=0.9, alpha=0.6, color="0.3", label="TESS template")
            if np.isfinite(gd["f_detected"]):
                ax.axvline(gd["f_detected"], lw=0.9, alpha=0.8, color="#009E73", linestyle=":", label="pol detection")
            ax.set_title(f"{pol0.name} local guided spectra (log y) | mode {gd['tess_mode']}")
            ax.set_ylabel("Power")
            ax.legend(loc="best", fontsize=8)
        axes[-1].set_xlabel("Frequency [c/d]")
    else:
        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(11, 7.5), sharex=False)
        axes[0].semilogy(freqs_tess, np.maximum(Pte_start, 1e-12), lw=0.9, color="#0072B2", label="tess start")
        axes[0].semilogy(freqs_tess, np.maximum(Pte_end, 1e-12), lw=0.9, color="#56B4E9", linestyle="--", label="tess end")
        axes[0].legend(loc="best", fontsize=9)
        axes[0].set_ylabel("TESS power")
        axes[0].set_title("TESS spectra (log y)")

        axes[1].semilogy(freqs_pol, np.maximum(Ppo_start, 1e-12), lw=0.9, color="#E69F00", label=f"{pol0.name} start")
        axes[1].semilogy(freqs_pol, np.maximum(Ppo_end, 1e-12), lw=0.9, color="#CC79A7", linestyle="--", label=f"{pol0.name} end")
        axes[1].legend(loc="best", fontsize=9)
        axes[1].set_xlabel("Frequency [c/d]")
        axes[1].set_ylabel(f"{pol0.name} power")
        axes[1].set_title(f"{pol0.name} spectra (log y)")
    fig.tight_layout()
    fig.savefig(outdir / prefixed_output_name(file_prefix, "spectra_start_end_log_offset.png"), dpi=180)
    if show_plots_inline:
        plt.show()
    plt.close(fig)

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(11, 7.2), sharex=False)
    axes[0].semilogy(freqs_tess, np.maximum(Pte_start, 1e-12), lw=0.9, color="#0072B2", label="tess start")
    axes[0].semilogy(freqs_tess, np.maximum(Pte_end, 1e-12), lw=0.9, color="#56B4E9", linestyle="--", label="tess end")
    if not tess_table.empty:
        for _, r in tess_table.iterrows():
            axes[0].axvline(float(r["f"]), lw=0.8, alpha=0.25, color="0.3")
    axes[0].set_ylabel("TESS power")
    axes[0].set_title("TESS spectra (log y)")
    axes[0].legend(loc="best", fontsize=9)

    axes[1].semilogy(freqs_pol_full, np.maximum(Ppo_start_full, 1e-12), lw=0.9, color="#E69F00", label=f"{pol0.name} start")
    axes[1].semilogy(freqs_pol_full, np.maximum(Ppo_end_full, 1e-12), lw=0.9, color="#CC79A7", linestyle="--", label=f"{pol0.name} end")
    if not pol_table.empty:
        for _, r in pol_table.iterrows():
            axes[1].axvline(float(r["f"]), lw=0.8, alpha=0.35, color="#009E73")
    axes[1].set_xlabel("Frequency [c/d]")
    axes[1].set_ylabel(f"{pol0.name} power")
    axes[1].set_title(f"{pol0.name} spectra (full-range, log y)")
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


def set_robust_phase_ylim(ax, *arrays, central_pct: float = 98.0, pad_frac: float = 0.12):
    """Set a robust y-range that ignores a small fraction of extreme outliers."""
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

def plot_tess_phased_modes(tess_snapshots: list[dict], mode_table: pd.DataFrame, outdir: Path,
                           n_phase_plots: int = 3, show_plots_inline: bool = False, file_prefix: str = ""):
    if mode_table.empty or not tess_snapshots:
        return
    snap_map = {int(s["mode"]): s for s in tess_snapshots if "mode" in s}
    pick = mode_table.sort_values("snr_local", ascending=False).head(n_phase_plots).reset_index(drop=True)
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
        ph = phase_fold_ts(prefit, f)
        ax.scatter(ph, y, s=8, alpha=0.7, color="0.25")
        ax.scatter(ph + 1.0, y, s=8, alpha=0.7, color="0.25")
        ph_grid = np.linspace(0.0, 2.0, 500)
        model = phase_model_on_grid(ph_grid, f, float(fit["s_coeff"]), float(fit["c_coeff"]), prefit)
        model = model - np.nanmedian(model)
        ax.plot(ph_grid, model, lw=1.4, color="0.0")
        set_robust_phase_ylim(ax, y, model, central_pct=98.0, pad_frac=0.15)
        ax.set_xlim(0.0, 2.0)
        ax.set_xlabel("Phase")
        ax.set_ylabel("TESS (median-subtracted)")
        ax.set_title(f"Mode {mode} | TESS | f={float(r['f']):.6f} c/d | amp={float(r['amp']):.4g} | SNR={float(r['snr_local']):.2f}")
    fig.tight_layout()
    fig.savefig(outdir / prefixed_output_name(file_prefix, "phased_top_modes.png"), dpi=180)
    if show_plots_inline:
        plt.show()
    plt.close(fig)

def plot_channel_phased_modes(tess_snapshots: list[dict], tess_table: pd.DataFrame,
                              pol_search_df: pd.DataFrame, pol_snapshots: list[dict], pol0: TimeSeries,
                              outdir: Path, n_phase_plots: int = 3, sort_by: str = "amp",
                              show_plots_inline: bool = False, file_prefix: str = ""):
    detected = pol_search_df.loc[pol_search_df["detected"]].copy() if pol_search_df is not None and len(pol_search_df) else pd.DataFrame()
    if detected.empty or not pol_snapshots:
        return
    snap_map = {int(s["tess_mode"]): s for s in pol_snapshots if "tess_mode" in s}
    tess_snap_map = {int(s["mode"]): s for s in tess_snapshots if "mode" in s}
    sort_col = sort_by if sort_by in detected.columns else ("amp_local" if "amp_local" in detected.columns else "snr_local")
    pick = detected.sort_values(sort_col, ascending=False).head(n_phase_plots).reset_index(drop=True)
    fig, axes = plt.subplots(nrows=len(pick), ncols=2, figsize=(12, 3.8 * len(pick)), squeeze=False)

    for i, (_, pr) in enumerate(pick.iterrows()):
        tess_mode = int(pr["tess_mode"])

        # TESS panel: use the local single-mode snapshot rather than a global component
        ax = axes[i, 0]
        tsnap = tess_snap_map.get(tess_mode)
        tr = tess_table.loc[tess_table["mode"] == tess_mode]
        if tsnap is not None and not tr.empty:
            tr = tr.iloc[0]
            tfit = tsnap["fit"]
            tprefit = tsnap["prefit"]
            f_t = float(tfit["best_f"])
            y_t = tprefit.y - np.nanmedian(tprefit.y)
            ph_t = phase_fold_ts(tprefit, f_t)
            ax.scatter(ph_t, y_t, s=8, alpha=0.7, color="0.25")
            ax.scatter(ph_t + 1.0, y_t, s=8, alpha=0.7, color="0.25")
            ph_grid = np.linspace(0.0, 2.0, 500)
            model_t = phase_model_on_grid(ph_grid, f_t, float(tfit["s_coeff"]), float(tfit["c_coeff"]), tprefit)
            model_t = model_t - np.nanmedian(model_t)
            ax.plot(ph_grid, model_t, lw=1.4, color="0.0")
            set_robust_phase_ylim(ax, y_t, model_t, central_pct=98.0, pad_frac=0.15)
            ax.set_title(f"Mode {tess_mode} | TESS | f={float(tr['f']):.6f} c/d | amp={float(tr['amp']):.4g} | SNR={float(tr['snr_local']):.2f}")
        ax.set_xlim(0.0, 2.0)
        ax.set_xlabel("Phase")
        ax.set_ylabel("TESS (median-subtracted)")

        # Polarimetry panel: show baseline-corrected folded data plus the local single-frequency fit
        ax = axes[i, 1]
        psnap = snap_map.get(tess_mode)
        if psnap is not None:
            pfit = psnap["fit"]
            pprefit = psnap["prefit"]
            f_p = float(pfit["best_f"])
            y_p = pprefit.y - np.asarray(pfit["baseline_model"], dtype=float)
            y_p = y_p - np.nanmedian(y_p)
            ph_p = phase_fold_ts(pprefit, f_p)
            ax.scatter(ph_p, y_p, s=12, alpha=0.55, color="0.25")
            ax.scatter(ph_p + 1.0, y_p, s=12, alpha=0.55, color="0.25")
            bph, by = phase_bin_medians(ph_p, y_p, nbin=PHASE_BIN_N)
            if len(bph):
                ax.scatter(bph, by, s=28, alpha=0.95, color="#D55E00", zorder=3)
                ax.scatter(bph + 1.0, by, s=28, alpha=0.95, color="#D55E00", zorder=3, label="phase-bin median")
            ph_grid = np.linspace(0.0, 2.0, 500)
            model_p = phase_model_on_grid(ph_grid, f_p, float(pfit["s_coeff"]), float(pfit["c_coeff"]), pprefit)
            model_p = model_p - np.nanmedian(model_p)
            ax.plot(ph_grid, model_p, lw=1.4, color="0.0")
            set_robust_phase_ylim(ax, y_p, model_p, by, central_pct=98.0, pad_frac=0.15)
            ax.set_title(f"Mode {tess_mode} | {pol0.name} | f={float(pr['f_local']):.6f} c/d | amp={float(pr['amp_local']):.4g} | SNR={float(pr['snr_local']):.2f}")
        ax.set_xlim(0.0, 2.0)
        ax.set_xlabel("Phase")
        ax.set_ylabel(f"{pol0.name} (median-subtracted)")

    fig.tight_layout()
    fig.savefig(outdir / prefixed_output_name(file_prefix, "phased_top_modes.png"), dpi=180)
    if show_plots_inline:
        plt.show()
    plt.close(fig)


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

def build_guided_local_diagnostics(pol0: TimeSeries, final_resid: TimeSeries, search_df: pd.DataFrame,
                                   trend_cfg: NightTrendConfig, tess_freq_resolution: float):
    Tfull_pol = max(compute_T_full(pol0), 1e-8)
    search_half_window = POL_SEARCH_WINDOW_MULT * max(float(tess_freq_resolution), 1.0 / Tfull_pol)
    df_local = max(min(float(tess_freq_resolution), 1.0 / Tfull_pol) / max(POL_OVERSAMPLE, 1e-8), 1e-8)
    baseline0 = make_pol_baseline_matrix(pol0, trend_cfg)
    baseline1 = make_pol_baseline_matrix(final_resid, trend_cfg)
    diags = []
    if search_df is None or len(search_df) == 0:
        return diags
    guided_pol_fmin = None if GUIDED_POL_FMIN is None else float(GUIDED_POL_FMIN)
    for _, row in search_df.iterrows():
        f_tess = float(row["f_tess"])
        if np.isfinite(guided_pol_fmin) and (f_tess < guided_pol_fmin):
            continue
        if str(row.get("skip_reason", "")).startswith("below_GUIDED_POL_FMIN"):
            continue
        local_freqs = make_local_frequency_grid(
            f_tess, search_half_window, df_local, fmin=max(FMIN, guided_pol_fmin if np.isfinite(guided_pol_fmin) else FMIN), fmax=FMAX,
            max_points=POL_LOCAL_MAX_POINTS
        )
        P0 = nuisance_periodogram(pol0, local_freqs, baseline_matrix=baseline0)
        P1 = nuisance_periodogram(final_resid, local_freqs, baseline_matrix=baseline1)
        diags.append({
            "tess_mode": int(row["tess_mode"]),
            "f_tess": f_tess,
            "f_detected": np.nan if pd.isna(row.get("f_local", np.nan)) else float(row["f_local"]),
            "detected": bool(row.get("detected", False)),
            "freqs": local_freqs,
            "Pstart": P0,
            "Pend": P1,
        })
    return diags

# ----------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------



def save_summary_table(summary_df: pd.DataFrame, csv_path: Path, xlsx_path: Path | None = None):
    summary_df.to_csv(csv_path, index=False)
    if xlsx_path is not None:
        try:
            summary_df.to_excel(xlsx_path, index=False)
        except Exception as exc:
            print(f"[summary] could not write Excel table {xlsx_path}: {exc}")

def run_analysis():
    print("TESS_INPUT_MODE =", TESS_INPUT_MODE)
    print("TESS_SNR_STOP =", TESS_SNR_STOP, "| POL_SNR_STOP =", POL_SNR_STOP)
    print("GLOBAL_FREQ_BOUND_MULT =", GLOBAL_FREQ_BOUND_MULT, "| GLOBAL_MAX_NFEV =", GLOBAL_MAX_NFEV)
    print("MIN_FREQ_SEP_MULT =", MIN_FREQ_SEP_MULT, "| TESS_DISCOVERY_EXCLUSION_MULT =", TESS_DISCOVERY_EXCLUSION_MULT)
    print("POL_SEARCH_WINDOW_MULT =", POL_SEARCH_WINDOW_MULT, "| POL_LOCAL_NOISE_KS =", POL_LOCAL_NOISE_KS,
          "| POL_LOCAL_NOISE_SIDE_BINS =", POL_LOCAL_NOISE_SIDE_BINS)
    print("GUIDED_POL_FMIN =", GUIDED_POL_FMIN, "| PHASE_BIN_N =", PHASE_BIN_N)
    print("VERBOSE =", VERBOSE, "| LSQ_VERBOSE =", LSQ_VERBOSE)
    print("POL channels =", POL_CHANNELS, "| p is treated as a directly observed channel in this version")

    trend_cfg = NightTrendConfig(
        use_offsets=USE_POL_NIGHT_OFFSETS,
        use_slopes=USE_POL_NIGHT_SLOPES,
        group_mode=POL_NIGHT_GROUP_MODE,
        gap_days=POL_NIGHT_GAP_HOURS / 24.0,
    )

    t_all = time.time()
    star_label, star_safe = infer_star_labels_from_pol_path(POL_CSV)
    print("STAR_LABEL =", star_label)
    print("PHASE_ZERO =", phase_zero_summary_text())
    tess = load_tess_input()
    pol_dict = load_polarimetry_csv(POL_CSV, product=POL_PRODUCT, trend_cfg=trend_cfg)
    vprint(1, f"Loaded datasets | TESS n={len(tess.t)} | " + ", ".join([f"{k} n={len(v.t)}" for k, v in pol_dict.items()]))

    if DO_DETREND and DETREND_POLY_ORDER > 0:
        tess_dt = detrend_poly(tess, order=DETREND_POLY_ORDER)
        pol_dt = {k: detrend_poly(v, order=DETREND_POLY_ORDER) for k, v in pol_dict.items()}
    else:
        tess_dt = tess
        pol_dt = pol_dict

    # TESS-only analysis
    tess_prefix = f"{star_safe}_guided_tess"
    tess_outdir = OUTROOT / tess_prefix
    tess_outdir.mkdir(parents=True, exist_ok=True)
    vprint(1, "Starting TESS-only analysis...")
    tess_local_df, tess_global_fit, tess_snapshots, tess_final_resid, freqs_tess, Pte_start, Pte_end = extract_tess_modes(tess_dt)
    tess_table = match_global_components_with_local(tess_local_df, tess_global_fit)
    tess_table_out = apply_phase_reference_to_columns(
        tess_table, tess_dt,
        [("phase_local", "f_local"), ("phase", "f")]
    )
    tess_table_out = add_period_columns(tess_table_out, [("f_local", "period_local_d"), ("f", "period_d")])
    tess_global_rows_out = apply_phase_reference_to_columns(
        pd.DataFrame(tess_global_fit["component_rows"]), tess_dt,
        [("phase", "f")]
    )
    tess_global_rows_out = add_period_columns(tess_global_rows_out, [("f", "period_d")])
    tess_table_out.to_csv(tess_outdir / prefixed_output_name(tess_prefix, "peaks_table.csv"), index=False)
    tess_global_rows_out.to_csv(tess_outdir / prefixed_output_name(tess_prefix, "global_components_raw.csv"), index=False)
    show_table(f"{star_label} TESS peaks_table", tess_table_out)
    plot_tess_timeseries_and_spectra(tess_dt, tess_final_resid, freqs_tess, Pte_start, Pte_end, tess_table,
                                     tess_outdir, show_plots_inline=SHOW_PLOTS_INLINE, file_prefix=tess_prefix)
    plot_tess_phased_modes(tess_snapshots, tess_table, tess_outdir,
                           n_phase_plots=N_PHASE_PLOTS, show_plots_inline=SHOW_PLOTS_INLINE, file_prefix=tess_prefix)

    outputs = {"tess": tess_table_out}
    summary_channel_tables = {}

    for k in POL_CHANNELS:
        if k not in pol_dt:
            print(f"Skipping channel {k!r}: not present.")
            continue
        channel_prefix = f"{star_safe}_guided_{k}"
        outdir = OUTROOT / channel_prefix
        outdir.mkdir(parents=True, exist_ok=True)
        print(f"\n=== Guided run for {star_label} {k} ===")
        (search_df, final_df, global_fit, pol_snapshots, pol_final_resid,
         freqs_pol, Ppo_start, Ppo_end, guided_diags,
         freqs_pol_full, Ppo_start_full, Ppo_end_full) = search_guided_channel(
            pol_dt[k], tess_table, 1.0 / max(compute_T_full(tess_dt), 1e-8), trend_cfg
        )
        search_df_out = apply_phase_reference_to_columns(
            search_df, pol_dt[k],
            [("phase_local", "f_local")]
        )
        search_df_out = add_period_columns(search_df_out, [("f_tess", "period_tess_d"), ("f_local", "period_local_d")])
        search_df_out = merge_tess_phase_info(search_df_out, tess_table_out)
        final_df_out = apply_phase_reference_to_columns(
            final_df, pol_dt[k],
            [("phase_local", "f_local"), ("phase", "f")]
        )
        final_df_out = add_period_columns(final_df_out, [("f_tess", "period_tess_d"), ("f_local", "period_local_d"), ("f", "period_d")])
        final_df_out = merge_tess_phase_info(final_df_out, tess_table_out)
        global_rows_out = apply_phase_reference_to_columns(
            pd.DataFrame(global_fit["component_rows"]), pol_dt[k],
            [("phase", "f")]
        )
        global_rows_out = add_period_columns(global_rows_out, [("f", "period_d")])
        search_df_out.to_csv(outdir / prefixed_output_name(channel_prefix, "matched_search_table.csv"), index=False)
        final_df_out.to_csv(outdir / prefixed_output_name(channel_prefix, "peaks_table.csv"), index=False)
        global_rows_out.to_csv(outdir / prefixed_output_name(channel_prefix, "global_components_raw.csv"), index=False)
        show_table(f"{star_label} {k} matched_search_table", search_df_out)
        show_table(f"{star_label} {k} peaks_table", final_df_out)
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
            show_plots_inline=SHOW_PLOTS_INLINE, file_prefix=channel_prefix
        )
        outputs[k] = final_df_out
        summary_channel_tables[k] = final_df_out

    summary_df = build_guided_summary_table(tess_table_out, summary_channel_tables)
    if not summary_df.empty:
        summary_prefix = f"{star_safe}_guided_summary"
        csv_path = OUTROOT / prefixed_output_name(summary_prefix, "frequency_table.csv")
        xlsx_path = OUTROOT / prefixed_output_name(summary_prefix, "frequency_table.xlsx")
        save_summary_table(summary_df, csv_path, xlsx_path)
        plot_guided_summary_amplitude_spectra(
            summary_df, tess_dt, pol_dt, trend_cfg,
            OUTROOT / prefixed_output_name(summary_prefix, "amplitude_spectra.png"),
            show_plots_inline=SHOW_PLOTS_INLINE,
        )
        outputs["summary"] = summary_df

    vprint(1, f"All analysis complete in {time.time()-t_all:.1f}s")
    return outputs

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Run the TESS-guided photometry + polarimetry analysis.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--phase-zero-mode", choices=["local_start", "btjd_zero", "custom_btjd"],
                   default=PHASE_ZERO_MODE,
                   help="Phase-reference convention for reported phases and phased plots.")
    p.add_argument("--phase-zero-btjd", type=float, default=PHASE_ZERO_BTJD,
                   help="Custom BTJD phase zero when --phase-zero-mode=custom_btjd.")
    p.add_argument("--summary-save-period", action="store_true", default=SUMMARY_SAVE_PERIOD_VERSION,
                   help="Also save period-axis versions of the summary amplitude-spectrum plot.")
    p.add_argument("--summary-save-log-amplitude", action="store_true", default=SUMMARY_SAVE_LOG_AMPLITUDE_VERSION,
                   help="Also save log-amplitude versions of the summary amplitude-spectrum plot.")
    return p.parse_args(argv)

if __name__ == "__main__":
    args = parse_args()
    PHASE_ZERO_MODE = args.phase_zero_mode
    PHASE_ZERO_BTJD = args.phase_zero_btjd
    SUMMARY_SAVE_PERIOD_VERSION = args.summary_save_period
    SUMMARY_SAVE_LOG_AMPLITUDE_VERSION = args.summary_save_log_amplitude
    outputs = run_analysis()
    if "tess" in outputs:
        print("\nTESS peaks:")
        print(outputs["tess"].head())


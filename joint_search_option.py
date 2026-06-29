# Auto-exported companion script from notebook v6

import numpy as np
import pandas as pd
from dataclasses import dataclass
from pathlib import Path
import matplotlib.pyplot as plt

try:
    from scipy.signal import find_peaks
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False

try:
    from astropy.timeseries import LombScargle
    _HAVE_ASTROPY = True
except Exception:
    _HAVE_ASTROPY = False

print("SciPy:", _HAVE_SCIPY, "| Astropy:", _HAVE_ASTROPY)


# --- File paths / input mode ---
TESS_INPUT_MODE = "spoc_csv"              # "spoc_csv" | "pipeline_dir"
TESS_CSV = Path("combined_filtered.csv")  # used if TESS_INPUT_MODE == "spoc_csv"

# If TESS_INPUT_MODE == "pipeline_dir", point this at a directory containing
# CSV outputs from your Voronoi/raw or detrend pipelines.
TESS_PIPELINE_DIR = Path("tess_pipeline_lcs")
TESS_PIPELINE_PATTERN = "*.csv"
TESS_PIPELINE_RECURSIVE = False
TESS_PIPELINE_FLUX = "raw"                # "raw" | "detrended" | "auto"
TESS_FORCE_Y_COL = None

# --- TESS uncertainty weighting ---
# none: equal weights; formal: use normalized flux uncertainties as written;
# sector_rescaled: preserve cadence-to-cadence formal-error variations while
# matching each sector's median error to its robust first-difference scatter.
TESS_WEIGHT_MODE = "sector_rescaled"      # "none" | "formal" | "sector_rescaled"
TESS_ERROR_FLOOR_FRAC = 0.25
TESS_SECTOR_RESCALE_MIN_POINTS = 20
TESS_SECTOR_SCALE_MIN = 0.10
TESS_SECTOR_SCALE_MAX = 100.0

POL_CSV  = Path("bet Cep_IND_nm_pchip_analysis_frame.csv")  # <-- change

# --- Polarimetry product to analyze ---
POL_PRODUCT = "resid_nm_pchip"  # 'nm' | 'resid_nm_pchip' | 'pw_resid_nm_pchip'

# --- Frequency range (cycles/day) ---
FMIN = 0.01
FMAX = 50.0

# --- Two-stage search ---
K_CANDIDATES = 10
TOP_N_RAW_TESS_CANDIDATES = 3
COARSE_OVERSAMPLE = 1.0
REFINE_FACTOR = 10

# --- Iterative extraction loop ---
MAX_ITERS = 20
KFIT = 3.0
SNR_STOP = 2.0
W_PREFILTER = 1.5

# --- Local noise floor ---
KS_TESS = 15.0
KS_POL  = 10.0
TRIM_TOP_FRAC = 0.10

# --- Whitening robustness ---
N_SIDE_BINS = 30

# --- Polarimetry night-model (Option C default; D-ready) ---
USE_POL_NIGHT_OFFSETS = True
USE_POL_NIGHT_SLOPES  = False    # keep False for now; easy upgrade to Option D later
POL_NIGHT_GROUP_MODE  = "gap"    # 'gap' | 'integer_jd' | 'run' | 'subrun'
POL_NIGHT_GAP_HOURS   = 8.0      # used if mode='gap'

# --- Optional broad detrending hook (defaults off) ---
DO_DETREND = False
DETREND_POLY_ORDER = 0

# --- Phased plots ---
N_PHASE_PLOTS = 3
PHASE_SORT_BY = "score_comb"     # 'score_comb' | 'amp_tess' | 'amp_pol'
PHASE_PLOT_STYLE = "isolated_mode"  # "isolated_mode" | "prefit_residual"

# --- Joint weighting ---
JOINT_WEIGHT_MODE = "equal"      # "equal" | "scale_free" | "manual"
SCALE_FREE_WEIGHT_BASIS = "tseg" # currently "tseg" (unit-free, modestly favors longer coherent coverage)
MANUAL_W_TESS = 1.0
MANUAL_W_POL = 1.0

# --- Phase reference ---
PHASE_ZERO_MODE = "local_start"   # "local_start" | "btjd_zero" | "custom_btjd"
PHASE_ZERO_BTJD = 0.0

# --- Optional extra summary-spectrum outputs ---
SUMMARY_SAVE_PERIOD_VERSION = False
SUMMARY_SAVE_LOG_AMPLITUDE_VERSION = False


# --- Notebook display ---
SHOW_PLOTS_INLINE = True

# --- Output ---
OUTROOT = Path("joint_phot_pol_outputs_optionC")
OUTROOT.mkdir(parents=True, exist_ok=True)

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

TESS_TIME_COL = "time_btjd"
TESS_TIME_COL_CANDIDATES = ["time_btjd", "btjd", "time", "bjd", "jd"]
TESS_Y_COL_CANDIDATES = [
    "flux_detrend_rel",
    "flux_orbital_corrected_rel",
    "flux_quaternion_corrected_rel",
    "flux_decor_only_rel",
    "flux_quaternion_only_corrected_rel",
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

# Custom TESS-pipeline CSV support:
#   raw extractor outputs are typically time_btjd + flux_rel, with one branch
#   writing flux_detrended_rel instead;
#   detrender outputs include flux_rel, flux_detrend_rel, and flux_decor_only_rel.
TESS_PIPELINE_RAW_Y_COLS = [
    "flux_rel", "flux_medscaled", "flux_detrended_rel",
    "relative_flux", "normalized_flux", "flux",
]
TESS_PIPELINE_DETRENDED_Y_COLS = [
    "flux_detrend_rel", "flux_orbital_corrected_rel", "flux_quaternion_corrected_rel",
    "flux_decor_only_rel", "flux_quaternion_only_corrected_rel",
    "flux_detrended_sub", "flux_detrended_div", "flux_detrended_rel",
    "flux_medscaled", "flux_rel",
]

POL_TIME_COL = "jd"
POL_ERR_COLS = {"q": "q_err", "u": "u_err", "p": "p_err"}

POL_PRODUCTS = {
    "nm":               {"q": "q_nm",               "u": "u_nm",               "p": "p_nm"},
    "resid_nm_pchip":   {"q": "q_resid_nm_pchip",   "u": "u_resid_nm_pchip",   "p": "p_resid_nm_pchip"},
    "pw_resid_nm_pchip":{"q": "q_pw_resid_nm_pchip","u": "u_pw_resid_nm_pchip","p": "p_pw_resid_nm_pchip"},
}


RAW_POL_RUN_CODE_CANDIDATES = [
    "run", "Run", "RUN",
    "run_id", "RunID", "runid",
    "run_code", "RunCode", "RUN_CODE",
    "run_label", "RunLabel",
    "subrun", "Subrun",
    "subrun_label", "SubrunLabel",
]

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

def _factorize_group_labels(labels: np.ndarray) -> np.ndarray:
    ser = pd.Series(labels, dtype="object")
    ser = ser.fillna("__MISSING_GROUP__")
    codes, _ = pd.factorize(ser, sort=False)
    return codes.astype(int)

def get_polarimetry_group_labels(df: pd.DataFrame, mode: str) -> np.ndarray:
    if mode == "run":
        if "run_label" in df.columns:
            vals = df["run_label"].map(_normalize_run_token).to_numpy(object)
        elif "subrun_label" in df.columns:
            vals = df["subrun_label"].map(_base_run_token).to_numpy(object)
        else:
            run_col = _pick_first_existing(df.columns, RAW_POL_RUN_CODE_CANDIDATES)
            if run_col is None:
                raise ValueError(
                    "POL_NIGHT_GROUP_MODE='run' requires a run-like column in the polarimetry CSV. "
                    "Expected one of run/run_id/run_code/run_label/subrun/subrun_label."
                )
            vals = pd.Series(df[run_col]).map(_base_run_token).to_numpy(object)
        return vals
    if mode == "subrun":
        if "subrun_label" in df.columns:
            vals = df["subrun_label"].map(_normalize_run_token).to_numpy(object)
        elif "run_label" in df.columns:
            vals = df["run_label"].map(_normalize_run_token).to_numpy(object)
        else:
            run_col = _pick_first_existing(df.columns, RAW_POL_RUN_CODE_CANDIDATES)
            if run_col is None:
                raise ValueError(
                    "POL_NIGHT_GROUP_MODE='subrun' requires a subrun-like column in the polarimetry CSV. "
                    "Expected one of run/run_id/run_code/subrun/subrun_label."
                )
            vals = pd.Series(df[run_col]).map(_normalize_run_token).to_numpy(object)
        return vals
    raise ValueError(f"Unsupported label-based group mode: {mode}")

def _clean_sort_with_group(t, y, yerr=None, t_abs=None, group_labels=None):
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
    if group_labels is not None:
        gl = np.asarray(group_labels, dtype=object)[m]
    else:
        gl = None
    t, y = t[m], y[m]
    idx = np.argsort(t)
    t, y = t[idx], y[idx]
    if yerr is not None:
        yerr = yerr[idx]
    if t_abs is not None:
        t_abs = t_abs[idx]
    if gl is not None:
        gl = gl[idx]
    return t, y, yerr, t_abs, gl

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
            raise ValueError(f"POL_NIGHT_GROUP_MODE={mode!r} requires group_labels.")
        return _factorize_group_labels(np.asarray(group_labels, dtype=object))
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

def _pick_first_existing(columns, candidates):
    column_map = {str(c).strip().lower(): c for c in columns}
    for candidate in candidates:
        found = column_map.get(str(candidate).strip().lower())
        if found is not None:
            return found
    return None


def _infer_numeric_flux_column(df: pd.DataFrame, excluded=()) -> str | None:
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
    lname = str(y_col).strip().lower()
    candidates = list(TESS_ERROR_COL_CANDIDATES_BY_FLUX.get(lname, []))
    for name in TESS_GENERIC_ERROR_COL_CANDIDATES:
        if name not in candidates:
            candidates.append(name)
    return candidates


def _tess_group_labels_from_frame(df: pd.DataFrame, path: Path) -> np.ndarray:
    sector_col = _pick_first_existing(df.columns, ["sector", "tess_sector"])
    source_col = _pick_first_existing(df.columns, ["source_file"])
    if sector_col is not None:
        labels = []
        for i, value in enumerate(df[sector_col]):
            if pd.notna(value) and str(value).strip() != "":
                try:
                    fval = float(value)
                    if np.isfinite(fval) and fval.is_integer():
                        labels.append(f"sector:{int(fval)}")
                    else:
                        labels.append(f"sector:{str(value).strip()}")
                except Exception:
                    labels.append(f"sector:{str(value).strip()}")
            elif source_col is not None and pd.notna(df[source_col].iloc[i]):
                labels.append(f"file:{str(df[source_col].iloc[i]).strip()}")
            else:
                labels.append(f"file:{path.name}")
        return np.asarray(labels, dtype=object)
    if source_col is not None:
        return np.asarray([
            f"file:{str(v).strip()}" if pd.notna(v) and str(v).strip() else f"file:{path.name}"
            for v in df[source_col]
        ], dtype=object)
    return np.full(len(df), f"file:{path.name}", dtype=object)


def _clean_sort_tess_arrays(t, y, yerr=None, t_abs=None, group_labels=None):
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(t) & np.isfinite(y)
    if t_abs is not None:
        t_abs = np.asarray(t_abs, dtype=float)
        m &= np.isfinite(t_abs)
    if yerr is not None:
        yerr = np.asarray(yerr, dtype=float)
    if group_labels is not None:
        group_labels = np.asarray(group_labels, dtype=object)
    t = t[m]
    y = y[m]
    if t_abs is not None:
        t_abs = t_abs[m]
    if yerr is not None:
        yerr = yerr[m]
    if group_labels is not None:
        group_labels = group_labels[m]
    idx = np.argsort(t)
    t = t[idx]
    y = y[idx]
    if t_abs is not None:
        t_abs = t_abs[idx]
    if yerr is not None:
        yerr = yerr[idx]
    if group_labels is not None:
        group_labels = group_labels[idx]
    return t, y, yerr, t_abs, group_labels


def _robust_first_difference_sigma(t: np.ndarray, y: np.ndarray) -> float:
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(t) & np.isfinite(y)
    t = t[m]
    y = y[m]
    if y.size < 3:
        return np.nan
    idx = np.argsort(t)
    t = t[idx]
    y = y[idx]
    dt = np.diff(t)
    dy = np.diff(y)
    good = np.isfinite(dt) & (dt > 0) & np.isfinite(dy)
    positive_dt = dt[good]
    if positive_dt.size:
        med_dt = np.nanmedian(positive_dt)
        if np.isfinite(med_dt) and med_dt > 0:
            good &= dt <= 10.0 * med_dt
    dy = dy[good]
    if dy.size < 2:
        return np.nan
    center = np.nanmedian(dy)
    mad = np.nanmedian(np.abs(dy - center))
    if np.isfinite(mad) and mad > 0:
        return float((1.4826 * mad) / np.sqrt(2.0))
    std = np.nanstd(dy)
    return float(std / np.sqrt(2.0)) if np.isfinite(std) and std > 0 else np.nan


def _prepare_tess_errors(t: np.ndarray, y: np.ndarray, yerr_raw: np.ndarray | None,
                         group_id: np.ndarray | None,
                         mode: str | None = None,
                         error_floor_frac: float | None = None) -> tuple[np.ndarray | None, list[dict]]:
    mode = str(TESS_WEIGHT_MODE if mode is None else mode).strip().lower()
    if mode not in {"none", "formal", "sector_rescaled"}:
        raise ValueError(f"TESS_WEIGHT_MODE must be 'none', 'formal', or 'sector_rescaled'; got {mode!r}")
    if mode == "none":
        return None, []
    if yerr_raw is None:
        print(f"[TESS weights] mode={mode}: no matching normalized uncertainty column; using equal weights.")
        return None, []
    err = np.asarray(yerr_raw, dtype=float).copy()
    good_global = np.isfinite(err) & (err > 0)
    if np.count_nonzero(good_global) < 3:
        print(f"[TESS weights] mode={mode}: fewer than three usable uncertainty values; using equal weights.")
        return None, []
    gid = np.zeros(len(err), dtype=int) if group_id is None else np.asarray(group_id, dtype=int)
    if gid.shape != err.shape:
        raise ValueError("TESS group_id and uncertainty arrays have different lengths.")
    floor_frac = float(TESS_ERROR_FLOOR_FRAC if error_floor_frac is None else error_floor_frac)
    if not np.isfinite(floor_frac) or floor_frac < 0:
        raise ValueError("TESS_ERROR_FLOOR_FRAC must be finite and non-negative.")
    global_med = float(np.nanmedian(err[good_global]))
    diagnostics = []
    for g in np.unique(gid):
        m = gid == g
        e = err[m]
        good = np.isfinite(e) & (e > 0)
        group_med = float(np.nanmedian(e[good])) if np.any(good) else global_med
        if not np.isfinite(group_med) or group_med <= 0:
            group_med = global_med
        e = np.where(good, e, group_med)
        scatter = _robust_first_difference_sigma(np.asarray(t)[m], np.asarray(y)[m])
        scale = 1.0
        if mode == "sector_rescaled" and np.count_nonzero(m) >= int(TESS_SECTOR_RESCALE_MIN_POINTS):
            if np.isfinite(scatter) and scatter > 0 and group_med > 0:
                scale = float(np.clip(scatter / group_med, float(TESS_SECTOR_SCALE_MIN), float(TESS_SECTOR_SCALE_MAX)))
        e = e * scale
        med_scaled = float(np.nanmedian(e[np.isfinite(e) & (e > 0)]))
        floor_value = floor_frac * med_scaled if np.isfinite(med_scaled) else 0.0
        if floor_value > 0:
            e = np.maximum(e, floor_value)
        err[m] = e
        diagnostics.append({
            "group": int(g), "n": int(np.count_nonzero(m)),
            "formal_median": group_med, "robust_fd_sigma": scatter,
            "scale_factor": scale, "used_median": float(np.nanmedian(e)),
            "floor_value": floor_value,
        })
    final_good = np.isfinite(err) & (err > 0)
    if np.count_nonzero(final_good) != len(err):
        final_med = float(np.nanmedian(err[final_good])) if np.any(final_good) else global_med
        err = np.where(final_good, err, final_med)
    print(f"[TESS weights] mode={mode} | groups={len(diagnostics)} | floor_frac={floor_frac:g}")
    for row in diagnostics:
        print(
            "  group={group} N={n} formal_med={formal_median:.6g} "
            "fd_sigma={robust_fd_sigma:.6g} scale={scale_factor:.6g} "
            "used_med={used_median:.6g} floor={floor_value:.6g}".format(**row)
        )
    return err, diagnostics


def _read_tess_csv_with_candidates(path: Path, y_candidates, label_prefix: str = "tess"):
    df = pd.read_csv(path)
    time_col = _pick_first_existing(df.columns, TESS_TIME_COL_CANDIDATES)
    if time_col is None:
        raise ValueError(f"{path}: missing a recognized TESS time column. Found: {list(df.columns)}")
    if TESS_FORCE_Y_COL is not None and str(TESS_FORCE_Y_COL).strip():
        y_col = _pick_first_existing(df.columns, [TESS_FORCE_Y_COL])
        if y_col is None:
            raise ValueError(f"{path}: required TESS column {TESS_FORCE_Y_COL!r} not found. Found: {list(df.columns)}")
    else:
        # flux_selected_rel is an explicit science-stream choice written by the
        # SPOC converter, so honor it ahead of generic raw/detrended heuristics.
        y_col = _pick_first_existing(df.columns, ["flux_selected_rel"])
        if y_col is None:
            y_col = _pick_first_existing(df.columns, y_candidates)
        if y_col is None:
            y_col = _infer_numeric_flux_column(df, excluded=[time_col])
        if y_col is None:
            raise ValueError(f"{path}: missing a usable flux column. Found: {list(df.columns)}")
    err_col = _pick_first_existing(df.columns, _tess_error_candidates_for_flux(str(y_col)))
    t_abs = pd.to_numeric(df[time_col], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(df[y_col], errors="coerce").to_numpy(dtype=float)
    yerr_raw = pd.to_numeric(df[err_col], errors="coerce").to_numpy(dtype=float) if err_col is not None else None
    group_labels = _tess_group_labels_from_frame(df, Path(path))
    t_rel = t_abs - np.nanmin(t_abs)
    t_rel, y, yerr_raw, t_abs, group_labels = _clean_sort_tess_arrays(
        t_rel, y, yerr_raw, t_abs=t_abs, group_labels=group_labels
    )
    print(f"Using TESS file: {path} | time={time_col} | flux={y_col} | error={err_col or 'none'}")
    return t_rel, y, yerr_raw, t_abs, group_labels, str(y_col), (None if err_col is None else str(err_col))


def load_tess_csv(path: Path) -> TimeSeries:
    t_rel, y, yerr_raw, t_abs, labels, y_col, err_col = _read_tess_csv_with_candidates(
        path, TESS_Y_COL_CANDIDATES, "spoc_csv"
    )
    group_id = _factorize_group_labels(labels) if labels is not None else np.zeros(len(y), dtype=int)
    yerr, _ = _prepare_tess_errors(t_rel, y, yerr_raw, group_id)
    return TimeSeries(
        t=t_rel, y=y, yerr=yerr,
        name=f"tess({y_col}; err={err_col or 'none'}; weights={TESS_WEIGHT_MODE})",
        t_abs=t_abs, group_id=group_id,
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
    skipped = []
    any_error_column = False
    for path in files:
        try:
            _, y, yerr_raw, t_abs, labels, y_col, err_col = _read_tess_csv_with_candidates(path, candidates, "pipeline_dir")
        except Exception as exc:
            skipped.append((path.name, str(exc)))
            continue
        if len(t_abs) == 0:
            skipped.append((path.name, "no finite rows after cleaning"))
            continue
        t_all.append(t_abs)
        y_all.append(y)
        err_all.append(yerr_raw if yerr_raw is not None else np.full(len(y), np.nan, dtype=float))
        any_error_column = any_error_column or (yerr_raw is not None)
        label_all.append(labels if labels is not None else np.full(len(y), f"file:{path.name}", dtype=object))
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
    t_rel, y, yerr_raw, t_abs, labels = _clean_sort_tess_arrays(
        t_rel, y, yerr_raw, t_abs=t_abs, group_labels=labels
    )
    group_id = _factorize_group_labels(labels)
    yerr, _ = _prepare_tess_errors(t_rel, y, yerr_raw, group_id)

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

def load_polarimetry_csv(path: Path, product: str, trend_cfg: NightTrendConfig) -> dict:
    if product not in POL_PRODUCTS:
        raise ValueError(f"POL_PRODUCT must be one of {list(POL_PRODUCTS.keys())}. Got: {product}")
    df = pd.read_csv(path)
    required = [POL_TIME_COL]
    required += [POL_PRODUCTS[product][k] for k in ["q","u","p"]]
    required += [POL_ERR_COLS[k] for k in ["q","u","p"]]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing columns: {missing}. Found: {list(df.columns)}")

    t_abs = df[POL_TIME_COL].to_numpy(dtype=float)
    t0 = np.nanmin(t_abs)
    t_rel = t_abs - t0

    group_labels = None
    if trend_cfg.group_mode in ("run", "subrun"):
        group_labels = get_polarimetry_group_labels(df, trend_cfg.group_mode)

    out = {}
    for k in ["q", "u", "p"]:
        y = df[POL_PRODUCTS[product][k]].to_numpy(dtype=float)
        yerr = df[POL_ERR_COLS[k]].to_numpy(dtype=float)
        tt, yy, ee, ta, glab = _clean_sort_with_group(t_rel, y, yerr, t_abs=t_abs, group_labels=group_labels)
        gid = build_night_groups(ta, mode=trend_cfg.group_mode, gap_days=trend_cfg.gap_days, group_labels=glab)
        out[k] = TimeSeries(t=tt, y=yy, yerr=ee, name=f"pol_{k}", t_abs=ta, group_id=gid)
    return out

def detrend_poly(ts: TimeSeries, order: int = 1) -> TimeSeries:
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
        group_id=None if ts.group_id is None else ts.group_id.copy()
    )

def estimate_time_domain_noise(ts: TimeSeries) -> float:
    if len(ts.y) < 5:
        return np.nan
    dy = np.diff(ts.y)
    med = np.nanmedian(dy)
    mad = np.nanmedian(np.abs(dy - med))
    sigma_dy = mad / 0.6745 if mad > 0 else np.nanstd(dy)
    return sigma_dy / np.sqrt(2.0) if np.isfinite(sigma_dy) else np.nan

def estimate_scale_free_weight(ts: TimeSeries, basis: str = "tseg") -> float:
    basis = str(basis).lower()
    if basis == "tseg":
        tseg = compute_Tseg(ts, gap_days=1.0)
        if not np.isfinite(tseg) or tseg <= 0:
            return 1.0
        return np.sqrt(tseg)
    raise ValueError(f"Unknown SCALE_FREE_WEIGHT_BASIS={basis!r}")

def estimate_dataset_weight(ts: TimeSeries, mode: str = "equal", scale_free_basis: str = "tseg") -> float:
    mode = str(mode).lower()
    if mode == "equal":
        return 1.0
    if mode == "scale_free":
        return estimate_scale_free_weight(ts, basis=scale_free_basis)
    if mode == "manual":
        nm = str(getattr(ts, "name", "")).lower()
        return float(MANUAL_W_TESS) if "tess" in nm else float(MANUAL_W_POL)
    raise ValueError(f"Unknown JOINT_WEIGHT_MODE={mode!r}")

def compute_T_full(ts: TimeSeries) -> float:
    return float(ts.t[-1] - ts.t[0]) if len(ts.t) >= 2 else 0.0

def longest_contiguous_segment_duration(t: np.ndarray, gap_days: float) -> float:
    if len(t) < 2:
        return 0.0
    ts = np.sort(t)
    dt = np.diff(ts)
    breaks = np.where(dt > gap_days)[0]
    starts = np.r_[0, breaks + 1]
    ends   = np.r_[breaks + 1, len(ts)]
    seg_durs = []
    for s, e in zip(starts, ends):
        if e - s >= 2:
            seg_durs.append(ts[e-1] - ts[s])
    return float(np.max(seg_durs)) if seg_durs else float(ts[-1] - ts[0])

def compute_Tseg(ts: TimeSeries, gap_days: float = 1.0) -> float:
    return longest_contiguous_segment_duration(ts.t, gap_days=gap_days)

def make_frequency_grid(fmin: float, fmax: float, df: float) -> np.ndarray:
    n = int(np.floor((fmax - fmin) / df)) + 1
    return fmin + df * np.arange(max(n, 1))

def lomb_scargle_power(ts: TimeSeries, freqs: np.ndarray) -> np.ndarray:
    if _HAVE_ASTROPY:
        if ts.yerr is not None:
            ls = LombScargle(ts.t, ts.y, dy=ts.yerr)
        else:
            ls = LombScargle(ts.t, ts.y)
        return np.asarray(ls.power(freqs, normalization="psd"), dtype=float)
    baseline = make_tess_baseline_matrix(ts)
    return nuisance_periodogram(ts, freqs, baseline_matrix=baseline)

def spectral_window(ts: TimeSeries, freqs: np.ndarray) -> np.ndarray:
    t = np.asarray(ts.t, dtype=float)
    w = np.ones_like(t, dtype=float)
    z = np.array([np.abs(np.sum(w * np.exp(-2j * np.pi * f * t)))**2 for f in freqs], dtype=float)
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
        x = np.partition(x, k-1)[:k]
    return float(np.nanmedian(x))

def local_noise_floor(freqs: np.ndarray, power: np.ndarray, f0: float, T: float,
                      kfit: float, ks: float, trim_top_frac: float) -> float:
    if len(freqs) < 5 or not np.isfinite(T) or T <= 0:
        return np.nan
    df_grid = np.nanmedian(np.diff(freqs)) if len(freqs) > 1 else np.nan
    if not np.isfinite(df_grid) or df_grid <= 0:
        return np.nan

    delta_fit = kfit * (2.0 / T)
    delta_side = ks * (1.0 / T)

    min_width = max(delta_side, N_SIDE_BINS * df_grid)
    lo1, hi1 = f0 - (delta_fit + min_width), f0 - delta_fit
    lo2, hi2 = f0 + delta_fit, f0 + (delta_fit + min_width)

    m = ((freqs >= lo1) & (freqs <= hi1)) | ((freqs >= lo2) & (freqs <= hi2))
    vals = power[m]
    noise = trimmed_median(vals, trim_top_frac=trim_top_frac)

    if not np.isfinite(noise) or noise <= 0:
        finite = power[np.isfinite(power)]
        noise = trimmed_median(finite, trim_top_frac=trim_top_frac)
    return noise

def whitened_power(freqs: np.ndarray, power: np.ndarray, T: float,
                   kfit: float, ks: float, trim_top_frac: float) -> np.ndarray:
    W = np.full_like(power, np.nan, dtype=float)
    for i, f0 in enumerate(freqs):
        n = local_noise_floor(freqs, power, f0, T=T, kfit=kfit, ks=ks, trim_top_frac=trim_top_frac)
        if np.isfinite(n) and n > 0:
            W[i] = power[i] / n
    return W

def combine_whitened(W1: np.ndarray, W2: np.ndarray, w1: float, w2: float) -> np.ndarray:
    eps = 1e-30
    W1c = np.nan_to_num(W1, nan=eps, posinf=eps, neginf=eps)
    W2c = np.nan_to_num(W2, nan=eps, posinf=eps, neginf=eps)
    a = np.log(np.maximum(W1c, eps))
    b = np.log(np.maximum(W2c, eps))
    denom = (w1 + w2) if (w1 + w2) > 0 else 1.0
    return np.exp((w1 * a + w2 * b) / denom)

def simple_find_peaks(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(y)
    peaks = []
    for i in range(1, len(y) - 1):
        if ok[i-1] and ok[i] and ok[i+1] and (y[i] > y[i-1]) and (y[i] > y[i+1]):
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
    if _HAVE_SCIPY:
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
    return dfp.sort_values("prominence", ascending=False).head(k).reset_index(drop=True)

def merge_peak_tables_with_min_sep(dfs: list[pd.DataFrame], min_sep: float) -> pd.DataFrame:
    kept = []
    for df in dfs:
        if df is None or len(df) == 0:
            continue
        for _, r in df.iterrows():
            f = float(r["f"])
            if any(abs(f - float(k["f"])) < min_sep for k in kept):
                continue
            kept.append({k: r[k] for k in r.index})
    if not kept:
        return pd.DataFrame(columns=["idx", "f", "height", "prominence", "source"])
    out = pd.DataFrame(kept)
    if "source" not in out.columns:
        out["source"] = "candidate"
    return out.reset_index(drop=True)

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

def make_tess_baseline_matrix(ts: TimeSeries) -> np.ndarray:
    return np.ones((len(ts.t), 1), dtype=float)

def design_matrix_with_sinusoid(t: np.ndarray, f: float, baseline: np.ndarray) -> np.ndarray:
    ang = 2.0 * np.pi * f * t
    s = np.sin(ang)
    c = np.cos(ang)
    return np.column_stack([s, c, baseline])

def sinusoid_coeffs_to_amp_phase(beta: np.ndarray) -> tuple[float, float]:
    s_coeff = float(beta[0])
    c_coeff = float(beta[1])
    amp = float(np.hypot(s_coeff, c_coeff))
    phase = float(np.arctan2(c_coeff, s_coeff))
    return amp, phase

def signal_from_beta(t: np.ndarray, f: float, beta: np.ndarray) -> np.ndarray:
    ang = 2.0 * np.pi * f * t
    return beta[0] * np.sin(ang) + beta[1] * np.cos(ang)

def baseline_from_beta(baseline_matrix: np.ndarray, beta: np.ndarray) -> np.ndarray:
    return baseline_matrix @ beta[2:]

def signal_weights(ts: TimeSeries) -> np.ndarray:
    if ts.yerr is None:
        return np.ones_like(ts.y, dtype=float)
    w = 1.0 / np.square(ts.yerr)
    w = np.where(np.isfinite(w) & (w > 0), w, 0.0)
    positive = w[w > 0]
    if positive.size:
        med = np.nanmedian(positive)
        if np.isfinite(med) and med > 0:
            w = w / med
    return w

def nuisance_periodogram(ts: TimeSeries, freqs: np.ndarray, baseline_matrix: np.ndarray) -> np.ndarray:
    w = signal_weights(ts)
    null_fit = weighted_linear_solve(ts.y, baseline_matrix, w)
    rss0 = max(null_fit["rss"], 1e-30)
    power = np.full_like(freqs, np.nan, dtype=float)
    for i, f in enumerate(freqs):
        X = design_matrix_with_sinusoid(ts.t, f, baseline_matrix)
        alt_fit = weighted_linear_solve(ts.y, X, w)
        power[i] = max(0.0, (rss0 - alt_fit["rss"]) / rss0)
    return power

def refine_peak(f0: float, df_ref: float, refine_factor: int, fmin: float, fmax: float, compute_combined_at_freqs):
    span = 3.0 * df_ref
    f_lo = max(fmin, f0 - span)
    f_hi = min(fmax, f0 + span)
    df_fine = df_ref / float(refine_factor)
    fine = make_frequency_grid(f_lo, f_hi, df_fine)
    Wcomb_f, W1_f, W2_f = compute_combined_at_freqs(fine)
    j = int(np.nanargmax(Wcomb_f))
    return float(fine[j]), {"fine_freqs": fine, "Wcomb": Wcomb_f, "W1": W1_f, "W2": W2_f}

def fit_frequency_with_design(ts: TimeSeries, f0: float, T: float, kfit: float, fmin: float, fmax: float,
                              baseline_matrix: np.ndarray, n_steps: int = 2001) -> dict:
    w = signal_weights(ts)
    df_fit = kfit * (2.0 / T)
    f_lo = max(fmin, f0 - df_fit)
    f_hi = min(fmax, f0 + df_fit)
    freqs = np.linspace(f_lo, f_hi, n_steps)
    rss = np.full_like(freqs, np.nan, dtype=float)
    fits = [None] * len(freqs)
    for i, f in enumerate(freqs):
        X = design_matrix_with_sinusoid(ts.t, f, baseline_matrix)
        fit = weighted_linear_solve(ts.y, X, w)
        fits[i] = fit
        rss[i] = fit["rss"]
    j = int(np.nanargmin(rss))
    f_best = float(freqs[j])
    fit_best = fits[j]
    beta = np.asarray(fit_best["beta"], dtype=float)
    amp, phase = sinusoid_coeffs_to_amp_phase(beta)
    signal_model = signal_from_beta(ts.t, f_best, beta)
    baseline_model = baseline_from_beta(baseline_matrix, beta)
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
        "df_fit": float(df_fit)
    }

def local_snr_from_power(freqs: np.ndarray, power: np.ndarray, f_fit: float, T: float, ks: float) -> tuple[float, float]:
    noise = local_noise_floor(freqs, power, f_fit, T=T, kfit=KFIT, ks=ks, trim_top_frac=TRIM_TOP_FRAC)
    j = int(np.argmin(np.abs(freqs - f_fit)))
    p0 = float(power[j])
    W = float(p0 / noise) if (np.isfinite(noise) and noise > 0) else np.nan
    snr = float(np.sqrt(W)) if np.isfinite(W) and W > 0 else np.nan
    return snr, W

def prewhiten(ts: TimeSeries, model: np.ndarray) -> TimeSeries:
    return TimeSeries(
        t=ts.t.copy(),
        y=ts.y - np.asarray(model, dtype=float),
        yerr=None if ts.yerr is None else ts.yerr.copy(),
        name=ts.name,
        t_abs=None if ts.t_abs is None else ts.t_abs.copy(),
        group_id=None if ts.group_id is None else ts.group_id.copy()
    )


def to_btjd_abs(ts: TimeSeries) -> np.ndarray:
    """Return an absolute BTJD-like time array for phase-reference calculations."""
    if ts.t_abs is None:
        return np.asarray(ts.t, dtype=float)
    t_abs = np.asarray(ts.t_abs, dtype=float)
    finite = t_abs[np.isfinite(t_abs)]
    if finite.size == 0:
        return np.asarray(ts.t, dtype=float)
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

def phase_fold(t: np.ndarray, f: float, t_ref: float = 0.0) -> np.ndarray:
    phase = ((t - t_ref) * f) % 1.0
    return phase

def sort_modes_for_phasing(res: pd.DataFrame, how: str, nmax: int) -> pd.DataFrame:
    if res.empty:
        return res
    if how not in res.columns:
        how = "score_comb"
    return res.sort_values(how, ascending=False).head(nmax).reset_index(drop=True)


def plot_phased_modes(mode_snapshots: list[dict], res: pd.DataFrame, outdir: Path, title_prefix: str,
                      final_tess_resid: TimeSeries | None = None,
                      final_pol_resid: TimeSeries | None = None,
                      n_phase_plots: int = 3, sort_by: str = "score_comb",
                      plot_style: str = "isolated_mode",
                      show_plots_inline: bool = False):
    if res.empty or not mode_snapshots:
        return
    pick = sort_modes_for_phasing(res, sort_by, n_phase_plots)
    by_n = {row["n"]: row for _, row in pick.iterrows()}
    chosen = [snap for snap in mode_snapshots if snap["n"] in set(by_n.keys())]
    if not chosen:
        return

    plot_style = str(plot_style).lower()
    chosen = sorted(chosen, key=lambda d: float(by_n[d["n"]][sort_by] if sort_by in by_n[d["n"]] else by_n[d["n"]]["score_comb"]), reverse=True)
    nrows = len(chosen)
    fig, axes = plt.subplots(nrows=nrows, ncols=2, figsize=(12, 3.8 * nrows), squeeze=False)

    point_color = "0.25"
    model_color = "0.0"

    for i, snap in enumerate(chosen):
        ph_grid = np.linspace(0.0, 2.0, 500)

        # TESS panel
        ax = axes[i, 0]
        ft = float(snap["fit_tess"]["best_f"])
        t_ref_t = _phase_reference_epoch_btjd(snap["tess_prefit"])
        phase_t = phase_fold(to_btjd_abs(snap["tess_prefit"]), ft, t_ref=t_ref_t)
        if plot_style == "isolated_mode" and final_tess_resid is not None and len(final_tess_resid.y) == len(snap["fit_tess"]["signal_model"]):
            y_t = np.asarray(final_tess_resid.y, dtype=float) + np.asarray(snap["fit_tess"]["signal_model"], dtype=float)
            ylab_t = "TESS (isolated mode; median-subtracted)"
        else:
            y_t = np.asarray(snap["tess_prefit"].y, dtype=float)
            ylab_t = "TESS (prefit residual; median-subtracted)"
        y_t = y_t - np.nanmedian(y_t)
        ax.scatter(phase_t, y_t, s=8, alpha=0.7, color=point_color)
        ax.scatter(phase_t + 1.0, y_t, s=8, alpha=0.7, color=point_color)
        y_model = (
            snap["fit_tess"]["beta"][0] * np.sin(2.0 * np.pi * ph_grid) +
            snap["fit_tess"]["beta"][1] * np.cos(2.0 * np.pi * ph_grid)
        )
        y_model = y_model - np.nanmedian(y_model)
        ax.plot(ph_grid, y_model, lw=1.4, color=model_color)
        ax.set_xlim(0.0, 2.0)
        ax.set_xlabel("Phase")
        ax.set_ylabel(ylab_t)
        ax.set_title(f"Mode {snap['n']} | TESS | f={ft:.6f} c/d | amp={snap['fit_tess']['amp']:.4g}")

        # Polarimetry panel
        ax = axes[i, 1]
        fp = float(snap["fit_pol"]["best_f"])
        t_ref_p = _phase_reference_epoch_btjd(snap["pol_prefit"])
        phase_p = phase_fold(to_btjd_abs(snap["pol_prefit"]), fp, t_ref=t_ref_p)
        if plot_style == "isolated_mode" and final_pol_resid is not None and len(final_pol_resid.y) == len(snap["fit_pol"]["signal_model"]):
            y_p = np.asarray(final_pol_resid.y, dtype=float) + np.asarray(snap["fit_pol"]["signal_model"], dtype=float)
            ylab_p = f"{snap['pol_prefit'].name} (isolated mode; median-subtracted)"
        else:
            y_p = np.asarray(snap["pol_prefit"].y - snap["fit_pol"]["baseline_model"], dtype=float)
            ylab_p = f"{snap['pol_prefit'].name} (prefit residual; median-subtracted)"
        y_p = y_p - np.nanmedian(y_p)
        ax.scatter(phase_p, y_p, s=10, alpha=0.7, color=point_color)
        ax.scatter(phase_p + 1.0, y_p, s=10, alpha=0.7, color=point_color)
        y_model_p = (
            snap["fit_pol"]["beta"][0] * np.sin(2.0 * np.pi * ph_grid) +
            snap["fit_pol"]["beta"][1] * np.cos(2.0 * np.pi * ph_grid)
        )
        y_model_p = y_model_p - np.nanmedian(y_model_p)
        ax.plot(ph_grid, y_model_p, lw=1.4, color=model_color)
        ax.set_xlim(0.0, 2.0)
        ax.set_xlabel("Phase")
        ax.set_ylabel(ylab_p)
        ax.set_title(f"Mode {snap['n']} | {snap['pol_prefit'].name} | f={fp:.6f} c/d | amp={snap['fit_pol']['amp']:.4g}")

    style_note = "isolated-mode baseline" if plot_style == "isolated_mode" else "prefit-residual baseline"
    fig.suptitle(f"{title_prefix}: phased top modes ({style_note})", y=1.01, fontsize=13)
    fig.tight_layout()
    fig.savefig(outdir / "phased_top_modes.png", dpi=180, bbox_inches="tight")
    if show_plots_inline:
        plt.show()
    plt.close(fig)


def run_joint_extraction_one(tess0: TimeSeries, pol0: TimeSeries, outdir: Path,
                             pol_trend_cfg: NightTrendConfig, show_plots_inline: bool = False) -> pd.DataFrame:
    outdir.mkdir(parents=True, exist_ok=True)

    tess = TimeSeries(
        t=tess0.t.copy(), y=tess0.y.copy(),
        yerr=None if tess0.yerr is None else tess0.yerr.copy(),
        name=tess0.name, t_abs=None if tess0.t_abs is None else tess0.t_abs.copy(),
        group_id=None if tess0.group_id is None else tess0.group_id.copy()
    )
    pol = TimeSeries(
        t=pol0.t.copy(), y=pol0.y.copy(),
        yerr=None if pol0.yerr is None else pol0.yerr.copy(),
        name=pol0.name, t_abs=None if pol0.t_abs is None else pol0.t_abs.copy(),
        group_id=None if pol0.group_id is None else pol0.group_id.copy()
    )

    w_te = estimate_dataset_weight(tess, mode=JOINT_WEIGHT_MODE, scale_free_basis=SCALE_FREE_WEIGHT_BASIS)
    w_po = estimate_dataset_weight(pol, mode=JOINT_WEIGHT_MODE, scale_free_basis=SCALE_FREE_WEIGHT_BASIS)

    Tseg_tess = compute_Tseg(tess, gap_days=1.0)
    Tseg_pol  = compute_Tseg(pol,  gap_days=1.0)
    Tseg_max = max(Tseg_tess, Tseg_pol)
    df_coarse = (1.0 / Tseg_max) / max(COARSE_OVERSAMPLE, 1e-6)
    freqs_coarse = make_frequency_grid(FMIN, FMAX, df_coarse)
    df_grid_run = float(np.nanmedian(np.diff(freqs_coarse))) if len(freqs_coarse) > 1 else np.nan
    print(f"[{pol.name}] Tseg_tess={Tseg_tess:.3f} d | Tseg_pol={Tseg_pol:.3f} d | df_grid={df_grid_run:.6g} c/d")

    win_te = spectral_window(tess, freqs_coarse)
    win_po = spectral_window(pol, freqs_coarse)
    Pte_start = lomb_scargle_power(tess, freqs_coarse)
    Bpol0 = make_pol_baseline_matrix(pol, pol_trend_cfg)
    Ppo_start = nuisance_periodogram(pol, freqs_coarse, baseline_matrix=Bpol0)

    accepted = []
    rows = []
    mode_snapshots = []

    def compute_combined_at_freqs(freqs_local: np.ndarray):
        P1 = lomb_scargle_power(tess, freqs_local)
        B2 = make_pol_baseline_matrix(pol, pol_trend_cfg)
        P2 = nuisance_periodogram(pol, freqs_local, baseline_matrix=B2)
        T1 = compute_T_full(tess)
        T2 = compute_T_full(pol)
        W1 = whitened_power(freqs_local, P1, T=T1, kfit=KFIT, ks=KS_TESS, trim_top_frac=TRIM_TOP_FRAC)
        W2 = whitened_power(freqs_local, P2, T=T2, kfit=KFIT, ks=KS_POL, trim_top_frac=TRIM_TOP_FRAC)
        Wc = combine_whitened(W1, W2, w_te, w_po)
        return Wc, W1, W2, P1, P2

    for n in range(1, MAX_ITERS + 1):
        Wc, W1, W2, P1, P2 = compute_combined_at_freqs(freqs_coarse)

        min_sep = 1.0 / Tseg_max
        peaks_joint_df = pick_top_peaks(freqs_coarse, Wc, k=K_CANDIDATES, min_sep=min_sep).copy()
        if len(peaks_joint_df) > 0:
            peaks_joint_df["source"] = "joint_whitened"
        peaks_tess_raw_df = pick_top_peaks(freqs_coarse, P1, k=TOP_N_RAW_TESS_CANDIDATES, min_sep=min_sep).copy()
        if len(peaks_tess_raw_df) > 0:
            peaks_tess_raw_df["source"] = "tess_raw"
        peaks_df = merge_peak_tables_with_min_sep([peaks_joint_df, peaks_tess_raw_df], min_sep=min_sep)
        if len(peaks_df) == 0:
            print(f"[{pol.name} iter {n}] No peaks found; stopping.")
            break

        best = None
        df_ref = 1.0 / Tseg_max

        for _, r in peaks_df.iterrows():
            f0 = float(r["f"])
            def _ccb(fine):
                Wc_f, W1_f, W2_f, _, _ = compute_combined_at_freqs(fine)
                return Wc_f, W1_f, W2_f
            f_ref, details = refine_peak(f0=f0, df_ref=df_ref, refine_factor=REFINE_FACTOR,
                                         fmin=FMIN, fmax=FMAX, compute_combined_at_freqs=_ccb)
            j = int(np.nanargmax(details["Wcomb"]))
            W1p = float(details["W1"][j])
            W2p = float(details["W2"][j])
            if (W1p > W_PREFILTER) and (W2p > W_PREFILTER):
                score = float(details["Wcomb"][j])
                if (best is None) or (score > best["score"]):
                    best = {"f_ref": f_ref, "score": score, "W1": W1p, "W2": W2p}

        if best is None:
            print(f"[{pol.name} iter {n}] Candidates failed prefilter; stopping.")
            break

        f_n = best["f_ref"]
        T1 = compute_T_full(tess)
        T2 = compute_T_full(pol)
        B_te = make_tess_baseline_matrix(tess)
        B_po = make_pol_baseline_matrix(pol, pol_trend_cfg)

        fit_te = fit_frequency_with_design(tess, f0=f_n, T=T1, kfit=KFIT, fmin=FMIN, fmax=FMAX,
                                           baseline_matrix=B_te, n_steps=2001)
        fit_po = fit_frequency_with_design(pol, f0=f_n, T=T2, kfit=KFIT, fmin=FMIN, fmax=FMAX,
                                           baseline_matrix=B_po, n_steps=2001)

        P1_loc = lomb_scargle_power(tess, freqs_coarse)
        P2_loc = nuisance_periodogram(pol, freqs_coarse, baseline_matrix=B_po)
        snr_te, W_te = local_snr_from_power(freqs_coarse, P1_loc, fit_te["best_f"], T=T1, ks=KS_TESS)
        snr_po, W_po = local_snr_from_power(freqs_coarse, P2_loc, fit_po["best_f"], T=T2, ks=KS_POL)

        if (np.isfinite(snr_te) and np.isfinite(snr_po)) and (snr_te < SNR_STOP) and (snr_po < SNR_STOP):
            print(f"[{pol.name} iter {n}] Both SNR < {SNR_STOP:.2f}; stopping.")
            break

        rows.append({
            "n": n,
            "f_comb": f_n,
            "score_comb": best["score"],
            "W_tess_at_pick": best["W1"],
            "W_pol_at_pick": best["W2"],
            "f_tess": fit_te["best_f"],
            "amp_tess": fit_te["amp"],
            "phase_tess": fit_te["phase"],
            "snr_tess": snr_te,
            "W_tess": W_te,
            "f_pol": fit_po["best_f"],
            "amp_pol": fit_po["amp"],
            "phase_pol": fit_po["phase"],
            "snr_pol": snr_po,
            "W_pol": W_po,
            "n_pol_groups": int(np.unique(pol.group_id).size) if pol.group_id is not None else 0,
            "pol_use_offsets": bool(pol_trend_cfg.use_offsets),
            "pol_use_slopes": bool(pol_trend_cfg.use_slopes),
        })
        accepted.append(f_n)
        mode_snapshots.append({
            "n": n,
            "tess_prefit": TimeSeries(
                t=tess.t.copy(), y=tess.y.copy(),
                yerr=None if tess.yerr is None else tess.yerr.copy(),
                name=tess.name,
                t_abs=None if tess.t_abs is None else tess.t_abs.copy(),
                group_id=None if tess.group_id is None else tess.group_id.copy()
            ),
            "pol_prefit": TimeSeries(
                t=pol.t.copy(), y=pol.y.copy(),
                yerr=None if pol.yerr is None else pol.yerr.copy(),
                name=pol.name,
                t_abs=None if pol.t_abs is None else pol.t_abs.copy(),
                group_id=None if pol.group_id is None else pol.group_id.copy()
            ),
            "fit_tess": fit_te,
            "fit_pol": fit_po,
        })

        tess = prewhiten(tess, model=fit_te["full_model"])
        pol  = prewhiten(pol,  model=fit_po["full_model"])

        print(f"[{pol.name} iter {n}] f~{f_n:.6f} c/d | SNR tess={snr_te:.2f} pol={snr_po:.2f}")

    res = pd.DataFrame(rows)
    res.to_csv(outdir / "peaks_table.csv", index=False)

    Pte_end = lomb_scargle_power(tess, freqs_coarse)
    Ppo_end = nuisance_periodogram(pol, freqs_coarse, baseline_matrix=make_pol_baseline_matrix(pol, pol_trend_cfg))

    # time-series plot (stacked panels: TESS and polarimetry separated; start/end overplotted)
    C_TESS_START = "#0072B2"
    C_TESS_END   = "#56B4E9"
    C_POL_START  = "#E69F00"
    C_POL_END    = "#CC79A7"

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(11, 7.5), sharex=True)

    ax = axes[0]
    tess0_plot = tess0.y - np.nanmedian(tess0.y)
    tess_plot  = tess.y  - np.nanmedian(tess.y)
    ax.scatter(tess0.t, tess0_plot, s=8, alpha=0.55, label="tess start", color=C_TESS_START)
    ax.scatter(tess.t, tess_plot, s=8, alpha=0.65, label="tess end (resid)", color=C_TESS_END)
    ax.set_ylabel("TESS value (median-subtracted)")
    ax.set_title("TESS time series")
    ax.legend(loc="best", fontsize=9)

    ax = axes[1]
    ax.scatter(pol0.t, pol0.y, s=10, alpha=0.55, label=f"{pol0.name} start", color=C_POL_START)
    ax.scatter(pol.t, pol.y, s=10, alpha=0.65, label=f"{pol0.name} end (resid)", color=C_POL_END)
    ax.set_xlabel("Time [days]")
    ax.set_ylabel(f"{pol0.name} value")
    ax.set_title(f"{pol0.name} time series")
    ax.legend(loc="best", fontsize=9)

    fig.suptitle(f"Time series: TESS + {pol0.name} (Option C night offsets)", y=0.98)
    fig.tight_layout()
    fig.savefig(outdir / "timeseries_start_end.png", dpi=180)
    if show_plots_inline:
        plt.show()
    plt.close(fig)

    # spectra plot
    def _norm_spec(P: np.ndarray, q: float = 95.0) -> np.ndarray:
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

    Pte_s = _norm_spec(Pte_start, q=99.0)
    Pte_e = _norm_spec(Pte_end,   q=99.0)
    Ppo_s = _norm_spec(Ppo_start, q=99.0)
    Ppo_e = _norm_spec(Ppo_end,   q=99.0)

    off_te_s = 0.0
    off_te_e = 1.2
    off_po_s = 2.8
    off_po_e = 4.0

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(11, 7.5), sharex=True)

    ax = axes[0]
    ax.plot(freqs_coarse, Pte_s + off_te_s, lw=0.9, label="tess start", color=C_TESS_START, linestyle="-")
    ax.plot(freqs_coarse, Pte_e + off_te_e, lw=0.9, label="tess end (offset)", color=C_TESS_END, linestyle="--")
    if len(accepted) > 0:
        y_top_t = np.nanmax(Pte_e + off_te_e) if np.isfinite(np.nanmax(Pte_e + off_te_e)) else (off_te_e + 1.0)
        for i, f in enumerate(accepted, start=1):
            ax.axvline(f, lw=0.8, alpha=0.35, color="0.3")
            ax.text(f, y_top_t, str(i), rotation=90, va="top", ha="center", fontsize=8)
    ax.set_ylabel("TESS norm. power + offset")
    ax.set_title("TESS power spectra")
    ax.legend(loc="best", fontsize=9)

    ax = axes[1]
    ax.plot(freqs_coarse, Ppo_s + off_po_s, lw=0.9, label=f"{pol0.name} start", color=C_POL_START, linestyle="-")
    ax.plot(freqs_coarse, Ppo_e + off_po_e, lw=0.9, label=f"{pol0.name} end (offset)", color=C_POL_END, linestyle="--")
    if len(accepted) > 0:
        y_top_p = np.nanmax(Ppo_e + off_po_e) if np.isfinite(np.nanmax(Ppo_e + off_po_e)) else (off_po_e + 1.0)
        for i, f in enumerate(accepted, start=1):
            ax.axvline(f, lw=0.8, alpha=0.35, color="0.3")
            ax.text(f, y_top_p, str(i), rotation=90, va="top", ha="center", fontsize=8)
    ax.set_xlabel("Frequency [cycles/day]")
    ax.set_ylabel(f"{pol0.name} norm. power + offset")
    ax.set_title(f"{pol0.name} power spectra")
    ax.legend(loc="best", fontsize=9)

    if len(accepted) > 0:
        fmax_plot = max(accepted)
        pad = 0.05 * fmax_plot
        axes[1].set_xlim(left=FMIN, right=fmax_plot + pad)

    fig.suptitle(f"Power spectra (normalized): TESS + {pol0.name} (night-aware polarimetry)", y=0.98)
    fig.tight_layout()
    fig.savefig(outdir / "spectra_start_end.png", dpi=180)
    if show_plots_inline:
        plt.show()
    plt.close(fig)

    # windows
    fig = plt.figure(figsize=(11, 4))
    plt.plot(freqs_coarse, win_te, lw=0.9, label="tess window (norm)")
    plt.plot(freqs_coarse, win_po, lw=0.9, label=f"{pol0.name} window (norm)")
    plt.xlabel("Frequency [cycles/day]")
    plt.ylabel("Window power (norm)")
    plt.title(f"Spectral windows: TESS + {pol0.name}")
    plt.legend(loc="best", fontsize=9)
    if len(accepted) > 0:
        fmax_plot = max(accepted)
        pad = 0.05 * fmax_plot
        plt.xlim(left=FMIN, right=fmax_plot + pad)
    fig.tight_layout()
    fig.savefig(outdir / "spectral_windows.png", dpi=180)
    if show_plots_inline:
        plt.show()
    plt.close(fig)

    # log spectra
    fig = plt.figure(figsize=(11, 6))
    eps = 1e-12
    OFF_TESS = 1e-3
    OFF_POL  = 1e-3
    plt.plot(freqs_coarse, Pte_start + eps, lw=0.9, label="tess start", color=C_TESS_START, linestyle="-")
    plt.plot(freqs_coarse, (Pte_end * OFF_TESS) + eps, lw=0.9, label=f"tess end (×{OFF_TESS:g})", color=C_TESS_END, linestyle="--")
    plt.plot(freqs_coarse, Ppo_start + eps, lw=0.9, label=f"{pol0.name} start", color=C_POL_START, linestyle="-")
    plt.plot(freqs_coarse, (Ppo_end * OFF_POL) + eps, lw=0.9, label=f"{pol0.name} end (×{OFF_POL:g})", color=C_POL_END, linestyle="--")
    if len(accepted) > 0:
        for i, f in enumerate(accepted, start=1):
            plt.axvline(f, lw=0.8, alpha=0.35, color="0.3")
    plt.yscale("log")
    plt.xlabel("Frequency [cycles/day]")
    plt.ylabel("Power (log scale)")
    plt.title(f"Power spectra (log scale): TESS + {pol0.name}")
    plt.legend(loc="best", fontsize=9)
    if len(accepted) > 0:
        fmax_plot = max(accepted)
        pad = 0.05 * fmax_plot
        plt.xlim(left=FMIN, right=fmax_plot + pad)
    fig.tight_layout()
    fig.savefig(outdir / "spectra_start_end_log_offset.png", dpi=180)
    if show_plots_inline:
        plt.show()
    plt.close(fig)

    plot_phased_modes(mode_snapshots, res, outdir=outdir,
                      title_prefix=f"TESS + {pol0.name}",
                      final_tess_resid=tess, final_pol_resid=pol,
                      n_phase_plots=N_PHASE_PLOTS, sort_by=PHASE_SORT_BY,
                      plot_style=PHASE_PLOT_STYLE,
                      show_plots_inline=show_plots_inline)
    return res

POL_TREND_CFG = NightTrendConfig(
    use_offsets=USE_POL_NIGHT_OFFSETS,
    use_slopes=USE_POL_NIGHT_SLOPES,
    group_mode=POL_NIGHT_GROUP_MODE,
    gap_days=POL_NIGHT_GAP_HOURS / 24.0
)

print("TESS_INPUT_MODE =", TESS_INPUT_MODE)
print("TESS_WEIGHT_MODE =", TESS_WEIGHT_MODE, "| TESS_ERROR_FLOOR_FRAC =", TESS_ERROR_FLOOR_FRAC)
if str(TESS_INPUT_MODE).lower() == "pipeline_dir":
    print("TESS_PIPELINE_DIR =", TESS_PIPELINE_DIR, "| flux =", TESS_PIPELINE_FLUX, "| pattern =", TESS_PIPELINE_PATTERN)
else:
    print("TESS_CSV =", TESS_CSV)

tess = load_tess_input()
pol_dict = load_polarimetry_csv(POL_CSV, product=POL_PRODUCT, trend_cfg=POL_TREND_CFG)

if DO_DETREND:
    tess_dt = detrend_poly(tess, order=DETREND_POLY_ORDER)
    pol_dt = {k: detrend_poly(v, order=DETREND_POLY_ORDER) for k, v in pol_dict.items()}
else:
    tess_dt = tess
    pol_dt = pol_dict

print("JOINT_WEIGHT_MODE =", JOINT_WEIGHT_MODE, "| SCALE_FREE_WEIGHT_BASIS =", SCALE_FREE_WEIGHT_BASIS)
print("TOP_N_RAW_TESS_CANDIDATES =", TOP_N_RAW_TESS_CANDIDATES)
print("TESS:", len(tess_dt.t), "Tfull=", compute_T_full(tess_dt), "weight=", estimate_dataset_weight(tess_dt, mode=JOINT_WEIGHT_MODE, scale_free_basis=SCALE_FREE_WEIGHT_BASIS))
for k in ["q", "u", "p"]:
    ts = pol_dt[k]
    ngrp = int(np.unique(ts.group_id).size) if ts.group_id is not None else 0
    print(ts.name, "N=", len(ts.t), "Tfull=", compute_T_full(ts), "weight=", estimate_dataset_weight(ts, mode=JOINT_WEIGHT_MODE, scale_free_basis=SCALE_FREE_WEIGHT_BASIS), "night groups=", ngrp)

results = {}
for k in ["q", "u", "p"]:
    outdir = OUTROOT / k
    print("\n=== RUN:", k, "->", outdir, "===")
    results[k] = run_joint_extraction_one(tess_dt, pol_dt[k], outdir=outdir, pol_trend_cfg=POL_TREND_CFG,
                                          show_plots_inline=SHOW_PLOTS_INLINE)

results["q"].head()


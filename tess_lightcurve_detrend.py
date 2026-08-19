#!/usr/bin/env python3
"""Standalone detrending for light curves produced by tess_voronoi_lc_pipeline.py."""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import pickle
import gzip
import lzma
import urllib.error
import urllib.request

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator

try:
    from astropy.io import fits
except Exception:
    fits = None

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


def infer_sector_from_csv_stem(stem: str):
    m = re.search(r'(^|[_-])s(\d{4})([_-]|$)', stem.lower())
    if m:
        return int(m.group(2))
    m = re.search(r's(\d{4})', stem.lower())
    if m:
        return int(m.group(1))
    return None


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



# =============================================================================
# QLP-style quaternion regression
# =============================================================================

_QUAT_FILE_SUFFIXES = (
    ".csv", ".csv.gz", ".csv.xz", ".ecsv", ".txt",
    ".fits", ".fit", ".fits.gz", ".fit.gz",
)

TESSVECTORS_BASE_URL = "https://heasarc.gsfc.nasa.gov/docs/tess/data/TESSVectors/Vectors"
DEFAULT_TESSVECTORS_CACHE = "~/.cache/photpol/tessvectors"


def _tessvectors_cadence_info(cadence_seconds: float | None) -> tuple[str, str]:
    """Return the HEASARC subdirectory and filename cadence code."""
    if cadence_seconds is None or not np.isfinite(cadence_seconds):
        raise ValueError("Could not determine the light-curve cadence for TESSVectors download.")
    cadence_seconds = float(cadence_seconds)
    if cadence_seconds <= 30.0:
        return "020_Cadence", "020"
    if cadence_seconds <= 180.0:
        return "120_Cadence", "120"
    return "FFI_Cadence", "FFI"


def download_tessvectors_to_cache(
    cache_dir: str | Path,
    sector: int,
    camera: int,
    cadence_seconds: float,
    base_url: str = TESSVECTORS_BASE_URL,
    timeout: float = 180.0,
) -> Path:
    """Download one missing sector/camera/cadence TESSVectors CSV atomically."""
    if sector is None or int(sector) < 1:
        raise ValueError("Could not infer the TESS sector needed for automatic TESSVectors download.")
    if camera is None or int(camera) not in (1, 2, 3, 4):
        raise ValueError(
            "Could not infer the TESS camera needed for automatic TESSVectors download. "
            "Select camera 1-4 explicitly in the GUI or provide camera metadata in the light-curve CSV."
        )

    subdir, code = _tessvectors_cadence_info(cadence_seconds)
    filename = f"TessVectors_S{int(sector):03d}_C{int(camera)}_{code}.csv"
    cache_root = Path(cache_dir).expanduser().resolve()
    destination = cache_root / subdir / filename
    if destination.exists() and destination.stat().st_size > 0:
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    url = f"{str(base_url).rstrip('/')}/{subdir}/{filename}"
    partial = destination.with_name(destination.name + ".part")
    try:
        partial.unlink()
    except FileNotFoundError:
        pass

    print(f"  Downloading missing TESSVectors file: {url}")
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "PhotPol-TESSVectors/1.0"},
    )
    try:
        with urllib.request.urlopen(request, timeout=float(timeout)) as response, open(partial, "wb") as out:
            while True:
                block = response.read(1024 * 1024)
                if not block:
                    break
                out.write(block)
        if not partial.exists() or partial.stat().st_size == 0:
            raise IOError("Downloaded file was empty.")
        partial.replace(destination)
    except urllib.error.HTTPError as exc:
        try:
            partial.unlink()
        except FileNotFoundError:
            pass
        if exc.code == 404:
            raise FileNotFoundError(
                f"No TESSVectors product is available at {url}. "
                "The public TESSVectors archive may not yet cover this sector/cadence."
            ) from exc
        raise RuntimeError(f"TESSVectors download failed with HTTP {exc.code}: {url}") from exc
    except Exception:
        try:
            partial.unlink()
        except FileNotFoundError:
            pass
        raise

    print(f"  Cached TESSVectors file: {destination} ({destination.stat().st_size / (1024**2):.1f} MiB)")
    return destination


def _norm_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name).lower())


def _first_matching_column(columns, candidates):
    norm_to_orig = {_norm_name(c): c for c in columns}
    for candidate in candidates:
        key = _norm_name(candidate)
        if key in norm_to_orig:
            return norm_to_orig[key]
    return None


def _numeric_series(values):
    return pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(float)


def _read_fits_tables(path: Path) -> list[pd.DataFrame]:
    if fits is None:
        raise ImportError(
            "Reading quaternion FITS files requires astropy. "
            "Install astropy or provide a CSV/TESSVectors file."
        )
    tables = []
    with fits.open(path, memmap=True) as hdul:
        for hdu_index, hdu in enumerate(hdul):
            data = getattr(hdu, "data", None)
            names = list(getattr(data, "names", []) or [])
            if data is None or not names:
                continue
            cols = {}
            nrow = len(data)
            for name in names:
                try:
                    arr = np.asarray(data[name])
                except Exception:
                    continue
                if arr.ndim == 1:
                    if arr.dtype.kind in "biufc":
                        cols[str(name)] = np.asarray(arr)
                    elif arr.dtype.kind in "SU":
                        cols[str(name)] = np.asarray(arr).astype(str)
                elif arr.ndim == 2 and arr.shape[0] == nrow and arr.shape[1] <= 16:
                    for j in range(arr.shape[1]):
                        cols[f"{name}_{j+1}"] = np.asarray(arr[:, j])
            if not cols:
                continue
            tab = pd.DataFrame(cols)
            for header_key in ("CAMERA", "CAM", "SECTOR"):
                if header_key in hdu.header and header_key.lower() not in tab.columns:
                    tab[header_key.lower()] = hdu.header[header_key]
            tab["__source_hdu__"] = hdu_index
            tables.append(tab)
    if not tables:
        raise ValueError(f"No readable binary-table HDU found in quaternion FITS file: {path}")
    return tables


def _open_text_auto(path: Path):
    name = path.name.lower()
    if name.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    if name.endswith(".xz"):
        return lzma.open(path, "rt", encoding="utf-8", errors="replace")
    return open(path, "rt", encoding="utf-8", errors="replace")


def _read_delimited_quaternion_table(path: Path) -> pd.DataFrame:
    name = path.name.lower()
    if name.endswith(".txt"):
        return pd.read_csv(path, sep=r"\s+", comment="#", compression="infer")

    # TESSVectors files have a long commented preamble; some releases contain
    # a wrapped description line that does not begin with '#'. Find the actual
    # comma-separated header rather than relying only on comment parsing.
    header_row = None
    with _open_text_auto(path) as fh:
        for lineno, line in enumerate(fh):
            low = line.lower()
            if "," in line and ("midtime" in low or "time" in low) and "quat" in low:
                header_row = lineno
                break
            if lineno > 500:
                break
    if header_row is None:
        df = pd.read_csv(path, comment="#", compression="infer")
    else:
        df = pd.read_csv(path, skiprows=header_row, compression="infer")
    unnamed = [c for c in df.columns if str(c).lower().startswith("unnamed:")]
    if unnamed:
        df = df.drop(columns=unnamed)
    return df


def read_quaternion_tables(path: Path) -> list[pd.DataFrame]:
    name = path.name.lower()
    if name.endswith((".fits", ".fit", ".fits.gz", ".fit.gz")):
        return _read_fits_tables(path)
    if name.endswith(".ecsv"):
        try:
            from astropy.table import Table
            return [Table.read(path, format="ascii.ecsv").to_pandas()]
        except Exception:
            return [_read_delimited_quaternion_table(path)]
    return [_read_delimited_quaternion_table(path)]


def _sector_tokens(sector: int) -> list[str]:
    return [
        f"s{sector:04d}", f"s{sector:03d}", f"sector{sector:04d}",
        f"sector{sector:03d}", f"sector{sector}",
    ]


def _camera_tokens(camera: int) -> list[str]:
    return [f"c{camera}", f"cam{camera}", f"camera{camera}"]


def infer_camera_from_lightcurve(df: pd.DataFrame, stem: str) -> int | None:
    for candidate in ("camera", "cam", "camera_number", "camera_num"):
        col = _first_matching_column(df.columns, [candidate])
        if col is None:
            continue
        vals = pd.to_numeric(df[col], errors="coerce")
        vals = vals[np.isfinite(vals)]
        unique = sorted(set(int(v) for v in vals if 1 <= int(v) <= 4))
        if len(unique) == 1:
            return unique[0]

    text = str(stem).lower()
    for pattern in (
        r"(?:^|[_-])camera([1-4])(?:[_-]|$)",
        r"(?:^|[_-])cam([1-4])(?:[_-]|$)",
        r"(?:^|[_-])c([1-4])(?:[_-]|$)",
    ):
        m = re.search(pattern, text)
        if m:
            return int(m.group(1))

    # TESSCut/Astrocut filenames encode camera and CCD immediately after the
    # sector: tess-sSSSS-camera-ccd_... .  Current extractor outputs retain
    # both tpf_name and tpf_path, so this also works without reopening FITS.
    for candidate in ("tpf_name", "tpf_path", "source_file", "source_path"):
        col = _first_matching_column(df.columns, [candidate])
        if col is None:
            continue
        for raw in df[col].dropna().astype(str).unique()[:20]:
            m = re.search(
                r"tess-s\d{4}-([1-4])-[1-4](?:_|-)",
                raw,
                flags=re.IGNORECASE,
            )
            if m:
                return int(m.group(1))
    return None


def _camera_from_tpf_header(path: Path) -> int | None:
    if fits is None or not path.is_file():
        return None
    try:
        with fits.open(path, memmap=True) as hdul:
            for hdu in hdul:
                for key in ("CAMERA", "CAM", "CAMERA_NUM", "CAMERANUM"):
                    if key not in hdu.header:
                        continue
                    try:
                        camera = int(float(hdu.header[key]))
                    except (TypeError, ValueError):
                        continue
                    if camera in (1, 2, 3, 4):
                        return camera
    except Exception:
        return None
    return None


def infer_camera_from_nearby_tpf(lc_df: pd.DataFrame, csv_path: Path) -> int | None:
    """Recover camera metadata for older extractor CSVs from their source TPF."""
    source_strings = []
    for candidate in ("tpf_path", "tpf_name", "source_path", "source_file"):
        col = _first_matching_column(lc_df.columns, [candidate])
        if col is not None:
            source_strings.extend(lc_df[col].dropna().astype(str).unique()[:20])
    if not source_strings:
        return None

    checked = set()
    search_roots = [csv_path.parent, csv_path.parent.parent]
    for raw in source_strings:
        raw_path = Path(raw).expanduser()
        candidates = []
        if raw_path.is_absolute():
            candidates.append(raw_path)
        else:
            candidates.extend(root / raw_path for root in search_roots)
            candidates.extend(root / raw_path.name for root in search_roots)
        for candidate in candidates:
            try:
                resolved = candidate.resolve()
            except Exception:
                resolved = candidate
            if resolved in checked:
                continue
            checked.add(resolved)
            camera = _camera_from_tpf_header(resolved)
            if camera is not None:
                return camera

        # The usual layout puts LC_output beside the original TPF. Search only
        # the two local roots and only for the exact recorded basename.
        basename = raw_path.name
        for root in search_roots:
            if not root.is_dir():
                continue
            try:
                matches = list(root.rglob(basename))
            except Exception:
                matches = []
            for match in matches[:10]:
                camera = _camera_from_tpf_header(match)
                if camera is not None:
                    return camera
    return None


def infer_camera_from_source_names(source: str | Path, sector: int | None) -> int | None:
    root = Path(source).expanduser()
    files = [root] if root.is_file() else [
        p for p in root.rglob("*") if p.is_file()
        and any(p.name.lower().endswith(suff) for suff in _QUAT_FILE_SUFFIXES)
    ] if root.is_dir() else []
    cameras = set()
    for p in files:
        lname = p.name.lower()
        if sector is not None and not any(tok in lname for tok in _sector_tokens(sector)):
            continue
        for pattern in (
            r"(?:^|[_-])camera([1-4])(?:[_-]|$)",
            r"(?:^|[_-])cam([1-4])(?:[_-]|$)",
            r"(?:^|[_-])c([1-4])(?:[_-]|$)",
        ):
            m = re.search(pattern, lname)
            if m:
                cameras.add(int(m.group(1)))
                break
    if len(cameras) == 1:
        return next(iter(cameras))
    if len(cameras) > 1:
        raise ValueError(
            f"Quaternion source contains files for multiple cameras {sorted(cameras)}. "
            "Select the target camera explicitly."
        )
    return None


def resolve_quaternion_files(source: str | Path, sector: int | None,
                             camera: int | None, cadence_seconds: float | None) -> list[Path]:
    root = Path(source).expanduser()
    if root.is_file():
        return [root.resolve()]
    if not root.exists():
        raise FileNotFoundError(f"Quaternion source does not exist: {root}")
    if not root.is_dir():
        raise ValueError(f"Quaternion source must be a file or directory: {root}")

    files = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        lname = p.name.lower()
        if any(lname.endswith(suff) for suff in _QUAT_FILE_SUFFIXES):
            files.append(p)
    if not files:
        raise FileNotFoundError(f"No quaternion/vector files found under: {root}")

    def score(p: Path):
        lname = p.name.lower()
        s = 0
        if sector is not None:
            if any(tok in lname for tok in _sector_tokens(sector)):
                s += 100
            else:
                s -= 100
        if camera is not None:
            if any(re.search(rf"(?:^|[_-]){re.escape(tok)}(?:[_-]|$)", lname)
                   for tok in _camera_tokens(camera)):
                s += 30
        if cadence_seconds is not None and np.isfinite(cadence_seconds):
            if cadence_seconds <= 30:
                if "20" in lname:
                    s += 10
            elif cadence_seconds <= 180:
                if "120" in lname or "2min" in lname or "2-min" in lname:
                    s += 10
            else:
                if "ffi" in lname or "600" in lname or "1800" in lname or "200" in lname:
                    s += 10
        if "quat" in lname:
            s += 5
        if "vector" in lname:
            s += 4
        return s

    scored = sorted(((score(p), p) for p in files), key=lambda item: (-item[0], str(item[1])))
    best = scored[0][0]
    if sector is not None and best < 0:
        raise FileNotFoundError(
            f"No quaternion file under {root} appears to match sector {sector}. "
            "Use a sector-specific file or directory."
        )

    # Keep files close to the best score. This allows the two orbit-half raw
    # engineering files for a sector to be concatenated, while avoiding files
    # for other cadences/cameras.
    keep = [p for s, p in scored if s >= best - 2]
    return keep


def _filter_table_camera(df: pd.DataFrame, camera: int | None) -> pd.DataFrame:
    if camera is None:
        return df
    col = _first_matching_column(df.columns, ["camera", "cam", "camera_number"])
    if col is None:
        return df
    vals = pd.to_numeric(df[col], errors="coerce")
    selected = df.loc[vals == int(camera)].copy()
    return selected if len(selected) else df


def _time_column(df: pd.DataFrame) -> str | None:
    return _first_matching_column(
        df.columns,
        [
            "MidTime", "time_btjd", "btjd", "tjd", "time", "bjd",
            "mjd", "timestamp", "quat_time", "quaternion_time",
        ],
    )


def _to_btjd_like(values, column_name: str = ""):
    arr = _numeric_series(values)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return arr
    med = float(np.nanmedian(finite))
    name = _norm_name(column_name)
    if "mjd" in name or (40000.0 < med < 100000.0):
        return arr - 56999.5
    if "jd" in name and "tjd" not in name and "btjd" not in name:
        if med > 2_000_000:
            return arr - 2457000.0
    if med > 2_000_000:
        return arr - 2457000.0
    return arr


def _column_for_component(columns, component: int, camera: int | None,
                          statistic: str | None = None) -> str | None:
    stat_aliases = {
        None: [""],
        "mean": ["mean", "avg", "average", "med", "median"],
        "std": ["std", "stddev", "sigma", "sigclip"],
        "skew": ["skew", "skewness"],
    }[statistic]
    prefixes = ["q", "quat", "quaternion"]
    candidates = []
    for pref in prefixes:
        for stat in stat_aliases:
            base_variants = [
                f"{pref}{component}{stat}",
                f"{pref}{component}_{stat}",
                f"{pref}_{component}_{stat}",
                f"{pref}{stat}{component}",
            ]
            candidates.extend(base_variants)
            if camera is not None:
                for camtok in (f"c{camera}", f"cam{camera}", f"camera{camera}"):
                    for base in base_variants:
                        candidates.extend([
                            f"{camtok}_{base}", f"{camtok}{base}",
                            f"{base}_{camtok}", f"{base}{camtok}",
                        ])
    return _first_matching_column(columns, candidates)


def _raw_quaternion_columns(df: pd.DataFrame, camera: int | None) -> list[str] | None:
    cols = []
    for component in (1, 2, 3):
        col = _column_for_component(df.columns, component, camera, statistic=None)
        if col is None:
            return None
        cols.append(col)
    return cols


def _prebinned_stat_columns(df: pd.DataFrame, camera: int | None) -> dict[str, list[str]]:
    found = {}
    for stat in ("mean", "std", "skew"):
        cols = []
        for component in (1, 2, 3):
            col = _column_for_component(df.columns, component, camera, statistic=stat)
            if col is None:
                cols = []
                break
            cols.append(col)
        if cols:
            found[stat] = cols
    return found


def _robust_skew(values):
    x = np.asarray(values, float)
    x = x[np.isfinite(x)]
    if x.size < 3:
        return np.nan
    mu = np.nanmean(x)
    sig = np.nanstd(x)
    if not np.isfinite(sig) or sig <= 0:
        return 0.0
    return float(np.nanmean(((x - mu) / sig) ** 3))


def _typical_cadence_seconds(t):
    tt = np.sort(np.asarray(t, float)[np.isfinite(t)])
    if tt.size < 2:
        return np.nan
    dt = np.diff(tt)
    dt = dt[(dt > 0) & np.isfinite(dt)]
    return float(np.nanmedian(dt) * 86400.0) if dt.size else np.nan


def _lightcurve_match_times(t, df_masked: pd.DataFrame,
                            source_times: np.ndarray,
                            time_offset_days: str | float = "auto"):
    t = np.asarray(t, float)
    for candidate in ("timecorr", "time_corr", "barycorr", "barycentric_correction"):
        col = _first_matching_column(df_masked.columns, [candidate])
        if col is not None:
            tc = pd.to_numeric(df_masked[col], errors="coerce").to_numpy(float)
            if len(tc) == len(t) and np.isfinite(tc).sum() >= max(3, len(t) // 2):
                return t - tc, float(np.nanmedian(tc)), f"column:{col}"

    if isinstance(time_offset_days, str) and time_offset_days.strip().lower() == "auto":
        src = np.asarray(source_times, float)
        src = src[np.isfinite(src)]
        if src.size == 0:
            return t.copy(), 0.0, "auto_failed"
        # Engineering/TESSVectors times are normally spacecraft times, while
        # the extracted TPF times are barycentric. Align their span centers.
        offset = 0.5 * ((np.nanmin(t) + np.nanmax(t)) - (np.nanmin(src) + np.nanmax(src)))
        return t - offset, float(offset), "auto_span_center"

    offset = float(time_offset_days)
    return t - offset, offset, "manual"


def _nearest_indices(source_t: np.ndarray, target_t: np.ndarray):
    source_t = np.asarray(source_t, float)
    target_t = np.asarray(target_t, float)
    order = np.argsort(source_t)
    st = source_t[order]
    pos = np.searchsorted(st, target_t)
    pos0 = np.clip(pos - 1, 0, len(st) - 1)
    pos1 = np.clip(pos, 0, len(st) - 1)
    choose1 = np.abs(st[pos1] - target_t) < np.abs(st[pos0] - target_t)
    chosen = np.where(choose1, pos1, pos0)
    return order[chosen], np.abs(st[chosen] - target_t)


def _build_qlp_feature_matrix(stat_arrays: dict[str, np.ndarray]):
    base_cols = []
    base_names = []
    for stat in ("mean", "std", "skew"):
        if stat not in stat_arrays:
            continue
        arr = np.asarray(stat_arrays[stat], float)
        if arr.ndim != 2 or arr.shape[1] != 3:
            continue
        for j in range(3):
            base_cols.append(arr[:, j])
            base_names.append(f"quat_{stat}_q{j+1}")
        for i, j in ((0, 1), (0, 2), (1, 2)):
            base_cols.append(arr[:, i] * arr[:, j])
            base_names.append(f"quat_{stat}_q{i+1}xq{j+1}")

    if not base_cols:
        raise ValueError("No usable quaternion statistics were available to build regressors.")

    first = np.column_stack(base_cols)
    first_names = list(base_names)
    squared = first ** 2
    squared_names = [f"{name}_sq" for name in first_names]
    X = np.column_stack([first, squared])
    names = first_names + squared_names
    return X, names


def _standardize_quaternion_features(X: np.ndarray, names: list[str]):
    X = np.asarray(X, float)
    center = np.nanmean(X, axis=0)
    scale = np.nanstd(X, axis=0)
    finite_fraction = np.mean(np.isfinite(X), axis=0)
    good_cols = (
        np.isfinite(center) & np.isfinite(scale) & (scale > 1e-14)
        & (finite_fraction >= 0.25)
    )
    if not np.any(good_cols):
        raise ValueError("All quaternion feature columns were empty or constant.")
    X = X[:, good_cols]
    center = center[good_cols]
    scale = scale[good_cols]
    names = [name for name, keep in zip(names, good_cols) if keep]
    Xz = (X - center) / scale
    coverage = np.all(np.isfinite(Xz), axis=1)
    X_predict = np.where(np.isfinite(Xz), Xz, 0.0)
    return X_predict, names, center, scale, coverage


def prepare_qlp_quaternion_regressors(
    source: str | Path,
    t_lc: np.ndarray,
    lc_df_masked: pd.DataFrame,
    stem: str,
    camera_setting: str = "auto",
    min_samples: int = 3,
    time_offset_days: str | float = "auto",
):
    sector = infer_sector_from_csv_stem(stem)
    cadence_seconds = _typical_cadence_seconds(t_lc)

    camera = None
    setting = str(camera_setting).strip().lower()
    if setting not in ("", "auto"):
        camera = int(setting)
        if camera not in (1, 2, 3, 4):
            raise ValueError("--quaternion-camera must be auto or one of 1,2,3,4.")
    if camera is None:
        camera = infer_camera_from_lightcurve(lc_df_masked, stem)
    if camera is None:
        camera = infer_camera_from_source_names(source, sector)

    files = resolve_quaternion_files(source, sector, camera, cadence_seconds)
    tables = []
    source_labels = []
    for path in files:
        for table in read_quaternion_tables(path):
            if len(table) == 0:
                continue
            tables.append(_filter_table_camera(table, camera))
            source_labels.append(str(path))
    tables = [tab for tab in tables if len(tab)]
    if not tables:
        raise ValueError("Quaternion source files contained no usable rows.")

    # Select tables that have a time column and either raw components or
    # pre-binned statistics.
    usable = []
    for tab in tables:
        tc = _time_column(tab)
        raw_cols = _raw_quaternion_columns(tab, camera)
        stat_cols = _prebinned_stat_columns(tab, camera)
        if tc is not None and (raw_cols is not None or stat_cols):
            usable.append((tab, tc, raw_cols, stat_cols))
    if not usable:
        examples = sorted(set(str(c) for tab in tables for c in tab.columns))[:40]
        raise ValueError(
            "Could not identify quaternion time/components in the selected source. "
            f"First available columns: {examples}"
        )

    # Prefer exact raw samples when available, because they allow the QLP
    # mean/std/skew construction. Otherwise use pre-binned statistics.
    raw_usable = [item for item in usable if item[2] is not None and not item[3]]
    if not raw_usable:
        # A table can contain both raw-looking Q1/Q2/Q3 and explicit statistics.
        # Explicit statistics identify it as pre-binned.
        raw_usable = [item for item in usable if item[2] is not None and not any(
            _norm_name(c).endswith(("mean", "med", "median", "std", "stddev", "skew", "skewness"))
            for c in item[2]
        ) and not item[3]]

    if raw_usable:
        raw_frames = []
        for tab, tc, raw_cols, _ in raw_usable:
            frame = pd.DataFrame({
                "time": _to_btjd_like(tab[tc], tc),
                "q1": _numeric_series(tab[raw_cols[0]]),
                "q2": _numeric_series(tab[raw_cols[1]]),
                "q3": _numeric_series(tab[raw_cols[2]]),
            })
            raw_frames.append(frame)
        raw = pd.concat(raw_frames, ignore_index=True)
        raw = raw.replace([np.inf, -np.inf], np.nan).dropna(subset=["time"])
        raw = raw.sort_values("time").drop_duplicates("time", keep="first")
        source_t = raw["time"].to_numpy(float)
        t_match, offset, align_method = _lightcurve_match_times(
            t_lc, lc_df_masked, source_t, time_offset_days=time_offset_days
        )

        cadence_days = max(_typical_cadence_seconds(t_lc) / 86400.0, 1.0 / 86400.0)
        half_width = 0.5 * cadence_days
        st = source_t
        q = raw[["q1", "q2", "q3"]].to_numpy(float)
        stats = {
            "mean": np.full((len(t_lc), 3), np.nan),
            "std": np.full((len(t_lc), 3), np.nan),
            "skew": np.full((len(t_lc), 3), np.nan),
        }
        counts = np.zeros(len(t_lc), dtype=int)
        for i, center in enumerate(t_match):
            lo = np.searchsorted(st, center - half_width, side="left")
            hi = np.searchsorted(st, center + half_width, side="right")
            vals = q[lo:hi]
            good_rows = np.all(np.isfinite(vals), axis=1)
            vals = vals[good_rows]
            counts[i] = len(vals)
            if len(vals) < int(min_samples):
                continue
            stats["mean"][i] = np.nanmean(vals, axis=0)
            stats["std"][i] = np.nanstd(vals, axis=0)
            stats["skew"][i] = [_robust_skew(vals[:, j]) for j in range(3)]
        Xraw, names = _build_qlp_feature_matrix(stats)
        source_mode = "raw_qlp_36"
        source_count = counts
    else:
        frames = []
        available_stats = set()
        for tab, tc, _, stat_cols in usable:
            if not stat_cols:
                continue
            frame = pd.DataFrame({"time": _to_btjd_like(tab[tc], tc)})
            for stat, cols in stat_cols.items():
                available_stats.add(stat)
                for j, col in enumerate(cols, start=1):
                    frame[f"{stat}_q{j}"] = _numeric_series(tab[col])
            frames.append(frame)
        if not frames:
            raise ValueError("No usable pre-binned quaternion statistics were found.")
        vec = pd.concat(frames, ignore_index=True)
        vec = vec.replace([np.inf, -np.inf], np.nan).dropna(subset=["time"])
        vec = vec.sort_values("time").drop_duplicates("time", keep="first")
        source_t = vec["time"].to_numpy(float)
        t_match, offset, align_method = _lightcurve_match_times(
            t_lc, lc_df_masked, source_t, time_offset_days=time_offset_days
        )
        idx, dist = _nearest_indices(source_t, t_match)
        tol = max(5.0 / 86400.0, 0.60 * max(_typical_cadence_seconds(t_lc), 1.0) / 86400.0)
        matched = dist <= tol
        stats = {}
        for stat in ("mean", "std", "skew"):
            cols = [f"{stat}_q{j}" for j in (1, 2, 3)]
            if all(c in vec.columns for c in cols):
                arr = vec[cols].to_numpy(float)[idx]
                arr[~matched, :] = np.nan
                stats[stat] = arr
        Xraw, names = _build_qlp_feature_matrix(stats)
        source_mode = "prebinned_" + "_".join(sorted(stats))
        source_count = matched.astype(int)

    X, names, feature_center, feature_scale, coverage = _standardize_quaternion_features(
        Xraw, names
    )
    return {
        "X": X,
        "names": names,
        "feature_center": feature_center,
        "feature_scale": feature_scale,
        "coverage": coverage,
        "source_count": source_count,
        "source_files": sorted(set(source_labels)),
        "source_mode": source_mode,
        "sector": sector,
        "camera": camera,
        "time_offset_days": offset,
        "alignment_method": align_method,
    }


def qlp_sigma_clipped_fit(X, y, initial_keep=None, sigma: float = 3.0,
                          n_iter: int = 5):
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    base = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    if initial_keep is not None:
        base &= np.asarray(initial_keep, bool)
    keep = base.copy()
    beta = np.zeros(X.shape[1], float)

    for _ in range(max(1, int(n_iter))):
        if keep.sum() < max(5, X.shape[1] + 1):
            break
        beta, *_ = np.linalg.lstsq(X[keep], y[keep], rcond=None)
        resid = y - X @ beta
        rr = resid[keep]
        med = np.nanmedian(rr)
        mad = np.nanmedian(np.abs(rr - med))
        scatter = 1.4826 * mad if np.isfinite(mad) and mad > 0 else np.nanstd(rr)
        if not np.isfinite(scatter) or scatter <= 0 or sigma <= 0:
            break
        new_keep = base & (np.abs(resid - med) <= float(sigma) * scatter)
        if np.array_equal(new_keep, keep):
            keep = new_keep
            break
        keep = new_keep

    if keep.sum() >= max(5, X.shape[1] + 1):
        beta, *_ = np.linalg.lstsq(X[keep], y[keep], rcond=None)
    elif base.sum() >= max(5, X.shape[1] + 1):
        beta, *_ = np.linalg.lstsq(X[base], y[base], rcond=None)
        keep = base
    else:
        raise ValueError(
            f"Too few valid cadences for quaternion regression: "
            f"{int(base.sum())} rows for {X.shape[1]} fitted columns."
        )
    return beta, keep


def save_quaternion_diagnostic_plot(t, flux_resid, q_model, flux_corrected,
                                    coverage, outpng: Path, title: str):
    fig, axes = plt.subplots(4, 1, figsize=(10.5, 8.5), sharex=True)
    axes[0].plot(t, flux_resid, ".", ms=2.0)
    axes[0].set_ylabel("PCHIP residual")
    axes[1].plot(t, q_model, ".", ms=2.0)
    axes[1].set_ylabel("Quaternion model")
    axes[2].plot(t, flux_corrected, ".", ms=2.0)
    axes[2].set_ylabel("Systematics corrected")
    axes[3].plot(t, np.asarray(coverage, int), ".", ms=2.0)
    axes[3].set_ylabel("Quat. coverage")
    axes[3].set_xlabel("Time [BTJD]")
    for ax in axes:
        ax.grid(alpha=0.25)
    fig.suptitle(title, y=0.995)
    fig.tight_layout()
    save_figure_with_optional_pickle(fig, outpng, dpi=180, bbox_inches="tight")
    plt.close(fig)


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
                   help="Skip centroid/background decorrelation entirely, but still allow the optional PCHIP correction.")
    p.add_argument("--use-quaternion-regression", action="store_true",
                   help="Add QLP-style camera-quaternion regressors to the systematics fit.")
    p.add_argument("--quaternion-source", type=str, default="",
                   help="Optional local quaternion engineering file, TESSVectors CSV, or directory containing sector files.")
    p.add_argument("--quaternion-auto-download", action="store_true",
                   help="When no local quaternion source is supplied, download the required TESSVectors CSV into the local cache.")
    p.add_argument("--tessvectors-cache-dir", type=str, default=DEFAULT_TESSVECTORS_CACHE,
                   help="Local cache root used by automatic TESSVectors downloading.")
    p.add_argument("--tessvectors-base-url", type=str, default=TESSVECTORS_BASE_URL,
                   help="Base URL of the HEASARC TESSVectors archive.")
    p.add_argument("--quaternion-camera", choices=["auto", "1", "2", "3", "4"], default="auto",
                   help="TESS camera for quaternion regressors. Auto uses light-curve metadata when available.")
    p.add_argument("--quaternion-min-samples", type=int, default=3,
                   help="Minimum raw 2-second quaternion samples required per light-curve cadence.")
    p.add_argument("--quaternion-clip-sigma", type=float, default=3.0,
                   help="Sigma threshold for iterative QLP-style regression clipping.")
    p.add_argument("--quaternion-clip-iters", type=int, default=5,
                   help="Maximum iterations of QLP-style sigma-clipped linear regression.")
    p.add_argument("--quaternion-time-offset-days", type=str, default="auto",
                   help="Barycentric-minus-spacecraft time offset in days, or 'auto'. A timecorr column is preferred when present.")
    p.add_argument("--save-quaternion-diagnostics", action="store_true",
                   help="Save quaternion feature/model NPZ data and a diagnostic PNG.")
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
            "flux_detrended_rel" if "flux_detrended_rel" in df.columns else
            ("flux_rel" if "flux_rel" in df.columns else
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

        use_quat = bool(getattr(args, "use_quaternion_regression", False))
        quaternion_source = str(getattr(args, "quaternion_source", "")).strip()
        quaternion_auto_download = bool(getattr(args, "quaternion_auto_download", False))
        if use_quat and not quaternion_source and not quaternion_auto_download:
            raise ValueError(
                "Quaternion regression requires either --quaternion-source or "
                "--quaternion-auto-download."
            )
        if use_quat and not getattr(args, "pre_model_pchip", False):
            print("  [WARN] Quaternion regression is enabled without --pre-model-pchip; "
                  "intrinsic stellar variability may be absorbed by the regression.")

        # Preserve the masked input dataframe so a timecorr/camera column can be
        # used for quaternion alignment when available.
        df_masked = df.iloc[:n].loc[mask].reset_index(drop=True)

        if getattr(args, "skip_xybg_decorrelation", False):
            X_xy = np.ones((len(t), 1), dtype=float)
            xy_names = ["intercept"]
        else:
            X_xy = build_design_matrix(t, ccol, crow, knot_spacing_days=args.knot_spacing_days)
            xy_names = [f"xybg_{i}" for i in range(X_xy.shape[1])]
            if bg is not None:
                dbg = bg - np.nanmedian(bg)
                X_xy = np.column_stack([X_xy, dbg])
                xy_names.append("background")

        quat_info = None
        X_quat = np.empty((len(t), 0), dtype=float)
        if use_quat:
            resolved_quaternion_source = quaternion_source
            if not resolved_quaternion_source:
                sector_for_download = infer_sector_from_csv_stem(stem_core)
                cadence_for_download = _typical_cadence_seconds(t)
                camera_for_download = None
                camera_setting = str(args.quaternion_camera).strip().lower()
                if camera_setting not in ("", "auto"):
                    camera_for_download = int(camera_setting)
                if camera_for_download is None:
                    camera_for_download = infer_camera_from_lightcurve(df_masked, stem_core)
                if camera_for_download is None:
                    camera_for_download = infer_camera_from_nearby_tpf(df_masked, csv_path)
                    if camera_for_download is not None:
                        print(
                            "  Inferred TESS camera from the source TPF associated "
                            f"with {csv_path.name}: camera {camera_for_download}"
                        )
                resolved_quaternion_source = str(download_tessvectors_to_cache(
                    args.tessvectors_cache_dir,
                    sector_for_download,
                    camera_for_download,
                    cadence_for_download,
                    base_url=args.tessvectors_base_url,
                ))

            quat_info = prepare_qlp_quaternion_regressors(
                resolved_quaternion_source,
                t,
                df_masked,
                stem_core,
                camera_setting=args.quaternion_camera,
                min_samples=int(args.quaternion_min_samples),
                time_offset_days=args.quaternion_time_offset_days,
            )
            X_quat = np.asarray(quat_info["X"], float)
            print(
                f"  Quaternion regressors: mode={quat_info['source_mode']} "
                f"features={X_quat.shape[1]} coverage={np.mean(quat_info['coverage']):.1%} "
                f"camera={quat_info['camera']} offset={quat_info['time_offset_days']:.6g} d "
                f"({quat_info['alignment_method']})"
            )
            if not any("quat_skew_" in name for name in quat_info["names"]):
                print("  [WARN] Quaternion source lacks skew statistics; using the available "
                      "mean/median and scatter features rather than the full 36-feature QLP set.")

        X = np.column_stack([X_xy, X_quat])
        y_center = np.nanmedian(flux_variability_resid)
        y = flux_variability_resid - y_center
        fit_keep = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
        if use_quat:
            fit_keep &= np.asarray(quat_info["coverage"], bool)
        if getattr(args, "clip_residuals_before_detrend", False):
            fit_keep &= iterative_sigma_keep(
                y,
                sigma_thresh=float(args.clip_residuals_sigma),
                n_iter=int(args.clip_residuals_iters),
            )

        if use_quat:
            beta, final_fit_keep = qlp_sigma_clipped_fit(
                X,
                y,
                initial_keep=fit_keep,
                sigma=float(args.quaternion_clip_sigma),
                n_iter=int(args.quaternion_clip_iters),
            )
        else:
            final_fit_keep = fit_keep.copy()
            if fit_keep.sum() >= max(5, X.shape[1] + 1):
                beta = robust_wls(
                    X[fit_keep], y[fit_keep],
                    n_iter=args.robust_iters, huber_k=args.huber_k
                )
            else:
                beta = robust_wls(
                    X, y, n_iter=args.robust_iters, huber_k=args.huber_k
                )

        full_systematics_model = X @ beta
        n_xy = X_xy.shape[1]
        xybg_model = X_xy @ beta[:n_xy]
        quaternion_model = (
            X_quat @ beta[n_xy:] if X_quat.shape[1] else np.zeros(len(t), float)
        )
        flux_decor_resid = y - full_systematics_model + y_center
        flux_quaternion_only_resid = y - quaternion_model + y_center

        flux_decor = flux_decor_resid.copy()
        flux_quaternion_only = flux_quaternion_only_resid.copy()
        if np.any(np.isfinite(variability_model)):
            restore = variability_model - np.nanmedian(variability_model)
            flux_decor = flux_decor + restore
            flux_quaternion_only = flux_quaternion_only + restore

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
            "flux_variability_resid_rel": flux_variability_resid,
            "flux_detrend_rel": flux_final,
            "flux_decor_only_rel": flux_decor,
            "centroid_col": ccol,
            "centroid_row": crow,
        })
        if use_quat and quat_info is not None:
            out_df["flux_quaternion_corrected_rel"] = flux_decor
            out_df["flux_quaternion_only_corrected_rel"] = flux_quaternion_only
            out_df["quaternion_systematics_model"] = quaternion_model
            out_df["xybg_systematics_model"] = xybg_model
            out_df["systematics_model"] = full_systematics_model
            out_df["quaternion_valid"] = np.asarray(quat_info["coverage"], bool)
            out_df["quaternion_samples_or_match"] = np.asarray(quat_info["source_count"])
        if bg is not None:
            out_df["background"] = bg
        if np.any(np.isfinite(variability_model)):
            out_df["pre_model_pchip_trend"] = variability_model
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

        if use_quat and quat_info is not None and getattr(args, "save_quaternion_diagnostics", False):
            qdiag_npz = out_root / f"{args.prefix}{simple_stem}_quaternion_diagnostics.npz"
            np.savez_compressed(
                qdiag_npz,
                time_btjd=np.asarray(t, float),
                feature_matrix=np.asarray(X_quat, float),
                feature_names=np.asarray(quat_info["names"], dtype=str),
                feature_center=np.asarray(quat_info["feature_center"], float),
                feature_scale=np.asarray(quat_info["feature_scale"], float),
                coefficients=np.asarray(beta[n_xy:], float),
                quaternion_model=np.asarray(quaternion_model, float),
                coverage=np.asarray(quat_info["coverage"], bool),
                fit_keep=np.asarray(final_fit_keep, bool),
                source_files=np.asarray(quat_info["source_files"], dtype=str),
                source_mode=np.asarray([quat_info["source_mode"]], dtype=str),
                camera=np.asarray([-1 if quat_info["camera"] is None else quat_info["camera"]], int),
                sector=np.asarray([-1 if quat_info["sector"] is None else quat_info["sector"]], int),
                time_offset_days=np.asarray([quat_info["time_offset_days"]], float),
                alignment_method=np.asarray([quat_info["alignment_method"]], dtype=str),
            )
            qdiag_png = out_root / f"{args.prefix}{simple_stem}_quaternion_diagnostics.png"
            save_quaternion_diagnostic_plot(
                t,
                flux_variability_resid,
                quaternion_model,
                flux_decor,
                quat_info["coverage"],
                qdiag_png,
                f"{simple_stem} QLP-style quaternion regression",
            )
            print(f"  Wrote quat diagnostics: {qdiag_npz.name}")
            print(f"  Wrote quat plot       : {qdiag_png.name}")

        comb_key = combined_output_stem(stem_core)
        combined_rows.setdefault(comb_key, []).append(out_df.copy())

        print(f"  RMS raw     : {rms_ppm(flux):.1f} ppm")
        if use_quat:
            print(f"  RMS quat    : {rms_ppm(flux_quaternion_only):.1f} ppm")
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

#!/usr/bin/env python3
"""
Watershed light-curve pipeline.

Extract multi-target light curves from TESS TPFs or TESScut astrocut FITS files.
This version uses watershed segmentation for initial target regions and writes flat outputs directly into --output-root.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import re

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
    from astroquery.simbad import Simbad
except Exception:
    Simbad = None

try:
    from astroquery.gaia import Gaia
except Exception as e:
    raise ImportError("astroquery.gaia is required for the Gaia/Voronoi step.") from e


# =============================================================================
# Pure MATLAB-style saturated single-target extractor
# =============================================================================

def read_tpf_arrays_for_matlab_port(path: Path):
    """Read basic arrays from a TESS TPF FITS file for the MATLAB-style extractor."""
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

        if "CADENCENO" in names:
            cadence = np.array(data["CADENCENO"])
        elif "CADENCE" in names:
            cadence = np.array(data["CADENCE"])
        else:
            cadence = np.arange(len(time))

        flux = np.array(data["FLUX"], dtype=float)
        if flux.ndim != 3:
            raise ValueError(f"Expected FLUX to have ndim=3, got shape {flux.shape} in {path.name}")

        _, nrow, ncol = flux.shape
        flux2d = flux.reshape(flux.shape[0], nrow * ncol)

    return time, flux2d, quality, cadence, nrow, ncol


def matlab_style_extract_lightcurve_pure(
    path: Path,
    threshold: float = 3000.0,
    nback: int = 20,
    use_legacy_geometry: bool = True,
    verbose: bool = True,
):
    """Pure port of the older Polaris MATLAB-style extractor."""
    time, flux, quality, cadence, nrow, ncol = read_tpf_arrays_for_matlab_port(path)
    npix = nrow * ncol

    keep = (quality == 0) & np.isfinite(time)
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

    # Track the actual aperture pixel set used at each iteration so the
    # reconstructed final mask matches the light curve that was evaluated.
    ts_flux = [np.nansum(flux[:, els], axis=1)]
    ts_pixel_sets = [np.array(els, dtype=int).copy()]
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
            ii1 = ii + 1
            if flag[ii] == 0 and np.isfinite(pixel_means[ii]) and pixel_means[ii] > 0:
                if ii1 > 1:
                    if use_legacy_geometry:
                        cond1 = (ii1 % nrow) > 1
                        cond2 = (ii1 % ncol) < 20
                        cond3 = (ii1 != 19)
                        geom_ok = cond1 and cond2 and cond3
                    else:
                        geom_ok = True

                    if geom_ok:
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
        ts_pixel_sets.append(np.concatenate([ts_pixel_sets[-1], np.array([bb], dtype=int)]))

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

    best_pixels_linear = np.array(ts_pixel_sets[best_idx], dtype=int)
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
        "n_pixels_in_best_curve": int(best_mask.sum()),
        "best_fom": float(fom[best_idx]),
        "threshold": float(threshold),
        "nback": int(nback),
        "legacy_geometry": bool(use_legacy_geometry),
    }

    if verbose:
        print(
            f"  Pure MATLAB extractor: shape={nrow}x{ncol}, cadences={len(time)}, "
            f"best_pixels={best_idx + 1}, best_fom={fom[best_idx]:.6g}"
        )

    out = pd.DataFrame({"time_btjd": time, "flux_detrended_rel": rel_flux})
    return out, meta, mean_image.reshape(nrow, ncol), best_mask_2d, back, keep, crow, ccol


def save_aperture_plot_matlab(mean_image_2d, ap_mask_2d, outpng: Path, title: str):
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
    fig.savefig(outpng, dpi=150)
    plt.close(fig)


def save_lightcurve_plot(time_btjd, flux_rel, outpng: Path, title: str):
    fig, ax = plt.subplots(figsize=(8.0, 3.8))
    ax.plot(np.asarray(time_btjd, float), np.asarray(flux_rel, float), ".", ms=2.5)
    ax.set_xlabel("Time [BTJD]")
    ax.set_ylabel("Relative Flux")
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(outpng, dpi=180, bbox_inches="tight")
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
# Sector orbital-frequency table helpers (for MATLAB-style saturated-star mode)
# =============================================================================

def load_sector_orbtable(csv_path: str | Path):
    """Load sector midpoint time + orbital frequency table.

    Expected columns (as in tess_sector_orbfreq_midpoints.csv):
      - sector (int)
      - mid_tjd  (BTJD-style day number in this workflow; despite the name)
      - freq_cyc/day (float; orbits/day)

    Returns a dict: sector -> (mid_btjd, freq_cyc_per_day)
    """
    p = Path(csv_path).expanduser()
    if not p.exists():
        raise FileNotFoundError(f"Orbital-frequency table not found: {p}")
    df = pd.read_csv(p)
    # Normalize column names a bit
    cols = {c.strip(): c for c in df.columns}
    need = ["sector", "mid_tjd", "freq_cyc/day"]
    missing = [c for c in need if c not in cols]
    if missing:
        raise ValueError(f"Orbital-frequency table missing columns: {missing}. Found: {list(df.columns)}")
    out = {}
    for _, r in df.iterrows():
        try:
            sec = int(r[cols["sector"]])
        except Exception:
            continue
        mid_btjd = float(r[cols["mid_tjd"]])
        freq = float(r[cols["freq_cyc/day"]])
        if np.isfinite(mid_btjd) and np.isfinite(freq):
            out[sec] = (mid_btjd, freq)
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


def metric_from_lc(lc):
    med = np.nanmedian(lc)
    if not np.isfinite(med) or med == 0:
        return np.inf
    rel = lc / med
    return hf_metric_from_flux(rel)


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
        "*tp.fits", "*tpf.fits", "*tp.fits.gz", "*tpf.fits.gz",
        "*_astrocut.fits", "*_astrocut.fits.gz",
    ]
    out = []
    if recursive:
        for pat in pats:
            out.extend(search_dir.rglob(pat))
    else:
        for pat in pats:
            out.extend(search_dir.glob(pat))
    return sorted({p.resolve() for p in out})


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
    alias such as ``Polaris`` or ``Lodestar``. That keeps filenames consistent
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


def infer_sector_tag(tpf_path: Path, tpf=None) -> str:
    sec = infer_sector_from_tpf(tpf_path, tpf)
    if sec is None:
        m = re.search(r"(s\d{4})", tpf_path.name.lower())
        if m:
            return m.group(1)
        return "nosector"
    return f"s{int(sec):04d}"


def build_output_stem(sector_tag: str, source_label: str, target_idx: int, method_tag: str) -> str:
    return f"{sanitize_token(sector_tag)}_{sanitize_token(source_label)}_target{int(target_idx)}_{sanitize_token(method_tag)}"



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


def infer_single_target_label(tpf, mean_image_2d=None, gaia_radius_arcmin: float = 12.0):
    """Infer a human-friendly label for a single-target or saturated-target run.

    Preference order:
      1) OBJECT/TARGNAME-style header metadata, if genuinely human-friendly
      2) SIMBAD/common-name resolution from the intended target coordinate in the FITS header
      3) SIMBAD/common-name resolution from the brightest image pixel (fallback only)
      4) TIC-style header metadata
      5) brightest in-stamp Gaia source
      6) unknown_target

    Returns (label, gaia_source_id_or_none, gaia_g_mag_or_nan).
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
    for key in ("TICID", "TARGETID"):
        try:
            if hasattr(tpf, "meta") and key in tpf.meta and tpf.meta[key] not in (None, ""):
                return sanitize_token(f"TIC_{tpf.meta[key]}"), None, np.nan
        except Exception:
            pass

    # Fifth choice: brightest Gaia source actually inside the stamp.
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
        description="Extract multi-target light curves from TPFs (Voronoi-then-grow).",
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
        "--products-subdir",
        type=str,
        default="products",
        help="Deprecated; outputs are now written directly into --output-root in a flat structure.",
    )
    p.add_argument(
        "--lightcurves-subdir",
        type=str,
        default="lightcurves_raw",
        help="Deprecated; light-curve files are now written directly into --output-root in a flat structure.",
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
        help="Max number of Gaia targets to extract per stamp (brightest in-stamp).",
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
    "--pure-sum",
    action="store_true",
    help="Extract LC by pure summation of all aperture pixels (no weighting / metric)."
    )

    # MATLAB-style saturated-star mode (optional)
    p.add_argument(
        "--matlab-sat-mode",
        action="store_true",
        help="Enable MATLAB-style saturated-star extraction/detrending (only meaningful for n-targets=1 and heavily saturated targets).",
    )
    p.add_argument(
        "--orbtable",
        type=str,
        default="",
        help="Path to tess_sector_orbfreq_midpoints.csv providing per-sector orbital frequency and mid-sector time (BTJD-style). Required if --matlab-sat-mode is set.",
    )
    p.add_argument("--sat-thresh", type=float, default=1e5, help="Mean-image threshold to consider a pixel saturated (counts).")
    p.add_argument("--sat-min-npix", type=int, default=20, help="Minimum number of pixels above --sat-thresh to treat target as heavily saturated.")
    p.add_argument("--back-nfaint", type=int, default=20, help="Number of faintest pixels to use for per-cadence background estimate.")
    p.add_argument("--phase-bin", type=float, default=0.01, help="Phase bin width for orbital-phase template subtraction.")
    p.add_argument(
        "--matlab-pure-single-sat",
        action="store_true",
        help="For a single heavily saturated source, bypass Gaia/jump/core entirely and run the pure MATLAB-style Polaris extractor with no extra detrending.",
    )
    p.add_argument(
        "--matlab-ap-thresh",
        type=float,
        default=3000.0,
        help="Initial mean-image threshold for the pure MATLAB-style saturated-source aperture seed.",
    )
    p.add_argument(
        "--matlab-no-legacy-geometry",
        action="store_true",
        help="Disable the preserved legacy MATLAB geometry restrictions in pure MATLAB-style saturated-source mode.",
    )

    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    # Optional: load sector orbital-frequency table once
    sector_orb = None
    if getattr(args, "matlab_sat_mode", False) and (not getattr(args, "matlab_pure_single_sat", False)):
        if not args.orbtable:
            raise ValueError("--matlab-sat-mode requires --orbtable path to tess_sector_orbfreq_midpoints.csv unless --matlab-pure-single-sat is used")
        sector_orb = load_sector_orbtable(args.orbtable)

    output_root = Path(args.output_root)
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

    print(f"TPFs to process: {len(tpf_paths)}")
    for i, p in enumerate(tpf_paths, 1):
        print(f"  [{i}] {p}")

    methods_to_run = [args.method] if args.method in ("jump", "core") else ["jump", "core"]

    for tpf_path in tpf_paths:
        print("\n" + "#" * 80)
        print("TPF:", tpf_path)

        outdir = output_root

        tpf = lk.read(str(tpf_path))
        # Sector inference (used by MATLAB saturated-star mode)
        sector_num = infer_sector_from_tpf(tpf_path, tpf)
        if getattr(args, "matlab_sat_mode", False) and (sector_num is None):
            print("  [WARN] MATLAB-sat-mode enabled but could not infer SECTOR from metadata/filename; will fall back to standard pipeline.")


        # Cadence selection
        if args.no_quality0:
            keep_idx = np.arange(len(tpf.time))
        else:
            try:
                qual = np.asarray(tpf.quality, int)
                keep_idx = np.where(qual == 0)[0]
                if keep_idx.size == 0:
                    keep_idx = np.arange(len(tpf.time))
            except Exception:
                keep_idx = np.arange(len(tpf.time))

        mid = int(keep_idx[len(keep_idx) // 2]) if len(keep_idx) else 0

        flux = np.asarray(tpf.flux, float)[keep_idx, :, :]
        time = np.asarray(tpf.time.value, float)[keep_idx]
        mean_img = np.nanmean(flux, axis=0)

        # Pure MATLAB-style saturated single-target branch (Option A)
        if getattr(args, "matlab_pure_single_sat", False):
            if int(args.n_targets) != 1:
                raise ValueError("--matlab-pure-single-sat requires --n-targets=1.")
            sat_ok, sat_npix = is_heavily_saturated(mean_img, thresh=args.sat_thresh, min_npix=args.sat_min_npix)
            if sat_ok:
                print(f"  Using pure MATLAB-style saturated-source extractor (npix_above_thresh={sat_npix})")
                lc_df, meta, mean_image_2d, ap_mask_2d, back_series, keep_mask, crow, ccol = matlab_style_extract_lightcurve_pure(
                    tpf_path,
                    threshold=args.matlab_ap_thresh,
                    nback=args.back_nfaint,
                    use_legacy_geometry=(not args.matlab_no_legacy_geometry),
                    verbose=True,
                )
                sector_tag = infer_sector_tag(tpf_path, tpf)
                target_label, target_gaia_id, target_gaia_g = infer_single_target_label(
                    tpf, mean_image_2d=mean_image_2d, gaia_radius_arcmin=max(float(args.gaia_radius_arcmin), 12.0)
                )
                output_stem = build_output_stem(sector_tag, target_label, 1, "matlab_pure")
                raw_csv_path = outdir / f"{output_stem}.csv"
                # Save only labeled versions for the pure MATLAB single-target branch,
                # and remove stale unlabeled files from older runs if they exist.
                for stale_name in ("keep_idx.npy", "mean_image.npy"):
                    stale_path = outdir / stale_name
                    if stale_path.exists():
                        try:
                            stale_path.unlink()
                        except Exception:
                            pass
                np.save(outdir / f"keep_idx_{output_stem}.npy", keep_idx)
                np.save(outdir / f"mean_image_{output_stem}.npy", mean_img)
                lc_df.rename(columns={"flux_detrended_rel": "flux_rel"}, inplace=True)
                lc_df["target_label"] = target_label
                lc_df["target_index"] = 1
                lc_df["method"] = "matlab_pure"
                lc_df["sector"] = sector_tag
                lc_df["tpf_name"] = tpf_path.name
                lc_df["gaia_source_id"] = target_gaia_id if target_gaia_id is not None else ""
                lc_df["gaia_g_mag"] = target_gaia_g
                lc_df.to_csv(raw_csv_path, index=False)
                save_lightcurve_plot(
                    lc_df["time_btjd"].to_numpy(float),
                    lc_df["flux_rel"].to_numpy(float),
                    outdir / f"preferred_lc_{output_stem}.png",
                    f"{tpf_path.stem} — {target_label} / target1 (matlab_pure)",
                )
                pd.DataFrame([meta]).to_csv(outdir / f"matlab_aperture_meta_{output_stem}.csv", index=False)
                np.save(outdir / f"aperture_mask_{output_stem}.npy", np.asarray(ap_mask_2d, bool))
                np.save(outdir / f"background_{output_stem}.npy", np.asarray(back_series, float))
                np.save(outdir / f"keep_mask_{output_stem}.npy", np.asarray(keep_mask, bool))
                np.save(outdir / f"centroid_row_{output_stem}.npy", np.asarray(crow, float))
                np.save(outdir / f"centroid_col_{output_stem}.npy", np.asarray(ccol, float))
                if not args.no_aperture_plots:
                    save_aperture_plot_matlab(
                        mean_image_2d,
                        ap_mask_2d,
                        outdir / f"aperture_{output_stem}.png",
                        f"{tpf_path.stem} — {target_label} / target1 (matlab_pure)",
                    )
                print(
                    f"  Wrote {target_label} / target1 (matlab_pure): {raw_csv_path.name}"
                    f"  Npix={int(np.count_nonzero(ap_mask_2d))}"
                )
                continue
            else:
                print(f"  [INFO] --matlab-pure-single-sat requested but saturation gate failed (npix_above_thresh={sat_npix}); using standard pipeline.")

        # Save unlabeled generic versions only for the standard / multi-target path.
        np.save(outdir / "keep_idx.npy", keep_idx)
        np.save(outdir / "mean_image.npy", mean_img)

        # Target definition / Gaia usage
        use_gaia = not (args.no_gaia or args.pure_sum or args.matlab_sat_mode)

        if args.no_gaia and int(args.n_targets) != 1:
            raise ValueError("--no-gaia is only supported when --n-targets=1 (otherwise target identification is ambiguous).")

        # Always build the coordinate grid (needed for seed placement / Voronoi when Gaia is available)
        coords_grid = tpf.get_coordinates(cadence=mid)
        if not isinstance(coords_grid, SkyCoord):
            ra_grid, dec_grid = coords_grid
            coords_grid = SkyCoord(
                ra=np.asarray(ra_grid, float) * u.deg,
                dec=np.asarray(dec_grid, float) * u.deg,
            )

        # --- Fix possible shape mismatch coords_grid vs flux ---
        ny_flux, nx_flux = flux.shape[1], flux.shape[2]
        ny_cg, nx_cg = coords_grid.shape
        if (ny_cg, nx_cg) != (ny_flux, nx_flux):
            ny0 = min(ny_cg, ny_flux)
            nx0 = min(nx_cg, nx_flux)
            print(
                f"  [WARN] Shape mismatch coords_grid={coords_grid.shape} vs flux={(ny_flux, nx_flux)}; cropping to {(ny0, nx0)}"
            )
            coords_grid = coords_grid[:ny0, :nx0]
            flux = flux[:, :ny0, :nx0]
            mean_img = mean_img[:ny0, :nx0]

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
            if n_use == 1 and (getattr(args, "matlab_sat_mode", False) or getattr(args, "no_gaia", False) or getattr(args, "matlab_pure_single_sat", False)):
                try:
                    single_label, _, _ = infer_single_target_label(tpf, mean_img, gaia_radius_arcmin=float(args.gaia_radius_arcmin))
                    if single_label:
                        source_labels[0] = single_label
                except Exception:
                    pass

            seeds = [nearest_pixel_to_coord(coords_grid, tc) for tc in target_coords]
            owner = watershed_owner_map(mean_img, seeds)

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

            target_coords = [coords_grid[seed[0], seed[1]]]
            seeds = [seed]
            owner = np.zeros(mean_img.shape, dtype=int)
            sid = ["pixel_seed"]
            gmag = [np.nan]
            try:
                single_label, _, _ = infer_single_target_label(tpf, mean_img, gaia_radius_arcmin=float(args.gaia_radius_arcmin))
                source_labels = [single_label if single_label else sanitize_token("unknown_target")]
            except Exception:
                source_labels = [sanitize_token("unknown_target")]

            print("  Gaia skipped/unavailable: using pixel-defined single target")
            print(f"    [1] seed={seed}  (brightest pixel in mean image)")

        # Per-target processing
        for k in range(n_use):
            tag = f"target{k+1}"
            source_label = source_labels[k] if source_labels is not None else sanitize_token(tag)
            allowed = (owner == k)
            if not np.any(allowed):
                print(f"  [WARN] {tag}: empty watershed region; skipping.")
                continue

            for meth in methods_to_run:
                if meth == "jump":
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
                denom = np.nansum(flux[:, ap], axis=1)
                yy, xx = np.indices(ap.shape)
                crow = np.nansum(flux[:, ap] * yy[ap][None, :], axis=1) / denom
                ccol = np.nansum(flux[:, ap] * xx[ap][None, :], axis=1) / denom

                # Optional PSF-width proxy
                psf_sig = None
                if args.psf_proxy:
                    psf_sig = psf_width_sigma(flux, ap, crow, ccol)
                # -----------------------------------------------------------------
                # Optional MATLAB-style saturated-star mode (only for single-target, heavy saturation)
                # -----------------------------------------------------------------
                use_matlab = bool(getattr(args, "matlab_sat_mode", False)) and (sector_orb is not None) and (n_use == 1)
                if use_matlab:
                    sat_ok, sat_npix = is_heavily_saturated(mean_img, thresh=args.sat_thresh, min_npix=args.sat_min_npix)
                    if not sat_ok:
                        print(f"  [INFO] MATLAB-sat-mode requested but saturation gate failed (npix_above_thresh={sat_npix}); using standard pipeline.")
                        use_matlab = False
                if use_matlab:
                    if sector_num not in sector_orb:
                        print(f"  [WARN] MATLAB-sat-mode: sector {sector_num} not found in orbtable; using standard pipeline.")
                        use_matlab = False

                # Build per-cadence background from faint pixels (if needed)
                back = None
                if use_matlab:
                    back = estimate_background_faint_pixels(flux, n_faint=args.back_nfaint)
                    # Optimize background scaling and subtract
                    k_back = optimize_background_scale(lc_raw, back)
                    lc_raw2 = lc_raw - k_back * back
                    med2 = np.nanmedian(lc_raw2)
                    lc_rel2 = lc_raw2 / med2 if np.isfinite(med2) and med2 != 0 else lc_raw2
                    # Orbit-phase template detrend
                    mid_btjd, freq_cpd = sector_orb[sector_num]
                    lc_rel2_det, phase, trend = phase_template_detrend(lc_rel2, t_g, freq_cpd, phase_bin=args.phase_bin)
                    lc_work = lc_rel2_det

                    # Save diagnostics later once the informative output stem is defined below.
                else:
                    lc_work = lc_rel

                # -----------------------------------------------------------------
                # Decorrelation (standard or split-at-mid-sector with background term)
                # -----------------------------------------------------------------
                if getattr(args, "pure_sum", False):
                    lc_xy = lc_work.copy()
                else:
                    if use_matlab:
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


                 # Write outputs (flat structure, labeled by sector/source/target/method)
                output_stem = build_output_stem(sector_tag, source_label, k + 1, meth_tag)
                pd.DataFrame({"time_btjd": t_g, "flux_detrended_rel": lc_xy}).to_csv(
                    outdir / f"preferred_lc_{output_stem}.csv", index=False
                )
                save_lightcurve_plot(
                    t_g,
                    lc_xy,
                    outdir / f"preferred_lc_{output_stem}.png",
                    f"{tpf_path.stem} — {source_label} / target{k+1} ({meth_tag})",
                )
                np.save(outdir / f"aperture_mask_{output_stem}.npy", ap_mask.astype(bool))
                np.save(outdir / f"centroid_row_{output_stem}.npy", np.asarray(crow, float))
                np.save(outdir / f"centroid_col_{output_stem}.npy", np.asarray(ccol, float))

                # Aperture image
                if not args.no_aperture_plots:
                    # gmag may be unavailable if Gaia was skipped/unavailable
                    gtxt = "Gaia G=NA"
                    try:
                        if gmag is not None and np.isfinite(float(gmag[k])):
                            gtxt = f"Gaia G={float(gmag[k]):.2f}"
                    except Exception:
                        pass
                    save_aperture_plot_matlab(
                        mean_img,
                        ap_mask.astype(bool),
                        outdir / f"aperture_{output_stem}.png",
                        f"{tpf_path.stem} — {source_label} / target{k+1} ({meth_tag})\n{gtxt}",
                    )

                print(
                    f"  Wrote {tag} ({meth_tag}): preferred_lc_{tag}_{meth_tag}.csv"
                    f"  Npix={int(np.count_nonzero(ap_mask))}"
                )

    print("\nDone.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
    # MATLAB-style saturated-star mode (optional)
    p.add_argument(
        "--matlab-sat-mode",
        action="store_true",
        help="Enable MATLAB-style saturated-star detrending (uses sector orbital frequency + mid-sector split + faint-pixel background).",
    )
    p.add_argument(
        "--orbtable",
        type=str,
        default="",
        help="CSV table with columns sector, mid_tjd (BTJD), freq_cyc/day (orbits/day). Required if --matlab-sat-mode.",
    )
    p.add_argument("--sat-thresh", type=float, default=1.0e5,
                   help="Mean-image threshold for considering pixels saturated (used only for gating --matlab-sat-mode).")
    p.add_argument("--sat-min-npix", type=int, default=20,
                   help="Minimum number of pixels above --sat-thresh to treat target as heavily saturated (gating for --matlab-sat-mode).")
    p.add_argument("--back-nfaint", type=int, default=20,
                   help="Number of faintest pixels to average per cadence for background estimate (MATLAB mode).")
    p.add_argument("--phase-bin", type=float, default=0.01,
                   help="Phase bin width for orbit-phase template detrending (MATLAB mode).")



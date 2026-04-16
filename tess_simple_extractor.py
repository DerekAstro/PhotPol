#!/usr/bin/env python3
"""
tess_simple_extractor.py

Robust light-curve extractor for TESS Target Pixel Files (TPFs) with:
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
  - <stem>_preferred_lc_target{k}_{mode}.csv  (time_btjd, flux_medscaled, npix)

Notes:
  - TIME saved as BTJD (native TIME column).
  - Flux comes from FLUX (e-/s) and is then median-scaled.

Example:
  python tess_simple_extractor.py --input "/path/*_tp.fits" --n-targets 1 --outdir lc_out --save-plots
"""

import argparse
import glob
import os
from pathlib import Path

import numpy as np

# Avoid Qt/Wayland issues when saving plots in batch runs.
# If you *want* interactive windows, remove these two lines or set MPLBACKEND=TkAgg.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from astropy.io import fits


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
    """Find up to n_peaks local maxima in the mean image (pure numpy).

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
    """Load TIME, FLUX cube, and QUALITY from a TESS TPF FITS."""
    with fits.open(path, memmap=True) as hdul:
        data = hdul[1].data
        hdr = hdul[1].header
        time = np.array(data["TIME"], dtype=float)
        flux = np.array(data["FLUX"], dtype=float)  # (nt, ny, nx)
        quality = np.array(data["QUALITY"], dtype=int) if ("QUALITY" in data.columns.names) else None
    return time, flux, quality, hdr



def median_scale(flux_1d: np.ndarray) -> np.ndarray:
    med = float(np.nanmedian(flux_1d))
    if not np.isfinite(med) or med == 0.0:
        return flux_1d * np.nan
    return flux_1d / med

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


def save_lc_csv(out_path: Path, time: np.ndarray, flux_medscaled: np.ndarray, npix: int):
    arr = np.column_stack([time, flux_medscaled, np.full_like(time, npix, dtype=float)])
    header = "time_btjd,flux_medscaled,npix"
    np.savetxt(out_path, arr, delimiter=",", header=header, comments="")


def quicklook_plot(out_png: Path, time: np.ndarray, flux_medscaled: np.ndarray, title: str):
    plt.figure(figsize=(10, 3))
    plt.plot(time, flux_medscaled, ".", ms=2)
    plt.xlabel("Time [BTJD]")
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

def choose_mode(args, mean_img: np.ndarray) -> str:
    """Return one of: 'fullstamp', 'fixedap', 'apgrow'."""
    if args.aperture_mode != "auto":
        return args.aperture_mode

    if args.n_targets > 1:
        return "apgrow"

    # n_targets == 1: decide based on saturation
    sat = is_saturated(mean_img,
                       top_frac=args.sat_top_frac,
                       flat_frac_of_max=args.sat_flat_frac_of_max,
                       min_flat_pixels=args.sat_min_flat_pixels)
    if not sat:
        return "apgrow"

    # Saturated single target: choose fullstamp vs fixedap
    vb = very_bright_fraction(mean_img, frac_of_peak=args.fullstamp_bright_frac_of_peak)
    # If bleed dominates a big chunk of the stamp, fullstamp is usually safest.
    if vb >= args.fullstamp_if_bright_frac_ge:
        return "fullstamp"
    return "fixedap"


# ----------------------------
#  Main
# ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True,
                    help="TPF path or glob pattern (e.g. '/dir/*_tp.fits').")
    ap.add_argument("--outdir", default="lc_out", help="Output directory.")
    ap.add_argument("--gap-days", type=float, default=0.5,
                    help="Gap threshold (days) for segment-wise median scaling.")
    ap.add_argument("--no-quality0", action="store_true",
                    help="Disable filtering to QUALITY==0 cadences.")
    ap.add_argument("--save-plots", action="store_true",
                    help="Save quicklook PNGs (light curves and aperture overlays).")

    ap.add_argument("--n-targets", type=int, default=1,
                    help="Number of targets to extract (image peaks).")

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

    args = ap.parse_args()

    paths = sorted(glob.glob(args.input))
    if len(paths) == 0 and os.path.exists(args.input):
        paths = [args.input]
    if len(paths) == 0:
        raise SystemExit(f"No files matched: {args.input}")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    for path in paths:
        base = Path(path).stem
        print(f"TPF: {path}")

        time, flux, quality, _hdr = load_tpf_cube(path)
        good = np.isfinite(time)
        if (quality is not None) and (not args.no_quality0):
            good &= (quality == 0)
        if not np.all(good):
            dropped = int(np.size(good) - np.sum(good))
            print(f"  [INFO] Dropping {dropped} cadences (non-finite time and/or QUALITY!=0)")
            time = time[good]
            flux = flux[good, :, :]


        mean_img = np.nanmean(flux, axis=0)
        ny, nx = mean_img.shape

        mode = choose_mode(args, mean_img)
        vb = very_bright_fraction(mean_img, frac_of_peak=args.fullstamp_bright_frac_of_peak)
        if args.aperture_mode == "auto":
            sat_flag = is_saturated(mean_img,
                                    top_frac=args.sat_top_frac,
                                    flat_frac_of_max=args.sat_flat_frac_of_max,
                                    min_flat_pixels=args.sat_min_flat_pixels)
            print(f"  [AUTO] saturated={sat_flag}  very_bright_frac={vb:.3f}  -> mode={mode}")

        if mode == "fullstamp":
            lc_raw = np.nansum(flux, axis=(1, 2))
            lc = median_scale_by_segment(time, lc_raw, gap_days=args.gap_days)
            npix = int(np.isfinite(mean_img).sum())
            out_csv = outdir / f"{base}_preferred_lc_target1_fullstamp.csv"
            save_lc_csv(out_csv, time, lc, npix=npix)
            if args.save_plots:
                out_png = outdir / f"{base}_target1_fullstamp.png"
                quicklook_plot(out_png, time, lc, f"{base} : full-stamp")
            continue

        if mode == "fixedap":
            if args.fixed_center == "center":
                seed = (ny // 2, nx // 2)
            else:
                seeds = find_peak_seeds(mean_img, n_peaks=1, min_sep=1, edge=args.fixed_edge)
                seed = seeds[0] if len(seeds) else (ny // 2, nx // 2)

            mask = circular_aperture_mask(ny, nx, seed, radius=args.fixed_radius)
            npix = int(mask.sum())
            lc_raw = np.nansum(flux[:, mask], axis=1)
            lc = median_scale_by_segment(time, lc_raw, gap_days=args.gap_days)

            tag = f"fixedapR{args.fixed_radius:g}"
            out_csv = outdir / f"{base}_preferred_lc_target1_{tag}.csv"
            save_lc_csv(out_csv, time, lc, npix=npix)

            print(f"  Fixed aperture: center={seed}  radius={args.fixed_radius:g}  npix={npix}")

            if args.save_plots:
                out_png = outdir / f"{base}_aperture_{tag}.png"
                mask_plot(out_png, mean_img, [mask], [seed], f"{base} : {tag}")
                out_png_lc = outdir / f"{base}_target1_{tag}.png"
                quicklook_plot(out_png_lc, time, lc, f"{base} : {tag} (seed={seed})")
            continue

        # mode == "apgrow"
        seeds = find_peak_seeds(mean_img, n_peaks=args.n_targets, min_sep=args.min_sep, edge=args.edge)
        if len(seeds) == 0:
            print("  [WARN] No peaks found; falling back to full-stamp")
            lc_raw = np.nansum(flux, axis=(1, 2))
            lc = median_scale_by_segment(time, lc_raw, gap_days=args.gap_days)
            out_csv = outdir / f"{base}_preferred_lc_target1_fullstamp_fallback.csv"
            save_lc_csv(out_csv, time, lc, npix=int(np.isfinite(mean_img).sum()))
            if args.save_plots:
                out_png = outdir / f"{base}_target1_fullstamp_fallback.png"
                quicklook_plot(out_png, time, lc, f"{base} : full-stamp fallback")
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
            save_lc_csv(out_csv, time, lc, npix=npix)

            if args.save_plots:
                out_png = outdir / f"{base}_target{k+1}_apgrow.png"
                quicklook_plot(out_png, time, lc, f"{base} : target{k+1} apgrow (seed={seed})")


if __name__ == "__main__":
    main()
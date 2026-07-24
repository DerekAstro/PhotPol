
#!/usr/bin/env python3
"""
Tkinter front end for:
  - tess_watershed_extractor.py
  - tess_lightcurve_detrend.py

This is intentionally a command-builder + subprocess wrapper rather than a
reimplementation of the extraction / detrending logic.
"""

from __future__ import annotations

import json
import os
import queue
import re
import shlex
import subprocess
import signal
import sys
import threading
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from PIL import Image, ImageTk


APP_TITLE = "TESS Photometry Pipeline GUI"
SETTINGS_SCHEMA_VERSION = 2
BASIC_PRESET_VERSION = 1
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_ORBITAL_TABLE = str(SCRIPT_DIR / "tess_sector_orbfreq_midpoints.csv")
STANDARD_APPROACH = "Standard aperture extraction"
SATURATED_APPROACH = "Saturation-optimized aperture extraction"


def default_tessvectors_cache() -> str:
    """Mirror the detrender's platform-specific cache default."""
    if os.name == "nt":
        local_appdata = os.environ.get("LOCALAPPDATA")
        base = Path(local_appdata) if local_appdata else Path.home() / "AppData" / "Local"
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Caches"
    else:
        xdg_cache = os.environ.get("XDG_CACHE_HOME")
        base = Path(xdg_cache) if xdg_cache else Path.home() / ".cache"
    return str(base / "photpol" / "tessvectors")


DEFAULT_TESSVECTORS_CACHE = default_tessvectors_cache()

# Expert-only values are reset to this documented preset when a user returns
# to Basic mode.  Keeping the preset explicit prevents hidden custom values
# from silently affecting a run after their controls disappear.
EXPERT_DEFAULTS = {
    "extractor_script": str(SCRIPT_DIR / "tess_watershed_extractor.py"),
    "detrender_script": str(SCRIPT_DIR / "tess_lightcurve_detrend.py"),
    "ex_gaia_radius": 6.0,
    "ex_no_gaia": False,
    "ex_gaia_fallback": False,
    "ex_no_quality0": False,
    "ex_full_region_sum": False,
    "ex_gaia_region_sum": False,
    "ex_save_pickled_figures": False,
    "ex_saturated_systematics": False,
    "ex_external_mask_file": "",
    "ex_orbtable": DEFAULT_ORBITAL_TABLE,
    "ex_aperture_fom": "stddiff",
    "ex_simple_aperture_mode": "auto",
    "ex_min_pixels": 10,
    "ex_amp_q_lo": 1.0,
    "ex_amp_q_hi": 99.0,
    "ex_amp_min_frac": 0.01,
    "ex_max_radius_pix": "inf",
    "ex_max_components": 3,
    "ex_min_seed_frac": 0.15,
    "ex_min_new_pixels": 1,
    "ex_core_npix": 12,
    "ex_core_min_frac": 0.25,
    "ex_sat_thresh": 1e5,
    "ex_sat_min_npix": 20,
    "ex_back_nfaint": 20,
    "ex_phase_bin": 0.01,
    "ex_saturated_aperture_threshold": 3000.0,
    "ex_prf_backend": "auto",
    "ex_prf_motion_source": "auto",
    "ex_prf_neighbor_treatment": "fixed",
    "ex_prf_source_output": "primary",
    "ex_prf_neighbor_dmag": 8.0,
    "ex_prf_neighbor_margin": 6.0,
    "ex_prf_max_scene_sources": 20,
    "ex_prf_min_neighbor_fraction": 1e-4,
    "ex_prf_background": "plane",
    "ex_prf_min_weight": 1e-5,
    "ex_prf_fit_radius": 6.0,
    "ex_prf_shift_quantization": 0.01,
    "ex_prf_max_shift": 2.0,
    "ex_prf_save_diagnostics": True,
    "ex_prf_allow_gaussian_fallback": True,
    "dt_diagnostics_dir": "LC_products_multi",
    "dt_pattern": "*.csv",
    "dt_recursive": False,
    "dt_prefix": "detrended_",
    "dt_skip_existing": False,
    "dt_use_pchip": False,
    "dt_skip_xybg": False,
    "dt_knot_spacing": "inf",
    "dt_robust_iters": 8,
    "dt_huber_k": 1.5,
    "dt_pchip_knot_spacing": 0.5,
    "dt_pre_model_bin_days": 0.25,
    "dt_pre_model_stat": "median",
    "dt_pre_model_min_points": 3,
    "dt_pre_model_sigma_clip": 0.0,
    "dt_pre_model_sigma_iters": 1,
    "dt_clip_residuals_before_detrend": False,
    "dt_clip_residuals_sigma": 5.0,
    "dt_clip_residuals_iters": 1,
    "dt_save_pickled_figures": False,
    "dt_apply_orbital_phase_template": False,
    "dt_orbtable": DEFAULT_ORBITAL_TABLE,
    "dt_phase_bin": 0.01,
    "dt_quaternion_source": "",
    "dt_quaternion_auto_download": True,
    "dt_tessvectors_cache_dir": DEFAULT_TESSVECTORS_CACHE,
    "dt_quaternion_camera": "auto",
    "dt_quaternion_min_samples": 3,
    "dt_quaternion_clip_sigma": 3.0,
    "dt_quaternion_clip_iters": 5,
    "dt_quaternion_time_offset": "auto",
    "dt_save_quaternion_diagnostics": True,
}


HELP_TEXT = """
TESS Photometry Pipeline GUI — option reference
===============================================

This GUI is a command-builder and launcher for:
  • tess_watershed_extractor.py
  • tess_lightcurve_detrend.py

It does not reimplement the algorithms. It builds the command line, runs the script,
and shows the log plus output PNG previews.

INTERFACE MODES
---------------
Basic
  Presents the routine workflow and uses Basic preset version 1 for hidden
  numerical controls. Returning from Expert to Basic offers to reset custom
  Expert values so invisible options cannot affect a later run.

Expert
  Exposes script paths, alternative extraction modes, numerical tuning,
  detailed PRF controls, and full detrending/quaternion options.

Settings files save the interface mode, preset version, high-level choices,
and every resolved numerical value. Loading a customized older settings file
opens Expert mode automatically.

GENERAL
-------
Script paths
  Path to the Python script that will be run for the Extractor or Detrender.
  These can live anywhere on disk.

Command preview
  The exact command that will be run. This is useful for debugging and for
  copying into a terminal.

RUN / PREVIEW
-------------
Clear log
  Erases the run log shown in the lower tab.

Save log
  Saves the current log text to a file.

Copy command
  Copies the currently visible command preview to the clipboard.

Refresh previews
  Rescans the preview folder for PNG files.

Preview folder
  Directory whose PNG files are shown in the lower preview pane.

EXTRACTOR TAB
-------------
Input mode
  Directory:
    Use --tpf-dir and optionally --recursive to search for TPF files.
  Single file:
    Use --single to run on just one FITS/TPF file.

TPF directory
  The directory searched by --tpf-dir.

Recursive
  Enables --recursive so subdirectories are searched too.

Single FITS file
  The path passed through --single.

Output root
  Directory passed as --output-root. Products are written here.

Save aperture plots
  If checked, aperture PNGs are written. If unchecked, the GUI adds
  --no-aperture-plots.

N targets
  --n-targets
  Number of targets to extract per stamp, usually the brightest Gaia targets
  inside the image. For TESSCut/Astrocut files, target 1 is instead the Gaia
  source nearest the requested cutout centre; additional targets retain
  brightness order. In no-Gaia modes this should usually be 1.

Method
  --method {jump, core, both}
  jump:
    Multi-component aperture growth.
  core:
    Bright-core preseed then growth.
  both:
    Run both and write both products.

Gaia radius [arcmin]
  --gaia-radius-arcmin
  Cone-search radius used for Gaia target lookup.

No Gaia
  --no-gaia
  Skip Gaia queries and define the target from image pixels instead.
  Normally appropriate only for single-target workflows.

Gaia fallback
  --gaia-fallback
  If Gaia fails, fall back to no-Gaia single-target mode.

Ignore quality flags
  --no-quality0
  Do not restrict to cadences with QUALITY == 0.

Full-region sum
  --full-region-sum
  Use the entire watershed-owned region for the selected target and sum all
  those pixels, instead of optimizing the aperture with jump/core growth.

Saturation-optimized aperture extraction
  --saturation-optimized-aperture
  A single-target approach for heavily saturated sources. It bypasses Gaia
  and jump/core growth and chooses an aperture by high-frequency scatter.

Advanced: Jump / core shared
  Min pixels
    --min-pixels
    Minimum aperture size before growth is allowed to stop just because the
    metric no longer improves.

  Amp q low / Amp q high
    --amp-q-lo, --amp-q-hi
    Percentile range used for the aperture-amplitude sanity check.

  Amp min frac
    --amp-min-frac
    Minimum allowed trial amplitude as a fraction of the full-region amplitude.

  Max radius pix
    --max-radius-pix
    Maximum distance an aperture can grow from its original seed. "inf" means
    no radial restriction.

Advanced: Jump only
  Max components
    --max-components
    Maximum number of separate connected components allowed in jump mode.

  Min seed frac of peak
    --min-seed-frac-of-peak
    Remaining pixels must still be at least this bright relative to the region
    peak to seed a new component.

  Min new pixels / component
    --min-new-pixels-per-component
    A newly found component must add at least this many new pixels.

Advanced: Core only
  Core npix
    --core-npix
    Size of the initial bright core assembled before normal growth.

  Core min frac of peak
    --core-min-frac-of-peak
    Brightness cutoff, relative to peak, for the initial core pixels.

Advanced: Saturated-target processing
  Sat thresh
    --sat-thresh
    Threshold used to decide whether the target is heavily saturated.

  Sat min npix
    --sat-min-npix
    Minimum number of pixels above sat-thresh to count as heavily saturated.

  Back nfaint
    --back-nfaint
    Number of faintest pixels used for per-cadence background estimation.

  Phase bin
    --phase-bin
    Phase bin width used in the orbital-phase template step.

  Initial saturated-aperture threshold
    --saturated-aperture-threshold
    Initial mean-image threshold for saturation-optimized aperture growth.

  Orbital table CSV
    --orbtable
    The bundled tess_sector_orbfreq_midpoints.csv is selected by default.
    Its preferred headings are sector, mid_btjd, and freq_cyc_per_day. Browse
    to a replacement only when using a separately maintained table.

Advanced: Fixed external aperture
  External mask file
    --external-mask-file
    Single-target text matrix that completely defines the aperture and
    overrides jump/core. Use one local image row per line and comma- or
    whitespace-separated 1/0 or Y/N values. The first data row is local row 0,
    the first value is column 0, and the matrix must exactly match the TPF
    image shape. Blank lines and text after # are ignored.

JITTER-AWARE PRF PHOTOMETRY
---------------------------
Enable jitter-aware PRF photometry
  Runs an additional PRF-weighted extraction alongside the selected aperture
  method. It does not replace jump/core/full-stamp products. The shared module
  tess_prf_photometry.py must be in the same directory or on PYTHONPATH.

PRF backend
  auto      : prefer lkprf, then TESS_PRF, with a warned Gaussian fallback.
  lkprf     : use the official engineering PRF through the lkprf package.
  tess_prf  : use the TESS_PRF package.
  gaussian  : use an approximate Gaussian model, mainly for testing/fallback.
  Install the preferred backend in the active environment with
  `pip install lkprf` (or `pip install TESS_PRF`). The first official-PRF use
  may download and cache camera/CCD calibration files.

Motion source
  auto      : usable, varying POS_CORR1/2, then ensemble image centroid, target
              centroid, fixed. Constant TESSCut placeholders are rejected.
  poscorr   : require usable, varying TPF POS_CORR1/2 values.
  ensemble  : derive common motion from bright pixels in the stamp.
  target    : derive motion from a target-centered pixel region.
  fixed     : use a fixed PRF for all cadences.

Background model
  none, constant, or plane. Plane is the recommended default because it can
  absorb a smooth background gradient without forcing it into the source flux.

Static PRF registration
  The nominal WCS/seed position is used only as a starting point. The native
  engineering PRF is fitted directly to the mean image to determine the static
  subpixel position. lkprf output is not flipped or transposed.

Motion calibration and quality check
  When POS_CORR is available, a robust full 2x2 linear mapping calibrates it to
  the image-derived row/column motion. After extraction, residual flux-motion
  coupling is measured. Strong coupling flags the PRF product while preserving
  it for diagnosis; the aperture product remains the preferred result.

Saturation safeguard
  Reliable target TESS magnitude, target-local vertical bleed morphology,
  unit-aware bright-pixel counts, and core shape are combined. TESSCut's
  placeholder TESSMAG is ignored, and unrelated sources elsewhere in a large
  cutout cannot veto the requested target. Saturated or bleed-dominated targets
  bypass ordinary PRF fitting because the engineering PRF does not model charge
  bleeding.

PRF grid step
  Requested spacing of the regular row/column motion grid on which the
  engineering PRF is precomputed. Cadence PRFs are obtained by bilinear
  interpolation. The implementation automatically limits each grid axis to
  21 nodes, so runtime does not scale with the number of distinct cadences.

Scene mode
  single : fit only the selected target.
  gaia   : build a target-first Gaia scene and jointly fit the reference image.
           At each cadence, one active source is fit while all other retained
           sources are held at their fitted reference-scene fluxes.

PRF light curves
  primary : write only the selected primary target (the default).
  all     : repeat the constrained cadence fit for every usable retained scene
            source. Sources with too little captured PRF flux, non-positive
            reference flux, or nearly degenerate PRF columns are skipped.

Gaia-neighbor controls
  Neighbor dG limits the initial scene by Gaia G magnitude difference. Neighbor
  margin includes source centers just outside the stamp. Max scene sources caps
  computation. Min neighbor fraction removes fitted neighbors whose predicted
  contribution inside the stamp is negligible relative to the target.

The multi-source output includes a scene CSV, reference-scene conditioning and
PRF-column correlations, predicted contamination, source-specific diagnostics,
and the usual motion-coupling quality checks. In all-source mode, each source is
fit separately with every other source held fixed; the code does not perform an
unstable unconstrained cadence-by-cadence fit of all source fluxes at once.
Saturated targets are skipped because the engineering PRF does not model charge
bleeding.

DETRENDER TAB
-------------
Lightcurve dir
  --lightcurve-dir
  Directory containing input CSV light curves.

Diagnostics dir
  --diagnostics-dir
  Directory containing centroid/background .npy files and related diagnostics.

Output dir
  --output-dir
  Directory for detrended CSVs and plots.

Pattern
  --pattern
  Filename pattern used to choose input CSVs.

Prefix
  --prefix
  Prefix added to output filenames.

Skip existing outputs
  --skip-existing
  If checked, the detrender skips an input light curve when the filename-based
  detrended output CSV already exists. Existing combined CSV outputs are also
  left untouched.

Recursive
  --recursive
  Search input CSVs recursively under the lightcurve directory.

Use background
  --use-background
  Add the saved background series as a regression term when decorrelating.

Use PCHIP high-pass
  --use-pchip-highpass
  After leveling/decorrelation, apply an extra smooth PCHIP trend removal.

Skip XY/background decorrelation
  --skip-xybg-decorrelation
  Skip the centroid/background regression entirely, but still do the chunk
  leveling and optional PCHIP high-pass / sector combining.

Combine sectors
  Inverse of --no-combine-sectors.
  If checked, write combined multi-sector CSV and PNG products.

Knot spacing days
  --knot-spacing-days
  Spacing for the optional time-basis hinge functions used in decorrelation.
  "inf" disables those terms.

Robust iters
  --robust-iters
  Number of robust weighted least-squares iterations.

Huber k
  --huber-k
  Huber tuning constant controlling how aggressively outliers are downweighted.

PCHIP knot spacing
  --pchip-knot-spacing
  Time spacing used when building the smooth PCHIP trend.

QLP-style quaternion regression
  --use-quaternion-regression
  Adds short-timescale spacecraft-pointing regressors built from camera
  quaternion engineering data. For raw 2-second quaternion files, the detrender
  bins the three quaternion axes to each light-curve cadence, computes their
  mean, standard deviation, and skew, forms the QLP pairwise products and
  squares, standardizes the resulting 36 features, and fits them with iterative
  3-sigma clipping.

Quaternion source
  --quaternion-source
  Optional local raw-quaternion file, TESSVectors CSV, or directory containing
  sector-specific files. An explicitly selected source takes priority over the
  automatic cache. Raw files provide the full 36 QLP-style regressors;
  pre-binned TESSVectors files use the statistics available in those products.

Automatic TESSVectors download / cache
  --quaternion-auto-download, --tessvectors-cache-dir
  When enabled and no explicit Quaternion source is supplied, the detrender
  infers the sector, camera, and cadence, downloads only the required HEASARC
  TESSVectors CSV if it is missing, and keeps it in the selected cache directory.
  Existing cached files are reused without downloading again. Camera must be
  selected explicitly if it cannot be inferred from the light-curve metadata.

Quaternion camera
  --quaternion-camera
  Select camera 1-4, or auto. Auto uses a camera column in the light-curve file
  or source table when possible. Choose the camera explicitly if the source
  contains multiple cameras and it cannot be inferred.

Quaternion time offset
  --quaternion-time-offset-days
  Use "auto" to align the span centers of barycentric light-curve time and
  spacecraft engineering time. If the light-curve CSV contains a timecorr
  column, that is used instead and is preferred. A numerical value supplies
  the barycentric-minus-spacecraft offset in days.

Quaternion min samples
  --quaternion-min-samples
  Minimum number of raw quaternion samples required inside a light-curve
  cadence before its mean/std/skew features are accepted.

Quaternion clip sigma / iterations
  --quaternion-clip-sigma, --quaternion-clip-iters
  Control the iterative outlier rejection in the QLP-style linear fit.

Save quaternion diagnostics
  --save-quaternion-diagnostics
  Writes an NPZ file containing features, coefficients, coverage, and alignment
  information, plus a PNG showing the fitted quaternion model and corrected
  light curve.

  For variability preservation, the GUI selects Pre-model PCHIP by default
  when quaternion regression is first turned on. The PCHIP model is removed
  before the quaternion fit and restored afterward. You may uncheck Pre-model
  PCHIP to fit the quaternion regressors directly to the original light curve.

Notes
-----
• The command preview is the authoritative description of what the GUI will do.
• If a script changes, the GUI may need to be updated to stay in sync.
• For debugging, always compare the GUI settings with the command preview and
  the run log.
"""


TOOLTIPS = {
    "ex_input_mode": "Choose whether to search a directory of TPF files or run on one specific FITS/TPF file.",
    "ex_tpf_dir": "Directory passed to --tpf-dir. The extractor searches for TESS, Kepler, K2, and TESSCut target-pixel files.",
    "ex_single_file": "Single TESS, Kepler, K2, or TESSCut target-pixel FITS file passed via --single.",
    "ex_recursive": "Search subdirectories under the chosen TPF directory.",
    "ex_output_root": "Directory passed as --output-root. Extracted CSVs, NPYs, and PNGs are written here.",
    "ex_n_targets": "Number of targets to extract per stamp. TESSCut target 1 is nearest the requested cutout centre; otherwise targets are ordered by Gaia brightness.",
    "ex_method": "Aperture-growth method: jump, core, or both.",
    "ex_gaia_radius": "Cone-search radius used for Gaia target lookup.",
    "ex_no_gaia": "Skip Gaia queries and define the target from image pixels instead. Best for single-target runs.",
    "ex_gaia_fallback": "If Gaia fails, fall back to no-Gaia single-target mode.",
    "ex_no_quality0": "Do not restrict to QUALITY==0 cadences.",
    "ex_full_region_sum": "Use the full allowed target region without optimizing a smaller aperture.",
"ex_gaia_region_sum": "For Gaia-defined targets, sum the full Gaia-owned watershed region for each target without aperture growth.",
"ex_external_mask_file": "Fixed aperture for a single target. Use one local image row per text line with comma- or whitespace-separated 1/0 or Y/N values. The shape must exactly match the TPF; row 0 and column 0 come first.",
"ex_aperture_fom": "Figure of merit used during watershed aperture growth.",
    "ex_saturated_systematics": "Apply background, TESS orbital-phase, and split-sector corrections for a heavily saturated target.",
    "ex_orbtable": "TESS sector midpoint/orbital-frequency CSV. The bundled table is selected by default; browse here only to use a replacement table.",
"ex_save_pickled_figures": "Also save pickled Matplotlib figures alongside PNGs.",
"ex_simple_aperture_mode": "Aperture mode used when running the simple extractor.",
    "ex_prf_photometry": "Write an additional cadence-dependent TESS PRF-weighted light curve. Kepler/K2 files are detected and skipped with a warning; aperture products are still generated.",
    "ex_prf_backend": "TESS-only PRF backend: auto prefers lkprf, then TESS_PRF, with a warned Gaussian fallback.",
    "ex_prf_motion_source": "Cadence motion source. Auto rejects constant TESSCut POS_CORR placeholders, then falls back through ensemble centroid, target centroid, and fixed position.",
    "ex_prf_scene_mode": "Single fits only the target. Gaia jointly fits a target-first reference scene and subtracts fixed neighbor models during cadence extraction.",
    "ex_prf_neighbor_treatment": "How non-active Gaia scene sources behave over time. Fixed uses their jointly fitted reference-scene fluxes.",
    "ex_prf_source_output": "Primary writes only the selected target and is the default. All writes one constrained light curve for every usable retained Gaia scene source.",
    "ex_prf_neighbor_dmag": "Initial Gaia scene includes neighbors no more than this many G magnitudes fainter than the target.",
    "ex_prf_neighbor_margin": "Include source centers this many pixels outside the stamp when their PRF wings may enter it.",
    "ex_prf_max_scene_sources": "Maximum target-plus-neighbor sources evaluated in the initial Gaia reference scene.",
    "ex_prf_min_neighbor_fraction": "After reference fitting, drop neighbors contributing less than this fraction of the target flux inside the stamp.",
    "ex_prf_background": "Background terms fitted simultaneously with the PRF flux. Plane is the recommended default and handles smooth stamp gradients.",
    "ex_prf_min_weight": "Minimum relative PRF weight used when constructing the source fitting region.",
    "ex_prf_fit_radius": "Radius around each source included in the PRF scene fit.",
    "ex_prf_shift_quantization": "Requested spacing of the precomputed regular PRF motion grid. Cadence PRFs are bilinearly interpolated; each axis is automatically capped at 21 nodes.",
    "ex_prf_max_shift": "Maximum absolute row/column shift retained after robust motion cleaning.",
    "ex_prf_save_diagnostics": "Save a PRF/image overlay, motion curves, extracted light curve, residual image, metadata JSON, and motion NPZ.",
    "ex_prf_allow_gaussian_fallback": "Allow a warned Gaussian approximation if neither lkprf nor TESS_PRF is installed or usable.",
    "ex_save_aperture_plots": "Save aperture-overlay PNGs. If unchecked, --no-aperture-plots is added.",
    "ex_extraction_approach": "Choose standard watershed aperture extraction or the single-target saturation-optimized aperture approach.",
    "ex_min_pixels": "Minimum aperture size before growth is allowed to stop just because the metric no longer improves.",
    "ex_amp_q_lo": "Lower percentile used in the aperture-amplitude sanity check.",
    "ex_amp_q_hi": "Upper percentile used in the aperture-amplitude sanity check.",
    "ex_amp_min_frac": "Minimum allowed trial amplitude as a fraction of the full-region amplitude.",
    "ex_max_radius_pix": "Maximum distance an aperture may grow from its original seed pixel. Use inf for no radius limit.",
    "ex_max_components": "Maximum number of separate connected components allowed in jump mode.",
    "ex_min_seed_frac": "A new jump component must be at least this bright relative to the region peak.",
    "ex_min_new_pixels": "A newly found jump component must add at least this many new pixels.",
    "ex_core_npix": "Size of the initial bright core assembled before normal growth in core mode.",
    "ex_core_min_frac": "Brightness cutoff, relative to the regional peak, for pixels eligible for the initial core.",
    "ex_sat_thresh": "Threshold used to decide whether the target is heavily saturated.",
    "ex_sat_min_npix": "Minimum number of pixels above sat-thresh to count as heavily saturated.",
    "ex_back_nfaint": "Number of faintest pixels used for per-cadence background estimation.",
    "ex_phase_bin": "Phase-bin width used in the orbital-phase template step.",
    "ex_saturated_aperture_threshold": "Initial mean-image threshold for the saturation-optimized aperture seed.",
    "ex_command_preview": "The exact extractor command the GUI will run.",
    "dt_lightcurve_dir": "Directory containing input CSV light curves.",
    "dt_diagnostics_dir": "Directory containing centroid/background NPY files and related diagnostics.",
    "dt_output_dir": "Directory for detrended CSVs and PNGs.",
    "dt_pattern": "Filename pattern used to choose input CSV light curves.",
    "dt_prefix": "Prefix added to detrended output filenames.",
    "dt_recursive": "Search input CSV files recursively under the lightcurve directory.",
    "dt_skip_existing": "Skip an input light curve if the filename-based detrended output CSV already exists. Existing combined CSV outputs are also left untouched.",
    "dt_use_background": "Include the saved background series as a regression term when decorrelating.",
    "dt_use_pchip": "Apply a smooth PCHIP high-pass step after position/background decorrelation.",
    "dt_skip_xybg": "Skip centroid/background regression while retaining optional orbital, PCHIP, and sector-combination steps.",
    "dt_combine_sectors": "Write combined multi-sector CSV and PNG products.",
    "dt_knot_spacing": "Spacing for optional time-basis hinge functions used in decorrelation. Use inf to disable them.",
    "dt_robust_iters": "Number of robust weighted least-squares iterations.",
    "dt_huber_k": "Huber tuning constant controlling how strongly outliers are downweighted.",
    "dt_pchip_knot_spacing": "Time spacing used when building the smooth PCHIP trend.",
    "dt_pre_model_pchip": "Before position-based detrending, fit a binned PCHIP spline to the light curve, subtract it, do the decorrelation on the residuals, then add the spline back afterward.",
    "dt_pre_model_bin_days": "Time-bin size in days used to build the optional pre-model PCHIP variability spline.",
    "dt_pre_model_stat": "Statistic used within each time bin for the pre-model PCHIP fit.",
    "dt_pre_model_min_points": "Minimum number of cadences required in a bin before that bin is used in the pre-model PCHIP fit.",
    "dt_pre_model_sigma_clip": "Optional sigma-clipping threshold applied within each pre-model time bin before computing the binned statistic. Set to 0 to disable.",
    "dt_pre_model_sigma_iters": "Number of sigma-clipping iterations used inside the pre-model time bins.",
    "dt_clip_residuals_before_detrend": "Optionally sigma-clip the residual light curve before fitting the centroid/background decorrelation model.",
    "dt_clip_residuals_sigma": "Sigma threshold used for clipping residuals before detrending.",
    "dt_clip_residuals_iters": "Number of sigma-clipping iterations used on the residuals before detrending.",
    "dt_save_pickled_figures": "Save pickled Matplotlib figure objects alongside PNGs so the plots can be reopened later for zooming and inspection.",
    "dt_apply_orbital_phase_template": "Apply a TESS orbital-phase template correction using the sector orbital-frequency table.",
    "dt_orbtable": "TESS sector midpoint/orbital-frequency CSV used during detrending. The bundled table is selected by default.",
    "dt_phase_bin": "Phase bin width used when building the orbital phase-template correction.",
    "dt_use_quaternion_regression": "Add QLP-style camera-quaternion regressors to remove short-timescale pointing systematics. Pre-model PCHIP is selected by default when this is first enabled, but may then be unchecked.",
    "dt_quaternion_source": "Optional explicit raw quaternion FITS/CSV, TESSVectors CSV, or directory. Leave blank to use automatic TESSVectors downloading.",
    "dt_quaternion_auto_download": "Automatically download only the missing sector/camera/cadence TESSVectors file and reuse it from the local cache on later runs.",
    "dt_tessvectors_cache_dir": "Local cache root for automatically downloaded TESSVectors files. Files are organized by 020_Cadence, 120_Cadence, and FFI_Cadence.",
    "dt_quaternion_camera": "TESS camera used for quaternion regressors. Auto infers it from metadata when possible; select 1-4 when automatic downloading cannot infer it.",
    "dt_quaternion_min_samples": "Minimum number of raw 2-second quaternion samples required within each light-curve cadence.",
    "dt_quaternion_clip_sigma": "Sigma threshold for iterative QLP-style outlier rejection during the quaternion regression.",
    "dt_quaternion_clip_iters": "Maximum number of QLP-style sigma-clipping iterations.",
    "dt_quaternion_time_offset": "Barycentric-minus-spacecraft time offset in days, or auto. A timecorr column in the light-curve CSV is preferred when available.",
    "dt_save_quaternion_diagnostics": "Save an NPZ with quaternion features/coefficients/coverage and a diagnostic PNG.",
    "dt_command_preview": "The exact detrender command the GUI will run.",
    "preview_dir": "Directory scanned for PNG previews.",
    "preview_scale_mode": "Choose how preview images are displayed: Fit scales to the visible preview pane; percentage modes use a fixed zoom level.",
    "preview_zoom": "Manual preview zoom level. Also adjustable with Ctrl + mouse wheel.",
}




def quote_cmd(cmd: list[str]) -> str:
    # list2cmdline follows Windows CreateProcess quoting rules; shlex.join is
    # appropriate for POSIX shells. Execution itself always uses the argument
    # list directly and therefore never depends on this display string.
    if os.name == "nt":
        return subprocess.list2cmdline([str(x) for x in cmd])
    return shlex.join([str(x) for x in cmd])


class ToolTip:
    def __init__(self, widget, text: str, delay_ms: int = 500, wraplength: int = 360):
        self.widget = widget
        self.text = text
        self.delay_ms = delay_ms
        self.wraplength = wraplength
        self.tipwindow = None
        self._after_id = None
        widget.bind("<Enter>", self._schedule, add="+")
        widget.bind("<Leave>", self._hide, add="+")
        widget.bind("<ButtonPress>", self._hide, add="+")

    def _schedule(self, event=None):
        self._cancel()
        self._after_id = self.widget.after(self.delay_ms, self._show)

    def _cancel(self):
        if self._after_id is not None:
            try:
                self.widget.after_cancel(self._after_id)
            except Exception:
                pass
            self._after_id = None

    def _show(self):
        if self.tipwindow is not None or not self.text:
            return
        try:
            x, y, _, h = self.widget.bbox("insert")
        except Exception:
            x, y, h = 0, 0, 0
        x = x + self.widget.winfo_rootx() + 18
        y = y + self.widget.winfo_rooty() + h + 18
        tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        label = tk.Label(
            tw,
            text=self.text,
            justify="left",
            relief="solid",
            borderwidth=1,
            background="#fff8dc",
            padx=6,
            pady=4,
            wraplength=self.wraplength,
        )
        label.pack()
        self.tipwindow = tw

    def _hide(self, event=None):
        self._cancel()
        tw = self.tipwindow
        self.tipwindow = None
        if tw is not None:
            try:
                tw.destroy()
            except Exception:
                pass
class ScrollableFrame(ttk.Frame):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self.canvas = tk.Canvas(self, highlightthickness=0)
        self.vsb = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.hsb = ttk.Scrollbar(self, orient="horizontal", command=self.canvas.xview)
        self.inner = ttk.Frame(self.canvas)

        self.inner.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )

        self.canvas.create_window((0, 0), window=self.inner, anchor="nw")
        self.canvas.configure(yscrollcommand=self.vsb.set, xscrollcommand=self.hsb.set)

        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.vsb.grid(row=0, column=1, sticky="ns")
        self.hsb.grid(row=1, column=0, sticky="ew")

        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)

        # Bind locally so the Extractor, Detrender, and preview canvases do not
        # steal one another's wheel events. Button-4/5 covers X11/Linux; the
        # MouseWheel event covers Windows and macOS.
        for widget in (self.canvas, self.inner):
            widget.bind("<MouseWheel>", self._on_mousewheel)
            widget.bind("<Button-4>", self._on_mousewheel)
            widget.bind("<Button-5>", self._on_mousewheel)

    def _on_mousewheel(self, event):
        try:
            if getattr(event, "num", None) == 4:
                steps = -1
            elif getattr(event, "num", None) == 5:
                steps = 1
            else:
                delta = int(getattr(event, "delta", 0))
                steps = -1 if delta > 0 else 1
            self.canvas.yview_scroll(steps, "units")
        except Exception:
            pass


class TESSGui(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(APP_TITLE)
        self._configure_platform_appearance()

        # Queue items carry the originating process so a delayed completion
        # event can never clear the state of a newer job.
        self.log_queue: queue.Queue[tuple[str, object, str]] = queue.Queue()
        self.current_process: subprocess.Popen | None = None
        self.current_job_name: str | None = None
        self.current_output_dir: Path | None = None
        self.preview_image = None

        self._build_vars()
        self._last_interface_mode = "basic"
        self._quaternion_was_enabled = False
        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self._poll_log_queue()
        self._update_extractor_state()
        self._update_detrender_state()
        self._apply_interface_mode()
        self._refresh_preview_list()

    def _configure_platform_appearance(self):
        """Choose a native-looking theme and a screen-safe initial size."""
        style = ttk.Style(self)
        available = set(style.theme_names())
        preferred = (
            ["vista", "xpnative", "clam"] if os.name == "nt" else
            ["aqua", "clam"] if sys.platform == "darwin" else
            ["clam", "alt", "default"]
        )
        for theme in preferred:
            if theme in available:
                try:
                    style.theme_use(theme)
                    break
                except tk.TclError:
                    continue

        screen_w = max(900, int(self.winfo_screenwidth()))
        screen_h = max(700, int(self.winfo_screenheight()))
        width = min(1220, int(screen_w * 0.94))
        height = min(860, int(screen_h * 0.90))
        self.geometry(f"{width}x{height}")
        self.minsize(min(860, width), min(640, height))

    @staticmethod
    def _settings_values_equal(left, right) -> bool:
        """Compare Tk setting values without treating 1 and True as different."""
        if isinstance(right, bool):
            return bool(left) is right
        if isinstance(right, float):
            try:
                return abs(float(left) - right) <= 1e-12 * max(1.0, abs(right))
            except Exception:
                return False
        return left == right

    @classmethod
    def _migrate_settings_data(cls, data: dict) -> tuple[dict, str]:
        """Translate older setting keys and infer their interface mode."""
        migrated = dict(data)
        aliases = {
            "ex_pure_sum": "ex_full_region_sum",
            "ex_matlab_sat_mode": "ex_saturated_systematics",
            "ex_matlab_ap_thresh": "ex_saturated_aperture_threshold",
        }
        for old, new in aliases.items():
            if new not in migrated and old in migrated:
                migrated[new] = migrated[old]

        # Older releases saved the bundled orbital table as a bare relative
        # filename. Promote only that known default to the new script-relative
        # absolute path; user-selected replacement paths remain untouched.
        for key in ("ex_orbtable", "dt_orbtable"):
            if str(migrated.get(key, "")).strip() == "tess_sector_orbfreq_midpoints.csv":
                migrated[key] = DEFAULT_ORBITAL_TABLE
        if "ex_extraction_approach" not in migrated:
            old_saturated = bool(migrated.get("ex_matlab_pure_single_sat", False))
            migrated["ex_extraction_approach"] = (
                SATURATED_APPROACH if old_saturated else STANDARD_APPROACH
            )
        elif migrated["ex_extraction_approach"] in {"standard", "saturation_optimized"}:
            migrated["ex_extraction_approach"] = (
                SATURATED_APPROACH
                if migrated["ex_extraction_approach"] == "saturation_optimized"
                else STANDARD_APPROACH
            )

        requested_mode = str(migrated.get("interface_mode", "")).strip().lower()
        if requested_mode not in {"basic", "expert"}:
            requested_mode = "basic"
            for name, default in EXPERT_DEFAULTS.items():
                if name in migrated and not cls._settings_values_equal(migrated[name], default):
                    requested_mode = "expert"
                    break
        return migrated, requested_mode

    def _expert_settings_are_custom(self) -> bool:
        for name, default in EXPERT_DEFAULTS.items():
            var = getattr(self, name, None)
            if var is not None and not self._settings_values_equal(var.get(), default):
                return True
        return False

    def _reset_expert_defaults(self):
        """Apply the versioned Basic preset to every hidden Expert control."""
        for name, default in EXPERT_DEFAULTS.items():
            var = getattr(self, name, None)
            if var is not None:
                var.set(default)

    def _on_interface_mode_change(self):
        requested = self.interface_mode.get().strip().lower()
        if requested == "basic" and self._last_interface_mode == "expert":
            if self._expert_settings_are_custom():
                proceed = messagebox.askyesno(
                    "Return to Basic mode?",
                    "Basic mode uses its documented preset. Custom Expert values "
                    "will be reset so hidden settings cannot affect future runs.\n\n"
                    "Continue?",
                )
                if not proceed:
                    self.interface_mode.set("expert")
                    return
            self._reset_expert_defaults()
        elif requested == "expert" and self._last_interface_mode == "basic":
            # Seed the newly visible diagnostics path with the value Basic mode
            # actually uses, rather than an unrelated hidden default.
            self.dt_diagnostics_dir.set(self.dt_lightcurve_dir.get())

        self._last_interface_mode = requested
        self._apply_interface_mode()
        self._update_extractor_state()
        self._update_detrender_state()

    def _apply_interface_mode(self):
        """Show only controls appropriate to the selected interface mode."""
        expert = self.interface_mode.get().strip().lower() == "expert"
        expert_frames = [
            getattr(self, "ex_script_paths_frame", None),
            getattr(self, "ex_expert_main_frame", None),
            getattr(self, "ex_advanced_frame", None),
            getattr(self, "ex_prf_expert_frame", None),
            getattr(self, "dt_script_paths_frame", None),
            getattr(self, "dt_expert_dirs_frame", None),
            getattr(self, "dt_expert_selection_frame", None),
            getattr(self, "dt_expert_options_frame", None),
            getattr(self, "dt_expert_quaternion_frame", None),
        ]
        basic_only_frames = [
            getattr(self, "ex_prf_basic_frame", None),
            getattr(self, "dt_basic_frame", None),
        ]
        for frame in expert_frames:
            if frame is not None:
                frame.grid() if expert else frame.grid_remove()
        for frame in basic_only_frames:
            if frame is not None:
                frame.grid_remove() if expert else frame.grid()

    def _resolved_diagnostics_dir(self) -> str:
        """Return the visible/resolved diagnostics path for the current mode."""
        if self.interface_mode.get().strip().lower() == "basic":
            return self.dt_lightcurve_dir.get().strip() or "LC_products_multi"
        return self.dt_diagnostics_dir.get().strip() or "LC_products_multi"

    def _on_close(self):
        """Stop a running process tree before closing the GUI."""
        self._reap_stale_process_if_needed()
        proc = self.current_process
        if proc is not None:
            if not messagebox.askyesno("Quit", "A job is still running. Stop it and quit?"):
                return
            self._terminate_process_tree(proc, force=True)
        self.destroy()

    def _build_vars(self):
        self.interface_mode = tk.StringVar(value="basic")
        self.extractor_script = tk.StringVar(value=str(SCRIPT_DIR / "tess_watershed_extractor.py"))
        self.detrender_script = tk.StringVar(value=str(SCRIPT_DIR / "tess_lightcurve_detrend.py"))

        self.ex_input_mode = tk.StringVar(value="directory")
        self.ex_tpf_dir = tk.StringVar(value=".")
        self.ex_single_file = tk.StringVar(value="")
        self.ex_recursive = tk.BooleanVar(value=False)
        self.ex_output_root = tk.StringVar(value="LC_products_multi")

        self.ex_n_targets = tk.IntVar(value=1)
        self.ex_method = tk.StringVar(value="jump")
        self.ex_gaia_radius = tk.DoubleVar(value=6.0)

        self.ex_no_gaia = tk.BooleanVar(value=False)
        self.ex_gaia_fallback = tk.BooleanVar(value=False)
        self.ex_no_quality0 = tk.BooleanVar(value=False)
        self.ex_full_region_sum = tk.BooleanVar(value=False)
        self.ex_gaia_region_sum = tk.BooleanVar(value=False)
        self.ex_save_aperture_plots = tk.BooleanVar(value=True)
        self.ex_save_pickled_figures = tk.BooleanVar(value=False)
        self.ex_saturated_systematics = tk.BooleanVar(value=False)
        self.ex_extraction_approach = tk.StringVar(value=STANDARD_APPROACH)
        self.ex_external_mask_file = tk.StringVar(value="")
        self.ex_orbtable = tk.StringVar(value=DEFAULT_ORBITAL_TABLE)
        self.ex_aperture_fom = tk.StringVar(value="stddiff")
        self.ex_simple_aperture_mode = tk.StringVar(value="auto")

        # Optional additional jitter-aware PRF extraction.
        self.ex_prf_photometry = tk.BooleanVar(value=False)
        self.ex_prf_backend = tk.StringVar(value="auto")
        self.ex_prf_motion_source = tk.StringVar(value="auto")
        self.ex_prf_scene_mode = tk.StringVar(value="single")
        self.ex_prf_neighbor_treatment = tk.StringVar(value="fixed")
        self.ex_prf_source_output = tk.StringVar(value="primary")
        self.ex_prf_neighbor_dmag = tk.DoubleVar(value=8.0)
        self.ex_prf_neighbor_margin = tk.DoubleVar(value=6.0)
        self.ex_prf_max_scene_sources = tk.IntVar(value=20)
        self.ex_prf_min_neighbor_fraction = tk.DoubleVar(value=1e-4)
        self.ex_prf_background = tk.StringVar(value="plane")
        self.ex_prf_min_weight = tk.DoubleVar(value=1e-5)
        self.ex_prf_fit_radius = tk.DoubleVar(value=6.0)
        self.ex_prf_shift_quantization = tk.DoubleVar(value=0.01)
        self.ex_prf_max_shift = tk.DoubleVar(value=2.0)
        self.ex_prf_save_diagnostics = tk.BooleanVar(value=True)
        self.ex_prf_allow_gaussian_fallback = tk.BooleanVar(value=True)

        self.ex_min_pixels = tk.IntVar(value=10)
        self.ex_amp_q_lo = tk.DoubleVar(value=1.0)
        self.ex_amp_q_hi = tk.DoubleVar(value=99.0)
        self.ex_amp_min_frac = tk.DoubleVar(value=0.01)
        self.ex_max_radius_pix = tk.StringVar(value="inf")

        self.ex_max_components = tk.IntVar(value=3)
        self.ex_min_seed_frac = tk.DoubleVar(value=0.15)
        self.ex_min_new_pixels = tk.IntVar(value=1)

        self.ex_core_npix = tk.IntVar(value=12)
        self.ex_core_min_frac = tk.DoubleVar(value=0.25)

        self.ex_sat_thresh = tk.DoubleVar(value=1e5)
        self.ex_sat_min_npix = tk.IntVar(value=20)
        self.ex_back_nfaint = tk.IntVar(value=20)
        self.ex_phase_bin = tk.DoubleVar(value=0.01)
        self.ex_saturated_aperture_threshold = tk.DoubleVar(value=3000.0)

        self.ex_command_preview = tk.StringVar(value="")

        self.dt_lightcurve_dir = tk.StringVar(value="LC_products_multi")
        self.dt_diagnostics_dir = tk.StringVar(value="LC_products_multi")
        self.dt_output_dir = tk.StringVar(value="LC_products_multi")
        self.dt_pattern = tk.StringVar(value="*.csv")
        self.dt_recursive = tk.BooleanVar(value=False)
        self.dt_prefix = tk.StringVar(value="detrended_")
        self.dt_skip_existing = tk.BooleanVar(value=False)

        self.dt_use_background = tk.BooleanVar(value=False)
        self.dt_use_pchip = tk.BooleanVar(value=False)
        self.dt_skip_xybg = tk.BooleanVar(value=False)
        self.dt_combine_sectors = tk.BooleanVar(value=True)

        self.dt_knot_spacing = tk.StringVar(value="inf")
        self.dt_robust_iters = tk.IntVar(value=8)
        self.dt_huber_k = tk.DoubleVar(value=1.5)
        self.dt_pchip_knot_spacing = tk.DoubleVar(value=0.5)
        self.dt_pre_model_pchip = tk.BooleanVar(value=False)
        self.dt_pre_model_bin_days = tk.DoubleVar(value=0.25)
        self.dt_pre_model_stat = tk.StringVar(value="median")
        self.dt_pre_model_min_points = tk.IntVar(value=3)
        self.dt_pre_model_sigma_clip = tk.DoubleVar(value=0.0)
        self.dt_pre_model_sigma_iters = tk.IntVar(value=1)
        self.dt_clip_residuals_before_detrend = tk.BooleanVar(value=False)
        self.dt_clip_residuals_sigma = tk.DoubleVar(value=5.0)
        self.dt_clip_residuals_iters = tk.IntVar(value=1)
        self.dt_save_pickled_figures = tk.BooleanVar(value=False)
        self.dt_apply_orbital_phase_template = tk.BooleanVar(value=False)
        self.dt_orbtable = tk.StringVar(value=DEFAULT_ORBITAL_TABLE)
        self.dt_phase_bin = tk.DoubleVar(value=0.01)

        self.dt_use_quaternion_regression = tk.BooleanVar(value=False)
        self.dt_quaternion_source = tk.StringVar(value="")
        self.dt_quaternion_auto_download = tk.BooleanVar(value=True)
        self.dt_tessvectors_cache_dir = tk.StringVar(value=DEFAULT_TESSVECTORS_CACHE)
        self.dt_quaternion_camera = tk.StringVar(value="auto")
        self.dt_quaternion_min_samples = tk.IntVar(value=3)
        self.dt_quaternion_clip_sigma = tk.DoubleVar(value=3.0)
        self.dt_quaternion_clip_iters = tk.IntVar(value=5)
        self.dt_quaternion_time_offset = tk.StringVar(value="auto")
        self.dt_save_quaternion_diagnostics = tk.BooleanVar(value=True)


        self.dt_command_preview = tk.StringVar(value="")

        self.preview_dir = tk.StringVar(value="")
        self.preview_scale_mode = tk.StringVar(value="Fit")
        self.preview_zoom = tk.IntVar(value=100)
        self.preview_source_image = None
        self.preview_source_path = None
        self.status_text = tk.StringVar(value="Ready.")


    def _tooltip(self, widget, key: str):
        txt = TOOLTIPS.get(key, "")
        if txt:
            ToolTip(widget, txt)
        return widget

    def _make_checkbutton(self, parent, text, variable, command=None, tooltip_key=None, **grid_kwargs):
        w = ttk.Checkbutton(parent, text=text, variable=variable, command=command)
        if tooltip_key:
            self._tooltip(w, tooltip_key)
        if grid_kwargs:
            w.grid(**grid_kwargs)
        return w


    def _preview_scale_factor(self):
        mode = self.preview_scale_mode.get().strip()
        if mode.lower() == "fit":
            return None
        m = re.match(r"^(\d+)\s*%$", mode)
        if m:
            try:
                return max(0.05, int(m.group(1)) / 100.0)
            except Exception:
                return 1.0
        try:
            return max(0.05, float(self.preview_zoom.get()) / 100.0)
        except Exception:
            return 1.0

    def _on_preview_scale_mode_change(self, event=None):
        mode = self.preview_scale_mode.get().strip()
        if mode.lower() != "fit":
            m = re.match(r"^(\d+)\s*%$", mode)
            if m:
                try:
                    self.preview_zoom.set(int(m.group(1)))
                except Exception:
                    pass
        self._render_current_preview()

    def _on_preview_zoom_change(self, event=None):
        try:
            z = int(float(self.preview_zoom.get()))
        except Exception:
            z = 100
        z = max(10, min(400, z))
        self.preview_zoom.set(z)
        self.preview_scale_mode.set("Custom")
        self._render_current_preview()

    def _change_preview_zoom(self, delta_steps: int):
        mode = self.preview_scale_mode.get().strip()
        if mode.lower() == "fit":
            current = 100
        else:
            try:
                current = int(self.preview_zoom.get())
            except Exception:
                current = 100
        new_zoom = min(400, max(10, current + 10 * int(delta_steps)))
        self.preview_zoom.set(new_zoom)
        self.preview_scale_mode.set(f"{new_zoom}%")
        self._render_current_preview()

    def _render_current_preview(self):
        if self.preview_source_image is None:
            return
        try:
            img = self.preview_source_image
            iw, ih = img.size
            if iw <= 0 or ih <= 0:
                return

            scale = self._preview_scale_factor()
            if scale is None:
                self.preview_canvas.update_idletasks()
                avail_w = max(50, self.preview_canvas.winfo_width() - 4)
                avail_h = max(50, self.preview_canvas.winfo_height() - 4)
                scale = min(avail_w / iw, avail_h / ih)
                scale = max(scale, 0.05)

            nw = max(1, int(round(iw * scale)))
            nh = max(1, int(round(ih * scale)))
            resample = Image.Resampling.LANCZOS if (nw < iw or nh < ih) else Image.Resampling.BICUBIC
            disp = img.resize((nw, nh), resample=resample)

            tkimg = ImageTk.PhotoImage(disp)
            self.preview_image = tkimg
            self.preview_panel.configure(image=tkimg, text="")
            self.preview_canvas.itemconfigure(self.preview_canvas_window, width=nw, height=nh)
            self.preview_canvas.coords(self.preview_canvas_window, 0, 0)
            self.preview_canvas.update_idletasks()
            self.preview_canvas.configure(scrollregion=(0, 0, nw, nh))
        except Exception as exc:
            self.preview_image = None
            name = self.preview_source_path.name if self.preview_source_path else "preview"
            self.preview_panel.configure(image="", text=f"Could not preview:\n{name}\n\n{exc}")

    def _build_ui(self):
        # The mode switch is global: both Extractor and Detrender present a
        # compact workflow in Basic mode and expose all numerical controls in
        # Expert mode.  Settings files persist this choice.
        mode_bar = ttk.Frame(self, padding=(10, 8, 10, 0))
        mode_bar.pack(fill="x")
        ttk.Label(mode_bar, text="Interface:").pack(side="left")
        ttk.Radiobutton(
            mode_bar, text="Basic", variable=self.interface_mode, value="basic",
            command=self._on_interface_mode_change,
        ).pack(side="left", padx=(8, 2))
        ttk.Radiobutton(
            mode_bar, text="Expert", variable=self.interface_mode, value="expert",
            command=self._on_interface_mode_change,
        ).pack(side="left", padx=2)
        ttk.Label(
            mode_bar,
            text="Basic uses a documented preset; Expert exposes every tuning control.",
        ).pack(side="left", padx=(14, 0))

        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill="both", expand=True, padx=8, pady=8)

        self.tab_extractor = ttk.Frame(self.notebook)
        self.tab_detrender = ttk.Frame(self.notebook)
        self.tab_run = ttk.Frame(self.notebook)
        self.tab_help = ttk.Frame(self.notebook)

        self.notebook.add(self.tab_extractor, text="Extractor")
        self.notebook.add(self.tab_detrender, text="Detrender")
        self.notebook.add(self.tab_run, text="Run / Preview")
        self.notebook.add(self.tab_help, text="Help")

        self._build_extractor_tab()
        self._build_detrender_tab()
        self._build_run_tab()
        self._build_help_tab()

    def _build_extractor_tab(self):
        sf = ScrollableFrame(self.tab_extractor)
        sf.pack(fill="both", expand=True)
        root = sf.inner

        self.ex_script_paths_frame = self._build_script_paths_frame(root)
        self.ex_script_paths_frame.grid(row=0, column=0, sticky="ew", padx=8, pady=6)

        input_frame = ttk.LabelFrame(root, text="TPF Input")
        input_frame.grid(row=1, column=0, sticky="ew", padx=8, pady=6)
        input_frame.columnconfigure(1, weight=1)

        ttk.Label(input_frame, text="Mode").grid(row=0, column=0, sticky="w", padx=6, pady=4)
        mode_row = ttk.Frame(input_frame)
        mode_row.grid(row=0, column=1, columnspan=3, sticky="w", padx=6, pady=4)
        rb = ttk.Radiobutton(mode_row, text="Directory", variable=self.ex_input_mode, value="directory",
                        command=self._update_extractor_state)
        rb.pack(side="left", padx=4)
        self._tooltip(rb, "ex_input_mode")
        rb = ttk.Radiobutton(mode_row, text="Single file", variable=self.ex_input_mode, value="single",
                        command=self._update_extractor_state)
        rb.pack(side="left", padx=4)
        self._tooltip(rb, "ex_input_mode")

        ttk.Label(input_frame, text="TPF directory").grid(row=1, column=0, sticky="w", padx=6, pady=4)
        self.ex_dir_entry = ttk.Entry(input_frame, textvariable=self.ex_tpf_dir)
        self._tooltip(self.ex_dir_entry, "ex_tpf_dir")
        self.ex_dir_entry.grid(row=1, column=1, sticky="ew", padx=6, pady=4)
        ttk.Button(input_frame, text="Browse...", command=lambda: self._browse_dir(self.ex_tpf_dir)).grid(row=1, column=2, padx=6, pady=4)
        self.ex_recursive_check = ttk.Checkbutton(input_frame, text="Recursive", variable=self.ex_recursive)
        self._tooltip(self.ex_recursive_check, "ex_recursive")
        self.ex_recursive_check.grid(row=1, column=3, sticky="w", padx=6, pady=4)

        ttk.Label(input_frame, text="Single FITS file").grid(row=2, column=0, sticky="w", padx=6, pady=4)
        self.ex_single_entry = ttk.Entry(input_frame, textvariable=self.ex_single_file)
        self._tooltip(self.ex_single_entry, "ex_single_file")
        self.ex_single_entry.grid(row=2, column=1, sticky="ew", padx=6, pady=4)
        ttk.Button(input_frame, text="Browse...", command=lambda: self._browse_file(self.ex_single_file, [("FITS files", "*.fits *.fits.gz"), ("All files", "*.*")])).grid(row=2, column=2, padx=6, pady=4)

        out_frame = ttk.LabelFrame(root, text="Output")
        out_frame.grid(row=2, column=0, sticky="ew", padx=8, pady=6)
        out_frame.columnconfigure(1, weight=1)
        ttk.Label(out_frame, text="Output root").grid(row=0, column=0, sticky="w", padx=6, pady=4)
        ttk.Entry(out_frame, textvariable=self.ex_output_root).grid(row=0, column=1, sticky="ew", padx=6, pady=4)
        ttk.Button(out_frame, text="Browse...", command=lambda: self._browse_dir(self.ex_output_root)).grid(row=0, column=2, padx=6, pady=4)
        self._make_checkbutton(out_frame, text="Save aperture plots", variable=self.ex_save_aperture_plots,
                        command=self._update_extractor_command_preview, tooltip_key="ex_save_aperture_plots",
                        row=1, column=1, sticky="w", padx=6, pady=4)

        # Basic settings contain only the choices needed for routine runs.
        self.ex_basic_settings_frame = ttk.LabelFrame(root, text="Extraction settings")
        self.ex_basic_settings_frame.grid(row=3, column=0, sticky="ew", padx=8, pady=6)
        for c in range(4):
            self.ex_basic_settings_frame.columnconfigure(c, weight=1)
        self.ex_approach_combo = self._combo(
            self.ex_basic_settings_frame,
            "Extraction approach",
            self.ex_extraction_approach,
            [STANDARD_APPROACH, SATURATED_APPROACH],
            0,
            0,
            tooltip_key="ex_extraction_approach",
        )
        self.ex_approach_combo.bind("<<ComboboxSelected>>", self._update_extractor_state)
        self.ex_n_targets_spin = self._spin(
            self.ex_basic_settings_frame, "N targets", self.ex_n_targets,
            1, 20, 1, 0, tooltip_key="ex_n_targets",
        )
        self.ex_method_combo = self._combo(
            self.ex_basic_settings_frame, "Aperture method", self.ex_method,
            ["jump", "core", "both"], 1, 2, tooltip_key="ex_method",
        )
        ttk.Label(
            self.ex_basic_settings_frame,
            text="Saturation-optimized extraction is a single-target approach and does not run Gaia, jump/core, or PRF photometry.",
            wraplength=980,
        ).grid(row=2, column=0, columnspan=4, sticky="w", padx=6, pady=(2, 6))
        self.ex_basic_warning_label = ttk.Label(
            self.ex_basic_settings_frame, text="", foreground="firebrick", wraplength=980,
        )
        self.ex_basic_warning_label.grid(row=3, column=0, columnspan=4, sticky="w", padx=6, pady=(0, 4))

        # These less common choices remain available in Expert mode.
        main_frame = ttk.LabelFrame(root, text="Expert extraction controls")
        self.ex_expert_main_frame = main_frame
        main_frame.grid(row=4, column=0, sticky="ew", padx=8, pady=6)
        for c in range(4):
            main_frame.columnconfigure(c, weight=1)

        self._entry(main_frame, "Gaia radius [arcmin]", self.ex_gaia_radius, 0, 0, tooltip_key="ex_gaia_radius")

        self._make_checkbutton(main_frame, text="No Gaia", variable=self.ex_no_gaia, command=self._update_extractor_state,
                        tooltip_key="ex_no_gaia", row=0, column=2, sticky="w", padx=6, pady=4)
        self._make_checkbutton(main_frame, text="Gaia fallback", variable=self.ex_gaia_fallback, command=self._update_extractor_command_preview,
                        tooltip_key="ex_gaia_fallback", row=0, column=3, sticky="w", padx=6, pady=4)
        self._make_checkbutton(main_frame, text="Ignore quality flags", variable=self.ex_no_quality0, command=self._update_extractor_command_preview,
                        tooltip_key="ex_no_quality0", row=2, column=0, sticky="w", padx=6, pady=4)
        self._make_checkbutton(main_frame, text="Full-region sum", variable=self.ex_full_region_sum, command=self._update_extractor_state,
                        tooltip_key="ex_full_region_sum", row=2, column=1, sticky="w", padx=6, pady=4)
        self._make_checkbutton(main_frame, text="Gaia region sum", variable=self.ex_gaia_region_sum, command=self._update_extractor_state,
                        tooltip_key="ex_gaia_region_sum", row=2, column=2, sticky="w", padx=6, pady=4)
        self._make_checkbutton(main_frame, text="Save pickled figures", variable=self.ex_save_pickled_figures, command=self._update_extractor_command_preview,
                        tooltip_key="ex_save_pickled_figures", row=2, column=3, sticky="w", padx=6, pady=4)
        self._make_checkbutton(
            main_frame,
            text="Saturated-target systematics correction",
            variable=self.ex_saturated_systematics,
            command=self._update_extractor_state,
            tooltip_key="ex_saturated_systematics",
            row=3,
            column=0,
            columnspan=3,
            sticky="w",
            padx=6,
            pady=4,
        )

        self.ex_warning_label = ttk.Label(main_frame, text="", foreground="firebrick")
        self.ex_warning_label.grid(row=4, column=0, columnspan=4, sticky="w", padx=6, pady=(4, 0))

        adv = ttk.LabelFrame(root, text="Expert numerical controls")
        self.ex_advanced_frame = adv
        adv.grid(row=5, column=0, sticky="ew", padx=8, pady=6)
        for c in range(4):
            adv.columnconfigure(c, weight=1)

        jumpcore = ttk.LabelFrame(adv, text="Jump / core shared")
        jumpcore.grid(row=0, column=0, columnspan=4, sticky="ew", padx=6, pady=6)
        for c in range(4):
            jumpcore.columnconfigure(c, weight=1)
        self._spin(jumpcore, "Min pixels", self.ex_min_pixels, 1, 1000, 0, 0, tooltip_key="ex_min_pixels")
        self._entry(jumpcore, "Amp q low", self.ex_amp_q_lo, 0, 2, tooltip_key="ex_amp_q_lo")
        self._entry(jumpcore, "Amp q high", self.ex_amp_q_hi, 1, 0, tooltip_key="ex_amp_q_hi")
        self._entry(jumpcore, "Amp min frac", self.ex_amp_min_frac, 1, 2, tooltip_key="ex_amp_min_frac")
        self._entry(jumpcore, "Max radius pix", self.ex_max_radius_pix, 2, 0, tooltip_key="ex_max_radius_pix")

        jump = ttk.LabelFrame(adv, text="Jump only")
        jump.grid(row=1, column=0, columnspan=2, sticky="ew", padx=6, pady=6)
        for c in range(2):
            jump.columnconfigure(c, weight=1)
        self._spin(jump, "Max components", self.ex_max_components, 1, 20, 0, 0, tooltip_key="ex_max_components")
        self._entry(jump, "Min seed frac of peak", self.ex_min_seed_frac, 1, 0, tooltip_key="ex_min_seed_frac")
        self._spin(jump, "Min new pixels / component", self.ex_min_new_pixels, 1, 500, 2, 0, tooltip_key="ex_min_new_pixels")

        core = ttk.LabelFrame(adv, text="Core only")
        core.grid(row=1, column=2, columnspan=2, sticky="ew", padx=6, pady=6)
        for c in range(2):
            core.columnconfigure(c, weight=1)
        self._spin(core, "Core npix", self.ex_core_npix, 1, 500, 0, 0, tooltip_key="ex_core_npix")
        self._entry(core, "Core min frac of peak", self.ex_core_min_frac, 1, 0, tooltip_key="ex_core_min_frac")

        saturated = ttk.LabelFrame(adv, text="Saturated-target processing")
        saturated.grid(row=2, column=0, columnspan=4, sticky="ew", padx=6, pady=6)
        for c in range(4):
            saturated.columnconfigure(c, weight=1)
        self._entry(saturated, "Saturation threshold", self.ex_sat_thresh, 0, 0, tooltip_key="ex_sat_thresh")
        self._spin(saturated, "Minimum saturated pixels", self.ex_sat_min_npix, 1, 10000, 0, 2, tooltip_key="ex_sat_min_npix")
        self._spin(saturated, "Faint background pixels", self.ex_back_nfaint, 1, 1000, 1, 0, tooltip_key="ex_back_nfaint")
        self._entry(saturated, "Orbital phase bin", self.ex_phase_bin, 1, 2, tooltip_key="ex_phase_bin")
        self._entry(saturated, "Initial saturated-aperture threshold", self.ex_saturated_aperture_threshold, 2, 0, tooltip_key="ex_saturated_aperture_threshold")
        self._entry(saturated, "Orbital table CSV", self.ex_orbtable, 3, 0, browse="file", tooltip_key="ex_orbtable")

        extras = ttk.LabelFrame(adv, text="Additional extractor options")
        extras.grid(row=3, column=0, columnspan=4, sticky="ew", padx=6, pady=6)
        for c in range(4):
            extras.columnconfigure(c, weight=1)
        self._combo(extras, "Aperture FOM", self.ex_aperture_fom, ["stddiff", "std", "mad"], 0, 0, tooltip_key="ex_aperture_fom")
        self._combo(extras, "Simple ap mode", self.ex_simple_aperture_mode, ["auto", "fullstamp", "fixedap", "apgrow"], 0, 2, tooltip_key="ex_simple_aperture_mode")
        self._entry(extras, "External mask file", self.ex_external_mask_file, 1, 0, browse="file", tooltip_key="ex_external_mask_file")

        self.ex_prf_basic_frame = ttk.LabelFrame(root, text="Optional jitter-aware PRF photometry")
        self.ex_prf_basic_frame.grid(row=4, column=0, sticky="ew", padx=8, pady=6)
        for c in range(4):
            self.ex_prf_basic_frame.columnconfigure(c, weight=1)
        self.ex_prf_basic_check = self._make_checkbutton(
            self.ex_prf_basic_frame,
            text="Also generate a jitter-aware PRF light curve",
            variable=self.ex_prf_photometry,
            command=self._update_extractor_state,
            tooltip_key="ex_prf_photometry",
            row=0,
            column=0,
            columnspan=2,
            sticky="w",
            padx=6,
            pady=4,
        )
        self.ex_prf_basic_scene_combo = self._combo(
            self.ex_prf_basic_frame,
            "PRF scene",
            self.ex_prf_scene_mode,
            ["single", "gaia"],
            0,
            2,
            tooltip_key="ex_prf_scene_mode",
        )

        prf = ttk.LabelFrame(root, text="Expert jitter-aware PRF controls")
        self.ex_prf_expert_frame = prf
        prf.grid(row=6, column=0, sticky="ew", padx=8, pady=6)
        for c in range(4):
            prf.columnconfigure(c, weight=1)
        self._make_checkbutton(
            prf, text="Enable jitter-aware PRF photometry", variable=self.ex_prf_photometry,
            command=self._update_extractor_state, tooltip_key="ex_prf_photometry",
            row=0, column=0, columnspan=2, sticky="w", padx=6, pady=4,
        )
        self.ex_prf_diag_check = self._make_checkbutton(
            prf, text="Save PRF diagnostics", variable=self.ex_prf_save_diagnostics,
            command=self._update_extractor_command_preview, tooltip_key="ex_prf_save_diagnostics",
            row=0, column=2, sticky="w", padx=6, pady=4,
        )
        self.ex_prf_gauss_check = self._make_checkbutton(
            prf, text="Allow Gaussian fallback", variable=self.ex_prf_allow_gaussian_fallback,
            command=self._update_extractor_command_preview, tooltip_key="ex_prf_allow_gaussian_fallback",
            row=0, column=3, sticky="w", padx=6, pady=4,
        )
        self.ex_prf_backend_combo = self._combo(
            prf, "PRF backend", self.ex_prf_backend, ["auto", "lkprf", "tess_prf", "gaussian"],
            1, 0, tooltip_key="ex_prf_backend",
        )
        self.ex_prf_motion_combo = self._combo(
            prf, "Motion source", self.ex_prf_motion_source, ["auto", "poscorr", "ensemble", "target", "fixed"],
            1, 2, tooltip_key="ex_prf_motion_source",
        )
        self.ex_prf_scene_combo = self._combo(
            prf, "Scene mode", self.ex_prf_scene_mode, ["single", "gaia"],
            2, 0, tooltip_key="ex_prf_scene_mode",
        )
        self.ex_prf_neighbor_treatment_combo = self._combo(
            prf, "Neighbor treatment", self.ex_prf_neighbor_treatment, ["fixed"],
            2, 2, tooltip_key="ex_prf_neighbor_treatment",
        )
        self.ex_prf_background_combo = self._combo(
            prf, "Background", self.ex_prf_background, ["none", "constant", "plane"],
            3, 0, tooltip_key="ex_prf_background",
        )
        self.ex_prf_min_weight_entry = self._entry(
            prf, "Min PRF weight", self.ex_prf_min_weight, 3, 2, tooltip_key="ex_prf_min_weight"
        )
        self.ex_prf_neighbor_dmag_entry = self._entry(
            prf, "Neighbor max dG", self.ex_prf_neighbor_dmag, 4, 0, tooltip_key="ex_prf_neighbor_dmag"
        )
        self.ex_prf_neighbor_margin_entry = self._entry(
            prf, "Neighbor margin [pix]", self.ex_prf_neighbor_margin, 4, 2, tooltip_key="ex_prf_neighbor_margin"
        )
        self.ex_prf_max_scene_sources_entry = self._entry(
            prf, "Max scene sources", self.ex_prf_max_scene_sources, 5, 0, tooltip_key="ex_prf_max_scene_sources"
        )
        self.ex_prf_min_neighbor_fraction_entry = self._entry(
            prf, "Min neighbor fraction", self.ex_prf_min_neighbor_fraction, 5, 2, tooltip_key="ex_prf_min_neighbor_fraction"
        )
        self.ex_prf_fit_radius_entry = self._entry(
            prf, "Fit radius [pix]", self.ex_prf_fit_radius, 6, 0, tooltip_key="ex_prf_fit_radius"
        )
        self.ex_prf_quant_entry = self._entry(
            prf, "PRF grid step [pix]", self.ex_prf_shift_quantization, 6, 2, tooltip_key="ex_prf_shift_quantization"
        )
        self.ex_prf_max_shift_entry = self._entry(
            prf, "Max shift [pix]", self.ex_prf_max_shift, 7, 0, tooltip_key="ex_prf_max_shift"
        )
        ttk.Label(
            prf,
            text="Gaia mode: one active source per fit; all other scene sources fixed. "
                 "Auto motion: POS_CORR → ensemble → target → fixed.",
        ).grid(row=7, column=2, columnspan=2, sticky="w", padx=6, pady=4)
        self.ex_prf_source_output_combo = self._combo(
            prf, "PRF light curves", self.ex_prf_source_output, ["primary", "all"],
            8, 0, tooltip_key="ex_prf_source_output",
        )
        ttk.Label(
            prf,
            text="Default: primary only. 'all' writes each usable retained Gaia source separately.",
        ).grid(row=8, column=2, columnspan=2, sticky="w", padx=6, pady=4)
        self.ex_prf_widgets = [
            self.ex_prf_backend_combo, self.ex_prf_motion_combo, self.ex_prf_scene_combo,
            self.ex_prf_neighbor_treatment_combo, self.ex_prf_source_output_combo, self.ex_prf_background_combo,
            self.ex_prf_min_weight_entry, self.ex_prf_neighbor_dmag_entry,
            self.ex_prf_neighbor_margin_entry, self.ex_prf_max_scene_sources_entry,
            self.ex_prf_min_neighbor_fraction_entry, self.ex_prf_fit_radius_entry,
            self.ex_prf_quant_entry, self.ex_prf_max_shift_entry,
            self.ex_prf_diag_check, self.ex_prf_gauss_check,
        ]

        action = ttk.LabelFrame(root, text="Actions")
        action.grid(row=7, column=0, sticky="ew", padx=8, pady=6)
        for c in range(5):
            action.columnconfigure(c, weight=1)
        ttk.Button(action, text="Show command", command=self._update_extractor_command_preview).grid(row=0, column=0, padx=6, pady=6, sticky="ew")
        ttk.Button(action, text="Run extractor", command=self.run_extractor).grid(row=0, column=1, padx=6, pady=6, sticky="ew")
        ttk.Button(action, text="Stop", command=self.stop_current_process).grid(row=0, column=2, padx=6, pady=6, sticky="ew")
        ttk.Button(action, text="Open output folder", command=lambda: self._open_folder(Path(self.ex_output_root.get()))).grid(row=0, column=3, padx=6, pady=6, sticky="ew")
        ttk.Button(action, text="Help", command=self.show_help_tab).grid(row=0, column=4, padx=6, pady=6, sticky="ew")
        ttk.Label(action, text="Command preview").grid(row=1, column=0, sticky="w", padx=6)
        ex_cmd_entry = ttk.Entry(action, textvariable=self.ex_command_preview)
        ex_cmd_entry.grid(row=2, column=0, columnspan=5, sticky="ew", padx=6, pady=6)
        self._tooltip(ex_cmd_entry, "ex_command_preview")

        root.columnconfigure(0, weight=1)
        self._trace_extractor_vars()

    def _build_detrender_tab(self):
        sf = ScrollableFrame(self.tab_detrender)
        sf.pack(fill="both", expand=True)
        root = sf.inner

        self.dt_script_paths_frame = self._build_script_paths_frame(root)
        self.dt_script_paths_frame.grid(row=0, column=0, sticky="ew", padx=8, pady=6)

        self.dt_basic_frame = ttk.LabelFrame(root, text="Detrending settings")
        self.dt_basic_frame.grid(row=1, column=0, sticky="ew", padx=8, pady=6)
        for c in range(4):
            self.dt_basic_frame.columnconfigure(c, weight=1)
        self._entry(self.dt_basic_frame, "Input light-curve directory", self.dt_lightcurve_dir, 0, 0, browse="dir", tooltip_key="dt_lightcurve_dir")
        self._entry(self.dt_basic_frame, "Output directory", self.dt_output_dir, 1, 0, browse="dir", tooltip_key="dt_output_dir")
        self._make_checkbutton(
            self.dt_basic_frame, text="Use background correction", variable=self.dt_use_background,
            command=self._update_detrender_state, tooltip_key="dt_use_background",
            row=2, column=0, sticky="w", padx=6, pady=4,
        )
        self._make_checkbutton(
            self.dt_basic_frame, text="Protect intrinsic variability (PCHIP)", variable=self.dt_pre_model_pchip,
            command=self._update_detrender_command_preview, tooltip_key="dt_pre_model_pchip",
            row=2, column=1, sticky="w", padx=6, pady=4,
        )
        self._make_checkbutton(
            self.dt_basic_frame, text="Use quaternion regression", variable=self.dt_use_quaternion_regression,
            command=self._update_detrender_state, tooltip_key="dt_use_quaternion_regression",
            row=2, column=2, sticky="w", padx=6, pady=4,
        )
        self._make_checkbutton(
            self.dt_basic_frame, text="Combine sectors", variable=self.dt_combine_sectors,
            command=self._update_detrender_command_preview, tooltip_key="dt_combine_sectors",
            row=2, column=3, sticky="w", padx=6, pady=4,
        )
        self._make_checkbutton(
            self.dt_basic_frame, text="Skip existing outputs", variable=self.dt_skip_existing,
            command=self._update_detrender_command_preview, tooltip_key="dt_skip_existing",
            row=3, column=0, columnspan=2, sticky="w", padx=6, pady=4,
        )

        dirs = ttk.LabelFrame(root, text="Directories")
        self.dt_expert_dirs_frame = dirs
        dirs.grid(row=1, column=0, sticky="ew", padx=8, pady=6)
        dirs.columnconfigure(1, weight=1)
        self._entry(dirs, "Lightcurve dir", self.dt_lightcurve_dir, 0, 0, browse="dir", tooltip_key="dt_lightcurve_dir")
        self._entry(dirs, "Diagnostics dir", self.dt_diagnostics_dir, 1, 0, browse="dir", tooltip_key="dt_diagnostics_dir")
        self._entry(dirs, "Output dir", self.dt_output_dir, 2, 0, browse="dir", tooltip_key="dt_output_dir")

        sel = ttk.LabelFrame(root, text="Input selection")
        self.dt_expert_selection_frame = sel
        sel.grid(row=2, column=0, sticky="ew", padx=8, pady=6)
        for c in range(4):
            sel.columnconfigure(c, weight=1)
        self._entry(sel, "Pattern", self.dt_pattern, 0, 0, tooltip_key="dt_pattern")
        self._entry(sel, "Prefix", self.dt_prefix, 0, 2, tooltip_key="dt_prefix")
        self._make_checkbutton(sel, text="Recursive", variable=self.dt_recursive, command=self._update_detrender_command_preview, tooltip_key="dt_recursive", row=1, column=0, sticky="w", padx=6, pady=4)
        self._make_checkbutton(sel, text="Skip existing outputs", variable=self.dt_skip_existing, command=self._update_detrender_command_preview, tooltip_key="dt_skip_existing", row=1, column=1, sticky="w", padx=6, pady=4)

        opts = ttk.LabelFrame(root, text="Detrending options")
        self.dt_expert_options_frame = opts
        opts.grid(row=3, column=0, sticky="ew", padx=8, pady=6)
        for c in range(4):
            opts.columnconfigure(c, weight=1)
        self.dt_chk_background = self._make_checkbutton(opts, text="Use background", variable=self.dt_use_background, command=self._update_detrender_state, tooltip_key="dt_use_background")
        self.dt_chk_background.grid(row=0, column=0, sticky="w", padx=6, pady=4)
        self.dt_chk_pchip = self._make_checkbutton(opts, text="Use PCHIP high-pass", variable=self.dt_use_pchip, command=self._update_detrender_state, tooltip_key="dt_use_pchip")
        self.dt_chk_pchip.grid(row=0, column=1, sticky="w", padx=6, pady=4)
        self.dt_chk_skip = self._make_checkbutton(opts, text="Skip XY/background decorrelation", variable=self.dt_skip_xybg, command=self._update_detrender_state, tooltip_key="dt_skip_xybg")
        self.dt_chk_skip.grid(row=0, column=2, sticky="w", padx=6, pady=4)
        self._make_checkbutton(opts, text="Combine sectors", variable=self.dt_combine_sectors, command=self._update_detrender_command_preview, tooltip_key="dt_combine_sectors", row=0, column=3, sticky="w", padx=6, pady=4)
        self._entry(opts, "Knot spacing days", self.dt_knot_spacing, 1, 0, tooltip_key="dt_knot_spacing")
        self._make_checkbutton(opts, text="Pre-model PCHIP (recommended with quaternion)", variable=self.dt_pre_model_pchip, command=self._update_detrender_command_preview, tooltip_key="dt_pre_model_pchip", row=1, column=2, sticky="w", padx=6, pady=4)
        self._spin(opts, "Robust iters", self.dt_robust_iters, 1, 100, 1, 0, tooltip_key="dt_robust_iters")
        self._entry(opts, "Huber k", self.dt_huber_k, 2, 0, tooltip_key="dt_huber_k")
        self.dt_pchip_entry = self._entry(opts, "PCHIP knot spacing", self.dt_pchip_knot_spacing, 2, 2, tooltip_key="dt_pchip_knot_spacing")
        self._entry(opts, "Pre-model bin days", self.dt_pre_model_bin_days, 3, 2, tooltip_key="dt_pre_model_bin_days")
        self._combo(opts, "Pre-model stat", self.dt_pre_model_stat, ["median", "mean"], 4, 0, tooltip_key="dt_pre_model_stat")
        self._spin(opts, "Pre-model min pts", self.dt_pre_model_min_points, 1, 9999, 4, 2, tooltip_key="dt_pre_model_min_points")
        self._entry(opts, "Pre-model sigma clip", self.dt_pre_model_sigma_clip, 5, 0, tooltip_key="dt_pre_model_sigma_clip")
        self._spin(opts, "Pre-model sigma iters", self.dt_pre_model_sigma_iters, 1, 99, 5, 2, tooltip_key="dt_pre_model_sigma_iters")
        self._make_checkbutton(opts, text="Clip residuals before detrend", variable=self.dt_clip_residuals_before_detrend, command=self._update_detrender_command_preview, tooltip_key="dt_clip_residuals_before_detrend", row=6, column=0, sticky="w", padx=6, pady=4)
        self._entry(opts, "Residual clip sigma", self.dt_clip_residuals_sigma, 6, 2, tooltip_key="dt_clip_residuals_sigma")
        self._spin(opts, "Residual clip iters", self.dt_clip_residuals_iters, 1, 99, 7, 0, tooltip_key="dt_clip_residuals_iters")
        self._make_checkbutton(opts, text="Save pickled figures", variable=self.dt_save_pickled_figures, command=self._update_detrender_command_preview, tooltip_key="dt_save_pickled_figures", row=7, column=2, sticky="w", padx=6, pady=4)
        self._make_checkbutton(opts, text="Apply orbital phase template", variable=self.dt_apply_orbital_phase_template, command=self._update_detrender_command_preview, tooltip_key="dt_apply_orbital_phase_template", row=8, column=0, sticky="w", padx=6, pady=4)
        self._entry(opts, "Orbtable CSV", self.dt_orbtable, 8, 2, browse="file", tooltip_key="dt_orbtable")
        self._entry(opts, "Orbital phase bin", self.dt_phase_bin, 9, 0, tooltip_key="dt_phase_bin")

        quat = ttk.LabelFrame(root, text="Expert quaternion controls")
        self.dt_expert_quaternion_frame = quat
        quat.grid(row=4, column=0, sticky="ew", padx=8, pady=6)
        for c in range(4):
            quat.columnconfigure(c, weight=1)

        self.dt_chk_quaternion = self._make_checkbutton(
            quat, text="Use quaternion regression",
            variable=self.dt_use_quaternion_regression,
            command=self._update_detrender_state,
            tooltip_key="dt_use_quaternion_regression",
            row=0, column=0, columnspan=2, sticky="w", padx=6, pady=4,
        )

        ttk.Label(quat, text="Quaternion source (optional)").grid(
            row=1, column=0, sticky="w", padx=6, pady=4
        )
        self.dt_quaternion_source_entry = ttk.Entry(
            quat, textvariable=self.dt_quaternion_source
        )
        self.dt_quaternion_source_entry.grid(
            row=1, column=1, sticky="ew", padx=6, pady=4
        )
        self._tooltip(self.dt_quaternion_source_entry, "dt_quaternion_source")
        self.dt_quaternion_file_button = ttk.Button(
            quat, text="Browse file...",
            command=lambda: self._browse_file(
                self.dt_quaternion_source,
                [("Quaternion/vector files", "*.fits *.fits.gz *.fit *.csv *.csv.gz *.csv.xz *.ecsv *.txt"),
                 ("All files", "*.*")]
            ),
        )
        self.dt_quaternion_file_button.grid(row=1, column=2, padx=6, pady=4)
        self._tooltip(self.dt_quaternion_file_button, "dt_quaternion_source")
        self.dt_quaternion_dir_button = ttk.Button(
            quat, text="Browse dir...",
            command=lambda: self._browse_dir(self.dt_quaternion_source),
        )
        self.dt_quaternion_dir_button.grid(row=1, column=3, padx=6, pady=4)
        self._tooltip(self.dt_quaternion_dir_button, "dt_quaternion_source")

        self.dt_chk_quaternion_auto_download = self._make_checkbutton(
            quat, text="Auto-download missing TESSVectors",
            variable=self.dt_quaternion_auto_download,
            command=self._update_detrender_state,
            tooltip_key="dt_quaternion_auto_download",
            row=2, column=0, columnspan=2, sticky="w", padx=6, pady=4,
        )
        ttk.Label(quat, text="TESSVectors cache").grid(
            row=3, column=0, sticky="w", padx=6, pady=4
        )
        self.dt_tessvectors_cache_entry = ttk.Entry(
            quat, textvariable=self.dt_tessvectors_cache_dir
        )
        self.dt_tessvectors_cache_entry.grid(
            row=3, column=1, columnspan=2, sticky="ew", padx=6, pady=4
        )
        self._tooltip(self.dt_tessvectors_cache_entry, "dt_tessvectors_cache_dir")
        self.dt_tessvectors_cache_button = ttk.Button(
            quat, text="Browse...",
            command=lambda: self._browse_dir(self.dt_tessvectors_cache_dir),
        )
        self.dt_tessvectors_cache_button.grid(row=3, column=3, padx=6, pady=4)
        self._tooltip(self.dt_tessvectors_cache_button, "dt_tessvectors_cache_dir")

        self.dt_quaternion_camera_combo = self._combo(
            quat, "Camera", self.dt_quaternion_camera,
            ["auto", "1", "2", "3", "4"], 4, 0,
            tooltip_key="dt_quaternion_camera"
        )
        self.dt_quaternion_min_samples_spin = self._spin(
            quat, "Min samples / cadence", self.dt_quaternion_min_samples,
            1, 999, 4, 2, tooltip_key="dt_quaternion_min_samples"
        )
        self.dt_quaternion_clip_sigma_entry = self._entry(
            quat, "Clip sigma", self.dt_quaternion_clip_sigma,
            5, 0, tooltip_key="dt_quaternion_clip_sigma"
        )
        self.dt_quaternion_clip_iters_spin = self._spin(
            quat, "Clip iterations", self.dt_quaternion_clip_iters,
            1, 99, 5, 2, tooltip_key="dt_quaternion_clip_iters"
        )
        self.dt_quaternion_time_offset_entry = self._entry(
            quat, "Time offset days", self.dt_quaternion_time_offset,
            6, 0, tooltip_key="dt_quaternion_time_offset"
        )
        self.dt_chk_quaternion_diagnostics = self._make_checkbutton(
            quat, text="Save quaternion diagnostics",
            variable=self.dt_save_quaternion_diagnostics,
            command=self._update_detrender_command_preview,
            tooltip_key="dt_save_quaternion_diagnostics",
            row=6, column=2, columnspan=2, sticky="w", padx=6, pady=4,
        )
        ttk.Label(
            quat,
            text="Leave Quaternion source blank to download and cache only the required TESSVectors file. "
                 "Raw 2-second files remain supported for the full 36-feature model.",
            wraplength=1050,
        ).grid(row=7, column=0, columnspan=4, sticky="w", padx=6, pady=(2, 6))

        self.dt_quaternion_widgets = [
            self.dt_quaternion_source_entry,
            self.dt_quaternion_file_button,
            self.dt_quaternion_dir_button,
            self.dt_chk_quaternion_auto_download,
            self.dt_quaternion_camera_combo,
            self.dt_quaternion_min_samples_spin,
            self.dt_quaternion_clip_sigma_entry,
            self.dt_quaternion_clip_iters_spin,
            self.dt_quaternion_time_offset_entry,
            self.dt_chk_quaternion_diagnostics,
        ]
        self.dt_quaternion_cache_widgets = [
            self.dt_tessvectors_cache_entry,
            self.dt_tessvectors_cache_button,
        ]

        action = ttk.LabelFrame(root, text="Actions")
        action.grid(row=5, column=0, sticky="ew", padx=8, pady=6)
        for c in range(5):
            action.columnconfigure(c, weight=1)
        ttk.Button(action, text="Show command", command=self._update_detrender_command_preview).grid(row=0, column=0, padx=6, pady=6, sticky="ew")
        ttk.Button(action, text="Run detrender", command=self.run_detrender).grid(row=0, column=1, padx=6, pady=6, sticky="ew")
        ttk.Button(action, text="Stop", command=self.stop_current_process).grid(row=0, column=2, padx=6, pady=6, sticky="ew")
        ttk.Button(action, text="Open output folder", command=lambda: self._open_folder(Path(self.dt_output_dir.get()))).grid(row=0, column=3, padx=6, pady=6, sticky="ew")
        ttk.Button(action, text="Help", command=self.show_help_tab).grid(row=0, column=4, padx=6, pady=6, sticky="ew")
        ttk.Label(action, text="Command preview").grid(row=1, column=0, sticky="w", padx=6)
        dt_cmd_entry = ttk.Entry(action, textvariable=self.dt_command_preview)
        dt_cmd_entry.grid(row=2, column=0, columnspan=5, sticky="ew", padx=6, pady=6)
        self._tooltip(dt_cmd_entry, "dt_command_preview")

        root.columnconfigure(0, weight=1)
        self._trace_detrender_vars()

    def _build_run_tab(self):
        top = ttk.Frame(self.tab_run)
        top.pack(fill="x", padx=8, pady=6)

        ttk.Button(top, text="Clear log", command=self.clear_log).pack(side="left", padx=4)
        ttk.Button(top, text="Save log", command=self.save_log).pack(side="left", padx=4)
        ttk.Button(top, text="Copy command", command=self.copy_active_command).pack(side="left", padx=4)
        ttk.Button(top, text="Refresh previews", command=self._refresh_preview_list).pack(side="left", padx=4)
        ttk.Button(top, text="Open preview folder", command=lambda: self._open_folder(Path(self.preview_dir.get()) if self.preview_dir.get() else Path.cwd())).pack(side="left", padx=4)
        ttk.Label(top, textvariable=self.status_text).pack(side="right", padx=4)

        paned = ttk.Panedwindow(self.tab_run, orient="vertical")
        paned.pack(fill="both", expand=True, padx=8, pady=6)

        log_frame = ttk.LabelFrame(paned, text="Log")
        preview_frame = ttk.LabelFrame(paned, text="Preview")

        paned.add(log_frame, weight=3)
        paned.add(preview_frame, weight=2)

        self.log_text = tk.Text(log_frame, wrap="word", height=20, bg="#111", fg="#ddd", insertbackground="#ddd")
        log_scroll = ttk.Scrollbar(log_frame, orient="vertical", command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=log_scroll.set)
        self.log_text.pack(side="left", fill="both", expand=True)
        log_scroll.pack(side="right", fill="y")

        preview_top = ttk.Frame(preview_frame)
        preview_top.pack(fill="x", padx=6, pady=6)
        ttk.Label(preview_top, text="Preview folder").pack(side="left")
        ent = ttk.Entry(preview_top, textvariable=self.preview_dir)
        ent.pack(side="left", fill="x", expand=True, padx=6)
        self._tooltip(ent, "preview_dir")
        ttk.Button(preview_top, text="Browse...", command=lambda: self._browse_dir(self.preview_dir, refresh=True)).pack(side="left")

        ttk.Label(preview_top, text="Scale").pack(side="left", padx=(12, 4))
        self.preview_scale_combo = ttk.Combobox(
            preview_top,
            textvariable=self.preview_scale_mode,
            values=["Fit", "25%", "50%", "75%", "100%", "125%", "150%", "200%", "Custom"],
            state="readonly",
            width=7,
        )
        self.preview_scale_combo.pack(side="left", padx=4)
        self._tooltip(self.preview_scale_combo, "preview_scale_mode")
        self.preview_scale_combo.bind("<<ComboboxSelected>>", self._on_preview_scale_mode_change)

        ttk.Label(preview_top, text="Zoom").pack(side="left", padx=(8, 4))
        self.preview_zoom_spin = ttk.Spinbox(preview_top, from_=10, to=400, increment=10, textvariable=self.preview_zoom, width=5, command=self._on_preview_zoom_change)
        self.preview_zoom_spin.pack(side="left", padx=4)
        self._tooltip(self.preview_zoom_spin, "preview_zoom")
        self.preview_zoom_spin.bind("<Return>", self._on_preview_zoom_change)
        self.preview_zoom_spin.bind("<FocusOut>", self._on_preview_zoom_change)

        preview_body = ttk.Frame(preview_frame)
        preview_body.pack(fill="both", expand=True, padx=6, pady=6)
        preview_body.columnconfigure(1, weight=1)
        preview_body.rowconfigure(0, weight=1)

        list_frame = ttk.Frame(preview_body)
        list_frame.grid(row=0, column=0, sticky="nsw")
        ttk.Label(list_frame, text="PNG files").pack(anchor="w")
        self.preview_list = tk.Listbox(list_frame, width=45)
        preview_scroll = ttk.Scrollbar(list_frame, orient="vertical", command=self.preview_list.yview)
        self.preview_list.configure(yscrollcommand=preview_scroll.set)
        self.preview_list.pack(side="left", fill="y")
        preview_scroll.pack(side="left", fill="y")
        self.preview_list.bind("<<ListboxSelect>>", self._on_preview_select)

        preview_canvas_frame = ttk.Frame(preview_body)
        preview_canvas_frame.grid(row=0, column=1, sticky="nsew", padx=(10, 0))
        preview_canvas_frame.columnconfigure(0, weight=1)
        preview_canvas_frame.rowconfigure(0, weight=1)

        self.preview_canvas = tk.Canvas(preview_canvas_frame, highlightthickness=0, bg="#ddd")
        self.preview_vscroll = ttk.Scrollbar(preview_canvas_frame, orient="vertical", command=self.preview_canvas.yview)
        self.preview_hscroll = ttk.Scrollbar(preview_canvas_frame, orient="horizontal", command=self.preview_canvas.xview)
        self.preview_canvas.configure(yscrollcommand=self.preview_vscroll.set, xscrollcommand=self.preview_hscroll.set)

        self.preview_canvas.grid(row=0, column=0, sticky="nsew")
        self.preview_vscroll.grid(row=0, column=1, sticky="ns")
        self.preview_hscroll.grid(row=1, column=0, sticky="ew")

        self.preview_panel = ttk.Label(self.preview_canvas, text="No preview selected.", anchor="nw")
        self.preview_canvas_window = self.preview_canvas.create_window((0, 0), window=self.preview_panel, anchor="nw")
        self.preview_canvas.bind("<Configure>", self._on_preview_canvas_configure)
        self.preview_canvas.bind("<MouseWheel>", self._on_preview_mousewheel)
        self.preview_canvas.bind("<Shift-MouseWheel>", self._on_preview_shift_mousewheel)
        self.preview_canvas.bind("<Control-MouseWheel>", self._on_preview_zoom_mousewheel)
        self.preview_canvas.bind("<Button-4>", self._on_preview_mousewheel)
        self.preview_canvas.bind("<Button-5>", self._on_preview_mousewheel)


    def _build_help_tab(self):
        container = ttk.Frame(self.tab_help)
        container.pack(fill="both", expand=True, padx=8, pady=8)

        top = ttk.Frame(container)
        top.pack(fill="x", pady=(0, 6))
        ttk.Button(top, text="Copy help text", command=self.copy_help_text).pack(side="left", padx=4)
        ttk.Button(top, text="Save help text", command=self.save_help_text).pack(side="left", padx=4)

        self.help_text_widget = scrolledtext.ScrolledText(container, wrap="word")
        self.help_text_widget.pack(fill="both", expand=True)
        self.help_text_widget.insert("1.0", HELP_TEXT)
        self.help_text_widget.configure(state="disabled")

    def show_help_tab(self):
        self.notebook.select(self.tab_help)

    def copy_help_text(self):
        self.clipboard_clear()
        self.clipboard_append(HELP_TEXT)

    def save_help_text(self):
        path = filedialog.asksaveasfilename(
            title="Save help text",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        if not path:
            return
        Path(path).write_text(HELP_TEXT, encoding="utf-8")

    def _build_script_paths_frame(self, master):
        frm = ttk.LabelFrame(master, text="Script paths")
        frm.columnconfigure(1, weight=1)
        frm.columnconfigure(4, weight=1)

        ttk.Label(frm, text="Extractor script").grid(row=0, column=0, sticky="w", padx=6, pady=4)
        ttk.Entry(frm, textvariable=self.extractor_script).grid(row=0, column=1, sticky="ew", padx=6, pady=4)
        ttk.Button(frm, text="Browse...", command=lambda: self._browse_file(self.extractor_script, [("Python files", "*.py"), ("All files", "*.*")])).grid(row=0, column=2, padx=6, pady=4)

        ttk.Label(frm, text="Detrender script").grid(row=0, column=3, sticky="w", padx=6, pady=4)
        ttk.Entry(frm, textvariable=self.detrender_script).grid(row=0, column=4, sticky="ew", padx=6, pady=4)
        ttk.Button(frm, text="Browse...", command=lambda: self._browse_file(self.detrender_script, [("Python files", "*.py"), ("All files", "*.*")])).grid(row=0, column=5, padx=6, pady=4)
        return frm

    def _entry(self, parent, label, variable, row, col, browse=None, tooltip_key=None):
        lab = ttk.Label(parent, text=label)
        lab.grid(row=row, column=col, sticky="w", padx=6, pady=4)
        if tooltip_key:
            self._tooltip(lab, tooltip_key)
        entry = ttk.Entry(parent, textvariable=variable)
        entry.grid(row=row, column=col + 1, sticky="ew", padx=6, pady=4)
        if tooltip_key:
            self._tooltip(entry, tooltip_key)
        if browse == "dir":
            btn = ttk.Button(parent, text="Browse...", command=lambda: self._browse_dir(variable))
            btn.grid(row=row, column=col + 2, padx=6, pady=4)
            if tooltip_key:
                self._tooltip(btn, tooltip_key)
        elif browse == "file":
            btn = ttk.Button(parent, text="Browse...", command=lambda: self._browse_file(variable, [("All files", "*.*")]))
            btn.grid(row=row, column=col + 2, padx=6, pady=4)
            if tooltip_key:
                self._tooltip(btn, tooltip_key)
        return entry

    def _spin(self, parent, label, variable, from_, to_, row, col, tooltip_key=None):
        lab = ttk.Label(parent, text=label)
        lab.grid(row=row, column=col, sticky="w", padx=6, pady=4)
        if tooltip_key:
            self._tooltip(lab, tooltip_key)
        sp = ttk.Spinbox(parent, textvariable=variable, from_=from_, to=to_)
        sp.grid(row=row, column=col + 1, sticky="ew", padx=6, pady=4)
        if tooltip_key:
            self._tooltip(sp, tooltip_key)
        return sp

    def _combo(self, parent, label, variable, values, row, col, tooltip_key=None):
        lab = ttk.Label(parent, text=label)
        lab.grid(row=row, column=col, sticky="w", padx=6, pady=4)
        if tooltip_key:
            self._tooltip(lab, tooltip_key)
        cb = ttk.Combobox(parent, textvariable=variable, values=values, state="readonly")
        cb.grid(row=row, column=col + 1, sticky="ew", padx=6, pady=4)
        if tooltip_key:
            self._tooltip(cb, tooltip_key)
        return cb

    def _trace_extractor_vars(self):
        vars_to_trace = [
            self.extractor_script, self.ex_input_mode, self.ex_tpf_dir, self.ex_single_file,
            self.ex_recursive, self.ex_output_root, self.ex_n_targets, self.ex_method,
            self.ex_gaia_radius, self.ex_no_gaia, self.ex_gaia_fallback, self.ex_no_quality0,
            self.ex_full_region_sum, self.ex_gaia_region_sum, self.ex_save_aperture_plots,
            self.ex_save_pickled_figures, self.ex_saturated_systematics,
            self.ex_extraction_approach,
            self.ex_min_pixels, self.ex_amp_q_lo, self.ex_amp_q_hi, self.ex_amp_min_frac,
            self.ex_max_radius_pix, self.ex_max_components, self.ex_min_seed_frac,
            self.ex_min_new_pixels, self.ex_core_npix, self.ex_core_min_frac,
            self.ex_sat_thresh, self.ex_sat_min_npix, self.ex_back_nfaint, self.ex_phase_bin,
            self.ex_saturated_aperture_threshold, self.ex_external_mask_file,
            self.ex_orbtable, self.ex_aperture_fom, self.ex_simple_aperture_mode,
            self.ex_prf_photometry, self.ex_prf_backend, self.ex_prf_motion_source,
            self.ex_prf_scene_mode, self.ex_prf_neighbor_treatment, self.ex_prf_source_output,
            self.ex_prf_neighbor_dmag, self.ex_prf_neighbor_margin,
            self.ex_prf_max_scene_sources, self.ex_prf_min_neighbor_fraction,
            self.ex_prf_background, self.ex_prf_min_weight, self.ex_prf_fit_radius,
            self.ex_prf_shift_quantization, self.ex_prf_max_shift,
            self.ex_prf_save_diagnostics, self.ex_prf_allow_gaussian_fallback
        ]
        for v in vars_to_trace:
            v.trace_add("write", lambda *_: self._update_extractor_state())

    def _trace_detrender_vars(self):
        vars_to_trace = [
            self.detrender_script, self.dt_lightcurve_dir, self.dt_diagnostics_dir,
            self.dt_output_dir, self.dt_pattern, self.dt_recursive, self.dt_prefix,
            self.dt_skip_existing, self.dt_use_background, self.dt_use_pchip, self.dt_skip_xybg,
            self.dt_combine_sectors, self.dt_knot_spacing, self.dt_robust_iters,
            self.dt_huber_k, self.dt_pchip_knot_spacing, self.dt_pre_model_pchip, self.dt_pre_model_bin_days, self.dt_pre_model_stat, self.dt_pre_model_min_points, self.dt_pre_model_sigma_clip, self.dt_pre_model_sigma_iters, self.dt_clip_residuals_before_detrend, self.dt_clip_residuals_sigma, self.dt_clip_residuals_iters, self.dt_save_pickled_figures, self.dt_apply_orbital_phase_template, self.dt_orbtable, self.dt_phase_bin,
            self.dt_use_quaternion_regression, self.dt_quaternion_source,
            self.dt_quaternion_auto_download, self.dt_tessvectors_cache_dir,
            self.dt_quaternion_camera, self.dt_quaternion_min_samples, self.dt_quaternion_clip_sigma,
            self.dt_quaternion_clip_iters, self.dt_quaternion_time_offset,
            self.dt_save_quaternion_diagnostics
        ]
        for v in vars_to_trace:
            v.trace_add("write", lambda *_: self._update_detrender_state())

    def _update_extractor_state(self, event=None):
        is_dir_mode = (self.ex_input_mode.get() == "directory")
        self.ex_dir_entry.configure(state=("normal" if is_dir_mode else "disabled"))
        self.ex_recursive_check.configure(state=("normal" if is_dir_mode else "disabled"))
        self.ex_single_entry.configure(state=("normal" if not is_dir_mode else "disabled"))

        saturation_optimized = self.ex_extraction_approach.get() == SATURATED_APPROACH
        if saturation_optimized:
            # This approach is deliberately a complete single-target workflow;
            # it does not inherit hidden standard-method or PRF choices.
            if self.ex_n_targets.get() != 1:
                self.ex_n_targets.set(1)
            if self.ex_prf_photometry.get():
                self.ex_prf_photometry.set(False)
        if self.ex_saturated_systematics.get() and self.ex_n_targets.get() != 1:
            self.ex_n_targets.set(1)
        external_mask_selected = bool(self.ex_external_mask_file.get().strip())
        if external_mask_selected and self.ex_n_targets.get() != 1:
            self.ex_n_targets.set(1)
        single_target_locked = (
            saturation_optimized
            or self.ex_saturated_systematics.get()
            or external_mask_selected
        )
        self.ex_n_targets_spin.configure(state=("disabled" if single_target_locked else "normal"))
        self.ex_method_combo.configure(
            state=("disabled" if saturation_optimized or external_mask_selected else "readonly")
        )

        warning = []
        if self.ex_no_gaia.get() and self.ex_n_targets.get() != 1:
            warning.append("No-Gaia mode requires n-targets = 1.")
        if self.ex_saturated_systematics.get() and not self.ex_orbtable.get().strip():
            warning.append("Saturated-target systematics correction requires an orbital table.")
        if self.ex_saturated_systematics.get() and saturation_optimized:
            warning.append("Saturation-optimized extraction bypasses the separate saturated-target correction.")
        if self.ex_full_region_sum.get() and saturation_optimized:
            warning.append("Saturation-optimized extraction bypasses the full-region sum setting.")
        if self.ex_gaia_region_sum.get() and saturation_optimized:
            warning.append("Saturation-optimized extraction bypasses the Gaia-region sum setting.")
        if self.ex_external_mask_file.get().strip() and saturation_optimized:
            warning.append("Saturation-optimized extraction bypasses the external aperture mask.")
        if external_mask_selected and self.ex_full_region_sum.get():
            warning.append("External aperture mask and full-region sum are alternative aperture definitions; select only one.")
        if external_mask_selected and self.ex_gaia_region_sum.get():
            warning.append("External aperture mask and Gaia-region sum are alternative aperture definitions; select only one.")
        if self.ex_prf_photometry.get() and self.ex_prf_backend.get() == "auto":
            warning.append("PRF auto mode prefers lkprf; install it in the active environment for official TESS PRFs.")
        if self.ex_prf_photometry.get() and self.ex_input_mode.get() != "directory":
            selected_name = Path(self.ex_single_file.get().strip()).name.lower()
            if selected_name.startswith(("kplr", "ktwo")) or "lpd-targ" in selected_name or "spd-targ" in selected_name:
                warning.append("Kepler/K2 file selected: PRF extraction will be skipped; aperture extraction will still run.")
        if self.ex_prf_photometry.get() and self.ex_prf_scene_mode.get() == "gaia" and self.ex_no_gaia.get():
            warning.append("Gaia PRF scene mode requires Gaia; current No Gaia setting will force a single-source fallback.")
        if self.ex_prf_photometry.get() and self.ex_prf_source_output.get() == "all" and self.ex_prf_scene_mode.get() != "gaia":
            warning.append("All-source PRF output requires Gaia scene mode; single mode will write only the primary source.")
        if self.ex_prf_photometry.get() and self.ex_prf_scene_mode.get() == "gaia":
            script_name = Path(self.extractor_script.get().strip()).name.lower()
            if "simple" in script_name and "watershed" not in script_name:
                warning.append("Gaia PRF scene mode is implemented in the watershed extractor; the simple extractor will fall back to single-source PRF fitting.")
        if self.ex_prf_photometry.get() and self.ex_prf_source_output.get() == "all":
            script_name = Path(self.extractor_script.get().strip()).name.lower()
            if "simple" in script_name and "watershed" not in script_name:
                warning.append("All-source PRF output is available only in the watershed extractor; the simple extractor will write primary sources only.")
        warning_text = "  ".join(warning)
        self.ex_warning_label.configure(text=warning_text)
        self.ex_basic_warning_label.configure(text=warning_text)

        prf_enabled = bool(self.ex_prf_photometry.get()) and not saturation_optimized
        self.ex_prf_basic_check.configure(state=("disabled" if saturation_optimized else "normal"))
        self.ex_prf_basic_scene_combo.configure(state=("readonly" if prf_enabled else "disabled"))
        for widget in getattr(self, "ex_prf_widgets", []):
            try:
                if widget in (getattr(self, "ex_prf_backend_combo", None),
                              getattr(self, "ex_prf_motion_combo", None),
                              getattr(self, "ex_prf_scene_combo", None),
                              getattr(self, "ex_prf_neighbor_treatment_combo", None),
                              getattr(self, "ex_prf_source_output_combo", None),
                              getattr(self, "ex_prf_background_combo", None)):
                    widget.configure(state=("readonly" if prf_enabled else "disabled"))
                else:
                    widget.configure(state=("normal" if prf_enabled else "disabled"))
            except Exception:
                pass
        self._update_extractor_command_preview()

    def _update_detrender_state(self, event=None):
        skip = self.dt_skip_xybg.get()
        pchip = self.dt_use_pchip.get()
        use_quat = self.dt_use_quaternion_regression.get()
        self.dt_chk_background.configure(state=("disabled" if skip else "normal"))
        try:
            self.dt_pchip_entry.configure(state=("normal" if pchip else "disabled"))
        except Exception:
            pass

        # Protect intrinsic variability by default, but only when quaternion
        # regression transitions from off to on. After that the user may
        # uncheck Pre-model PCHIP and it will remain off.
        was_quat = bool(getattr(self, "_quaternion_was_enabled", False))
        if use_quat and not was_quat and not self.dt_pre_model_pchip.get():
            self.dt_pre_model_pchip.set(True)
        self._quaternion_was_enabled = bool(use_quat)

        for widget in getattr(self, "dt_quaternion_widgets", []):
            try:
                if widget is self.dt_quaternion_camera_combo:
                    widget.configure(state=("readonly" if use_quat else "disabled"))
                else:
                    widget.configure(state=("normal" if use_quat else "disabled"))
            except Exception:
                pass
        cache_enabled = use_quat and bool(self.dt_quaternion_auto_download.get())
        for widget in getattr(self, "dt_quaternion_cache_widgets", []):
            try:
                widget.configure(state=("normal" if cache_enabled else "disabled"))
            except Exception:
                pass
        self._update_detrender_command_preview()

    def _validate_extractor_values(self) -> None:
        """Validate GUI values before constructing an extractor command."""
        approach = self.ex_extraction_approach.get()
        if approach not in {STANDARD_APPROACH, SATURATED_APPROACH}:
            raise ValueError(f"Unknown extraction approach: {approach!r}")
        if int(self.ex_n_targets.get()) < 1:
            raise ValueError("N targets must be at least 1.")
        if approach == SATURATED_APPROACH and int(self.ex_n_targets.get()) != 1:
            raise ValueError("Saturation-optimized aperture extraction requires exactly one target.")
        if self.ex_no_gaia.get() and int(self.ex_n_targets.get()) != 1:
            raise ValueError("No-Gaia extraction requires exactly one target.")
        if self.ex_saturated_systematics.get() and not self.ex_orbtable.get().strip():
            raise ValueError("Saturated-target systematics correction requires an orbital table.")
        external_mask = self.ex_external_mask_file.get().strip()
        if external_mask:
            if int(self.ex_n_targets.get()) != 1:
                raise ValueError("External aperture masks require exactly one target.")
            if not Path(external_mask).expanduser().is_file():
                raise ValueError(f"External aperture mask file not found: {external_mask}")
            if self.ex_full_region_sum.get() or self.ex_gaia_region_sum.get():
                raise ValueError(
                    "External aperture mask cannot be combined with full-region or Gaia-region summation."
                )
        if self.ex_saturated_systematics.get():
            orbital_table = self.ex_orbtable.get().strip()
            if not Path(orbital_table).expanduser().is_file():
                raise ValueError(f"Orbital table file not found: {orbital_table}")
        q_lo = float(self.ex_amp_q_lo.get())
        q_hi = float(self.ex_amp_q_hi.get())
        if not (0.0 <= q_lo < q_hi <= 100.0):
            raise ValueError("Amplitude percentiles must satisfy 0 <= low < high <= 100.")

    def _validate_detrender_values(self) -> None:
        """Validate GUI values before constructing a detrender command."""
        if int(self.dt_robust_iters.get()) < 1 or float(self.dt_huber_k.get()) <= 0:
            raise ValueError("Robust iterations and Huber k must be positive.")
        if self.dt_use_pchip.get() and float(self.dt_pchip_knot_spacing.get()) <= 0:
            raise ValueError("PCHIP knot spacing must be positive.")
        if self.dt_pre_model_pchip.get() and float(self.dt_pre_model_bin_days.get()) <= 0:
            raise ValueError("Pre-model PCHIP bin size must be positive.")
        if self.dt_apply_orbital_phase_template.get():
            orbital_table = self.dt_orbtable.get().strip()
            if not orbital_table:
                raise ValueError("Orbital phase-template correction requires an orbital table.")
            if not Path(orbital_table).expanduser().is_file():
                raise ValueError(f"Orbital table file not found: {orbital_table}")

    def _append_prf_extractor_args(self, cmd: list[str]) -> None:
        if not self.ex_prf_photometry.get():
            return
        cmd.append("--prf-photometry")
        cmd += ["--prf-backend", self.ex_prf_backend.get().strip() or "auto"]
        cmd += ["--prf-motion-source", self.ex_prf_motion_source.get().strip() or "auto"]
        cmd += ["--prf-scene-mode", self.ex_prf_scene_mode.get().strip() or "single"]
        cmd += ["--prf-neighbor-treatment", self.ex_prf_neighbor_treatment.get().strip() or "fixed"]
        cmd += ["--prf-source-output", self.ex_prf_source_output.get().strip() or "primary"]
        cmd += ["--prf-neighbor-dmag", str(float(self.ex_prf_neighbor_dmag.get()))]
        cmd += ["--prf-neighbor-margin", str(float(self.ex_prf_neighbor_margin.get()))]
        cmd += ["--prf-max-scene-sources", str(int(self.ex_prf_max_scene_sources.get()))]
        cmd += ["--prf-min-neighbor-fraction", str(float(self.ex_prf_min_neighbor_fraction.get()))]
        cmd += ["--prf-background", self.ex_prf_background.get().strip() or "plane"]
        cmd += ["--prf-min-weight", str(float(self.ex_prf_min_weight.get()))]
        cmd += ["--prf-fit-radius", str(float(self.ex_prf_fit_radius.get()))]
        cmd += ["--prf-shift-quantization", str(float(self.ex_prf_shift_quantization.get()))]
        cmd += ["--prf-max-shift", str(float(self.ex_prf_max_shift.get()))]
        if not self.ex_prf_save_diagnostics.get():
            cmd.append("--prf-no-diagnostics")
        if not self.ex_prf_allow_gaussian_fallback.get():
            cmd.append("--prf-no-gaussian-fallback")

    def build_extractor_command(self) -> list[str]:
        self._validate_extractor_values()
        script = self.extractor_script.get().strip()
        if not script:
            raise ValueError("Extractor script path is empty.")
        cmd = [sys.executable, "-u", script]
        script_name = Path(script).name.lower()
        is_simple = ("simple" in script_name) and ("watershed" not in script_name)
        saturation_optimized = self.ex_extraction_approach.get() == SATURATED_APPROACH

        if is_simple:
            if self.ex_extraction_approach.get() == SATURATED_APPROACH:
                raise ValueError(
                    "Saturation-optimized aperture extraction is available in "
                    "tess_watershed_extractor.py, not the simple extractor."
                )
            if self.ex_input_mode.get() == "directory":
                root = self.ex_tpf_dir.get().strip() or "."
                # Let the simple extractor perform mission-aware discovery;
                # constructing a TESS-only *_tp.fits glob here used to omit
                # Kepler/K2 and compressed/TESSCut products.
                cmd += ["--input", root]
                if self.ex_recursive.get():
                    cmd.append("--recursive")
            else:
                single = self.ex_single_file.get().strip()
                if not single:
                    raise ValueError("Single-file mode is selected but no FITS file is set.")
                cmd += ["--input", single]
            cmd += ["--outdir", self.ex_output_root.get().strip() or "LC_products_multi"]
            cmd += ["--n-targets", str(int(self.ex_n_targets.get()))]
            cmd += ["--aperture-mode", self.ex_simple_aperture_mode.get().strip() or "auto"]
            self._append_prf_extractor_args(cmd)
            return cmd

        if self.ex_input_mode.get() == "directory":
            cmd += ["--tpf-dir", self.ex_tpf_dir.get().strip() or "."]
            if self.ex_recursive.get():
                cmd.append("--recursive")
        else:
            single = self.ex_single_file.get().strip()
            if not single:
                raise ValueError("Single-file mode is selected but no FITS file is set.")
            cmd += ["--single", single]

        cmd += ["--output-root", self.ex_output_root.get().strip() or "LC_products_multi"]
        cmd += ["--n-targets", str(int(self.ex_n_targets.get()))]
        if not saturation_optimized and not (
            self.ex_full_region_sum.get()
            or self.ex_gaia_region_sum.get()
            or self.ex_external_mask_file.get().strip()
        ):
            cmd += ["--method", self.ex_method.get()]
        cmd += ["--gaia-radius-arcmin", str(float(self.ex_gaia_radius.get()))]

        if self.ex_no_gaia.get():
            cmd.append("--no-gaia")
        if self.ex_gaia_fallback.get():
            cmd.append("--gaia-fallback")
        if self.ex_no_quality0.get():
            cmd.append("--no-quality0")
        if self.ex_full_region_sum.get() and not saturation_optimized:
            cmd.append("--full-region-sum")
        if self.ex_gaia_region_sum.get() and not saturation_optimized:
            cmd.append("--gaia-region-sum")
        if self.ex_external_mask_file.get().strip() and not saturation_optimized:
            cmd += ["--external-mask-file", self.ex_external_mask_file.get().strip()]
        if not self.ex_save_aperture_plots.get():
            cmd.append("--no-aperture-plots")
        if self.ex_save_pickled_figures.get():
            cmd.append("--save-figure-pickles")
        if self.ex_saturated_systematics.get() and not saturation_optimized:
            cmd.append("--saturated-systematics-correction")
            if self.ex_orbtable.get().strip():
                cmd += ["--orbtable", self.ex_orbtable.get().strip()]
        if saturation_optimized:
            cmd.append("--saturation-optimized-aperture")

        cmd += ["--min-pixels", str(int(self.ex_min_pixels.get()))]
        cmd += ["--amp-q-lo", str(float(self.ex_amp_q_lo.get()))]
        cmd += ["--amp-q-hi", str(float(self.ex_amp_q_hi.get()))]
        cmd += ["--amp-min-frac", str(float(self.ex_amp_min_frac.get()))]
        cmd += ["--aperture-fom", self.ex_aperture_fom.get().strip() or "stddiff"]
        cmd += ["--max-radius-pix", str(self.ex_max_radius_pix.get()).strip() or "inf"]
        cmd += ["--max-components", str(int(self.ex_max_components.get()))]
        cmd += ["--min-seed-frac-of-peak", str(float(self.ex_min_seed_frac.get()))]
        cmd += ["--min-new-pixels-per-component", str(int(self.ex_min_new_pixels.get()))]
        cmd += ["--core-npix", str(int(self.ex_core_npix.get()))]
        cmd += ["--core-min-frac-of-peak", str(float(self.ex_core_min_frac.get()))]
        cmd += ["--sat-thresh", str(float(self.ex_sat_thresh.get()))]
        cmd += ["--sat-min-npix", str(int(self.ex_sat_min_npix.get()))]
        cmd += ["--back-nfaint", str(int(self.ex_back_nfaint.get()))]
        cmd += ["--phase-bin", str(float(self.ex_phase_bin.get()))]
        cmd += ["--saturated-aperture-threshold", str(float(self.ex_saturated_aperture_threshold.get()))]
        if not saturation_optimized:
            self._append_prf_extractor_args(cmd)
        return cmd

    def build_detrender_command(self) -> list[str]:
        self._validate_detrender_values()
        script = self.detrender_script.get().strip()
        if not script:
            raise ValueError("Detrender script path is empty.")
        cmd = [sys.executable, "-u", script]
        cmd += ["--lightcurve-dir", self.dt_lightcurve_dir.get().strip() or "LC_products_multi"]
        cmd += ["--diagnostics-dir", self._resolved_diagnostics_dir()]
        cmd += ["--output-dir", self.dt_output_dir.get().strip() or "LC_products_multi"]
        cmd += ["--pattern", self.dt_pattern.get().strip() or "*.csv"]
        cmd += ["--prefix", self.dt_prefix.get().strip() or "detrended_"]
        if self.dt_recursive.get():
            cmd.append("--recursive")
        if self.dt_skip_existing.get():
            cmd.append("--skip-existing")
        if self.dt_use_background.get() and not self.dt_skip_xybg.get():
            cmd.append("--use-background")
        if self.dt_use_pchip.get():
            cmd.append("--use-pchip-highpass")
        if self.dt_pre_model_pchip.get():
            cmd.append("--pre-model-pchip")
        if self.dt_skip_xybg.get():
            cmd.append("--skip-xybg-decorrelation")
        if not self.dt_combine_sectors.get():
            cmd.append("--no-combine-sectors")
        cmd += ["--knot-spacing-days", str(self.dt_knot_spacing.get()).strip() or "inf"]
        cmd += ["--robust-iters", str(int(self.dt_robust_iters.get()))]
        cmd += ["--huber-k", str(float(self.dt_huber_k.get()))]
        cmd += ["--pchip-knot-spacing", str(float(self.dt_pchip_knot_spacing.get()))]
        cmd += ["--pre-model-bin-days", str(float(self.dt_pre_model_bin_days.get()))]
        cmd += ["--pre-model-stat", self.dt_pre_model_stat.get().strip() or "median"]
        cmd += ["--pre-model-min-points", str(int(self.dt_pre_model_min_points.get()))]
        cmd += ["--pre-model-sigma-clip", str(float(self.dt_pre_model_sigma_clip.get()))]
        cmd += ["--pre-model-sigma-iters", str(int(self.dt_pre_model_sigma_iters.get()))]
        if self.dt_clip_residuals_before_detrend.get():
            cmd.append("--clip-residuals-before-detrend")
        cmd += ["--clip-residuals-sigma", str(float(self.dt_clip_residuals_sigma.get()))]
        cmd += ["--clip-residuals-iters", str(int(self.dt_clip_residuals_iters.get()))]
        if self.dt_save_pickled_figures.get():
            cmd.append("--save-figure-pickles")
        if self.dt_apply_orbital_phase_template.get():
            cmd.append("--apply-orbital-phase-template")
            if self.dt_orbtable.get().strip():
                cmd += ["--orbtable", self.dt_orbtable.get().strip()]
            cmd += ["--phase-bin", str(float(self.dt_phase_bin.get()))]
        if self.dt_use_quaternion_regression.get():
            source = self.dt_quaternion_source.get().strip()
            auto_download = bool(self.dt_quaternion_auto_download.get())
            if not source and not auto_download:
                raise ValueError(
                    "Quaternion regression needs an explicit source or Auto-download missing TESSVectors."
                )
            cmd.append("--use-quaternion-regression")
            if source:
                cmd += ["--quaternion-source", source]
            if auto_download:
                cache_dir = self.dt_tessvectors_cache_dir.get().strip()
                if not cache_dir:
                    raise ValueError("TESSVectors cache directory is empty.")
                cmd.append("--quaternion-auto-download")
                cmd += ["--tessvectors-cache-dir", cache_dir]
            cmd += ["--quaternion-camera", self.dt_quaternion_camera.get().strip() or "auto"]
            cmd += ["--quaternion-min-samples", str(int(self.dt_quaternion_min_samples.get()))]
            cmd += ["--quaternion-clip-sigma", str(float(self.dt_quaternion_clip_sigma.get()))]
            cmd += ["--quaternion-clip-iters", str(int(self.dt_quaternion_clip_iters.get()))]
            cmd += ["--quaternion-time-offset-days", self.dt_quaternion_time_offset.get().strip() or "auto"]
            if self.dt_save_quaternion_diagnostics.get():
                cmd.append("--save-quaternion-diagnostics")
        return cmd

    def _update_extractor_command_preview(self):
        try:
            self.ex_command_preview.set(quote_cmd(self.build_extractor_command()))
        except Exception as exc:
            self.ex_command_preview.set(f"<invalid: {exc}>")

    def _update_detrender_command_preview(self):
        try:
            self.dt_command_preview.set(quote_cmd(self.build_detrender_command()))
        except Exception as exc:
            self.dt_command_preview.set(f"<invalid: {exc}>")

    def run_extractor(self):
        try:
            cmd = self.build_extractor_command()
        except Exception as exc:
            messagebox.showerror("Invalid extractor command", str(exc))
            return
        self._run_subprocess("Extractor", cmd, Path(self.ex_output_root.get().strip() or "."))

    def run_detrender(self):
        try:
            cmd = self.build_detrender_command()
        except Exception as exc:
            messagebox.showerror("Invalid detrender command", str(exc))
            return
        self._run_subprocess("Detrender", cmd, Path(self.dt_output_dir.get().strip() or "."))


    def _clear_current_process_state(self, status: str = "Ready."):
        self.current_process = None
        self.current_job_name = None

        self.status_text.set(status)

    def _reap_stale_process_if_needed(self) -> bool:
        proc = self.current_process
        if proc is None:
            return False
        try:
            rc = proc.poll()
        except Exception:
            rc = None
        if rc is None:
            return False
        self.log_text.insert("end", f"\n[INFO] Cleared stale finished process (exit code {rc}).\n")
        self.log_text.see("end")
        self._clear_current_process_state("Ready.")
        return True

    def _terminate_process_tree(self, proc: subprocess.Popen, force: bool = False):
        if os.name == "nt":
            cmd = ["taskkill", "/PID", str(proc.pid), "/T"]
            if force:
                cmd.append("/F")
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
            return
        try:
            pgid = os.getpgid(proc.pid)
        except Exception:
            pgid = None
        sig = signal.SIGKILL if force else signal.SIGTERM
        if pgid is not None:
            try:
                os.killpg(pgid, sig)
                return
            except Exception:
                pass
        try:
            proc.kill() if force else proc.terminate()
        except Exception:
            pass

    def _stop_process_worker(self, proc: subprocess.Popen):
        try:
            self._terminate_process_tree(proc, force=False)
            try:
                proc.wait(timeout=3.0)
            except Exception:
                self._terminate_process_tree(proc, force=True)
                try:
                    proc.wait(timeout=2.0)
                except Exception:
                    pass
        finally:
            if self.current_process is proc:
                try:
                    rc = proc.poll()
                except Exception:
                    rc = None
                self.log_queue.put(("line", proc, f"Process stop requested; final exit code {rc}.\n"))

    def _run_subprocess(self, job_name: str, cmd: list[str], output_dir: Path):
        self._reap_stale_process_if_needed()
        if self.current_process is not None:
            messagebox.showwarning("Job already running", "Stop the current process before starting another one.")
            return

        self.current_job_name = job_name
        self.current_output_dir = output_dir
        self.preview_dir.set(str(output_dir))
        self.status_text.set(f"{job_name} running...")
        self.notebook.select(self.tab_run)

        self.log_text.insert("end", f"\n=== {job_name} ===\n")
        self.log_text.insert("end", quote_cmd(cmd) + "\n\n")
        self.log_text.see("end")

        try:
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            popen_kwargs = dict(
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=env,
            )
            if os.name == "nt":
                popen_kwargs["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
            else:
                popen_kwargs["start_new_session"] = True
            self.current_process = subprocess.Popen(cmd, **popen_kwargs)
        except Exception as exc:
            self._clear_current_process_state("Ready.")
            messagebox.showerror("Failed to start process", str(exc))
            return

        proc = self.current_process

        def reader_thread(proc=proc):
            try:
                if proc and proc.stdout is not None:
                    for line in proc.stdout:
                        self.log_queue.put(("line", proc, line))
                rc = proc.wait() if proc else -1
                self.log_queue.put(("done", proc, f"{job_name} finished with exit code {rc}.\n"))
            except Exception as exc:
                self.log_queue.put(("done", proc, f"{job_name} failed: {exc}\n"))

        threading.Thread(target=reader_thread, daemon=True).start()

    def stop_current_process(self):
        self._reap_stale_process_if_needed()
        proc = self.current_process
        if proc is None:
            return
        try:
            self.status_text.set("Stopping process tree...")
            threading.Thread(target=self._stop_process_worker, args=(proc,), daemon=True).start()
        except Exception as exc:
            messagebox.showerror("Stop failed", str(exc))

    def _poll_log_queue(self):
        try:
            while True:
                kind, event_proc, payload = self.log_queue.get_nowait()
                if kind == "line":
                    self.log_text.insert("end", payload)
                    self.log_text.see("end")
                elif kind == "done":
                    self.log_text.insert("end", "\n" + payload + "\n")
                    self.log_text.see("end")
                    if self.current_process is event_proc:
                        self._clear_current_process_state("Ready.")
                        self._refresh_preview_list()
        except queue.Empty:
            pass

        proc = self.current_process
        if proc is not None:
            try:
                rc = proc.poll()
            except Exception:
                rc = None
            if rc is not None:
                self.log_text.insert("end", f"\n[INFO] Detected exited process without clean completion (exit code {rc}).\n")
                self.log_text.see("end")
                self._clear_current_process_state("Ready.")
                self._refresh_preview_list()

        self.after(150, self._poll_log_queue)

    def clear_log(self):
        self.log_text.delete("1.0", "end")

    def save_log(self):
        path = filedialog.asksaveasfilename(
            title="Save log",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        if not path:
            return
        Path(path).write_text(self.log_text.get("1.0", "end"), encoding="utf-8")

    def copy_active_command(self):
        txt = ""
        current = self.notebook.select()
        if current == str(self.tab_extractor):
            txt = self.ex_command_preview.get()
        elif current == str(self.tab_detrender):
            txt = self.dt_command_preview.get()
        else:
            txt = self.ex_command_preview.get() or self.dt_command_preview.get()
        if txt:
            self.clipboard_clear()
            self.clipboard_append(txt)

    def _on_preview_mousewheel(self, event):
        try:
            if getattr(event, "num", None) == 4:
                steps = -1
            elif getattr(event, "num", None) == 5:
                steps = 1
            else:
                steps = -1 if int(getattr(event, "delta", 0)) > 0 else 1
            self.preview_canvas.yview_scroll(steps, "units")
        except Exception:
            pass

    def _on_preview_shift_mousewheel(self, event):
        try:
            self.preview_canvas.xview_scroll(int(-1 * (event.delta / 120)), "units")
        except Exception:
            pass

    def _on_preview_zoom_mousewheel(self, event):
        try:
            steps = 1 if event.delta > 0 else -1
            self._change_preview_zoom(steps)
        except Exception:
            pass

    def _refresh_preview_list(self):
        self.preview_list.delete(0, "end")
        path_str = self.preview_dir.get().strip()
        if not path_str and self.current_output_dir is not None:
            self.preview_dir.set(str(self.current_output_dir))
            path_str = str(self.current_output_dir)
        if not path_str:
            self.preview_panel.configure(text="No preview folder selected.", image="")
            return
        p = Path(path_str).expanduser()
        if not p.exists():
            self.preview_panel.configure(text=f"Preview folder not found:\n{p}", image="")
            return
        pngs = sorted(p.glob("*.png"))
        self._preview_paths = pngs
        for item in pngs:
            self.preview_list.insert("end", item.name)
        if pngs:
            self.preview_list.selection_set(0)
            self._show_preview(pngs[0])
        else:
            self.preview_panel.configure(text="No PNG files found in preview folder.", image="")

    def _on_preview_select(self, event=None):
        sel = self.preview_list.curselection()
        if not sel:
            return
        idx = int(sel[0])
        if 0 <= idx < len(getattr(self, "_preview_paths", [])):
            self._show_preview(self._preview_paths[idx])

    def _on_preview_canvas_configure(self, event=None):
        self.preview_canvas.configure(scrollregion=self.preview_canvas.bbox("all"))
        if self.preview_scale_mode.get().strip().lower() == "fit":
            self._render_current_preview()

    def _show_preview(self, path: Path):
        try:
            # Load pixels while the file handle is inside a context manager so
            # Windows can immediately overwrite/delete a preview on the next run.
            with Image.open(path) as source:
                self.preview_source_image = source.convert("RGBA")
            self.preview_source_path = path
            self._render_current_preview()
        except Exception as exc:
            self.preview_source_image = None
            self.preview_source_path = path
            self.preview_image = None
            self.preview_panel.configure(image="", text=f"Could not preview:\n{path.name}\n\n{exc}")
            self.preview_canvas.update_idletasks()
            self.preview_canvas.configure(scrollregion=self.preview_canvas.bbox("all"))

    def _settings_dict(self) -> dict:
        return {
            "settings_schema_version": SETTINGS_SCHEMA_VERSION,
            "basic_preset_version": BASIC_PRESET_VERSION,
            "interface_mode": self.interface_mode.get(),
            "extractor_script": self.extractor_script.get(),
            "detrender_script": self.detrender_script.get(),
            "ex_input_mode": self.ex_input_mode.get(),
            "ex_tpf_dir": self.ex_tpf_dir.get(),
            "ex_single_file": self.ex_single_file.get(),
            "ex_recursive": self.ex_recursive.get(),
            "ex_output_root": self.ex_output_root.get(),
            "ex_n_targets": self.ex_n_targets.get(),
            "ex_method": self.ex_method.get(),
            "ex_gaia_radius": self.ex_gaia_radius.get(),
            "ex_no_gaia": self.ex_no_gaia.get(),
            "ex_gaia_fallback": self.ex_gaia_fallback.get(),
            "ex_no_quality0": self.ex_no_quality0.get(),
            "ex_full_region_sum": self.ex_full_region_sum.get(),
            "ex_gaia_region_sum": self.ex_gaia_region_sum.get(),
            "ex_save_pickled_figures": self.ex_save_pickled_figures.get(),
            "ex_external_mask_file": self.ex_external_mask_file.get(),
            "ex_aperture_fom": self.ex_aperture_fom.get(),
            "ex_simple_aperture_mode": self.ex_simple_aperture_mode.get(),
            "ex_save_aperture_plots": self.ex_save_aperture_plots.get(),
            "ex_saturated_systematics": self.ex_saturated_systematics.get(),
            "ex_extraction_approach": self.ex_extraction_approach.get(),
            "ex_orbtable": self.ex_orbtable.get(),
            "ex_min_pixels": self.ex_min_pixels.get(),
            "ex_amp_q_lo": self.ex_amp_q_lo.get(),
            "ex_amp_q_hi": self.ex_amp_q_hi.get(),
            "ex_amp_min_frac": self.ex_amp_min_frac.get(),
            "ex_max_radius_pix": self.ex_max_radius_pix.get(),
            "ex_max_components": self.ex_max_components.get(),
            "ex_min_seed_frac": self.ex_min_seed_frac.get(),
            "ex_min_new_pixels": self.ex_min_new_pixels.get(),
            "ex_core_npix": self.ex_core_npix.get(),
            "ex_core_min_frac": self.ex_core_min_frac.get(),
            "ex_sat_thresh": self.ex_sat_thresh.get(),
            "ex_sat_min_npix": self.ex_sat_min_npix.get(),
            "ex_back_nfaint": self.ex_back_nfaint.get(),
            "ex_phase_bin": self.ex_phase_bin.get(),
            "ex_saturated_aperture_threshold": self.ex_saturated_aperture_threshold.get(),
            "ex_prf_photometry": self.ex_prf_photometry.get(),
            "ex_prf_backend": self.ex_prf_backend.get(),
            "ex_prf_motion_source": self.ex_prf_motion_source.get(),
            "ex_prf_scene_mode": self.ex_prf_scene_mode.get(),
            "ex_prf_neighbor_treatment": self.ex_prf_neighbor_treatment.get(),
            "ex_prf_source_output": self.ex_prf_source_output.get(),
            "ex_prf_neighbor_dmag": self.ex_prf_neighbor_dmag.get(),
            "ex_prf_neighbor_margin": self.ex_prf_neighbor_margin.get(),
            "ex_prf_max_scene_sources": self.ex_prf_max_scene_sources.get(),
            "ex_prf_min_neighbor_fraction": self.ex_prf_min_neighbor_fraction.get(),
            "ex_prf_background": self.ex_prf_background.get(),
            "ex_prf_min_weight": self.ex_prf_min_weight.get(),
            "ex_prf_fit_radius": self.ex_prf_fit_radius.get(),
            "ex_prf_shift_quantization": self.ex_prf_shift_quantization.get(),
            "ex_prf_max_shift": self.ex_prf_max_shift.get(),
            "ex_prf_save_diagnostics": self.ex_prf_save_diagnostics.get(),
            "ex_prf_allow_gaussian_fallback": self.ex_prf_allow_gaussian_fallback.get(),
            "dt_lightcurve_dir": self.dt_lightcurve_dir.get(),
            "dt_diagnostics_dir": self._resolved_diagnostics_dir(),
            "dt_output_dir": self.dt_output_dir.get(),
            "dt_pattern": self.dt_pattern.get(),
            "dt_recursive": self.dt_recursive.get(),
            "dt_prefix": self.dt_prefix.get(),
            "dt_skip_existing": self.dt_skip_existing.get(),
            "dt_use_background": self.dt_use_background.get(),
            "dt_use_pchip": self.dt_use_pchip.get(),
            "dt_skip_xybg": self.dt_skip_xybg.get(),
            "dt_combine_sectors": self.dt_combine_sectors.get(),
            "dt_knot_spacing": self.dt_knot_spacing.get(),
            "dt_robust_iters": self.dt_robust_iters.get(),
            "dt_huber_k": self.dt_huber_k.get(),
            "dt_pchip_knot_spacing": self.dt_pchip_knot_spacing.get(),
            "dt_pre_model_pchip": self.dt_pre_model_pchip.get(),
            "dt_pre_model_bin_days": self.dt_pre_model_bin_days.get(),
            "dt_pre_model_stat": self.dt_pre_model_stat.get(),
            "dt_pre_model_min_points": self.dt_pre_model_min_points.get(),
            "dt_pre_model_sigma_clip": self.dt_pre_model_sigma_clip.get(),
            "dt_pre_model_sigma_iters": self.dt_pre_model_sigma_iters.get(),
            "dt_clip_residuals_before_detrend": self.dt_clip_residuals_before_detrend.get(),
            "dt_clip_residuals_sigma": self.dt_clip_residuals_sigma.get(),
            "dt_clip_residuals_iters": self.dt_clip_residuals_iters.get(),
            "dt_save_pickled_figures": self.dt_save_pickled_figures.get(),
            "dt_apply_orbital_phase_template": self.dt_apply_orbital_phase_template.get(),
            "dt_orbtable": self.dt_orbtable.get(),
            "dt_phase_bin": self.dt_phase_bin.get(),
            "dt_use_quaternion_regression": self.dt_use_quaternion_regression.get(),
            "dt_quaternion_source": self.dt_quaternion_source.get(),
            "dt_quaternion_auto_download": self.dt_quaternion_auto_download.get(),
            "dt_tessvectors_cache_dir": self.dt_tessvectors_cache_dir.get(),
            "dt_quaternion_camera": self.dt_quaternion_camera.get(),
            "dt_quaternion_min_samples": self.dt_quaternion_min_samples.get(),
            "dt_quaternion_clip_sigma": self.dt_quaternion_clip_sigma.get(),
            "dt_quaternion_clip_iters": self.dt_quaternion_clip_iters.get(),
            "dt_quaternion_time_offset": self.dt_quaternion_time_offset.get(),
            "dt_save_quaternion_diagnostics": self.dt_save_quaternion_diagnostics.get(),
            "preview_dir": self.preview_dir.get(),
            "preview_scale_mode": self.preview_scale_mode.get(),
            "preview_zoom": self.preview_zoom.get(),
        }

    def save_settings_json(self):
        path = filedialog.asksaveasfilename(
            title="Save settings",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if not path:
            return
        try:
            Path(path).write_text(
                json.dumps(self._settings_dict(), indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            self.status_text.set(f"Saved settings: {Path(path).name}")
        except Exception as exc:
            messagebox.showerror("Could not save settings", str(exc))

    def load_settings_json(self):
        path = filedialog.askopenfilename(
            title="Load settings",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if not path:
            return
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                raise ValueError("The settings file must contain a JSON object.")
        except Exception as exc:
            messagebox.showerror("Could not load settings", str(exc))
            return

        # Old files are promoted to Expert only when they contain a customized
        # value; all-default files remain pleasantly compact when reopened.
        migrated, requested_mode = self._migrate_settings_data(data)

        for key, value in migrated.items():
            if hasattr(self, key):
                var = getattr(self, key)
                try:
                    var.set(value)
                except Exception:
                    pass

        self.interface_mode.set(requested_mode)
        if requested_mode == "basic":
            self._reset_expert_defaults()
        self._last_interface_mode = requested_mode

        # Preserve an explicitly saved Pre-model PCHIP choice. For older
        # settings files that do not contain that key, retain the default-on
        # behavior when quaternion regression is enabled.
        if "dt_pre_model_pchip" in migrated:
            self._quaternion_was_enabled = bool(self.dt_use_quaternion_regression.get())
        else:
            self._quaternion_was_enabled = False

        self._update_extractor_state()
        self._update_detrender_state()
        self._apply_interface_mode()
        self._refresh_preview_list()
        self.status_text.set(f"Loaded settings: {Path(path).name} ({requested_mode.title()} mode)")

    def _browse_dir(self, var: tk.Variable, refresh: bool = False):
        path = filedialog.askdirectory()
        if path:
            var.set(path)
            if refresh:
                self._refresh_preview_list()

    def _browse_file(self, var: tk.Variable, filetypes):
        path = filedialog.askopenfilename(filetypes=filetypes)
        if path:
            var.set(path)

    def _open_folder(self, path: Path):
        path = path.expanduser().resolve()
        if not path.exists():
            messagebox.showwarning("Folder not found", f"Folder does not exist:\n{path}")
            return
        try:
            if sys.platform.startswith("linux"):
                subprocess.Popen(["xdg-open", str(path)])
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(path)])
            elif os.name == "nt":
                os.startfile(str(path))  # type: ignore[attr-defined]
            else:
                messagebox.showinfo("Folder", str(path))
        except Exception as exc:
            messagebox.showerror("Open folder failed", str(exc))


def attach_menu(app: TESSGui):
    menubar = tk.Menu(app)
    filemenu = tk.Menu(menubar, tearoff=0)
    filemenu.add_command(label="Load settings...", command=app.load_settings_json)
    filemenu.add_command(label="Save settings...", command=app.save_settings_json)
    filemenu.add_separator()
    filemenu.add_command(label="Exit", command=app._on_close)
    menubar.add_cascade(label="File", menu=filemenu)

    helpmenu = tk.Menu(menubar, tearoff=0)
    helpmenu.add_command(label="Option reference", command=app.show_help_tab)
    helpmenu.add_command(label="Copy help text", command=app.copy_help_text)
    menubar.add_cascade(label="Help", menu=helpmenu)

    app.config(menu=menubar)


def main():
    app = TESSGui()
    attach_menu(app)
    app.mainloop()


if __name__ == "__main__":
    main()

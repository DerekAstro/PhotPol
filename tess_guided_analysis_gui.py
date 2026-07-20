#!/usr/bin/env python3
"""Cross-platform GUI for Guided and Joint TESS/polarimetry analysis.

The interface owns configuration, input conversion, process management, and
output previewing.  Scientific calculations remain in the two backend modules;
the GUI launches a small generated runner so a long analysis cannot freeze the
Tk event loop.
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
import tempfile
import threading
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from PIL import Image, ImageTk

APP_TITLE = "TESS Guided and Joint Analysis"
DEFAULT_WINDOW_WIDTH = 1220
DEFAULT_WINDOW_HEIGHT = 880
SCRIPT_DIR = Path(__file__).resolve().parent


def _wheel_scroll_units(event) -> int:
    """Normalize mouse-wheel events from Windows, macOS, and X11/Linux."""
    button = getattr(event, "num", None)
    if button == 4:
        return -1
    if button == 5:
        return 1
    delta = int(getattr(event, "delta", 0) or 0)
    if delta == 0:
        return 0
    # Windows commonly reports +/-120 while macOS often reports small values.
    magnitude = max(1, abs(delta) // 120)
    return -magnitude if delta > 0 else magnitude

HELP_TEXT = """
TESS Guided and Joint Analysis GUI
==================================

This GUI wraps the guided photometry + polarimetry analysis code and keeps the
same general style as the extractor/detrender GUI.

Main ideas
----------
• The GUI does not reimplement the analysis.
• It writes a temporary Python runner that imports your guided-analysis script,
  sets the requested configuration variables, and calls run_analysis().
• Output is streamed live into the Run / Preview log.

Analysis mode
-------------
Guided analysis
  Runs the current guided photometry + polarimetry workflow.

Joint analysis
  Runs the joint photometry + polarimetry search workflow using the
  companion script you provide.

  This mode uses the same general TESS-input controls and the same SPOC
  conversion option, but it has its own joint-search settings and runner path.

TESS input modes
----------------
Existing CSV
  Use an already-converted TESS CSV with columns the guided-analysis code can
  read (for example combined_filtered.csv or similar).

Pipeline/custom CSV directory
  Point at a directory of CSV light curves and let guided analysis load them
  using TESS_INPUT_MODE = 'pipeline_dir'.

Pipeline/custom CSV directory (batch one file at a time)
  Point at a directory of CSV light curves and run the guided analysis once per file.
  This is intended for photometry-only batch work over many CSVs.

SPOC lc.fits conversion
  Run spoc_lightcurve_converter.py first, then automatically feed the CSV it
  creates into guided or joint analysis. The input may be one light-curve FITS
  file, a directory, or a glob pattern, including paths containing spaces.
  Target-pixel files (*-tp.fits and *-fast-tp.fits) are not accepted here;
  process those through a photometric extractor first, or choose the matching
  *-lc.fits / *-fast-lc.fits product.

SPOC multi-sector conversion
----------------------------
The converter can process multiple sectors for one TIC. Each input product is
quality-filtered and normalized independently before combination. The default
normalization is a sigma-clipped median division, with flux errors divided by
the same normalization factor.

When more than one cadence product exists for the same TIC and sector, the
default cadence policy keeps the shortest cadence for the combined file. The
other available policies are longest and all. Per-sector CSVs may still be
written even when only one cadence is selected for the combined file.

The converter remains a standalone command-line program. The GUI calls its
normal switches directly and reads a JSON manifest afterward to identify the
combined CSV that should be sent into the analysis. If the selected inputs
contain more than one TIC, the GUI stops with a clear ambiguity message rather
than silently choosing a target.

Common guided-analysis options
------------------------------
POL product
  nm                  : night-mean-subtracted polarimetry
  resid_nm_pchip      : night-mean-subtracted and PCHIP-cleaned
  pw_resid_nm_pchip   : prewhitened residual product (only if explicitly wanted)

Inline plots
  Normally leave this off in GUI mode. Turning it on may create separate plot
  windows depending on the backend and script behavior.

Polarimetry-periodogram smoothing
---------------------------------
Optional Gaussian or boxcar smoothing is available in both Guided and Joint
analysis. Width is measured in independent polarimetry resolution elements
(1/T_pol), not computational grid samples. Gaussian width means FWHM; boxcar
width means the full box width. Smoothing is used only to select periodogram
candidates. Final frequencies, amplitudes, phases, and local SNR values are
fitted from the original unsmoothed time-series data.
If a kernel would be narrower than two samples on a particular frequency grid,
that grid is left unsmoothed rather than silently broadening the requested
width. Joint mode increases its local refinement sampling when needed so the
kernel is resolved during candidate refinement.

Basic and Expert modes
----------------------
Basic mode shows the controls needed for a typical run. Expert mode reveals
numerical search, baseline, diagnostic, script-path, and converter-tuning
controls. Switching modes does not reset hidden values, and saved settings
include both Basic and Expert controls.

Guided spectrum diagnostic markers
----------------------------------
Gray
  TESS template frequency.

Orange
  Maximum of the spectrum actually used to initialize the local fit (smoothed
  when smoothing was applied).

Green
  A local polarimetry fit that passed the detection threshold.

Magenta
  A marginal local fit admitted as a seed for the simultaneous global fit.

Purple
  The retained frequency from the final simultaneous global fit.

Rejected local fits are identified in the panel title and do not receive a
green detection marker. Each panel shows the exact sequential residual
spectrum used at that search iteration.

Run / Preview
-------------
The preview tab scans the chosen output root recursively for PNG files.

Polarimetry grouping modes
--------------------------
gap
  Start a new baseline group whenever the time gap exceeds Gap hours.

integer_jd
  Group all measurements with the same integer Julian Date.

run
  Group subruns such as 12A, 12B, and 12C under their parent run 12.
  The polarimetry input must contain a recognized run/subrun label column.

subrun
  Keep complete labels such as 12A, 12B, and 12C as separate baseline groups.
  The polarimetry input must contain a recognized run/subrun label column.

TESS uncertainty weighting
--------------------------
none
  Ignore TESS uncertainty columns and give every retained cadence equal weight.

formal
  Use the normalized uncertainty column corresponding to the selected flux
  stream. Nonfinite values are replaced robustly and a configurable lower
  error floor prevents a few tiny formal errors from dominating the fit.

sector_rescaled
  Use the formal relative errors, but rescale them separately within each TESS
  sector so the median uncertainty matches that sector's robust
  first-difference scatter. This is the default for converted multi-sector
  SPOC data. If a usable uncertainty column is absent, analysis falls back to
  equal weights with a message in the log.

The same selected TESS weights are used consistently for frequency discovery,
local refinement, prewhitening, local-SNR calculations, and final sinusoidal
fits in both guided and joint modes.

TESS-only frequency limit
-------------------------
Cap Fmax to TESS Nyquist
  When enabled, the effective maximum TESS search frequency is the smaller of
  the GUI Fmax value and the TESS Nyquist frequency for the current dataset.
"""

TOOLTIPS = {
    "guided_script": "Path to the guided-analysis Python script that will be imported and run.",
    "converter_script": "Path to spoc_lightcurve_converter.py used when SPOC lc.fits conversion mode is selected.",
    "analysis_mode": "Choose whether to run the Guided or Joint analysis workflow.",
    "ui_mode": "Basic shows typical-run controls; Expert reveals numerical and diagnostic controls without resetting their values.",
    "pol_smooth_enabled": "Smooth the polarimetry periodogram used for candidate selection in Guided or Joint analysis. Final time-domain fits remain unsmoothed.",
    "pol_smooth_kernel": "Gaussian uses the entered width as FWHM; Boxcar uses it as the full box width.",
    "pol_smooth_width": "Width in independent polarimetry resolution elements (1/T_pol). Default: 10.",
    "joint_script": "Path to the joint-search companion Python script used when Analysis mode = joint_search.",
    "joint_k_candidates": "Number of top peaks retained from the whitened joint spectrum in each iteration.",
    "joint_top_n_raw_tess": "Number of extra raw-TESS candidates kept each iteration.",
    "joint_coarse_oversample": "Coarse-grid oversampling factor for the joint-search frequency grid.",
    "joint_refine_factor": "Refinement factor used around candidate frequencies in the joint search.",
    "joint_max_iters": "Maximum number of accepted iterations in the joint search.",
    "joint_kfit": "Frequency-fit half-width multiplier used when refining local fits in the joint search.",
    "joint_snr_stop": "Stop the joint search when both TESS and polarimetry SNRs fall below this threshold.",
    "joint_w_prefilter": "Whitened-power prefilter threshold used before detailed joint fitting.",
    "joint_ks_tess": "Local-noise sideband width used for TESS in the joint search.",
    "joint_ks_pol": "Local-noise sideband width used for polarimetry in the joint search.",
    "joint_trim_top_frac": "Fraction of the highest sideband values trimmed when estimating local noise in the joint search.",
    "joint_weight_mode": "How TESS and polarimetry are relatively weighted in the joint combined score.",
    "joint_scale_free_basis": "Scale-free basis used when joint_weight_mode = scale_free.",
    "joint_manual_w_tess": "Manual relative weight for photometry when joint_weight_mode = manual.",
    "joint_manual_w_pol": "Manual relative weight for polarimetry when joint_weight_mode = manual.",
    "tess_input_mode": "Choose whether TESS input comes from an existing CSV, a pipeline/custom CSV directory, a batch-per-file CSV directory, or SPOC lc.fits conversion.",
    "tess_csv": "Existing TESS CSV file to load directly in spoc_csv mode.",
    "pipeline_dir": "Directory of pipeline/custom CSV light curves for pipeline_dir mode.",
    "pipeline_pattern": "Filename pattern used when scanning the pipeline/custom light-curve directory.",
    "pipeline_flux": "Preferred flux-family selection in pipeline_dir mode: raw, detrended, or auto.",
    "pipeline_recursive": "Search the pipeline/custom light-curve directory recursively.",
    "pipeline_batch_skip_existing": "In batch-per-file CSV directory mode, skip files whose output directory already contains a peaks-table CSV.",
    "spoc_input": "One SPOC *-lc.fits/*-fast-lc.fits file, a directory, or a glob pattern. Target-pixel *-tp.fits files are rejected with a clear message. Paths containing spaces are passed safely.",
    "spoc_pattern": "Glob pattern used when the SPOC input is a directory, for example *lc.fits.",
    "spoc_recursive": "Search recursively when the SPOC input is a directory or recursive glob.",
    "spoc_output_dir": "Directory where the converter writes per-sector CSVs, combined TIC CSVs, and its JSON manifest.",
    "spoc_flux": "Science stream placed in flux_medscaled/flux_selected_rel: auto prefers PDCSAP, or explicitly choose PDCSAP or SAP.",
    "spoc_quality": "good keeps only QUALITY==0 cadences; all retains every finite cadence.",
    "spoc_normalization": "Per-sector normalization: sigma-clipped median division, ordinary median division, or none.",
    "spoc_normalization_sigma": "Sigma threshold used by sigma-clipped-median normalization.",
    "spoc_normalization_iters": "Maximum iterations used by sigma-clipped-median normalization.",
    "spoc_cadence_policy": "When a TIC has more than one cadence product in the same sector, shortest keeps the fastest product in the combined CSV.",
    "spoc_combine_by_tic": "Write a normalized, time-sorted multi-sector CSV for each TIC and automatically use it for analysis when exactly one TIC is present.",
    "spoc_write_per_sector": "Also retain one converted CSV for each individual SPOC input product.",
    "pol_csv": "Polarimetry CSV: either raw/basic polarimetry or a precomputed analysis-frame CSV.",
    "use_polarimetry": "Guided mode only. When unchecked, run only the TESS frequency search and global multisinusoid fit, skipping all polarimetry steps.",
    "pol_product": "Polarimetry product to analyze: nm, resid_nm_pchip, or pw_resid_nm_pchip.",
    "save_generated_frame": "Save the generated analysis-frame CSV when the input polarimetry is raw/basic.",
    "generated_analysis_dir": "Optional directory for the generated analysis-frame CSV. Blank means next to the polarimetry file.",
    "output_target_subdir": "If enabled, place analysis outputs inside a subdirectory named after the target to keep runs separated and reduce accidental overwriting.",
    "outroot": "Top-level output directory. The analysis writes target/channel tables, figures, diagnostics, and run provenance beneath it.",
    "show_inline": "Inline plots / interactively. Usually best left off in GUI mode.",
    "verbose": "General verbosity level printed by the guided-analysis script.",
    "lsq_verbose": "least_squares verbosity level for the global fits.",
    "fmin": "Minimum frequency in cycles/day.",
    "fmax": "Maximum frequency in cycles/day.",
    "tess_cap_fmax_to_nyquist": "When enabled, use the smaller of the GUI Fmax value and the TESS Nyquist frequency for the current dataset. This is the default and is especially useful in batch mode.",
    "tess_weight_mode": "TESS cadence weighting: none gives equal weights; formal uses normalized flux errors; sector_rescaled additionally matches each sector's median error to its robust first-difference scatter.",
    "tess_error_floor_frac": "Lower uncertainty floor as a fraction of the median error within each sector/group. A value of 0.25 limits any cadence to at most 16 times the median inverse-variance weight.",
    "tess_grid_mode": "Use the full baseline or the longest contiguous chunk to set the TESS discovery grid.",
    "tess_snr_stop": "Stop the sequential TESS search when the local SNR drops below this value.",
    "max_tess_modes": "Maximum number of sequential TESS modes to extract before the global fit.",
    "pol_snr_stop": "Guided polarimetry detections must reach at least this local SNR.",
    "max_pol_modes": "Maximum number of guided polarimetry modes to test / keep.",
    "guided_pol_fmin": "Do not test TESS template frequencies below this value in the guided polarimetric matching.",
    "search_window_mult": "Half-width of the guided polarimetric frequency window in units of max(1/T_tess, 1/T_pol).",
    "noise_ks": "Sideband half-width used for the guided local-noise estimate.",
    "noise_bins": "Minimum sideband width, in local-grid bins, for the guided local-noise estimate.",
    "channels": "Choose which polarimetric channels to analyze. Usually q, u, and p.",
    "use_offsets": "Include per-night offsets in the polarimetric baseline model.",
    "use_slopes": "Include per-night slopes in the polarimetric baseline model.",
    "group_mode": "How polarimetry baseline groups are defined: by time gap, integer JD, parent observing run, or full subrun label.",
    "gap_hours": "Gap threshold in hours used when group_mode = gap.",
    "do_detrend": "Apply the optional broad polynomial detrending hook before the main analysis.",
    "detrend_order": "Polynomial order used by the optional broad detrending hook.",
    "n_phase_plots": "Number of strongest modes to show in the phased-summary plots.",
    "phase_sort_by": "How to rank modes when selecting phased-summary plots.",
    "phase_plot_style": "Joint only: whether phased plots use isolated_mode or prefit_residual style.",
    "phase_zero_mode": "Joint-analysis phase-reference convention. Guided analysis retains its established local-start convention.",
    "phase_zero_btjd": "Custom BTJD phase zero for Joint analysis when phase_zero_mode = custom_btjd.",
    "preview_dir": "Directory scanned recursively for PNG previews.",
    "preview_scale_mode": "Choose how preview images are displayed: Fit scales the image to the visible preview pane; percentage modes use a fixed zoom level.",
    "preview_zoom": "Manual preview zoom level. Also adjustable with Ctrl + mouse wheel.",
}


def quote_cmd(cmd: list[str]) -> str:
    return " ".join(shlex.quote(str(x)) for x in cmd)


def _looks_like_tess_target_pixel_path(value: str | Path) -> bool:
    """Return True for standard TESS target-pixel product filenames."""
    name = Path(str(value)).name.lower()
    return name.endswith(("-tp.fits", "-tp.fits.gz", "_tp.fits", "_tp.fits.gz"))


def _looks_like_spoc_lightcurve_path(value: str | Path) -> bool:
    """Return True for standard SPOC light-curve product filenames."""
    name = Path(str(value)).name.lower()
    return name.endswith(("-lc.fits", "-lc.fits.gz", "_lc.fits", "_lc.fits.gz"))


def _target_pixel_guidance(path: str | Path) -> str:
    name = Path(str(path)).name
    return (
        f"{name} is a TESS target-pixel file, not a SPOC light-curve file.\n\n"
        "Choose the corresponding *-lc.fits or *-fast-lc.fits product, or "
        "process the target-pixel file through the PhotPol extractor first."
    )


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
            foreground="#111111",
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
        self.inner.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.create_window((0, 0), window=self.inner, anchor="nw")
        self.canvas.configure(yscrollcommand=self.vsb.set, xscrollcommand=self.hsb.set)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.vsb.grid(row=0, column=1, sticky="ns")
        self.hsb.grid(row=1, column=0, sticky="ew")
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)


class GuidedAnalysisGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(APP_TITLE)
        # Keep the default roomy on large displays, but fit on typical laptop
        # screens across Windows, Linux, and macOS.
        width = min(DEFAULT_WINDOW_WIDTH, max(720, self.winfo_screenwidth() - 80))
        height = min(DEFAULT_WINDOW_HEIGHT, max(600, self.winfo_screenheight() - 120))
        self.geometry(f"{width}x{height}")
        self.protocol("WM_DELETE_WINDOW", self.close_application)

        self.log_queue: queue.Queue[tuple[str, object]] = queue.Queue()
        self.current_process: subprocess.Popen | None = None
        self.current_output_dir: Path | None = None
        self.preview_image = None
        self._preview_paths: list[Path] = []
        self._temp_runner_path: Path | None = None

        self._build_vars()
        self._build_ui()
        self._poll_log_queue()
        self._update_ui_mode_state()
        self._update_tess_mode_state()
        self._update_analysis_mode_state()
        self._update_polarimetry_state()
        self._update_smoothing_state()
        self._update_joint_weight_mode_state()
        self._update_phase_zero_mode_state()
        self._update_run_plan_preview()
        self._refresh_preview_list()

    def _build_vars(self):
        # script paths
        self.guided_script = tk.StringVar(value=str(SCRIPT_DIR / "tess_guided_analysis.py"))
        self.joint_script = tk.StringVar(value=str(SCRIPT_DIR / "joint_search_option.py"))
        self.converter_script = tk.StringVar(value=str(SCRIPT_DIR / "spoc_lightcurve_converter.py"))

        # top-level mode
        self.ui_mode = tk.StringVar(value="basic")
        self.analysis_mode = tk.StringVar(value="guided_analysis")

        # TESS input
        self.tess_input_mode = tk.StringVar(value="existing_csv")
        self.tess_csv = tk.StringVar(value="combined_filtered.csv")
        self.pipeline_dir = tk.StringVar(value="tess_pipeline_lcs")
        self.pipeline_pattern = tk.StringVar(value="*.csv")
        self.pipeline_recursive = tk.BooleanVar(value=False)
        self.pipeline_batch_skip_existing = tk.BooleanVar(value=True)
        self.pipeline_flux = tk.StringVar(value="auto")
        self.tess_force_y_col = tk.StringVar(value="")

        self.spoc_input = tk.StringVar(value="")
        self.spoc_pattern = tk.StringVar(value="*lc.fits")
        self.spoc_recursive = tk.BooleanVar(value=True)
        self.spoc_output_dir = tk.StringVar(value="spoc_csvs")
        self.spoc_flux = tk.StringVar(value="auto")
        self.spoc_quality = tk.StringVar(value="good")
        self.spoc_normalization = tk.StringVar(value="sigma-clipped-median")
        self.spoc_normalization_sigma = tk.DoubleVar(value=4.0)
        self.spoc_normalization_iters = tk.IntVar(value=5)
        self.spoc_cadence_policy = tk.StringVar(value="shortest")
        self.spoc_combine_by_tic = tk.BooleanVar(value=True)
        self.spoc_write_per_sector = tk.BooleanVar(value=True)

        # polarimetry / output
        self.use_polarimetry = tk.BooleanVar(value=True)
        self.pol_csv = tk.StringVar(value="target_IND.csv")
        self.pol_product = tk.StringVar(value="resid_nm_pchip")
        self.save_generated_frame = tk.BooleanVar(value=True)
        self.generated_analysis_dir = tk.StringVar(value="")
        self.output_target_subdir = tk.BooleanVar(value=False)
        self.outroot = tk.StringVar(value="guided_analysis_outputs")
        self.show_plots_inline = tk.BooleanVar(value=False)
        self.verbose = tk.IntVar(value=1)
        self.lsq_verbose = tk.IntVar(value=0)

        # main analysis controls
        self.fmin = tk.DoubleVar(value=0.01)
        self.fmax = tk.DoubleVar(value=50.0)
        self.tess_cap_fmax_to_nyquist = tk.BooleanVar(value=True)
        self.tess_weight_mode = tk.StringVar(value="sector_rescaled")
        self.tess_error_floor_frac = tk.DoubleVar(value=0.25)
        self.tess_grid_mode = tk.StringVar(value="full_baseline")
        self.tess_snr_stop = tk.DoubleVar(value=4.0)
        self.max_tess_modes = tk.IntVar(value=20)
        self.pol_snr_stop = tk.DoubleVar(value=2.0)
        self.max_pol_modes = tk.IntVar(value=99)
        self.guided_pol_fmin = tk.DoubleVar(value=0.2)
        self.search_window_mult = tk.DoubleVar(value=10.0)
        self.noise_ks = tk.DoubleVar(value=3.0)
        self.noise_bins = tk.IntVar(value=6)
        self.pol_smooth_enabled = tk.BooleanVar(value=False)
        self.pol_smooth_kernel = tk.StringVar(value="gaussian")
        self.pol_smooth_width = tk.DoubleVar(value=10.0)

        self.channel_q = tk.BooleanVar(value=True)
        self.channel_u = tk.BooleanVar(value=True)
        self.channel_p = tk.BooleanVar(value=True)

        self.use_offsets = tk.BooleanVar(value=True)
        self.use_slopes = tk.BooleanVar(value=False)
        self.group_mode = tk.StringVar(value="gap")
        self.gap_hours = tk.DoubleVar(value=8.0)

        self.do_detrend = tk.BooleanVar(value=False)
        self.detrend_order = tk.IntVar(value=0)
        self.n_phase_plots = tk.IntVar(value=3)
        self.phase_sort_by = tk.StringVar(value="amp")
        self.phase_plot_style = tk.StringVar(value="isolated_mode")
        self.phase_zero_mode = tk.StringVar(value="local_start")
        self.phase_zero_btjd = tk.DoubleVar(value=0.0)

        # joint-search settings
        self.joint_k_candidates = tk.IntVar(value=10)
        self.joint_top_n_raw_tess = tk.IntVar(value=3)
        self.joint_coarse_oversample = tk.DoubleVar(value=1.0)
        self.joint_refine_factor = tk.IntVar(value=10)
        self.joint_max_iters = tk.IntVar(value=20)
        self.joint_kfit = tk.DoubleVar(value=3.0)
        self.joint_snr_stop = tk.DoubleVar(value=2.0)
        self.joint_w_prefilter = tk.DoubleVar(value=1.5)
        self.joint_ks_tess = tk.DoubleVar(value=15.0)
        self.joint_ks_pol = tk.DoubleVar(value=10.0)
        self.joint_trim_top_frac = tk.DoubleVar(value=0.10)
        self.joint_weight_mode = tk.StringVar(value="equal")
        self.joint_scale_free_basis = tk.StringVar(value="tseg")
        self.joint_manual_w_tess = tk.DoubleVar(value=1.0)
        self.joint_manual_w_pol = tk.DoubleVar(value=1.0)

        self.run_plan_preview = tk.StringVar(value="")
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
        # Manual zoom entry should take effect immediately, regardless of the
        # current scale-mode selection. Switch into a non-percent "Custom" mode
        # so the renderer uses preview_zoom instead of the combobox value.
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
        modebar = ttk.Frame(self)
        modebar.pack(fill="x", padx=12, pady=(10, 0))
        ttk.Label(modebar, text="Interface:").pack(side="left", padx=(0, 6))
        for value, label in (("basic", "Basic"), ("expert", "Expert")):
            rb = ttk.Radiobutton(
                modebar,
                text=label,
                value=value,
                variable=self.ui_mode,
                command=self._update_ui_mode_state,
            )
            rb.pack(side="left", padx=4)
            self._tooltip(rb, "ui_mode")
        ttk.Label(
            modebar,
            text="Basic keeps the normal workflow compact; Expert exposes numerical and diagnostic controls.",
        ).pack(side="left", padx=14)

        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill="both", expand=True, padx=8, pady=8)

        self.tab_analysis = ttk.Frame(self.notebook)
        self.tab_tess = ttk.Frame(self.notebook)
        self.tab_run = ttk.Frame(self.notebook)
        self.tab_help = ttk.Frame(self.notebook)

        self.notebook.add(self.tab_analysis, text="Analysis")
        self.notebook.add(self.tab_tess, text="TESS Input")
        self.notebook.add(self.tab_run, text="Run / Preview")
        self.notebook.add(self.tab_help, text="Help")

        self._build_analysis_tab()
        self._build_tess_tab()
        self._build_run_tab()
        self._build_help_tab()
        # A single dispatcher prevents the preview pane from stealing wheel
        # events from the two scrollable settings tabs.
        self.bind_all("<MouseWheel>", self._on_app_mousewheel)
        self.bind_all("<Button-4>", self._on_app_mousewheel)
        self.bind_all("<Button-5>", self._on_app_mousewheel)

    def _entry(self, parent, label, variable, row, col, browse=None, tooltip_key=None, filetypes=None):
        lab = ttk.Label(parent, text=label)
        lab.grid(row=row, column=col, sticky="w", padx=6, pady=4)
        if tooltip_key:
            self._tooltip(lab, tooltip_key)
        ent = ttk.Entry(parent, textvariable=variable)
        ent.grid(row=row, column=col + 1, sticky="ew", padx=6, pady=4)
        if tooltip_key:
            self._tooltip(ent, tooltip_key)
        if browse == "dir":
            btn = ttk.Button(parent, text="Browse...", command=lambda: self._browse_dir(variable))
            btn.grid(row=row, column=col + 2, padx=6, pady=4)
            if tooltip_key:
                self._tooltip(btn, tooltip_key)
        elif browse == "file":
            btn = ttk.Button(parent, text="Browse...", command=lambda: self._browse_file(variable, filetypes or [("All files", "*.*")]))
            btn.grid(row=row, column=col + 2, padx=6, pady=4)
            if tooltip_key:
                self._tooltip(btn, tooltip_key)
        return ent

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

    def _combo(self, parent, label, variable, values, row, col, tooltip_key=None, state="readonly"):
        lab = ttk.Label(parent, text=label)
        lab.grid(row=row, column=col, sticky="w", padx=6, pady=4)
        if tooltip_key:
            self._tooltip(lab, tooltip_key)
        cb = ttk.Combobox(parent, textvariable=variable, values=values, state=state)
        cb.grid(row=row, column=col + 1, sticky="ew", padx=6, pady=4)
        if tooltip_key:
            self._tooltip(cb, tooltip_key)
        return cb

    def _check(self, parent, text, variable, row, col, command=None, tooltip_key=None, colspan=1):
        cb = ttk.Checkbutton(parent, text=text, variable=variable, command=command)
        cb.grid(row=row, column=col, columnspan=colspan, sticky="w", padx=6, pady=4)
        if tooltip_key:
            self._tooltip(cb, tooltip_key)
        return cb

    def _build_analysis_tab(self):
        sf = ScrollableFrame(self.tab_analysis)
        sf.pack(fill="both", expand=True)
        self.analysis_scroll_canvas = sf.canvas
        root = sf.inner
        root.columnconfigure(0, weight=1)

        basic = ttk.LabelFrame(root, text="Basic analysis")
        basic.grid(row=0, column=0, sticky="ew", padx=8, pady=6)
        for c in range(4):
            basic.columnconfigure(c, weight=1)
        self._combo(basic, "Analysis track", self.analysis_mode, ["guided_analysis", "joint_search"], 0, 0, tooltip_key="analysis_mode")
        self._entry(basic, "Fmin [c/d]", self.fmin, 1, 0, tooltip_key="fmin")
        self._entry(basic, "Fmax [c/d]", self.fmax, 1, 2, tooltip_key="fmax")

        scripts = ttk.LabelFrame(root, text="Script paths / mode")
        scripts.grid(row=1, column=0, sticky="ew", padx=8, pady=6)
        for c in range(5):
            scripts.columnconfigure(c, weight=1)
        self._entry(scripts, "Guided-analysis script", self.guided_script, 0, 0, browse="file", tooltip_key="guided_script", filetypes=[("Python files", "*.py"), ("All files", "*.*")])
        self._entry(scripts, "Joint-search script", self.joint_script, 1, 0, browse="file", tooltip_key="joint_script", filetypes=[("Python files", "*.py"), ("All files", "*.*")])
        self._entry(scripts, "SPOC converter script", self.converter_script, 2, 0, browse="file", tooltip_key="converter_script", filetypes=[("Python files", "*.py"), ("All files", "*.*")])
        self._combo(scripts, "Analysis mode", self.analysis_mode, ["guided_analysis", "joint_search"], 3, 0, tooltip_key="analysis_mode")

        pol = ttk.LabelFrame(root, text="Polarimetry")
        pol.grid(row=2, column=0, sticky="ew", padx=8, pady=6)
        for c in range(4):
            pol.columnconfigure(c, weight=1)
        self.use_polarimetry_chk = self._check(pol, "Use polarimetry", self.use_polarimetry, 0, 0, tooltip_key="use_polarimetry", command=self._update_polarimetry_state, colspan=2)

        self.pol_csv_entry = self._entry(pol, "Polarimetry CSV", self.pol_csv, 1, 0, browse="file", tooltip_key="pol_csv", filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        self.pol_product_combo = self._combo(pol, "POL product", self.pol_product, ["nm", "resid_nm_pchip", "pw_resid_nm_pchip"], 2, 0, tooltip_key="pol_product")
        self.save_generated_frame_chk = self._check(pol, "Save generated analysis frame", self.save_generated_frame, 2, 2, tooltip_key="save_generated_frame", command=self._update_run_plan_preview)
        self.generated_analysis_dir_entry = self._entry(pol, "Generated analysis dir", self.generated_analysis_dir, 3, 0, browse="dir", tooltip_key="generated_analysis_dir")

        smoothing = ttk.LabelFrame(root, text="Polarimetry periodogram smoothing")
        smoothing.grid(row=3, column=0, sticky="ew", padx=8, pady=6)
        for c in range(4):
            smoothing.columnconfigure(c, weight=1)
        self.pol_smooth_enabled_chk = self._check(
            smoothing,
            "Smooth before Guided/Joint candidate selection",
            self.pol_smooth_enabled,
            0,
            0,
            command=self._update_smoothing_state,
            tooltip_key="pol_smooth_enabled",
            colspan=2,
        )
        self.pol_smooth_kernel_combo = self._combo(
            smoothing, "Kernel", self.pol_smooth_kernel, ["gaussian", "boxcar"], 0, 2,
            tooltip_key="pol_smooth_kernel",
        )
        self.pol_smooth_width_entry = self._entry(
            smoothing, "Width [resolution elements]", self.pol_smooth_width, 1, 0,
            tooltip_key="pol_smooth_width",
        )
        ttk.Label(
            smoothing,
            text="Gaussian width is FWHM; boxcar width is full width. Final time-series fits are not smoothed.",
        ).grid(row=1, column=2, columnspan=2, sticky="w", padx=6, pady=4)

        basic_output = ttk.LabelFrame(root, text="Output")
        basic_output.grid(row=4, column=0, sticky="ew", padx=8, pady=6)
        for c in range(4):
            basic_output.columnconfigure(c, weight=1)
        self._entry(basic_output, "Output root", self.outroot, 0, 0, browse="dir", tooltip_key="outroot")
        self._check(basic_output, "Use target subdirectory", self.output_target_subdir, 0, 2, tooltip_key="output_target_subdir", command=self._update_run_plan_preview, colspan=2)

        outdisp = ttk.LabelFrame(root, text="Expert output / display")
        outdisp.grid(row=5, column=0, sticky="ew", padx=8, pady=6)
        for c in range(4):
            outdisp.columnconfigure(c, weight=1)
        self.outroot_entry = self._entry(outdisp, "Output root", self.outroot, 0, 0, browse="dir", tooltip_key="outroot")
        self.output_target_subdir_chk = self._check(outdisp, "Place outputs in target subdirectory", self.output_target_subdir, 0, 2, tooltip_key="output_target_subdir", command=self._update_run_plan_preview, colspan=2)
        self.inline_plots_chk = self._check(outdisp, "Inline plots", self.show_plots_inline, 1, 0, tooltip_key="show_inline", command=self._update_run_plan_preview)
        self._spin(outdisp, "Verbose", self.verbose, 0, 5, 1, 2, tooltip_key="verbose")
        self._spin(outdisp, "LSQ verbose", self.lsq_verbose, 0, 5, 2, 0, tooltip_key="lsq_verbose")
        self._combo(outdisp, "TESS error weighting", self.tess_weight_mode, ["none", "formal", "sector_rescaled"], 3, 0, tooltip_key="tess_weight_mode")
        self._entry(outdisp, "TESS error floor / median", self.tess_error_floor_frac, 3, 2, tooltip_key="tess_error_floor_frac")

        main = ttk.LabelFrame(root, text="Main analysis settings")
        main.grid(row=6, column=0, sticky="ew", padx=8, pady=6)
        for c in range(4):
            main.columnconfigure(c, weight=1)
        self._entry(main, "Fmin [c/d]", self.fmin, 0, 0, tooltip_key="fmin")
        self._entry(main, "Fmax [c/d]", self.fmax, 0, 2, tooltip_key="fmax")
        self._check(main, "Cap Fmax to TESS Nyquist", self.tess_cap_fmax_to_nyquist, 1, 2, tooltip_key="tess_cap_fmax_to_nyquist", command=self._update_run_plan_preview, colspan=2)
        self._combo(main, "TESS grid mode", self.tess_grid_mode, ["full_baseline", "longest_chunk"], 2, 0, tooltip_key="tess_grid_mode")
        self._entry(main, "TESS SNR stop", self.tess_snr_stop, 2, 2, tooltip_key="tess_snr_stop")
        self._spin(main, "Max TESS modes", self.max_tess_modes, 1, 999, 3, 0, tooltip_key="max_tess_modes")
        self._entry(main, "POL SNR stop", self.pol_snr_stop, 3, 2, tooltip_key="pol_snr_stop")
        self._spin(main, "Max POL modes", self.max_pol_modes, 1, 999, 4, 0, tooltip_key="max_pol_modes")
        self._entry(main, "Guided POL fmin", self.guided_pol_fmin, 4, 2, tooltip_key="guided_pol_fmin")
        self._entry(main, "Search window mult", self.search_window_mult, 5, 0, tooltip_key="search_window_mult")
        self._entry(main, "Noise KS", self.noise_ks, 5, 2, tooltip_key="noise_ks")
        self._spin(main, "Noise side bins", self.noise_bins, 1, 1000, 6, 0, tooltip_key="noise_bins")

        basic_channels = ttk.LabelFrame(root, text="Polarimetry channels")
        basic_channels.grid(row=7, column=0, sticky="ew", padx=8, pady=6)
        self._check(basic_channels, "q", self.channel_q, 0, 0, tooltip_key="channels", command=self._update_run_plan_preview)
        self._check(basic_channels, "u", self.channel_u, 0, 1, tooltip_key="channels", command=self._update_run_plan_preview)
        self._check(basic_channels, "p", self.channel_p, 0, 2, tooltip_key="channels", command=self._update_run_plan_preview)

        chans = ttk.LabelFrame(root, text="Channels / baseline model")
        chans.grid(row=8, column=0, sticky="ew", padx=8, pady=6)
        for c in range(4):
            chans.columnconfigure(c, weight=1)
        self._check(chans, "q", self.channel_q, 0, 0, tooltip_key="channels", command=self._update_run_plan_preview)
        self._check(chans, "u", self.channel_u, 0, 1, tooltip_key="channels", command=self._update_run_plan_preview)
        self._check(chans, "p", self.channel_p, 0, 2, tooltip_key="channels", command=self._update_run_plan_preview)
        self._check(chans, "Use night offsets", self.use_offsets, 1, 0, tooltip_key="use_offsets", command=self._update_run_plan_preview)
        self._check(chans, "Use night slopes", self.use_slopes, 1, 1, tooltip_key="use_slopes", command=self._update_run_plan_preview)
        self._combo(chans, "Group mode", self.group_mode, ["gap", "integer_jd", "run", "subrun"], 2, 0, tooltip_key="group_mode")
        self._entry(chans, "Gap hours", self.gap_hours, 2, 2, tooltip_key="gap_hours")

        extras = ttk.LabelFrame(root, text="Optional detrending and plots")
        extras.grid(row=9, column=0, sticky="ew", padx=8, pady=6)
        for c in range(4):
            extras.columnconfigure(c, weight=1)
        self._check(extras, "Apply broad polynomial detrend", self.do_detrend, 0, 0, tooltip_key="do_detrend", command=self._update_run_plan_preview)
        self._spin(extras, "Detrend poly order", self.detrend_order, 0, 10, 0, 2, tooltip_key="detrend_order")
        self._spin(extras, "N phase plots", self.n_phase_plots, 0, 20, 1, 0, tooltip_key="n_phase_plots")
        self.phase_sort_by_combo = self._combo(extras, "Phase sort by", self.phase_sort_by, ["amp", "snr", "mode"], 1, 2, tooltip_key="phase_sort_by")
        self.phase_plot_style_combo = self._combo(
            extras,
            "Joint phase plot style",
            self.phase_plot_style,
            ["isolated_mode", "prefit_residual"],
            2,
            0,
            tooltip_key="phase_plot_style",
        )
        self.phase_zero_mode_combo = self._combo(extras, "Phase zero mode", self.phase_zero_mode, ["local_start", "btjd_zero", "custom_btjd"], 2, 2, tooltip_key="phase_zero_mode")
        ttk.Label(extras, text="BTJD=0.0 uses the absolute TESS zero point.").grid(row=3, column=0, columnspan=2, sticky="w", padx=6, pady=2)
        self.phase_zero_btjd_entry = self._entry(extras, "Custom phase BTJD", self.phase_zero_btjd, 3, 2, tooltip_key="phase_zero_btjd")

        jointf = ttk.LabelFrame(root, text="Joint-search settings")
        jointf.grid(row=10, column=0, sticky="ew", padx=8, pady=6)
        for c in range(4):
            jointf.columnconfigure(c, weight=1)
        self._spin(jointf, "K candidates", self.joint_k_candidates, 1, 999, 0, 0, tooltip_key="joint_k_candidates")
        self._spin(jointf, "Top N raw TESS", self.joint_top_n_raw_tess, 0, 999, 0, 2, tooltip_key="joint_top_n_raw_tess")
        self._entry(jointf, "Coarse oversample", self.joint_coarse_oversample, 1, 0, tooltip_key="joint_coarse_oversample")
        self._spin(jointf, "Refine factor", self.joint_refine_factor, 1, 999, 1, 2, tooltip_key="joint_refine_factor")
        self._spin(jointf, "Max iters", self.joint_max_iters, 1, 999, 2, 0, tooltip_key="joint_max_iters")
        self._entry(jointf, "KFIT", self.joint_kfit, 2, 2, tooltip_key="joint_kfit")
        self._entry(jointf, "Joint SNR stop", self.joint_snr_stop, 3, 0, tooltip_key="joint_snr_stop")
        self._entry(jointf, "W prefilter", self.joint_w_prefilter, 3, 2, tooltip_key="joint_w_prefilter")
        self._entry(jointf, "KS TESS", self.joint_ks_tess, 4, 0, tooltip_key="joint_ks_tess")
        self._entry(jointf, "KS POL", self.joint_ks_pol, 4, 2, tooltip_key="joint_ks_pol")
        self._entry(jointf, "Trim top frac", self.joint_trim_top_frac, 5, 0, tooltip_key="joint_trim_top_frac")
        self.joint_weight_mode_combo = self._combo(jointf, "Weight mode", self.joint_weight_mode, ["equal", "scale_free", "manual"], 5, 2, tooltip_key="joint_weight_mode")
        self.joint_scale_free_basis_combo = self._combo(jointf, "Scale-free basis", self.joint_scale_free_basis, ["tseg"], 6, 0, tooltip_key="joint_scale_free_basis")
        self.joint_manual_tess_entry = self._entry(jointf, "Manual TESS weight", self.joint_manual_w_tess, 6, 2, tooltip_key="joint_manual_w_tess")
        self.joint_manual_pol_entry = self._entry(jointf, "Manual POL weight", self.joint_manual_w_pol, 7, 0, tooltip_key="joint_manual_w_pol")

        actions = ttk.LabelFrame(root, text="Actions")
        actions.grid(row=11, column=0, sticky="ew", padx=8, pady=6)
        for c in range(5):
            actions.columnconfigure(c, weight=1)
        ttk.Button(actions, text="Show run plan", command=self._update_run_plan_preview).grid(row=0, column=0, padx=6, pady=6, sticky="ew")
        ttk.Button(actions, text="Run analysis", command=self.run_analysis_job).grid(row=0, column=1, padx=6, pady=6, sticky="ew")
        ttk.Button(actions, text="Stop", command=self.stop_current_process).grid(row=0, column=2, padx=6, pady=6, sticky="ew")
        ttk.Button(actions, text="Open output folder", command=lambda: self._open_folder(Path(self.outroot.get() or "."))).grid(row=0, column=3, padx=6, pady=6, sticky="ew")
        ttk.Button(actions, text="Help", command=self.show_help_tab).grid(row=0, column=4, padx=6, pady=6, sticky="ew")
        ttk.Label(actions, text="Run plan preview").grid(row=1, column=0, sticky="w", padx=6)
        self.plan_text = scrolledtext.ScrolledText(actions, height=8, wrap="word")
        self.plan_text.grid(row=2, column=0, columnspan=5, sticky="ew", padx=6, pady=6)
        self.plan_text.configure(state="disabled")

        self.analysis_mode_groups = {
            "guided": [main],
            "joint": [jointf],
        }
        self.basic_only_frames = [basic, basic_output, basic_channels]
        self.expert_only_frames = [scripts, outdisp, main, chans, extras, jointf]
        self.polarimetry_toggle_widgets = [
            getattr(self, "pol_csv_entry", None),
            getattr(self, "pol_product_combo", None),
            getattr(self, "save_generated_frame_chk", None),
            getattr(self, "generated_analysis_dir_entry", None),
        ]

        self._trace_vars([
            self.guided_script, self.joint_script, self.converter_script, self.ui_mode, self.analysis_mode, self.use_polarimetry,
            self.pol_csv, self.pol_product, self.save_generated_frame, self.generated_analysis_dir,
            self.output_target_subdir, self.outroot, self.show_plots_inline, self.verbose, self.lsq_verbose,
            self.fmin, self.fmax, self.tess_weight_mode, self.tess_error_floor_frac,
            self.tess_grid_mode, self.tess_snr_stop, self.max_tess_modes,
            self.pol_snr_stop, self.max_pol_modes, self.guided_pol_fmin, self.search_window_mult,
            self.noise_ks, self.noise_bins, self.pol_smooth_enabled, self.pol_smooth_kernel,
            self.pol_smooth_width, self.channel_q, self.channel_u, self.channel_p,
            self.use_offsets, self.use_slopes, self.group_mode, self.gap_hours,
            self.do_detrend, self.detrend_order, self.n_phase_plots, self.phase_sort_by,
            self.phase_plot_style, self.phase_zero_mode, self.phase_zero_btjd,
            self.joint_k_candidates, self.joint_top_n_raw_tess, self.joint_coarse_oversample,
            self.joint_refine_factor, self.joint_max_iters, self.joint_kfit,
            self.joint_snr_stop, self.joint_w_prefilter, self.joint_ks_tess,
            self.joint_ks_pol, self.joint_trim_top_frac, self.joint_weight_mode,
            self.joint_scale_free_basis, self.joint_manual_w_tess, self.joint_manual_w_pol,
        ], self._update_run_plan_preview)
        self._trace_vars([self.ui_mode], self._update_ui_mode_state)
        self._trace_vars([self.analysis_mode], self._update_analysis_mode_state)
        self._trace_vars([self.use_polarimetry], self._update_polarimetry_state)
        self._trace_vars([self.joint_weight_mode], self._update_joint_weight_mode_state)
        self._trace_vars([self.phase_zero_mode], self._update_phase_zero_mode_state)
        self._trace_vars([self.pol_smooth_enabled], self._update_smoothing_state)

    def _build_tess_tab(self):
        sf = ScrollableFrame(self.tab_tess)
        sf.pack(fill="both", expand=True)
        self.tess_scroll_canvas = sf.canvas
        root = sf.inner
        root.columnconfigure(0, weight=1)

        modef = ttk.LabelFrame(root, text="TESS input mode")
        modef.grid(row=0, column=0, sticky="ew", padx=8, pady=6)
        for c in range(4):
            modef.columnconfigure(c, weight=1)
        self._combo(modef, "TESS input mode", self.tess_input_mode,
                    ["existing_csv", "pipeline_dir", "pipeline_dir_batch", "spoc_lc_fits"], 0, 0, tooltip_key="tess_input_mode")

        csvf = ttk.LabelFrame(root, text="Existing CSV mode")
        csvf.grid(row=1, column=0, sticky="ew", padx=8, pady=6)
        csvf.columnconfigure(1, weight=1)
        self._entry(csvf, "TESS CSV", self.tess_csv, 0, 0, browse="file", tooltip_key="tess_csv", filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])

        pipef = ttk.LabelFrame(root, text="Pipeline/custom CSV directory mode")
        pipef.grid(row=2, column=0, sticky="ew", padx=8, pady=6)
        for c in range(4):
            pipef.columnconfigure(c, weight=1)
        self._entry(pipef, "Pipeline dir", self.pipeline_dir, 0, 0, browse="dir", tooltip_key="pipeline_dir")
        self._entry(pipef, "Pattern", self.pipeline_pattern, 1, 0, tooltip_key="pipeline_pattern")
        self._combo(pipef, "Flux family", self.pipeline_flux, ["raw", "detrended", "auto"], 1, 2, tooltip_key="pipeline_flux")
        self._check(pipef, "Recursive", self.pipeline_recursive, 2, 0, tooltip_key="pipeline_recursive", command=self._update_run_plan_preview)
        self._check(pipef, "Batch mode: skip existing", self.pipeline_batch_skip_existing, 2, 1, tooltip_key="pipeline_batch_skip_existing", command=self._update_run_plan_preview)
        self._entry(pipef, "Force TESS y column (optional)", self.tess_force_y_col, 3, 0)

        spocf = ttk.LabelFrame(root, text="SPOC lc.fits conversion mode")
        spocf.grid(row=3, column=0, sticky="ew", padx=8, pady=6)
        for c in range(4):
            spocf.columnconfigure(c, weight=1)

        self._entry(
            spocf, "SPOC input file / directory / glob", self.spoc_input, 0, 0,
            browse=None, tooltip_key="spoc_input",
        )
        file_btn = ttk.Button(spocf, text="Browse lc.fits...", command=self._browse_spoc_lc_file)
        file_btn.grid(row=0, column=2, padx=6, pady=4)
        self._tooltip(file_btn, "spoc_input")
        dir_btn = ttk.Button(spocf, text="Browse dir...", command=lambda: self._browse_dir(self.spoc_input))
        dir_btn.grid(row=0, column=3, padx=6, pady=4)
        self._tooltip(dir_btn, "spoc_input")

        self._entry(spocf, "Directory pattern", self.spoc_pattern, 1, 0, tooltip_key="spoc_pattern")
        self._check(spocf, "Recursive", self.spoc_recursive, 1, 2, tooltip_key="spoc_recursive", command=self._update_run_plan_preview)
        self._entry(spocf, "Converted output directory", self.spoc_output_dir, 2, 0, browse="dir", tooltip_key="spoc_output_dir")

        self._combo(spocf, "Science flux", self.spoc_flux, ["auto", "pdcsap", "sap"], 3, 0, tooltip_key="spoc_flux")
        self._combo(spocf, "Quality policy", self.spoc_quality, ["good", "all"], 3, 2, tooltip_key="spoc_quality")
        self._combo(
            spocf, "Normalization", self.spoc_normalization,
            ["sigma-clipped-median", "median", "none"], 4, 0,
            tooltip_key="spoc_normalization",
        )
        self._combo(
            spocf, "Cadence policy", self.spoc_cadence_policy,
            ["shortest", "longest", "all"], 4, 2,
            tooltip_key="spoc_cadence_policy",
        )
        self._entry(spocf, "Normalization sigma", self.spoc_normalization_sigma, 5, 0, tooltip_key="spoc_normalization_sigma")
        self._spin(spocf, "Normalization iterations", self.spoc_normalization_iters, 1, 99, 5, 2, tooltip_key="spoc_normalization_iters")
        self._check(
            spocf, "Combine sectors by TIC", self.spoc_combine_by_tic, 6, 0,
            tooltip_key="spoc_combine_by_tic", command=self._update_run_plan_preview,
        )
        self._check(
            spocf, "Write per-sector CSVs", self.spoc_write_per_sector, 6, 2,
            tooltip_key="spoc_write_per_sector", command=self._update_run_plan_preview,
        )

        self._trace_vars([
            self.tess_input_mode, self.tess_csv, self.pipeline_dir, self.pipeline_pattern,
            self.pipeline_recursive, self.pipeline_batch_skip_existing, self.pipeline_flux, self.tess_force_y_col,
            self.spoc_input, self.spoc_pattern, self.spoc_recursive, self.spoc_output_dir,
            self.spoc_flux, self.spoc_quality, self.spoc_normalization,
            self.spoc_normalization_sigma, self.spoc_normalization_iters,
            self.spoc_cadence_policy, self.spoc_combine_by_tic, self.spoc_write_per_sector,
        ], self._update_run_plan_preview)
        self._trace_vars([self.tess_input_mode], self._update_tess_mode_state)

        self.tess_mode_groups = {
            "csv": csvf,
            "pipe": pipef,
            "spoc": spocf,
        }

    def _build_run_tab(self):
        top = ttk.Frame(self.tab_run)
        top.pack(fill="x", padx=8, pady=6)
        ttk.Button(top, text="Clear log", command=self.clear_log).pack(side="left", padx=4)
        ttk.Button(top, text="Save log", command=self.save_log).pack(side="left", padx=4)
        ttk.Button(top, text="Copy run plan", command=self.copy_run_plan).pack(side="left", padx=4)
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
        ttk.Label(list_frame, text="PNG files (recursive)").pack(anchor="w")
        self.preview_list = tk.Listbox(list_frame, width=50)
        preview_scroll = ttk.Scrollbar(list_frame, orient="vertical", command=self.preview_list.yview)
        self.preview_list.configure(yscrollcommand=preview_scroll.set)
        self.preview_list.pack(side="left", fill="y")
        preview_scroll.pack(side="left", fill="y")
        self.preview_list.bind("<<ListboxSelect>>", self._on_preview_select)

        canvas_frame = ttk.Frame(preview_body)
        canvas_frame.grid(row=0, column=1, sticky="nsew", padx=(10, 0))
        canvas_frame.rowconfigure(0, weight=1)
        canvas_frame.columnconfigure(0, weight=1)
        self.preview_canvas = tk.Canvas(canvas_frame, highlightthickness=0)
        self.preview_vsb = ttk.Scrollbar(canvas_frame, orient="vertical", command=self.preview_canvas.yview)
        self.preview_hsb = ttk.Scrollbar(canvas_frame, orient="horizontal", command=self.preview_canvas.xview)
        self.preview_canvas.configure(yscrollcommand=self.preview_vsb.set, xscrollcommand=self.preview_hsb.set)
        self.preview_canvas.grid(row=0, column=0, sticky="nsew")
        self.preview_vsb.grid(row=0, column=1, sticky="ns")
        self.preview_hsb.grid(row=1, column=0, sticky="ew")
        self.preview_panel = ttk.Label(self.preview_canvas, text="No preview selected.", anchor="nw")
        self.preview_canvas_window = self.preview_canvas.create_window((0, 0), window=self.preview_panel, anchor="nw")
        self.preview_canvas.bind("<Configure>", self._on_preview_canvas_configure)

    def _build_help_tab(self):
        frame = ttk.Frame(self.tab_help)
        frame.pack(fill="both", expand=True, padx=8, pady=8)
        top = ttk.Frame(frame)
        top.pack(fill="x", pady=(0, 6))
        ttk.Button(top, text="Copy help text", command=self.copy_help_text).pack(side="left", padx=4)
        ttk.Button(top, text="Save help text", command=self.save_help_text).pack(side="left", padx=4)
        self.help_box = scrolledtext.ScrolledText(frame, wrap="word")
        self.help_box.pack(fill="both", expand=True)
        self.help_box.insert("1.0", HELP_TEXT)
        self.help_box.configure(state="disabled")

    def _trace_vars(self, vars_list, callback):
        for v in vars_list:
            v.trace_add("write", lambda *_args, cb=callback: cb())

    def _set_state_recursive(self, widget, state: str):
        try:
            widget.configure(state=state)
        except Exception:
            pass
        for child in widget.winfo_children():
            self._set_state_recursive(child, state)

    def _update_ui_mode_state(self):
        """Show the compact workflow or reveal the full numerical controls."""
        expert = self.ui_mode.get().strip().lower() == "expert"
        for frame in getattr(self, "basic_only_frames", []):
            if expert:
                frame.grid_remove()
            else:
                frame.grid()
        for frame in getattr(self, "expert_only_frames", []):
            if expert:
                frame.grid()
            else:
                frame.grid_remove()
        if hasattr(self, "analysis_mode_groups"):
            self._update_analysis_mode_state()
        self._update_run_plan_preview()

    def _update_smoothing_state(self):
        state = "readonly" if bool(self.pol_smooth_enabled.get()) else "disabled"
        try:
            self.pol_smooth_kernel_combo.configure(state=state)
            self.pol_smooth_width_entry.configure(
                state="normal" if bool(self.pol_smooth_enabled.get()) else "disabled"
            )
        except Exception:
            pass
        self._update_run_plan_preview()

    def _update_tess_mode_state(self):
        mode = self.tess_input_mode.get()
        active_key = "csv" if mode == "existing_csv" else "pipe" if mode in ("pipeline_dir", "pipeline_dir_batch") else "spoc"
        for key, frame in self.tess_mode_groups.items():
            if key == active_key:
                frame.grid()
                self._set_state_recursive(frame, "normal")
            else:
                frame.grid_remove()
        self._update_run_plan_preview()

    def _update_analysis_mode_state(self):
        mode = self.analysis_mode.get().strip()

        # Keep separate sensible defaults for guided vs joint mode, but do not
        # overwrite a custom user path.
        cur_out = self.outroot.get().strip()
        if mode == "guided_analysis":
            if cur_out in ("", "joint_analysis_outputs", "guided_joint_analysis_outputs", "tess_joint_outputs", "tess_guided_outputs"):
                self.outroot.set("guided_analysis_outputs")
        elif mode == "joint_search":
            if cur_out in ("", "guided_analysis_outputs", "guided_joint_analysis_outputs", "tess_guided_outputs", "tess_joint_outputs"):
                self.outroot.set("joint_analysis_outputs")

        for frame in self.analysis_mode_groups.get("guided", []):
            self._set_state_recursive(frame, "normal" if mode == "guided_analysis" else "disabled")
        for frame in self.analysis_mode_groups.get("joint", []):
            self._set_state_recursive(frame, "normal" if mode == "joint_search" else "disabled")

        # The two backends use different ranking column names for phase plots.
        if mode == "joint_search":
            phase_values = ["score_comb", "amp_tess", "amp_pol"]
            if self.phase_sort_by.get() not in phase_values:
                self.phase_sort_by.set("score_comb")
        else:
            phase_values = ["amp", "snr", "mode"]
            if self.phase_sort_by.get() not in phase_values:
                self.phase_sort_by.set("amp")
        try:
            self.phase_sort_by_combo.configure(values=phase_values, state="readonly")
            self.phase_plot_style_combo.configure(
                state="readonly" if mode == "joint_search" else "disabled"
            )
        except Exception:
            pass
        self._update_phase_zero_mode_state()

        self._update_run_plan_preview()

    def _update_polarimetry_state(self):
        guided_mode = (self.analysis_mode.get().strip() == "guided_analysis")
        # Joint analysis always requires polarimetry; Guided analysis may be
        # run in a TESS-only mode.
        use_pol = bool(self.use_polarimetry.get()) if guided_mode else True
        try:
            self.use_polarimetry_chk.configure(state=("normal" if guided_mode else "disabled"))
        except Exception:
            pass
        for w in getattr(self, "polarimetry_toggle_widgets", []):
            if w is None:
                continue
            try:
                if hasattr(w, "configure"):
                    if w.__class__.__name__.lower().endswith("combobox"):
                        w.configure(state=("readonly" if use_pol else "disabled"))
                    else:
                        w.configure(state=("normal" if use_pol else "disabled"))
            except Exception:
                pass
        self._update_run_plan_preview()

    def _update_joint_weight_mode_state(self):
        mode = self.joint_weight_mode.get().strip().lower()
        scale_state = "normal" if mode == "scale_free" else "disabled"
        manual_state = "normal" if mode == "manual" else "disabled"
        try:
            self.joint_scale_free_basis_combo.configure(state=("readonly" if mode == "scale_free" else "disabled"))
        except Exception:
            pass
        for w in [getattr(self, "joint_manual_tess_entry", None), getattr(self, "joint_manual_pol_entry", None)]:
            if w is None:
                continue
            try:
                w.configure(state=manual_state)
            except Exception:
                pass
        self._update_run_plan_preview()

    def _update_phase_zero_mode_state(self):
        mode = self.phase_zero_mode.get().strip().lower()
        joint = self.analysis_mode.get().strip() == "joint_search"
        state = "normal" if joint and mode == "custom_btjd" else "disabled"
        try:
            self.phase_zero_mode_combo.configure(state="readonly" if joint else "disabled")
            self.phase_zero_btjd_entry.configure(state=state)
        except Exception:
            pass
        self._update_run_plan_preview()

    def _channels_list(self):
        ch = []
        if self.channel_q.get():
            ch.append("q")
        if self.channel_u.get():
            ch.append("u")
        if self.channel_p.get():
            ch.append("p")
        return ch

    def _spoc_manifest_path(self) -> str:
        outdir = self.spoc_output_dir.get().strip()
        if not outdir:
            return ""
        return str((Path(outdir).expanduser().resolve() / "spoc_conversion_manifest.json"))

    def _build_converter_command(self) -> list[str] | None:
        if self.tess_input_mode.get() != "spoc_lc_fits":
            return None

        script = self.converter_script.get().strip()
        input_spec = self.spoc_input.get().strip()
        output_dir = self.spoc_output_dir.get().strip()
        pattern = self.spoc_pattern.get().strip() or "*lc.fits"
        if not script:
            raise ValueError("SPOC converter mode selected but converter script path is empty.")
        if not input_spec:
            raise ValueError("SPOC converter mode selected but the input file/directory/glob is empty.")

        # Catch the common accidental selection of a TPF before launching a
        # subprocess. Directory and glob inputs are validated by the converter.
        explicit_path = Path(input_spec).expanduser()
        if explicit_path.is_file() and _looks_like_tess_target_pixel_path(explicit_path):
            raise ValueError(_target_pixel_guidance(explicit_path))

        if not output_dir:
            raise ValueError("SPOC converter mode selected but the converted output directory is empty.")
        if not bool(self.spoc_combine_by_tic.get()) and not bool(self.spoc_write_per_sector.get()):
            raise ValueError("Enable at least one SPOC output type: combined by TIC or per-sector CSVs.")

        command = [
            sys.executable,
            "-u",
            script,
            "--input",
            input_spec,
            "--pattern",
            pattern,
            "--output-dir",
            str(Path(output_dir).expanduser().resolve()),
            "--flux",
            self.spoc_flux.get().strip(),
            "--quality",
            self.spoc_quality.get().strip(),
            "--normalization",
            self.spoc_normalization.get().strip(),
            "--normalization-sigma",
            str(float(self.spoc_normalization_sigma.get())),
            "--normalization-iters",
            str(int(self.spoc_normalization_iters.get())),
            "--cadence-policy",
            self.spoc_cadence_policy.get().strip(),
            "--manifest-json",
            self._spoc_manifest_path(),
        ]
        if bool(self.spoc_recursive.get()):
            command.append("--recursive")
        if bool(self.spoc_combine_by_tic.get()):
            command.append("--combine-by-tic")
        if bool(self.spoc_write_per_sector.get()):
            command.append("--write-per-sector")
        else:
            command.append("--no-write-per-sector")
        return command

    def _build_config(self) -> dict:
        channels = self._channels_list()
        analysis_track = self.analysis_mode.get().strip()
        use_polarimetry_active = bool(self.use_polarimetry.get()) if analysis_track == "guided_analysis" else True
        if self.tess_input_mode.get().strip() == "pipeline_dir_batch":
            if analysis_track != "guided_analysis":
                raise ValueError("Batch-one-file-at-a-time mode is available only for Guided photometry-only analysis.")
            use_polarimetry_active = False
            channels = []
        if use_polarimetry_active and not channels:
            raise ValueError("Select at least one polarimetric channel.")
        cfg = {
            "guided_script": self.guided_script.get().strip(),
            "joint_script": self.joint_script.get().strip(),
            "ui_mode": self.ui_mode.get().strip(),
            "analysis_mode": analysis_track,
            "tess_input_mode": self.tess_input_mode.get().strip(),
            "tess_csv": self.tess_csv.get().strip(),
            "pipeline_dir": self.pipeline_dir.get().strip(),
            "pipeline_pattern": self.pipeline_pattern.get().strip(),
            "pipeline_recursive": bool(self.pipeline_recursive.get()),
            "pipeline_batch_skip_existing": bool(self.pipeline_batch_skip_existing.get()),
            "pipeline_flux": self.pipeline_flux.get().strip(),
            "tess_force_y_col": self.tess_force_y_col.get().strip(),
            "spoc_input": self.spoc_input.get().strip(),
            "spoc_pattern": self.spoc_pattern.get().strip(),
            "spoc_recursive": bool(self.spoc_recursive.get()),
            "spoc_output_dir": self.spoc_output_dir.get().strip(),
            "spoc_flux": self.spoc_flux.get().strip(),
            "spoc_quality": self.spoc_quality.get().strip(),
            "spoc_normalization": self.spoc_normalization.get().strip(),
            "spoc_normalization_sigma": float(self.spoc_normalization_sigma.get()),
            "spoc_normalization_iters": int(self.spoc_normalization_iters.get()),
            "spoc_cadence_policy": self.spoc_cadence_policy.get().strip(),
            "spoc_combine_by_tic": bool(self.spoc_combine_by_tic.get()),
            "spoc_write_per_sector": bool(self.spoc_write_per_sector.get()),
            "use_polarimetry": use_polarimetry_active,
            "pol_csv": self.pol_csv.get().strip(),
            "pol_product": self.pol_product.get().strip(),
            "save_generated_frame": bool(self.save_generated_frame.get()),
            "generated_analysis_dir": self.generated_analysis_dir.get().strip(),
            "output_target_subdir": bool(self.output_target_subdir.get()),
            "outroot": self.outroot.get().strip(),
            "show_plots_inline": bool(self.show_plots_inline.get()),
            "verbose": int(self.verbose.get()),
            "lsq_verbose": int(self.lsq_verbose.get()),
            "fmin": float(self.fmin.get()),
            "fmax": float(self.fmax.get()),
            "tess_cap_fmax_to_nyquist": bool(self.tess_cap_fmax_to_nyquist.get()),
            "tess_weight_mode": self.tess_weight_mode.get().strip(),
            "tess_error_floor_frac": float(self.tess_error_floor_frac.get()),
            "tess_grid_mode": self.tess_grid_mode.get().strip(),
            "tess_snr_stop": float(self.tess_snr_stop.get()),
            "max_tess_modes": int(self.max_tess_modes.get()),
            "pol_snr_stop": float(self.pol_snr_stop.get()),
            "max_pol_modes": int(self.max_pol_modes.get()),
            "guided_pol_fmin": float(self.guided_pol_fmin.get()),
            "search_window_mult": float(self.search_window_mult.get()),
            "noise_ks": float(self.noise_ks.get()),
            "noise_bins": int(self.noise_bins.get()),
            "pol_smooth_enabled": bool(self.pol_smooth_enabled.get()),
            "pol_smooth_kernel": self.pol_smooth_kernel.get().strip().lower(),
            "pol_smooth_width": float(self.pol_smooth_width.get()),
            "channels": (channels if use_polarimetry_active else []),
            "use_offsets": bool(self.use_offsets.get()),
            "use_slopes": bool(self.use_slopes.get()),
            "group_mode": self.group_mode.get().strip(),
            "gap_hours": float(self.gap_hours.get()),
            "do_detrend": bool(self.do_detrend.get()),
            "detrend_order": int(self.detrend_order.get()),
            "n_phase_plots": int(self.n_phase_plots.get()),
            "phase_sort_by": self.phase_sort_by.get().strip(),
            "phase_plot_style": self.phase_plot_style.get().strip(),
            "phase_zero_mode": self.phase_zero_mode.get().strip(),
            "phase_zero_btjd": float(self.phase_zero_btjd.get()),
            "joint_k_candidates": int(self.joint_k_candidates.get()),
            "joint_top_n_raw_tess": int(self.joint_top_n_raw_tess.get()),
            "joint_coarse_oversample": float(self.joint_coarse_oversample.get()),
            "joint_refine_factor": int(self.joint_refine_factor.get()),
            "joint_max_iters": int(self.joint_max_iters.get()),
            "joint_kfit": float(self.joint_kfit.get()),
            "joint_snr_stop": float(self.joint_snr_stop.get()),
            "joint_w_prefilter": float(self.joint_w_prefilter.get()),
            "joint_ks_tess": float(self.joint_ks_tess.get()),
            "joint_ks_pol": float(self.joint_ks_pol.get()),
            "joint_trim_top_frac": float(self.joint_trim_top_frac.get()),
            "joint_weight_mode": self.joint_weight_mode.get().strip(),
            "joint_scale_free_basis": self.joint_scale_free_basis.get().strip(),
            "joint_manual_w_tess": float(self.joint_manual_w_tess.get()),
            "joint_manual_w_pol": float(self.joint_manual_w_pol.get()),
        }
        if cfg["analysis_mode"] == "guided_analysis":
            if not cfg["guided_script"]:
                raise ValueError("Guided-analysis script path is empty.")
        elif cfg["analysis_mode"] == "joint_search":
            if not cfg["joint_script"]:
                raise ValueError("Joint-search script path is empty.")
        else:
            raise ValueError(f"Unsupported analysis mode: {cfg['analysis_mode']}")
        if cfg["use_polarimetry"] and not cfg["pol_csv"]:
            raise ValueError("Polarimetry CSV path is empty.")
        if not cfg["outroot"]:
            raise ValueError("Output root is empty.")
        if cfg["tess_weight_mode"] not in {"none", "formal", "sector_rescaled"}:
            raise ValueError("TESS error weighting must be none, formal, or sector_rescaled.")
        if cfg["pol_product"] not in {"nm", "resid_nm_pchip", "pw_resid_nm_pchip"}:
            raise ValueError("Polarimetry product must be nm, resid_nm_pchip, or pw_resid_nm_pchip.")
        if cfg["fmin"] <= 0 or cfg["fmax"] <= cfg["fmin"]:
            raise ValueError("Frequency limits must satisfy 0 < Fmin < Fmax.")
        if cfg["tess_error_floor_frac"] < 0:
            raise ValueError("TESS error floor / median must be non-negative.")
        if cfg["pol_smooth_kernel"] not in {"gaussian", "boxcar"}:
            raise ValueError("Polarimetry smoothing kernel must be gaussian or boxcar.")
        if cfg["pol_smooth_width"] <= 0:
            raise ValueError("Polarimetry smoothing width must be positive.")
        if cfg["n_phase_plots"] < 0:
            raise ValueError("Number of phase plots cannot be negative.")

        if cfg["tess_input_mode"] == "existing_csv":
            if not cfg["tess_csv"]:
                raise ValueError("Existing CSV mode selected but no TESS CSV is set.")
            cfg["converter_command"] = None
        elif cfg["tess_input_mode"] == "pipeline_dir":
            if not cfg["pipeline_dir"]:
                raise ValueError("Pipeline/custom CSV mode selected but no pipeline directory is set.")
            cfg["converter_command"] = None
        elif cfg["tess_input_mode"] == "pipeline_dir_batch":
            if not cfg["pipeline_dir"]:
                raise ValueError("Batch pipeline/custom CSV mode selected but no pipeline directory is set.")
            cfg["converter_command"] = None
        elif cfg["tess_input_mode"] == "spoc_lc_fits":
            cfg["converter_command"] = self._build_converter_command()
            cfg["converter_manifest"] = self._spoc_manifest_path()
            # The temporary runner replaces this with the preferred CSV from
            # the converter manifest before importing either analysis backend.
            cfg["tess_csv"] = ""
        else:
            raise ValueError(f"Unsupported TESS input mode: {cfg['tess_input_mode']}")
        return cfg

    def _update_run_plan_preview(self):
        try:
            cfg = self._build_config()
        except Exception as exc:
            preview = f"<invalid configuration: {exc}>"
        else:
            lines = [
                f"Analysis mode: {cfg['analysis_mode']}",
                f"Guided script: {cfg['guided_script']}",
                f"Joint script: {cfg['joint_script']}",
                f"TESS input mode: {cfg['tess_input_mode']}",
            ]
            if cfg["tess_input_mode"] == "existing_csv":
                lines.append(f"TESS CSV: {cfg['tess_csv']}")
            elif cfg["tess_input_mode"] == "pipeline_dir":
                lines.append(f"Pipeline dir: {cfg['pipeline_dir']}")
                lines.append(f"Pattern: {cfg['pipeline_pattern']} | recursive={cfg['pipeline_recursive']} | flux={cfg['pipeline_flux']}")
            elif cfg["tess_input_mode"] == "pipeline_dir_batch":
                lines.append(f"Batch pipeline dir: {cfg['pipeline_dir']}")
                lines.append(f"Pattern: {cfg['pipeline_pattern']} | recursive={cfg['pipeline_recursive']} | flux={cfg['pipeline_flux']} | skip_existing={cfg['pipeline_batch_skip_existing']}")
            else:
                lines.append("SPOC converter command:")
                lines.append("  " + quote_cmd(cfg["converter_command"]))
                lines.append(f"Converted output directory: {cfg['spoc_output_dir']}")
                lines.append(f"Manifest: {cfg['converter_manifest']}")
                lines.append(
                    "Conversion: "
                    f"flux={cfg['spoc_flux']} | quality={cfg['spoc_quality']} | "
                    f"normalization={cfg['spoc_normalization']} | "
                    f"cadence={cfg['spoc_cadence_policy']} | "
                    f"combine_by_tic={cfg['spoc_combine_by_tic']}"
                )
            lines.extend([
                f"Use polarimetry: {cfg['use_polarimetry']}",
                f"Polarimetry CSV: {cfg['pol_csv'] if cfg['use_polarimetry'] else '(not used)'}",
                f"POL product: {cfg['pol_product'] if cfg['use_polarimetry'] else '(not used)'}",
                f"Channels: {', '.join(cfg['channels']) if cfg['use_polarimetry'] else '(none)'}",
                "Polarimetry smoothing: " + (
                    f"{cfg['pol_smooth_kernel']}, width={cfg['pol_smooth_width']:g} resolution elements"
                    if cfg["pol_smooth_enabled"] else "off"
                ),
                f"Output root: {cfg['outroot']}",
                f"Target subdirectory: {cfg['output_target_subdir']}",
                f"Cap Fmax to Nyquist: {cfg['tess_cap_fmax_to_nyquist']}",
                f"TESS weighting: {cfg['tess_weight_mode']} | error floor/median={cfg['tess_error_floor_frac']}",
            ])
            if cfg["analysis_mode"] == "guided_analysis":
                if cfg["tess_input_mode"] == "pipeline_dir_batch":
                    lines.append("Run style: wrapper script imports guided module and runs one photometry-only guided analysis per matching CSV.")
                else:
                    lines.append("Run style: wrapper script imports guided module, applies settings, and calls run_analysis().")
            else:
                lines.append(
                    f"Phase zero: {cfg['phase_zero_mode']}" +
                    (f" (BTJD={cfg['phase_zero_btjd']})" if cfg['phase_zero_mode'] == 'custom_btjd' else "")
                )
                lines.append("Run style: wrapper imports the joint backend, applies settings, and calls run_joint_analysis().")
                lines.append(
                    "Joint settings: "
                    f"Kcand={cfg['joint_k_candidates']}, TopRawTESS={cfg['joint_top_n_raw_tess']}, "
                    f"MaxIters={cfg['joint_max_iters']}, SNRstop={cfg['joint_snr_stop']}, "
                    f"WeightMode={cfg['joint_weight_mode']}"
                )
                if cfg['joint_weight_mode'] == 'manual':
                    lines.append(
                        f"Manual weights: TESS={cfg['joint_manual_w_tess']}, POL={cfg['joint_manual_w_pol']}"
                    )
            preview = "\n".join(lines)
        self.plan_text.configure(state="normal")
        self.plan_text.delete("1.0", "end")
        self.plan_text.insert("1.0", preview)
        self.plan_text.configure(state="disabled")


    def _runner_script_text(self, cfg: dict) -> str:
        cfg_json = json.dumps(cfg, indent=2)
        return f'''#!/usr/bin/env python3
from __future__ import annotations
import importlib.util
import json
import os
import shlex
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path

import pandas as pd

CFG = json.loads(r"""{cfg_json}""")

# Some managed or read-only installations do not provide a writable default
# Matplotlib configuration directory.  Keep GUI-launched jobs self-contained.
mpl_config_dir = Path(tempfile.gettempdir()) / "tess_guided_analysis_matplotlib"
mpl_config_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))

print("GUI runner starting")
print("Analysis mode =", CFG["analysis_mode"])

if CFG.get("converter_command"):
    manifest_path = Path(CFG.get("converter_manifest", "")).expanduser().resolve()
    try:
        manifest_path.unlink()
    except FileNotFoundError:
        pass

    print("Running SPOC converter:")
    print("  " + " ".join(shlex.quote(str(arg)) for arg in CFG["converter_command"]))
    rc = subprocess.run(CFG["converter_command"]).returncode
    if rc != 0:
        raise SystemExit(f"SPOC converter failed with exit code {{rc}}")

    if not manifest_path.exists():
        raise SystemExit(f"SPOC converter finished but did not write its manifest: {{manifest_path}}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    preferred = manifest.get("preferred_analysis_csv")
    if not preferred:
        candidates = manifest.get("combined_outputs") or manifest.get("per_sector_outputs") or []
        if len(candidates) > 1:
            listing = "\\n".join(f"  - {{item}}" for item in candidates[:20])
            raise SystemExit(
                "SPOC conversion produced more than one possible analysis CSV, usually because "
                "the input contained multiple TICs or sector combination was disabled. "
                "Select inputs for one TIC or choose one converted CSV explicitly.\\n" + listing
            )
        if len(candidates) == 1:
            preferred = candidates[0]
    if not preferred:
        failures = manifest.get("failures") or []
        raise SystemExit(
            "SPOC conversion did not produce a usable analysis CSV. "
            f"Manifest failures: {{failures[:5]}}"
        )
    preferred_path = Path(preferred).expanduser().resolve()
    if not preferred_path.exists():
        raise SystemExit(f"Manifest-selected SPOC CSV does not exist: {{preferred_path}}")
    CFG["tess_csv"] = str(preferred_path)
    print("SPOC conversion complete; analysis input CSV:")
    print("  ", preferred_path)

def _load_guided_module():
    script_path = Path(CFG["guided_script"]).expanduser().resolve()
    if not script_path.exists():
        raise SystemExit(f"Guided-analysis script not found: {{script_path}}")
    spec = importlib.util.spec_from_file_location("guided_analysis_gui_module", script_path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"Could not import guided-analysis script: {{script_path}}")
    mod = importlib.util.module_from_spec(spec)
    # Dataclasses and postponed annotations expect an executing module to be
    # registered, just as it is during a normal Python import.
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod

def _apply_guided_common_settings(mod, *, outroot_override: Path | None = None, force_use_polarimetry: bool | None = None):
    use_pol = bool(CFG["use_polarimetry"]) if force_use_polarimetry is None else bool(force_use_polarimetry)
    mod.USE_POLARIMETRY = use_pol
    mod.POL_CSV = Path(CFG["pol_csv"]) if CFG["pol_csv"] else Path(".")
    mod.POL_PRODUCT = CFG["pol_product"]
    mod.POL_SAVE_GENERATED_ANALYSIS_FRAME = bool(CFG["save_generated_frame"])
    mod.POL_GENERATED_ANALYSIS_DIR = (None if not CFG["generated_analysis_dir"] else str(CFG["generated_analysis_dir"]))
    mod.OUTROOT = Path(CFG["outroot"]) if outroot_override is None else Path(outroot_override)
    mod.OUTROOT.mkdir(parents=True, exist_ok=True)
    if hasattr(mod, "OUTPUT_TARGET_SUBDIR"):
        mod.OUTPUT_TARGET_SUBDIR = bool(CFG.get("output_target_subdir", False))
    mod.SHOW_PLOTS_INLINE = bool(CFG["show_plots_inline"])
    mod.VERBOSE = int(CFG["verbose"])
    mod.LSQ_VERBOSE = int(CFG["lsq_verbose"])

    mod.FMIN = float(CFG["fmin"])
    mod.FMAX = float(CFG["fmax"])
    setattr(mod, "TESS_CAP_FMAX_TO_NYQUIST", bool(CFG.get("tess_cap_fmax_to_nyquist", True)))
    setattr(mod, "TESS_WEIGHT_MODE", CFG.get("tess_weight_mode", "sector_rescaled"))
    setattr(mod, "TESS_ERROR_FLOOR_FRAC", float(CFG.get("tess_error_floor_frac", 0.25)))
    mod.TESS_GRID_MODE = CFG["tess_grid_mode"]
    mod.TESS_SNR_STOP = float(CFG["tess_snr_stop"])
    mod.MAX_TESS_MODES = int(CFG["max_tess_modes"])
    mod.POL_SNR_STOP = float(CFG["pol_snr_stop"])
    mod.MAX_POL_MODES = int(CFG["max_pol_modes"])
    mod.GUIDED_POL_FMIN = float(CFG["guided_pol_fmin"])
    mod.POL_SEARCH_WINDOW_MULT = float(CFG["search_window_mult"])
    mod.POL_LOCAL_NOISE_KS = float(CFG["noise_ks"])
    mod.POL_LOCAL_NOISE_SIDE_BINS = int(CFG["noise_bins"])
    mod.POL_SMOOTH_ENABLED = bool(CFG["pol_smooth_enabled"])
    mod.POL_SMOOTH_KERNEL = CFG["pol_smooth_kernel"]
    mod.POL_SMOOTH_WIDTH_RES_ELEMS = float(CFG["pol_smooth_width"])
    mod.POL_CHANNELS = list(CFG["channels"]) if use_pol else []
    mod.USE_POL_NIGHT_OFFSETS = bool(CFG["use_offsets"])
    mod.USE_POL_NIGHT_SLOPES = bool(CFG["use_slopes"])
    mod.POL_NIGHT_GROUP_MODE = CFG["group_mode"]
    mod.POL_NIGHT_GAP_HOURS = float(CFG["gap_hours"])
    mod.DO_DETREND = bool(CFG["do_detrend"])
    mod.DETREND_POLY_ORDER = int(CFG["detrend_order"])
    mod.N_PHASE_PLOTS = int(CFG["n_phase_plots"])
    mod.PHASE_SORT_BY = CFG["phase_sort_by"]

def _configure_guided_tess_input_for_mode(mod, mode: str, csv_path: Path | None = None):
    if mode == "existing_csv":
        mod.TESS_INPUT_MODE = "spoc_csv"
        mod.TESS_CSV = Path(CFG["tess_csv"])
    elif mode == "pipeline_dir":
        mod.TESS_INPUT_MODE = "pipeline_dir"
        mod.TESS_PIPELINE_DIR = Path(CFG["pipeline_dir"])
        mod.TESS_PIPELINE_PATTERN = CFG["pipeline_pattern"]
        mod.TESS_PIPELINE_RECURSIVE = bool(CFG["pipeline_recursive"])
        mod.TESS_PIPELINE_FLUX = CFG["pipeline_flux"]
    elif mode == "pipeline_dir_batch":
        if csv_path is None:
            raise SystemExit("pipeline_dir_batch requires a csv_path")
        mod.TESS_INPUT_MODE = "spoc_csv"
        mod.TESS_CSV = Path(csv_path)
    elif mode == "spoc_lc_fits":
        mod.TESS_INPUT_MODE = "spoc_csv"
        mod.TESS_CSV = Path(CFG["tess_csv"])
    else:
        raise SystemExit(f"Unsupported TESS input mode: {{mode}}")
    setattr(mod, "TESS_FORCE_Y_COL", CFG["tess_force_y_col"] if CFG["tess_force_y_col"] else None)

if CFG["analysis_mode"] == "guided_analysis":
    mode = CFG["tess_input_mode"]
    if mode != "pipeline_dir_batch":
        mod = _load_guided_module()
        _apply_guided_common_settings(mod)
        _configure_guided_tess_input_for_mode(mod, mode)
        print("Running guided analysis...")
        outputs = mod.run_analysis()
        if isinstance(outputs, dict):
            print("Output keys:", sorted(outputs.keys()))
        print("GUI runner finished")
    else:
        pipe_root = Path(CFG["pipeline_dir"]).expanduser()
        if not pipe_root.exists():
            raise SystemExit(f"Batch pipeline directory not found: {{pipe_root}}")
        pattern = CFG["pipeline_pattern"]
        files = sorted(pipe_root.rglob(pattern) if CFG["pipeline_recursive"] else pipe_root.glob(pattern))
        files = [p for p in files if p.is_file()]
        if not files:
            raise SystemExit(f"No CSV files found in {{pipe_root}} matching {{pattern!r}}")

        outroot = Path(CFG["outroot"]).expanduser()
        outroot.mkdir(parents=True, exist_ok=True)
        print(f"Guided batch mode: {{len(files)}} CSV file(s) found")
        print("  Input dir =", pipe_root)
        print("  Output root =", outroot)

        n_ok = 0
        n_skip = 0
        n_fail = 0
        failures = []

        for i, csv_path in enumerate(files, start=1):
            print("\\n" + "-" * 88)
            print(f"[{{i}}/{{len(files)}}] CSV: {{csv_path}}")
            target_out = outroot / csv_path.stem
            existing = list(target_out.rglob("*peaks_table*.csv")) if target_out.exists() else []
            if bool(CFG.get("pipeline_batch_skip_existing", False)) and existing:
                print(f"  [SKIP] Existing output found: {{existing[0]}}")
                n_skip += 1
                continue

            mod = _load_guided_module()
            _apply_guided_common_settings(mod, outroot_override=target_out, force_use_polarimetry=False)
            _configure_guided_tess_input_for_mode(mod, "pipeline_dir_batch", csv_path=csv_path)

            try:
                outputs = mod.run_analysis()
                if isinstance(outputs, dict):
                    print("  [OK] Output keys:", sorted(outputs.keys()))
                n_ok += 1
            except Exception as exc:
                n_fail += 1
                failures.append((str(csv_path), str(exc)))
                print(f"  [FAIL] {{type(exc).__name__}}: {{exc}}")
                traceback.print_exc()
                continue

        print("\\n" + "=" * 88)
        print(f"Batch summary: success={{n_ok}} | skipped={{n_skip}} | failed={{n_fail}} | total={{len(files)}}")
        if failures:
            print("Failed files:")
            for name, reason in failures[:20]:
                print("  -", name, "->", reason)
            if len(failures) > 20:
                print(f"  ... and {{len(failures) - 20}} more")
        print("GUI runner finished")

elif CFG["analysis_mode"] == "joint_search":
    script_path = Path(CFG["joint_script"]).expanduser().resolve()
    if not script_path.exists():
        raise SystemExit(f"Joint-search script not found: {{script_path}}")

    spec = importlib.util.spec_from_file_location("joint_analysis_gui_module", script_path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"Could not import joint-analysis script: {{script_path}}")
    jmod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = jmod
    spec.loader.exec_module(jmod)

    if CFG["tess_input_mode"] == "pipeline_dir":
        tess_input_mode = "pipeline_dir"
        tess_pipeline_dir = CFG["pipeline_dir"]
        tess_pipeline_pattern = CFG["pipeline_pattern"]
        tess_pipeline_recursive = bool(CFG["pipeline_recursive"])
        tess_pipeline_flux = CFG["pipeline_flux"]
    else:
        tess_input_mode = "pipeline_dir"
        tcsv = Path(CFG["tess_csv"]).expanduser()
        tess_pipeline_dir = str(tcsv.parent)
        tess_pipeline_pattern = tcsv.name
        tess_pipeline_recursive = False
        tess_pipeline_flux = "auto"
        print("Joint mode: treating single TESS CSV as pipeline_dir input:")
        print("  dir    =", tess_pipeline_dir)
        print("  pattern=", tess_pipeline_pattern)

    pol_csv_for_joint = str(Path(CFG["pol_csv"]).expanduser().resolve())
    pol_product = CFG["pol_product"]
    pol_required = {{
        "nm": ["q_nm", "u_nm", "p_nm"],
        "resid_nm_pchip": ["q_resid_nm_pchip", "u_resid_nm_pchip", "p_resid_nm_pchip"],
        "pw_resid_nm_pchip": ["q_pw_resid_nm_pchip", "u_pw_resid_nm_pchip", "p_pw_resid_nm_pchip"],
    }}
    pol_head = pd.read_csv(pol_csv_for_joint, nrows=5)
    # Only require columns for channels the user selected.  A valid q-only
    # analysis frame should not be rejected merely because it lacks u or p.
    chosen_channels = list(CFG["channels"])
    product_columns = pol_required.get(pol_product, [])
    by_channel = dict(zip(("q", "u", "p"), product_columns))
    needed = ["jd"] + [by_channel[c] for c in chosen_channels] + [f"{{c}}_err" for c in chosen_channels]
    missing = [c for c in needed if c not in pol_head.columns]
    if missing:
        print("Joint mode: polarimetry CSV is not already an analysis frame; generating one from raw/basic polarimetry input.")
        print("  missing columns:", missing)
        guided_path = Path(CFG["guided_script"]).expanduser().resolve()
        if not guided_path.exists():
            raise SystemExit(
                "Joint mode needs the guided-analysis script path in order to preprocess raw polarimetry into an analysis frame."
            )
        spec = importlib.util.spec_from_file_location("guided_preprocess_for_joint", guided_path)
        if spec is None or spec.loader is None:
            raise SystemExit(f"Could not import guided-analysis script for polarimetry preprocessing: {{guided_path}}")
        gmod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = gmod
        spec.loader.exec_module(gmod)

        gmod.OUTROOT = Path(CFG["outroot"])
        gmod.OUTROOT.mkdir(parents=True, exist_ok=True)
        gmod.SHOW_PLOTS_INLINE = bool(CFG["show_plots_inline"])

        df_pol = gmod.build_analysis_frame_from_raw_pol(Path(pol_csv_for_joint), save_generated=False)
        temp_pol_dir = Path(tempfile.gettempdir()) / f"joint_pol_analysis_frames_{{os.getpid()}}"
        temp_pol_dir.mkdir(parents=True, exist_ok=True)
        temp_pol_path = temp_pol_dir / f"{{Path(pol_csv_for_joint).stem}}_nm_pchip_analysis_frame.csv"
        df_pol.to_csv(temp_pol_path, index=False)
        pol_csv_for_joint = str(temp_pol_path)
        print("  wrote generated analysis frame:", pol_csv_for_joint)
    else:
        print("Joint mode: using existing polarimetry analysis-frame CSV:")
        print("  ", pol_csv_for_joint)

    jmod.TESS_INPUT_MODE = tess_input_mode
    jmod.TESS_CSV = Path(CFG["tess_csv"])
    jmod.TESS_PIPELINE_DIR = Path(tess_pipeline_dir)
    jmod.TESS_PIPELINE_PATTERN = tess_pipeline_pattern
    jmod.TESS_PIPELINE_RECURSIVE = bool(tess_pipeline_recursive)
    jmod.TESS_PIPELINE_FLUX = tess_pipeline_flux
    jmod.TESS_FORCE_Y_COL = CFG["tess_force_y_col"] or None
    jmod.TESS_WEIGHT_MODE = CFG.get("tess_weight_mode", "sector_rescaled")
    jmod.TESS_ERROR_FLOOR_FRAC = float(CFG.get("tess_error_floor_frac", 0.25))
    jmod.POL_CSV = Path(pol_csv_for_joint)
    jmod.POL_PRODUCT = pol_product
    jmod.FMIN = float(CFG["fmin"])
    jmod.FMAX = float(CFG["fmax"])
    jmod.TESS_CAP_FMAX_TO_NYQUIST = bool(CFG.get("tess_cap_fmax_to_nyquist", True))
    jmod.K_CANDIDATES = int(CFG["joint_k_candidates"])
    jmod.TOP_N_RAW_TESS_CANDIDATES = int(CFG["joint_top_n_raw_tess"])
    jmod.COARSE_OVERSAMPLE = float(CFG["joint_coarse_oversample"])
    jmod.REFINE_FACTOR = int(CFG["joint_refine_factor"])
    jmod.MAX_ITERS = int(CFG["joint_max_iters"])
    jmod.KFIT = float(CFG["joint_kfit"])
    jmod.SNR_STOP = float(CFG["joint_snr_stop"])
    jmod.W_PREFILTER = float(CFG["joint_w_prefilter"])
    jmod.KS_TESS = float(CFG["joint_ks_tess"])
    jmod.KS_POL = float(CFG["joint_ks_pol"])
    jmod.TRIM_TOP_FRAC = float(CFG["joint_trim_top_frac"])
    jmod.USE_POL_NIGHT_OFFSETS = bool(CFG["use_offsets"])
    jmod.USE_POL_NIGHT_SLOPES = bool(CFG["use_slopes"])
    jmod.POL_NIGHT_GROUP_MODE = CFG["group_mode"]
    jmod.POL_NIGHT_GAP_HOURS = float(CFG["gap_hours"])
    jmod.DO_DETREND = bool(CFG["do_detrend"])
    jmod.DETREND_POLY_ORDER = int(CFG["detrend_order"])
    jmod.N_PHASE_PLOTS = int(CFG["n_phase_plots"])
    jmod.PHASE_SORT_BY = CFG["phase_sort_by"]
    jmod.PHASE_PLOT_STYLE = CFG["phase_plot_style"]
    jmod.PHASE_ZERO_MODE = CFG["phase_zero_mode"]
    jmod.PHASE_ZERO_BTJD = float(CFG["phase_zero_btjd"])
    jmod.JOINT_WEIGHT_MODE = CFG["joint_weight_mode"]
    jmod.SCALE_FREE_WEIGHT_BASIS = CFG["joint_scale_free_basis"]
    jmod.MANUAL_W_TESS = float(CFG["joint_manual_w_tess"])
    jmod.MANUAL_W_POL = float(CFG["joint_manual_w_pol"])
    jmod.POL_SMOOTH_ENABLED = bool(CFG["pol_smooth_enabled"])
    jmod.POL_SMOOTH_KERNEL = CFG["pol_smooth_kernel"]
    jmod.POL_SMOOTH_WIDTH_RES_ELEMS = float(CFG["pol_smooth_width"])
    jmod.SHOW_PLOTS_INLINE = bool(CFG["show_plots_inline"])
    jmod.OUTROOT = Path(CFG["outroot"])
    jmod.OUTROOT.mkdir(parents=True, exist_ok=True)

    print("Running joint analysis through run_joint_analysis()...")
    results = jmod.run_joint_analysis(channels=CFG["channels"])
    print("Output keys:", sorted(results.keys()))
    print("GUI runner finished")

else:
    raise SystemExit(f"Unsupported analysis mode: {{CFG['analysis_mode']}}")
'''


    def run_analysis_job(self):
        try:
            cfg = self._build_config()
        except Exception as exc:
            messagebox.showerror("Invalid configuration", str(exc))
            return

        runner_text = self._runner_script_text(cfg)
        runner_dir = Path(tempfile.gettempdir())
        runner_path = runner_dir / f"guided_analysis_gui_runner_{os.getpid()}.py"
        runner_path.write_text(runner_text, encoding="utf-8")
        self._temp_runner_path = runner_path

        cmd = [sys.executable, "-u", str(runner_path)]
        job_name = "Joint analysis" if cfg["analysis_mode"] == "joint_search" else "Guided analysis"
        self._run_subprocess(job_name, cmd, Path(cfg["outroot"]))

    def _run_subprocess(self, job_name: str, cmd: list[str], output_dir: Path):
        if self.current_process is not None:
            messagebox.showwarning("Job already running", "Stop the current process before starting another one.")
            return
        self.current_output_dir = output_dir
        self.preview_dir.set(str(output_dir))
        self.status_text.set(f"{job_name} running...")
        self.notebook.select(self.tab_run)
        self.log_text.insert("end", f"\n=== {job_name} ===\n")
        self.log_text.insert("end", quote_cmd(cmd) + "\n\n")
        self.log_text.see("end")
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        popen_kwargs = {
            "stdout": subprocess.PIPE,
            "stderr": subprocess.STDOUT,
            "text": True,
            "bufsize": 1,
            "env": env,
        }
        if os.name == "nt":
            # Give the analysis job its own Windows process group.
            popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            # Put the runner and all subprocesses it launches into a new Unix
            # session/process group so Stop cannot signal the GUI or terminal.
            popen_kwargs["start_new_session"] = True

        try:
            proc = subprocess.Popen(cmd, **popen_kwargs)
            self.current_process = proc
        except Exception as exc:
            self.current_process = None
            self.status_text.set("Ready.")
            messagebox.showerror("Failed to start process", str(exc))
            return

        def reader_thread(proc=proc):
            try:
                if proc.stdout is not None:
                    for line in proc.stdout:
                        self.log_queue.put(("line", line))
                rc = proc.wait()
                self.log_queue.put(("done", (proc, f"{job_name} finished with exit code {rc}.\n")))
            except Exception as exc:
                self.log_queue.put(("done", (proc, f"{job_name} failed: {exc}\n")))

        threading.Thread(target=reader_thread, daemon=True).start()

    def _force_kill_process(self, proc: subprocess.Popen):
        """Force-kill a still-running analysis process and its descendants."""
        if proc.poll() is not None:
            return
        try:
            if os.name == "nt":
                subprocess.run(
                    ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                )
            else:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass

    def stop_current_process(self):
        proc = self.current_process
        if proc is None or proc.poll() is not None:
            return
        try:
            if os.name == "nt":
                # /T includes subprocesses launched by the temporary runner.
                subprocess.run(
                    ["taskkill", "/PID", str(proc.pid), "/T"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                )
            else:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                except ProcessLookupError:
                    return
                except Exception:
                    proc.terminate()
            self.status_text.set("Stopping process...")
            # Escalate only if the process group has not exited after 3 s.
            self.after(3000, lambda p=proc: self._force_kill_process(p))
        except Exception as exc:
            messagebox.showerror("Stop failed", str(exc))

    def _poll_log_queue(self):
        try:
            while True:
                kind, payload = self.log_queue.get_nowait()
                if kind == "line":
                    self.log_text.insert("end", payload)
                    self.log_text.see("end")
                elif kind == "done":
                    proc, message = payload
                    self.log_text.insert("end", "\n" + str(message) + "\n")
                    self.log_text.see("end")
                    # Do not let a delayed completion message from an older job
                    # clear the state of a newer process.
                    if self.current_process is proc:
                        self.current_process = None
                        self.status_text.set("Ready.")
                        self._refresh_preview_list()
        except queue.Empty:
            pass
        self.after(150, self._poll_log_queue)

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
        pngs = sorted(p.rglob("*.png"))
        self._preview_paths = pngs
        for item in pngs:
            try:
                rel = item.relative_to(p)
            except Exception:
                rel = item.name
            self.preview_list.insert("end", str(rel))
        if pngs:
            self.preview_list.selection_set(0)
            self._show_preview(pngs[0])
        else:
            self.preview_panel.configure(text="No PNG files found under preview folder.", image="")

    def _on_preview_select(self, event=None):
        sel = self.preview_list.curselection()
        if not sel:
            return
        idx = int(sel[0])
        if 0 <= idx < len(self._preview_paths):
            self._show_preview(self._preview_paths[idx])

    def _show_preview(self, path: Path):
        try:
            self.preview_source_image = Image.open(path).convert("RGBA")
            self.preview_source_path = path
            self._render_current_preview()
        except Exception as exc:
            self.preview_source_image = None
            self.preview_source_path = path
            self.preview_image = None
            self.preview_panel.configure(image="", text=f"Could not preview:\n{path.name}\n\n{exc}")
            self.preview_canvas.update_idletasks()
            self.preview_canvas.configure(scrollregion=self.preview_canvas.bbox("all"))

    def _on_preview_canvas_configure(self, event=None):
        self.preview_canvas.configure(scrollregion=self.preview_canvas.bbox("all"))
        if self.preview_scale_mode.get().strip().lower() == "fit":
            self._render_current_preview()

    def _on_app_mousewheel(self, event):
        """Route wheel input to the visible settings or preview region."""
        try:
            selected = self.nametowidget(self.notebook.select())
            units = _wheel_scroll_units(event)
            if selected is self.tab_analysis:
                if units:
                    self.analysis_scroll_canvas.yview_scroll(units, "units")
                return "break"
            if selected is self.tab_tess:
                if units:
                    self.tess_scroll_canvas.yview_scroll(units, "units")
                return "break"
            if selected is not self.tab_run:
                return None

            # Preserve ordinary list scrolling when the pointer is over the
            # PNG selector.  Elsewhere in Run / Preview, modifiers control the
            # image pane (Shift = horizontal, Control = zoom).
            if event.widget is self.preview_list:
                if units:
                    self.preview_list.yview_scroll(units, "units")
                return "break"
            state = int(getattr(event, "state", 0) or 0)
            if state & 0x4:  # Control / Command-style zoom modifier
                self._on_preview_zoom_mousewheel(event)
            elif state & 0x1:  # Shift
                self._on_preview_shift_mousewheel(event)
            else:
                self._on_preview_mousewheel(event)
            return "break"
        except Exception:
            return None

    def _on_preview_mousewheel(self, event):
        try:
            units = _wheel_scroll_units(event)
            if units:
                self.preview_canvas.yview_scroll(units, "units")
        except Exception:
            pass

    def _on_preview_shift_mousewheel(self, event):
        try:
            units = _wheel_scroll_units(event)
            if units:
                self.preview_canvas.xview_scroll(units, "units")
        except Exception:
            pass

    def _on_preview_zoom_mousewheel(self, event):
        try:
            steps = 1 if event.delta > 0 else -1
            self._change_preview_zoom(steps)
        except Exception:
            pass

    def clear_log(self):
        self.log_text.delete("1.0", "end")

    def close_application(self):
        """Close the GUI without leaving an analysis subprocess behind."""
        proc = self.current_process
        if proc is not None and proc.poll() is None:
            if not messagebox.askyesno(
                "Analysis is running",
                "Stop the current analysis and close the application?",
            ):
                return
            self.stop_current_process()
        self.destroy()

    def save_log(self):
        path = filedialog.asksaveasfilename(title="Save log", defaultextension=".txt", filetypes=[("Text files", "*.txt"), ("All files", "*.*")])
        if not path:
            return
        Path(path).write_text(self.log_text.get("1.0", "end"), encoding="utf-8")

    def copy_run_plan(self):
        txt = self.plan_text.get("1.0", "end")
        self.clipboard_clear()
        self.clipboard_append(txt)

    def show_help_tab(self):
        self.notebook.select(self.tab_help)

    def copy_help_text(self):
        self.clipboard_clear()
        self.clipboard_append(HELP_TEXT)

    def save_help_text(self):
        path = filedialog.asksaveasfilename(title="Save help text", defaultextension=".txt", filetypes=[("Text files", "*.txt"), ("All files", "*.*")])
        if not path:
            return
        Path(path).write_text(HELP_TEXT, encoding="utf-8")

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

    def _browse_spoc_lc_file(self):
        path = filedialog.askopenfilename(
            title="Choose a SPOC light-curve FITS file",
            filetypes=[
                ("SPOC light-curve FITS", ("*lc.fits", "*lc.fits.gz")),
                ("All FITS files", ("*.fits", "*.fits.gz")),
            ],
        )
        if not path:
            return
        if _looks_like_tess_target_pixel_path(path):
            messagebox.showerror("Target-pixel file selected", _target_pixel_guidance(path))
            return
        if not _looks_like_spoc_lightcurve_path(path):
            proceed = messagebox.askyesno(
                "Unrecognized light-curve filename",
                f"{Path(path).name} does not have the usual *-lc.fits or *-fast-lc.fits name.\n\n"
                "Continue only if this FITS file contains a standard SPOC light-curve table "
                "with SAP_FLUX or PDCSAP_FLUX columns.",
            )
            if not proceed:
                return
        self.spoc_input.set(path)

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

    def _settings_dict(self) -> dict:
        keys = [
            "guided_script", "joint_script", "converter_script", "ui_mode", "analysis_mode",
            "tess_input_mode", "tess_csv", "pipeline_dir", "pipeline_pattern",
            "pipeline_recursive", "pipeline_batch_skip_existing", "pipeline_flux",
            "tess_force_y_col", "spoc_input", "spoc_pattern", "spoc_recursive",
            "spoc_output_dir", "spoc_flux", "spoc_quality", "spoc_normalization",
            "spoc_normalization_sigma", "spoc_normalization_iters",
            "spoc_cadence_policy", "spoc_combine_by_tic", "spoc_write_per_sector",
            "use_polarimetry", "pol_csv", "pol_product", "save_generated_frame",
            "generated_analysis_dir", "output_target_subdir", "outroot",
            "show_plots_inline", "verbose", "lsq_verbose", "fmin", "fmax",
            "tess_cap_fmax_to_nyquist", "tess_weight_mode", "tess_error_floor_frac",
            "tess_grid_mode", "tess_snr_stop",
            "max_tess_modes", "pol_snr_stop", "max_pol_modes", "guided_pol_fmin",
            "search_window_mult", "noise_ks", "noise_bins", "pol_smooth_enabled",
            "pol_smooth_kernel", "pol_smooth_width", "use_offsets",
            "use_slopes", "group_mode", "gap_hours", "do_detrend", "detrend_order",
            "n_phase_plots", "phase_sort_by", "phase_plot_style", "phase_zero_mode",
            "phase_zero_btjd",
            "joint_k_candidates", "joint_top_n_raw_tess", "joint_coarse_oversample",
            "joint_refine_factor", "joint_max_iters", "joint_kfit",
            "joint_snr_stop", "joint_w_prefilter", "joint_ks_tess", "joint_ks_pol",
            "joint_trim_top_frac", "joint_weight_mode", "joint_scale_free_basis",
            "joint_manual_w_tess", "joint_manual_w_pol", "preview_dir",
            "preview_scale_mode", "preview_zoom",
        ]
        out = {}
        for k in keys:
            if hasattr(self, k):
                try:
                    out[k] = getattr(self, k).get()
                except Exception:
                    pass
        out["channel_q"] = self.channel_q.get()
        out["channel_u"] = self.channel_u.get()
        out["channel_p"] = self.channel_p.get()
        return out

    def save_settings_json(self):
        path = filedialog.asksaveasfilename(title="Save settings", defaultextension=".json", filetypes=[("JSON files", "*.json"), ("All files", "*.*")])
        if not path:
            return
        Path(path).write_text(json.dumps(self._settings_dict(), indent=2), encoding="utf-8")

    def load_settings_json(self):
        path = filedialog.askopenfilename(title="Load settings", filetypes=[("JSON files", "*.json"), ("All files", "*.*")])
        if not path:
            return
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        for key, filename in {
            "guided_script": "tess_guided_analysis.py",
            "joint_script": "joint_search_option.py",
            "converter_script": "spoc_lightcurve_converter.py",
        }.items():
            if data.get(key) == filename:
                data[key] = str(SCRIPT_DIR / filename)
        for key, value in data.items():
            if hasattr(self, key):
                try:
                    getattr(self, key).set(value)
                except Exception:
                    pass
        self._update_tess_mode_state()
        self._update_ui_mode_state()
        self._update_analysis_mode_state()
        self._update_polarimetry_state()
        self._update_smoothing_state()
        self._update_joint_weight_mode_state()
        self._update_phase_zero_mode_state()
        self._update_run_plan_preview()
        self._refresh_preview_list()


def attach_menu(app: GuidedAnalysisGUI):
    menubar = tk.Menu(app)
    filemenu = tk.Menu(menubar, tearoff=0)
    filemenu.add_command(label="Load settings...", command=app.load_settings_json)
    filemenu.add_command(label="Save settings...", command=app.save_settings_json)
    filemenu.add_separator()
    filemenu.add_command(label="Exit", command=app.close_application)
    menubar.add_cascade(label="File", menu=filemenu)

    helpmenu = tk.Menu(menubar, tearoff=0)
    helpmenu.add_command(label="Help tab", command=app.show_help_tab)
    helpmenu.add_command(label="Copy help text", command=app.copy_help_text)
    helpmenu.add_command(label="Save help text", command=app.save_help_text)
    menubar.add_cascade(label="Help", menu=helpmenu)
    app.config(menu=menubar)


def main():
    app = GuidedAnalysisGUI()
    attach_menu(app)
    app.mainloop()


if __name__ == "__main__":
    main()


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
import sys
import threading
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from PIL import Image, ImageTk


APP_TITLE = "TESS Photometry Pipeline GUI"
DEFAULT_GEOMETRY = "1280x900"


HELP_TEXT = """
TESS Photometry Pipeline GUI — option reference
===============================================

This GUI is a command-builder and launcher for:
  • tess_watershed_extractor.py
  • tess_lightcurve_detrend.py

It does not reimplement the algorithms. It builds the command line, runs the script,
and shows the log plus output PNG previews.

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
  inside the image. In no-Gaia modes this should usually be 1.

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

Pure sum
  --pure-sum
  Use the entire watershed-owned region for the selected target and sum all
  those pixels, instead of optimizing the aperture with jump/core growth.

MATLAB pure single saturated mode
  --matlab-pure-single-sat
  Single-target saturated-star branch modeled on the older MATLAB workflow.
  Best for extremely saturated stars where the standard aperture logic is not
  what you want.

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

Advanced: MATLAB saturated branch
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

  MATLAB ap thresh
    --matlab-ap-thresh
    Initial threshold for the pure MATLAB-style saturated aperture seed.

  Disable legacy geometry
    --matlab-no-legacy-geometry
    Turns off the preserved old MATLAB geometric restrictions.

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

Gap days
  --gap-days
  Gap threshold used to split the light curve into chunks for chunk-wise
  median leveling.

Notes
-----
• The command preview is the authoritative description of what the GUI will do.
• If a script changes, the GUI may need to be updated to stay in sync.
• For debugging, always compare the GUI settings with the command preview and
  the run log.
"""


TOOLTIPS = {
    "ex_input_mode": "Choose whether to search a directory of TPF files or run on one specific FITS/TPF file.",
    "ex_tpf_dir": "Directory passed to --tpf-dir. The extractor searches here for TPF files.",
    "ex_single_file": "Single TPF/FITS file passed via --single.",
    "ex_recursive": "Search subdirectories under the chosen TPF directory.",
    "ex_output_root": "Directory passed as --output-root. Extracted CSVs, NPYs, and PNGs are written here.",
    "ex_n_targets": "Number of targets to extract per stamp, usually the brightest Gaia sources inside the image.",
    "ex_method": "Aperture-growth method: jump, core, or both.",
    "ex_gaia_radius": "Cone-search radius used for Gaia target lookup.",
    "ex_no_gaia": "Skip Gaia queries and define the target from image pixels instead. Best for single-target runs.",
    "ex_gaia_fallback": "If Gaia fails, fall back to no-Gaia single-target mode.",
    "ex_no_quality0": "Do not restrict to QUALITY==0 cadences.",
    "ex_pure_sum": "Use the full watershed-owned region for the target and sum all of those pixels instead of optimizing the aperture.",
    "ex_save_aperture_plots": "Save aperture-overlay PNGs. If unchecked, --no-aperture-plots is added.",
    "ex_matlab_pure_single_sat": "Use the pure single-target saturated-star branch modeled on the older MATLAB workflow.",
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
    "ex_matlab_ap_thresh": "Initial threshold for the pure MATLAB-style saturated aperture seed.",
    "ex_disable_legacy_geometry": "Turn off the preserved legacy MATLAB geometric restrictions.",
    "ex_command_preview": "The exact extractor command the GUI will run.",
    "dt_lightcurve_dir": "Directory containing input CSV light curves.",
    "dt_diagnostics_dir": "Directory containing centroid/background NPY files and related diagnostics.",
    "dt_output_dir": "Directory for detrended CSVs and PNGs.",
    "dt_pattern": "Filename pattern used to choose input CSV light curves.",
    "dt_prefix": "Prefix added to detrended output filenames.",
    "dt_recursive": "Search input CSV files recursively under the lightcurve directory.",
    "dt_use_background": "Include the saved background series as a regression term when decorrelating.",
    "dt_use_pchip": "Apply a smooth PCHIP high-pass step after leveling/decorrelation.",
    "dt_skip_xybg": "Skip centroid/background regression, but still do chunk leveling and optional PCHIP/sector combining.",
    "dt_combine_sectors": "Write combined multi-sector CSV and PNG products.",
    "dt_knot_spacing": "Spacing for optional time-basis hinge functions used in decorrelation. Use inf to disable them.",
    "dt_robust_iters": "Number of robust weighted least-squares iterations.",
    "dt_huber_k": "Huber tuning constant controlling how strongly outliers are downweighted.",
    "dt_pchip_knot_spacing": "Time spacing used when building the smooth PCHIP trend.",
    "dt_gap_days": "Gap threshold used to split the light curve into chunks for chunk-wise median leveling.",
    "dt_command_preview": "The exact detrender command the GUI will run.",
    "preview_dir": "Directory scanned for PNG previews.",
    "preview_scale_mode": "Choose how preview images are displayed: Fit scales to the visible preview pane; percentage modes use a fixed zoom level.",
    "preview_zoom": "Manual preview zoom level. Also adjustable with Ctrl + mouse wheel.",
}




def quote_cmd(cmd: list[str]) -> str:
    return " ".join(shlex.quote(str(x)) for x in cmd)


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

        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)

    def _on_mousewheel(self, event):
        try:
            self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        except Exception:
            pass


class TESSGui(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(APP_TITLE)
        self.geometry(DEFAULT_GEOMETRY)

        self.log_queue: queue.Queue[tuple[str, str]] = queue.Queue()
        self.current_process: subprocess.Popen | None = None
        self.current_job_name: str | None = None
        self.current_output_dir: Path | None = None
        self.preview_image = None

        self._build_vars()
        self._build_ui()
        self._poll_log_queue()
        self._update_extractor_state()
        self._update_detrender_state()
        self._refresh_preview_list()

    def _build_vars(self):
        self.extractor_script = tk.StringVar(value="tess_watershed_extractor.py")
        self.detrender_script = tk.StringVar(value="tess_lightcurve_detrend.py")

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
        self.ex_pure_sum = tk.BooleanVar(value=False)
        self.ex_save_aperture_plots = tk.BooleanVar(value=True)
        self.ex_matlab_pure_single_sat = tk.BooleanVar(value=False)

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
        self.ex_matlab_ap_thresh = tk.DoubleVar(value=3000.0)
        self.ex_disable_legacy_geometry = tk.BooleanVar(value=False)

        self.ex_command_preview = tk.StringVar(value="")

        self.dt_lightcurve_dir = tk.StringVar(value="LC_products_multi")
        self.dt_diagnostics_dir = tk.StringVar(value="LC_products_multi")
        self.dt_output_dir = tk.StringVar(value="LC_products_multi")
        self.dt_pattern = tk.StringVar(value="*.csv")
        self.dt_recursive = tk.BooleanVar(value=False)
        self.dt_prefix = tk.StringVar(value="detrended_")

        self.dt_use_background = tk.BooleanVar(value=False)
        self.dt_use_pchip = tk.BooleanVar(value=False)
        self.dt_skip_xybg = tk.BooleanVar(value=False)
        self.dt_combine_sectors = tk.BooleanVar(value=True)

        self.dt_knot_spacing = tk.StringVar(value="inf")
        self.dt_robust_iters = tk.IntVar(value=8)
        self.dt_huber_k = tk.DoubleVar(value=1.5)
        self.dt_pchip_knot_spacing = tk.DoubleVar(value=0.5)
        self.dt_gap_days = tk.DoubleVar(value=0.5)

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

        self._build_script_paths_frame(root).grid(row=0, column=0, sticky="ew", padx=8, pady=6)

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

        main_frame = ttk.LabelFrame(root, text="Main settings")
        main_frame.grid(row=3, column=0, sticky="ew", padx=8, pady=6)
        for c in range(4):
            main_frame.columnconfigure(c, weight=1)

        self._spin(main_frame, "N targets", self.ex_n_targets, 1, 20, 0, 0, tooltip_key="ex_n_targets")
        self._combo(main_frame, "Method", self.ex_method, ["jump", "core", "both"], 0, 2, tooltip_key="ex_method")
        self._entry(main_frame, "Gaia radius [arcmin]", self.ex_gaia_radius, 1, 0, tooltip_key="ex_gaia_radius")

        self._make_checkbutton(main_frame, text="No Gaia", variable=self.ex_no_gaia, command=self._update_extractor_state,
                        tooltip_key="ex_no_gaia", row=1, column=2, sticky="w", padx=6, pady=4)
        self._make_checkbutton(main_frame, text="Gaia fallback", variable=self.ex_gaia_fallback, command=self._update_extractor_command_preview,
                        tooltip_key="ex_gaia_fallback", row=1, column=3, sticky="w", padx=6, pady=4)
        self._make_checkbutton(main_frame, text="Ignore quality flags", variable=self.ex_no_quality0, command=self._update_extractor_command_preview,
                        tooltip_key="ex_no_quality0", row=2, column=0, sticky="w", padx=6, pady=4)
        self._make_checkbutton(main_frame, text="Pure sum", variable=self.ex_pure_sum, command=self._update_extractor_state,
                        tooltip_key="ex_pure_sum", row=2, column=1, sticky="w", padx=6, pady=4)
        self._make_checkbutton(main_frame, text="MATLAB pure single saturated mode", variable=self.ex_matlab_pure_single_sat, command=self._update_extractor_state,
                        tooltip_key="ex_matlab_pure_single_sat", row=2, column=2, columnspan=2, sticky="w", padx=6, pady=4)

        self.ex_warning_label = ttk.Label(main_frame, text="", foreground="firebrick")
        self.ex_warning_label.grid(row=3, column=0, columnspan=4, sticky="w", padx=6, pady=(4, 0))

        adv = ttk.LabelFrame(root, text="Advanced")
        adv.grid(row=4, column=0, sticky="ew", padx=8, pady=6)
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

        matlab = ttk.LabelFrame(adv, text="MATLAB saturated branch")
        matlab.grid(row=2, column=0, columnspan=4, sticky="ew", padx=6, pady=6)
        for c in range(4):
            matlab.columnconfigure(c, weight=1)
        self._entry(matlab, "Sat thresh", self.ex_sat_thresh, 0, 0, tooltip_key="ex_sat_thresh")
        self._spin(matlab, "Sat min npix", self.ex_sat_min_npix, 1, 10000, 0, 2, tooltip_key="ex_sat_min_npix")
        self._spin(matlab, "Back nfaint", self.ex_back_nfaint, 1, 1000, 1, 0, tooltip_key="ex_back_nfaint")
        self._entry(matlab, "Phase bin", self.ex_phase_bin, 1, 2, tooltip_key="ex_phase_bin")
        self._entry(matlab, "MATLAB ap thresh", self.ex_matlab_ap_thresh, 2, 0, tooltip_key="ex_matlab_ap_thresh")
        self._make_checkbutton(matlab, text="Disable legacy geometry", variable=self.ex_disable_legacy_geometry,
                        command=self._update_extractor_command_preview, tooltip_key="ex_disable_legacy_geometry",
                        row=2, column=2, columnspan=2, sticky="w", padx=6, pady=4)

        action = ttk.LabelFrame(root, text="Actions")
        action.grid(row=5, column=0, sticky="ew", padx=8, pady=6)
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

        self._build_script_paths_frame(root).grid(row=0, column=0, sticky="ew", padx=8, pady=6)

        dirs = ttk.LabelFrame(root, text="Directories")
        dirs.grid(row=1, column=0, sticky="ew", padx=8, pady=6)
        dirs.columnconfigure(1, weight=1)
        self._entry(dirs, "Lightcurve dir", self.dt_lightcurve_dir, 0, 0, browse="dir", tooltip_key="dt_lightcurve_dir")
        self._entry(dirs, "Diagnostics dir", self.dt_diagnostics_dir, 1, 0, browse="dir", tooltip_key="dt_diagnostics_dir")
        self._entry(dirs, "Output dir", self.dt_output_dir, 2, 0, browse="dir", tooltip_key="dt_output_dir")

        sel = ttk.LabelFrame(root, text="Input selection")
        sel.grid(row=2, column=0, sticky="ew", padx=8, pady=6)
        for c in range(4):
            sel.columnconfigure(c, weight=1)
        self._entry(sel, "Pattern", self.dt_pattern, 0, 0, tooltip_key="dt_pattern")
        self._entry(sel, "Prefix", self.dt_prefix, 0, 2, tooltip_key="dt_prefix")
        self._make_checkbutton(sel, text="Recursive", variable=self.dt_recursive, command=self._update_detrender_command_preview, tooltip_key="dt_recursive", row=1, column=0, sticky="w", padx=6, pady=4)

        opts = ttk.LabelFrame(root, text="Detrending options")
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
        self._spin(opts, "Robust iters", self.dt_robust_iters, 1, 100, 1, 0, tooltip_key="dt_robust_iters")
        self._entry(opts, "Huber k", self.dt_huber_k, 1, 2, tooltip_key="dt_huber_k")
        self.dt_pchip_entry = self._entry(opts, "PCHIP knot spacing", self.dt_pchip_knot_spacing, 2, 0, tooltip_key="dt_pchip_knot_spacing")
        self._entry(opts, "Gap days", self.dt_gap_days, 2, 2, tooltip_key="dt_gap_days")

        action = ttk.LabelFrame(root, text="Actions")
        action.grid(row=4, column=0, sticky="ew", padx=8, pady=6)
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
        self.preview_canvas.bind_all("<MouseWheel>", self._on_preview_mousewheel)
        self.preview_canvas.bind_all("<Shift-MouseWheel>", self._on_preview_shift_mousewheel)
        self.preview_canvas.bind_all("<Control-MouseWheel>", self._on_preview_zoom_mousewheel)


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
            self.ex_pure_sum, self.ex_save_aperture_plots, self.ex_matlab_pure_single_sat,
            self.ex_min_pixels, self.ex_amp_q_lo, self.ex_amp_q_hi, self.ex_amp_min_frac,
            self.ex_max_radius_pix, self.ex_max_components, self.ex_min_seed_frac,
            self.ex_min_new_pixels, self.ex_core_npix, self.ex_core_min_frac,
            self.ex_sat_thresh, self.ex_sat_min_npix, self.ex_back_nfaint, self.ex_phase_bin,
            self.ex_matlab_ap_thresh, self.ex_disable_legacy_geometry
        ]
        for v in vars_to_trace:
            v.trace_add("write", lambda *_: self._update_extractor_command_preview())

    def _trace_detrender_vars(self):
        vars_to_trace = [
            self.detrender_script, self.dt_lightcurve_dir, self.dt_diagnostics_dir,
            self.dt_output_dir, self.dt_pattern, self.dt_recursive, self.dt_prefix,
            self.dt_use_background, self.dt_use_pchip, self.dt_skip_xybg,
            self.dt_combine_sectors, self.dt_knot_spacing, self.dt_robust_iters,
            self.dt_huber_k, self.dt_pchip_knot_spacing, self.dt_gap_days
        ]
        for v in vars_to_trace:
            v.trace_add("write", lambda *_: self._update_detrender_command_preview())

    def _update_extractor_state(self):
        is_dir_mode = (self.ex_input_mode.get() == "directory")
        self.ex_dir_entry.configure(state=("normal" if is_dir_mode else "disabled"))
        self.ex_recursive_check.configure(state=("normal" if is_dir_mode else "disabled"))
        self.ex_single_entry.configure(state=("normal" if not is_dir_mode else "disabled"))

        warning = []
        if self.ex_no_gaia.get() and self.ex_n_targets.get() != 1:
            warning.append("No-Gaia mode requires n-targets = 1.")
        if self.ex_matlab_pure_single_sat.get() and self.ex_n_targets.get() != 1:
            warning.append("MATLAB pure single saturated mode requires n-targets = 1.")
        self.ex_warning_label.configure(text="  ".join(warning))
        self._update_extractor_command_preview()

    def _update_detrender_state(self):
        skip = self.dt_skip_xybg.get()
        pchip = self.dt_use_pchip.get()
        self.dt_chk_background.configure(state=("disabled" if skip else "normal"))
        try:
            self.dt_pchip_entry.configure(state=("normal" if pchip else "disabled"))
        except Exception:
            pass
        self._update_detrender_command_preview()

    def build_extractor_command(self) -> list[str]:
        script = self.extractor_script.get().strip()
        if not script:
            raise ValueError("Extractor script path is empty.")
        cmd = [sys.executable, "-u", script]

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
        if not self.ex_pure_sum.get():
            cmd += ["--method", self.ex_method.get()]
        cmd += ["--gaia-radius-arcmin", str(float(self.ex_gaia_radius.get()))]

        if self.ex_no_gaia.get():
            cmd.append("--no-gaia")
        if self.ex_gaia_fallback.get():
            cmd.append("--gaia-fallback")
        if self.ex_no_quality0.get():
            cmd.append("--no-quality0")
        if self.ex_pure_sum.get():
            cmd.append("--pure-sum")
        if not self.ex_save_aperture_plots.get():
            cmd.append("--no-aperture-plots")
        if self.ex_matlab_pure_single_sat.get():
            cmd.append("--matlab-pure-single-sat")

        cmd += ["--min-pixels", str(int(self.ex_min_pixels.get()))]
        cmd += ["--amp-q-lo", str(float(self.ex_amp_q_lo.get()))]
        cmd += ["--amp-q-hi", str(float(self.ex_amp_q_hi.get()))]
        cmd += ["--amp-min-frac", str(float(self.ex_amp_min_frac.get()))]
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
        cmd += ["--matlab-ap-thresh", str(float(self.ex_matlab_ap_thresh.get()))]
        if self.ex_disable_legacy_geometry.get():
            cmd.append("--matlab-no-legacy-geometry")
        return cmd

    def build_detrender_command(self) -> list[str]:
        script = self.detrender_script.get().strip()
        if not script:
            raise ValueError("Detrender script path is empty.")
        cmd = [sys.executable, "-u", script]
        cmd += ["--lightcurve-dir", self.dt_lightcurve_dir.get().strip() or "LC_products_multi"]
        cmd += ["--diagnostics-dir", self.dt_diagnostics_dir.get().strip() or "LC_products_multi"]
        cmd += ["--output-dir", self.dt_output_dir.get().strip() or "LC_products_multi"]
        cmd += ["--pattern", self.dt_pattern.get().strip() or "*.csv"]
        cmd += ["--prefix", self.dt_prefix.get().strip() or "detrended_"]
        if self.dt_recursive.get():
            cmd.append("--recursive")
        if self.dt_use_background.get() and not self.dt_skip_xybg.get():
            cmd.append("--use-background")
        if self.dt_use_pchip.get():
            cmd.append("--use-pchip-highpass")
        if self.dt_skip_xybg.get():
            cmd.append("--skip-xybg-decorrelation")
        if not self.dt_combine_sectors.get():
            cmd.append("--no-combine-sectors")
        cmd += ["--knot-spacing-days", str(self.dt_knot_spacing.get()).strip() or "inf"]
        cmd += ["--robust-iters", str(int(self.dt_robust_iters.get()))]
        cmd += ["--huber-k", str(float(self.dt_huber_k.get()))]
        cmd += ["--pchip-knot-spacing", str(float(self.dt_pchip_knot_spacing.get()))]
        cmd += ["--gap-days", str(float(self.dt_gap_days.get()))]
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

    def _run_subprocess(self, job_name: str, cmd: list[str], output_dir: Path):
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
            self.current_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=env,
            )
        except Exception as exc:
            self.current_process = None
            self.status_text.set("Ready.")
            messagebox.showerror("Failed to start process", str(exc))
            return

        def reader_thread():
            proc = self.current_process
            try:
                if proc and proc.stdout is not None:
                    for line in proc.stdout:
                        self.log_queue.put(("line", line))
                rc = proc.wait() if proc else -1
                self.log_queue.put(("done", f"{job_name} finished with exit code {rc}.\n"))
            except Exception as exc:
                self.log_queue.put(("done", f"{job_name} failed: {exc}\n"))

        threading.Thread(target=reader_thread, daemon=True).start()

    def stop_current_process(self):
        if self.current_process is None:
            return
        try:
            self.current_process.terminate()
            self.status_text.set("Stopping process...")
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
                    self.log_text.insert("end", "\n" + payload + "\n")
                    self.log_text.see("end")
                    self.current_process = None
                    self.status_text.set("Ready.")
                    self._refresh_preview_list()
        except queue.Empty:
            pass
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
            self.preview_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
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

    def _settings_dict(self) -> dict:
        return {
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
            "ex_pure_sum": self.ex_pure_sum.get(),
            "ex_save_aperture_plots": self.ex_save_aperture_plots.get(),
            "ex_matlab_pure_single_sat": self.ex_matlab_pure_single_sat.get(),
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
            "ex_matlab_ap_thresh": self.ex_matlab_ap_thresh.get(),
            "ex_disable_legacy_geometry": self.ex_disable_legacy_geometry.get(),
            "dt_lightcurve_dir": self.dt_lightcurve_dir.get(),
            "dt_diagnostics_dir": self.dt_diagnostics_dir.get(),
            "dt_output_dir": self.dt_output_dir.get(),
            "dt_pattern": self.dt_pattern.get(),
            "dt_recursive": self.dt_recursive.get(),
            "dt_prefix": self.dt_prefix.get(),
            "dt_use_background": self.dt_use_background.get(),
            "dt_use_pchip": self.dt_use_pchip.get(),
            "dt_skip_xybg": self.dt_skip_xybg.get(),
            "dt_combine_sectors": self.dt_combine_sectors.get(),
            "dt_knot_spacing": self.dt_knot_spacing.get(),
            "dt_robust_iters": self.dt_robust_iters.get(),
            "dt_huber_k": self.dt_huber_k.get(),
            "dt_pchip_knot_spacing": self.dt_pchip_knot_spacing.get(),
            "dt_gap_days": self.dt_gap_days.get(),
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
        Path(path).write_text(json.dumps(self._settings_dict(), indent=2), encoding="utf-8")

    def load_settings_json(self):
        path = filedialog.askopenfilename(
            title="Load settings",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if not path:
            return
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        for key, value in data.items():
            if hasattr(self, key):
                var = getattr(self, key)
                try:
                    var.set(value)
                except Exception:
                    pass
        self._update_extractor_state()
        self._update_detrender_state()
        self._refresh_preview_list()

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
    filemenu.add_command(label="Exit", command=app.destroy)
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

#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import queue
import re
import shlex
import subprocess
import sys
import tempfile
import threading
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from PIL import Image, ImageTk

APP_TITLE = "TESS Guided Analysis GUI"
DEFAULT_GEOMETRY = "1320x940"

HELP_TEXT = """
TESS Guided Analysis GUI
========================

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
  Runs the earlier joint photometry + polarimetry search workflow using the
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

SPOC lc.fits conversion
  Run spoc_lightcurve_converter.py first, then feed the converted CSV into the
  guided analysis as a normal TESS CSV.

SPOC converter command template
--------------------------------
The converter command is configurable because different converter versions may
not use exactly the same CLI.

Placeholders available in the template:
  {python}   Python executable
  {script}   converter script path
  {input}    selected SPOC input path / glob
  {output}   converted CSV output path

Default template:
  {python} -u {script} --input {input} --output {output}

Common guided-analysis options
------------------------------
POL product
  nm                  : night-mean-subtracted polarimetry
  resid_nm_pchip      : night-mean-subtracted and PCHIP-cleaned
  pw_resid_nm_pchip   : prewhitened residual product (only if explicitly wanted)

Preprocess diagnostic plots
  When enabled, writes optional before/after polarimetry preprocessing plots.

Inline plots
  Normally leave this off in GUI mode. Turning it on may create separate plot
  windows depending on the backend and script behavior.

Run / Preview
-------------
The preview tab scans the chosen output root recursively for PNG files.
"""

TOOLTIPS = {
    "guided_script": "Path to the guided-analysis Python script that will be imported and run.",
    "converter_script": "Path to spoc_lightcurve_converter.py used when SPOC lc.fits conversion mode is selected.",
    "analysis_mode": "Choose whether to run the guided-analysis workflow or the earlier joint-search workflow.",
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
    "tess_input_mode": "Choose whether TESS input comes from an existing CSV, a pipeline/custom CSV directory, or SPOC lc.fits conversion.",
    "tess_csv": "Existing TESS CSV file to load directly in spoc_csv mode.",
    "pipeline_dir": "Directory of pipeline/custom CSV light curves for pipeline_dir mode.",
    "pipeline_pattern": "Filename pattern used when scanning the pipeline/custom light-curve directory.",
    "pipeline_flux": "Preferred flux-family selection in pipeline_dir mode: raw, detrended, or auto.",
    "pipeline_recursive": "Search the pipeline/custom light-curve directory recursively.",
    "spoc_input": "SPOC lc.fits input path or glob used by the converter.",
    "spoc_output_csv": "CSV path that the converter should write before the guided analysis starts.",
    "spoc_template": "Command template used to run the SPOC converter. Use {python}, {script}, {input}, and {output} placeholders.",
    "pol_csv": "Polarimetry CSV: either raw/basic polarimetry or a precomputed analysis-frame CSV.",
    "use_polarimetry": "Guided mode only. When unchecked, run only the TESS frequency search and global multisinusoid fit, skipping all polarimetry steps.",
    "pol_product": "Polarimetry product to analyze: nm, resid_nm_pchip, or pw_resid_nm_pchip.",
    "save_generated_frame": "Save the generated analysis-frame CSV when the input polarimetry is raw/basic.",
    "generated_analysis_dir": "Optional directory for the generated analysis-frame CSV. Blank means next to the polarimetry file.",
    "outroot": "Top-level output directory. The script writes subdirectories like tess/, q/, u/, p/, and optional preprocessing diagnostics here.",
    "show_inline": "Inline plots / interactively. Usually best left off in GUI mode.",
    "verbose": "General verbosity level printed by the guided-analysis script.",
    "lsq_verbose": "least_squares verbosity level for the global fits.",
    "fmin": "Minimum frequency in cycles/day.",
    "fmax": "Maximum frequency in cycles/day.",
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
    "group_mode": "How nights/groups are defined for the polarimetric baseline model.",
    "gap_hours": "Gap threshold in hours used when group_mode = gap.",
    "do_detrend": "Apply the optional broad polynomial detrending hook before the main analysis.",
    "detrend_order": "Polynomial order used by the optional broad detrending hook.",
    "n_phase_plots": "Number of strongest modes to show in the phased-summary plots.",
    "phase_sort_by": "How to rank modes when selecting phased-summary plots.",
    "phase_plot_style": "Whether phased plots use isolated_mode or prefit_residual style.",
    "phase_zero_mode": "Phase-reference convention for reported phases and phased plots.",
    "phase_zero_btjd": "Custom BTJD phase zero used when phase_zero_mode = custom_btjd.",
    "plot_preprocess": "Write the optional before/after polarimetry preprocessing diagnostic plots.",
    "preplot_chunk_days": "Length of each time chunk in the preprocessing diagnostic plots.",
    "preplot_panels": "Maximum number of chunk panels per preprocessing-diagnostic figure page.",
    "preplot_include_pw": "Include the prewhitened preprocessing product in the diagnostic plots.",
    "compute_pw": "Explicitly compute the pw_resid_nm_pchip product. Off by default.",
    "summary_save_period": "Also save summary amplitude-spectrum plots with the x-axis shown as period in days.",
    "summary_save_log_amplitude": "Also save summary amplitude-spectrum plots with log-scaled amplitude.",
    "preview_dir": "Directory scanned recursively for PNG previews.",
    "preview_scale_mode": "Choose how preview images are displayed: Fit scales the image to the visible preview pane; percentage modes use a fixed zoom level.",
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
        self.inner.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
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


class GuidedAnalysisGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(APP_TITLE)
        self.geometry(DEFAULT_GEOMETRY)

        self.log_queue: queue.Queue[tuple[str, str]] = queue.Queue()
        self.current_process: subprocess.Popen | None = None
        self.current_output_dir: Path | None = None
        self.preview_image = None
        self._preview_paths: list[Path] = []
        self._temp_runner_path: Path | None = None

        self._build_vars()
        self._build_ui()
        self._poll_log_queue()
        self._update_tess_mode_state()
        self._update_analysis_mode_state()
        self._update_polarimetry_state()
        self._update_joint_weight_mode_state()
        self._update_phase_zero_mode_state()
        self._update_run_plan_preview()
        self._refresh_preview_list()

    def _build_vars(self):
        # script paths
        self.guided_script = tk.StringVar(value="tess_guided_analysis.py")
        self.joint_script = tk.StringVar(value="joint_search_option.py")
        self.converter_script = tk.StringVar(value="spoc_lightcurve_converter.py")

        # top-level mode
        self.analysis_mode = tk.StringVar(value="guided_analysis")

        # TESS input
        self.tess_input_mode = tk.StringVar(value="existing_csv")
        self.tess_csv = tk.StringVar(value="combined_filtered.csv")
        self.pipeline_dir = tk.StringVar(value="tess_pipeline_lcs")
        self.pipeline_pattern = tk.StringVar(value="*.csv")
        self.pipeline_recursive = tk.BooleanVar(value=False)
        self.pipeline_flux = tk.StringVar(value="auto")
        self.tess_force_y_col = tk.StringVar(value="")

        self.spoc_input = tk.StringVar(value="")
        self.spoc_output_csv = tk.StringVar(value="combined_filtered.csv")
        self.spoc_template = tk.StringVar(value="{python} -u {script} --input {input} --output {output}")

        # polarimetry / output
        self.use_polarimetry = tk.BooleanVar(value=True)
        self.pol_csv = tk.StringVar(value="target_IND.csv")
        self.pol_product = tk.StringVar(value="resid_nm_pchip")
        self.save_generated_frame = tk.BooleanVar(value=True)
        self.generated_analysis_dir = tk.StringVar(value="")
        self.outroot = tk.StringVar(value="tess_guided_outputs")
        self.show_plots_inline = tk.BooleanVar(value=False)
        self.verbose = tk.IntVar(value=1)
        self.lsq_verbose = tk.IntVar(value=0)

        # main analysis controls
        self.fmin = tk.DoubleVar(value=0.01)
        self.fmax = tk.DoubleVar(value=50.0)
        self.tess_grid_mode = tk.StringVar(value="full_baseline")
        self.tess_snr_stop = tk.DoubleVar(value=4.0)
        self.max_tess_modes = tk.IntVar(value=20)
        self.pol_snr_stop = tk.DoubleVar(value=2.0)
        self.max_pol_modes = tk.IntVar(value=99)
        self.guided_pol_fmin = tk.DoubleVar(value=0.2)
        self.search_window_mult = tk.DoubleVar(value=10.0)
        self.noise_ks = tk.DoubleVar(value=3.0)
        self.noise_bins = tk.IntVar(value=6)

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

        # preprocess diagnostics
        self.plot_preprocess = tk.BooleanVar(value=False)
        self.preplot_chunk_days = tk.DoubleVar(value=3.0)
        self.preplot_panels = tk.IntVar(value=6)
        self.preplot_include_pw = tk.BooleanVar(value=False)
        self.compute_pw = tk.BooleanVar(value=False)
        self.summary_save_period = tk.BooleanVar(value=False)
        self.summary_save_log_amplitude = tk.BooleanVar(value=False)

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
        root = sf.inner
        root.columnconfigure(0, weight=1)

        scripts = ttk.LabelFrame(root, text="Script paths / mode")
        scripts.grid(row=0, column=0, sticky="ew", padx=8, pady=6)
        for c in range(5):
            scripts.columnconfigure(c, weight=1)
        self._entry(scripts, "Guided-analysis script", self.guided_script, 0, 0, browse="file", tooltip_key="guided_script", filetypes=[("Python files", "*.py"), ("All files", "*.*")])
        self._entry(scripts, "Joint-search script", self.joint_script, 1, 0, browse="file", tooltip_key="joint_script", filetypes=[("Python files", "*.py"), ("All files", "*.*")])
        self._entry(scripts, "SPOC converter script", self.converter_script, 2, 0, browse="file", tooltip_key="converter_script", filetypes=[("Python files", "*.py"), ("All files", "*.*")])
        self._combo(scripts, "Analysis mode", self.analysis_mode, ["guided_analysis", "joint_search"], 3, 0, tooltip_key="analysis_mode")

        pol = ttk.LabelFrame(root, text="Polarimetry / output")
        pol.grid(row=1, column=0, sticky="ew", padx=8, pady=6)
        for c in range(4):
            pol.columnconfigure(c, weight=1)
        self.use_polarimetry_chk = self._check(pol, "Use polarimetry", self.use_polarimetry, 0, 0, tooltip_key="use_polarimetry", command=self._update_polarimetry_state, colspan=2)

        self.pol_csv_entry = self._entry(pol, "Polarimetry CSV", self.pol_csv, 1, 0, browse="file", tooltip_key="pol_csv", filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        self.pol_product_combo = self._combo(pol, "POL product", self.pol_product, ["nm", "resid_nm_pchip", "pw_resid_nm_pchip"], 2, 0, tooltip_key="pol_product")
        self.save_generated_frame_chk = self._check(pol, "Save generated analysis frame", self.save_generated_frame, 2, 2, tooltip_key="save_generated_frame", command=self._update_run_plan_preview)
        self.generated_analysis_dir_entry = self._entry(pol, "Generated analysis dir", self.generated_analysis_dir, 3, 0, browse="dir", tooltip_key="generated_analysis_dir")
        self._entry(pol, "Output root", self.outroot, 4, 0, browse="dir", tooltip_key="outroot")
        self._check(pol, "Inline plots", self.show_plots_inline, 4, 2, tooltip_key="show_inline", command=self._update_run_plan_preview)
        self._spin(pol, "Verbose", self.verbose, 0, 5, 5, 0, tooltip_key="verbose")
        self._spin(pol, "LSQ verbose", self.lsq_verbose, 0, 5, 5, 2, tooltip_key="lsq_verbose")

        main = ttk.LabelFrame(root, text="Main analysis settings")
        main.grid(row=2, column=0, sticky="ew", padx=8, pady=6)
        for c in range(4):
            main.columnconfigure(c, weight=1)
        self._entry(main, "Fmin [c/d]", self.fmin, 0, 0, tooltip_key="fmin")
        self._entry(main, "Fmax [c/d]", self.fmax, 0, 2, tooltip_key="fmax")
        self._combo(main, "TESS grid mode", self.tess_grid_mode, ["full_baseline", "longest_chunk"], 1, 0, tooltip_key="tess_grid_mode")
        self._entry(main, "TESS SNR stop", self.tess_snr_stop, 1, 2, tooltip_key="tess_snr_stop")
        self._spin(main, "Max TESS modes", self.max_tess_modes, 1, 999, 2, 0, tooltip_key="max_tess_modes")
        self._entry(main, "POL SNR stop", self.pol_snr_stop, 2, 2, tooltip_key="pol_snr_stop")
        self._spin(main, "Max POL modes", self.max_pol_modes, 1, 999, 3, 0, tooltip_key="max_pol_modes")
        self._entry(main, "Guided POL fmin", self.guided_pol_fmin, 3, 2, tooltip_key="guided_pol_fmin")
        self._entry(main, "Search window mult", self.search_window_mult, 4, 0, tooltip_key="search_window_mult")
        self._entry(main, "Noise KS", self.noise_ks, 4, 2, tooltip_key="noise_ks")
        self._spin(main, "Noise side bins", self.noise_bins, 1, 1000, 5, 0, tooltip_key="noise_bins")

        chans = ttk.LabelFrame(root, text="Channels / baseline model")
        chans.grid(row=3, column=0, sticky="ew", padx=8, pady=6)
        for c in range(4):
            chans.columnconfigure(c, weight=1)
        self._check(chans, "q", self.channel_q, 0, 0, tooltip_key="channels", command=self._update_run_plan_preview)
        self._check(chans, "u", self.channel_u, 0, 1, tooltip_key="channels", command=self._update_run_plan_preview)
        self._check(chans, "p", self.channel_p, 0, 2, tooltip_key="channels", command=self._update_run_plan_preview)
        self._check(chans, "Use night offsets", self.use_offsets, 1, 0, tooltip_key="use_offsets", command=self._update_run_plan_preview)
        self._check(chans, "Use night slopes", self.use_slopes, 1, 1, tooltip_key="use_slopes", command=self._update_run_plan_preview)
        self._combo(chans, "Group mode", self.group_mode, ["gap", "integer_jd"], 2, 0, tooltip_key="group_mode")
        self._entry(chans, "Gap hours", self.gap_hours, 2, 2, tooltip_key="gap_hours")

        extras = ttk.LabelFrame(root, text="Optional detrending / plots / preprocessing diagnostics")
        extras.grid(row=4, column=0, sticky="ew", padx=8, pady=6)
        for c in range(4):
            extras.columnconfigure(c, weight=1)
        self._check(extras, "Apply broad polynomial detrend", self.do_detrend, 0, 0, tooltip_key="do_detrend", command=self._update_run_plan_preview)
        self._spin(extras, "Detrend poly order", self.detrend_order, 0, 10, 0, 2, tooltip_key="detrend_order")
        self._spin(extras, "N phase plots", self.n_phase_plots, 0, 20, 1, 0, tooltip_key="n_phase_plots")
        self._combo(extras, "Phase sort by", self.phase_sort_by, ["amp", "snr", "mode"], 1, 2, tooltip_key="phase_sort_by")
        self._combo(extras, "Phase plot style", self.phase_plot_style, ["isolated_mode", "prefit_residual"], 2, 0, tooltip_key="phase_plot_style")
        self.phase_zero_mode_combo = self._combo(extras, "Phase zero mode", self.phase_zero_mode, ["local_start", "btjd_zero", "custom_btjd"], 2, 2, tooltip_key="phase_zero_mode")
        ttk.Label(extras, text="BTJD=0.0 uses the absolute TESS zero point.").grid(row=3, column=0, columnspan=2, sticky="w", padx=6, pady=2)
        self.phase_zero_btjd_entry = self._entry(extras, "Custom phase BTJD", self.phase_zero_btjd, 3, 2, tooltip_key="phase_zero_btjd")
        self._check(extras, "Plot preprocess diagnostics", self.plot_preprocess, 4, 0, tooltip_key="plot_preprocess", command=self._update_run_plan_preview)
        self._entry(extras, "Preplot chunk days", self.preplot_chunk_days, 4, 2, tooltip_key="preplot_chunk_days")
        self._spin(extras, "Panels / fig", self.preplot_panels, 5, 20, 5, 0, tooltip_key="preplot_panels")
        self._check(extras, "Include prewhitened product in plots", self.preplot_include_pw, 5, 2, tooltip_key="preplot_include_pw", command=self._update_run_plan_preview)
        self._check(extras, "Compute pw_resid_nm_pchip product", self.compute_pw, 6, 0, tooltip_key="compute_pw", command=self._update_run_plan_preview)
        self.summary_save_period_chk = self._check(
            extras, "Save period summary", self.summary_save_period, 6, 2,
            command=self._update_run_plan_preview, tooltip_key="summary_save_period"
        )
        self.summary_save_log_amplitude_chk = self._check(
            extras, "Save log-amplitude summary", self.summary_save_log_amplitude, 7, 0,
            command=self._update_run_plan_preview, tooltip_key="summary_save_log_amplitude"
        )

        jointf = ttk.LabelFrame(root, text="Joint-search settings")
        jointf.grid(row=5, column=0, sticky="ew", padx=8, pady=6)
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
        actions.grid(row=6, column=0, sticky="ew", padx=8, pady=6)
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
            "guided": [main, extras],
            "joint": [jointf],
        }
        self.polarimetry_toggle_widgets = [
            getattr(self, "pol_csv_entry", None),
            getattr(self, "pol_product_combo", None),
            getattr(self, "save_generated_frame_chk", None),
            getattr(self, "generated_analysis_dir_entry", None),
        ]

        self._trace_vars([
            self.guided_script, self.joint_script, self.converter_script, self.analysis_mode, self.use_polarimetry,
            self.pol_csv, self.pol_product, self.save_generated_frame, self.generated_analysis_dir,
            self.outroot, self.show_plots_inline, self.verbose, self.lsq_verbose,
            self.fmin, self.fmax, self.tess_grid_mode, self.tess_snr_stop, self.max_tess_modes,
            self.pol_snr_stop, self.max_pol_modes, self.guided_pol_fmin, self.search_window_mult,
            self.noise_ks, self.noise_bins, self.channel_q, self.channel_u, self.channel_p,
            self.use_offsets, self.use_slopes, self.group_mode, self.gap_hours,
            self.do_detrend, self.detrend_order, self.n_phase_plots, self.phase_sort_by,
            self.phase_plot_style, self.phase_zero_mode, self.phase_zero_btjd, self.plot_preprocess, self.preplot_chunk_days,
            self.preplot_panels, self.preplot_include_pw, self.compute_pw, self.summary_save_period, self.summary_save_log_amplitude,
            self.joint_k_candidates, self.joint_top_n_raw_tess, self.joint_coarse_oversample,
            self.joint_refine_factor, self.joint_max_iters, self.joint_kfit,
            self.joint_snr_stop, self.joint_w_prefilter, self.joint_ks_tess,
            self.joint_ks_pol, self.joint_trim_top_frac, self.joint_weight_mode,
            self.joint_scale_free_basis, self.joint_manual_w_tess, self.joint_manual_w_pol,
        ], self._update_run_plan_preview)
        self._trace_vars([self.analysis_mode], self._update_analysis_mode_state)
        self._trace_vars([self.use_polarimetry], self._update_polarimetry_state)
        self._trace_vars([self.joint_weight_mode], self._update_joint_weight_mode_state)
        self._trace_vars([self.phase_zero_mode], self._update_phase_zero_mode_state)

    def _build_tess_tab(self):
        sf = ScrollableFrame(self.tab_tess)
        sf.pack(fill="both", expand=True)
        root = sf.inner
        root.columnconfigure(0, weight=1)

        modef = ttk.LabelFrame(root, text="TESS input mode")
        modef.grid(row=0, column=0, sticky="ew", padx=8, pady=6)
        for c in range(4):
            modef.columnconfigure(c, weight=1)
        self._combo(modef, "TESS input mode", self.tess_input_mode,
                    ["existing_csv", "pipeline_dir", "spoc_lc_fits"], 0, 0, tooltip_key="tess_input_mode")

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
        self._entry(pipef, "Force TESS y column (optional)", self.tess_force_y_col, 2, 2)

        spocf = ttk.LabelFrame(root, text="SPOC lc.fits conversion mode")
        spocf.grid(row=3, column=0, sticky="ew", padx=8, pady=6)
        for c in range(4):
            spocf.columnconfigure(c, weight=1)
        self._entry(spocf, "SPOC input path or glob", self.spoc_input, 0, 0, browse="file", tooltip_key="spoc_input", filetypes=[("FITS files", "*.fits *.fits.gz"), ("All files", "*.*")])
        self._entry(spocf, "Converted CSV output", self.spoc_output_csv, 1, 0, browse="file", tooltip_key="spoc_output_csv", filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        self._entry(spocf, "Converter command template", self.spoc_template, 2, 0, tooltip_key="spoc_template")

        self._trace_vars([
            self.tess_input_mode, self.tess_csv, self.pipeline_dir, self.pipeline_pattern,
            self.pipeline_recursive, self.pipeline_flux, self.tess_force_y_col,
            self.spoc_input, self.spoc_output_csv, self.spoc_template,
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
        self.preview_canvas.bind_all("<MouseWheel>", self._on_preview_mousewheel)
        self.preview_canvas.bind_all("<Shift-MouseWheel>", self._on_preview_shift_mousewheel)
        self.preview_canvas.bind_all("<Control-MouseWheel>", self._on_preview_zoom_mousewheel)

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

    def _update_tess_mode_state(self):
        mode = self.tess_input_mode.get()
        self._set_state_recursive(self.tess_mode_groups["csv"], "normal" if mode == "existing_csv" else "disabled")
        self._set_state_recursive(self.tess_mode_groups["pipe"], "normal" if mode == "pipeline_dir" else "disabled")
        self._set_state_recursive(self.tess_mode_groups["spoc"], "normal" if mode == "spoc_lc_fits" else "disabled")
        self._update_run_plan_preview()

    def _update_analysis_mode_state(self):
        mode = self.analysis_mode.get().strip()

        # Keep separate sensible defaults for guided vs joint mode, but do not
        # overwrite a custom user path.
        cur_out = self.outroot.get().strip()
        if mode == "guided_analysis":
            if cur_out in ("", "tess_joint_outputs", "tess_guided_outputs"):
                self.outroot.set("tess_guided_outputs")
        elif mode == "joint_search":
            if cur_out in ("", "tess_guided_outputs", "tess_joint_outputs"):
                self.outroot.set("tess_joint_outputs")

        for frame in self.analysis_mode_groups.get("guided", []):
            self._set_state_recursive(frame, "normal" if mode == "guided_analysis" else "disabled")
        for frame in self.analysis_mode_groups.get("joint", []):
            self._set_state_recursive(frame, "normal" if mode == "joint_search" else "disabled")

        # These summary-output controls apply to both guided and joint workflows,
        # so keep them enabled regardless of the current analysis mode.
        for w in [
            getattr(self, "summary_save_period_chk", None),
            getattr(self, "summary_save_log_amplitude_chk", None),
        ]:
            if w is not None:
                try:
                    w.configure(state="normal")
                except Exception:
                    pass

        self._update_run_plan_preview()

    def _update_polarimetry_state(self):
        guided_mode = (self.analysis_mode.get().strip() == "guided_analysis")
        use_pol = guided_mode and bool(self.use_polarimetry.get())
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
        state = "normal" if mode == "custom_btjd" else "disabled"
        try:
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

    def _build_converter_command(self) -> list[str] | None:
        if self.tess_input_mode.get() != "spoc_lc_fits":
            return None
        script = self.converter_script.get().strip()
        input_path = self.spoc_input.get().strip()
        output_csv = self.spoc_output_csv.get().strip()
        template = self.spoc_template.get().strip()
        if not script:
            raise ValueError("SPOC converter mode selected but converter script path is empty.")
        if not input_path:
            raise ValueError("SPOC converter mode selected but SPOC input path/glob is empty.")
        if not output_csv:
            raise ValueError("SPOC converter mode selected but converted CSV output path is empty.")
        if not template:
            raise ValueError("SPOC converter command template is empty.")
        rendered = template.format(
            python=sys.executable,
            script=script,
            input=input_path,
            output=output_csv,
        )
        return shlex.split(rendered)

    def _build_config(self) -> dict:
        channels = self._channels_list()
        use_polarimetry_active = (self.analysis_mode.get().strip() == "guided_analysis") and bool(self.use_polarimetry.get())
        if use_polarimetry_active and not channels:
            raise ValueError("Select at least one polarimetric channel.")
        cfg = {
            "guided_script": self.guided_script.get().strip(),
            "joint_script": self.joint_script.get().strip(),
            "analysis_mode": self.analysis_mode.get().strip(),
            "tess_input_mode": self.tess_input_mode.get().strip(),
            "tess_csv": self.tess_csv.get().strip(),
            "pipeline_dir": self.pipeline_dir.get().strip(),
            "pipeline_pattern": self.pipeline_pattern.get().strip(),
            "pipeline_recursive": bool(self.pipeline_recursive.get()),
            "pipeline_flux": self.pipeline_flux.get().strip(),
            "tess_force_y_col": self.tess_force_y_col.get().strip(),
            "use_polarimetry": use_polarimetry_active,
            "pol_csv": self.pol_csv.get().strip(),
            "pol_product": self.pol_product.get().strip(),
            "save_generated_frame": bool(self.save_generated_frame.get()),
            "generated_analysis_dir": self.generated_analysis_dir.get().strip(),
            "outroot": self.outroot.get().strip(),
            "show_plots_inline": bool(self.show_plots_inline.get()),
            "verbose": int(self.verbose.get()),
            "lsq_verbose": int(self.lsq_verbose.get()),
            "fmin": float(self.fmin.get()),
            "fmax": float(self.fmax.get()),
            "tess_grid_mode": self.tess_grid_mode.get().strip(),
            "tess_snr_stop": float(self.tess_snr_stop.get()),
            "max_tess_modes": int(self.max_tess_modes.get()),
            "pol_snr_stop": float(self.pol_snr_stop.get()),
            "max_pol_modes": int(self.max_pol_modes.get()),
            "guided_pol_fmin": float(self.guided_pol_fmin.get()),
            "search_window_mult": float(self.search_window_mult.get()),
            "noise_ks": float(self.noise_ks.get()),
            "noise_bins": int(self.noise_bins.get()),
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
            "plot_preprocess": bool(self.plot_preprocess.get()),
            "preplot_chunk_days": float(self.preplot_chunk_days.get()),
            "preplot_panels": int(self.preplot_panels.get()),
            "preplot_include_pw": bool(self.preplot_include_pw.get()),
            "compute_pw": bool(self.compute_pw.get()),
            "summary_save_period": bool(self.summary_save_period.get()),
            "summary_save_log_amplitude": bool(self.summary_save_log_amplitude.get()),
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

        if cfg["tess_input_mode"] == "existing_csv":
            if not cfg["tess_csv"]:
                raise ValueError("Existing CSV mode selected but no TESS CSV is set.")
            cfg["converter_command"] = None
        elif cfg["tess_input_mode"] == "pipeline_dir":
            if not cfg["pipeline_dir"]:
                raise ValueError("Pipeline/custom CSV mode selected but no pipeline directory is set.")
            cfg["converter_command"] = None
        elif cfg["tess_input_mode"] == "spoc_lc_fits":
            cfg["converter_command"] = self._build_converter_command()
            cfg["tess_csv"] = self.spoc_output_csv.get().strip()
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
            else:
                lines.append("SPOC converter command:")
                lines.append("  " + quote_cmd(cfg["converter_command"]))
                lines.append(f"Converted CSV: {cfg['tess_csv']}")
            lines.extend([
                f"Use polarimetry: {cfg['use_polarimetry']}",
                f"Polarimetry CSV: {cfg['pol_csv'] if cfg['use_polarimetry'] else '(not used)'}",
                f"POL product: {cfg['pol_product'] if cfg['use_polarimetry'] else '(not used)'}",
                f"Channels: {', '.join(cfg['channels']) if cfg['use_polarimetry'] else '(none)'}",
                f"Output root: {cfg['outroot']}",
                f"Phase zero: {cfg['phase_zero_mode']}" + (f" (BTJD={cfg['phase_zero_btjd']})" if cfg['phase_zero_mode']=='custom_btjd' else (" (absolute TESS BTJD=0.0)" if cfg['phase_zero_mode']=='btjd_zero' else "")),
            ])
            if cfg["analysis_mode"] == "guided_analysis":
                lines.append("Run style: wrapper script imports guided module, applies settings, and calls run_analysis().")
            else:
                lines.append("Run style: wrapper script patches the joint-search companion script with the chosen settings and runs the patched copy.")
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
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

CFG = json.loads(r"""{cfg_json}""")

print("GUI runner starting")
print("Analysis mode =", CFG["analysis_mode"])

if CFG.get("converter_command"):
    print("Running SPOC converter:")
    print("  " + " ".join(CFG["converter_command"]))
    rc = subprocess.run(CFG["converter_command"]).returncode
    if rc != 0:
        raise SystemExit(f"SPOC converter failed with exit code {{rc}}")

def _py_literal(value):
    if isinstance(value, str):
        if value.startswith("Path("):
            return value
        return repr(value)
    if isinstance(value, bool):
        return "True" if value else "False"
    if value is None:
        return "None"
    return repr(value)

def _replace_assignment(src: str, varname: str, py_expr: str) -> str:
    pattern = re.compile(rf"(?m)^{{re.escape(varname)}}\s*=\s*.*$")
    repl = f"{{varname}} = {{py_expr}}"
    new_src, n = pattern.subn(repl, src, count=1)
    if n == 0:
        raise SystemExit(f"Could not patch variable {{varname}} in target script.")
    return new_src

if CFG["analysis_mode"] == "guided_analysis":
    script_path = Path(CFG["guided_script"]).expanduser().resolve()
    if not script_path.exists():
        raise SystemExit(f"Guided-analysis script not found: {{script_path}}")

    spec = importlib.util.spec_from_file_location("guided_analysis_gui_module", script_path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"Could not import guided-analysis script: {{script_path}}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    mode = CFG["tess_input_mode"]
    if mode == "existing_csv":
        mod.TESS_INPUT_MODE = "spoc_csv"
        mod.TESS_CSV = Path(CFG["tess_csv"])
    elif mode == "pipeline_dir":
        mod.TESS_INPUT_MODE = "pipeline_dir"
        mod.TESS_PIPELINE_DIR = Path(CFG["pipeline_dir"])
        mod.TESS_PIPELINE_PATTERN = CFG["pipeline_pattern"]
        mod.TESS_PIPELINE_RECURSIVE = bool(CFG["pipeline_recursive"])
        mod.TESS_PIPELINE_FLUX = CFG["pipeline_flux"]
    elif mode == "spoc_lc_fits":
        mod.TESS_INPUT_MODE = "spoc_csv"
        mod.TESS_CSV = Path(CFG["tess_csv"])
    else:
        raise SystemExit(f"Unsupported TESS input mode: {{mode}}")

    mod.TESS_FORCE_Y_COL = CFG["tess_force_y_col"] if CFG["tess_force_y_col"] else None

    mod.USE_POLARIMETRY = bool(CFG["use_polarimetry"])
    mod.POL_CSV = Path(CFG["pol_csv"]) if CFG["pol_csv"] else Path(".")
    mod.POL_PRODUCT = CFG["pol_product"]
    mod.POL_SAVE_GENERATED_ANALYSIS_FRAME = bool(CFG["save_generated_frame"])
    mod.POL_GENERATED_ANALYSIS_DIR = (None if not CFG["generated_analysis_dir"] else str(CFG["generated_analysis_dir"]))
    mod.OUTROOT = Path(CFG["outroot"])
    mod.OUTROOT.mkdir(parents=True, exist_ok=True)
    mod.SHOW_PLOTS_INLINE = bool(CFG["show_plots_inline"])
    mod.VERBOSE = int(CFG["verbose"])
    mod.LSQ_VERBOSE = int(CFG["lsq_verbose"])

    mod.FMIN = float(CFG["fmin"])
    mod.FMAX = float(CFG["fmax"])
    mod.TESS_GRID_MODE = CFG["tess_grid_mode"]
    mod.TESS_SNR_STOP = float(CFG["tess_snr_stop"])
    mod.MAX_TESS_MODES = int(CFG["max_tess_modes"])
    mod.POL_SNR_STOP = float(CFG["pol_snr_stop"])
    mod.MAX_POL_MODES = int(CFG["max_pol_modes"])
    mod.GUIDED_POL_FMIN = float(CFG["guided_pol_fmin"])
    mod.POL_SEARCH_WINDOW_MULT = float(CFG["search_window_mult"])
    mod.POL_LOCAL_NOISE_KS = float(CFG["noise_ks"])
    mod.POL_LOCAL_NOISE_SIDE_BINS = int(CFG["noise_bins"])
    mod.POL_CHANNELS = list(CFG["channels"]) if CFG["use_polarimetry"] else []
    mod.USE_POL_NIGHT_OFFSETS = bool(CFG["use_offsets"])
    mod.USE_POL_NIGHT_SLOPES = bool(CFG["use_slopes"])
    mod.POL_NIGHT_GROUP_MODE = CFG["group_mode"]
    mod.POL_NIGHT_GAP_HOURS = float(CFG["gap_hours"])
    mod.DO_DETREND = bool(CFG["do_detrend"])
    mod.DETREND_POLY_ORDER = int(CFG["detrend_order"])
    mod.N_PHASE_PLOTS = int(CFG["n_phase_plots"])
    mod.PHASE_SORT_BY = CFG["phase_sort_by"]
    mod.PHASE_PLOT_STYLE = CFG["phase_plot_style"]
    mod.PHASE_ZERO_MODE = CFG["phase_zero_mode"]
    mod.PHASE_ZERO_BTJD = float(CFG["phase_zero_btjd"])

    mod.POL_PLOT_PREPROCESS_DIAGNOSTICS = bool(CFG["plot_preprocess"])
    mod.POL_PREPROCESS_PLOT_CHUNK_DAYS = float(CFG["preplot_chunk_days"])
    mod.POL_PREPROCESS_PLOT_PANELS_PER_FIG = int(CFG["preplot_panels"])
    mod.POL_PREPROCESS_PLOT_INCLUDE_PREWHITEN = bool(CFG["preplot_include_pw"])
    mod.POL_COMPUTE_PREWHITEN_PRODUCT = bool(CFG["compute_pw"])
    mod.SUMMARY_SAVE_PERIOD_VERSION = bool(CFG["summary_save_period"])
    mod.SUMMARY_SAVE_LOG_AMPLITUDE_VERSION = bool(CFG["summary_save_log_amplitude"])

    print("Running guided analysis...")
    outputs = mod.run_analysis()
    if isinstance(outputs, dict):
        print("Output keys:", sorted(outputs.keys()))
    print("GUI runner finished")

elif CFG["analysis_mode"] == "joint_search":
    script_path = Path(CFG["joint_script"]).expanduser().resolve()
    if not script_path.exists():
        raise SystemExit(f"Joint-search script not found: {{script_path}}")

    src = script_path.read_text(encoding="utf-8")
    if 'import matplotlib.pyplot as plt' in src and 'matplotlib.use(' not in src:
        src = src.replace(
            'import matplotlib.pyplot as plt',
            'import matplotlib\\nmatplotlib.use("Agg")\\nimport matplotlib.pyplot as plt',
            1
        )

    if CFG["tess_input_mode"] == "pipeline_dir":
        tess_input_mode = "pipeline_dir"
        tess_pipeline_dir = CFG["pipeline_dir"]
        tess_pipeline_pattern = CFG["pipeline_pattern"]
        tess_pipeline_recursive = bool(CFG["pipeline_recursive"])
        tess_pipeline_flux = CFG["pipeline_flux"]
    else:
        # For joint-search mode, a single CSV (including converted SPOC CSVs and
        # detrender outputs) is most robustly handled by the pipeline_dir loader
        # using the parent directory + exact filename pattern.
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
    needed = ["jd"] + pol_required.get(pol_product, []) + ["q_err", "u_err", "p_err"]
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
        spec.loader.exec_module(gmod)

        gmod.OUTROOT = Path(CFG["outroot"])
        gmod.OUTROOT.mkdir(parents=True, exist_ok=True)
        gmod.POL_PLOT_PREPROCESS_DIAGNOSTICS = bool(CFG["plot_preprocess"])
        gmod.POL_PREPROCESS_PLOT_CHUNK_DAYS = float(CFG["preplot_chunk_days"])
        gmod.POL_PREPROCESS_PLOT_PANELS_PER_FIG = int(CFG["preplot_panels"])
        gmod.POL_PREPROCESS_PLOT_INCLUDE_PREWHITEN = bool(CFG["preplot_include_pw"])
        gmod.POL_COMPUTE_PREWHITEN_PRODUCT = bool(CFG["compute_pw"]) or (pol_product == "pw_resid_nm_pchip")
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

    assign_map = {{
        "TESS_INPUT_MODE": _py_literal(tess_input_mode),
        "TESS_CSV": _py_literal(f'Path({{repr(CFG["tess_csv"])}})'),
        "TESS_PIPELINE_DIR": _py_literal(f'Path({{repr(tess_pipeline_dir)}})'),
        "TESS_PIPELINE_PATTERN": _py_literal(tess_pipeline_pattern),
        "TESS_PIPELINE_RECURSIVE": _py_literal(bool(tess_pipeline_recursive)),
        "TESS_PIPELINE_FLUX": _py_literal(tess_pipeline_flux),
        "POL_CSV": _py_literal(f'Path({{repr(pol_csv_for_joint)}})'),
        "POL_PRODUCT": _py_literal(pol_product),
        "FMIN": _py_literal(float(CFG["fmin"])),
        "FMAX": _py_literal(float(CFG["fmax"])),
        "K_CANDIDATES": _py_literal(int(CFG["joint_k_candidates"])),
        "TOP_N_RAW_TESS_CANDIDATES": _py_literal(int(CFG["joint_top_n_raw_tess"])),
        "COARSE_OVERSAMPLE": _py_literal(float(CFG["joint_coarse_oversample"])),
        "REFINE_FACTOR": _py_literal(int(CFG["joint_refine_factor"])),
        "MAX_ITERS": _py_literal(int(CFG["joint_max_iters"])),
        "KFIT": _py_literal(float(CFG["joint_kfit"])),
        "SNR_STOP": _py_literal(float(CFG["joint_snr_stop"])),
        "W_PREFILTER": _py_literal(float(CFG["joint_w_prefilter"])),
        "KS_TESS": _py_literal(float(CFG["joint_ks_tess"])),
        "KS_POL": _py_literal(float(CFG["joint_ks_pol"])),
        "TRIM_TOP_FRAC": _py_literal(float(CFG["joint_trim_top_frac"])),
        "USE_POL_NIGHT_OFFSETS": _py_literal(bool(CFG["use_offsets"])),
        "USE_POL_NIGHT_SLOPES": _py_literal(bool(CFG["use_slopes"])),
        "POL_NIGHT_GROUP_MODE": _py_literal(CFG["group_mode"]),
        "POL_NIGHT_GAP_HOURS": _py_literal(float(CFG["gap_hours"])),
        "DO_DETREND": _py_literal(bool(CFG["do_detrend"])),
        "DETREND_POLY_ORDER": _py_literal(int(CFG["detrend_order"])),
        "N_PHASE_PLOTS": _py_literal(int(CFG["n_phase_plots"])),
        "PHASE_SORT_BY": _py_literal(CFG["phase_sort_by"]),
        "PHASE_PLOT_STYLE": _py_literal(CFG["phase_plot_style"]),
        "PHASE_ZERO_MODE": _py_literal(CFG["phase_zero_mode"]),
        "PHASE_ZERO_BTJD": _py_literal(float(CFG["phase_zero_btjd"])),
        "SUMMARY_SAVE_PERIOD_VERSION": _py_literal(bool(CFG["summary_save_period"])),
        "SUMMARY_SAVE_LOG_AMPLITUDE_VERSION": _py_literal(bool(CFG["summary_save_log_amplitude"])),
        "JOINT_WEIGHT_MODE": _py_literal(CFG["joint_weight_mode"]),
        "SCALE_FREE_WEIGHT_BASIS": _py_literal(CFG["joint_scale_free_basis"]),
        "MANUAL_W_TESS": _py_literal(float(CFG["joint_manual_w_tess"])),
        "MANUAL_W_POL": _py_literal(float(CFG["joint_manual_w_pol"])),
        "SHOW_PLOTS_INLINE": _py_literal(bool(CFG["show_plots_inline"])),
        "OUTROOT": _py_literal(f'Path({{repr(CFG["outroot"])}})'),
    }}

    for varname, py_expr in assign_map.items():
        src = _replace_assignment(src, varname, py_expr)

    channels = list(CFG["channels"])
    src = src.replace('for k in ["q", "u", "p"]:', f"for k in {{channels!r}}:")
    src = src.replace('for k in ["q", "u", "p"]:', f"for k in {{channels!r}}:")
    src = src.replace('results["q"].head()', 'print("Joint output channels:", sorted(results.keys()))')

    patched_path = Path(tempfile.gettempdir()) / f"joint_search_gui_patched_{{os.getpid()}}.py"
    patched_path.write_text(src, encoding="utf-8")
    print("Running joint search via patched companion script:")
    print("  " + str(patched_path))
    run_env = os.environ.copy()
    run_env["MPLBACKEND"] = "Agg"
    run_env.setdefault("QT_QPA_PLATFORM", "offscreen")
    rc = subprocess.run([sys.executable, "-u", str(patched_path)], env=run_env).returncode
    if rc != 0:
        raise SystemExit(f"Joint-search script failed with exit code {{rc}}")
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
        try:
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

    def clear_log(self):
        self.log_text.delete("1.0", "end")

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
        return {
            k: getattr(self, k).get() for k in [
                "guided_script", "joint_script", "converter_script", "analysis_mode", "tess_input_mode", "tess_csv",
                "pipeline_dir", "pipeline_pattern", "pipeline_recursive", "pipeline_flux", "tess_force_y_col",
                "spoc_input", "spoc_output_csv", "spoc_template", "pol_csv", "pol_product",
                "save_generated_frame", "generated_analysis_dir", "outroot", "show_plots_inline",
                "verbose", "lsq_verbose", "fmin", "fmax", "tess_grid_mode", "tess_snr_stop",
                "max_tess_modes", "pol_snr_stop", "max_pol_modes", "guided_pol_fmin",
                "search_window_mult", "noise_ks", "noise_bins", "use_offsets", "use_slopes",
                "group_mode", "gap_hours", "do_detrend", "detrend_order", "n_phase_plots",
                "phase_sort_by", "phase_plot_style", "phase_zero_mode", "phase_zero_btjd", "plot_preprocess", "preplot_chunk_days",
                "preplot_panels", "preplot_include_pw", "compute_pw", "summary_save_period", "summary_save_log_amplitude",
                "joint_k_candidates", "joint_top_n_raw_tess", "joint_coarse_oversample",
                "joint_refine_factor", "joint_max_iters", "joint_kfit", "joint_snr_stop",
                "joint_w_prefilter", "joint_ks_tess", "joint_ks_pol", "joint_trim_top_frac",
                "joint_weight_mode", "joint_scale_free_basis", "joint_manual_w_tess", "joint_manual_w_pol", "preview_dir",
            ]
        } | {
            "channel_q": self.channel_q.get(),
            "channel_u": self.channel_u.get(),
            "channel_p": self.channel_p.get(),
        }

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
        for key, value in data.items():
            if hasattr(self, key):
                try:
                    getattr(self, key).set(value)
                except Exception:
                    pass
        self._update_tess_mode_state()
        self._update_analysis_mode_state()
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
    filemenu.add_command(label="Exit", command=app.destroy)
    menubar.add_cascade(label="File", menu=filemenu)

    helpmenu = tk.Menu(menubar, tearoff=0)
    helpmenu.add_command(label="Help tab", command=app.show_help_tab)
    helpmenu.add_command(label="Copy help text", command=app.copy_help_text)
    menubar.add_cascade(label="Help", menu=helpmenu)
    app.config(menu=menubar)


def main():
    app = GuidedAnalysisGUI()
    attach_menu(app)
    app.mainloop()


if __name__ == "__main__":
    main()

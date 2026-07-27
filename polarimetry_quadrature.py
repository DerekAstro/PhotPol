"""Shared constrained Stokes-q/u sinusoid fitting utilities.

The Guided and Joint backends deliberately keep their existing periodogram
searches.  This module is used only after candidate selection, where the
original, unsmoothed time series are refit with a common frequency and a
physically motivated q/u phase relation.

The fitted convention is

    q(t) = baseline_q(t) + A_q sin(2 pi f t + phi)
    u(t) = baseline_u(t) + A_u sin(2 pi f t + phi + s pi/2)

with non-negative amplitudes and s in {-1, +1}.  For ``auto`` mode, both
allowed signs are tested independently for every candidate before the final
simultaneous multisinusoid fit.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    from scipy.optimize import least_squares
except Exception:  # pragma: no cover - the package requirements include SciPy
    least_squares = None


QUADRATURE_MODES = ("free", "auto", "force_plus", "force_minus")


def normalize_quadrature_mode(value: str | None) -> str:
    """Return the canonical quadrature-mode name or raise a clear error."""
    text = str(value or "free").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "off": "free",
        "none": "free",
        "unconstrained": "free",
        "automatic": "auto",
        "auto_quadrature": "auto",
        "+pi/2": "force_plus",
        "plus": "force_plus",
        "positive": "force_plus",
        "-pi/2": "force_minus",
        "minus": "force_minus",
        "negative": "force_minus",
    }
    text = aliases.get(text, text)
    if text not in QUADRATURE_MODES:
        raise ValueError(
            "q/u phase mode must be one of "
            f"{', '.join(QUADRATURE_MODES)}; received {value!r}."
        )
    return text


def wrap_phase_radians(value):
    """Wrap a phase or phase array to [-pi, pi)."""
    arr = np.asarray(value, dtype=float)
    wrapped = (arr + np.pi) % (2.0 * np.pi) - np.pi
    return float(wrapped) if wrapped.ndim == 0 else wrapped


def phase_at_reference(
    phase_local: float,
    frequency_cpd: float,
    local_zero_time: float,
    reference_time: float,
) -> float:
    """Transform a local-time sinusoid phase to a shared absolute epoch."""
    return float(
        wrap_phase_radians(
            float(phase_local)
            + 2.0
            * np.pi
            * float(frequency_cpd)
            * (float(reference_time) - float(local_zero_time))
        )
    )


def _safe_weights(values: np.ndarray | None, size: int) -> np.ndarray:
    if values is None:
        return np.ones(size, dtype=float)
    w = np.asarray(values, dtype=float)
    if w.shape != (size,):
        raise ValueError("Weight array length does not match its time series.")
    w = np.where(np.isfinite(w) & (w > 0), w, 0.0)
    positive = w[w > 0]
    if positive.size:
        median = float(np.nanmedian(positive))
        if np.isfinite(median) and median > 0:
            w = w / median
    return w


def _weighted_linear_solve(
    y: np.ndarray,
    design: np.ndarray,
    weights: np.ndarray,
) -> dict:
    sw = np.sqrt(np.asarray(weights, dtype=float))
    beta, *_ = np.linalg.lstsq(design * sw[:, None], y * sw, rcond=None)
    model = design @ beta
    resid = y - model
    return {
        "beta": np.asarray(beta, dtype=float),
        "model": np.asarray(model, dtype=float),
        "resid": np.asarray(resid, dtype=float),
        "rss": float(np.sum(weights * np.square(resid))),
    }


def _unconstrained_multisin(
    time: np.ndarray,
    values: np.ndarray,
    weights: np.ndarray,
    baseline: np.ndarray,
    frequencies: np.ndarray,
) -> dict:
    columns = []
    for frequency in frequencies:
        angle = 2.0 * np.pi * float(frequency) * time
        columns.extend((np.sin(angle), np.cos(angle)))
    design = (
        np.column_stack(columns + [baseline[:, i] for i in range(baseline.shape[1])])
        if columns
        else np.asarray(baseline, dtype=float)
    )
    fit = _weighted_linear_solve(values, design, weights)
    rows = []
    for index, frequency in enumerate(frequencies):
        sine = float(fit["beta"][2 * index])
        cosine = float(fit["beta"][2 * index + 1])
        rows.append(
            {
                "frequency": float(frequency),
                "sine": sine,
                "cosine": cosine,
                "amplitude": float(np.hypot(sine, cosine)),
                "phase": float(np.arctan2(cosine, sine)),
            }
        )
    signal_columns = 2 * len(frequencies)
    signal_model = (
        design[:, :signal_columns] @ fit["beta"][:signal_columns]
        if signal_columns
        else np.zeros_like(values)
    )
    baseline_model = baseline @ fit["beta"][signal_columns:]
    return {
        **fit,
        "rows": rows,
        "signal_model": np.asarray(signal_model, dtype=float),
        "baseline_model": np.asarray(baseline_model, dtype=float),
    }


@dataclass
class _Series:
    time: np.ndarray
    values: np.ndarray
    weights: np.ndarray
    baseline: np.ndarray


def _prepare_series(
    time,
    values,
    weights,
    baseline,
    label: str,
) -> _Series:
    time = np.asarray(time, dtype=float)
    values = np.asarray(values, dtype=float)
    baseline = np.asarray(baseline, dtype=float)
    if time.ndim != 1 or values.ndim != 1 or time.size != values.size:
        raise ValueError(f"{label} time and value arrays must be equal-length 1-D arrays.")
    if baseline.ndim != 2 or baseline.shape[0] != time.size:
        raise ValueError(f"{label} baseline matrix has an incompatible shape.")
    if not np.all(np.isfinite(time)) or not np.all(np.isfinite(values)):
        raise ValueError(f"{label} time series contains nonfinite values.")
    if not np.all(np.isfinite(baseline)):
        raise ValueError(f"{label} baseline matrix contains nonfinite values.")
    return _Series(
        time=time,
        values=values,
        weights=_safe_weights(weights, time.size),
        baseline=baseline,
    )


def _frequency_bounds(
    seeds: np.ndarray,
    lower: np.ndarray | None,
    upper: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    if lower is None or upper is None:
        raise ValueError("Frequency bounds are required when frequency optimization is enabled.")
    lower = np.asarray(lower, dtype=float).copy()
    upper = np.asarray(upper, dtype=float).copy()
    if lower.shape != seeds.shape or upper.shape != seeds.shape:
        raise ValueError("Frequency-bound arrays must match the seed-frequency array.")
    order = np.argsort(seeds)
    sorted_seeds = seeds[order]
    for position in range(1, len(sorted_seeds)):
        midpoint = 0.5 * (sorted_seeds[position - 1] + sorted_seeds[position])
        previous = order[position - 1]
        current = order[position]
        upper[previous] = min(upper[previous], midpoint - 1e-10)
        lower[current] = max(lower[current], midpoint + 1e-10)
    if np.any(~np.isfinite(lower)) or np.any(~np.isfinite(upper)):
        raise ValueError("Frequency bounds must be finite.")
    if np.any(lower >= upper) or np.any(seeds < lower) or np.any(seeds > upper):
        raise ValueError("Frequency bounds do not contain distinct seed frequencies.")
    return lower, upper


def _initial_shared_phase(phase_q: float, phase_u: float, sign: int, amp_q: float, amp_u: float) -> float:
    aligned_u = float(phase_u) - float(sign) * np.pi / 2.0
    vector = max(float(amp_q), 1e-12) * np.exp(1j * float(phase_q))
    vector += max(float(amp_u), 1e-12) * np.exp(1j * aligned_u)
    if abs(vector) == 0:
        return float(wrap_phase_radians(phase_q))
    return float(np.angle(vector))


def _fit_with_signs(
    q: _Series,
    u: _Series,
    frequencies: np.ndarray,
    signs: np.ndarray,
    *,
    optimize_frequencies: bool,
    frequency_lower: np.ndarray | None,
    frequency_upper: np.ndarray | None,
    tess: _Series | None,
    tess_dataset_weight: float,
    pol_dataset_weight: float,
    max_nfev: int,
) -> dict:
    if least_squares is None:
        raise RuntimeError("SciPy least_squares is required for constrained q/u fitting.")

    nmode = len(frequencies)
    unc_q = _unconstrained_multisin(q.time, q.values, q.weights, q.baseline, frequencies)
    unc_u = _unconstrained_multisin(u.time, u.values, u.weights, u.baseline, frequencies)
    unc_t = (
        _unconstrained_multisin(tess.time, tess.values, tess.weights, tess.baseline, frequencies)
        if tess is not None
        else None
    )

    phase0 = np.array(
        [
            _initial_shared_phase(
                unc_q["rows"][i]["phase"],
                unc_u["rows"][i]["phase"],
                int(signs[i]),
                unc_q["rows"][i]["amplitude"],
                unc_u["rows"][i]["amplitude"],
            )
            for i in range(nmode)
        ],
        dtype=float,
    )
    amp_q0 = np.maximum(
        [row["amplitude"] for row in unc_q["rows"]],
        np.finfo(float).eps,
    )
    amp_u0 = np.maximum(
        [row["amplitude"] for row in unc_u["rows"]],
        np.finfo(float).eps,
    )

    pieces = []
    lower_pieces = []
    upper_pieces = []
    if optimize_frequencies:
        flo, fhi = _frequency_bounds(frequencies, frequency_lower, frequency_upper)
        pieces.append(frequencies)
        lower_pieces.append(flo)
        upper_pieces.append(fhi)
    pieces.extend((phase0, amp_q0, amp_u0))
    lower_pieces.extend(
        (
            np.full(nmode, -np.pi),
            np.zeros(nmode),
            np.zeros(nmode),
        )
    )
    upper_pieces.extend(
        (
            np.full(nmode, np.pi),
            np.full(nmode, np.inf),
            np.full(nmode, np.inf),
        )
    )

    if tess is not None and unc_t is not None:
        tess_s0 = np.array([row["sine"] for row in unc_t["rows"]], dtype=float)
        tess_c0 = np.array([row["cosine"] for row in unc_t["rows"]], dtype=float)
        pieces.extend((tess_s0, tess_c0))
        lower_pieces.extend((np.full(nmode, -np.inf), np.full(nmode, -np.inf)))
        upper_pieces.extend((np.full(nmode, np.inf), np.full(nmode, np.inf)))

    pieces.extend(
        (
            unc_q["beta"][2 * nmode :],
            unc_u["beta"][2 * nmode :],
        )
    )
    lower_pieces.extend(
        (
            np.full(q.baseline.shape[1], -np.inf),
            np.full(u.baseline.shape[1], -np.inf),
        )
    )
    upper_pieces.extend(
        (
            np.full(q.baseline.shape[1], np.inf),
            np.full(u.baseline.shape[1], np.inf),
        )
    )
    if tess is not None and unc_t is not None:
        pieces.append(unc_t["beta"][2 * nmode :])
        lower_pieces.append(np.full(tess.baseline.shape[1], -np.inf))
        upper_pieces.append(np.full(tess.baseline.shape[1], np.inf))

    x0 = np.concatenate([np.asarray(piece, dtype=float) for piece in pieces])
    lower_bounds = np.concatenate(lower_pieces)
    upper_bounds = np.concatenate(upper_pieces)

    def unpack(parameters: np.ndarray):
        cursor = 0
        if optimize_frequencies:
            fit_frequencies = parameters[cursor : cursor + nmode]
            cursor += nmode
        else:
            fit_frequencies = frequencies
        phases = parameters[cursor : cursor + nmode]
        cursor += nmode
        amp_q = parameters[cursor : cursor + nmode]
        cursor += nmode
        amp_u = parameters[cursor : cursor + nmode]
        cursor += nmode
        tess_s = tess_c = None
        if tess is not None:
            tess_s = parameters[cursor : cursor + nmode]
            cursor += nmode
            tess_c = parameters[cursor : cursor + nmode]
            cursor += nmode
        baseline_q = parameters[cursor : cursor + q.baseline.shape[1]]
        cursor += q.baseline.shape[1]
        baseline_u = parameters[cursor : cursor + u.baseline.shape[1]]
        cursor += u.baseline.shape[1]
        baseline_t = None
        if tess is not None:
            baseline_t = parameters[cursor : cursor + tess.baseline.shape[1]]
            cursor += tess.baseline.shape[1]
        if cursor != len(parameters):
            raise RuntimeError("Internal constrained-fit parameter indexing error.")
        return (
            fit_frequencies,
            phases,
            amp_q,
            amp_u,
            tess_s,
            tess_c,
            baseline_q,
            baseline_u,
            baseline_t,
        )

    def models(parameters: np.ndarray):
        (
            fit_frequencies,
            phases,
            amp_q,
            amp_u,
            tess_s,
            tess_c,
            baseline_q,
            baseline_u,
            baseline_t,
        ) = unpack(parameters)
        signal_q = np.zeros_like(q.values)
        signal_u = np.zeros_like(u.values)
        signal_t = np.zeros_like(tess.values) if tess is not None else None
        for i, frequency in enumerate(fit_frequencies):
            signal_q += amp_q[i] * np.sin(
                2.0 * np.pi * frequency * q.time + phases[i]
            )
            signal_u += amp_u[i] * np.sin(
                2.0 * np.pi * frequency * u.time
                + phases[i]
                + signs[i] * np.pi / 2.0
            )
            if tess is not None and signal_t is not None:
                angle_t = 2.0 * np.pi * frequency * tess.time
                signal_t += tess_s[i] * np.sin(angle_t) + tess_c[i] * np.cos(angle_t)
        base_q = q.baseline @ baseline_q
        base_u = u.baseline @ baseline_u
        base_t = tess.baseline @ baseline_t if tess is not None else None
        return {
            "frequencies": np.asarray(fit_frequencies, dtype=float),
            "phases": np.asarray(phases, dtype=float),
            "amp_q": np.asarray(amp_q, dtype=float),
            "amp_u": np.asarray(amp_u, dtype=float),
            "tess_s": None if tess_s is None else np.asarray(tess_s, dtype=float),
            "tess_c": None if tess_c is None else np.asarray(tess_c, dtype=float),
            "signal_q": signal_q,
            "signal_u": signal_u,
            "signal_tess": signal_t,
            "baseline_q": base_q,
            "baseline_u": base_u,
            "baseline_tess": base_t,
            "model_q": signal_q + base_q,
            "model_u": signal_u + base_u,
            "model_tess": None if tess is None else signal_t + base_t,
        }

    sqrt_q = np.sqrt(q.weights * max(float(pol_dataset_weight), 0.0))
    sqrt_u = np.sqrt(u.weights * max(float(pol_dataset_weight), 0.0))
    sqrt_t = (
        np.sqrt(tess.weights * max(float(tess_dataset_weight), 0.0))
        if tess is not None
        else None
    )

    def residual_vector(parameters: np.ndarray):
        model = models(parameters)
        residuals = [
            (q.values - model["model_q"]) * sqrt_q,
            (u.values - model["model_u"]) * sqrt_u,
        ]
        if tess is not None:
            residuals.append((tess.values - model["model_tess"]) * sqrt_t)
        return np.concatenate(residuals)

    result = least_squares(
        residual_vector,
        x0=x0,
        bounds=(lower_bounds, upper_bounds),
        method="trf",
        xtol=1e-11,
        ftol=1e-11,
        gtol=1e-11,
        max_nfev=max(20, int(max_nfev)),
    )
    fitted = models(result.x)
    fitted["success"] = bool(result.success)
    fitted["message"] = str(result.message)
    fitted["nfev"] = int(result.nfev)
    fitted["weighted_rss"] = float(np.sum(np.square(residual_vector(result.x))))
    fitted["resid_q"] = q.values - fitted["model_q"]
    fitted["resid_u"] = u.values - fitted["model_u"]
    fitted["resid_tess"] = (
        None if tess is None else tess.values - fitted["model_tess"]
    )
    return fitted


def fit_quadrature_multisin(
    *,
    q_time,
    q_values,
    q_weights,
    q_baseline,
    u_time,
    u_values,
    u_weights,
    u_baseline,
    seed_frequencies,
    mode: str,
    optimize_frequencies: bool = False,
    frequency_lower=None,
    frequency_upper=None,
    tess_time=None,
    tess_values=None,
    tess_weights=None,
    tess_baseline=None,
    tess_dataset_weight: float = 1.0,
    pol_dataset_weight: float = 1.0,
    max_nfev: int = 300,
) -> dict:
    """Fit a simultaneous multisinusoid model with constrained q/u phases."""
    canonical_mode = normalize_quadrature_mode(mode)
    if canonical_mode == "free":
        raise ValueError("fit_quadrature_multisin requires an active quadrature mode.")

    frequencies = np.asarray(seed_frequencies, dtype=float)
    if frequencies.ndim != 1 or frequencies.size == 0:
        raise ValueError("At least one finite seed frequency is required.")
    if not np.all(np.isfinite(frequencies)) or np.any(frequencies <= 0):
        raise ValueError("Seed frequencies must be finite and positive.")
    order = np.argsort(frequencies)
    frequencies = frequencies[order]
    if frequency_lower is not None:
        frequency_lower = np.asarray(frequency_lower, dtype=float)[order]
    if frequency_upper is not None:
        frequency_upper = np.asarray(frequency_upper, dtype=float)[order]

    q = _prepare_series(q_time, q_values, q_weights, q_baseline, "q")
    u = _prepare_series(u_time, u_values, u_weights, u_baseline, "u")
    tess = None
    if tess_time is not None:
        if tess_values is None or tess_baseline is None:
            raise ValueError("TESS time, values, and baseline must be supplied together.")
        tess = _prepare_series(
            tess_time,
            tess_values,
            tess_weights,
            tess_baseline,
            "TESS",
        )

    if canonical_mode == "force_plus":
        signs = np.ones(len(frequencies), dtype=int)
    elif canonical_mode == "force_minus":
        signs = -np.ones(len(frequencies), dtype=int)
    else:
        # Determine the handedness independently for each selected frequency.
        # Both signs are fitted to the unsmoothed q/u data; the lower weighted
        # residual sum of squares is retained for that frequency.
        signs = np.ones(len(frequencies), dtype=int)
        for i, frequency in enumerate(frequencies):
            sign_scores = {}
            for sign in (-1, 1):
                trial = _fit_with_signs(
                    q,
                    u,
                    np.array([frequency], dtype=float),
                    np.array([sign], dtype=int),
                    optimize_frequencies=False,
                    frequency_lower=None,
                    frequency_upper=None,
                    tess=None,
                    tess_dataset_weight=1.0,
                    pol_dataset_weight=pol_dataset_weight,
                    max_nfev=max(50, min(int(max_nfev), 160)),
                )
                sign_scores[sign] = float(trial["weighted_rss"])
            signs[i] = min(sign_scores, key=sign_scores.get)

    fitted = _fit_with_signs(
        q,
        u,
        frequencies,
        signs,
        optimize_frequencies=bool(optimize_frequencies),
        frequency_lower=frequency_lower,
        frequency_upper=frequency_upper,
        tess=tess,
        tess_dataset_weight=tess_dataset_weight,
        pol_dataset_weight=pol_dataset_weight,
        max_nfev=max_nfev,
    )

    final_frequencies = np.asarray(fitted["frequencies"], dtype=float)
    unc_q = _unconstrained_multisin(
        q.time, q.values, q.weights, q.baseline, final_frequencies
    )
    unc_u = _unconstrained_multisin(
        u.time, u.values, u.weights, u.baseline, final_frequencies
    )
    unc_t = (
        _unconstrained_multisin(
            tess.time, tess.values, tess.weights, tess.baseline, final_frequencies
        )
        if tess is not None
        else None
    )

    rows = []
    for i, frequency in enumerate(final_frequencies):
        phase_q = float(wrap_phase_radians(fitted["phases"][i]))
        phase_u = float(
            wrap_phase_radians(
                fitted["phases"][i] + signs[i] * np.pi / 2.0
            )
        )
        tess_s = tess_c = amp_tess = phase_tess = np.nan
        if tess is not None:
            tess_s = float(fitted["tess_s"][i])
            tess_c = float(fitted["tess_c"][i])
            amp_tess = float(np.hypot(tess_s, tess_c))
            phase_tess = float(np.arctan2(tess_c, tess_s))
        rows.append(
            {
                "component": i + 1,
                "frequency_cpd": float(frequency),
                "amp_q": float(fitted["amp_q"][i]),
                "phase_q_rad": phase_q,
                "amp_u": float(fitted["amp_u"][i]),
                "phase_u_rad": phase_u,
                "phase_diff_u_minus_q_rad": float(
                    wrap_phase_radians(phase_u - phase_q)
                ),
                "quadrature_sign": int(signs[i]),
                "quadrature_relation": "+pi/2" if signs[i] > 0 else "-pi/2",
                "unconstrained_amp_q": float(unc_q["rows"][i]["amplitude"]),
                "unconstrained_phase_q_rad": float(
                    wrap_phase_radians(unc_q["rows"][i]["phase"])
                ),
                "unconstrained_amp_u": float(unc_u["rows"][i]["amplitude"]),
                "unconstrained_phase_u_rad": float(
                    wrap_phase_radians(unc_u["rows"][i]["phase"])
                ),
                "unconstrained_phase_diff_u_minus_q_rad": float(
                    wrap_phase_radians(
                        unc_u["rows"][i]["phase"] - unc_q["rows"][i]["phase"]
                    )
                ),
                "amp_tess": amp_tess,
                "phase_tess_rad": phase_tess,
                "tess_sine_coefficient": tess_s,
                "tess_cosine_coefficient": tess_c,
            }
        )

    fitted.update(
        {
            "mode": canonical_mode,
            "signs": signs,
            "component_rows": rows,
            "unconstrained_q": unc_q,
            "unconstrained_u": unc_u,
            "unconstrained_tess": unc_t,
            "unconstrained_weighted_rss_q_u": float(
                pol_dataset_weight * (unc_q["rss"] + unc_u["rss"])
            ),
        }
    )
    if unc_t is not None:
        fitted["unconstrained_weighted_rss_total"] = float(
            fitted["unconstrained_weighted_rss_q_u"]
            + tess_dataset_weight * unc_t["rss"]
        )
    else:
        fitted["unconstrained_weighted_rss_total"] = fitted[
            "unconstrained_weighted_rss_q_u"
        ]
    return fitted

from __future__ import annotations

import numpy as np
import scipy.optimize
from scipy.special import erf

from hplc.models.peak import MathPeak2D


def _skew_normal(t, amplitude, mu, sigma, alpha):
    """Skew-normal peak shape: I(t) = 2 * A * phi(t; mu, sigma) * Phi(alpha * z)."""
    z = (t - mu) / sigma
    phi = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * z ** 2)
    cdf = 0.5 * (1 + erf(alpha * z / np.sqrt(2)))
    return 2 * amplitude * phi * cdf


def _multi_skew_normal(t, *params):
    """Sum of N skew-normals. params = [amp1, mu1, sigma1, alpha1, amp2, ...]"""
    result = np.zeros_like(t, dtype=float)
    for i in range(len(params) // 4):
        result += _skew_normal(t, *params[i * 4:(i + 1) * 4])
    return result


def _initial_guess(time, signal, start, end, dt):
    """Seed one peak's parameters from the tallest point inside its window."""
    window = (time >= start) & (time <= end)
    if not window.any():
        raise ValueError(
            f"No data points between {start:g} and {end:g}; "
            f"signal spans {time[0]:g}-{time[-1]:g}"
        )

    t_win = time[window]
    s_win = signal[window]
    mu = float(t_win[np.argmax(s_win)])
    height = max(float(s_win.max()), 0.0)

    # Half-width at half maximum inside the window, falling back to a
    # fraction of the window when the peak fills it completely.
    above = t_win[s_win >= height / 2]
    hwhm = (float(above[-1]) - float(above[0])) / 2 if len(above) > 1 else (end - start) / 6
    sigma = max(hwhm / np.sqrt(2 * np.log(2)), dt)

    amplitude = max(height * sigma * np.sqrt(2 * np.pi), dt)
    return [amplitude, mu, sigma, 0.0]


def detect_peaks(
    time: np.ndarray,
    signal: np.ndarray,
    peak_retention_times: list[tuple[float, float]],
) -> list[MathPeak2D]:
    """Fit one skew-normal peak per requested retention-time window.

    `peak_retention_times` is a (start, end) window per expected peak; the
    fitted retention time is constrained to lie inside its own window, which
    is what keeps overlapping peaks from collapsing onto each other. All the
    peaks are fitted together, so shared signal is deconvolved rather than
    double-counted.

    Returns one MathPeak2D per window, sorted by retention time.
    """
    time = np.asarray(time, dtype=float)
    signal = np.asarray(signal, dtype=float)

    if len(time) != len(signal):
        raise ValueError(
            f"time and signal must have the same length, "
            f"got {len(time)} and {len(signal)}"
        )
    if not peak_retention_times:
        return []

    dt = float(np.mean(np.diff(time))) if len(time) > 1 else 1.0
    span = float(time[-1] - time[0]) or dt

    p0: list[float] = []
    lower: list[float] = []
    upper: list[float] = []

    for start, end in peak_retention_times:
        if end <= start:
            raise ValueError(
                f"Each retention-time window must be (start, end) with "
                f"end > start, got ({start}, {end})"
            )
        guess = _initial_guess(time, signal, start, end, dt)
        p0.extend(guess)
        # Retention time is pinned to its window; the shape is free.
        lower.extend([0.0, start, dt / 2, -50.0])
        upper.extend([np.inf, end, span, 50.0])

    popt, _ = scipy.optimize.curve_fit(
        _multi_skew_normal,
        time,
        signal,
        p0=p0,
        bounds=(lower, upper),
        maxfev=20000,
    )

    peaks = [
        MathPeak2D(
            mu=float(mu),
            amplitude=float(amplitude),
            sigma=float(sigma),
            alpha=float(alpha),
            time=time,
        )
        for amplitude, mu, sigma, alpha in (popt[i:i + 4] for i in range(0, len(popt), 4))
    ]
    return sorted(peaks, key=lambda peak: peak.retention_time)

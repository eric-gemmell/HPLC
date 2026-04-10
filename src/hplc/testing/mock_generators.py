import numpy as np
from scipy.special import erf
from hplc.models.signal import Signal2D
from hplc.models.chromatogram import Chromatogram


def _skew_normal(time, amplitude, mu, sigma, alpha):
    """
    Skew-normal peak shape.
    I(t) = 2 * A * φ(t; μ, σ) * Φ(α(t - μ) / σ)

    Parameters:
        amplitude: peak height scaling factor (not the max height directly)
        mu: retention time (location)
        sigma: scale (width)
        alpha: skew parameter (0 = symmetric, >0 = right-tailing, <0 = fronting)
    """
    z = (time - mu) / sigma
    phi = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * z ** 2)
    Phi = 0.5 * (1 + erf(alpha * z / np.sqrt(2)))
    return 2 * amplitude * phi * Phi


def make_raw_signal(peaks, duration=900, dt=1, noise_scale=2, seed=42):
    """
    Returns (time, signal) arrays.

    peaks: list of dicts, each with:
        - mu: float, retention time (seconds)
        - amplitude: float, scaling factor
        - sigma: float, width (seconds)
        - alpha: float, skew (0 = symmetric, >0 = tailing, <0 = fronting)
    """
    rng = np.random.default_rng(seed)
    time = np.arange(0, duration, dt, dtype=float)
    signal = rng.normal(loc=0, scale=noise_scale, size=len(time))

    for p in peaks:
        signal += _skew_normal(time, p["amplitude"], p["mu"], p["sigma"], p["alpha"])

    return time, signal


def expected_peak_properties(peaks, n_sigma=5.0):
    """
    Computes expected properties from peak definitions by evaluating the
    skew-normal on a fine grid rather than using analytical formulas,
    since the skew-normal's max height and center shift with alpha.
    """
    results = []
    for p in peaks:
        # Fine grid around the peak for numerical evaluation
        t_start = p["mu"] - n_sigma * p["sigma"] * (1 + abs(p["alpha"]))
        t_end = p["mu"] + n_sigma * p["sigma"] * (1 + abs(p["alpha"]))
        time = np.linspace(t_start, t_end, 100000)
        curve = _skew_normal(time, p["amplitude"], p["mu"], p["sigma"], p["alpha"])

        max_idx = np.argmax(curve)
        max_height = curve[max_idx]
        center = time[max_idx]

        # Start/end where signal drops below 0.1% of max
        threshold = max_height * 0.001
        above = np.where(curve > threshold)[0]
        start = time[above[0]]
        end = time[above[-1]]

        # Numerical area
        area = np.trapezoid(curve, time)

        results.append({
            "mu": p["mu"],
            "retention_time": center,
            "max_height": max_height,
            "start_time": start,
            "end_time": end,
            "area": area,
            "sigma": p["sigma"],
            "alpha": p["alpha"],
        })
    return results


def make_signal2d(peaks, detector_name="UV-Detektor S 2600: 254 nm", signal_unit="mAU",
                  duration=900, dt=1, noise_scale=0.1, seed=42, filename=None):
    time, signal = make_raw_signal(peaks, duration=duration, dt=dt, noise_scale=noise_scale, seed=seed)
    return Signal2D(
        detector_name=detector_name,
        time=time,
        time_unit="s",
        signal=signal,
        signal_unit=signal_unit,
        filename=filename,
    )


def make_chromatogram(signal_definitions, filename="test_sample.dat"):
    """
    signal_definitions: dict mapping detector name to kwargs for make_signal2d.
    """
    signals = {}
    for name, kwargs in signal_definitions.items():
        kwargs.setdefault("detector_name", name)
        signals[name] = make_signal2d(**kwargs)
    return Chromatogram(signals=signals, filename=filename)
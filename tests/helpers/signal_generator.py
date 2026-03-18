import numpy as np
from HPLC.models.signal import Signal2D



def make_signal_with_peaks(peaks, detector_name="UV-Detektor S 2600: 254 nm", signal_unit="mAU", duration=900, dt=1, noise_scale=2, seed=42):
    """
    peaks: list of dicts, each with:
        - retention_time: float (seconds)
        - height: float (mAU)
        - half_life: float (seconds)
        - front_skew: float (1.0 = no skew, <1 = sharper front, >1 = stretched front)
        - tail_skew: float (1.0 = no skew, <1 = sharper tail, >1 = stretched tail)
    """
    rng = np.random.default_rng(seed)
    time = np.arange(0, duration, dt, dtype=float)
    signal = rng.normal(loc=0, scale=noise_scale, size=len(time))

    for p in peaks:
        sigma_base = p["half_life"] / np.sqrt(2 * np.log(2))
        sigma_front = sigma_base * p.get("front_skew", 1.0)
        sigma_tail = sigma_base * p.get("tail_skew", 1.0)

        left = np.exp(-0.5 * ((time - p["retention_time"]) / sigma_front) ** 2)
        right = np.exp(-0.5 * ((time - p["retention_time"]) / sigma_tail) ** 2)

        peak = p["max_height"] * np.where(time <= p["retention_time"], left, right)
        signal += peak

    return Signal2D(
        detector_name=detector_name,
        time=time,
        time_unit="s",
        signal=signal,
        signal_unit=signal_unit,
    )


def expected_peak_properties(peaks, n_sigma=3.0):
    """
    Given the same peak dicts passed to make_signal_with_peaks,
    returns a list of dicts with: retention_time, height, start, end, area.

    n_sigma: how many base sigmas out to define start/end (4.0 captures ~99.99%)
    """
    results = []
    for p in peaks:
        sigma_base = p["half_life"] / np.sqrt(2 * np.log(2))
        sigma_front = sigma_base * p.get("front_skew", 1.0)
        sigma_tail = sigma_base * p.get("tail_skew", 1.0)

        start = p["retention_time"] - n_sigma * sigma_front
        end = p["retention_time"] + n_sigma * sigma_tail

        # Area of a split Gaussian: half-Gaussian left + half-Gaussian right
        area = p["max_height"] * np.sqrt(2 * np.pi) * (sigma_front + sigma_tail) / 2

        results.append({
            "retention_time": p["retention_time"],
            "max_height": p["max_height"],
            "start": start,
            "end": end,
            "area": area,
            "sigma_front": sigma_front,
            "sigma_tail": sigma_tail,
        })
    return results
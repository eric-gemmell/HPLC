"""Constrained peak detection: the caller supplies the retention-time windows.

detect_peaks(time, signal, peak_retention_times) is handed one (start, end)
window per expected peak and returns one MathPeak2D per window, sorted by
retention time.
"""

import pytest

from hplc.analysis.peak_detection import detect_peaks
from hplc.models.peak import MathPeak2D
from hplc.testing.mock_generators import make_raw_signal, expected_peak_properties


PEAKS = [
    {"mu": 300, "amplitude": 100, "sigma": 12, "alpha": 0.0},
    {"mu": 250, "amplitude": 10, "sigma": 12, "alpha": 0.0},
    {"mu": 100, "amplitude": 30, "sigma": 3, "alpha": 0.0},
]

NOISE = 0.01
DURATION = 600
WINDOW = 20.0
TOLERANCE = 0.05

# One (start, end) window per peak, in the order the peaks were defined —
# deliberately not sorted, so the sorted-by-retention-time contract bites.
PEAK_RETENTION_TIMES = [(p["mu"] - WINDOW, p["mu"] + WINDOW) for p in PEAKS]

# Ground truth, sorted to match the order detect_peaks returns.
EXPECTED = sorted(
    expected_peak_properties(PEAKS, n_sigma=3.0),
    key=lambda p: p["retention_time"],
)


@pytest.fixture(scope="module")
def detected_peaks():
    time, signal = make_raw_signal(PEAKS, noise_scale=NOISE, duration=DURATION)
    return detect_peaks(time, signal, PEAK_RETENTION_TIMES)


def test_detect_peaks_finds_one_peak_per_requested_window(detected_peaks):
    assert len(detected_peaks) == len(PEAK_RETENTION_TIMES)
    assert all(isinstance(peak, MathPeak2D) for peak in detected_peaks)

    retention_times = [peak.retention_time for peak in detected_peaks]
    assert retention_times == sorted(retention_times)

    for peak, expected in zip(detected_peaks, EXPECTED):
        assert abs(peak.retention_time - expected["retention_time"]) / expected["retention_time"] < TOLERANCE
        assert abs(peak.max_height - expected["max_height"]) / expected["max_height"] < TOLERANCE
        assert abs(peak.area - expected["area"]) / expected["area"] < TOLERANCE

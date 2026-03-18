import pytest
import numpy as np
from unittest.mock import patch

from HPLC.models import Signal2D, Peak2D
from tests.helpers import make_signal_with_peaks, expected_peak_properties

@pytest.fixture
def sig():
    return Signal2D("UV", np.array([1.0, 2.0]), "min", np.array([10.0, 20.0]), "mAU")

@pytest.fixture
def baseline_signal():
    time = np.arange(0, 600, 1, dtype=float)  # 0 to 599 seconds, 1 point/sec
    noise = np.random.normal(loc=0, sigma=1, size=len(time))
    return Signal2D(
        detector_name="UV-Detektor S 2600: 254 nm",
        time=time,
        time_unit="s",
        signal=noise,
        signal_unit="mAU",
    )


@pytest.fixture
def make_peak_signal(peak_center=300.0, peak_height=100.0, half_life=10.0):
    time = np.arange(0, 600, 1, dtype=float)
    noise = np.random.normal(loc=0, sigma=1, size=len(time))

    # t1/2 = sigma * sqrt(2 * ln(2)), so sigma = t1/2 / sqrt(2 * ln(2))
    sigma = half_life / np.sqrt(2 * np.log(2))
    peak = peak_height * np.exp(-0.5 * ((time - peak_center) / sigma) ** 2)

    return Signal2D(
        detector_name="UV-Detektor S 2600: 254 nm",
        time=time,
        time_unit="s",
        signal=noise + peak,
        signal_unit="mAU",
    )


def test_construction(sig):
    assert sig.detector_name == "UV"
    assert len(sig) == 2
    np.testing.assert_array_equal(sig.signal, [10.0, 20.0])


@pytest.mark.parametrize("kwargs, match", [
    (dict(detector_name="", time=np.array([1.0]), time_unit="s", signal=np.array([1.0]), signal_unit="V"), "Detector name"),
    (dict(detector_name="UV", time=np.array([]), time_unit="s", signal=np.array([]), signal_unit="V"), "at least one"),
    (dict(detector_name="UV", time=np.array([1.0, 2.0]), time_unit="s", signal=np.array([1.0]), signal_unit="V"), "same length"),
])
def test_validation(kwargs, match):
    with pytest.raises(ValueError, match=match):
        Signal2D(**kwargs)


@patch("HPLC.models.signal.plot_signal_2d")
def test_display(mock_display, sig):
    assert sig.display() is sig
    mock_display.assert_called_once()

def test_signal_has_optional_peaks_attribute(sig):
    assert hasattr(sig, "peaks")
    assert sig.peaks == []




DIFFICULT_PEAKS = [
    {"retention_time": 800, "max_height": 80,  "half_life": 5, "front_skew": 20.0, "tail_skew": 1.0},
    {"retention_time": 200, "max_height": 100, "half_life": 10, "front_skew": 1.0, "tail_skew": 1.0},
    {"retention_time": 350, "max_height": 60,  "half_life": 10, "front_skew": 1.0, "tail_skew": 2.5},
    {"retention_time": 480, "max_height": 90,  "half_life": 10, "front_skew": 1.5, "tail_skew": 2.0},
]

EXPECTED = expected_peak_properties(DIFFICULT_PEAKS, n_sigma=3.0)

@pytest.fixture(scope="module")
def detected():
    signal = make_signal_with_peaks(DIFFICULT_PEAKS, duration=1200)
    return signal.detect_peaks()


# --- Basic detection ---

def test_finds_correct_number_of_peaks(detected):
    assert len(detected) == 4

def test_all_peaks_are_peak_objects(detected):
    for peak in detected:
        assert isinstance(peak, Peak2D)

def test_peaks_have_no_compound_by_default(detected):
    for peak in detected:
        assert peak.compound is None


# --- Per-peak properties ---

@pytest.mark.parametrize("i, label", [
    (0, "extreme fronting"),
    (1, "symmetric"),
    (2, "tailing"),
    (3, "both skewed"),
])
class TestPeakProperties:
    def _match(self, detected, i):
        """Find the detected peak closest to the expected retention time."""
        exp = EXPECTED[i]
        return min(detected, key=lambda p: abs(p.retention_time - exp["retention_time"]))

    def test_retention_time(self, detected, i, label):
        peak = self._match(detected, i)
        assert abs(peak.retention_time - EXPECTED[i]["retention_time"]) < 5.0, f"{label}: retention time off"

    def test_height(self, detected, i, label):
        peak = self._match(detected, i)
        assert abs(peak.max_height - EXPECTED[i]["max_height"]) / EXPECTED[i]["max_height"] < 0.15, f"{label}: height off"

    def test_area(self, detected, i, label):
        peak = self._match(detected, i)
        assert abs(peak.area - EXPECTED[i]["area"]) / EXPECTED[i]["area"] < 0.15, f"{label}: area off"

    def test_start_time(self, detected, i, label):
        peak = self._match(detected, i)
        assert peak.start_time < EXPECTED[i]["retention_time"], f"{label}: start should be before retention time"
        assert abs(peak.start_time - EXPECTED[i]["start"]) < 10.0, f"{label}: start off"

    def test_end_time(self, detected, i, label):
        peak = self._match(detected, i)
        assert peak.end_time > EXPECTED[i]["retention_time"], f"{label}: end should be after retention time"
        assert abs(peak.end_time - EXPECTED[i]["end"]) < 10.0, f"{label}: end off"

    def test_unit(self, detected, i, label):
        peak = self._match(detected, i)
        assert peak.unit == "mAU", f"{label}: wrong unit"


# --- Edge cases ---

def test_no_peaks_on_pure_noise():
    signal = make_signal_with_peaks([], duration=600, noise_scale=3.0)
    peaks = signal.detect_peaks()
    assert len(peaks) == 0

def test_single_small_peak_near_noise():
    signal = make_signal_with_peaks(
        [{"retention_time": 300, "max_height": 5, "half_life": 10, "front_skew": 1.0, "tail_skew": 1.0, "noise_scale": 1.0}],
        noise_scale=1.0,
    )
    peaks = signal.detect_peaks()
    # Either finds it or doesn't — but should not find more than 1
    assert len(peaks) <= 1

def test_detect_peaks_returns_signal2d():
    signal = make_signal_with_peaks([], duration=20)
    result = signal.detect_peaks()
    assert isinstance(result, Signal2D)

def test_detect_peaks_returns_copy():
    signal = make_signal_with_peaks([], duration=20)
    result = signal.detect_peaks()
    assert result is not signal
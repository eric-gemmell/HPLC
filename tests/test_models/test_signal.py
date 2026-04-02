from typing import Literal

from numpy._typing._array_like import NDArray
import pytest
import numpy as np
from unittest.mock import patch

from hplc.models import Signal2D, Peak2D
from hplc.testing.mock_generators import make_raw_signal, make_signal2d, expected_peak_properties
from hplc.analysis.peak_detection import detect_peaks


@pytest.fixture
def sig():
    return Signal2D(
        filename="test_signal", 
        detector_name="UV", 
        time=np.array([1.0, 2.0]), 
        time_unit="min", 
        signal=np.array([10.0, 20.0]), 
        signal_unit="mAU"
    )


def test_construction(sig: Signal2D):
    assert sig.filename == "test_signal"
    assert sig.detector_name == "UV"
    assert len(sig) == 2
    np.testing.assert_array_equal(sig.signal, [10.0, 20.0])


@pytest.mark.parametrize("kwargs, match", [
    (dict(filename="s", detector_name="", time=np.array([1.0]), time_unit="s", signal=np.array([1.0]), signal_unit="V"), "Detector name"),
    (dict(filename="s", detector_name="UV", time=np.array([]), time_unit="s", signal=np.array([]), signal_unit="V"), "at least one"),
    (dict(filename="s", detector_name="UV", time=np.array([1.0, 2.0]), time_unit="s", signal=np.array([1.0]), signal_unit="V"), "same length"),
    # (dict(filename="", detector_name="UV", time=np.array([1.0]), time_unit="s", signal=np.array([1.0]), signal_unit="V"), "Signal name"),
])
def test_validation(kwargs: dict[str, str | NDArray[Any]], match: Literal['Detector name'] | Literal['at least one'] | Literal['same length']):
    with pytest.raises(ValueError, match=match):
        Signal2D(**kwargs)


@patch("hplc.models.signal.plot_signal_2d")
def test_display(mock_display, sig: Signal2D):
    assert sig.display() is sig
    mock_display.assert_called_once()


@patch("hplc.models.signal.plot_signal_2d")
def test_display_accepts_optional_parameters(mock_display, sig: Signal2D):
    sig.display(width=800, height=600, filename="out.png", dpi=150, style="bw", title="My Signal", x_lim=(0, 10), y_lim=(0, 100))
    mock_display.assert_called_once()


@patch("hplc.models.signal.plot_signal_2d")
def test_display_publication(mock_display, sig: Signal2D):
    assert sig.display_publication() is sig
    mock_display.assert_called_once()


@patch("hplc.models.signal.plot_signal_2d")
def test_display_publication_accepts_optional_parameters(mock_display, sig: Signal2D):
    sig.display_publication(width=800, height=600, filename="out.png", dpi=150, title="My Signal", x_lim=(0, 10), y_lim=(0, 100))
    mock_display.assert_called_once()


def test_signal_has_optional_peaks_attribute(sig: Signal2D):
    assert hasattr(sig, "peaks")
    assert sig.peaks == []


def test_detect_peaks_returns_signal2d():
    signal = make_signal2d([], duration=20)
    result = signal.detect_peaks()
    assert isinstance(result, Signal2D)


def test_detect_peaks_returns_copy():
    signal = make_signal2d([], duration=20)
    result = signal.detect_peaks()
    assert result is not signal


def test_detect_peaks_calls_detect_peaks_once():
    time = np.array([1.0, 2.0, 3.0])
    signal = np.array([0.0, 1.0, 0.0])

    sig = Signal2D(
        filename="test",
        detector_name="UV",
        time=time,
        time_unit="min",
        signal=signal,
        signal_unit="mAU",
    )

    with patch("hplc.models.signal.detect_peaks", return_value=[]) as mock_fp:
        sig.detect_peaks()
        mock_fp.assert_called_once()

def test_detect_peaks_assigns_peaks_to_result():
    time = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    signal = np.array([0.0, 1.0, 3.0, 1.0, 0.0])

    sig = Signal2D(
        filename="test",
        detector_name="UV",
        time=time,
        time_unit="min",
        signal=signal,
        signal_unit="mAU",
    )

    stub_peak_data = [
        {
            "retention_time": 3.0,
            "max_height": 3.0,
            "area": 6.0,
            "start_time": 1.0,
            "end_time": 5.0,
            "mu": 3.0,
            "amplitude": 3.0,
            "sigma": 0.5,
            "alpha": 0.1,
        }
    ]

    with patch("hplc.models.signal.detect_peaks", return_value=stub_peak_data):
        result = sig.detect_peaks()

    assert len(result.peaks) == 1

    peak = result.peaks[0]
    assert isinstance(peak, Peak2D)
    assert peak.retention_time == 3.0
    assert peak.max_height == 3.0
    assert peak.area == 6.0
    assert peak.start_time == 1.0
    assert peak.end_time == 5.0
    assert peak.unit == "mAU"
    assert peak.mu == 3.0
    assert peak.amplitude == 3.0
    assert peak.sigma == 0.5
    assert peak.alpha == 0.1


def test_signal_has_stack_method():
    sig = make_signal2d([], filename="test_signal", detector_name="UV", duration=20)
    sig2 = make_signal2d([], filename="test_signal_2", detector_name="UV", duration=20)
    assert hasattr(sig, "stack")
    stack = sig.stack(sig2)
    assert stack.signals[0].filename is sig.filename
    assert stack.signals[1].filename is sig2.filename
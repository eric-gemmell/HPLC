import pytest
import numpy as np
from unittest.mock import patch, MagicMock

from HPLC.models.chromatogram import Chromatogram
from HPLC.models.signal import Signal2D

from tests.helpers import make_signal_with_peaks

@pytest.fixture
def make_signal():
    def _make(name="UV", signal_unit="mAU"):
        return Signal2D(name, np.array([1.0, 2.0, 3.0]), "min", np.array([10.0, 20.0, 30.0]), signal_unit)
    return _make


@pytest.fixture
def chrom(make_signal):
    return Chromatogram(
        signals={"UV": make_signal("UV"), "FLD": make_signal("FLD", "RFU")},
        filename="sample_001.txt",
    )


def test_construction(chrom):
    assert chrom.filename == "sample_001.txt"
    assert set(chrom.signals.keys()) == {"UV", "FLD"}
    assert chrom.additional_data == {}
    assert chrom.peaks == []


def test_empty_signals_raises():
    with pytest.raises(ValueError, match="at least one signal"):
        Chromatogram(signals={}, filename="empty.txt")


@patch("HPLC.models.chromatogram.plot_chromatogram")
def test_display_all(mock_plot, chrom):
    assert chrom.display() is chrom
    mock_plot.assert_called_once()


@patch("HPLC.models.chromatogram.plot_signal_2d")
def test_display_single_detector(mock_plot, chrom):
    assert chrom.display(detector_name="UV") is chrom
    mock_plot.assert_called_once()


@patch("HPLC.models.chromatogram.plot_chromatogram")
def test_display_multiple_detectors(mock_plot, chrom):
    assert chrom.display(detector_name=["UV", "FLD"]) is chrom
    mock_plot.assert_called_once()

@pytest.fixture
def complex_chrom_with_peaks():
    signal254: Signal2D = make_signal_with_peaks([], noise_scale=5, detector_name="UV-Detektor S 2600: 254 nm")
    signal200: Signal2D = make_signal_with_peaks([
        {"center": 200, "height": 50, "half_life": 20},
        {"center": 400, "height": 70, "half_life": 30},
    ], detector_name="UV-Detektor S 2600: 200 nm")

    lichtschtreu_signal: Signal2D = make_signal_with_peaks([
        {"center": 300, "height": 80, "half_life": 15},
        {"center": 700, "height": 60, "half_life": 25},
    ], detector_name="Lichtschtreudetektor")

    chrom = Chromatogram(
        signals={
            "UV-Detektor S 2600: 254 nm": signal254,
            "UV-Detektor S 2600: 200 nm": signal200,
            "Lichtschtreudetektor": lichtschtreu_signal,
        },
        filename="complex_sample.txt",
    )
    return chrom.identify_peaks()


@pytest.mark.parametrize("detector_name, expected_peaks", [
    ("UV-Detektor S 2600: 254 nm", 0),
    ("UV-Detektor S 2600: 200 nm", 2),
    ("Lichtschtreudetektor", 2),
])
def test_chromatogram_identify_peaks_identifies_peaks_in_all_signals(complex_chrom_with_peaks, detector_name, expected_peaks):
    chrom = complex_chrom_with_peaks
    signal = chrom.signals[detector_name]
    assert len(signal.peaks) == expected_peaks


def test_chromatogram_identify_peaks_returns_chromatogram(complex_chrom_with_peaks):
    assert isinstance(complex_chrom_with_peaks, Chromatogram)

def test_identify_peaks_returns_copy():
    signal = make_signal_with_peaks([
        {"center": 200, "height": 50, "half_life": 20},
    ], detector_name="UV-Detektor S 2600: 200 nm")
    chrom = Chromatogram(
        signals={"UV-Detektor S 2600: 200 nm": signal},
        filename="test.dat",
    )
    result = chrom.identify_peaks()
    assert result is not chrom
    assert result.signals["UV-Detektor S 2600: 200 nm"] is not signal
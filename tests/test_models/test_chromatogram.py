import pytest
import numpy as np
from unittest.mock import patch

from HPLC.models.chromatogram import Chromatogram
from HPLC.models.signal import Signal2D
from HPLC.testing.mock_generators import make_signal2d, make_chromatogram


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


def test_detect_peaks_returns_chromatogram(chrom):
    result = chrom.detect_peaks()
    assert isinstance(result, Chromatogram)


def test_detect_peaks_returns_copy(chrom):
    result = chrom.detect_peaks()
    assert result is not chrom
    for name in chrom.signals:
        assert result.signals[name] is not chrom.signals[name]


def test_detect_peaks_calls_signal_detect_peaks_once_per_signal(chrom):
    with patch.object(Signal2D, "detect_peaks", return_value=chrom.signals["UV"]) as mock:
        chrom.detect_peaks()
        assert mock.call_count == len(chrom.signals)
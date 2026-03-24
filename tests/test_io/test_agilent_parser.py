import pytest
import numpy as np

from HPLC.io import parse_agilent_data_file, is_agilent_chromatogram
from HPLC.models import Chromatogram, Signal2D

example_agilent_filename = "tests/example_chromatograms/example_agilent_chromatogram.D"
example_shimadzu_filename = "tests/example_chromatograms/example_shimadzu_chromatogram.lcd"
example_ez_chrom_filename = "tests/example_chromatograms/example_ez_chrom_chromatogram.dat"

EXPECTED_SIGNALS = {
    "200 nm",
    "254 nm",
    "405 nm"
}

@pytest.fixture(scope="module")
def agilent_chromatogram():
    return parse_agilent_data_file(example_agilent_filename)


def test_is_agilent_chromatogram_detects_agilent_filetype():
    assert is_agilent_chromatogram(example_agilent_filename) is True

def test_is_agilent_chromatogram_rejects_non_agilent_filetype():
    assert is_agilent_chromatogram(example_shimadzu_filename) is False
    assert is_agilent_chromatogram(example_ez_chrom_filename) is False


def test_parse_agilent_data_file_returns_chromatogram_object(agilent_chromatogram):
    assert isinstance(agilent_chromatogram, Chromatogram)

def test_parse_agilent_data_file_extract_relevant_chromatogram_metadata(agilent_chromatogram):
    extra_data = agilent_chromatogram.additional_data

    assert agilent_chromatogram.filename == "example_agilent_chromatogram.D"

    # assert extra_data["sample_name"] == "NaCl_0.2mM"

    # assert extra_data["method_path"] == 'C:\\EZChrom Elite\\Enterprise\\Projects\\Default\\Method\\Eric\\2026_03_13\\Isocratic_100B_15min_1ml.met'

    # assert extra_data["software_version"] == "Version 3.1.7"

    # assert extra_data["instrument"] == "Single_Pump_B_UV_&_ELSD"


def test_parse_agilent_data_file_extracts_relevant_signals(agilent_chromatogram):
    signals = agilent_chromatogram.signals
    for signal in signals.values():
        assert isinstance(signal, Signal2D)
    assert agilent_chromatogram.filename == "example_agilent_chromatogram.D"
    assert signals.keys() == EXPECTED_SIGNALS

@pytest.mark.parametrize("detector_name, expected_unit", [
    ("200 nm", "mAU"),
    ("254 nm", "mAU"),
    ("405 nm", "mAU"),
])
def test_parse_agilent_data_file_extracts_relevant_signal_metadata(agilent_chromatogram, detector_name, expected_unit):
    signals = agilent_chromatogram.signals

    sig = signals[detector_name]
    assert sig.detector_name == detector_name
    assert sig.time_unit == "s"
    assert sig.signal_unit == expected_unit

    assert "sampling_rate_hz" in sig.additional_data
    # assert sig.additional_data["sampling_rate_hz"] == 10

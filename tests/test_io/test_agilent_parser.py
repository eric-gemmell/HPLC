import pytest
import numpy as np

from hplc.io import parse_agilent_data_file, is_agilent_chromatogram
from hplc.models import Chromatogram, Signal2D

example_agilent_filename = "tests/example_chromatograms/example_agilent_chromatogram.D"
example_agilent_filename_modified_1 = "tests/example_chromatograms/example_agilent_chromatogram"
example_agilent_filename_modified_2 = "tests/example_chromatograms/example_agilent_chromatogram.D/"
example_shimadzu_filename = "tests/example_chromatograms/example_shimadzu_chromatogram.lcd"
example_ez_chrom_filename = "tests/example_chromatograms/example_ez_chrom_chromatogram.dat"

EXPECTED_SIGNALS = {
    "DAD 254.0 nm",
    "DAD 300.0 nm",
    "DAD 405.0 nm"
}

@pytest.fixture(scope="module")
def agilent_chromatogram():
    return parse_agilent_data_file(example_agilent_filename)

@pytest.fixture(scope="module")
def agilent_chromatogram_1():
    return parse_agilent_data_file(example_agilent_filename)

@pytest.fixture(scope="module")
def agilent_chromatogram_2():
    return parse_agilent_data_file(example_agilent_filename)


def test_is_agilent_chromatogram_detects_agilent_filetype():
    assert is_agilent_chromatogram(example_agilent_filename) is True
    assert is_agilent_chromatogram(example_agilent_filename_modified_1) is True
    assert is_agilent_chromatogram(example_agilent_filename_modified_2) is True

def test_is_agilent_chromatogram_rejects_non_agilent_filetype():
    assert is_agilent_chromatogram(example_shimadzu_filename) is False
    assert is_agilent_chromatogram(example_ez_chrom_filename) is False


def test_parse_agilent_data_file_returns_chromatogram_object(agilent_chromatogram, agilent_chromatogram_1, agilent_chromatogram_2):
    assert isinstance(agilent_chromatogram, Chromatogram)
    assert isinstance(agilent_chromatogram_1, Chromatogram)
    assert isinstance(agilent_chromatogram_2, Chromatogram)

def test_parse_agilent_data_file_extract_relevant_chromatogram_metadata(agilent_chromatogram):
    extra_data = agilent_chromatogram.additional_data

    assert agilent_chromatogram.filename == "example_agilent_chromatogram.D"

    assert extra_data["sample_name"] == "EGG_32_GlcNAc-6S"

    assert extra_data["method_path"] == "C:\\Chem32\\2\\Methods\\BiomatMethods\\DAD_30kV 25min_Eric.M"

    # assert extra_data["software_version"] == "Version 3.1.7"

    # assert extra_data["instrument"] == "Single_Pump_B_UV_&_ELSD"


def test_parse_agilent_data_file_extracts_relevant_signals(agilent_chromatogram):
    signals = agilent_chromatogram.signals
    for signal in signals.values():
        assert isinstance(signal, Signal2D)
    assert agilent_chromatogram.filename == "example_agilent_chromatogram.D"
    assert signals.keys() == EXPECTED_SIGNALS

@pytest.mark.parametrize("detector_name, expected_unit", [
    ("DAD 254.0 nm", "mAU"),
    ("DAD 300.0 nm", "mAU"),
    ("DAD 405.0 nm", "mAU"),
])
def test_parse_agilent_data_file_extracts_relevant_signal_metadata(agilent_chromatogram, detector_name, expected_unit):
    signals = agilent_chromatogram.signals

    sig = signals[detector_name]
    assert sig.detector_name == detector_name
    assert sig.time_unit == "s"
    assert sig.signal_unit == expected_unit

    assert "sampling_rate_hz" in sig.additional_data
    assert sig.additional_data["sampling_rate_hz"] == 2.5

def test_parse_agilent_data_file_gets_signals_in_the_right_range(agilent_chromatogram):
    signals = agilent_chromatogram.signals

    for sig in signals.values():
        assert np.all(sig.signal <= 400)
        assert np.all(sig.signal >= -100)
        assert len(sig.time) < 4.5*60*26  # less than 25 minutes at 4.5 Hz sampling rate

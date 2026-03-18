import pytest
import numpy as np

from HPLC.io import parse_ezchrom, is_ezchrom_file
from HPLC.models import Chromatogram, Signal2D

example_ezchrom_filename = "tests/example_chromatograms/example_ez_chrom_chromatogram.dat"
example_shimadzu_filename = "tests/example_chromatograms/example_shimadzu_chromatogram.lcd"

EXPECTED_SIGNALS = {
    "UV-Detektor S 2600: 200 nm",
    "UV-Detektor S 2600: 210 nm",
    "UV-Detektor S 2600: 220 nm",
    "UV-Detektor S 2600: 254 nm",
    "Lichtstreudetektor",
}

@pytest.fixture(scope="module")
def ezchrom_chromatogram():
    return parse_ezchrom(example_ezchrom_filename)


def test_is_ezchrom_file_detects_ezchrom_filetype():
    assert is_ezchrom_file(example_ezchrom_filename) is True

def test_is_ezchrom_file_rejects_non_ezchrom_filetype():
    assert is_ezchrom_file(example_shimadzu_filename) is False


def test_parse_ezchrom_returns_chromatogram_object(ezchrom_chromatogram):
    assert isinstance(ezchrom_chromatogram, Chromatogram)

def test_parse_ezchrom_extract_relevant_chromatogram_metadata(ezchrom_chromatogram):
    extra_data = ezchrom_chromatogram.additional_data

    assert ezchrom_chromatogram.filename == "example_ez_chrom_chromatogram.dat"

    assert extra_data["sample_name"] == "NaCl_0.2mM"

    assert extra_data["method_path"] == 'C:\\EZChrom Elite\\Enterprise\\Projects\\Default\\Method\\Eric\\2026_03_13\\Isocratic_100B_15min_1ml.met'

    assert extra_data["software_version"] == "Version 3.1.7"

    assert extra_data["instrument"] == "Single_Pump_B_UV_&_ELSD"


def test_parse_ezchrom_extracts_relevant_signals(ezchrom_chromatogram):
    signals = ezchrom_chromatogram.signals
    for signal in signals.values():
        assert isinstance(signal, Signal2D)
    assert ezchrom_chromatogram.filename == "example_ez_chrom_chromatogram.dat"
    assert signals.keys() == EXPECTED_SIGNALS

@pytest.mark.parametrize("detector_name, expected_unit", [
    ("UV-Detektor S 2600: 200 nm", "mAU"),
    ("UV-Detektor S 2600: 210 nm", "mAU"),
    ("UV-Detektor S 2600: 220 nm", "mAU"),
    ("UV-Detektor S 2600: 254 nm", "mAU"),
    ("Lichtstreudetektor", "mV")
])
def test_parse_ezchrom_extracts_relevant_signal_metadata(ezchrom_chromatogram, detector_name, expected_unit):
    signals = ezchrom_chromatogram.signals
    assert signals.keys() == EXPECTED_SIGNALS

    sig = signals[detector_name]
    assert sig.detector_name == detector_name
    assert sig.time_unit == "s"
    assert sig.signal_unit == expected_unit

    assert "sampling_rate_hz" in sig.additional_data
    assert sig.additional_data["sampling_rate_hz"] == 10

import pytest
import numpy as np

from hplc.io import parse_shimadzu, is_shimadzu_file
from hplc.models import Chromatogram, Signal2D, Signal3D

# example_shimadzu_filename = "tests/example_chromatograms/example_shimadzu_chromatogram_3d.lcd"
example_shimadzu_filename = "tests/example_chromatograms/example_shimadzu_chromatogram_3d.lcd"
example_ezchrom_filename = "tests/example_chromatograms/example_ez_chrom_chromatogram.dat"

EXPECTED_SIGNALS = {
    "PDA",
    "RID",
}

@pytest.fixture(scope="module")
def shimadzu_chromatogram():
    return parse_shimadzu(example_shimadzu_filename)


def test_is_shimadzu_file_detects_shimadzu_filetype():
    assert is_shimadzu_file(example_shimadzu_filename) is True

def test_is_shimadzu_file_rejects_non_shimadzu_filetype():
    assert is_shimadzu_file(example_ezchrom_filename) is False


def test_parse_shimadzu_returns_chromatogram_object(shimadzu_chromatogram):
    assert isinstance(shimadzu_chromatogram, Chromatogram)

def test_parse_shimadzu_extract_relevant_chromatogram_metadata(shimadzu_chromatogram):
    extra_data = shimadzu_chromatogram.additional_data

    assert shimadzu_chromatogram.filename == "example_shimadzu_chromatogram_3d.lcd"

    assert extra_data["wavelengths"] == 786

    assert extra_data["start_wavelength"] == 189.91

    assert extra_data["end_wavelength"] == 600.42

    assert extra_data["wavelength_step"] == 0.52


def test_parse_shimadzu_extracts_relevant_signals(shimadzu_chromatogram):
    signals = shimadzu_chromatogram.signals
    for signal in signals.values():
        assert isinstance(signal, Signal2D) or isinstance(signal, Signal3D)
    assert shimadzu_chromatogram.filename == "example_shimadzu_chromatogram_3d.lcd"
    assert signals.keys() == EXPECTED_SIGNALS

@pytest.mark.parametrize("detector_name, expected_unit", [
    ("PDA", "mAU"),
    ("RID", "mV"),
])
def test_parse_shimadzu_extracts_relevant_signal_metadata(shimadzu_chromatogram, detector_name, expected_unit):
    signals = shimadzu_chromatogram.signals
    assert signals.keys() == EXPECTED_SIGNALS

    sig = signals[detector_name]
    assert sig.detector_name == detector_name
    assert sig.time_unit == "s"
    assert sig.signal_unit == expected_unit


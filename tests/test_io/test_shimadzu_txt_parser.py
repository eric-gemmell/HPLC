import pytest
import numpy as np

import hplc
from hplc.io import parse_shimadzu_txt, is_shimadzu_txt_file, is_shimadzu_file
from hplc.models import Chromatogram, Signal2D, Signal3D

example_shimadzu_txt_filename = "tests/example_chromatograms/example_shimadzu_chromatogram_3d.txt"
example_shimadzu_lcd_filename = "tests/example_chromatograms/example_shimadzu_chromatogram_3d.lcd"
example_ezchrom_filename = "tests/example_chromatograms/example_ez_chrom_chromatogram.dat"

EXPECTED_SIGNALS = {
    "PDA",
    "AD2",
}

@pytest.fixture(scope="module")
def shimadzu_txt_chromatogram():
    return parse_shimadzu_txt(example_shimadzu_txt_filename)


def test_is_shimadzu_txt_file_detects_shimadzu_txt_filetype():
    assert is_shimadzu_txt_file(example_shimadzu_txt_filename) is True

@pytest.mark.parametrize("filename", [
    example_shimadzu_lcd_filename,
    example_ezchrom_filename,
])
def test_is_shimadzu_txt_file_rejects_non_shimadzu_txt_filetype(filename):
    assert is_shimadzu_txt_file(filename) is False

def test_is_shimadzu_file_rejects_shimadzu_txt_filetype():
    assert is_shimadzu_file(example_shimadzu_txt_filename) is False

def test_load_dispatches_shimadzu_txt_file():
    assert isinstance(hplc.load(example_shimadzu_txt_filename), Chromatogram)


def test_parse_shimadzu_txt_returns_chromatogram_object(shimadzu_txt_chromatogram):
    assert isinstance(shimadzu_txt_chromatogram, Chromatogram)

def test_parse_shimadzu_txt_extract_relevant_chromatogram_metadata(shimadzu_txt_chromatogram):
    extra_data = shimadzu_txt_chromatogram.additional_data

    assert shimadzu_txt_chromatogram.filename == "example_shimadzu_chromatogram_3d.txt"

    assert extra_data["wavelengths"] == 329

    assert extra_data["start_wavelength"] == 188.96

    assert extra_data["end_wavelength"] == 600.50

    assert extra_data["wavelength_step"] == 1.26


def test_parse_shimadzu_txt_extracts_relevant_signals(shimadzu_txt_chromatogram):
    signals = shimadzu_txt_chromatogram.signals
    for signal in signals.values():
        assert isinstance(signal, Signal2D) or isinstance(signal, Signal3D)
    assert shimadzu_txt_chromatogram.filename == "example_shimadzu_chromatogram_3d.txt"
    assert signals.keys() == EXPECTED_SIGNALS

@pytest.mark.parametrize("detector_name, expected_unit", [
    ("PDA", "mAU"),
    ("AD2", "mV"),
])
def test_parse_shimadzu_txt_extracts_relevant_signal_metadata(shimadzu_txt_chromatogram, detector_name, expected_unit):
    signals = shimadzu_txt_chromatogram.signals
    assert signals.keys() == EXPECTED_SIGNALS

    sig = signals[detector_name]
    assert sig.detector_name == detector_name
    assert sig.time_unit == "s"
    assert sig.signal_unit == expected_unit


def test_parse_shimadzu_txt_pda_is_3d_with_expected_shape(shimadzu_txt_chromatogram):
    pda = shimadzu_txt_chromatogram.signals["PDA"]
    assert isinstance(pda, Signal3D)
    assert pda.signal.shape == (12001, 329)
    assert len(pda.time) == 12001
    assert len(pda.wavelengths) == 329

@pytest.mark.parametrize("detector_name, expected_points", [
    ("PDA", 12001),
    ("AD2", 3001),
])
def test_parse_shimadzu_txt_time_axis_in_seconds(shimadzu_txt_chromatogram, detector_name, expected_points):
    # Run is 5 min; the file stores minutes with decimal commas.
    sig = shimadzu_txt_chromatogram.signals[detector_name]
    assert len(sig.time) == expected_points
    assert sig.time[0] == pytest.approx(0.0)
    assert sig.time[-1] == pytest.approx(300.0, abs=0.1)

def test_parse_shimadzu_txt_pda_matches_exported_254nm_chromatogram(shimadzu_txt_chromatogram):
    # The file also exports a 254 nm (bandwidth 4 nm) PDA chromatogram in uAU;
    # averaging the cube over that band should reproduce it in mAU.
    pda = shimadzu_txt_chromatogram.signals["PDA"]
    band = np.abs(pda.wavelengths - 254) <= 2
    extracted = pda.signal[:, band].mean(axis=1)

    lines = open(example_shimadzu_txt_filename, encoding="latin-1").read().splitlines()
    start = lines.index("[PDA Multi Chromatogram(Ch1)]") + 11
    exported = np.array([float(l.split("\t")[1].replace(",", ".")) for l in lines[start:start + 12001]]) / 1000

    assert np.max(np.abs(extracted - exported)) < 0.05 * np.max(np.abs(exported))

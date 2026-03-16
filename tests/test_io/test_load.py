import pytest
from pathlib import Path

from hplc.core.chromatogram import Chromatogram
from hplc.core.signal import Signal2D
from hplc.io import load


EXAMPLE_FILE = (
    Path(__file__).parent.parent
    / "example_files"
    / "EGG_HILIC_NaCl_0.2mM_20ul_80%MeCN_pH3_20mM_NH4OHC13.03.2026 19-13-41.dat"
)

DATA_DIR = Path(__file__).parent.parent.parent / "examples" / "data"


def get_dat_files():
    """Collect all .dat files from the examples directory."""
    files = list(DATA_DIR.glob("*.dat"))
    assert len(files) > 0, "No .dat files found in examples/data"
    return files


class TestParseEzchromLoader:
    """Verify that load() correctly parses an EZChrom .dat file."""

    def test_returns_chromatogram(self):
        chrom = load(str(EXAMPLE_FILE))
        assert isinstance(chrom, Chromatogram)

    def test_sample_name(self):
        chrom = load(str(EXAMPLE_FILE))
        assert chrom.metadata["sample_name"] == "NaCl_0.2mM"

    def test_signals_are_dict_of_signal2d(self):
        chrom = load(str(EXAMPLE_FILE))
        assert isinstance(chrom.signals, dict)
        assert len(chrom.signals) == 5
        for name, sig in chrom.signals.items():
            assert isinstance(name, str)
            assert isinstance(sig, Signal2D)
            assert len(sig.time) == len(sig.signal)
            assert len(sig.time) > 0

    def test_metadata_has_expected_keys(self):
        chrom = load(str(EXAMPLE_FILE))
        for key in ("filepath", "sample_name", "method_path", "software_version",
                     "instrument", "timestamp_serial", "wavelengths"):
            assert key in chrom.metadata


class TestLoadAllExampleFiles:
    """Smoke test: every .dat file in examples/data should load without error."""

    @pytest.mark.parametrize("dat_file", get_dat_files(), ids=lambda f: f.name[:40])
    def test_load_returns_chromatogram(self, dat_file):
        chrom = load(str(dat_file))
        assert isinstance(chrom, Chromatogram)
        assert len(chrom.signals) > 0


class TestLoadEdgeCases:
    def test_nonexistent_file_raises(self):
        with pytest.raises((FileNotFoundError, OSError)):
            load("nonexistent_file.dat")

    def test_unsupported_extension_raises(self):
        with pytest.raises(ValueError, match="Unsupported file type"):
            load("somefile.xyz")

    def test_accepts_string_path(self):
        chrom = load(str(EXAMPLE_FILE))
        assert isinstance(chrom, Chromatogram)

    def test_accepts_path_object(self):
        chrom = load(EXAMPLE_FILE)
        assert isinstance(chrom, Chromatogram)

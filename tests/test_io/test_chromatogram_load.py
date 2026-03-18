import pytest
from HPLC.io import load
from HPLC.models import Chromatogram

example_ezchrom_filename = "tests/example_chromatograms/example_ez_chrom_chromatogram.dat"
example_shimadzu_filename = "tests/example_chromatograms/example_shimadzu_chromatogram.lcd"

def test_load_ezchrom_returns_chromatogram():
    chrom = load(example_ezchrom_filename)
    assert isinstance(chrom, Chromatogram)

# def test_load_shimadzu_returns_chromatogram():
#     chrom = load(example_shimadzu_filename)
#     assert isinstance(chrom, Chromatogram)

def test_load_unknown_format_raises():
    with pytest.raises(ValueError):
        load("tests/example_chromatograms/not_a_real_file.xyz")
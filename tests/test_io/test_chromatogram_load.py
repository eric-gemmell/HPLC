import pytest
from HPLC.io import load
from HPLC.models import Chromatogram

example_ezchrom_filename = "tests/example_chromatograms/example_ez_chrom_chromatogram.dat"
example_agilent_filename = "tests/example_chromatograms/example_agilent_chromatogram.D"
example_shimadzu_filename = "tests/example_chromatograms/example_shimadzu_chromatogram.lcd"


@pytest.fixture(scope="module")
def ezchrom_chrom():
    return load(example_ezchrom_filename)

@pytest.fixture(scope="module")
def agilent_chrom():
    return load(example_agilent_filename)

@pytest.fixture(params=[example_ezchrom_filename, example_agilent_filename])                                                                                                                                                                
def chrom(request):                                                                                                                                                                                                                         
    return load(request.param)   

def test_load_ezchrom_returns_chromatogram(ezchrom_chrom):
    assert isinstance(ezchrom_chrom, Chromatogram)

def test_load_agilent_chromatogram_returns_chromatogram(agilent_chrom):
    assert isinstance(agilent_chrom, Chromatogram)

# def test_load_shimadzu_returns_chromatogram():
#     chrom = load(example_shimadzu_filename)
#     assert isinstance(chrom, Chromatogram)

def test_load_unknown_format_raises():
    with pytest.raises(ValueError):
        load("tests/example_chromatograms/not_a_real_file.xyz")

def test_load_chromatogram_signals_have_the_same_filename_as_chromatogram(chrom):
    filename = chrom.filename
    for name in chrom.signals.keys():
        assert chrom.signals[name].filename == filename


def test_load_chromatogram_signals_have_sample_name_in_additional_info(chrom):
    sample_name = chrom.additional_data["sample_name"]
    for name in chrom.signals.keys():
        assert chrom.signals[name].additional_data["sample_name"] == sample_name

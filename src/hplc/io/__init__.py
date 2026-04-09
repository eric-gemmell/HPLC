from hplc.io.loader import load
from hplc.io.parsers import parse_ezchrom, is_ezchrom_file
from hplc.io.agilent_parser import parse_agilent_data_file, is_agilent_chromatogram

# .D format — aliases for the existing agilent parser
parse_agilent_d_data_file = parse_agilent_data_file
is_agilent_d_chromatogram = is_agilent_chromatogram

# .aia format
from hplc.io.agilent_aia_parser import parse_agilent_aia_data_file, is_agilent_aia_chromatogram

__all__ = [
    "load",
    "parse_ezchrom",
    "is_ezchrom_file",
    "parse_agilent_data_file",
    "is_agilent_chromatogram",
    "parse_agilent_d_data_file",
    "is_agilent_d_chromatogram",
    "parse_agilent_aia_data_file",
    "is_agilent_aia_chromatogram",
]

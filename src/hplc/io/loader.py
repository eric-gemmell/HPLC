from __future__ import annotations

from hplc.models.chromatogram import Chromatogram
from hplc.io.parsers import parse_ezchrom, is_ezchrom_file
from hplc.io.agilent_parser import parse_agilent_data_file, is_agilent_chromatogram
from hplc.io.agilent_aia_parser import parse_agilent_aia_data_file, is_agilent_aia_chromatogram

_PARSERS = [
    (is_agilent_chromatogram, parse_agilent_data_file),
    (is_agilent_aia_chromatogram, parse_agilent_aia_data_file),
    (is_ezchrom_file, parse_ezchrom),
]


def load(filepath: str) -> Chromatogram:
    """Load a chromatogram file. Iteratively tries each registered parser."""
    for can_parse, parser in _PARSERS:
        if can_parse(filepath):
            return parser(filepath)
    raise ValueError(f"No parser found for file: {filepath}")

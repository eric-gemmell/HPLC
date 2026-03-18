from __future__ import annotations

from HPLC.models.chromatogram import Chromatogram
from HPLC.io.parsers import parse_ezchrom, is_ezchrom_file

_PARSERS = [
    (is_ezchrom_file, parse_ezchrom),
]


def load(filepath: str) -> Chromatogram:
    """Load a chromatogram file. Iteratively tries each registered parser."""
    for can_parse, parser in _PARSERS:
        if can_parse(filepath):
            return parser(filepath)
    raise ValueError(f"No parser found for file: {filepath}")

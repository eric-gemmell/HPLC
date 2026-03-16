from __future__ import annotations

from pathlib import Path

from hplc.core.chromatogram import Chromatogram
from hplc.io.parsers import parse_ezchrom

_PARSERS = {
    ".dat": parse_ezchrom,
}


def load(filepath: str) -> Chromatogram:
    """Load a chromatogram file. Dispatches to the correct parser by extension."""
    ext = Path(filepath).suffix.lower()
    parser = _PARSERS.get(ext)
    if parser is None:
        supported = ", ".join(sorted(_PARSERS))
        raise ValueError(f"Unsupported file type '{ext}'. Supported: {supported}")
    return parser(filepath)

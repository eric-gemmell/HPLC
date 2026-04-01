"""
Agilent ChemStation .D Folder Parser
=====================================
Parses Agilent .D directories containing version-130 .ch binary files
(DAD / diode-array detector signals).

Returns a Chromatogram with typed Signal2D objects whose keys are
wavelength labels like "DAD 254.0 nm".

Binary layout (version 130, "LC DATA FILE"):
    0x11A       Start retention time (int32 BE, units of 1/60000 min)
    0x11E       End retention time   (int32 BE, units of 1/60000 min)
    0x127C      Y-axis scaling factor (float64 BE)
    0x104C      Y-axis unit string (pascal UTF-16, gap=2)
    0x1075      Signal description (pascal UTF-16, gap=2)
    0x1800      Data start — segment-framed delta-encoded signal

Data encoding:
    Data is stored in segments.  Each segment starts with a 0x10 header
    byte followed by a count byte.  Within a segment, each point is either
    a 16-bit BE signed delta added to a running accumulator, or the sentinel
    -0x8000 followed by a 32-bit BE signed int that replaces the accumulator.
    Final signal = raw_accumulated_value * scaling_factor.
"""

import re
import struct
from pathlib import Path

import numpy as np

from HPLC.models.chromatogram import Chromatogram
from HPLC.models.signal import Signal2D

# ---------------------------------------------------------------------------
# Constants for version-130 .ch binary layout
# ---------------------------------------------------------------------------
_TIME_START_OFFSET = 0x11A
_TIME_END_OFFSET = 0x11E
_SCALING_FACTOR_OFFSET = 0x127C
_DATA_START_OFFSET = 0x1800
_TIME_UNIT = 60_000.0  # ticks per minute
_UNIT_OFFSET = 0x104C
_SIGNAL_DESC_OFFSET = 0x1075


# ---------------------------------------------------------------------------
# Low-level binary helpers
# ---------------------------------------------------------------------------

def _read_pascal_string(data: bytes, offset: int, gap: int = 2) -> str:
    """Read a pascal-style string (length byte + chars with stride *gap*)."""
    if offset >= len(data):
        return ""
    char_count = data[offset]
    raw = data[offset + 1: offset + 1 + char_count * gap]
    try:
        return raw[::gap].decode("utf-8", errors="replace").strip()
    except Exception:
        return ""


def _decode_segments(data: bytes, start: int = _DATA_START_OFFSET) -> np.ndarray:
    """Decode segment-framed, delta-encoded signal data.

    Each segment begins with 0x10 followed by a count byte.  A signed
    int16 of -0x8000 is a sentinel: the next 4 bytes are an absolute
    int32 replacement for the accumulator.
    """
    values: list[int] = []
    offset = start
    accumulator = 0
    length = len(data)

    while offset < length:
        if data[offset] != 0x10:
            break
        if offset + 1 >= length:
            break
        n_points = data[offset + 1]
        offset += 2

        for _ in range(n_points):
            if offset + 2 > length:
                break
            delta = struct.unpack(">h", data[offset: offset + 2])[0]
            if delta == -0x8000:
                if offset + 6 > length:
                    break
                accumulator = struct.unpack(">i", data[offset + 2: offset + 6])[0]
                offset += 6
            else:
                accumulator += delta
                offset += 2
            values.append(accumulator)

    return np.array(values, dtype=np.int64)


# ---------------------------------------------------------------------------
# Wavelength extraction from acq.macaml
# ---------------------------------------------------------------------------

def _parse_acq_macaml(d_folder: Path) -> dict[str, float]:
    """Return mapping of channel key (e.g. 'DAD1A') to wavelength in nm."""
    acq_file = d_folder / "acq.macaml"
    if not acq_file.exists():
        return {}
    text = acq_file.read_text(encoding="utf-8", errors="replace")

    wavelengths: dict[str, float] = {}
    rows = re.findall(r"<Row>(.*?)</Row>", text, re.DOTALL)
    for row in rows:
        sig_m = re.search(
            r"Signals_Signal_ID.*?<Value>(Signal \w+)</Value>", row, re.DOTALL
        )
        wl_m = re.search(
            r"Signals_Signal_Wavelength.*?<Value>([\d.]+)</Value>", row, re.DOTALL
        )
        use_m = re.search(
            r"Signals_Signal_UseSignal.*?<Value>(\w+)</Value>", row, re.DOTALL
        )
        if sig_m and wl_m:
            enabled = use_m.group(1).lower() == "yes" if use_m else False
            if enabled:
                letter = sig_m.group(1).split()[-1]
                key = f"DAD1{letter}"
                wavelengths[key] = float(wl_m.group(1))
    return wavelengths


# ---------------------------------------------------------------------------
# SAMPLE.XML metadata
# ---------------------------------------------------------------------------

def _parse_sample_xml(d_folder: Path) -> dict:
    """Extract metadata from SAMPLE.XML (UTF-16 encoded)."""
    sample_xml = d_folder / "SAMPLE.XML"
    if not sample_xml.exists():
        return {}
    raw = sample_xml.read_bytes()
    try:
        text = raw.decode("utf-16")
    except Exception:
        text = raw.decode("utf-8", errors="replace")

    meta: dict[str, str] = {}
    tag_map = {
        "Name": "sample_name",
        "ACQMethodPath": "method_path",
    }
    for xml_tag, key in tag_map.items():
        m = re.search(rf"<{xml_tag}>(.*?)</{xml_tag}>", text, re.DOTALL)
        if m:
            meta[key] = m.group(1).strip()
    return meta


# ---------------------------------------------------------------------------
# .ch file parser
# ---------------------------------------------------------------------------

def _parse_ch_file(filepath: Path) -> dict:
    """Parse a single version-130 .ch file into a dict of raw data."""
    data = filepath.read_bytes()

    if len(data) < _DATA_START_OFFSET:
        raise ValueError(f"{filepath.name}: file too small ({len(data)} bytes)")

    version = data[1:4].decode("ascii", errors="replace").strip("\x00")
    if version not in ("130", "30"):
        raise ValueError(
            f"{filepath.name}: expected version 130, got '{version}'"
        )

    start_raw, end_raw = struct.unpack(
        ">ii", data[_TIME_START_OFFSET: _TIME_START_OFFSET + 8]
    )
    start_min = start_raw / _TIME_UNIT
    end_min = end_raw / _TIME_UNIT

    scaling_factor = struct.unpack(
        ">d", data[_SCALING_FACTOR_OFFSET: _SCALING_FACTOR_OFFSET + 8]
    )[0]

    signal_desc = _read_pascal_string(data, _SIGNAL_DESC_OFFSET)
    unit = _read_pascal_string(data, _UNIT_OFFSET) or "mAU"

    raw_signal = _decode_segments(data)
    signal = raw_signal.astype(np.float64) * scaling_factor
    n_points = len(signal)

    duration_s = (end_min - start_min) * 60.0
    sampling_rate_hz = round(n_points / duration_s, 1) if duration_s > 0 else 0.0

    time = np.linspace(start_min * 60.0, end_min * 60.0, n_points)

    return {
        "channel_key": filepath.stem.upper(),
        "signal_description": signal_desc,
        "unit": unit,
        "time": time,
        "signal": signal,
        "sampling_rate_hz": sampling_rate_hz,
        "time_unit": "s",
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _resolve_d_folder(filepath: str) -> Path | None:
    """Resolve a filepath to an Agilent .D directory.

    Accepts paths with or without the .D extension, and with or without
    a trailing slash.
    """
    p = Path(filepath)
    if p.is_dir():
        ch_files = list(p.glob("*.ch")) + list(p.glob("*.CH"))
        if ch_files:
            return p
    # Try appending .D if the path doesn't already end with it
    if p.suffix.upper() != ".D":
        p_with_ext = Path(str(p) + ".D")
        if p_with_ext.is_dir():
            ch_files = list(p_with_ext.glob("*.ch")) + list(p_with_ext.glob("*.CH"))
            if ch_files:
                return p_with_ext
    return None


def is_agilent_chromatogram(filepath: str) -> bool:
    """Check whether *filepath* is an Agilent .D directory with .ch files."""
    return _resolve_d_folder(filepath) is not None


def parse_agilent_data_file(filepath: str) -> Chromatogram:
    """Parse an Agilent .D folder into a Chromatogram.

    Only DAD channels are included as signals (keyed by wavelength,
    e.g. "DAD 254.0 nm").  CE instrument traces are skipped.
    """
    d_folder = _resolve_d_folder(filepath)
    if d_folder is None:
        raise ValueError(f"Not a valid Agilent .D directory: {filepath}")
    wavelengths = _parse_acq_macaml(d_folder)
    sample_meta = _parse_sample_xml(d_folder)

    ch_files = sorted(d_folder.glob("*.ch")) + sorted(d_folder.glob("*.CH"))

    chrom_filename = d_folder.name
    sample_name = sample_meta.get("sample_name")

    signals: dict[str, Signal2D] = {}
    for chf in ch_files:
        key = chf.stem.upper()
        if not key.startswith("DAD"):
            continue

        parsed = _parse_ch_file(chf)

        wl = wavelengths.get(key)
        if wl is not None:
            detector_name = f"DAD {wl} nm"
        else:
            detector_name = parsed["signal_description"] or key

        signals[detector_name] = Signal2D(
            time=parsed["time"],
            signal=parsed["signal"],
            detector_name=detector_name,
            time_unit=parsed["time_unit"],
            signal_unit=parsed["unit"],
            filename=chrom_filename,
            additional_data={
                "channel_key": key,
                "signal_description": parsed["signal_description"],
                "sampling_rate_hz": parsed["sampling_rate_hz"],
                "sample_name": sample_name,
            },
        )

    additional_data = {}
    additional_data.update(sample_meta)

    return Chromatogram(
        signals=signals,
        filename=chrom_filename,
        additional_data=additional_data,
    )

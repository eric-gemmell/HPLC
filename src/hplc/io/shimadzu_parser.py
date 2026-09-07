"""
Shimadzu LabSolutions .lcd File Parser
======================================
Parses .lcd files from Shimadzu LabSolutions into a Chromatogram carrying a
Signal3D for the PDA cube and a Signal2D for each 2D detector trace.

Requirements: pip install olefile numpy

File format notes:
    - OLE2 Compound Document container (magic: d0 cf 11 e0 a1 b1 1a e1)
    - "LSS Raw Data/Chromatogram Ch<N>" holds one 2D detector trace
    - "PDA 3D Raw Data/3D Raw Data" holds one block per time point, each block
      a full spectrum across the wavelength axis
    - "PDA 3D Raw Data/Wavelength Table" is int32 count + int32 values in 0.01 nm
    - "PDA 3D Raw Data/Max Plot" carries the PDA time axis (interval, n_rows)

    Both 2D and 3D streams share one block encoding:
      * 24-byte block header; int16 at offset 8 is the number of values
      * the payload is split into chunks, each framed by a 2-byte length before
        and after it; the delta accumulator resets at every chunk boundary
      * values are delta-encoded. The high nibble of a value's first byte is a
        sign digit: it carries `hex1 // 2` extra bytes, and an odd sign digit
        means the delta is negative, stored as a two's complement
      * 0x82 is a padding byte and emits no value; 0x00 is a zero delta

Delta encoding credit: the sign-digit scheme was documented by the
chromConverter project (https://github.com/ethanbass/chromConverter).
"""

from __future__ import annotations

import io
import struct

import numpy as np
import olefile

from hplc.models.chromatogram import Chromatogram
from hplc.models.signal import Signal2D, Signal3D

_BLOCK_HEADER_BYTES = 24
_PAD_BYTE = 0x82

# Raw counts are the smallest unit of each detector's unit ladder (uAU, uV);
# scaling by 1/1000 puts them in the units we report.
_COUNTS_PER_UNIT = 1000.0

_DETECTOR_UNITS = {"PDA": "mAU", "RID": "mV"}


def is_shimadzu_file(filepath: str) -> bool:
    """Check whether a file is a Shimadzu LabSolutions .lcd OLE2 file."""
    try:
        if not olefile.isOleFile(filepath):
            return False
        ole = olefile.OleFileIO(filepath)
        has_streams = ole.exists("LSS Raw Data") or ole.exists("PDA 3D Raw Data")
        ole.close()
        return has_streams
    except Exception:
        return False


def _decode_value(chunk: bytes) -> int:
    """Decode one delta. The high nibble of the first byte is the sign digit."""
    value_bits = 8 * len(chunk) - 4
    x = int.from_bytes(chunk, "big")
    sign = (x >> value_bits) & 0xF
    value = x & ((1 << value_bits) - 1)
    if sign % 2:
        return -((1 << value_bits) - value)
    return value


def _decode_block(stream: io.BytesIO) -> np.ndarray | None:
    """Decode one block of delta-encoded values, advancing the stream past it."""
    header = stream.read(_BLOCK_HEADER_BYTES)
    if len(header) < _BLOCK_HEADER_BYTES:
        return None
    n_values = struct.unpack_from("<h", header, 8)[0]
    if n_values <= 0:
        return None

    signal = np.empty(n_values, dtype=float)
    count = 0
    accumulator = 0

    while count < n_values:
        raw_len = stream.read(2)
        if len(raw_len) < 2:
            break
        chunk = stream.read(struct.unpack("<h", raw_len)[0])

        pos = 0
        while pos < len(chunk) and count < n_values:
            byte = chunk[pos]
            if byte == _PAD_BYTE:
                pos += 1
                continue
            if byte == 0x00:
                delta = 0
                pos += 1
            else:
                sign_digit = byte >> 4
                if sign_digit == 0:
                    delta = byte
                    pos += 1
                else:
                    extra = sign_digit // 2
                    delta = _decode_value(chunk[pos:pos + 1 + extra])
                    pos += 1 + extra
            accumulator += delta
            signal[count] = accumulator
            count += 1

        stream.read(2)      # trailing length marker
        accumulator = 0     # the accumulator resets at every chunk boundary

    return signal[:count]


def _read_interval(data: bytes) -> int:
    """Sampling interval in milliseconds, stored as int32 at offset 4."""
    return struct.unpack_from("<i", data, 4)[0]


def _read_wavelengths(ole: olefile.OleFileIO) -> np.ndarray:
    data = ole.openstream("PDA 3D Raw Data/Wavelength Table").read()
    count = struct.unpack_from("<I", data, 0)[0]
    return np.frombuffer(data[4:4 + 4 * count], dtype="<u4") / 100.0


def _time_axis(n: int, interval_ms: int) -> np.ndarray:
    return np.arange(n, dtype=float) * interval_ms / 1000.0


def _parse_pda(ole: olefile.OleFileIO, filename: str) -> Signal3D | None:
    if not ole.exists("PDA 3D Raw Data/3D Raw Data"):
        return None

    wavelengths = _read_wavelengths(ole)
    interval_ms = _read_interval(ole.openstream("PDA 3D Raw Data/Max Plot").read())

    raw = ole.openstream("PDA 3D Raw Data/3D Raw Data").read()
    stream = io.BytesIO(raw)
    spectra = []
    while stream.tell() < len(raw):
        block = _decode_block(stream)
        if block is None or len(block) != len(wavelengths):
            break
        spectra.append(block)

    if not spectra:
        return None

    cube = np.array(spectra) / _COUNTS_PER_UNIT
    return Signal3D(
        detector_name="PDA",
        time=_time_axis(len(cube), interval_ms),
        time_unit="s",
        wavelengths=wavelengths,
        wavelength_unit="nm",
        signal=cube,
        signal_unit=_DETECTOR_UNITS["PDA"],
        filename=filename,
        additional_data={"sampling_rate_hz": 1000.0 / interval_ms},
    )


def _parse_2d_traces(ole: olefile.OleFileIO, filename: str) -> dict[str, Signal2D]:
    signals: dict[str, Signal2D] = {}
    for entry in ole.listdir():
        path = "/".join(entry)
        if not path.startswith("LSS Raw Data/Chromatogram Ch"):
            continue
        if ole.get_size(path) == 0:
            continue

        data = ole.openstream(path).read()
        values = _decode_block(io.BytesIO(data))
        if values is None or len(values) == 0:
            continue

        interval_ms = _read_interval(data)
        name = "RID"
        signals[name] = Signal2D(
            detector_name=name,
            time=_time_axis(len(values), interval_ms),
            time_unit="s",
            signal=values / _COUNTS_PER_UNIT,
            signal_unit=_DETECTOR_UNITS[name],
            filename=filename,
            additional_data={"sampling_rate_hz": 1000.0 / interval_ms},
        )
    return signals


def parse_shimadzu(filepath: str) -> Chromatogram:
    """Parse a Shimadzu LabSolutions .lcd file into a Chromatogram."""
    ole = olefile.OleFileIO(filepath)
    try:
        filename = filepath.replace("\\", "/").rstrip("/").split("/")[-1]

        signals: dict[str, Signal2D | Signal3D] = {}
        pda = _parse_pda(ole, filename)
        if pda is not None:
            signals["PDA"] = pda
        signals.update(_parse_2d_traces(ole, filename))

        if not signals:
            raise ValueError(f"No detector signals found in {filepath}")

        additional_data: dict = {}
        if pda is not None:
            wavelengths = pda.wavelengths
            additional_data.update({
                "wavelengths": len(wavelengths),
                "start_wavelength": float(wavelengths[0]),
                "end_wavelength": float(wavelengths[-1]),
                "wavelength_step": round(float(np.median(np.diff(wavelengths))), 2),
            })
        first = next(iter(signals.values()))
        additional_data["sampling_rate_hz"] = first.additional_data["sampling_rate_hz"]

        return Chromatogram(
            signals=signals, filename=filename, additional_data=additional_data
        )
    finally:
        ole.close()

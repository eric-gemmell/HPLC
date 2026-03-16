"""
EZChrom Elite .dat File Parser
================================
Parses binary .dat files from EZChrom Elite HPLC software (Version 3.1.7).
Returns a Chromatogram with typed Signal2D objects.

Requirements: pip install olefile numpy

Reverse-engineered from binary analysis, March 2026.

File format notes:
    - OLE2 Compound Document container (magic: d0 cf 11 e0 a1 b1 1a e1)
    - Detector traces live in streams named "Detector Data/Detector N Trace"
    - Each trace stream: 56-byte header | int32[] intensity data | 40-byte footer
    - Header uint32 at offset 4 = number of data points
    - Scale factor 0.1 converts raw ints to mAU (UV) or mV (ELSD)
    - 10 Hz sampling rate (0.1 sec/point)
"""

import olefile
import struct
import numpy as np
import re

from hplc.core.chromatogram import Chromatogram
from hplc.core.signal import Signal2D


def parse_ezchrom(filepath: str) -> Chromatogram:
    """Parse an EZChrom Elite .dat file into a Chromatogram."""
    ole = olefile.OleFileIO(filepath)

    header = _parse_header(ole)
    detector_info = _parse_trace_handler(ole)
    wavelengths = _extract_wavelengths(ole)

    traces = []
    for entry in ole.listdir():
        path = "/".join(entry)
        if path.startswith("Detector Data/Detector ") and path.endswith(" Trace"):
            det_id = int(entry[1].split()[1])
            raw_bytes = ole.openstream(path).read()
            trace = _parse_trace(raw_bytes, det_id, detector_info)
            traces.append(trace)

    traces.sort(key=lambda t: t["detector_id"])
    ole.close()

    signals = {
        trace["name"]: Signal2D(
            time=trace["time"],
            signal=trace["raw"].astype(float) * trace["scale_factor"],
            detector_name=trace["name"],
            time_unit=trace["time_unit"],
            measurement_unit=trace["unit"],
            metadata={
                "detector_id": trace["detector_id"],
                
                "scale_factor": trace["scale_factor"],
                "sampling_rate_hz": trace["sampling_rate_hz"],
            },
        )
        for trace in traces
    }

    metadata = {
        "filepath": filepath,
        "sample_name": header["sample_name"],
        "method_path": header["method_path"],
        "software_version": header["software_version"],
        "instrument": header["instrument"],
        "timestamp_serial": header["timestamp_serial"],
        "wavelengths": wavelengths,
    }

    return Chromatogram(signals=signals, metadata=metadata)


# ── Internal parsers ───────────────────────────────────────────────────────

def _parse_header(ole):
    """
    Parse the 'Chrom Header' stream.

    Layout:
      0x00: uint32 flag
      0x08: float64 Excel date serial
      0x10: length-prefixed string (method path)
            length-prefixed string (sample name)
            length-prefixed string (software version)
      later: instrument config name
    """
    data = ole.openstream("Chrom Header").read()

    result = {
        "timestamp_serial": struct.unpack_from("<d", data, 8)[0],
        "method_path": "",
        "sample_name": "",
        "software_version": "",
        "instrument": "",
    }

    # Length-prefixed strings starting at offset 0x10
    pos = 0x10
    for key in ("method_path", "sample_name", "software_version"):
        s, pos = _read_lpstring(data, pos)
        if s:
            result[key] = s

    # Instrument config name is typically after the "System" string
    text = data.decode("ascii", errors="replace")
    match = re.search(
        r"(Single_|Dual_|Gradient_|Binary_|Quaternary_)[A-Za-z0-9_&]+", text
    )
    if match:
        result["instrument"] = match.group(0)
    else:
        match = re.search(r"System.{0,10}?([A-Z][A-Za-z0-9_&]{10,})", text)
        if match:
            result["instrument"] = match.group(1)

    return result


def _read_lpstring(data, pos):
    """Read a length-prefixed ASCII string. Returns (string, new_pos)."""
    if pos >= len(data):
        return None, pos
    length = data[pos]
    if length == 0 or pos + 1 + length > len(data):
        return None, pos + 1
    raw = data[pos + 1 : pos + 1 + length]
    printable = sum(1 for b in raw if 32 <= b < 127)
    if printable / max(len(raw), 1) < 0.8:
        return None, pos + 1 + length
    return raw.decode("ascii", errors="replace"), pos + 1 + length


def _parse_trace_handler(ole):
    """
    Parse 'Detector Trace Handler' to get detector names and units.

    Returns dict: detector_id -> {"name": str, "unit": str, "scale": float}
    """
    info = {}
    try:
        data = ole.openstream("Detector Trace Handler").read()
    except Exception:
        return info

    known_ids = set()
    for entry in ole.listdir():
        path = "/".join(entry)
        if path.startswith("Detector Data/Detector ") and path.endswith(" Trace"):
            known_ids.add(int(entry[1].split()[1]))

    for det_id in sorted(known_ids):
        needle = struct.pack("<I", det_id)
        pos = 0
        while True:
            pos = data.find(needle, pos)
            if pos < 0:
                break

            for name_off in range(pos + 4, min(pos + 30, len(data) - 1)):
                b = data[name_off]
                if 8 <= b <= 60 and name_off + 1 + b <= len(data):
                    candidate = data[name_off + 1 : name_off + 1 + b]
                    if all(32 <= c < 127 for c in candidate):
                        name = candidate.decode("ascii")
                        if "etektor" in name or "etector" in name:
                            after = name_off + 1 + b
                            scale = 0.1
                            unit = "mAU"

                            if after + 4 <= len(data):
                                s = struct.unpack_from("<f", data, after)[0]
                                if 1e-6 < abs(s) < 1e6:
                                    scale = s

                            upos = after + 4
                            if upos < len(data):
                                ulen = data[upos]
                                if 1 <= ulen <= 10 and upos + 1 + ulen <= len(data):
                                    uc = data[upos + 1 : upos + 1 + ulen]
                                    if all(32 <= c < 127 for c in uc):
                                        unit = uc.decode("ascii")

                            if det_id not in info:
                                info[det_id] = {
                                    "name": name,
                                    "unit": unit,
                                    "scale": scale,
                                }
                            break
            pos += 4

    return info


def _extract_wavelengths(ole):
    """
    Extract UV wavelength settings from method substorages.
    Wavelengths are stored as uint32 at offset 0x51 in InstrDAD_00.
    """
    try:
        data = ole.openstream(
            "Method10/Method/DUIMethodSubStorage/InstrDAD_00"
        ).read()
        if len(data) > 0x61:
            wls = []
            for i in range(4):
                wl = struct.unpack_from("<I", data, 0x51 + i * 4)[0]
                if 190 <= wl <= 800:
                    wls.append(wl)
            return wls
    except Exception:
        pass
    return []


def _parse_trace(raw, det_id, detector_info):
    """
    Parse a single detector trace stream.

    Stream layout:
      [0:16]   4 x uint32 header (_, n_points, n_points, _)
      [16:56]  zeros (padding)
      [56:]    int32 array, followed by 40-byte footer
    """
    HEADER = 56
    FOOTER = 40

    _, n_points, _, _ = struct.unpack_from("<4I", raw, 0)

    max_points = (len(raw) - HEADER - FOOTER) // 4
    n_points = min(n_points, max_points)

    values = np.frombuffer(raw, dtype="<i4", count=n_points, offset=HEADER)

    meta = detector_info.get(
        det_id, {"name": f"Detector {det_id}", "unit": "mAU", "scale": 0.1}
    )

    rate = 10.0  # Hz
    time = np.linspace(0, n_points / (rate * 60), n_points)

    return {
        "detector_id": det_id,
        "name": meta["name"],
        "unit": meta["unit"],
        "scale_factor": meta["scale"],
        "sampling_rate_hz": rate,
        "time": time,
        "raw": values,
        "time_unit": "mins",
    }

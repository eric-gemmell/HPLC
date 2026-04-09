"""
Agilent AIA (.aia) Folder Parser
=================================
Parses Agilent .aia directories containing ANDI/AIA netCDF (.cdf) signal
files exported from ChemStation.

Each .cdf file is a self-contained signal with global attributes for
metadata (sample_name, detector_name, detector_unit, etc.) and a single
``ordinate_values`` variable for the signal data.

Returns a Chromatogram with typed Signal2D objects keyed by wavelength
labels like "DAD 254.0 nm".
"""

import re
from pathlib import Path

import numpy as np
from scipy.io import netcdf_file

from hplc.models.chromatogram import Chromatogram
from hplc.models.signal import Signal2D


def _read_attr(ds, name: str, default: str = "") -> str:
    """Read a global attribute from a netCDF dataset as a stripped string."""
    val = getattr(ds, name, None)
    if val is None:
        return default
    if isinstance(val, bytes):
        return val.decode("latin-1").strip()
    if isinstance(val, str):
        return val.strip()
    return str(val).strip()


def _parse_detector_label(raw_name: str) -> str:
    """Convert e.g. 'DAD1 A, Sig=254,4 Ref=off' -> 'DAD 254.0 nm'."""
    m = re.search(r"Sig=(\d+)", raw_name)
    if m:
        return f"DAD {float(m.group(1)):.1f} nm"
    return raw_name


def _parse_cdf_file(filepath: Path) -> dict:
    """Parse a single AIA .cdf signal file."""
    ds = netcdf_file(str(filepath), "r", mmap=False)
    try:
        raw_detector = _read_attr(ds, "detector_name")
        detector_label = _parse_detector_label(raw_detector)
        unit = _read_attr(ds, "detector_unit", "mAU")
        sample_name = _read_attr(ds, "sample_name")
        injection_time = _read_attr(ds, "HP_injection_time")
        method_name = _read_attr(ds, "detection_method_name")
        source_ref = _read_attr(ds, "source_file_reference")
        retention_unit = _read_attr(ds, "retention_unit", "seconds")

        v = ds.variables
        signal = np.array(v["ordinate_values"].data, dtype=np.float64)
        delay = float(v["actual_delay_time"].data)
        run_length = float(v["actual_run_time_length"].data)
        interval = float(v["actual_sampling_interval"].data)

        n_points = len(signal)
        sampling_rate_hz = round(1.0 / interval, 1) if interval > 0 else 0.0

        time = np.linspace(delay, delay + run_length, n_points)

        time_unit = "s" if retention_unit.lower() == "seconds" else retention_unit
    finally:
        ds.close()

    return {
        "detector_label": detector_label,
        "raw_detector_name": raw_detector,
        "unit": unit,
        "time": time,
        "time_unit": time_unit,
        "signal": signal,
        "sampling_rate_hz": sampling_rate_hz,
        "sample_name": sample_name,
        "injection_time": injection_time,
        "method_name": method_name,
        "source_ref": source_ref,
    }


def _resolve_aia_folder(filepath: str) -> Path | None:
    """Resolve a filepath to an Agilent .aia directory with .cdf files."""
    p = Path(filepath)
    if p.is_dir():
        cdf_files = list(p.glob("*.cdf")) + list(p.glob("*.CDF"))
        if cdf_files:
            return p
    if p.suffix.lower() != ".aia":
        p_with_ext = Path(str(p) + ".aia")
        if p_with_ext.is_dir():
            cdf_files = list(p_with_ext.glob("*.cdf")) + list(p_with_ext.glob("*.CDF"))
            if cdf_files:
                return p_with_ext
    return None


def is_agilent_aia_chromatogram(filepath: str) -> bool:
    """Check whether *filepath* is an Agilent .aia directory with .cdf files."""
    return _resolve_aia_folder(filepath) is not None


def parse_agilent_aia_data_file(filepath: str) -> Chromatogram:
    """Parse an Agilent .aia folder into a Chromatogram."""
    aia_folder = _resolve_aia_folder(filepath)
    if aia_folder is None:
        raise ValueError(f"Not a valid Agilent .aia directory: {filepath}")

    cdf_files = sorted(aia_folder.glob("*.cdf")) + sorted(aia_folder.glob("*.CDF"))

    signals: dict[str, Signal2D] = {}
    chromatogram_meta: dict = {}

    for cdf_path in cdf_files:
        parsed = _parse_cdf_file(cdf_path)

        if not chromatogram_meta:
            # Extract injection date (just the date part from "28-Nov-25, 13:55:58")
            injection_time = parsed["injection_time"]
            injection_date = injection_time.split(",")[0].strip() if injection_time else ""

            # Extract method path from source_file_reference
            source_ref = parsed["source_ref"]
            method_name = parsed["method_name"]

            chromatogram_meta = {
                "sample_name": parsed["sample_name"],
                "injection_date": injection_date,
                "method_path": method_name,
                "source_file_reference": source_ref,
            }

        detector_label = parsed["detector_label"]
        signals[detector_label] = Signal2D(
            time=parsed["time"],
            signal=parsed["signal"],
            detector_name=detector_label,
            time_unit=parsed["time_unit"],
            signal_unit=parsed["unit"],
            filename=aia_folder.name,
            additional_data={
                "raw_detector_name": parsed["raw_detector_name"],
                "sampling_rate_hz": parsed["sampling_rate_hz"],
                "sample_name": parsed["sample_name"],
            },
        )

    return Chromatogram(
        signals=signals,
        filename=aia_folder.name,
        additional_data=chromatogram_meta,
    )

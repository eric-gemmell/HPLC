"""
Shimadzu LabSolutions ASCII (.txt) Export Parser
================================================
Parses the tab-separated text export of a LabSolutions data file into a
Chromatogram carrying a Signal3D for the PDA cube and a Signal2D for each
"[LC Chromatogram(<detector>)]" section.

File format notes:
    - Sections are introduced by "[Section Name]" lines; the first is "[Header]"
      followed by "Application Name<TAB>LabSolutions"
    - Numbers use the locale of the exporting PC, so decimals may be commas
    - Times are in minutes; we convert to seconds to match the .lcd parser
    - "[LC Chromatogram(<det>)]": key/value header (Interval(msec), # of Points,
      Intensity Units, Intensity Multiplier), then "R.Time (min)<TAB>Intensity"
      rows of raw counts; counts * multiplier gives the stated unit
    - "[PDA 3D]": key/value header, an "R.Time (min)" line, a row of
      wavelengths in 0.01 nm, then one row per time point (time, then one
      value per wavelength) in uAU
    - "[PDA Multi Chromatogram(...)]" holds single-wavelength extractions of the
      cube; they are not loaded, as they can be re-derived from the PDA signal

    "AD2" is the instrument's second analog input. What it records depends on
    the wiring; on the instrument this was developed against it carries the
    PDA's analog output, not an RID.
"""

from __future__ import annotations

import numpy as np

from hplc.models.chromatogram import Chromatogram
from hplc.models.signal import Signal2D, Signal3D

_ENCODING = "latin-1"

# PDA cube values are exported in uAU; report them in mAU like the .lcd parser.
_PDA_COUNTS_PER_MAU = 1000.0
_PDA_WAVELENGTH_SCALE = 100.0   # wavelengths are exported in 0.01 nm

_CHROMATOGRAM_PREFIX = "[LC Chromatogram("


def is_shimadzu_txt_file(filepath: str) -> bool:
    """Check whether a file is a Shimadzu LabSolutions ASCII export."""
    try:
        with open(filepath, "r", encoding=_ENCODING) as fh:
            first = fh.readline().strip()
            second = fh.readline().strip()
        return first == "[Header]" and second.split("\t") == ["Application Name", "LabSolutions"]
    except Exception:
        return False


def _number(text: str) -> float:
    return float(text.strip().replace(",", "."))


def _split_sections(lines: list[str]) -> dict[str, list[str]]:
    sections: dict[str, list[str]] = {}
    current: list[str] | None = None
    for line in lines:
        if line.startswith("[") and line.rstrip().endswith("]"):
            current = sections.setdefault(line.strip(), [])
        elif current is not None:
            current.append(line)
    return sections


def _read_header(body: list[str]) -> tuple[dict[str, str], int]:
    """Key/value lines up to the "R.Time (min)" column line; returns the index after it."""
    header: dict[str, str] = {}
    for i, line in enumerate(body):
        if line.startswith("R.Time"):
            return header, i + 1
        key, _, value = line.partition("\t")
        header[key.strip()] = value.strip()
    raise ValueError("Section has no 'R.Time' data block")


def _parse_table(rows: list[str], n_rows: int, n_cols: int) -> np.ndarray:
    text = "\n".join(rows[:n_rows]).replace(",", ".")
    values = np.array(text.split(), dtype=float)
    return values.reshape(n_rows, n_cols)


def _parse_2d(name: str, body: list[str], filename: str) -> Signal2D:
    header, start = _read_header(body)
    n_points = int(header["# of Points"])
    table = _parse_table(body[start:], n_points, 2)
    interval_ms = _number(header["Interval(msec)"])
    multiplier = _number(header.get("Intensity Multiplier", "1"))
    return Signal2D(
        detector_name=name,
        time=table[:, 0] * 60.0,
        time_unit="s",
        signal=table[:, 1] * multiplier,
        signal_unit=header["Intensity Units"],
        filename=filename,
        additional_data={"sampling_rate_hz": 1000.0 / interval_ms},
    )


def _parse_pda(body: list[str], filename: str) -> Signal3D:
    header, start = _read_header(body)
    n_wavelengths = int(header["# of Wavelength Axis Points"])
    n_times = int(header["# of Time Axis Points"])
    interval_ms = _number(header["Interval(msec)"])

    wavelengths = np.array(body[start].split(), dtype=float) / _PDA_WAVELENGTH_SCALE
    if len(wavelengths) != n_wavelengths:
        raise ValueError(
            f"PDA 3D wavelength row has {len(wavelengths)} values, expected {n_wavelengths}"
        )

    table = _parse_table(body[start + 1:], n_times, n_wavelengths + 1)
    return Signal3D(
        detector_name="PDA",
        time=table[:, 0] * 60.0,
        time_unit="s",
        wavelengths=wavelengths,
        wavelength_unit="nm",
        signal=table[:, 1:] / _PDA_COUNTS_PER_MAU,
        signal_unit="mAU",
        filename=filename,
        additional_data={"sampling_rate_hz": 1000.0 / interval_ms},
    )


def parse_shimadzu_txt(filepath: str) -> Chromatogram:
    """Parse a Shimadzu LabSolutions ASCII export into a Chromatogram."""
    with open(filepath, "r", encoding=_ENCODING) as fh:
        lines = fh.read().splitlines()
    filename = filepath.replace("\\", "/").rstrip("/").split("/")[-1]
    sections = _split_sections(lines)

    signals: dict[str, Signal2D | Signal3D] = {}
    pda = None
    if "[PDA 3D]" in sections:
        pda = _parse_pda(sections["[PDA 3D]"], filename)
        signals["PDA"] = pda
    for title, body in sections.items():
        if title.startswith(_CHROMATOGRAM_PREFIX):
            name = title[len(_CHROMATOGRAM_PREFIX):-2]
            signals[name] = _parse_2d(name, body, filename)

    if not signals:
        raise ValueError(f"No detector signals found in {filepath}")

    additional_data: dict = {}
    sample_info, _ = _read_header(sections.get("[Sample Information]", []) + ["R.Time"])
    if "Sample Name" in sample_info:
        additional_data["sample_name"] = sample_info["Sample Name"]
    if "Acquired" in sample_info:
        additional_data["acquired"] = sample_info["Acquired"]
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

    return Chromatogram(signals=signals, filename=filename, additional_data=additional_data)

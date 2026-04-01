"""
agilent_ce_parser.py — Parse Agilent ChemStation .D folders (version 130 .ch files)

Supports Agilent CE (capillary electrophoresis) and DAD (diode array detector)
data files produced by ChemStation software. Handles the segment-framed,
delta-encoded binary .ch format used in version-130 "LC DATA FILE" containers.

Binary layout (version 130):
    0x000       Version string ("130")
    0x11A       Start retention time (int32 BE, units of 1/60000 min)
    0x11E       End retention time   (int32 BE, units of 1/60000 min)
    0x127C      Y-axis scaling factor (float64 BE)
    0x15C       File type marker ("LC DATA FILE", UTF-16LE)
    0x35A       Sample / notebook name (pascal UTF-16, gap=2)
    0x957       Acquisition date
    0xA0E       Method name
    0xC11       Instrument name
    0x104C      Y-axis unit string
    0x1075      Signal description (e.g. "DAD1A, Sig=254,4  Ref=off")
    0x1800      Data start — segment-framed delta-encoded signal

Data encoding:
    Data is stored in segments.  Each segment starts with a 0x10 header byte
    followed by a count byte (number of data points in the segment).  Within
    a segment, each point is either:
      • A 16-bit big-endian signed delta added to a running accumulator, OR
      • The sentinel value -0x8000 (−32768) followed by a 32-bit big-endian
        signed integer that *replaces* the accumulator (used when the signal
        exceeds int16 range).
    The final signal in real units = raw_accumulated_value × scaling_factor.

Usage:
    from agilent_ce_parser import AgilentDFolder

    run = AgilentDFolder("032-0301.D")   # or path to a .zip
    print(run)

    ch = run["DAD1A"]
    ch.time_minutes   # numpy array (min)
    ch.signal          # numpy array (mAU, µA, kV, … depending on channel)
    ch.plot()          # quick matplotlib figure
"""

from __future__ import annotations

import os
import re
import struct
import zipfile
import tempfile
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Binary layout constants — version 130
# ---------------------------------------------------------------------------
_HEADER_OFFSETS_130 = {
    "time_range":     0x11A,     # int32 BE × 2: start and end time (1/60000 min)
    "scaling_factor": 0x127C,    # float64 BE
    "data_start":     0x1800,    # first segment header byte
}

_METADATA_OFFSETS_130 = {
    "notebook":   0x35A,
    "date":       0x957,
    "method":     0xA0E,
    "instrument": 0xC11,
    "unit":       0x104C,
    "signal":     0x1075,
}

_METADATA_GAP_130 = 2           # UTF-16: every other byte is NUL
_TIME_UNIT = 60_000.0           # ticks → minutes


# ---------------------------------------------------------------------------
# Low-level binary helpers
# ---------------------------------------------------------------------------

def _read_pascal_string(data: bytes, offset: int, gap: int = 2) -> str:
    """
    Read a pascal-style string.  First byte is the character count, then
    ``count × gap`` raw bytes.  Characters are extracted by stepping with
    stride ``gap`` (gap=2 for UTF-16LE where every other byte is 0x00).
    """
    if offset >= len(data):
        return ""
    char_count = data[offset]
    raw = data[offset + 1 : offset + 1 + char_count * gap]
    try:
        return raw[::gap].decode("utf-8", errors="replace").strip()
    except Exception:
        return ""


def _decode_segments(data: bytes, start: int) -> np.ndarray:
    """
    Decode segment-framed, delta-encoded signal data.

    Each segment begins with ``0x10`` followed by a count byte.  Inside a
    segment, a signed int16 value of ``-0x8000`` is a sentinel indicating
    that the next 4 bytes are an absolute int32 replacement for the running
    accumulator; any other int16 is a delta added to the accumulator.

    Returns a 1-D int64 NumPy array of the reconstructed raw signal.
    """
    values: list[int] = []
    offset = start
    accumulator = 0
    length = len(data)

    while offset < length:
        # Segment header
        if data[offset] != 0x10:
            break
        if offset + 1 >= length:
            break
        n_points = data[offset + 1]
        offset += 2

        for _ in range(n_points):
            if offset + 2 > length:
                break
            delta = struct.unpack(">h", data[offset : offset + 2])[0]
            if delta == -0x8000:
                # Sentinel: next 4 bytes are an absolute value
                if offset + 6 > length:
                    break
                accumulator = struct.unpack(">i", data[offset + 2 : offset + 6])[0]
                offset += 6
            else:
                accumulator += delta
                offset += 2
            values.append(accumulator)

    return np.array(values, dtype=np.int64)


# ---------------------------------------------------------------------------
# Channel data class
# ---------------------------------------------------------------------------

@dataclass
class Channel:
    """One signal channel parsed from a .ch file."""

    name: str
    filepath: str
    version: str

    # Timing
    start_time_min: float
    end_time_min: float

    # Signal
    _raw_signal: np.ndarray        # integer ADC counts (before scaling)
    scaling_factor: float           # multiply raw to get real units

    # Metadata from the .ch header
    notebook: str = ""
    date: str = ""
    method: str = ""
    instrument: str = ""
    unit: str = ""
    signal_description: str = ""

    # ------ derived properties ------

    @property
    def n_points(self) -> int:
        return len(self._raw_signal)

    @property
    def duration_min(self) -> float:
        return self.end_time_min - self.start_time_min

    @property
    def sampling_rate_hz(self) -> float:
        if self.duration_min <= 0 or self.n_points < 2:
            return 0.0
        return self.n_points / (self.duration_min * 60.0)

    @property
    def time_minutes(self) -> np.ndarray:
        """Evenly-spaced time axis in minutes."""
        return np.linspace(self.start_time_min, self.end_time_min, self.n_points)

    @property
    def time_seconds(self) -> np.ndarray:
        """Evenly-spaced time axis in seconds."""
        return self.time_minutes * 60.0

    @property
    def signal(self) -> np.ndarray:
        """Signal in real units (mAU, µA, kV, … ) after applying scaling."""
        return self._raw_signal.astype(np.float64) * self.scaling_factor

    @property
    def signal_raw(self) -> np.ndarray:
        """Unscaled integer signal (ADC counts)."""
        return self._raw_signal

    @property
    def metadata(self) -> dict:
        return {
            "name":               self.name,
            "signal_description": self.signal_description,
            "unit":               self.unit,
            "notebook":           self.notebook,
            "date":               self.date,
            "method":             self.method,
            "instrument":         self.instrument,
            "version":            self.version,
            "scaling_factor":     self.scaling_factor,
            "start_time_min":     round(self.start_time_min, 6),
            "end_time_min":       round(self.end_time_min, 6),
            "n_points":           self.n_points,
            "duration_min":       round(self.duration_min, 4),
            "sampling_rate_hz":   round(self.sampling_rate_hz, 2),
            "signal_min":         round(float(self.signal.min()), 4),
            "signal_max":         round(float(self.signal.max()), 4),
        }

    # ------ convenience ------

    def to_dataframe(self):
        """Return a pandas DataFrame with columns ``time_min`` and ``signal``."""
        import pandas as pd
        return pd.DataFrame({
            "time_min": self.time_minutes,
            f"signal ({self.unit})": self.signal,
        })

    def plot(self, ax=None, **kwargs):
        """Quick matplotlib plot.  Returns the Axes."""
        import matplotlib.pyplot as plt
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 3))
        label = kwargs.pop("label", f"{self.name} — {self.signal_description}")
        ax.plot(self.time_minutes, self.signal, label=label, linewidth=0.7, **kwargs)
        ax.set_xlabel("Time (min)")
        ax.set_ylabel(self.unit or "Signal")
        ax.set_title(label)
        ax.legend(fontsize=8)
        ax.set_xlim(self.start_time_min, self.end_time_min)
        return ax

    def __repr__(self) -> str:
        return (
            f"Channel('{self.name}', \"{self.signal_description}\", "
            f"{self.n_points} pts, "
            f"{self.start_time_min:.2f}–{self.end_time_min:.2f} min, "
            f"{self.sampling_rate_hz:.1f} Hz, "
            f"[{self.signal.min():.2f} .. {self.signal.max():.2f}] {self.unit})"
        )


# ---------------------------------------------------------------------------
# File-level parser
# ---------------------------------------------------------------------------

def parse_ch_file(filepath: str | Path) -> Channel:
    """
    Parse a single Agilent .ch file (version 130, segment-framed delta encoding).

    Parameters
    ----------
    filepath : path to the .ch file

    Returns
    -------
    Channel object with time/signal arrays and metadata.

    Raises
    ------
    ValueError
        If the file version is not 130 or the file is too small.
    """
    filepath = Path(filepath)
    with open(filepath, "rb") as f:
        data = f.read()

    min_size = _HEADER_OFFSETS_130["data_start"]
    if len(data) < min_size:
        raise ValueError(
            f"{filepath.name}: file too small ({len(data)} bytes, "
            f"need at least {min_size} for version 130)"
        )

    # Version check
    version = data[1:4].decode("ascii", errors="replace").strip("\x00")
    if version not in ("130", "30"):
        raise ValueError(
            f"{filepath.name}: expected version 130 (or 30), got '{version}'. "
            "Versions 179/181 (FID) use a different layout and are not yet supported."
        )

    # Timing
    t_off = _HEADER_OFFSETS_130["time_range"]
    start_raw, end_raw = struct.unpack(">ii", data[t_off : t_off + 8])
    start_min = start_raw / _TIME_UNIT
    end_min = end_raw / _TIME_UNIT

    # Scaling factor
    sf_off = _HEADER_OFFSETS_130["scaling_factor"]
    scaling_factor = struct.unpack(">d", data[sf_off : sf_off + 8])[0]

    # Signal data
    raw_signal = _decode_segments(data, _HEADER_OFFSETS_130["data_start"])

    # Metadata strings
    meta = {}
    for key, off in _METADATA_OFFSETS_130.items():
        meta[key] = _read_pascal_string(data, off, gap=_METADATA_GAP_130)

    ch_name = filepath.stem.upper()

    return Channel(
        name=ch_name,
        filepath=str(filepath),
        version=version,
        start_time_min=start_min,
        end_time_min=end_min,
        _raw_signal=raw_signal,
        scaling_factor=scaling_factor,
        notebook=meta.get("notebook", ""),
        date=meta.get("date", ""),
        method=meta.get("method", ""),
        instrument=meta.get("instrument", ""),
        unit=meta.get("unit", ""),
        signal_description=meta.get("signal", ch_name),
    )


# ---------------------------------------------------------------------------
# .D folder parser
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
    meta = {}
    for tag in ("Name", "Description", "ACQMethodPath", "DAMethodPath"):
        m = re.search(rf"<{tag}>(.*?)</{tag}>", text, re.DOTALL)
        if m:
            meta[tag.lower()] = m.group(1).strip()
    return meta


def _parse_acq_macaml(d_folder: Path) -> dict:
    """Extract DAD signal wavelengths from acq.macaml."""
    acq_file = d_folder / "acq.macaml"
    if not acq_file.exists():
        return {}
    text = acq_file.read_text(encoding="utf-8", errors="replace")
    wavelengths: dict[str, dict] = {}
    rows = re.findall(r"<Row>(.*?)</Row>", text, re.DOTALL)
    for row in rows:
        sig_m = re.search(
            r"Signals_Signal_ID.*?<Value>(Signal \w+)</Value>", row, re.DOTALL
        )
        wl_m = re.search(
            r"Signals_Signal_Wavelength.*?<Value>([\d.]+)</Value>", row, re.DOTALL
        )
        bw_m = re.search(
            r"Signals_Signal_Bandwidth.*?<Value>([\d.]+)</Value>", row, re.DOTALL
        )
        use_m = re.search(
            r"Signals_Signal_UseSignal.*?<Value>(\w+)</Value>", row, re.DOTALL
        )
        if sig_m:
            letter = sig_m.group(1).split()[-1]
            key = f"DAD1{letter}"
            wavelengths[key] = {
                "wavelength_nm": float(wl_m.group(1)) if wl_m else None,
                "bandwidth_nm":  float(bw_m.group(1)) if bw_m else None,
                "enabled":       (use_m.group(1).lower() == "yes") if use_m else False,
            }
    return wavelengths


class AgilentDFolder:
    """
    Parse an Agilent ChemStation .D folder (or zipped .D folder).

    Parameters
    ----------
    path : str or Path
        Path to a ``.D`` directory **or** a ``.zip`` file containing one.

    Examples
    --------
    >>> run = AgilentDFolder("032-0301.D")
    >>> run.channel_names
    ['DAD1A', 'DAD1B', 'DAD1C', 'HPCE1C', 'HPCE1P', 'HPCE1V']

    >>> dad = run["DAD1A"]
    >>> dad.signal          # numpy array in mAU
    >>> dad.time_minutes    # numpy array in minutes
    >>> dad.plot()
    """

    def __init__(self, path: str | Path):
        path = Path(path)
        self._tempdir: Optional[str] = None

        if path.suffix.lower() == ".zip":
            self._tempdir = tempfile.mkdtemp()
            with zipfile.ZipFile(path) as zf:
                zf.extractall(self._tempdir)
            candidates = [
                p for p in Path(self._tempdir).rglob("*")
                if p.is_dir() and p.suffix.upper() == ".D"
            ]
            if not candidates:
                candidates = [Path(self._tempdir)]
            self._d_folder = candidates[0]
        elif path.is_dir():
            self._d_folder = path
        else:
            raise FileNotFoundError(f"Not a directory or zip: {path}")

        self._channels: Dict[str, Channel] = {}
        self._sample_meta: dict = {}
        self._signal_config: dict = {}
        self._parse()

    def _parse(self):
        ch_files = sorted(
            self._d_folder.glob("*.ch"),
            key=lambda p: p.name.upper(),
        ) + sorted(
            self._d_folder.glob("*.CH"),
            key=lambda p: p.name.upper(),
        )
        # Deduplicate (case-insensitive filesystems)
        seen = set()
        for chf in ch_files:
            key = chf.name.upper()
            if key in seen:
                continue
            seen.add(key)
            try:
                ch = parse_ch_file(chf)
                self._channels[ch.name] = ch
            except Exception as exc:
                import warnings
                warnings.warn(f"Skipping {chf.name}: {exc}")

        self._sample_meta = _parse_sample_xml(self._d_folder)
        self._signal_config = _parse_acq_macaml(self._d_folder)

    # ------ access ------

    @property
    def folder_name(self) -> str:
        return self._d_folder.name

    @property
    def sample_name(self) -> str:
        return self._sample_meta.get("name", "")

    @property
    def method(self) -> str:
        acq = self._sample_meta.get("acqmethodpath", "")
        return acq.rsplit("\\", 1)[-1] if acq else ""

    @property
    def channels(self) -> Dict[str, Channel]:
        return dict(self._channels)

    @property
    def channel_names(self) -> List[str]:
        return sorted(self._channels.keys())

    @property
    def dad_channels(self) -> Dict[str, Channel]:
        """Only DAD (UV absorbance) channels."""
        return {k: v for k, v in self._channels.items() if k.startswith("DAD")}

    @property
    def ce_channels(self) -> Dict[str, Channel]:
        """Only CE instrument channels (current, voltage, power)."""
        return {k: v for k, v in self._channels.items() if k.startswith("HPCE")}

    @property
    def signal_config(self) -> dict:
        """DAD signal configuration from acq.macaml."""
        return dict(self._signal_config)

    def __getitem__(self, key: str) -> Channel:
        key = key.upper()
        if key in self._channels:
            return self._channels[key]
        raise KeyError(
            f"Channel '{key}' not found. Available: {self.channel_names}"
        )

    def __contains__(self, key: str) -> bool:
        return key.upper() in self._channels

    # ------ plotting ------

    def plot_dad(self, ax=None, **kwargs):
        """Plot all DAD channels on one axis."""
        import matplotlib.pyplot as plt
        if ax is None:
            _, ax = plt.subplots(figsize=(12, 4))
        for name, ch in sorted(self.dad_channels.items()):
            ch.plot(ax=ax, **kwargs)
        ax.set_title(f"{self.folder_name} — DAD signals")
        ax.legend(fontsize=8)
        return ax

    def plot_ce(self, **kwargs):
        """Plot CE instrument traces as subplots."""
        import matplotlib.pyplot as plt
        ce = sorted(self.ce_channels.items())
        if not ce:
            raise ValueError("No CE channels found.")
        fig, axes = plt.subplots(len(ce), 1, figsize=(12, 3 * len(ce)), sharex=True)
        if len(ce) == 1:
            axes = [axes]
        for ax, (name, ch) in zip(axes, ce):
            ch.plot(ax=ax, **kwargs)
        fig.suptitle(f"{self.folder_name} — CE traces", fontsize=12)
        fig.tight_layout()
        return fig

    def plot_all(self, **kwargs):
        """Plot everything: DAD + CE channels."""
        import matplotlib.pyplot as plt
        n_dad = len(self.dad_channels)
        n_ce = len(self.ce_channels)
        n_total = (1 if n_dad else 0) + n_ce
        if n_total == 0:
            raise ValueError("No channels to plot.")
        fig, axes = plt.subplots(n_total, 1, figsize=(12, 3 * n_total), sharex=True)
        if n_total == 1:
            axes = [axes]
        idx = 0
        if n_dad:
            self.plot_dad(ax=axes[idx], **kwargs)
            idx += 1
        for name, ch in sorted(self.ce_channels.items()):
            ch.plot(ax=axes[idx], **kwargs)
            idx += 1
        fig.suptitle(
            f"{self.folder_name} — {self.sample_name}",
            fontsize=12, y=1.01,
        )
        fig.tight_layout()
        return fig

    # ------ repr ------

    def __repr__(self) -> str:
        lines = [
            f"AgilentDFolder('{self.folder_name}')",
            f"  Sample     : {self.sample_name}",
            f"  Method     : {self.method}",
            f"  Instrument : {next(iter(self._channels.values())).instrument if self._channels else ''}",
            f"  Date       : {next(iter(self._channels.values())).date if self._channels else ''}",
            f"  Channels ({len(self._channels)}):",
        ]
        for name in self.channel_names:
            ch = self._channels[name]
            lines.append(f"    {ch}")
        return "\n".join(lines)

    def __del__(self):
        if self._tempdir and os.path.isdir(self._tempdir):
            shutil.rmtree(self._tempdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# CLI convenience
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python agilent_ce_parser.py <path-to-.D-or-.zip>")
        sys.exit(1)

    run = AgilentDFolder(sys.argv[1])
    print(run)
    print()
    for name, ch in run.channels.items():
        for k, v in ch.metadata.items():
            print(f"  {k}: {v}")
        print()

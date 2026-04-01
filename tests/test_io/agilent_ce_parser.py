"""
agilent_ce_parser.py — Parse Agilent ChemStation .D folders (version 130 .ch files)

Supports Agilent CE (capillary electrophoresis) and DAD (diode array detector)
data files produced by ChemStation software. Handles the delta-encoded binary
.ch format used in version-130 "LC DATA FILE" containers.

Usage:
    from agilent_ce_parser import AgilentDFolder

    run = AgilentDFolder("032-0301.D")            # or path to a .zip
    print(run)                                      # summary of all channels
    print(run.channels)                             # dict of channel names

    ch = run["DAD1A"]                               # access by channel name
    ch.plot()                                       # quick matplotlib plot

    # raw numpy arrays
    time_min = ch.time_minutes
    signal   = ch.signal

    # metadata
    ch.metadata                                     # dict with start/end time, n_points, etc.
"""

from __future__ import annotations

import os
import re
import struct
import zipfile
import tempfile
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Constants for version-130 .ch binary layout
# ---------------------------------------------------------------------------
_VERSION_OFFSET = 1          # 3-byte ASCII version string starts at byte 1
_FILE_TYPE_OFFSET = 0x15C    # UTF-16LE "LC DATA FILE" marker
_TIME_START_OFFSET = 0x11A   # int32 BE — start retention time (1/60000 min)
_TIME_END_OFFSET = 0x11E     # int32 BE — end retention time   (1/60000 min)
_DATA_START_OFFSET = 0x200   # delta-encoded data begins here
_TIME_UNIT = 60_000.0        # ticks per minute

# Delta-encoding sentinel: when a 16-bit delta equals 0x7FFF the next 4 bytes
# are an absolute 32-bit value that *replaces* the running accumulator.
_DELTA_SENTINEL = 0x7FFF

# Channel name conventions
_CHANNEL_INFO = {
    "HPCE1C": {"description": "CE Current",   "units": "\u00b5A"},
    "HPCE1V": {"description": "CE Voltage",   "units": "kV"},
    "HPCE1P": {"description": "CE Pressure",  "units": "mbar"},
    "DAD1A":  {"description": "DAD Signal A",  "units": "mAU"},
    "DAD1B":  {"description": "DAD Signal B",  "units": "mAU"},
    "DAD1C":  {"description": "DAD Signal C",  "units": "mAU"},
    "DAD1D":  {"description": "DAD Signal D",  "units": "mAU"},
    "DAD1E":  {"description": "DAD Signal E",  "units": "mAU"},
    "DAD1F":  {"description": "DAD Signal F",  "units": "mAU"},
    "DAD1G":  {"description": "DAD Signal G",  "units": "mAU"},
    "DAD1H":  {"description": "DAD Signal H",  "units": "mAU"},
}


# ---------------------------------------------------------------------------
# Low-level binary helpers
# ---------------------------------------------------------------------------

def _read_be_int32(data: bytes, offset: int) -> int:
    """Read a big-endian signed 32-bit integer."""
    return struct.unpack(">i", data[offset : offset + 4])[0]


def _read_be_int16(data: bytes, offset: int) -> int:
    """Read a big-endian signed 16-bit integer."""
    return struct.unpack(">h", data[offset : offset + 2])[0]


def _decode_version(data: bytes) -> str:
    """Return the ASCII version string from bytes 1-3."""
    return data[1:4].decode("ascii", errors="replace").strip("\x00")


def _decode_utf16le(data: bytes, offset: int, max_len: int = 40) -> str:
    """Try to read a UTF-16LE string at *offset* (up to *max_len* chars)."""
    raw = data[offset : offset + max_len * 2]
    try:
        return raw.decode("utf-16-le").strip("\x00").strip()
    except Exception:
        return ""


def _delta_decode(data: bytes, start: int = _DATA_START_OFFSET) -> np.ndarray:
    """
    Delta-decode the run-length-compressed chromatogram data.

    Starting from *start*, reads big-endian int16 deltas.  Each value is added
    to a running accumulator.  When a delta equals 0x7FFF (the sentinel), the
    next four bytes are read as a big-endian int32 absolute value that replaces
    the accumulator outright.

    Returns a 1-D int64 NumPy array of the reconstructed signal.
    """
    values: list[int] = []
    offset = start
    current = 0
    length = len(data)

    while offset + 2 <= length:
        delta = struct.unpack(">h", data[offset : offset + 2])[0]
        if delta == _DELTA_SENTINEL:
            if offset + 6 <= length:
                current = struct.unpack(">i", data[offset + 2 : offset + 6])[0]
                offset += 6
            else:
                break
        else:
            current += delta
            offset += 2
        values.append(current)

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
    file_type: str
    start_time_min: float
    end_time_min: float
    _raw_signal: np.ndarray
    description: str = ""
    units: str = ""

    # ------ derived properties ------

    @property
    def n_points(self) -> int:
        return len(self._raw_signal)

    @property
    def duration_min(self) -> float:
        return self.end_time_min - self.start_time_min

    @property
    def sampling_rate_hz(self) -> float:
        if self.duration_min <= 0:
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
        """Raw signal values (integer ADC counts / instrument units)."""
        return self._raw_signal

    @property
    def metadata(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "units": self.units,
            "version": self.version,
            "file_type": self.file_type,
            "start_time_min": self.start_time_min,
            "end_time_min": self.end_time_min,
            "n_points": self.n_points,
            "duration_min": self.duration_min,
            "sampling_rate_hz": round(self.sampling_rate_hz, 2),
            "signal_min": int(self._raw_signal.min()),
            "signal_max": int(self._raw_signal.max()),
        }

    # ------ convenience ------

    def to_dataframe(self):
        """Return a pandas DataFrame with columns ``time_min`` and ``signal``."""
        import pandas as pd
        return pd.DataFrame({
            "time_min": self.time_minutes,
            "signal": self.signal,
        })

    def plot(self, ax=None, **kwargs):
        """Quick matplotlib plot.  Returns the Axes."""
        import matplotlib.pyplot as plt
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 3))
        label = kwargs.pop("label", f"{self.name} ({self.description})")
        ax.plot(self.time_minutes, self.signal, label=label, linewidth=0.7, **kwargs)
        ax.set_xlabel("Time (min)")
        ax.set_ylabel(self.units or "Signal")
        ax.set_title(label)
        ax.legend(fontsize=8)
        ax.set_xlim(self.start_time_min, self.end_time_min)
        return ax

    def __repr__(self) -> str:
        return (
            f"Channel('{self.name}', {self.description}, "
            f"{self.n_points} pts, "
            f"{self.start_time_min:.2f}-{self.end_time_min:.2f} min, "
            f"{self.sampling_rate_hz:.1f} Hz)"
        )


# ---------------------------------------------------------------------------
# File-level parser
# ---------------------------------------------------------------------------

def parse_ch_file(filepath: str | Path) -> Channel:
    """
    Parse a single Agilent .ch file (version 130, delta-encoded).

    Parameters
    ----------
    filepath : path to the .ch file

    Returns
    -------
    Channel object with time/signal arrays and metadata.
    """
    filepath = Path(filepath)
    with open(filepath, "rb") as f:
        data = f.read()

    if len(data) < _DATA_START_OFFSET:
        raise ValueError(f"{filepath.name}: file too small ({len(data)} bytes)")

    version = _decode_version(data)
    if version != "130":
        raise ValueError(
            f"{filepath.name}: expected version 130, got '{version}'. "
            "Other versions (8, 179, 181) are not yet supported."
        )

    file_type = _decode_utf16le(data, _FILE_TYPE_OFFSET)
    start_raw = _read_be_int32(data, _TIME_START_OFFSET)
    end_raw = _read_be_int32(data, _TIME_END_OFFSET)
    start_min = start_raw / _TIME_UNIT
    end_min = end_raw / _TIME_UNIT

    signal = _delta_decode(data)

    # Derive channel name from filename (e.g. "DAD1A.ch" -> "DAD1A")
    ch_name = filepath.stem.upper()
    info = _CHANNEL_INFO.get(ch_name, {})

    return Channel(
        name=ch_name,
        filepath=str(filepath),
        version=version,
        file_type=file_type,
        start_time_min=start_min,
        end_time_min=end_min,
        _raw_signal=signal,
        description=info.get("description", ch_name),
        units=info.get("units", ""),
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
    # Find Signal table rows
    rows = re.findall(
        r"<Row>(.*?)</Row>", text, re.DOTALL
    )
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
            sig_letter = sig_m.group(1).split()[-1]  # "Signal A" -> "A"
            key = f"DAD1{sig_letter}"
            wavelengths[key] = {
                "wavelength_nm": float(wl_m.group(1)) if wl_m else None,
                "bandwidth_nm": float(bw_m.group(1)) if bw_m else None,
                "enabled": (use_m.group(1).lower() == "yes") if use_m else False,
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
    >>> dad.time_minutes   # numpy array
    >>> dad.signal          # numpy array
    >>> dad.plot()          # matplotlib figure
    """

    def __init__(self, path: str | Path):
        path = Path(path)
        self._tempdir: Optional[str] = None

        if path.suffix.lower() == ".zip":
            self._tempdir = tempfile.mkdtemp()
            with zipfile.ZipFile(path) as zf:
                zf.extractall(self._tempdir)
            # Find the .D folder inside
            candidates = [
                p for p in Path(self._tempdir).rglob("*")
                if p.is_dir() and p.suffix.upper() == ".D"
            ]
            if not candidates:
                # Maybe the zip root itself is the .D contents
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
        # Parse all .ch files
        ch_files = sorted(self._d_folder.glob("*.ch")) + sorted(
            self._d_folder.glob("*.CH")
        )
        for chf in ch_files:
            try:
                ch = parse_ch_file(chf)
                self._channels[ch.name] = ch
            except Exception as exc:
                import warnings
                warnings.warn(f"Skipping {chf.name}: {exc}")

        # Parse supplementary metadata
        self._sample_meta = _parse_sample_xml(self._d_folder)
        self._signal_config = _parse_acq_macaml(self._d_folder)

        # Enrich DAD channels with wavelength info
        for key, cfg in self._signal_config.items():
            if key in self._channels and cfg.get("wavelength_nm"):
                ch = self._channels[key]
                wl = cfg["wavelength_nm"]
                bw = cfg.get("bandwidth_nm", "")
                ch.description = f"DAD {wl} nm" + (f" (bw {bw} nm)" if bw else "")

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
        """Only CE instrument channels (current, voltage, pressure)."""
        return {k: v for k, v in self._channels.items() if k.startswith("HPCE")}

    @property
    def signal_config(self) -> dict:
        """DAD signal configuration from acq.macaml (wavelengths, etc.)."""
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
            ch.plot(ax=ax, label=f"{name} ({ch.description})", **kwargs)
        ax.set_title(f"{self.folder_name} \u2014 DAD signals")
        ax.legend(fontsize=8)
        return ax

    def plot_ce(self, **kwargs):
        """Plot CE instrument traces (current, voltage, pressure) as subplots."""
        import matplotlib.pyplot as plt
        ce = sorted(self.ce_channels.items())
        if not ce:
            raise ValueError("No CE channels found.")
        fig, axes = plt.subplots(len(ce), 1, figsize=(12, 3 * len(ce)), sharex=True)
        if len(ce) == 1:
            axes = [axes]
        for ax, (name, ch) in zip(axes, ce):
            ch.plot(ax=ax, **kwargs)
        fig.suptitle(f"{self.folder_name} \u2014 CE traces", fontsize=12)
        fig.tight_layout()
        return fig

    def plot_all(self, **kwargs):
        """Plot everything: DAD + CE channels."""
        import matplotlib.pyplot as plt
        n_dad = len(self.dad_channels)
        n_ce = len(self.ce_channels)
        n_total = (1 if n_dad else 0) + n_ce
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
            f"{self.folder_name} \u2014 {self.sample_name}",
            fontsize=12, y=1.01,
        )
        fig.tight_layout()
        return fig

    # ------ repr ------

    def __repr__(self) -> str:
        lines = [
            f"AgilentDFolder('{self.folder_name}')",
            f"  Sample : {self.sample_name}",
            f"  Method : {self.method}",
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
        print(f"{name}: {ch.metadata}")

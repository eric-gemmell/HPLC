from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class FillRegion:
    time: np.ndarray
    upper: np.ndarray
    lower: np.ndarray | None = None
    label: str = ""

    def __eq__(self, other):
        if not isinstance(other, FillRegion):
            return NotImplemented
        return (
            np.array_equal(self.time, other.time)
            and np.array_equal(self.upper, other.upper)
            and np.array_equal(self.lower, other.lower)
            and self.label == other.label
        )


@dataclass
class TraceProperties2D:
    name: str
    time: np.ndarray
    time_unit: str
    signal: np.ndarray
    signal_unit: str
    fills: list[FillRegion] | None = None
    baseline: np.ndarray | None = None
    color: str | None = None
    line_width: float | None = None

    def __eq__(self, other):
        if not isinstance(other, TraceProperties2D):
            return NotImplemented
        return (
            self.name == other.name
            and np.array_equal(self.time, other.time)
            and self.time_unit == other.time_unit
            and np.array_equal(self.signal, other.signal)
            and self.signal_unit == other.signal_unit
            and self.fills == other.fills
            and np.array_equal(self.baseline, other.baseline)
            and self.color == other.color
            and self.line_width == other.line_width
        )


@dataclass
class TraceProperties3D:
    """A time x wavelength intensity surface (e.g. a PDA cube).

    `signal` is indexed time-first: signal[i] is the spectrum at time[i],
    so its shape is (len(time), len(wavelengths)).

    `colors` / `color_values` are parallel: color_values[k] is the absolute
    signal value at which colors[k] is reached. Absolute rather than
    normalised, so two traces coloured the same way stay comparable.
    Both default to None; the graphing layer supplies the defaults.
    """

    name: str
    time: np.ndarray
    time_unit: str
    wavelengths: np.ndarray
    wavelength_unit: str
    signal: np.ndarray
    signal_unit: str
    colors: list[str] | None = None
    color_values: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.time = np.asarray(self.time, dtype=float)
        self.wavelengths = np.asarray(self.wavelengths, dtype=float)
        self.signal = np.asarray(self.signal, dtype=float)

        if self.signal.ndim != 2:
            raise ValueError(
                f"signal must be a 2D array, got {self.signal.ndim} dimension(s)"
            )
        if self.time.size == 0:
            raise ValueError("time must contain at least one point")
        if self.wavelengths.size == 0:
            raise ValueError("wavelengths must contain at least one point")

        expected = (len(self.time), len(self.wavelengths))
        if self.signal.shape != expected:
            raise ValueError(
                f"signal shape {self.signal.shape} does not match "
                f"(len(time), len(wavelengths)) = {expected}"
            )

        if self.colors is None and self.color_values is None:
            return
        if self.colors is None or self.color_values is None:
            raise ValueError("colors and color_values must be given together")

        self.colors = list(self.colors)
        self.color_values = np.asarray(self.color_values, dtype=float)
        if len(self.colors) < 2:
            raise ValueError(
                f"colors must contain at least 2 entries, got {len(self.colors)}"
            )
        if len(self.color_values) != len(self.colors):
            raise ValueError(
                f"color_values has {len(self.color_values)} entries but "
                f"colors has {len(self.colors)}"
            )

    def __eq__(self, other):
        if not isinstance(other, TraceProperties3D):
            return NotImplemented
        return (
            self.name == other.name
            and np.array_equal(self.time, other.time)
            and self.time_unit == other.time_unit
            and np.array_equal(self.wavelengths, other.wavelengths)
            and self.wavelength_unit == other.wavelength_unit
            and np.array_equal(self.signal, other.signal)
            and self.signal_unit == other.signal_unit
            and self.colors == other.colors
            and _optional_arrays_equal(self.color_values, other.color_values)
        )


def _optional_arrays_equal(a, b) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return np.array_equal(a, b)


@dataclass
class GraphProperties2D:
    signals: list[TraceProperties2D]
    title: str = ""
    x_min: float | None = None
    x_max: float | None = None
    y_min: float | None = None
    y_max: float | None = None

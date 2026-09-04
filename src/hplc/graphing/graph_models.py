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
class GraphProperties2D:
    signals: list[TraceProperties2D]
    title: str = ""
    x_min: float | None = None
    x_max: float | None = None
    y_min: float | None = None
    y_max: float | None = None

from __future__ import annotations

import copy

import numpy as np
from scipy.special import erf

from hplc.graphing.graph_models import FillRegion


class Peak():
    def __init__(
        self,
        start_time: float,
        end_time: float,
        retention_time: float | None = None,
        area: float | None = None,
        compound: str | None = None,
    ) -> None:
        self.retention_time = retention_time
        self.area = area
        self.start_time = start_time
        self.end_time = end_time
        self.compound = compound

    def _copy(self) -> Peak:
        return copy.deepcopy(self)


class Peak2D(Peak):
    def __init__(
        self,
        start_time: float,
        end_time: float,
        time: np.ndarray,
        signal: np.ndarray,
        baseline: np.ndarray | None = None,
        compound: str | None = None,
    ) -> None:
        self._time = time
        self._signal = signal
        self._baseline = baseline
        self._mask = (time >= start_time) & (time <= end_time)

        area, max_height, retention_time = self._compute_properties()

        super().__init__(
            retention_time=retention_time,
            area=area,
            start_time=start_time,
            end_time=end_time,
            compound=compound,
        )
        self.max_height = max_height

    @property
    def baseline(self) -> np.ndarray | None:
        return self._baseline

    def _baseline_masked(self) -> np.ndarray:
        if self._baseline is not None:
            return self._baseline[self._mask]
        return np.zeros(np.count_nonzero(self._mask))

    def _compute_properties(self):
        t = self._time[self._mask]
        net = self._signal[self._mask] - self._baseline_masked()
        area = float(np.trapezoid(net, t))
        max_height = float(np.max(net))
        retention_time = float(t[np.argmax(self._signal[self._mask])])
        return area, max_height, retention_time

    def _label(self) -> str:
        return self.compound or f"Peak @ {self.retention_time:.1f}s"

    def to_fill_region(self) -> FillRegion:
        t = self._time[self._mask]
        upper = self._signal[self._mask]
        lower = self._baseline_masked()
        return FillRegion(time=t, upper=upper, lower=lower, label=self._label())

    def recalculate(self, signal: np.ndarray | None = None, baseline: np.ndarray | None = None) -> None:
        if signal is not None:
            self._signal = signal
        if baseline is not None:
            self._baseline = baseline
        self.area, self.max_height, self.retention_time = self._compute_properties()

    def __repr__(self) -> str:
        return (
            f"Peak2D(retention_time={self.retention_time}, "
            f"max_height={self.max_height}, "
            f"area={self.area}, "
            f"start_time={self.start_time}, "
            f"end_time={self.end_time})"
        )


AreaPeak2D = Peak2D


class MathPeak2D(Peak2D):
    def __init__(
        self,
        mu: float,
        amplitude: float,
        sigma: float,
        alpha: float,
        time: np.ndarray,
        baseline: np.ndarray | None = None,
        compound: str | None = None,
    ) -> None:
        self.mu = mu
        self.amplitude = amplitude
        self.sigma = sigma
        self.alpha = alpha

        curve = self._compute_curve(time)

        # Derive start/end from the curve (where signal > 0.1% of max)
        max_val = np.max(curve)
        threshold = max_val * 0.001
        above = np.where(curve > threshold)[0]
        start_time = float(time[above[0]])
        end_time = float(time[above[-1]])

        # The "signal" for Peak2D storage is curve + baseline
        signal = curve + baseline if baseline is not None else curve

        super().__init__(
            start_time=start_time,
            end_time=end_time,
            time=time,
            signal=signal,
            baseline=baseline,
            compound=compound,
        )

    def _compute_curve(self, time: np.ndarray) -> np.ndarray:
        z = (time - self.mu) / self.sigma
        phi = (1 / (self.sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * z ** 2)
        Phi = 0.5 * (1 + erf(self.alpha * z / np.sqrt(2)))
        return 2 * self.amplitude * phi * Phi

    def _compute_properties(self):
        t = self._time[self._mask]
        curve = self._compute_curve(t)
        area = float(np.trapezoid(curve, t))
        max_height = float(np.max(curve))
        retention_time = float(t[np.argmax(curve)])
        return area, max_height, retention_time

    def to_fill_region(self) -> FillRegion:
        t = self._time[self._mask]
        curve = self._compute_curve(t)
        bl = self._baseline_masked()
        upper = curve + bl
        lower = bl
        return FillRegion(time=t, upper=upper, lower=lower, label=self._label())

    def recalculate(self, signal: np.ndarray | None = None, baseline: np.ndarray | None = None) -> None:
        if baseline is not None:
            self._baseline = baseline
            # Recompute stored signal as curve + new baseline
            self._signal = self._compute_curve(self._time) + self._baseline
        # signal argument is intentionally ignored — shape is defined by math params
        # Properties don't change since they're derived from the curve alone
        self.area, self.max_height, self.retention_time = self._compute_properties()

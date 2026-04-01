from __future__ import annotations

from abc import ABC, abstractmethod
import copy

from HPLC.models.signal import Signal, Signal2D
from HPLC.graphing import plot_signal_stack_2d


class SignalStack(ABC):
    """Base class for stacked signal views."""

    def __init__(self, signals: list[Signal]) -> None:
        if not signals:
            raise ValueError("Must provide at least one signal")
        self.signals = list(signals)
        self.deltas: list[float] = [0.0] * len(signals)
        self.stretch_factors: list[float] = [1.0] * len(signals)

    def __len__(self) -> int:
        return len(self.signals)

    def __iter__(self):
        return iter(self.signals)

    def _resolve_index(
        self,
        signal: int | None = None,
        filename: str | None = None,
        detector_name: str | None = None,
    ) -> int:
        if signal is not None:
            if signal < 0 or signal >= len(self.signals):
                raise IndexError(f"Signal index {signal} out of range for stack of size {len(self.signals)}")
            return signal

        if filename is not None:
            matches = [i for i, s in enumerate(self.signals) if s.filename == filename]
            if not matches:
                raise KeyError(f"No signal with filename '{filename}'")
            if len(matches) > 1:
                raise ValueError(f"Multiple signals match filename '{filename}'")
            return matches[0]

        if detector_name is not None:
            matches = [i for i, s in enumerate(self.signals) if s.detector_name == detector_name]
            if not matches:
                raise KeyError(f"No signal with detector_name '{detector_name}'")
            if len(matches) > 1:
                raise ValueError(f"Multiple signals match detector_name '{detector_name}'")
            return matches[0]

        # Default to last signal
        return len(self.signals) - 1

    def _copy(self) -> SignalStack:
        return copy.deepcopy(self)

    @abstractmethod
    def display(self): ...


class SignalStack2D(SignalStack):
    """A vertical stack of 2D signals with shift/stretch alignment."""

    def __init__(self, signals: list[Signal2D]) -> None:
        super().__init__(signals)
        self.signals: list[Signal2D] = self.signals

    def shift(
        self,
        delta: float | None = None,
        from_time: float | None = None,
        to_time: float | None = None,
        signal: int | None = None,
        filename: str | None = None,
        detector_name: str | None = None,
    ) -> SignalStack2D:
        idx = self._resolve_index(signal=signal, filename=filename, detector_name=detector_name)

        has_delta = delta is not None
        has_from = from_time is not None
        has_to = to_time is not None

        if has_delta and (has_from or has_to):
            raise ValueError("Cannot specify both delta and from_time/to_time")
        if not has_delta and not (has_from and has_to):
            raise ValueError("Must specify either delta or both from_time and to_time")

        if has_delta:
            self.deltas[idx] += delta
        else:
            self.deltas[idx] += to_time - from_time

        return self._copy()

    def stretch(
        self,
        factor: float,
        anchor: float = 0.0,
        signal: int | None = None,
        filename: str | None = None,
        detector_name: str | None = None,
    ) -> SignalStack2D:
        if factor <= 0:
            raise ValueError("Factor must be positive")

        idx = self._resolve_index(signal=signal, filename=filename, detector_name=detector_name)

        self.stretch_factors[idx] *= factor
        self.deltas[idx] += anchor * (1 - factor)

        return self._copy()

    def fit(
        self,
        from_times: tuple[float, float],
        to_times: tuple[float, float],
        signal: int | None = None,
        filename: str | None = None,
        detector_name: str | None = None,
    ) -> SignalStack2D:
        f1, f2 = from_times
        t1, t2 = to_times

        if f1 == f2:
            raise ValueError("from_times must be distinct")

        idx = self._resolve_index(signal=signal, filename=filename, detector_name=detector_name)

        scale = (t2 - t1) / (f2 - f1)
        offset = t1 - scale * f1

        self.stretch_factors[idx] = scale
        self.deltas[idx] = offset

        return self._copy()

    def display(
        self,
        scrunch: float = 1.0,
        title: str = "Signal Stack",
        width: int = 700,
        height: int | None = None,
        filename: str | None = None,
        dpi: int = 300,
    ) -> SignalStack2D:
        signal_dicts = []
        for i, sig in enumerate(self.signals):
            peak_dicts = [
                {
                    "mu": p.mu,
                    "amplitude": p.amplitude,
                    "sigma": p.sigma,
                    "alpha": p.alpha,
                    "label": p.compound or f"Peak @ {p.retention_time:.1f}",
                }
                for p in sig.peaks
                if p.mu is not None
            ]

            signal_dicts.append({
                "name": sig.filename or sig.detector_name or f"Signal {i + 1}",
                "time": sig.time,
                "time_unit": sig.time_unit,
                "signal": sig.signal,
                "signal_unit": sig.signal_unit,
                "peaks": peak_dicts or None,
                "delta": self.deltas[i],
                "stretch_factor": self.stretch_factors[i],
            })

        plot_signal_stack_2d(
            signals=signal_dicts,
            title=title,
            scrunch=scrunch,
            width=width,
            height=height,
            filename=filename,
            dpi=dpi,
        )
        return self

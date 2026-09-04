from __future__ import annotations

from abc import ABC, abstractmethod
import copy

from hplc.models.signal import Signal, Signal2D
from hplc.graphing import plot_2d_graph


class SignalStack(ABC):
    """Base class for stacked signal views."""

    def __init__(self, signals: list[Signal]) -> None:
        if not signals:
            raise ValueError("Must provide at least one signal")
        self.signals = list(signals)
        self.deltas: list[float] = [0.0] * len(signals)
        self.stretch_factors: list[float] = [1.0] * len(signals)
        self.y_scales: list[float] = [1.0] * len(signals)
        self.labels: list[str | None] = [None] * len(signals)
        self.colors: list[str | None] = [None] * len(signals)
        self.line_widths: list[float | None] = [None] * len(signals)

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

    def set_label(self, signal: int, name: str) -> SignalStack:
        idx = self._resolve_index(signal=signal)
        self.labels[idx] = name
        return self._copy()

    def set_color(self, signal: int, color: str) -> SignalStack:
        idx = self._resolve_index(signal=signal)
        self.colors[idx] = color
        return self._copy()

    def set_line_width(self, signal: int, line_width: float) -> SignalStack:
        idx = self._resolve_index(signal=signal)
        self.line_widths[idx] = line_width
        return self._copy()

    @abstractmethod
    def display(self): ...


class SignalStack2D(SignalStack):
    """A vertical stack of 2D signals with shift/stretch alignment."""

    def __init__(self, signals: list[Signal2D], title: str = "signal stack") -> None:
        super().__init__(signals)
        self.signals: list[Signal2D] = self.signals
        self.title = title

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

    def scale_y(
        self,
        factor: float,
        signal: int | None = None,
        filename: str | None = None,
        detector_name: str | None = None,
    ) -> SignalStack2D:
        if factor <= 0:
            raise ValueError("Factor must be positive")

        idx = self._resolve_index(signal=signal, filename=filename, detector_name=detector_name)

        self.y_scales[idx] *= factor

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
        y_offset: float = 0.0,
        width: int = 1000,
        height: int = 600,
        filename: str | None = None,
        dpi: int = 300,
        style: str = "color",
        title: str | None = None,
        x_min: float | None = None,
        x_max: float | None = None,
        y_min: float | None = None,
        y_max: float | None = None,
        labels: list[str] | None = None,
        colors: list[str] | None = None,
        line_widths: list[float] | None = None,
    ) -> SignalStack2D:
        if labels is not None and len(labels) != len(self.signals):
            raise ValueError(
                f"labels must have one entry per signal ({len(self.signals)}), got {len(labels)}"
            )
        if colors is not None and len(colors) != len(self.signals):
            raise ValueError(
                f"colors must have one entry per signal ({len(self.signals)}), got {len(colors)}"
            )
        if line_widths is not None and len(line_widths) != len(self.signals):
            raise ValueError(
                f"line_widths must have one entry per signal ({len(self.signals)}), got {len(line_widths)}"
            )

        traces = []
        for i, sig in enumerate(self.signals):
            trace = sig.to_trace_properties()
            if labels is not None:
                trace.name = labels[i]
            elif self.labels[i] is not None:
                trace.name = self.labels[i]

            if colors is not None:
                trace.color = colors[i]
            elif self.colors[i] is not None:
                trace.color = self.colors[i]

            if line_widths is not None:
                trace.line_width = line_widths[i]
            elif self.line_widths[i] is not None:
                trace.line_width = self.line_widths[i]

            # Apply stretch and shift to time axis
            stretch = self.stretch_factors[i]
            delta = self.deltas[i]
            if stretch != 1.0 or delta != 0.0:
                trace.time = trace.time * stretch + delta
                for fill in (trace.fills or []):
                    fill.time = fill.time * stretch + delta

            # Apply per-signal y scale
            y_scale = self.y_scales[i]
            if y_scale != 1.0:
                trace.signal = trace.signal * y_scale
                if trace.baseline is not None:
                    trace.baseline = trace.baseline * y_scale
                for fill in (trace.fills or []):
                    fill.upper = fill.upper * y_scale
                    fill.lower = fill.lower * y_scale

            # Apply vertical offset
            offset = i * y_offset
            trace.signal = trace.signal + offset
            if trace.baseline is not None:
                trace.baseline = trace.baseline + offset
            for fill in (trace.fills or []):
                fill.upper = fill.upper + offset
                fill.lower = fill.lower + offset
            traces.append(trace)

        plot_2d_graph(
            signals=traces,
            title=title or self.title,
            width=width,
            height=height,
            filename=filename,
            dpi=dpi,
            style=style,
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
            fig=None,
        )
        return self

    def display_publication(
        self,
        y_offset: float = 0.0,
        width: int = 1000,
        height: int = 600,
        filename: str | None = None,
        dpi: int = 300,
        title: str | None = None,
        x_min: float | None = None,
        x_max: float | None = None,
        y_min: float | None = None,
        y_max: float | None = None,
        labels: list[str] | None = None,
        colors: list[str] | None = None,
        line_widths: list[float] | None = None,
    ) -> SignalStack2D:
        return self.display(
            y_offset=y_offset,
            width=width,
            height=height,
            filename=filename,
            dpi=dpi,
            style="bw",
            title=title,
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
            labels=labels,
            colors=colors,
            line_widths=line_widths,
        )

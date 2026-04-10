from __future__ import annotations
import copy

from hplc.models.signal import Signal
from hplc.graphing import plot_graphs, plot_2d_graph
from hplc.graphing.graph_models import GraphProperties2D


class Chromatogram:

    def __init__(
        self,
        signals: dict[str, Signal],
        filename: str,
        additional_data: dict | None = None
    ) -> None:

        if not signals:
            raise ValueError("Chromatogram must contain at least one signal")

        self.signals: dict[str, Signal] = signals
        self.filename: str = filename
        self.additional_data: dict = additional_data or {}

    def _select_signals(self, detectors: list[str] | None = None) -> dict[str, Signal]:
        if detectors is None:
            return self.signals
        return {name: self.signals[name] for name in detectors}

    def _copy(self) -> Chromatogram:
        return copy.deepcopy(self)

    def detect_peaks(self) -> Chromatogram:
        result = self._copy()
        for name, sig in result.signals.items():
            result.signals[name] = sig.detect_peaks()
        return result

    def display(
        self,
        detectors: list[str] | None = None,
        width: int = 540,
        height: int = 400,
        filename: str | None = None,
        dpi: int = 300,
        style: str = "color",
        title: str | None = None,
        x_min: float | None = None,
        x_max: float | None = None,
        y_min: float | None = None,
        y_max: float | None = None,
    ) -> Chromatogram:
        selected = self._select_signals(detectors)

        graphs = [
            GraphProperties2D(
                signals=[sig.to_trace_properties()],
                title=name,
                x_min=x_min,
                x_max=x_max,
                y_min=y_min,
                y_max=y_max,
            )
            for name, sig in selected.items()
        ]

        plot_graphs(
            graphs=graphs,
            title=title if title is not None else self.filename,
            width=width,
            height=height,
            filename=filename,
            dpi=dpi,
            style=style,
        )
        return self

    def display_publication(
        self,
        detectors: list[str] | None = None,
        width: int = 540,
        height: int = 400,
        filename: str | None = None,
        dpi: int = 300,
        title: str | None = None,
        x_min: float | None = None,
        x_max: float | None = None,
        y_min: float | None = None,
        y_max: float | None = None,
    ) -> Chromatogram:
        return self.display(
            detectors=detectors,
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
        )

    def display_stacked(
        self,
        detectors: list[str] | None = None,
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
    ) -> Chromatogram:
        selected = self._select_signals(detectors)

        traces = []
        for i, (name, sig) in enumerate(selected.items()):
            trace = sig.to_trace_properties()
            trace.name = name
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
            title=title if title is not None else self.filename,
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
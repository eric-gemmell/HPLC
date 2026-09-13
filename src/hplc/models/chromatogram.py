from __future__ import annotations
import copy

import numpy as np

from hplc.models.signal import Signal, Signal2D, Signal3D
from hplc.graphing import plot_graphs, plot_2d_graph
from hplc.graphing.graph_models import GraphProperties2D, GraphProperties3D


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

    def detect_peaks(self, peak_retention_times: list[tuple[float, float]]) -> Chromatogram:
        """Detect peaks on every 2D signal, using the same retention-time windows.

        3D signals have no peak concept yet and are left untouched.
        """
        result = self._copy()
        for name, sig in list(result.signals.items()):
            if isinstance(sig, Signal2D):
                result.signals[name] = sig.detect_peaks(peak_retention_times)
        return result

    def compute_baseline(self, **kwargs) -> Chromatogram:
        """Estimate a baseline for every 2D signal.

        Keyword arguments pass straight through to each signal. 3D signals
        have no baseline concept yet and are left untouched.
        """
        result = self._copy()
        for name, sig in list(result.signals.items()):
            if isinstance(sig, Signal2D):
                result.signals[name] = sig.compute_baseline(**kwargs)
        return result

    def define_peak_by_bounds(self, start_time: float, end_time: float) -> Chromatogram:
        """Define a peak manually on every 2D signal, from its integration bounds.

        Bounds are in each signal's time unit. 3D signals have no peak concept
        yet and are left untouched.
        """
        result = self._copy()
        for name, sig in list(result.signals.items()):
            if isinstance(sig, Signal2D):
                result.signals[name] = sig.define_peak_by_bounds(start_time, end_time)
        return result

    def export_peaks(self) -> dict[str, list[dict]]:
        """Peaks per 2D signal, keyed by detector name.

        3D signals have no peak concept yet and are omitted.
        """
        return {
            name: sig.export_peaks()
            for name, sig in self.signals.items()
            if isinstance(sig, Signal2D)
        }

    def to_graph_properties(
        self,
        detectors: list[str] | None = None,
        x_min: float | None = None,
        x_max: float | None = None,
        y_min: float | None = None,
        y_max: float | None = None,
        z_min: float | None = None,
        z_max: float | None = None,
    ) -> list[GraphProperties2D | GraphProperties3D]:
        """One graphing panel per signal, dispatched on the signal's dimensionality."""
        selected = self._select_signals(detectors)

        graphs: list[GraphProperties2D | GraphProperties3D] = []
        for name, sig in selected.items():
            if isinstance(sig, Signal3D):
                graphs.append(GraphProperties3D(
                    signal=sig.to_trace_properties(),
                    title=name,
                    x_min=x_min,
                    x_max=x_max,
                    y_min=y_min,
                    y_max=y_max,
                    z_min=z_min,
                    z_max=z_max,
                ))
            else:
                graphs.append(GraphProperties2D(
                    signals=[sig.to_trace_properties()],
                    title=name,
                    x_min=x_min,
                    x_max=x_max,
                    y_min=y_min,
                    y_max=y_max,
                ))
        return graphs

    def extract_wavelengths(
        self,
        wavelengths: float | list[float],
        bandwidth: float = 0.0,
        detector: str | None = None,
    ) -> Chromatogram:

        source = self._select_3d_signal(detector)

        if isinstance(wavelengths, (int, float, np.number)):
            wavelengths = [wavelengths]
        if len(wavelengths) == 0:
            raise ValueError("Must request at least one wavelength")

        extracted = [source.at_wavelength(wl, bandwidth=bandwidth) for wl in wavelengths]

        result = self._copy()
        for sig in extracted:
            result.signals[sig.detector_name] = sig
        return result

    def _select_3d_signal(self, detector: str | None = None) -> Signal3D:
        if detector is not None:
            if detector not in self.signals:
                raise ValueError(
                    f"Detector {detector!r} not found in chromatogram; "
                    f"have {sorted(self.signals)}"
                )
            source = self.signals[detector]
            if not isinstance(source, Signal3D):
                raise ValueError(f"Detector {detector!r} is not a 3D signal")
            return source

        candidates = {n: s for n, s in self.signals.items() if isinstance(s, Signal3D)}
        if not candidates:
            raise ValueError("Chromatogram contains no 3D signal to extract from")
        if len(candidates) > 1:
            raise ValueError(
                f"Multiple 3D signals present ({', '.join(sorted(candidates))}); "
                f"pass detector= to choose one"
            )
        return next(iter(candidates.values()))

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
        z_min: float | None = None,
        z_max: float | None = None,
    ) -> Chromatogram:
        graphs = self.to_graph_properties(
            detectors=detectors,
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
            z_min=z_min,
            z_max=z_max,
        )

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
        selected = {
            name: sig
            for name, sig in self._select_signals(detectors).items()
            if isinstance(sig, Signal2D)
        }
        if not selected:
            raise ValueError("display_stacked requires at least one 2D signal")

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
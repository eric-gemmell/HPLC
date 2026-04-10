from __future__ import annotations
 
from abc import ABC, abstractmethod
 
import numpy as np
import copy
from hplc.graphing import plot_2d_graph
from hplc.graphing.graph_models import TraceProperties2D
from hplc.models.peak import Peak, Peak2D
from hplc.analysis.peak_detection import detect_peaks
 
class Signal(ABC):
    """Base class for all signal types."""
 
    def __init__(
        self,
        detector_name: str,
        time: np.ndarray,
        time_unit: str,
        signal_unit: str,
        filename: str | None = None,
        additional_data: dict | None = None,
        peaks: list[Peak] | None = None,

    ) -> None:
        
        if not detector_name:
            raise ValueError("Detector name must not be empty")
        if not time_unit:
            raise ValueError("Time unit required")
        if not signal_unit:
            raise ValueError("Signal unit required")
        if len(time) == 0:
            raise ValueError("Signal must contain at least one data point")
 
        self.detector_name = detector_name
        self.time = np.asarray(time, dtype=float)
        self.time_unit = time_unit
        self.signal_unit = signal_unit
        self.filename = filename
        self.additional_data = additional_data or {}
        self.peaks: list[Peak] = peaks or []
        self._peaks_detected: bool = bool(peaks)
 
    def __len__(self) -> int:
        if self._peaks_detected:
            return len(self.peaks)
        return len(self.time)

    def __iter__(self):
        return iter(self.peaks)

    def _copy(self) -> Signal:
        return copy.deepcopy(self)
        
    @abstractmethod
    def display(self):
        ...
 
 
class Signal2D(Signal):
 
    def __init__(
        self,
        detector_name: str,
        time: np.ndarray,
        time_unit: str,
        signal: np.ndarray,
        signal_unit: str,
        filename: str | None = None,
        baseline: np.ndarray | None = None,
        additional_data: dict | None = None,
        peaks: list[Peak2D] | None = None,
    ) -> None:
        super().__init__(
            detector_name=detector_name,
            time=time,
            time_unit=time_unit,
            signal_unit=signal_unit,
            filename=filename,
            additional_data=additional_data,
            peaks=peaks,
        )

        if len(signal) != len(time):
            raise ValueError(
                f"time and signal must have the same length, "
                f"got {len(time)} and {len(signal)}"
            )

        self.signal = np.asarray(signal, dtype=float)
        self.baseline = np.asarray(baseline, dtype=float) if baseline is not None else None
        self.peaks: list[Peak2D] = peaks or []
 
    def detect_peaks(self) -> Signal2D:
        result = self._copy()
        raw_peaks = detect_peaks(self.time, self.signal)
        result.peaks = [
            Peak2D(
                retention_time=p["retention_time"],
                max_height=p["max_height"],
                area=p["area"],
                start_time=p["start_time"],
                end_time=p["end_time"],
                unit=self.signal_unit,
                mu=p.get("mu"),
                amplitude=p.get("amplitude"),
                sigma=p.get("sigma"),
                alpha=p.get("alpha"),
            )
            for p in raw_peaks
        ]
        result._peaks_detected = True
        return result

    def to_trace_properties(self) -> TraceProperties2D:
        fills = [p.to_fill_region() for p in self.peaks]
        return TraceProperties2D(
            name=f"{self.filename}_{self.detector_name}",
            time=self.time,
            time_unit=self.time_unit,
            signal=self.signal,
            signal_unit=self.signal_unit,
            fills=fills,
            baseline=self.baseline,
        )

    def display(
        self,
        width: int = 520,
        height: int = 340,
        filename: str | None = None,
        dpi: int = 300,
        style: str = "color",
        title: str | None = None,
        x_min: float | None = None,
        x_max: float | None = None,
        y_min: float | None = None,
        y_max: float | None = None,
    ) -> Signal2D:
        trace = self.to_trace_properties()
        plot_2d_graph(
            signals=[trace],
            title=title or trace.name,
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
        width: int = 520,
        height: int = 340,
        filename: str | None = None,
        dpi: int = 300,
        title: str | None = None,
        x_min: float | None = None,
        x_max: float | None = None,
        y_min: float | None = None,
        y_max: float | None = None,
    ) -> Signal2D:
        return self.display(
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

    def stack(self, *others: Signal2D) -> SignalStack2D:
        from hplc.models.signal_stack import SignalStack2D  # avoid circular import
        return SignalStack2D([self, *others])
from __future__ import annotations
 
from abc import ABC, abstractmethod
 
import numpy as np
import copy
from HPLC.graphing import plot_signal_2d
from HPLC.models.peak import Peak, Peak2D
from HPLC.analysis.peak_detection import detect_peaks
 
class Signal(ABC):
    """Base class for all signal types."""
 
    def __init__(
        self,
        detector_name: str,
        time: np.ndarray,
        time_unit: str,
        signal_unit: str,
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
        additional_data: dict | None = None,
        peaks: list[Peak2D] | None = None,
    ) -> None:
        super().__init__(
            detector_name=detector_name,
            time=time,
            time_unit=time_unit,
            signal_unit=signal_unit,
            additional_data=additional_data,
            peaks=peaks,
        )
 
        if len(signal) != len(time):
            raise ValueError(
                f"time and signal must have the same length, "
                f"got {len(time)} and {len(signal)}"
            )
 
        self.signal = np.asarray(signal, dtype=float)
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

    def display(
        self,
        width: int = 520,
        height: int = 340,
        filename: str | None = None,
        dpi: int = 300,
    ) -> Signal2D:
        peak_dicts = [
            {
                "mu": p.mu,
                "amplitude": p.amplitude,
                "sigma": p.sigma,
                "alpha": p.alpha,
                "label": p.compound or f"Peak @ {p.retention_time:.1f}s",
            }
            for p in self.peaks
        ]

        plot_signal_2d(
            name=self.detector_name,
            time=self.time,
            time_unit=self.time_unit,
            signal=self.signal,
            signal_unit=self.signal_unit,
            peaks=peak_dicts if peak_dicts else None,
            width=width,
            height=height,
            filename=filename,
            dpi=dpi,
        )
        return self
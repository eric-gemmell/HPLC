from __future__ import annotations
 
from abc import ABC, abstractmethod
 
import numpy as np
import copy
from HPLC.graphing import plot_signal_2d
from HPLC.models.peak import Peak, Peak2D
 
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
 
    def __len__(self) -> int:
        return len(self.time)

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
 
    def display(
        self,
        width: int = 520,
        height: int = 340,
        filename: str | None = None,
        dpi: int = 300,
    ) -> Signal:
        plot_signal_2d(
            name=self.detector_name,
            time=self.time,
            time_unit=self.time_unit,
            signal=self.signal,
            signal_unit=self.signal_unit,
            width=width,
            height=height,
            filename=filename,
            dpi=dpi,
        )
        return self
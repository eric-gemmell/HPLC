from __future__ import annotations
 
from abc import ABC, abstractmethod
 
import numpy as np
 
 
class Signal(ABC):
    """Base class for all signal types."""
 
    def __init__(
        self,
        detector_name: str,
        time: np.ndarray,
        time_unit: str,
        signal_unit: str,
        additional_data: dict | None = None,
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
 
    def __len__(self) -> int:
        return len(self.time)

    def _copy(self) -> Signal:
        return copy.deepcopy(self)
        
    @abstractmethod
    def display(self):
        """Display the signal. Implemented by concrete subclasses."""
        ...
 
 
class Signal2DBase(Signal):
    """2D signal (time + single signal array). Display-agnostic base."""
 
    def __init__(
        self,
        detector_name: str,
        time: np.ndarray,
        time_unit: str,
        signal: np.ndarray,
        signal_unit: str,
        additional_data: dict | None = None,
    ) -> None:
        super().__init__(
            detector_name=detector_name,
            time=time,
            time_unit=time_unit,
            signal_unit=signal_unit,
            additional_data=additional_data,
        )
 
        if len(signal) != len(time):
            raise ValueError(
                f"time and signal must have the same length, "
                f"got {len(time)} and {len(signal)}"
            )
 
        self.signal = np.asarray(signal, dtype=float)
 
    def display(self) -> Signal:
        raise NotImplementedError(
            "Use Signal2D from the package __init__ which has display wired in"
        )
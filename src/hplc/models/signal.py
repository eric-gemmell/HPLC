from __future__ import annotations
 
from abc import ABC, abstractmethod
 
import numpy as np
import copy
from hplc.graphing import plot_2d_graph, plot_3d_graph
from hplc.graphing.graph_models import TraceProperties2D, TraceProperties3D
from hplc.models.peak import AreaPeak2D, Peak, Peak2D
from hplc.analysis.baseline import compute_baseline
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
        return len(self.time)

    def __iter__(self):
        return iter(self.peaks)

    def _copy(self) -> Signal:
        return copy.deepcopy(self)
        
    @abstractmethod
    def to_trace_properties(self):
        """Convert to the graphing layer's DTO for this signal's dimensionality."""
        ...

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
 
    def detect_peaks(self, peak_retention_times: list[tuple[float, float]]) -> Signal2D:
        """Fit a peak inside each of the given (start, end) retention-time windows.

        The analysis layer sees the baseline-corrected signal, so the fitted
        peak shapes describe the peaks themselves rather than the drift
        underneath them. Returns a copy carrying the detected peaks.
        """
        result = self._copy()
        corrected = (
            self.signal - self.baseline if self.baseline is not None else self.signal
        )
        peaks = detect_peaks(self.time, corrected, peak_retention_times)

        if result.baseline is not None:
            # Sit the fitted curves back on top of the baseline for plotting.
            for peak in peaks:
                peak.recalculate(baseline=result.baseline)

        result.peaks = [*result.peaks, *peaks]
        result._peaks_detected = True
        return result

    def compute_baseline(self, **kwargs) -> Signal2D:
        """Estimate this signal's baseline over the whole trace.

        Any keyword arguments are handed to the analysis layer's
        compute_baseline untouched, so it can grow parameters without the
        models needing to know about them. Returns a copy carrying the
        computed baseline.
        """
        result = self._copy()
        result.baseline = compute_baseline(self.time, self.signal, **kwargs)
        return result

    def define_peak_by_bounds(self, start_time: float, end_time: float) -> Signal2D:
        if end_time <= start_time:
            raise ValueError(
                f"end_time ({end_time}) must be greater than start_time ({start_time})"
            )
        if not ((self.time >= start_time) & (self.time <= end_time)).any():
            raise ValueError(
                f"No data points between {start_time} and {end_time} {self.time_unit}; "
                f"signal spans {self.time[0]:g}-{self.time[-1]:g} {self.time_unit}"
            )

        result = self._copy()
        result.peaks = [
            *result.peaks,
            AreaPeak2D(
                start_time=start_time,
                end_time=end_time,
                time=result.time,
                signal=result.signal,
                baseline=result.baseline,
            ),
        ]
        result._peaks_detected = True
        return result

    def export_peaks(self) -> list[dict]:
        """Every peak on this signal as a list of plain dicts."""
        return [peak.export() for peak in self.peaks]

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

class Signal3D(Signal):
    """A time x wavelength intensity surface, e.g. a PDA/DAD cube.

    `signal` is indexed time-first: signal[i] is the spectrum at time[i],
    so its shape is (len(time), len(wavelengths)).
    """

    def __init__(
        self,
        detector_name: str,
        time: np.ndarray,
        time_unit: str,
        wavelengths: np.ndarray,
        wavelength_unit: str,
        signal: np.ndarray,
        signal_unit: str,
        filename: str | None = None,
        additional_data: dict | None = None,
    ) -> None:
        super().__init__(
            detector_name=detector_name,
            time=time,
            time_unit=time_unit,
            signal_unit=signal_unit,
            filename=filename,
            additional_data=additional_data,
        )

        if not wavelength_unit:
            raise ValueError("Wavelength unit required")

        self.wavelength_unit = wavelength_unit
        self.wavelengths = np.asarray(wavelengths, dtype=float)
        self.signal = np.asarray(signal, dtype=float)

        if self.wavelengths.size == 0:
            raise ValueError("wavelengths must contain at least one point")
        if self.signal.ndim != 2:
            raise ValueError(
                f"signal must be a 2D array, got {self.signal.ndim} dimension(s)"
            )

        expected = (len(self.time), len(self.wavelengths))
        if self.signal.shape != expected:
            raise ValueError(
                f"signal shape {self.signal.shape} does not match "
                f"(len(time), len(wavelengths)) = {expected}"
            )

    # -- graphing -----------------------------------------------------------

    def to_trace_properties(
        self,
        colors: list[str] | None = None,
        color_values: np.ndarray | None = None,
    ) -> TraceProperties3D:
        return TraceProperties3D(
            name=f"{self.filename}_{self.detector_name}",
            time=self.time,
            time_unit=self.time_unit,
            wavelengths=self.wavelengths,
            wavelength_unit=self.wavelength_unit,
            signal=self.signal,
            signal_unit=self.signal_unit,
            colors=colors,
            color_values=color_values,
        )

    def display(
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
        z_min: float | None = None,
        z_max: float | None = None,
        colors: list[str] | None = None,
        color_values: np.ndarray | None = None,
    ) -> Signal3D:
        trace = self.to_trace_properties(colors=colors, color_values=color_values)
        plot_3d_graph(
            signal=trace,
            title=title or trace.name,
            width=width,
            height=height,
            filename=filename,
            dpi=dpi,
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
            z_min=z_min,
            z_max=z_max,
            fig=None,
        )
        return self

    # -- extracting 2D traces -----------------------------------------------

    def _check_wavelength(self, wavelength: float) -> None:
        low = float(self.wavelengths.min())
        high = float(self.wavelengths.max())
        if not low <= wavelength <= high:
            raise ValueError(
                f"wavelength {wavelength:g} {self.wavelength_unit} lies outside the "
                f"measured range {low:g}-{high:g} {self.wavelength_unit}"
            )

    def at_wavelength(self, wavelength: float, bandwidth: float = 0.0) -> Signal2D:
        """Extract a single 2D trace at `wavelength`.

        Snaps to the nearest measured wavelength. With `bandwidth` > 0 the
        columns within +/- bandwidth/2 are averaged, as a real detector does.
        """
        if bandwidth < 0:
            raise ValueError("bandwidth must not be negative")
        self._check_wavelength(wavelength)

        nearest = int(np.argmin(np.abs(self.wavelengths - wavelength)))
        used = float(self.wavelengths[nearest])

        if bandwidth > 0:
            mask = np.abs(self.wavelengths - wavelength) <= bandwidth / 2
            if not mask.any():
                mask = np.zeros(len(self.wavelengths), dtype=bool)
                mask[nearest] = True
            signal = self.signal[:, mask].mean(axis=1)
        else:
            signal = self.signal[:, nearest]

        return Signal2D(
            detector_name=f"{self.detector_name} {used:g} {self.wavelength_unit}",
            time=self.time,
            time_unit=self.time_unit,
            signal=signal,
            signal_unit=self.signal_unit,
            filename=self.filename,
            additional_data={
                **self.additional_data,
                "wavelength": used,
                "bandwidth": bandwidth,
            },
        )

    def at_wavelengths(
        self,
        wavelengths: list[float],
        bandwidth: float = 0.0,
    ) -> SignalStack2D:
        """Extract several 2D traces at once, stacked for overlay."""
        from hplc.models.signal_stack import SignalStack2D  # avoid circular import

        if len(wavelengths) == 0:
            raise ValueError("Must request at least one wavelength")
        return SignalStack2D(
            [self.at_wavelength(wl, bandwidth=bandwidth) for wl in wavelengths]
        )

    def max_plot(self) -> Signal2D:
        """The maximum absorbance across all wavelengths at each time point."""
        return Signal2D(
            detector_name=f"{self.detector_name} max plot",
            time=self.time,
            time_unit=self.time_unit,
            signal=self.signal.max(axis=1),
            signal_unit=self.signal_unit,
            filename=self.filename,
            additional_data=dict(self.additional_data),
        )

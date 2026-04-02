from __future__ import annotations
import copy

from hplc.models.signal import Signal
from hplc.graphing import plot_chromatogram, plot_signal_2d

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


    def _copy(self) -> Chromatogram:
        return copy.deepcopy(self)

    def detect_peaks(self) -> Chromatogram:
        result = self._copy()
        for name, sig in result.signals.items():
            result.signals[name] = sig.detect_peaks()
        return result

    def display(
        self,
        detector_name: str | list[str] | tuple[str, ...] | None = None,
        width: int = 520,
        height: int = 340,
        filename: str | None = None,
        dpi: int = 300,
        style: str = "color",
        title: str | None = None,
        shared_x: bool = False,
        x_lim: tuple[float | None, float | None] | None = None,
        y_lim: tuple[float | None, float | None] | None = None,
    ) -> Chromatogram:
        if detector_name is None:
            selected = self.signals
        elif isinstance(detector_name, str):
            selected = {detector_name: self.signals[detector_name]}
        else:
            selected = {name: self.signals[name] for name in detector_name}

        display_title = title if title is not None else self.filename

        if len(selected) == 1:
            name, sig = next(iter(selected.items()))
            plot_signal_2d(
                signals=[{
                    "name": name,
                    "time": sig.time,
                    "time_unit": sig.time_unit,
                    "signal": sig.signal,
                    "signal_unit": sig.signal_unit,
                }],
                title=display_title,
                width=width,
                height=height,
                filename=filename,
                dpi=dpi,
                style=style,
                x_lim=x_lim,
                y_lim=y_lim,
            )
        else:
            plot_chromatogram(
                signals={
                    name: dict(
                        time=sig.time,
                        time_unit=sig.time_unit,
                        signal=sig.signal,
                        signal_unit=sig.signal_unit,
                    )
                    for name, sig in selected.items()
                },
                title=display_title,
                subplot_width=width,
                subplot_height=height,
                filename=filename,
                dpi=dpi,
                style=style,
                shared_x=shared_x,
            )
        return self

    def display_publication(
        self,
        detector_name: str | list[str] | tuple[str, ...] | None = None,
        width: int = 520,
        height: int = 340,
        filename: str | None = None,
        dpi: int = 300,
        title: str | None = None,
        shared_x: bool = False,
        x_lim: tuple[float | None, float | None] | None = None,
        y_lim: tuple[float | None, float | None] | None = None,
    ) -> Chromatogram:
        return self.display(
            detector_name=detector_name,
            width=width,
            height=height,
            filename=filename,
            dpi=dpi,
            style="bw",
            title=title,
            shared_x=shared_x,
            x_lim=x_lim,
            y_lim=y_lim,
        )
        
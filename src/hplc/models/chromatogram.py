from __future__ import annotations
import copy

from HPLC.models.signal import Signal
from HPLC.graphing import plot_chromatogram, plot_signal_2d

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
        self.peaks: list = []


    def _copy(self) -> Chromatogram:
        return copy.deepcopy(self)

    def display(
        self,
        detector_name: str | list[str] | tuple[str, ...] | None = None,
        filename: str | None = None,
        size: tuple[int, int] = (520, 340),
        dpi: int = 300,
        shared_x: bool = False,
    ) -> Chromatogram:
        
        
        if detector_name is None:
            selected = self.signals
        elif isinstance(detector_name, str):
            selected = {detector_name: self.signals[detector_name]}
        else:
            selected = {name: self.signals[name] for name in detector_name}

        if len(selected) == 1:
            name, sig = next(iter(selected.items()))
            plot_signal_2d(
                name=name,
                time=sig.time,
                time_unit=sig.time_unit,
                signal=sig.signal,
                signal_unit=sig.signal_unit,
                width=size[0],
                height=size[1],
                filename=filename,
                dpi=dpi,
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
                title=self.filename,
                subplot_width=size[0],
                subplot_height=size[1],
                filename=filename,
                dpi=dpi,
                shared_x=shared_x,
            )
        return self
        